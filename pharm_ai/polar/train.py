# encoding: utf-8
'''
@author: zyl
@file: train.py
@time: 2021/7/21 下午11:55
@desc:
'''
import os
import time
from sklearn.utils import resample
import pandas as pd
import wandb
from loguru import logger
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from pharm_ai.config import ConfigFilePaths
from pharm_ai.ddp.ddpclassification_model import DDPClassificationModel


class PolarTrainer:
    def __init__(self):
        self.start_time = '2021-07-22'
        self.end_time = ''

        self.wandb_proj = 'polar'
        self.model_version = 'test'

        self.model_type = 'bert'
        self.pretrained_model = ConfigFilePaths.bert_dir_remote
        self.use_cuda = True
        self.cuda_device = 0
        self.num_labels = 2
        self.args = PolarTrainer.set_model_parameter(model_version=self.model_version)

    def run(self):
        # self.train_0722()
        self.train_0723()
        pass

    @staticmethod
    def set_model_parameter(model_version, args=ClassificationArgs(), save_dir='polar'):
        # multiprocess
        args.use_multiprocessing = False
        args.use_multiprocessing_for_evaluation = False

        # base config
        args.reprocess_input_data = True
        args.fp16 = False
        args.manual_seed = 234
        # args.gradient_accumulation_steps = 8  # ==increase batch size,Use time for memory,

        # save
        args.no_save = False
        args.save_eval_checkpoints = False
        args.save_model_every_epoch = False
        args.save_optimizer_and_scheduler = True
        args.save_steps = -1

        # eval
        args.evaluate_during_training = True
        args.evaluate_during_training_verbose = True

        args.no_cache = False
        args.use_early_stopping = False
        args.encoding = None
        args.do_lower_case = False
        args.dynamic_quantize = False
        args.quantized_model = False
        args.silent = False

        args.overwrite_output_dir = True
        saved_dirs = ConfigFilePaths.project_dir + '/' + save_dir + '/'
        args.output_dir = saved_dirs + 'outputs/' + model_version + '/'
        args.cache_dir = saved_dirs + 'cache/' + model_version + '/'
        args.best_model_dir = saved_dirs + 'best_model/' + model_version + '/'
        args.tensorboard_dir = saved_dirs + 'runs/' + model_version + '/' + time.strftime("%Y%m%d_%H%M%S",
                                                                                          time.localtime()) + '/'
        return args

    def get_train_model(self):
        self.args.use_cached_eval_features = True
        if self.args.n_gpu <= 1:
            return ClassificationModel(model_type=self.model_type, model_name=self.pretrained_model,
                                       use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args)
        else:
            return DDPClassificationModel(model_type=self.model_type, model_name=self.pretrained_model, use_cuda=True,
                                          args=self.args)

    def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, shuffled=False):
        # deal with dt
        train_df = train_df[['text', 'labels']]
        if shuffled:
            train_df = resample(train_df, replace=False, n_samples=len(train_df), random_state=3242)
        eval_df = eval_df[['text', 'labels']]

        # config some parameters
        train_size = train_df.shape[0]
        all_steps = train_size / self.args.train_batch_size
        self.args.logging_steps = int(all_steps / 20)
        self.args.evaluate_during_training_steps = int(all_steps / 6)

        self.args.wandb_project = self.wandb_proj
        self.args.wandb_kwargs = {
            'name': self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
            'tags': [self.model_version, 'train']}

        # get model
        model = self.get_train_model()
        print(f'train length: {train_size}')
        model.args.update_from_dict({'train_length': train_size})

        # train
        try:
            start_time = time.time()
            model.train_model(train_df=train_df, eval_df=eval_df)
            logger.info('train finished!!!')
            end_time = time.time()
            need_time = round((end_time - start_time) / train_size, 4)
            training_time = round(need_time * train_size, 4)
            print(f'train time: {need_time} s * {train_size} = {training_time} s')
        except Exception as error:
            logger.error(f'train failed!!! ERROR:{error}')
        finally:
            wandb.finish()

    def train_0722(self):
        self.model_version = 'v0.0.0.1'
        self.args = PolarTrainer.set_model_parameter(model_version=self.model_version)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        self.cuda_device = 4
        self.args.n_gpu = 4

        self.args.num_train_epochs = 5
        self.args.learning_rate = 5e-6
        self.args.train_batch_size = 512
        self.args.eval_batch_size = 256
        train_df = pd.read_excel('./data/processed_dt_0722.xlsx', 'train_up')
        eval_df = pd.read_excel('./data/processed_dt_0722.xlsx', 'eval')
        self.train(train_df=train_df, eval_df=eval_df, shuffled=True)

    def train_0723(self):
        self.pretrained_model = "voidful/albert_chinese_base"
        self.model_type = 'bert'
        # self.pretrained_model = 'roberta-'
        self.model_version = 'v0.0.0.t'
        self.args = PolarTrainer.set_model_parameter(model_version=self.model_version)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        self.cuda_device = 4
        self.args.n_gpu = 1

        self.args.num_train_epochs = 5
        self.args.learning_rate = 5e-6
        self.args.train_batch_size = 100  # 512
        self.args.eval_batch_size = 100  # 256
        train_df = pd.read_excel('./data/processed_dt_0722.xlsx', 'train_up')[0:500]
        eval_df = pd.read_excel('./data/processed_dt_0722.xlsx', 'eval')[0:400]
        self.train(train_df=train_df, eval_df=eval_df, shuffled=True)


if __name__ == '__main__':
    PolarTrainer().run()
