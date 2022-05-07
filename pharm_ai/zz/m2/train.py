import time
import warnings

import pandas as pd
import wandb
from loguru import logger
# from sklearn.utils import resample
# import os
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from pharm_ai.config import ConfigFilePaths
from pharm_ai.ddp.ddpclassification_model import DDPClassificationModel
from pharm_ai.util.utils import Utilfuncs

Utilfuncs.fix_torch_multiprocessing()
warnings.filterwarnings("ignore")


class ZZTrainerM2:
    def __init__(self):
        self.start_time = ''
        self.end_time = ''

        self.wandb_proj = 'zz'
        self.model_version = 'test'

        self.model_type = 'bert'
        self.pretrained_model = ConfigFilePaths.bert_dir_remote
        self.use_cuda = True
        self.cuda_device = 0
        self.args = ZZTrainerM2.set_model_parameter(model_version=self.model_version)

    def run(self):
        self.train_0713()

    @staticmethod
    def set_model_parameter(model_version, args=ClassificationArgs()):
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
        saved_dirs = ConfigFilePaths.project_dir + '/' + 'zz/m2/'
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
                                          cuda_device=self.cuda_device, args=self.args)

    def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame):
        # deal with dt
        train_df = train_df[['text_a', 'text_b', 'labels']]
        eval_df = eval_df[['text_a', 'text_b', 'labels']]
        train_df['text_a'] = train_df['text_a'].astype(str)
        train_df['text_b'] = train_df['text_b'].astype(str)
        eval_df['text_a'] = eval_df['text_a'].astype(str)
        eval_df['text_b'] = eval_df['text_b'].astype(str)

        # config some parameters
        train_size = train_df.shape[0]
        all_steps = train_size / self.args.train_batch_size
        self.args.logging_steps = int(all_steps / 20)
        self.args.evaluate_during_training_steps = int(all_steps / 6)

        self.args.wandb_project = self.wandb_proj
        self.args.wandb_kwargs = {
            'name': self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
            'tags': [self.model_version, 'train', 'm2']}

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


    def train_0713(self):
        self.time = '2021-07-13'
        self.wandb_proj = 'zz'
        self.model_version = 'v1.7.0.5'

        self.args = ZZTrainerM2.set_model_parameter(model_version=self.model_version)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        self.cuda_device = 4
        self.args.n_gpu = 1

        self.args.num_train_epochs = 5
        self.args.learning_rate = 9e-6
        self.args.train_batch_size = 148
        self.args.eval_batch_size = 128
        # self.args.hyper_args.max_seq_length = 128

        train_df = pd.read_excel("../data/v1.7/processed_dt0708_up.xlsx", 'm2_train_up')
        # train_df = pd.read_excel("../data/v1.7/processed_dt0708.xlsx", 'm2_train')
        eval_df = pd.read_excel("../data/v1.7/processed_dt0708.xlsx", 'm2_eval')
        self.train(train_df, eval_df)


#
# def config_usual_args(model_version):
#     args = ClassificationArgs()
#
#     # multiprocess
#     args.use_multiprocessing = False
#     args.use_multiprocessing_for_evaluation = False
#
#     # base config
#     args.reprocess_input_data = True
#     args.fp16 = False
#     args.manual_seed = 234
#     # args.gradient_accumulation_steps = 8  # ==increase batch size,Use time for memory,
#
#     # save
#     args.no_save = False
#     args.save_eval_checkpoints = False
#     args.save_model_every_epoch = False
#     args.save_optimizer_and_scheduler = True
#     args.save_steps = -1
#
#     # eval
#     args.evaluate_during_training = True
#     args.evaluate_during_training_verbose = True
#
#     args.no_cache = False
#     args.use_early_stopping = False
#     args.encoding = None
#     args.do_lower_case = False
#     args.dynamic_quantize = False
#     args.quantized_model = False
#     args.silent = False
#
#     args.overwrite_output_dir = True
#     saved_dirs = ConfigFilePaths.project_dir + '/' + 'zz/m2/'
#     args.output_dir = saved_dirs + 'outputs/' + model_version + '/'
#     args.cache_dir = saved_dirs + 'cache/' + model_version + '/'
#     args.best_model_dir = saved_dirs + 'best_model/' + model_version + '/'
#     args.tensorboard_dir = saved_dirs + 'runs/' + model_version + '/' + time.strftime("%Y%m%d_%H%M%S",
#                                                                                       time.localtime()) + '/'
#     return args
#
#
# class ZZTrainer2:
#     def __init__(self):
#         self.start_time = ''
#         self.end_time = ''
#         self.pretrained_model = ConfigFilePaths.bert_dir_remote
#         self.wandb_proj = 'zz'
#
#         self.use_cuda = True
#         self.cuda_device = 0
#
#         self.model_version = 'test'
#         self.args = self.get_args()
#         self.args.n_gpu = 1
#
#     def get_args(self):
#         return config_usual_args(model_version=self.model_version)
#
#     def get_train_model(self):
#         self.args.use_cached_eval_features = True
#         if self.args.n_gpu <= 1:
#             return ClassificationModel(model_type='bert', model_name=self.pretrained_model,
#                                        use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args)
#         else:
#             return DDPClassificationModel(model_type='bert', model_name=self.pretrained_model, use_cuda=True,
#                                           cuda_device=self.cuda_device, args=self.args)
#
#     def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame):
#         # deal with dt
#         train_df = train_df[['text_a', 'text_b', 'labels']]
#         eval_df = eval_df[['text_a', 'text_b', 'labels']]
#
#         # config some parameters
#         train_size = train_df.shape[0]
#         all_steps = train_size / self.args.train_batch_size
#         self.args.logging_steps = int(all_steps / 20)
#         self.args.evaluate_during_training_steps = int(all_steps / 5)
#         self.args.eval_batch_size = self.args.train_batch_size
#
#         self.args.wandb_project = self.wandb_proj
#         self.args.wandb_kwargs = {
#             'name': self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
#             'tags': [self.model_version, 'train', 'm2']}
#
#         # get model
#         model = self.get_train_model()
#         print(f'train length: {train_df.shape[0]}')
#         model.args.update_from_dict({'train_length': train_df.shape[0]})
#
#         # train
#         try:
#             # wandb.init()
#             start_time = time.time()
#             model.train_model(train_df=train_df, eval_df=eval_df)
#             logger.info('train finished!!!')
#             end_time = time.time()
#             need_time = round((end_time - start_time) / train_size, 4)
#             training_time = round(need_time * train_size, 4)
#             print(f'train time: {need_time} s * {train_size} = {training_time} s')
#         except Exception as error:
#             logger.error(f'train failed!!! ERROR:{error}')
#             # DTUtils.send_to_me(f'train failed!!! ERROR:{error}')
#         finally:
#             wandb.finish()
#

class ZZM2V1_7(ZZTrainerM2):
    def __init__(self):
        super(ZZM2V1_7, self).__init__()

    def run(self):
        self.train_0713()

    def train_0713(self):
        self.time = '2021-07-13'
        self.model_version = 'v1.7.0.5'
        self.args = ZZTrainerM2.set_model_parameter(model_version=self.model_version )
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        self.cuda_device = 4
        self.args.n_gpu = 1

        self.args.num_train_epochs = 5
        self.args.learning_rate = 9e-6
        self.args.train_batch_size = 148
        self.args.eval_batch_size = 128
        # self.args.hyper_args.max_seq_length = 128

        train_df = pd.read_excel("../data/v1.7/processed_dt0708_up.xlsx", 'm2_train_up')
        # train_df = pd.read_excel("../data/v1.7/processed_dt0708.xlsx", 'm2_train')
        eval_df = pd.read_excel("../data/v1.7/processed_dt0708.xlsx", 'm2_eval')
        self.train(train_df, eval_df)


if __name__ == '__main__':
    ZZTrainerM2().run()
    # ZZM2V1_7().run()
