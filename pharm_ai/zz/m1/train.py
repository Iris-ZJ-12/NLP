import time

import pandas as pd
import wandb
from loguru import logger
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from pharm_ai.config import ConfigFilePaths
from pharm_ai.ddp.ddpclassification_model import DDPClassificationModel


class ZZTrainerM1:
    def __init__(self):
        self.start_time = ''
        self.end_time = ''

        self.wandb_proj = 'zz'
        self.model_version = 'test'

        self.model_type = 'bert'
        self.pretrained_model = ConfigFilePaths.bert_dir_remote
        self.use_cuda = True
        self.cuda_device = 0
        self.num_labels = 2
        self.args = ZZTrainerM1.set_model_parameter(model_version=self.model_version)

    def run(self):
        self.train_0708()

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
        saved_dirs = ConfigFilePaths.project_dir + '/' + 'zz/m1/'
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
                                       num_labels=self.num_labels,
                                       use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args)
        else:
            return DDPClassificationModel(model_type=self.model_type, model_name=self.pretrained_model, use_cuda=True,
                                          cuda_device=self.cuda_device, args=self.args)

    def get_predict_model(self):
        self.args.use_cached_eval_features = False
        return ClassificationModel(model_type=self.model_type, model_name=self.args.best_model_dir,
                                   num_labels=self.num_labels, use_cuda=self.use_cuda, cuda_device=self.cuda_device,
                                   args=self.args)

    def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame):
        # deal with dt
        train_df = train_df[['text', 'labels']]
        eval_df = eval_df[['text', 'labels']]

        # config some parameters
        train_size = train_df.shape[0]
        all_steps = train_size / self.args.train_batch_size
        self.args.logging_steps = int(all_steps / 20)
        self.args.evaluate_during_training_steps = int(all_steps / 6)

        self.args.wandb_project = self.wandb_proj
        self.args.wandb_kwargs = {
            'name': self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
            'tags': [self.model_version, 'train', 'm1']}

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

    def train_0708(self):
        self.time = '2021-07-08'
        self.wandb_proj = 'zz'
        self.model_version = 'v1.7.0.5'

        self.args = ZZTrainerM1.set_model_parameter(model_version=self.model_version)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        self.cuda_device = 4
        self.args.n_gpu = 1

        self.args.num_train_epochs = 5
        self.args.learning_rate = 6e-6
        self.args.train_batch_size = 108
        self.args.eval_batch_size = 108
        train_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/processed_dt0708_up.xlsx", 'm1_train_up')
        eval_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/processed_dt0708.xlsx", 'm1_eval')
        self.train(train_df=train_df, eval_df=eval_df)


# class ZZTrainerM1:
#     def __init__(self, version, h5='../data/zz.h5'):
#         # sweep_configs = {
#         #     "method": "bayes",  # grid, random
#         #     "metric": {"name": "f1", "goal": "maximize"},
#         #     "parameters": {
#         #         "num_train_epochs": {"values": [3]},
#         #         "learning_rate": {"values": [5e-6, 4e-7, 2e-4, 9e-7]},
#         #         "train_batch_size": {"values": [96]},
#         #     },
#         # }
#         # self.sweep_id = wandb.sweep(sweep_configs, project="zz_m1_" + version)
#         # self.h5 = h5
#         self.ver = version
#         self.m1 = ZZModelM1(version=version)
#         self.m1.hyper_args.num_train_epochs = 5
#         self.m1.hyper_args.train_batch_size = 108
#         self.m1.hyper_args.eval_batch_size = self.m1.hyper_args.train_batch_size
#         self.m1.hyper_args.learning_rate = 6e-6
#         self.m1.hyper_args.n_gpu =4
#         # self.m1.hyper_args.wandb_project = 'zz_m1_' + version
#
#     def get_model(self, use_sweep=False):
#         if use_sweep:
#             return self.m1.get_train_model(use_sweep=wandb.config)
#         else:
#             return self.m1.get_train_model()
#
#     def train(self, train_df, eval_df, up_sampling=False, shuffled=False):
#         if up_sampling:
#             train_df = None
#         if shuffled:
#             train_df = None
#
#         train_dt_length = self.get_train_data_length(train_df)
#         one_epoch_steps = train_dt_length / self.m1.hyper_args.train_batch_size
#         self.m1.hyper_args.logging_steps = int(one_epoch_steps / 10)
#         self.m1.hyper_args.evaluate_during_training_steps = int(one_epoch_steps / 5)
#         self.m1.hyper_args.wandb_kwargs = {
#             'name': self.ver + time.strftime("_m1_%m%d_%H:%M:%S", time.localtime()),
#             'tags': [self.ver, 'train','m1']}
#         self.m1.hyper_args.wandb_project = 'zz'
#
#         # model
#         model = self.get_model(use_sweep=False)
#
#         # train
#         global_step, training_details = model.train_model(train_df=train_df, eval_df=eval_df, show_running_loss=True
#                                                           )
#
#         return global_step, training_details
#
#     def get_train_data_length(self, train_df: pd.DataFrame):
#         length = len(train_df)
#         return length

# def do_train(self):
#     start_time = time.time()
#     train_df = pd.read_hdf(self.h5, self.ver + '/m1_train')
#     eval_df = pd.read_hdf(self.h5, self.ver + '/m1_eval')
#
#     global_step, training_details = self.train(train_df=train_df, eval_df=eval_df, up_sampling=False,
#                                                shuffled=False)
#     print('global step:', global_step)
#     print('training details:', training_details)
#     # SMUtil().auto_rm_outputs_dir(outputs_dir=self.model.args.output_dir)
#     end_time = time.time()
#     print('m1 training time: {}s * {}'.format((end_time - start_time) / train_df.shape[0], train_df.shape[0]))
#
#     # Sync wandb
#     wandb.join()
#
# def do(self):
#     wandb.agent(self.sweep_id, self.do_train)

# def train_0708(self):
#     train_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/processed_dt0708_up.xlsx",'m1_train_up')
#     eval_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/processed_dt0708.xlsx",'m1_eval')
#     global_step, training_details = self.train(train_df=train_df, eval_df=eval_df, up_sampling=False,
#                                                shuffled=False)
#     print('global step:', global_step)

if __name__ == '__main__':
    ZZTrainerM1().run()
    # ZZTrainerM1('v7_0708').train_0708()
    # ZZTrainerM1('v6_0128').do_train()
    # ZZTrainerM1('v6_0126').do()
