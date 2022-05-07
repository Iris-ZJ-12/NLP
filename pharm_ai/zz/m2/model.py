"""
Author      : Zhang Yuliang
Datetime    : 2021/1/25 下午9:17
Description : 
"""
import time

import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from pharm_ai.config import ConfigFilePaths
from pharm_ai.ddp.ddpclassification_model import DDPClassificationModel


class ZZModel2:
    def __init__(self, model_version, cuda_device, n_gpu, project_name='zz'):
        self.saved_dirs = ConfigFilePaths.project_dir + '/' + project_name
        self.model_version = model_version
        self.model_type = 'bert'

        self.hyper_args = ClassificationArgs()
        if torch.cuda.is_available():
            self.use_cuda = True
            self.hyper_args.use_cuda = True
        else:
            self.use_cuda = False
            self.hyper_args.use_cuda = True

        self.hyper_args.pretrained_model = ConfigFilePaths.bert_dir_remote
        self.pretrained_model = ConfigFilePaths.bert_dir_remote
        self.hyper_args.model_version = model_version
        self.model_version = model_version
        self.hyper_args.cuda_device = cuda_device
        self.cuda_device = cuda_device
        self.hyper_args.project_name = project_name
        self.project_name = project_name
        self.hyper_args.n_gpu = n_gpu

        # multiprocess
        self.hyper_args.use_multiprocessing = False
        self.hyper_args.use_multiprocessing_for_evaluation = False

        # base config
        self.hyper_args.reprocess_input_data = True
        self.hyper_args.fp16 = False
        self.hyper_args.manual_seed = 234
        # "gradient_accumulation_steps": 8,  # ==increase batch size,Use time for memory,

        # save
        self.hyper_args.no_save = False
        self.hyper_args.save_eval_checkpoints = False
        self.hyper_args.save_model_every_epoch = False
        self.hyper_args.save_optimizer_and_scheduler = True
        self.hyper_args.save_steps = -1

        # eval
        self.hyper_args.evaluate_during_training = True
        self.hyper_args.evaluate_during_training_verbose = True

        self.hyper_args.no_cache = False
        self.hyper_args.use_early_stopping = False
        self.hyper_args.encoding = None
        self.hyper_args.do_lower_case = False
        self.hyper_args.dynamic_quantize = False
        self.hyper_args.quantized_model = False
        self.hyper_args.silent = False
        # self.hyper_args.not_saved_args = []
        self.hyper_args.wandb_project = 'test'
        self.hyper_args.wandb_kwargs = dict()
        # self.saved_drs()

    def set_saved_dirs(self):
        self.hyper_args.overwrite_output_dir = True
        self.hyper_args.output_dir = self.saved_dirs + '/m2/outputs/' + self.model_version + '/'
        self.hyper_args.cache_dir = self.saved_dirs + '/m2/cache/' + self.model_version + '/'
        self.hyper_args.best_model_dir = self.saved_dirs + '/m2/best_model/' + self.model_version + '/'
        self.hyper_args.tensorboard_dir = self.saved_dirs + '/m2/runs/' + self.model_version + '/' + \
                                          time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '/'

    def get_train_model(self):
        self.hyper_args.use_cached_eval_features = True
        self.set_saved_dirs()
        if self.hyper_args.n_gpu <= 1:
            return ClassificationModel(model_type=self.model_type, model_name=self.pretrained_model,
                                       use_cuda=self.use_cuda,
                                       cuda_device=self.cuda_device, args=self.hyper_args)
        else:
            return DDPClassificationModel(model_type=self.model_type, model_name=self.pretrained_model, use_cuda=True,
                                          cuda_device=self.cuda_device, args=self.hyper_args)

    def get_predict_model(self):
        self.hyper_args.use_cached_eval_features = False
        self.set_saved_dirs()
        return ClassificationModel(model_type=self.model_type, model_name=self.hyper_args.best_model_dir,
                                   use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.hyper_args)


if __name__ == '__main__':
    pass
