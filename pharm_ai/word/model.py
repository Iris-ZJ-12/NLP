# -*- coding: UTF-8 -*-
"""
Description : 
"""
import time

import torch
from simpletransformers.t5 import T5Model, T5Args

from pharm_ai.config import ConfigFilePaths


class WordModelT5:
    def __init__(self, version, cuda_device=0, n_gpu=1, project_name='word'):
        self.cuda_device = cuda_device
        self.pn = project_name
        self.pre_model = '/large_files/pretrained_pytorch/mt5-base/'
        self.hyper_args = T5Args()

        if torch.cuda.is_available():
            self.use_cuda = True
            self.hyper_args.n_gpu = n_gpu
        else:
            self.use_cuda = False
            self.hyper_args.n_gpu = n_gpu

        # base config
        self.hyper_args.reprocess_input_data = True
        self.hyper_args.use_cached_eval_features = True
        self.hyper_args.fp16 = False
        self.hyper_args.manual_seed = 234
        # "gradient_accumulation_steps": 8,  # ==increase batch size,Use time for memory,

        # save
        self.hyper_args.save_eval_checkpoints = False
        self.hyper_args.save_model_every_epoch = False
        self.hyper_args.save_optimizer_and_scheduler = True
        self.hyper_args.save_steps = -1

        # eval
        self.hyper_args.evaluate_during_training = True
        self.hyper_args.evaluate_during_training_verbose = True

        # output
        self.hyper_args.overwrite_output_dir = True
        self.hyper_args.output_dir = ConfigFilePaths.project_dir + '/' + self.pn + '/outputs/' + version + '/'
        self.hyper_args.cache_dir = ConfigFilePaths.project_dir + '/' + self.pn + '/cache/' + version + '/'
        self.hyper_args.best_model_dir = ConfigFilePaths.project_dir + '/' + self.pn + '/best_model/' + version + '/'
        self.hyper_args.tensorboard_dir = ConfigFilePaths.project_dir + '/' + self.pn + '/runs/' + version + '/' + \
                                          time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '/'

    def get_train_model(self):
        self.hyper_args.use_cached_eval_features = True
        return T5Model(model_type='mt5', model_name=self.pre_model, use_cuda=self.use_cuda,
                       cuda_device=self.cuda_device, args=self.hyper_args)

    def get_predict_model(self):
        self.hyper_args.use_cached_eval_features = True
        self.hyper_args.use_multiprocessing = True
        return T5Model(model_type='mt5', model_name=self.hyper_args.best_model_dir, use_cuda=self.use_cuda,
                       cuda_device=self.cuda_device, args=self.hyper_args)


if __name__ == '__main__':
    pass
