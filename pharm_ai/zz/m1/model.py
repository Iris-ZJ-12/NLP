"""
Author      : Zhang Yuliang
Datetime    : 2021/1/25 下午9:17
Description : 
"""
import torch
from pharm_ai.config import ConfigFilePaths as cfp
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pharm_ai.ddp.ddpclassification_model import DDPClassificationModel


class ZZModelM1:
    def __init__(self, version, cuda_device=3):
        torch.cuda.empty_cache()
        self.cuda_device = cuda_device
        self.num_labels = 2
        self.pre_model = cfp.bert_dir_remote
        self.hyper_args = ClassificationArgs()

        if torch.cuda.is_available():
            self.use_cuda = True
            self.hyper_args.n_gpu = 1
        else:
            self.use_cuda = False
            self.hyper_args.n_gpu = 1

        # base config
        self.hyper_args.reprocess_input_data = True
        self.hyper_args.use_cached_eval_features = True

        self.hyper_args.fp16 = False
        self.hyper_args.manual_seed = 234

        # save
        self.hyper_args.save_eval_checkpoints = False
        self.hyper_args.save_model_every_epoch = False
        self.hyper_args.save_optimizer_and_scheduler = False
        self.hyper_args.save_steps = -1

        # eval
        self.hyper_args.evaluate_during_training = True
        self.hyper_args.evaluate_during_training_verbose = True

        # output
        self.hyper_args.overwrite_output_dir = True
        self.hyper_args.output_dir = cfp.project_dir + '/zz/m1/outputs/' + version + '/'
        self.hyper_args.cache_dir = cfp.project_dir + '/zz/m1/cache/' + version + '/'
        self.hyper_args.best_model_dir = cfp.project_dir + '/zz/m1/best_model/' + version + '/'
        # self.model_args.tensorboard_dir = None

    def get_train_model(self, use_sweep=False):
        if use_sweep:
            return ClassificationModel(model_type='bert', model_name=self.pre_model, num_labels=self.num_labels,
                                       use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.hyper_args,
                                       sweep_config=use_sweep)
        else:
            return DDPClassificationModel(model_type='bert', model_name=self.pre_model, num_labels=self.num_labels,
                                          use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.hyper_args)

    def get_predict_model(self):
        self.hyper_args.use_multiprocessing = False
        self.hyper_args.use_cached_eval_features = False
        return ClassificationModel(model_type='bert', model_name=self.hyper_args.best_model_dir,
                                   num_labels=self.num_labels, use_cuda=self.use_cuda, cuda_device=self.cuda_device,
                                   args=self.hyper_args)


if __name__ == '__main__':
    t = ZZModelM1(version='1')
    print(t.hyper_args)
