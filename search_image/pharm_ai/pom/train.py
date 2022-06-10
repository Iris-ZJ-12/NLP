# -*- coding: UTF-8 -*-
"""
Description : train Pom 😀
"""
import os
import time
import pandas as pd
import torch.multiprocessing

from datetime import datetime
from pharm_ai.pom.dt import TaskManager
from pharm_ai.pom.model import PomModel
from pharm_ai.util.utils import Utilfuncs

torch.cuda.empty_cache()
Utilfuncs.fix_torch_multiprocessing()


class PomTrainer:

    def __init__(self, version, n_gpu, project_name='pom'):  # modify###
        self.ver = version
        self.args = PomModel(version=version, n_gpu=n_gpu, project_name=project_name)
        self.wandb_proj = 'pom'
        self.manager = TaskManager()

    def get_train_model(self):
        return self.args.get_train_model()

    def train(self, task=None, up_sampling=True):
        train_df, eval_df = self.manager.get_train_and_eval_df(task, up_sampling=up_sampling)
        eval_df.to_excel('eval_df.xlsx')
        use_extra = True
        if use_extra:
            self.manager.tasks['content_label'].rename('content_label_long')
            train_df = pd.concat([train_df, self.manager.tasks['content_label'].df], ignore_index=True).sample(frac=1)

        self.args.hyper_args.wandb_kwargs = {'name': self.ver + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
                                             'tags': [self.ver, 'train']}
        # self.args.hyper_args.wandb_project = self.wandb_proj

        model = self.get_train_model()
        cache_dir = self.args.hyper_args.cache_dir
        if os.path.exists(cache_dir):
            cmd = 'rm -rf ' + cache_dir
            os.system(cmd)
        now = datetime.now()
        model.train_model(train_data=train_df, eval_data=eval_df)
        print("Training finished with time:", datetime.now()-now)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,5,6,7'
    # tasks: classification_label, content_label, is_medical_news
    PomTrainer('v1.3', n_gpu=4).train(task=['classification_label', 'content_label_long', 'is_medical_news'], up_sampling=True)
    # PomTrainer('v1.3', n_gpu=4).train(task=['classification_label', 'content_label_long'], up_sampling=True)
