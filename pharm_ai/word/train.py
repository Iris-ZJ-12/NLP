# -*- coding: UTF-8 -*-
"""
Description : 
"""

import pandas as pd
import torch
from sklearn.utils import resample
import wandb
from pharm_ai.word.model import WordModelT5

torch.cuda.empty_cache()


class WordTrainerT5:
    def __init__(self, version, cuda_device, n_gpu):
        self.ver = version
        self.args = WordModelT5(version=version, cuda_device=cuda_device, n_gpu=n_gpu)

        self.args.hyper_args.num_train_epochs = 3
        self.args.hyper_args.learning_rate = 1e-4
        self.args.hyper_args.train_batch_size = 24
        self.args.hyper_args.max_seq_length = 128

        self.args.hyper_args.use_cached_eval_features = True

        self.args.hyper_args.eval_batch_size = self.args.hyper_args.train_batch_size
        self.args.hyper_args.wandb_kwargs = {'name': version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
                                             'tags': ['.'.join(version.split('.')[0:-1]) + '_train'],
                                             }

        self.args.hyper_args.wandb_project = self.args.pn

    def get_model(self):
        return self.args.get_train_model()

    def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, shuffled=True):
        if shuffled:
            train_df = resample(train_df, random_state=42, replace=False)

        train_size = train_df.shape[0]
        all_steps = train_size / self.args.hyper_args.train_batch_size
        self.args.hyper_args.logging_steps = int(all_steps / 20)
        self.args.hyper_args.evaluate_during_training_steps = int(all_steps / 5)

        # model
        model = self.get_model()

        # train
        start_time = time.time()
        global_step, training_details = model.train_model(train_data=train_df, eval_data=eval_df)
        end_time = time.time()
        need_time = round((end_time - start_time) / train_size, 5)

        print('training details:', training_details)
        print('training time: {}s * {} = {}s'.format(need_time, train_size, round(need_time * train_size, 5)))

        return global_step, training_details


class V2_0(WordTrainerT5):
    def __init__(self, version, cuda_device=-1, n_gpu=2):
        super(V2_0, self).__init__(version=version, cuda_device=cuda_device, n_gpu=n_gpu)

    def do_train_2(self):
        train_df = pd.read_excel("./data/v2.0/v2_0_2_0309.xlsx", sheet_name='train')
        eval_df = pd.read_excel("./data/v2.0/v2_0_2_0309.xlsx", sheet_name='eval')
        train_df = train_df.astype('str')
        label_1_df = train_df[train_df['target_text'] != '0']

        up_df = resample(label_1_df, replace=True, n_samples=4000)
        train_df = pd.concat([train_df, up_df], ignore_index=True)

        eval_df = eval_df.astype('str')
        self.train(train_df, eval_df)

    def do_train_5(self):
        train_df = pd.read_excel("./data/v2.0/v2_0_5_0310_160449.xlsx", sheet_name='train')
        eval_df = pd.read_excel("./data/v2.0/v2_0_5_0310_160449.xlsx", sheet_name='eval')
        train_df = train_df.astype('str')
        label_1_df = train_df[train_df['target_text'] != '|']

        up_df = resample(label_1_df, replace=True, n_samples=4000)
        train_df = pd.concat([train_df, up_df], ignore_index=True)

        eval_df = eval_df.astype('str')
        self.train(train_df, eval_df)

    def do_train_test(self):
        train_df = pd.read_excel("./data/v2.0/v2_0_2_0309.xlsx", sheet_name='train')[0:50]
        eval_df = pd.read_excel("./data/v2.0/v2_0_2_0309.xlsx", sheet_name='eval')[0:50]
        train_df = train_df.astype('str')
        eval_df = eval_df.astype('str')
        self.train(train_df, eval_df)


if __name__ == '__main__':
    import time

    V2_0(version='v2.0.t').do_train_test()
