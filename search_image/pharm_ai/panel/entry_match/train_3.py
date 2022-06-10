# encoding: utf-8
'''
@author: zyl
@file: train_3.py
@time: 2021/8/31 15:48
@desc:
'''
from pharm_ai.panel.my_utils import MyModel, DTUtils, ModelUtils
from simpletransformers.t5 import T5Args
from pharm_ai.panel.panel_utils import ModelUtils
import time
import warnings

import pandas as pd
import wandb
from loguru import logger
# from sklearn.utils import resample
# import os

from simpletransformers.classification import ClassificationModel, ClassificationArgs, DDPClassificationModel

from pharm_ai.config import ConfigFilePaths
# from pharm_ai.ddp.ddpclassification_model import DDPClassificationModel
from pharm_ai.util.utils import Utilfuncs

Utilfuncs.fix_torch_multiprocessing()
warnings.filterwarnings("ignore")
import os
from scipy.stats import pearsonr, spearmanr
class TryBert(MyModel):
    def __init__(self):
        super(TryBert, self).__init__()
        self.start_time = '2021-08'
        self.end_time = '2021-0~'
        self.bm_version = '~'

        self.wandb_proj = 'panel_entry_match'
        self.use_model = 'classification'  # mt5 /classification
        self.model_type = 'bert'
        self.pretrained_model = ConfigFilePaths.bert_dir_remote

    def run(self):
        self.train_0831()
        # self.eval_0831()
        pass

    def train_0831(self):
        self.model_version = 'emv2.3.0.0'
        # self.pretrained_model = '/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/emv2.2.0.2/'
        self.args = MyModel.set_model_parameter(model_version=self.model_version, args=ClassificationArgs(),
                                                save_dir='panel/entry_match')
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
        self.cuda_device = 0
        self.args.n_gpu = 5

        self.args.num_train_epochs = 5
        self.args.learning_rate = 1e-5
        self.args.train_batch_size = 64  # 512
        self.args.eval_batch_size = 32  # 256
        # self.args.max_seq_length = 128
        self.args.gradient_accumulation_steps = 1  # 256

        train_df = pd.read_excel('./data/em_0830.xlsx', 'train')
        from sklearn.utils import resample

        train_neg = train_df[train_df['target_text']==0][0:12000]
        train_pos = train_df[train_df['target_text']==1]
        train_df = pd.concat([train_neg,train_pos],ignore_index=True)
        train_df = resample(train_df, replace=False)
        train_df = train_df.rename(columns={'input_text':'text','target_text':'labels'})

        eval_df = pd.read_excel('./data/em_0830.xlsx', 'eval')
        eval_neg = eval_df[eval_df['target_text'] == 0][0:1000]
        eval_pos = eval_df[eval_df['target_text'] == 1]
        eval_df = pd.concat([eval_neg, eval_pos], ignore_index=True)
        eval_df = resample(eval_df, replace=False)
        eval_df = eval_df.rename(columns={'input_text': 'text', 'target_text': 'labels'})
        self.train(train_df=train_df, eval_df=eval_df)



if __name__ == '__main__':
    TryBert().run()