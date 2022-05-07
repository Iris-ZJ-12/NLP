import os.path
from time import time

import pandas as pd
from loguru import logger

from mart.sm_util.sm_util import hide_labels_arg, auto_rm_outputs_dir
from pharm_ai.config import ConfigFilePaths as cfp
from simpletransformers.ner import NERModel

date='20201204'
od = os.path.join(cfp.project_dir, f'prophet/org_ner/outputs/{date}')
cd = os.path.join(cfp.project_dir, f'prophet/org_ner/cache_dir/{date}')
bm = os.path.join(cfp.project_dir, f'prophet/org_ner/best_model/{date}')

h5 = os.path.join(os.path.dirname(__file__),'data.h5')

logger.add('result.log', filter=lambda record: record["extra"].get("task") == "orgtrain")
orgtrain_logger = logger.bind(task="orgtrain")


class Trainer:
    def __init__(self, labels=None):
        self.output_dir = od
        self.cache_dir = cd
        self.best_model_dir = bm
        self.model_args = {'reprocess_input_data': True,
                           'use_cached_eval_features': True,
                           'overwrite_output_dir': True,
                           'fp16': True,
                           'num_train_epochs': 5,
                           'n_gpu': 1,
                           'learning_rate': 4e-5,
                           'logging_steps': 10,
                           'train_batch_size': 90,
                           'max_seq_length': 190,
                           'save_eval_checkpoints': False,
                           'save_model_every_epoch': False,
                           'evaluate_during_training': True,
                           'evaluate_during_training_verbose': True,
                           'evaluate_during_training_steps': 10,
                           'use_multiprocessing': False,
                           'output_dir': self.output_dir, 'cache_dir': self.cache_dir,
                           'best_model_dir': self.best_model_dir
                           }
        self.label_mode = 'auto'
        if labels is not None:
            self.model_args.labels_list = labels
            self.label_mode = 'non_auto'

    def train(self, cuda_device=-1):
        t1 = time()
        self.model = NERModel('bert', cfp.bert_dir_remote, use_cuda=True,
                              cuda_device=cuda_device, args=self.model_args)
        orgtrain_logger.info('Training begin.')
        if self.label_mode == 'auto':
            func_train_model = hide_labels_arg(self.model)
        elif self.label_mode == 'non_auto':
            func_train_model = self.model.train_model
        train_df = pd.read_hdf(h5, 'dt1204/train')
        func_train_model(train_df, eval_df=self.get_eval_df())
        t2 = time()
        orgtrain_logger.info('Training Done, total running time: {}', t2 - t1)
        auto_rm_outputs_dir()

    def get_eval_df(self):
        eval_df = pd.read_hdf(h5, 'dt1204/test')
        return eval_df


if __name__ == '__main__':
    # labels = ['b-c1', 'b-c2', 'b-c3', 'i-c1', 'i-c2', 'i-c3', 'b-f', 'i-f', 'O']
    x = Trainer()
    x.train(cuda_device=1)
else:
    logger.remove()
