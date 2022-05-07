from time import time

import pandas as pd
from loguru import logger
from mart.sm_util.sm_util import auto_rm_outputs_dir

from pharm_ai.config import ConfigFilePaths as cfp
from simpletransformers.classification import ClassificationModel

f = cfp.project_dir + '/prophet/news_filter/'
train_date = ['20201111', '20201123']

logger.add('result.log', filter=lambda record: record["extra"].get("task") == "train")
train_logger = logger.bind(task="train")

h5 = 'data.h5'

class TitleTrainer:
    def __init__(self):
        od = f + 'outputs/{}/'.format(train_date[0])
        cd = f + 'cache/{}/'.format(train_date[0])
        self.bm = f + 'best_model/{}/'.format(train_date[0])
        self.training_args = {'reprocess_input_data': True,
                              'use_cached_eval_features': True,
                              'overwrite_output_dir': True,
                              'fp16': True,
                              'num_train_epochs': 4,
                              'n_gpu': 1,
                              'learning_rate': 8e-5,
                              'logging_steps': 20,
                              'train_batch_size': 200,
                              'save_eval_checkpoints': False,
                              'save_model_every_epoch': False,
                              'evaluate_during_training': True,
                              'evaluate_during_training_verbose': True,
                              'evaluate_during_training_steps': 20,
                              'use_multiprocessing': False,
                              'output_dir': od, 'cache_dir': cd,
                              'best_model_dir': self.bm}

    def do_training(self):
        t1 = time()
        train_df, eval_df = pd.read_hdf(h5, 'v6-1/train'), pd.read_hdf(h5, 'v6-1/test')
        model = ClassificationModel('bert', cfp.bert_dir_remote,
                                    num_labels=4, use_cuda=True,
                                    args=self.training_args)
        model.train_model(train_df, eval_df=eval_df)
        t2 = time()
        train_logger.success('Training model 1 done. Total running time: {} s', t2 - t1)
        auto_rm_outputs_dir()

class UncertainTitleTrainer:
    def __init__(self):
        od = f + 'outputs/{}/'.format(train_date[1])
        cd = f + 'cache/{}/'.format(train_date[1])
        self.bm = f + 'best_model/{}/'.format(train_date[1])
        self.training_args = {'reprocess_input_data': True,
                              'use_cached_eval_features': True,
                              'overwrite_output_dir': True,
                              'fp16': True,
                              'num_train_epochs': 2,
                              'n_gpu': 1,
                              'learning_rate': 8e-6,
                              'logging_steps': 20,
                              'train_batch_size': 200,
                              'save_eval_checkpoints': False,
                              'save_model_every_epoch': False,
                              'evaluate_during_training': True,
                              'evaluate_during_training_verbose': True,
                              'evaluate_during_training_steps': 20,
                              'use_multiprocessing': False,
                              'output_dir': od, 'cache_dir': cd,
                              'best_model_dir': self.bm}

    def do_training(self):
        t1 = time()
        train_df, eval_df = pd.read_hdf(h5, 'v6-2/train'), pd.read_hdf(h5, 'v6-2/test')
        model = ClassificationModel('bert', cfp.bert_dir_remote, num_labels=2, use_cuda=True,
                                    args=self.training_args)
        model.train_model(train_df, eval_df=eval_df)
        t2 = time()
        train_logger.success('Training model 2 done. Total running time: {} s', t2 - t1)
        auto_rm_outputs_dir()

class GroceryTrainer:
    def __init__(self, name="simple_model"):
        from tgrocery import Grocery
        self.model = Grocery(name)
        self.train_df = pd.read_hdf(h5, 'v6-3/train')
        self.mapping = {1:"医药", 2:"非医药"}

    def do_training(self):
        train_df = self.train_df
        train_df['label'] = train_df['label'].map(self.mapping)
        train_src = train_df[['label','text']].apply(tuple, axis=1).tolist()
        self.model.train(train_src)
        self.model.save()


if __name__=='__main__':
    t = GroceryTrainer()
    t.do_training()
else:
    logger.remove()