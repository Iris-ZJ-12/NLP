from simpletransformers.classification import ClassificationModel
from pharm_ai.config import ConfigFilePaths as cfp
from time import time
from pharm_ai.prophet.org_filter.dt import OrgFilterPreprocessor
import pandas as pd
from sklearn.metrics import f1_score
from loguru import logger

class OrgFilterTrainer:
    def __init__(self, args={}, version='v1' ,date='20201128'):
        self.preprocessor = OrgFilterPreprocessor(version=version)
        self.h5 = self.preprocessor.h5
        f = cfp.project_dir + '/prophet/org_filter/'
        outputs_dir = f + f'outputs/{date}/'
        cache_dir = f + f'cache/{date}/'
        self.bestmodel_dir = f + f'best_model/{date}/'
        self.modal_args = {'reprocess_input_data': True,
                           'use_cached_eval_features': True,
                           'overwrite_output_dir': True,
                           'fp16': False,
                           'n_gpu': 1,
                           'save_eval_checkpoints': False,
                           'save_model_every_epoch': False,
                           'evaluate_during_training': True,
                           'evaluate_during_training_verbose': True,
                           'evaluate_during_training_steps': 40,
                           'use_multiprocessing':False,
                           'output_dir': outputs_dir, 'cache_dir': cache_dir,
                           'best_model_dir': self.bestmodel_dir}
        self.modal_args.update(args)
        logger.add('train.log')

    def train(self, cuda_device=-1):
        t1 = time()
        logger.info("Training args: {}", self.modal_args)
        model = ClassificationModel('bert', cfp.bert_dir_remote, 2, use_cuda=True, cuda_device=cuda_device,
                                    args = self.modal_args)
        train_df = pd.read_hdf(self.h5, self.preprocessor.h5_keys['train'])
        eval_df = pd.read_hdf(self.h5, self.preprocessor.h5_keys['test'])
        model.train_model(train_df, eval_df=eval_df, f1_score = f1_score)
        t2 = time()
        logger.info("Training time :{}", t2-t1)



if __name__ == '__main__':
    x=OrgFilterTrainer(args={'num_train_epochs': 3,
                             'learning_rate': 8e-5,
                             'logging_steps': 40,
                             'train_batch_size': 150
                             },
                       version='v1-1',
                       date='20201204')
    x.train(cuda_device=1)