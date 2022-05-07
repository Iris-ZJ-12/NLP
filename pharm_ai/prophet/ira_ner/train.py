from simpletransformers.ner import NERModel
import torch
from time import time
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.util.sm_util import SMUtil
import pandas as pd

# import os
# d = os.path.abspath(os.path.dirname(__file__))
torch.cuda.empty_cache()
od = cfp.project_dir + '/prophet/ira_ner/outputs/20201201/'
cd = cfp.project_dir + '/prophet/ira_ner/cache/20201201/'
bm = cfp.project_dir + '/prophet/ira_ner/best_model/20201201/'

h5 = 'train_test_20201201.h5'
train = pd.read_hdf(h5, 'train')
test = pd.read_hdf(h5, 'test')

labels = ['b-financed#', 'i-financed#', 'b-investee', 'i-investee', 'O']

t1 = time()
# Training
model = NERModel('bert', cfp.bert_dir_remote, use_cuda=True,
                 cuda_device=1, args={'reprocess_input_data': True,
                                      'use_cached_eval_features': True,
                                      'overwrite_output_dir': True,
                                      'fp16': False,
                                      'num_train_epochs': 7,
                                      'n_gpu': 1,
                                      'learning_rate': 2e-4,
                                      'logging_steps': 5,
                                      'train_batch_size': 80,
                                      'max_seq_length': 190,
                                      'save_eval_checkpoints': False,
                                      'save_model_every_epoch': False,
                                      'evaluate_during_training': True,
                                      'evaluate_during_training_verbose': True,
                                      'evaluate_during_training_steps': 5,
                                      'use_multiprocessing': False,
                                      'output_dir': od, 'cache_dir': cd,
                                      'best_model_dir': bm}, labels=labels)

model.train_model(train, eval_df=test)

t2 = time()
print(t2-t1)