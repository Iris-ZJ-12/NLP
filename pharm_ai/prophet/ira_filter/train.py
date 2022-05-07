from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
import logging

from time import time
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.util.sm_util import SMUtil
from pharm_ai.prophet.ira_filter.dt import train_test
import torch

f = cfp.project_dir + '/prophet/ira_filter/'
od = f + 'outputs/20201123/'
cd = f + 'cache/20201123/'
bm = f + 'best_model/20201123/'

torch.cuda.empty_cache()

# Training
t1 = time()
model = ClassificationModel('bert', cfp.bert_dir_remote, cuda_device=1,
                            num_labels=2, use_cuda=True,
                            args={'reprocess_input_data': True,
                                  'use_cached_eval_features': True,
                                  'overwrite_output_dir': True,
                                  'fp16': False,
                                  'num_train_epochs': 7,
                                  'n_gpu': 1,
                                  # 'gradient_accumulation_steps': 20,
                                  'learning_rate': 8e-7,
                                  'logging_steps': 40,
                                  'train_batch_size': 80,
                                  'save_eval_checkpoints': False,
                                  'save_model_every_epoch': False,
                                  'evaluate_during_training': True,
                                  'evaluate_during_training_verbose': True,
                                  'evaluate_during_training_steps': 40,
                                  'use_multiprocessing': True,
                                  'output_dir': od, 'cache_dir': cd,
                                  'best_model_dir': bm})

train, test = train_test()

model.train_model(train, eval_df=test)
t2 = time()
SMUtil.auto_rm_outputs_dir(outputs_dir=od)
print(t2-t1)