from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
import logging
from time import time
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.PC.dt import train_test0824
import torch

f = cfp.project_dir + '/patent_claims/claims_non_claims_classifier/'
od = f + 'outputs/20200824/'
cd = f + 'cache/20200824/'
bm = f + 'best_model/20200824/'

torch.cuda.empty_cache()

# Training
t1 = time()
model = ClassificationModel('bert', cfp.bert_dir_remote, cuda_device=1,
                            num_labels=2, use_cuda=True,
                            args={'reprocess_input_data': True,
                                  'use_cached_eval_features': True,
                                  'overwrite_output_dir': True,
                                  'fp16': True,
                                  'num_train_epochs': 3,
                                  'n_gpu': 1,
                                  # 'gradient_accumulation_steps': 20,
                                  'learning_rate': 8e-6,
                                  'logging_steps': 32,
                                  'train_batch_size': 32,
                                  'save_eval_checkpoints': False,
                                  'save_model_every_epoch': False,
                                  'evaluate_during_training': True,
                                  'evaluate_during_training_verbose': True,
                                  'evaluate_during_training_steps': 32,
                                  'use_multiprocessing': True,
                                  'output_dir': od, 'cache_dir': cd,
                                  'best_model_dir': bm})

train1, test1, train2, test2, train3, test3 = train_test0824()

model.train_model(train1, eval_df=test1)
t2 = time()
print(t2-t1)