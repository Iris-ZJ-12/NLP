import logging
from time import time
import pandas as pd
import numpy as np
from simpletransformers.seq2seq import Seq2SeqModel
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.prophet.ira_round.dt import train_test
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.util.sm_util import SMUtil

f = cfp.project_dir + '/prophet/ira_round/'
od = f + 'outputs/20201124-2/'
cd = f + 'cache/20201124-2/'
bm = f + 'best_model/20201124-2/'

train, test = train_test()

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 180,
    "learning_rate": 4e-5,
    "train_batch_size": 20,
    "num_train_epochs": 20,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": True,
    "max_length": 10,
    "manual_seed": 4,
    'output_dir': od,
    'cache_dir': cd,
    'best_model_dir': bm}

# Initialize model
model = Seq2SeqModel('bert', cfp.bert_dir_remote, cfp.bert_dir_remote,
                     args=model_args, cuda_device=1)

t1 = time()
# Train the model
model.train_model(train, eval_data=test)
t2 = time()
SMUtil.auto_rm_outputs_dir()

print(str((t2-t1)/60) + ' minutes for training.')