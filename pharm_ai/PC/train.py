from pharm_ai.config import ConfigFilePaths as cfp
from simpletransformers.t5 import T5Model, T5Args
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pharm_ai.PC.dt import GenerativeDataProcessor, ClassificationProcessor
import os
from pharm_ai.util.sm_util import SMUtil
from time import time
from loguru import logger
from pharm_ai.util.utils import Utilfuncs as u
from sklearn.metrics import accuracy_score, f1_score
import wandb

version = 'v4.0'
cuda_device=-1
n_gpu=5
run_with_sweep=False

output_dir = os.path.join(os.path.dirname(__file__), 'outputs', version)
cache_dir = os.path.join(os.path.dirname(__file__), 'cache', version)
best_model_dir = os.path.join(os.path.dirname(__file__), 'best_model', version)

if version=='v2.5':
    model_args=ClassificationArgs(
        n_gpu=n_gpu,
        overwrite_output_dir=True,
        reprocess_input_data=False,
        use_cached_eval_features=True,
        use_multiprocessing=True,
        save_eval_checkpoints=False,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        evaluate_during_training_steps=5,
        train_batch_size=30,
        eval_batch_size=30,
        learning_rate=4e-5,
        num_train_epochs=1,
        output_dir=output_dir,
        best_model_dir=best_model_dir,
        cache_dir=cache_dir,
        wandb_project='patent_claims'
    )
    prepro = ClassificationProcessor(version=version)
else:
    model_args = T5Args(
        n_gpu=n_gpu,
        reprocess_input_data=False,
        use_cached_eval_features=True,
        overwrite_output_dir=True,
        save_eval_checkpoints=False,
        save_model_every_epoch=False,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        evaluate_generated_text=False,
        use_multiprocessing=True,
        use_multiprocessing_for_evaluation=True,
        use_multiprocessed_decoding=True,
        fp16=False,
        no_cache=False,
        train_batch_size=70,
        eval_batch_size=45,
        logging_steps=10,
        max_seq_length=64,
        max_length=10,
        num_train_epochs=1,
        evaluate_during_training_steps=30,
        output_dir=output_dir,
        cache_dir=cache_dir,
        best_model_dir=best_model_dir,
        wandb_project='patent_claims',
        wandb_kwargs={'tags':[version, 'train']}
    )
    prepro = GenerativeDataProcessor(version=version)

train_df, eval_df = prepro.get_train_eval_datasets()

sweep_config = {
    "method": "grid",
    "metric": {"name": "eval_loss", "goal": "minimize"},
    "parameters": {
        "train_batch_size" : {"values": [16, 15, 13, 9, 7, 5, 3, 2]},
        "num_train_epochs": {"values": [2, 1]}
    }
}

def train_model(auto_adjust_batch_size=False, run_sweep=True):
    u.fix_torch_multiprocessing()
    if auto_adjust_batch_size:
        # Auto reduce to best batch_size
        select_cuda = u.select_best_cuda() if cuda_device==-1 else cuda_device
        logger.info('Select cuda_device={}', select_cuda)
        while True:
            try:
                t1 = time()
                model = T5Model('mt5', cfp.mt5_base_remote, args=model_args, cuda_device=select_cuda,
                                accuracy=accuracy_score)
                logger.info('Train with train_batch_size={}', model.args.train_batch_size)
                model.train_model(train_df, eval_data=eval_df)
                t2 = time()
                SMUtil.auto_rm_outputs_dir()
                logger.info('Training spent {} min totally.', (t2-t1)/60)
                break
            except RuntimeError as ex:
                modelargs = model.args
                if model_args.train_batch_size>1:
                    logger.error('train_batch_size is too large! Adjusting to a smaller automatically...')
                    modelargs.train_batch_size = modelargs.train_batch_size*3 //4
                    # create a new instance of the model to avoid "wandb watch" error
                    model = T5Model('mt5', cfp.mt5_base_remote, args=modelargs, cuda_device=select_cuda,
                                    accuracy = accuracy_score)
                    t1 = time()
                else:
                    logger.error('No suitable train_batch_size. Stopping...')
                    break
    else:
        if run_sweep:
            wandb.init()
        select_cuda = (u.select_best_cuda() if cuda_device==-1 else cuda_device) if n_gpu==1 else -1
        logger.info('Select cuda_device={}', select_cuda)
        if version=='v2.5':
            if run_sweep:
                model = ClassificationModel('bert', cfp.bert_dir_remote, num_labels=2, args=model_args,
                                            cuda_device=select_cuda, sweep_config=wandb.config)
            else:
                model = ClassificationModel('bert', cfp.bert_dir_remote, num_labels=2,
                                            args=model_args, cuda_device=select_cuda)
            t1 = time()
            model.train_model(train_df, eval_df=eval_df, f1=f1_score)
        else:
            if run_sweep:
                model = T5Model('mt5', cfp.mt5_base_remote, args=model_args, cuda_device=select_cuda,
                                sweep_config=wandb.config, accuracy=accuracy_score)
            else:
                model = T5Model('mt5', cfp.mt5_base_remote, args=model_args, cuda_device=select_cuda,
                                accuracy=accuracy_score)
            t1 = time()
            model.train_model(train_df, eval_data=eval_df)
        t2 = time()
        if run_sweep:
            wandb.join()
        SMUtil.auto_rm_outputs_dir()
        logger.info('Training spent {} min totally.', (t2-t1)/60)

if run_with_sweep:
    sweep_id = wandb.sweep(sweep_config, project='patent_claims')
    wandb.agent(sweep_id, train_model)
else:
    train_model(run_sweep=False)