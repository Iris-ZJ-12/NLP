from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from simpletransformers.t5 import T5Model, T5Args
from pharm_ai.prophet.ira.prepro import IraPreprocessor
import os
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.util.sm_util import SMUtil
from pharm_ai.util.utils import Utilfuncs as u
from time import time
from loguru import logger
from sklearn.metrics import accuracy_score
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import wandb


u.fix_torch_multiprocessing()


def do_train(version='v2.0', cuda_device=-1, n_gpu=1, run_name=None, task=None):
    prepro = IraPreprocessor(version)
    train_df, eval_df = prepro.get_train_eval_dataset(task=task)
    output_dir = os.path.join(cfp.project_dir, 'prophet', 'ira', 'outputs', version)
    cache_dir = os.path.join(cfp.project_dir, 'prophet', 'ira', 'cache', version)
    best_model_dir = os.path.join(cfp.project_dir, 'prophet', 'ira', 'best_model', version)
    if version=='v2.6' and task:
        output_dir = os.path.join(output_dir, task)
        cache_dir = os.path.join(cache_dir, task)
        best_model_dir = os.path.join(best_model_dir, task)
    if n_gpu==2:
        cuda_device=-1
    elif cuda_device==-1 and n_gpu==1:
        cuda_device = select_best_cuda()
        logger.info('Select cuda_device={}', cuda_device)
    if version in ['v2.0', 'v2.6']:
        model_args = Seq2SeqArgs(
            n_gpu=n_gpu,
            reprocess_input_data=False,
            overwrite_output_dir=True,
            learning_rate=6e-5,
            train_batch_size=12,
            eval_batch_size=12,
            num_train_epochs=22,
            max_seq_length=200,
            max_length=15,
            save_eval_checkpoints=False,
            save_model_every_epoch=False,
            evaluate_during_training=True,
            evaluate_during_training_verbose=True,
            evaluate_during_training_steps=200,
            logging_steps=10,
            evaluate_generated_text=True,
            use_cached_eval_features=True,
            no_cache=False,
            use_multiprocessing=True,
            output_dir=output_dir,
            cache_dir=cache_dir,
            best_model_dir=best_model_dir,
            scheduler='constant_schedule_with_warmup',
            wandb_project='prophet'
        )
        if run_name:
            model_args.update_from_dict({'wandb_kwargs': {'name': run_name}})

        model = Seq2SeqModel('bert', cfp.bert_dir_remote, cfp.bert_dir_remote,
                             args=model_args, cuda_device=cuda_device)

    else:
        # use t5 model
        model_args = T5Args(
            n_gpu=n_gpu,
            dataloader_num_workers=14,
            num_train_epochs=7,
            learning_rate=4e-5,
            train_batch_size=10,
            eval_batch_size=10,
            max_seq_length=300,
            max_length=10,
            save_eval_checkpoints=False,
            save_model_every_epoch=False,
            reprocess_input_data=False,
            use_cached_eval_features=True,
            no_cache=False,
            evaluate_during_training=True,
            evaluate_during_training_verbose=True,
            evaluate_during_training_steps=250,
            overwrite_output_dir=True,
            use_multiprocessing=True,
            use_multiprocessing_for_evaluation=True,
            evaluate_generated_text=True,
            fp16=False,
            logging_steps=10,
            output_dir=output_dir,
            cache_dir=cache_dir,
            best_model_dir=best_model_dir,
            wandb_project='prophet'
        )
        if run_name:
            model_args.update_from_dict({'wandb_kwargs': {'name': run_name}})

        # v2.3 and v2.4 use mt5_small model
        pretrained_path = cfp.mt5_base_remote if version in ['v2.1', 'v2.2', 'v2.5'] else cfp.mt5_small_remote
        model = T5Model('mt5', pretrained_path, model_args, cuda_device=cuda_device)

    t1 = time()
    # logger.debug('dataloader_num_workers={}', model.args.dataloader_num_workers)
    model.train_model(train_df, eval_data=eval_df, accuracy = accuracy_score,
                      rouge=lambda trues, preds: u.get_edit_distance_ratios(trues, preds)['edit_distances'].mean())
    t2 = time()
    SMUtil.auto_rm_outputs_dir()
    logger.info('Training spent {} min totally.', (t2-t1)/60)

def select_best_cuda():
    nvmlInit()
    h0 = nvmlDeviceGetHandleByIndex(0)
    info0 = nvmlDeviceGetMemoryInfo(h0)
    h1 = nvmlDeviceGetHandleByIndex(1)
    info1 = nvmlDeviceGetMemoryInfo(h1)
    res_cuda = 0 if info0.free>info1.free else 1
    return res_cuda

def do_sweep(sweep_name=None, sweep_id=None, *args, **kwargs):
    if not sweep_id:
        sweep_config = {
            "method": "bayes",
            "metric": {"goal": "minimize",
                "name": "eval_loss"
            },
            "parameters": {
                "learning_rate":{
                    "min": 1e-6,
                    "max": 1e-4,
                    "distribution": "uniform"
                },
                "num_train_epochs":{
                    "min": 15,
                    "max": 25,
                    "distribution": "int_uniform"
                },
                "max_seq_length":{
                    "min": 150,
                    "max": 300,
                    "distribution": "int_uniform"
                },
                "max_length":{
                    "min": 8,
                    "max": 15,
                    "distribution": "int_uniform"
                },
                "scheduler": {
                    "values": [
                        "constant_schedule_with_warmup",
                        "linear_schedule_with_warmup",
                        "cosine_schedule_with_warmup",
                        "cosine_with_hard_restarts_schedule_with_warmup",
                        "polynomial_decay_schedule_with_warmup"
                    ],
                    "distribution": "categorical"
                }
            }
        }
        if sweep_name:
            sweep_config.update({"name": sweep_name})

        # create sweep
        sweep_id = wandb.sweep(sweep_config, project="prophet")

    # launch agent
    def train_fun():
        wandb.init()
        do_train(*args, **kwargs)
        wandb.join()
    wandb.agent(sweep_id, function=train_fun)


if __name__ == '__main__':
    do_sweep(sweep_id='7tay4xm1', version='v2.6', cuda_device=-1, n_gpu=2, task='amount')