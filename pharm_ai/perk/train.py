from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from simpletransformers.t5 import T5Model, T5Args, DDPT5Model
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.perk.dt import ClassifyPreprocessor, NerPreprocessor
from sklearn.metrics import f1_score
from pharm_ai.util.sm_util import SMUtil
from pharm_ai.util.utils import Utilfuncs as u
from time import time
import os
import torch
from pathlib import Path

def train(version='v1.0', sub_task:int=None, task='type', use_generative=False,
          cuda_devices:list=None, ddp=True, train_classify=False, t5_model_type='mt5'):
    if cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(c) for c in cuda_devices)
    output_dir = Path('outputs')/version
    cache_dir = Path('cache')/version
    if version.startswith('v1') or train_classify:
        if not use_generative:
            best_model_dir = Path('best_model')/task/(f'{version}.{sub_task}' if sub_task else version)
            args = ClassificationArgs(
                wandb_project='perk',
                wandb_kwargs={'tags':[version, 'classify', task]},
                n_gpu=len(cuda_devices) if cuda_devices else 1,
                reprocess_input_data=True,
                use_cached_eval_features=True,
                overwrite_output_dir=True,
                evaluate_during_training=True,
                evaluate_during_training_verbose=True,
                evaluate_during_training_steps=3,
                sliding_window=True,
                train_batch_size=400,
                eval_batch_size=40,
                learning_rate=8e-5,
                num_train_epochs=2,
                output_dir=output_dir.as_posix(),
                cache_dir=cache_dir.as_posix(),
                best_model_dir=best_model_dir.as_posix()
            )
            prep = ClassifyPreprocessor(version=version)
            task_id = {'type': 0, 'result': 1}.get(task)
            df_train, df_eval = prep.get_train_eval_dataset(task=task_id)
            model = ClassificationModel(
                'bert', cfp.bert_dir_remote, num_labels=df_train['labels'].nunique(),
                args=args
            )
            if version.startswith('v1'):
                model.train_model(df_train, eval_df=df_eval, f1=f1_score)
            else:
                model.train_model(df_train, eval_df=df_eval)
            SMUtil.auto_rm_outputs_dir()
        else:
            best_model_dir = Path('best_model') / 'seq' / (f'{version}.{sub_task}' if sub_task else version)
            t1 = time()
            model_args = Seq2SeqArgs(
                wandb_project='perk',
                wandb_kwargs={'tags':[version, 'classify']},
                n_gpu=len(cuda_devices) if cuda_devices else 1,
                num_train_epochs=50,
                evaluate_generated_text=True,
                evaluate_during_training=True,
                evaluate_during_training_verbose=True,
                save_model_every_epoch=False,
                save_optimizer_and_scheduler=False,
                save_steps=-1,
                overwrite_output_dir=True,
                use_cached_eval_features=True,
                use_multiprocessing=True,
                use_multiprocessing_for_evaluation=True,
                use_multiprocessed_decoding=True,
                output_dir=output_dir.as_posix(),
                cache_dir=cache_dir.as_posix(),
                best_model_dir=best_model_dir.as_posix(),
                train_batch_size=75,
                logging_steps=2,
                evaluate_during_training_steps=2,
                eval_batch_size=10
            )
            model = Seq2SeqModel('bert', cfp.bert_dir_remote, cfp.bert_dir_remote,
                                 args=model_args, cuda_device=-1)
            prep = ClassifyPreprocessor(version=version, generative=True)
            df_train, df_eval = prep.get_train_eval_dataset()
            u.fix_torch_multiprocessing()
            model.train_model(df_train, eval_data=df_eval)
            SMUtil.auto_rm_outputs_dir()
            t2 = time()
            print(f'{(t2-t1)} s was used.')
    elif version in ['v2.0', 'v2.1']:
        # train ner model for drug
        best_model_dir = Path('best_model') / 'ner' / (f'{version}.{sub_task}' if sub_task else version)
        model_args = T5Args(
            wandb_project='perk',
            wandb_kwargs={'tags': [version, 'NER']},
            output_dir=output_dir.as_posix(),
            cache_dir=cache_dir.as_posix(),
            best_model_dir=best_model_dir.as_posix(),
            n_gpu=len(cuda_devices) if cuda_devices else 1,
            reprocess_input_data=False,
            use_cached_eval_features=True,
            overwrite_output_dir=True,
            save_model_every_epoch=False,
            save_eval_checkpoints=False,
            use_multiprocessing=True,
            use_multiprocessing_for_evaluation=True,
            use_multiprocessed_decoding=True,
            fp16=False,
            train_batch_size=200,
            eval_batch_size=100,
            logging_steps=30,
            evaluate_during_training=True,
            evaluate_during_training_steps=20,
            evaluate_during_training_silent=False,
            max_seq_length=64,
            max_length=10,
            num_train_epochs=2
        )
        model_type = 't5' if t5_model_type=='t5' else 'mt5'
        model_path = cfp.mt5_zh_en if t5_model_type=='zh_en' else (cfp.t5_base if t5_model_type=='t5' else cfp.mt5_base_remote)
        u.fix_torch_multiprocessing()
        if ddp:
            model = DDPT5Model(model_type, model_path, args=model_args, cuda_device=-1)
        else:
            model = T5Model(model_type, model_path, args=model_args, cuda_device=-1)
        prepro = NerPreprocessor(version)
        train_df = prepro.get_from_h5('train')
        eval_df = prepro.get_from_h5('eval')
        model.train_model(train_df, eval_data=eval_df)
        SMUtil.auto_rm_outputs_dir(output_dir.as_posix())

if __name__ == '__main__':
    train(version='v2.1', sub_task=3, cuda_devices=[1,2,3,4], ddp=True, t5_model_type='t5')