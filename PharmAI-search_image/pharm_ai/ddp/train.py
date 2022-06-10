import os
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from simpletransformers.t5 import T5Model, T5Args
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.perk.dt import NerPreprocessor
from pharm_ai.util.utils import Utilfuncs as u
from datetime import datetime
# from ddpt5_model import DDPT5Model
from simpletransformers.t5 import DDPT5Model

def train(version='v1.0', task='type', use_generative=False, n_gpu=2, cuda_device=-1, ddp=True, train_classify=False):
    output_dir = f'outputs/{version}/'
    cache_dir = f'cache/{version}'
    if version.startswith('v1') or train_classify:
        pass
    elif version in ['v2.0', 'v2.1']:
        # train ner model for drug
        best_model_dir = f'best_model/ner/{version}'
        model_args = T5Args(
            wandb_project='perk',
            wandb_kwargs={'tags': [version, 'NER']},
            output_dir=output_dir,
            cache_dir=cache_dir,
            best_model_dir=best_model_dir,
            n_gpu=n_gpu,
            reprocess_input_data=False,
            use_cached_eval_features=True,
            overwrite_output_dir=True,
            save_model_every_epoch=False,
            save_eval_checkpoints=False,
            save_optimizer_and_scheduler=False,
            use_multiprocessing=False,
            use_multiprocessing_for_evaluation=False,
            use_multiprocessed_decoding=True,
            fp16=True,
            train_batch_size=128,
            eval_batch_size=128,
            logging_steps=10,
            warmup_ratio=0.1,
            evaluate_during_training=True,
            evaluate_during_training_steps=10,
            evaluate_during_training_silent=False,
            max_seq_length=64,
            max_length=10,
            num_train_epochs=2,
            learning_rate=1e-4,
            r_drop=True
        )
        u.fix_torch_multiprocessing()
        path = '/home/clr/disk/playground/models/mt5_zh_en/'
        # path = cfp.mt5_base_remote
        if ddp:
            model = DDPT5Model('mt5', path, args=model_args)
        else:
            model = T5Model('mt5', path, args=model_args)
        prepro = NerPreprocessor(version)
        train_df = prepro.get_from_h5('train')
        eval_df = prepro.get_from_h5('eval')
        model.train_model(train_df, eval_data=eval_df)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    now = datetime.now()
    train(version='v2.1', n_gpu=4, use_generative=True, ddp=True)
    print("Training finished with time:", datetime.now()-now)
