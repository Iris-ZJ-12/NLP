import os
import shutil
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from edit_distance import SequenceMatcher
from loguru import logger
from rouge.rouge import Rouge
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score

from mart.prepro_util.utils import remove_illegal_chars
from mart.sm_util.sm_util import fix_torch_multiprocessing, eval_ner_v2
from mart.utils import email_wrapper
from pharm_ai.config import ConfigFilePaths as cfp
from simpletransformers.classification import ClassificationArgs, ClassificationModel, DDPClassificationModel
from simpletransformers.t5 import DDPT5Model, T5Model, T5Args

email_receiver = 'fanzuquan@pharmcube.com'


class AkModelBase:
    project_root = Path(__file__).parent

    def __init__(self, version, task_id, cuda_devices=None):
        self.output_dir = self.project_root / 'outputs' / version
        self.cache_dir = self.project_root / 'cache' / version
        self.best_model_dir = self.project_root / 'best_model' / f'{version}.{task_id}'

        self.cuda_devices = cuda_devices

    def remove_output_dir(self):
        shutil.rmtree(self.output_dir)
        logger.info(f'Output directory {self.output_dir} removed.')


def get_cuda_device(cuda_device):
    previous = [int(c) for c in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    return previous.index(cuda_device)


def set_cuda_environ(cuda_devices):
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(c) for c in set(cuda_devices))
    else:
        previous = [int(c) for c in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([
            os.environ['CUDA_VISIBLE_DEVICES'],
            *(str(c) for c in set(cuda_devices) if c not in previous)
        ])


class AkT5ModelBase(AkModelBase):
    model_args = T5Args(
        reprocess_input_data=False,
        overwrite_output_dir=True,
        save_eval_checkpoints=False,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        evaluate_generated_text=False,
        save_model_every_epoch=False,
        wandb_project='ak'
    )

    def __init__(self, version, cuda_devices=None, task_id=0):

        super().__init__(version, task_id, cuda_devices)
        self.model_args = copy(AkT5ModelBase.model_args)
        self.model_args.wandb_kwargs = {'tags': [version]}
        self.model_args.output_dir = self.output_dir.as_posix()
        self.model_args.cache_dir = self.cache_dir.as_posix()
        self.model_args.best_model_dir = self.best_model_dir.as_posix()

        if cuda_devices:
            set_cuda_environ(cuda_devices)
            self.model_args.n_gpu = len(self.cuda_devices)

    def update_training_args(self, ddp=True):
        if not ddp:
            self.set_multiprocessing(True)
        else:
            self.set_multiprocessing(False)
        self.model_args.use_cached_eval_features = True,
        self.model_args.wandb_kwargs['tags'].append('train')

    def update_eval_args(self):
        self.model_args.no_cache = True
        self.set_multiprocessing(False)

    def set_multiprocessing(self, value: bool):
        self.model_args.use_multiprocessing = value
        self.model_args.use_multiprocessing_for_evaluation = value
        self.model_args.use_multiprocessed_decoding = value

    def train(self, train_df, eval_df, ddp=True, notification=False, pretrained_model=None, **training_args):
        pre_model = self.get_pretrained_model_path(pretrained_model)
        self.update_training_args(ddp)
        self.model_args.update_from_dict(training_args)
        model_class = DDPT5Model if ddp else T5Model
        self.sm_model = model_class('mt5', pre_model, self.model_args)
        fix_torch_multiprocessing()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if not notification:
                self.sm_model.train_model(train_df, eval_data=eval_df)
            else:
                email_wrapper(self.sm_model.train_model, email_receiver)(train_df, eval_data=eval_df)
        self.remove_output_dir()

    def get_pretrained_model_path(self, pretrained_model):
        if pretrained_model == 'zh_en':
            pre_model = cfp.mt5_zh_en
        else:
            pre_model = cfp.mt5_base_remote
        return pre_model

    def predict(self, prefix, to_predict_texts):
        to_predicts = [p + ': ' + t for p, t in zip(prefix, to_predict_texts)]
        results = self.sm_model.predict(to_predicts)
        return results

    def eval(self, eval_df, delimiter):
        return eval_ner_v2(eval_df, self.sm_model, delimiter=delimiter)


class AkClassificationModelBase(AkModelBase):
    model_args = ClassificationArgs(
        reprocess_input_data=False,
        overwrite_output_dir=True,
        save_eval_checkpoints=False,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        save_model_every_epoch=False,
        no_cache=False,
        wandb_project='ak'
    )

    def __init__(self, version, sub_project_name=None, cuda_devices=None, task_id=0):

        super(AkClassificationModelBase, self).__init__(version, task_id, cuda_devices)

        self.sub_project_name = sub_project_name
        self.model_args = copy(AkClassificationModelBase.model_args)
        self.model_args.wandb_kwargs = {'tags': [version, sub_project_name] if sub_project_name else [version]}
        self.model_args.output_dir = self.output_dir.as_posix()
        self.model_args.cache_dir = self.cache_dir.as_posix()
        self.model_args.best_model_dir = self.best_model_dir.as_posix()

        if cuda_devices:
            set_cuda_environ(cuda_devices)
            self.model_args.n_gpu = len(self.cuda_devices)

    def update_training_args(self):
        self.model_args.use_cached_eval_features = True
        self.model_args.use_multiprocessing = True
        self.model_args.wandb_kwargs['tags'].append('train')

    def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, ddp=True, notification=False, **training_args):
        self.update_training_args()
        self.model_args.update_from_dict(training_args)
        model_class = DDPClassificationModel if ddp else ClassificationModel
        self.sm_model = model_class('bert', cfp.bert_dir_remote, num_labels=get_num_labels(train_df),
                                    args=self.model_args)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if not notification:
                self.sm_model.train_model(train_df, eval_df=eval_df)
            else:
                email_wrapper(self.sm_model.train_model, email_receiver)(train_df, eval_df=eval_df)
        self.remove_output_dir()

    def predict(self, to_predicts: list):
        res, raw_outputs = self.sm_model.predict(to_predicts)
        if not isinstance(raw_outputs, list):
            probs = softmax(raw_outputs, axis=1)
        else:
            probs = np.array(list(map(lambda ar: list(softmax(ar, axis=1).T), raw_outputs)))
        return res.tolist(), probs

    def eval(self, eval_df: pd.DataFrame, label_names=None):
        true_labels = eval_df['labels'].tolist()
        predict_labels, probs = AkClassificationModelBase.predict(self, eval_df['text'].tolist())
        eval_res = f1_score(true_labels, predict_labels, average='macro')
        res_eval_df = pd.DataFrame({'text': eval_df['text'].tolist(), 'true_labels': eval_df['labels'].tolist(),
                                    'predict_labels': predict_labels})
        if label_names:
            for ind, label in enumerate(label_names):
                res_eval_df[f'P(text={label})'] = probs[:, ind]
        else:
            for label in range(eval_df['labels'].nunique()):
                res_eval_df[f'P(text={label}'] = probs[:, label]
        print(classification_report(true_labels, predict_labels, target_names=label_names, digits=4, zero_division=0))
        return eval_res, res_eval_df


def get_num_labels(df: pd.DataFrame):
    return df['labels'].nunique()


def get_rouge_f1_scores(trues, predicts):
    """Compute average rouge scores and edit-distance."""
    rough_obj = Rouge()
    score_avg = rough_obj.get_scores(predicts, trues, avg=True, ignore_empty=True)
    res = {k + '-f1': v['f'] for k, v in score_avg.items()}
    sm = [SequenceMatcher(t, p).ratio() for t, p in zip(trues, predicts)]
    res['avg-edit-distance'] = sum(sm)/len(sm)
    return res

def get_edit_distance(s1, s2):
    return SequenceMatcher(s1, s2).ratio()


def remove_illegal_char(x):
    if x and isinstance(x, str):
        return remove_illegal_chars(x)
    else:
        return x
