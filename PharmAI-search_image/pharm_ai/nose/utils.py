import os
import re
import shutil
import warnings
from copy import copy, deepcopy
from functools import partial
from pathlib import Path

import Levenshtein
import json
import numpy as np
import pandas as pd
from edit_distance import SequenceMatcher
from loguru import logger
from rouge.rouge import Rouge
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from mart.prepro_util.utils import remove_illegal_chars
from mart.sm_util.sm_util import fix_torch_multiprocessing, eval_ner_v2
from mart.utils import email_wrapper
from pharm_ai.config import ConfigFilePaths as cfp
from simpletransformers.classification import ClassificationArgs, ClassificationModel, DDPClassificationModel
from simpletransformers.t5 import DDPT5Model, T5Model, T5Args

email_receiver = 'fanzuquan@pharmcube.com'


class NoseModelBase:
    project_root = Path(__file__).parent

    def __init__(self, version, task_id, cuda_devices=None):
        self.version = version
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


class NoseT5ModelBase(NoseModelBase):
    model_args = T5Args(
        reprocess_input_data=False,
        overwrite_output_dir=True,
        save_eval_checkpoints=False,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        evaluate_generated_text=False,
        save_model_every_epoch=False,
        wandb_project='nose'
    )

    def __init__(self, version, cuda_devices=None, task_id=0):

        super().__init__(version, task_id, cuda_devices)
        self.model_args = copy(NoseT5ModelBase.model_args)
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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            to_predicts = [p + ': ' + t for p, t in zip(prefix, to_predict_texts)]
            results = self.sm_model.predict(to_predicts)
        return results

    def eval(self, eval_df, delimiter):
        return eval_ner_v2(eval_df, self.sm_model, delimiter=delimiter)


class NoseClassificationModelBase(NoseModelBase):
    model_args = ClassificationArgs(
        reprocess_input_data=False,
        overwrite_output_dir=True,
        save_eval_checkpoints=False,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        save_model_every_epoch=False,
        no_cache=False,
        wandb_project='nose'
    )
    TEXT_COL = ['text']

    def __init__(self, version, sub_project_name, cuda_devices=None, task_id=0):

        super(NoseClassificationModelBase, self).__init__(version, task_id, cuda_devices)

        self.sub_project_name = sub_project_name
        self.model_args = copy(NoseClassificationModelBase.model_args)
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

    def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, ddp=True, notification=False,
              weight=None, **training_args):
        self.update_training_args()
        self.model_args.update_from_dict(training_args)
        model_class = DDPClassificationModel if ddp else ClassificationModel
        self.sm_model = model_class('bert', cfp.bert_dir_remote, num_labels=get_num_labels(train_df),
                                    weight=weight, args=self.model_args)
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
        predict_labels, probs = NoseClassificationModelBase.predict(self, eval_df[self.TEXT_COL].values.squeeze().tolist())
        eval_res = f1_score(true_labels, predict_labels, average='macro')
        res_eval_df = pd.DataFrame({**{col: eval_df[col].tolist() for col in self.TEXT_COL},
                                    'true_labels': eval_df['labels'].tolist(),
                                    'predict_labels': predict_labels})
        if label_names:
            for ind, label in enumerate(label_names):
                res_eval_df[f'P(text={label})'] = probs[:, ind]
        else:
            for label in range(eval_df['labels'].nunique()):
                res_eval_df[f'P(text={label})'] = probs[:, label]
        print(classification_report(true_labels, predict_labels, target_names=label_names, digits=4, zero_division=0))
        return eval_res, res_eval_df

    def set_multiprocessing(self, value: bool):
        self.sm_model.args.use_multiprocessing = value
        self.sm_model.args.use_multiprocessing_for_evaluation = value


def get_num_labels(df: pd.DataFrame):
    return df['labels'].nunique()


def get_rouge_f1_scores(trues, predicts):
    """Compute average rouge scores and edit-distance."""
    rough_obj = Rouge()
    score_avg = rough_obj.get_scores(predicts, trues, avg=True, ignore_empty=True)
    res = {k + '-f1': v['f'] for k, v in score_avg.items()}
    sm = [SequenceMatcher(t, p).ratio() for t, p in zip(trues, predicts)]
    res['avg-edit-distance'] = sum(sm) / len(sm)
    return res


def remove_illegal_char(x):
    if x and isinstance(x, str):
        return remove_illegal_chars(x)
    else:
        return x


class Indication:
    objects = []
    ids = []
    names = []
    fields = ['esid', 'name', 'name_en', 'name_short', 'name_synonyms', 'department', 'indication_type',
              'medgen', 'medgen_synonyms', 'wiki', 'ct_disease']
    synonym_fields = ['name', 'name_en', 'name_short', 'name_synonyms', 'medgen_synonyms', 'ct_disease']
    ambiguous_names = ['肿瘤', '癌症', '其他']
    if_set_included = True

    def __init__(self, **kwargs):
        self.parent = []
        self.children = []
        esid = kwargs.get('esid')
        name = kwargs.get('name')
        for field in self.fields:
            val = kwargs.get(field) and kwargs.pop(field)
            if val:
                setattr(self, field, val)
        self.objects.append(self)
        self.ids.append(esid)
        self.names.append(name)
        self.included = set()
        if self.if_set_included:
            self.__checked_included = set()
            self.set_included_indication()

    def add_child(self, node):
        assert isinstance(node, Indication)
        if node not in self.children:
            self.children.append(node)
        if self not in node.parent:
            node.parent.append(self)

    def __repr__(self):
        return '<Indication %s>' % self.esid

    @classmethod
    def get_object(cls, /, esid=None, name=None):
        assert any([esid, name])
        try:
            if esid:
                ind = cls.ids.index(esid)
            elif name:
                ind = cls.names.index(name)
            res = cls.objects[ind]
        except:
            res = None
        return res

    def add_parent(self, node):
        assert isinstance(node, Indication)
        node.add_child(self)

    @classmethod
    def get_roots(cls):
        return [ind for ind in cls.objects if not ind.parent]

    @classmethod
    def clear_objects(cls):
        cls.objects = []
        cls.ids = []
        cls.names = []

    @classmethod
    def from_dict(cls, items_of_kwargs: list, cache_file: str = None):
        """
        Load indication objects from dictionary.
        :param List[Dict] items_of_kwargs: init args.
        :param cache_file: The json file to load/dump for included indications, 
        to reduce time-consuming loading process.
        """
        use_cache = bool(cache_file) and Path(cache_file).exists()
        cls.if_set_included = not use_cache
        ls_kwargs = deepcopy(items_of_kwargs)
        for kwargs in tqdm(ls_kwargs, desc='Loading indications'):
            parent_ids = kwargs.pop('parent_indication') if kwargs.get('parent_indication') else []
            indication = cls.get_object(esid=kwargs.get('esid')) or Indication(**kwargs)
            for parent_id in parent_ids:
                parent = cls.get_object(esid=parent_id)
                if not parent:
                    parent_kwargs = next(t for t in items_of_kwargs if t['esid'] == parent_id)
                    parent = Indication(**parent_kwargs)
                parent.add_child(indication)
        if use_cache:
            # Set included indication objects post hoc
            with open(cache_file, 'r') as f:
                included_map = json.load(f)
            for ind in tqdm(cls.objects, desc='Loading included'):
                ind.included = {cls.get_object(esid=included_esid) for included_esid in included_map.get(ind.esid)}
            logger.info('Included relations loaded from {}.', cache_file)
        elif cache_file:
            dump_dict = {ind.esid: [included.esid for included in ind.included] for ind in cls.objects}
            with open(cache_file, 'w') as f:
                json.dump(dump_dict, f)
            logger.info('Included relations dumped to {}.', cache_file)
        else:
            logger.info('No cache_file specified, skipping dumping included relations.')

    def find_path(self, node):
        path = []
        for child in self.children:
            if child is node:
                path = [self, node]
            elif child.children:
                child_path = child.find_path(node)
                if child_path:
                    path = [child] + child_path
        return path

    def iter_sibling(self):
        for parent in self.parent:
            for sibling in parent.children:
                if sibling is not self:
                    yield sibling

    def iter_ancestors(self):
        for parent in self.parent:
            yield parent
            if parent.parent:
                yield from parent.iter_ancestors()

    def iter_descendants(self):
        for child in self.children:
            yield child
            if child.children:
                yield from child.iter_descendants()

    @classmethod
    def drop_parent_indications(cls, indication_names):
        res_names = copy(indication_names)
        for name in indication_names:
            indication = cls.get_object(name=name)
            for ans in indication.iter_ancestors():
                if ans.name in res_names:
                    res_names.remove(ans.name)
        return res_names

    def iter_synonyms(self, exclude_fields: list=None):
        filter_field = lambda x: bool(x.replace('-', ''))
        for field in self.synonym_fields:
            if exclude_fields and field in exclude_fields:
                continue
            name = getattr(self, field, None)
            if field in ['name_synonyms', 'medgen_synonyms', 'ct_disease'] and name:
                yield from filter(filter_field, name)
            elif name and filter_field:
                yield name

    def iter_siblings_of_ancestors(self):
        for ancestor in self.iter_ancestors():
            yield from ancestor.iter_sibling()

    illegal_remover = lambda x: re.sub(r'[\(\)\-–,\s]+', ' ', x).strip()

    def match_synonym(self, text):
        matcher = np.vectorize(partial(Levenshtein.ratio, text))
        synonyms = np.array(list(self.iter_synonyms()))
        ratios = matcher(synonyms)
        return synonyms[ratios.argmax()]

    @classmethod
    def rule_match(cls, text):
        matched = []
        t = cls.illegal_remover(text)
        for obj in cls.objects:
            for synonym in obj.iter_synonyms():
                synonym = cls.illegal_remover(synonym)
                if (
                        not synonym.isdigit()
                        and re.findall(r'(?<![a-z])' + synonym.lower() + r'(?![a-z])', t.lower())
                        and obj.name not in matched
                        and obj.name not in cls.ambiguous_names
                ):
                    matched.append(obj.name)
        return cls.drop_included_indication(cls.drop_parent_indications(matched))

    @classmethod
    def batch_rule_match(cls, texts):
        return list(map(cls.rule_match, texts))

    @classmethod
    def get_esid_from_name(cls, name):
        obj = cls.get_object(name=name)
        return obj.esid if obj else None

    def set_included_indication(self):
        for obj in self.objects:
            if obj not in self.__checked_included:
                self.check_included(self, obj)
            if self not in obj.__checked_included:
                self.check_included(obj, self)

    @classmethod
    def check_included(cls, obj, ref_obj):
        for synonym in ref_obj.iter_synonyms():
            if any(synonym0 in synonym for synonym0 in obj.iter_synonyms()) and obj is not ref_obj:
                obj.included.add(ref_obj)
            obj.__checked_included.add(ref_obj)

    @classmethod
    def drop_included_indication(cls, indication_names):
        res_names = copy(indication_names)
        for name in indication_names:
            indication = cls.get_object(name=name)
            for included in indication.included:
                if included.name in indication_names:
                    res_names.remove(name)
                    break
        return res_names

def accuracy(list1, list2):
    top, bottom = 0, 0
    for l1, l2 in zip(list1, list2):
        top += sum(1 for x in l1 if x in l2)
        bottom += max(len(l1), len(l2))
    return top / bottom

def drop_substr(lst):
    return [l1 for l1 in lst if not any(l1 in l2 and l1 != l2 for l2 in lst)]