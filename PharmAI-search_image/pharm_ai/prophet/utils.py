import os
import shutil
import warnings

import Levenshtein
import pandas as pd
from pathlib import Path
import json

from elasticsearch import Elasticsearch
from loguru import logger
import re
from pharm_ai.config import ConfigFilePaths as cfp
from mart.prepro_util.utils import remove_illegal_chars
from mart.utils import email_wrapper
from mart.sm_util.sm_util import fix_torch_multiprocessing, eval_ner_v2
from simpletransformers.t5 import DDPT5Model, T5Model, T5Args
from simpletransformers.classification import ClassificationArgs, ClassificationModel, DDPClassificationModel
from sklearn.metrics import classification_report, f1_score
from scipy.special import softmax
from functools import partial, lru_cache, wraps
from tqdm.auto import tqdm
from rouge.rouge import Rouge
from edit_distance import SequenceMatcher
import numpy as np
from typing import List
from copy import copy, deepcopy
from bs4 import BeautifulSoup
import time
from datetime import datetime
from file_read_backwards import FileReadBackwards

email_receiver = 'fanzuquan@pharmcube.com'
index = 'invest_news'


class ProphetModelBase:
    project_root = Path(__file__).parent

    def __init__(self, version, task_id, cuda_devices=None):
        self.output_dir = self.project_root/'outputs'/version
        self.cache_dir = self.project_root/'cache'/version
        self.best_model_dir = self.project_root/'best_model'/f'{version}.{task_id}'

        self.cuda_devices=cuda_devices

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


blank_table = {160: 32, 9: 32, 10: 32}

class ProphetT5ModelBase(ProphetModelBase):
    model_args = T5Args(
        reprocess_input_data=False,
        overwrite_output_dir=True,
        save_eval_checkpoints=False,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        evaluate_generated_text=False,
        save_model_every_epoch=False,
        wandb_project='prophet'
    )

    def __init__(self, version, cuda_devices=None, task_id=0):

        super().__init__(version, task_id, cuda_devices)
        self.model_args = copy(ProphetT5ModelBase.model_args)
        self.model_args.wandb_kwargs={'tags':[version]}
        self.model_args.output_dir=self.output_dir.as_posix()
        self.model_args.cache_dir=self.cache_dir.as_posix()
        self.model_args.best_model_dir=self.best_model_dir.as_posix()

        if cuda_devices:
            set_cuda_environ(cuda_devices)
            self.model_args.n_gpu=len(self.cuda_devices)

    def update_training_args(self, ddp=True):
        if not ddp:
            self.set_multiprocessing(True)
        else:
            self.set_multiprocessing(False)
        self.model_args.use_cached_eval_features=True,
        self.model_args.wandb_kwargs['tags'].append('train')

    def update_eval_args(self):
        self.model_args.no_cache=True
        self.set_multiprocessing(False)


    def set_multiprocessing(self, value:bool):
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

    def predict(self, prefix, to_predict_texts, separator=None, na_value=None, to_refine=True, sliding=True):
        df = pd.DataFrame({'prefix': prefix, 'to_predicts': to_predict_texts})
        if sliding:
            df['to_predicts'] = df['to_predicts'].map(self.get_sliding_windows)
            df = df.explode('to_predicts')
            df['to_predicts'] = df['to_predicts'].str.translate(blank_table) # substitute illegal blank
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df['raw_predicts'] = self.sm_model.predict((df['prefix'] + ': ' + df['to_predicts']).tolist())
        if df.index.duplicated().any():
            results = df.groupby(level=0)['raw_predicts'].apply(
                partial(self.merge_sliding_window_results, separator=separator, na_value=na_value)
            ).tolist()
        else:
            results = df['raw_predicts'].tolist()
        if to_refine:
            results = list(map(partial(refine_entity, na_value=na_value, separator=separator),
                               df['to_predicts'], results))
        return results

    def eval(self, eval_df, delimiter):
        return eval_ner_v2(eval_df, self.sm_model, delimiter=delimiter)

    def get_sliding_windows(self, text: str, overlap=0.1):
        res = []
        trunk_size = self.sm_model.args.max_seq_length
        text_len = len(text)
        for trunk in range(text_len // trunk_size + 1):
            start_pos = max(0, int((trunk - overlap) * trunk_size))
            end_pos = min(text_len, (trunk + 1) * trunk_size)
            res.append(text[start_pos:end_pos])
        return res

    def merge_sliding_window_results(self, raw: pd.Series, separator, na_value=None):
        s0 = raw.str.split(separator).explode()
        # drop predicted results within overlap windows
        cond = s0.apply(lambda x: s0[s0.ne(x)].str.contains(x).any())
        # drop NA value if possible
        if na_value:
            cond |= s0.eq(na_value)
        return na_value if cond.all() else separator.join(s0[~cond])


class ProphetClassificationModelBase(ProphetModelBase):
    model_args = ClassificationArgs(
        reprocess_input_data=False,
        overwrite_output_dir=True,
        save_eval_checkpoints=False,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        save_model_every_epoch=False,
        no_cache=False,
        wandb_project='prophet'
    )

    def __init__(self, version, sub_project_name, cuda_devices=None, task_id=0):
        """sub_project_name: (news_filter, ira_filter, org_filter)."""
        super(ProphetClassificationModelBase, self).__init__(version, task_id, cuda_devices)

        self.sub_project_name=sub_project_name
        self.model_args = copy(ProphetClassificationModelBase.model_args)
        self.model_args.wandb_kwargs={'tags':[version, sub_project_name]}
        self.model_args.output_dir=self.output_dir.as_posix()
        self.model_args.cache_dir=self.cache_dir.as_posix()
        self.model_args.best_model_dir=self.best_model_dir.as_posix()

        if cuda_devices:
            set_cuda_environ(cuda_devices)
            self.model_args.n_gpu=len(self.cuda_devices)

    def update_training_args(self):
        self.model_args.use_cached_eval_features=True
        self.model_args.use_multiprocessing=True
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

    def predict(self, to_predicts:list):
        res, raw_outputs = self.sm_model.predict(to_predicts)
        if not isinstance(raw_outputs, list):
            probs = softmax(raw_outputs, axis=1)
        else:
            probs = np.array(list(map(lambda ar: list(softmax(ar, axis=1).T), raw_outputs)))
        return res.tolist(), probs

    def eval(self, eval_df: pd.DataFrame, label_names=None):
        true_labels = eval_df['labels'].tolist()
        predict_labels, probs = ProphetClassificationModelBase.predict(self, eval_df['text'].tolist())
        eval_res = f1_score(true_labels, predict_labels, average='macro')
        res_eval_df = pd.DataFrame({'text': eval_df['text'].tolist(), 'true_labels': eval_df['labels'].tolist(),
                                    'predict_labels': predict_labels})
        if label_names:
            for ind, label in enumerate(label_names):
                res_eval_df[f'P(text={label})']=probs[:, ind]
        else:
            for label in range(eval_df['labels'].nunique()):
                res_eval_df[f'P(text={label}']=probs[:, label]
        print(classification_report(true_labels, predict_labels, target_names=label_names, digits=4, zero_division=0))
        return eval_res, res_eval_df

def remove_illegal_char(x):
    if x and isinstance(x, str):
        return remove_illegal_chars(x)
    else:
        return x

def get_num_labels(df: pd.DataFrame):
    return df['labels'].nunique()

def reverse_dict_key_value(dic: dict):
    return dict(zip(dic.values(), dic.keys()))

def freeze_lru_cache(func):
    def func_with_frozen_arg(obj, *args):
        orig_args = (list(ag) if isinstance(ag, tuple) else ag for ag in args)
        return func(obj, *orig_args)
    cached_func = lru_cache(func_with_frozen_arg)

    @wraps(func)
    def wrapped_func(obj, *args):
        frozen_args = (tuple(ag) if isinstance(ag, list) else ag for ag in args)
        res = cached_func(obj, *frozen_args)
        return res

    wrapped_func.cache_info = cached_func.cache_info
    wrapped_func.cache_clear = cached_func.cache_clear

    info = cached_func.cache_info()
    logger.info("{} cache: hits={}, misses={}", func.__name__, info.hits, info.misses)

    return wrapped_func


class PseudoT5Model:
    """A utility t5 analogue model from historical predict functions."""

    from pharm_ai.prophet.news_filter.news_filter import NewsFilter as _NewsFilter
    from pharm_ai.prophet.ira_filter.ira_filter import IraFilter as _IraFilter
    from pharm_ai.prophet.ira.predictor import Ira as _Ira
    from pharm_ai.prophet.org_filter.org_filter import OrgFilter as _OrgFilter
    from pharm_ai.prophet.org_ner.org_ner import OrgNER as _OrgNER

    def get_cuda(self, cuda_device):
        select_cuda = os.environ['CUDA_VISIBLE_DEVICES'].index(str(cuda_device)) if cuda_device != -1 and os.environ.get(
            'CUDA_VISIBLE_DEVICES') else cuda_device
        return select_cuda

    def _news_filter_init(self) -> _NewsFilter:
        predictor = self._NewsFilter(cuda_device=self.cuda_device)
        predictor.title_classifier.model.args.use_multiprocessing_for_evaluation=False
        predictor.title_classifier.model.args.use_multiprocessing=False
        predictor.title_classifier.model.args.silent=True,
        predictor.uncertain_classifier.model.args.use_multiprocessing=False
        predictor.uncertain_classifier.model.args.use_multiprocessing_for_evaluation=False
        predictor.uncertain_classifier.model.args.silent=True
        return predictor

    def _org_filter_init(self) -> _OrgFilter:
        predictor = self._OrgFilter(cuda_device=self.cuda_device, version='v1-1', date='20201204')
        predictor.model.args.use_multiprocessing_for_evaluation=False
        predictor.model.args.use_multiprocessing=False
        return predictor

    def _ira_filter_init(self) -> _IraFilter:
        predictor = self._IraFilter(cuda_device=self.cuda_device)
        predictor.model.args.use_multiprocessing_for_evaluation=False
        predictor.model.args.use_multiprocessing=False
        return predictor

    def _ira_init(self) -> _Ira:
        predictor = self._Ira(version='v2.5',cuda_device=self.cuda_device)
        return predictor

    def _org_ner_init(self) -> _OrgNER:
        predictor = self._OrgNER(cuda_device=self.cuda_device)
        predictor.model.args.silent=True
        return predictor

    def _news_filter_predict(self, obj: _NewsFilter, to_predict):
        s = pd.Series(to_predict)
        pattern = r'(?:标题：(?P<title_cn>.*)。全文：(?P<fulltext_cn>.*))|(?:Title: (?P<title_en>.*). Fulltext: (?P<fulltext_en>.*))'
        extracted = s.str.extract(pattern)
        titles = extracted['title_cn'].where(extracted['title_cn'].notna(), extracted['title_en']).fillna('')
        fulltexts = extracted['fulltext_cn'].where(extracted['fulltext_cn'].notna(), extracted['fulltext_en']).fillna('')
        res = [obj.predict(title_, [fulltext_]) for title_, fulltext_ in
               tqdm(zip(titles, fulltexts), total=len(titles))]
        return [str(r) for r in res]

    def _ira_filter_predict(self, obj: _IraFilter, to_predict):
        res = obj.predict_(to_predict)
        return [str(r) for r in res]

    def _org_filter_predict(self, obj: _OrgFilter, to_predict):
        res = obj.predict(is_folding_head=True, texts=to_predict)
        return [str(r) for r in res]

    def _ira_predict(self, obj: _Ira, prefix, to_predict):
        model_to_predicts = [prefix + ': ' + raw for raw in to_predict]
        return obj.model.predict(model_to_predicts)

    def _org_ner(self, obj: _OrgNER, to_predict):
        return PseudoT5Model._OrgNER.predict_ner(obj, to_predict, return_raw=False)

    def _org_ner_sub(self, obj: _OrgNER, sub, to_predict):
        """sub: (c1, c2, c3, f)."""
        res_ner = self._org_ner(obj, tqdm(to_predict, desc=f'org_ner.{sub}'))
        res_dict = {'c1': [], 'c2': [], 'c3': [], 'f': []}
        for r in res_ner:
            for k in res_dict:
                rd = ';'.join(r.get(k) or [''])
                res_dict[k].append(rd)
        return res_dict.get(sub)


    IRA = ['investee', 'round', 'amount']

    def __init__(self, sub_projects: List[str], cuda_device=-1):
        self.sub_projects = sub_projects
        self.cuda_device = self.get_cuda(cuda_device)
        self.predictors={}

    model_init_func = {"news_filter": _news_filter_init,
                       "ira_filter": _ira_filter_init,
                       "org_filter": _org_filter_init,
                       "ira": _ira_init,
                       "org_ner": _org_ner_init}
    predict_func = {"news_filter": _news_filter_predict, "ira_filter": _ira_filter_predict,
                    "org_filter": _org_filter_predict, "investee": partial(_ira_predict, prefix='investee'),
                    "round": partial(_ira_predict, prefix='round'),
                    "amount": partial(_ira_predict, prefix='amount'),
                    "org_ner.c1": partial(_org_ner_sub, sub='c1'),
                    "org_ner.c2": partial(_org_ner_sub, sub='c2'),
                    "org_ner.c3": partial(_org_ner_sub, sub='c3'),
                    "org_ner.f": partial(_org_ner_sub, sub='f')}

    predict_config = {'fillna': None}

    def predict(self, to_predict: List[str]):
        """to_predict: each format as '<prefix>: <text>'."""
        prefixes, texts = zip(*[t.split(': ', maxsplit=1) for t in to_predict])
        return self.get_history_predicts(list(prefixes), list(texts))

    def get_history_predict_funcs(self):
        """Join prophet-v2.x predict function to a dictionary."""
        res = {}
        self.predictors = {}
        for prefix in self.sub_projects:
            if prefix in self.IRA:
                if self.predictors.get('ira'):
                    predictor = self.predictors.get('ira')
                else:
                    predictor_func = self.model_init_func.get('ira')
                    predictor = predictor_func(self)
                    self.predictors['ira'] = predictor
            elif prefix.startswith('org_ner'):
                if self.predictors.get('org_ner'):
                    predictor = self.predictors.get('org_ner')
                else:
                    predictor_func = self.model_init_func.get('org_ner')
                    predictor = predictor_func(self)
                    self.predictors['org_ner'] = predictor
            else:
                predictor_func = self.model_init_func.get(prefix)
                predictor = predictor_func(self)
                self.predictors[prefix] = predictor
            func = self.predict_func.get(prefix)
            res[prefix] = partial(func, obj=predictor)
        return res

    @freeze_lru_cache
    def get_history_predicts(self, prefix, to_predict_texts):
        assert len(prefix) == len(to_predict_texts)
        prefixes = set(prefix)
        ind_results = []
        history_predict_funcs = self.get_history_predict_funcs()
        for pre in prefixes:
            fun = history_predict_funcs.get(pre)
            to_pred_ind, to_pred = zip(*filter(lambda x: x[1][0]==pre, enumerate(zip(prefix, to_predict_texts))))
            _, to_pred = zip(*to_pred)
            to_pred = list(to_pred)
            res = fun(self, to_predict=to_pred)
            ind_results.extend(list(zip(to_pred_ind, res)))
        ind_results.sort(key=lambda x: x[0])
        if ind_results:
            _, results = zip(*ind_results)
            fillna = self.predict_config.get('fillna')
            if not fillna:
                return list(results)
            else:
                return list(res or fillna for res in results)
        else:
            return []


def get_rouge_f1_scores(trues, predicts):
    """Compute average rouge scores and edit-distance."""
    rough_obj = Rouge()
    score_avg = rough_obj.get_scores(predicts, trues, avg=True, ignore_empty=True)
    res = {k + '-f1': v['f'] for k, v in score_avg.items()}
    sm = [SequenceMatcher(t, p).ratio() for t, p in zip(trues, predicts)]
    res['avg-edit-distance'] = sum(sm)/len(sm)
    return res


def get_news_by(host='online', *, title=None, esid=None, number: int = 10):
    ls = None
    es = get_es(host)
    fields = ['title', 'content', 'publish_date', 'resource']
    if esid:
        raw = es.get(index, esid, _source_includes=fields)
        ls = {'esid': raw['_id'], **raw['_source']}
    elif title:
        body = {"query": {"match_phrase": {"title.raw": title}},
                "sort": [{"publish_date": {"order": "desc"}}]}
        raw = es.search(body, index, _source_includes=fields)
        ls = [{'esid': r['_id'], **r['_source']} for r in raw['hits']['hits']]
    elif number:
        # None field specified
        raw = es.search(index=index, _source_includes=fields, sort={'spider_wormtime': 'desc'}, size=number)
        ls = [{'esid': r['_id'], **r['_source']} for r in raw['hits']['hits']]
    return ls

def tidy_paragraphs(content, split_break=True):
    soup = BeautifulSoup(content, features='html.parser')
    if split_break:
        res = [ss.replace('\n','') for s in soup.contents if hasattr(s, 'text') and s.text
               for ss in s.text.split('\n\n') if ss]
    else:
        res = [s.text.replace('\n','') for s in soup.contents if hasattr(s, 'text') and s.text]
    return res

def int2date(int_date:int):
    res = time.strftime('%Y-%m-%d', time.localtime(int_date/1000))
    return res

def date2int(date_str):
    """Year-Month-Day -> timestamp"""
    return int(datetime.strptime(date_str, '%Y-%m-%d').timestamp()*1000)

def format_prophet_input_data(esid, title, content, publish_date:int, resource):
    res = {esid: {'title': title, 'paragraphs': tidy_paragraphs(content), 'publish_date': int2date(publish_date),
                  'news_source': resource}}
    return res

def parse_log_data(log_file, start_string):
    pattern = r'([0-9\-\:\s\.]+?)\s*\|\s*([A-Z]+?)\s*\|.*Input data: (.*)'
    input_data_pattern = r'SingleArgBody\(title=\'(.*)\', paragraphs=(.*), publish_date=\'(.*)\', news_source=\'(.*)\'\)'
    result = {}
    with FileReadBackwards(log_file, encoding='utf-8') as log_f:
        while True:
            line = log_f.readline()
            if line.startswith(start_string):
                r = re.match(pattern, line)
                date_str, log_level, input_str = r.groups()
                r_input = re.search(input_data_pattern, input_str)
                if not r_input:
                    input_dic_raw = eval(input_str)
                    result = input_dic_raw['input_dic']
                else:
                    title, para_str, publish_date, news_source = r_input.groups()
                    paras = [p.strip("'\"") for p in para_str.strip('[]').split("', '")] if para_str!='[]' else []
                    esid_pattern = r'\{\'input_dic\': \{\'([0-9a-z]+)\''
                    r_esid = re.search(esid_pattern, input_str)
                    esid = r_esid.groups()[0]
                    result[esid] = {'title': title, 'paragraphs': paras, 'publish_date': publish_date, 'news_source': news_source}
                break
    return result


def get_one_log(saving_json_path, start_string, log_file=None):
    """
    Get one prophet data from log by matching the given starting string,
    and save to json."""
    if not log_file:
        log_file = Path('result.log')
    if saving_json_path.exists():
        with saving_json_path.open('r') as f:
            data = json.load(f)
    else:
        data = parse_log_data(log_file, start_string)
    return data


def get_news_by_esids_and_save(es_host, esids, saving_json_path):
    if saving_json_path.exists():
        with saving_json_path.open('r') as f:
            ls = json.load(f)
    else:
        ls = {}
        for esid in esids:
            cur_res = get_news_by(es_host, esid=esid)
            ls.update(format_prophet_input_data(
                cur_res['esid'], cur_res['title'], cur_res['content'],
                cur_res['publish_date'], cur_res['resource']
            ))
        with saving_json_path.open('w') as f:
            json.dump(ls, f, ensure_ascii=False, indent=4)
    return ls


def get_recent_data_save(num_data, saving_json_path):
    if saving_json_path.exists():
        with saving_json_path.open('r') as f:
            res = json.load(f)
    else:
        raw_ls = get_news_by(number=num_data)
        res = {}
        for entry in raw_ls:
            res.update(
                format_prophet_input_data(entry['esid'], entry['title'], entry['content'], entry['publish_date'],
                                          entry['resource']))
        with saving_json_path.open('w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
    return res


def get_es(host='test'):
    """
    Get Elasticsearch instance.
    :param str host: 'test', 'online', '176'
    :return: an Elasticsearch instance.
    """
    assert host in ['test', 'online', '176']
    if host == 'test':
        return Elasticsearch(hosts=[{"host": 'test112', 'port': '9200', 'http_auth': 'esjava:esjava123abc'}])
    elif host == '176':
        return Elasticsearch(hosts=[{'host': 'gpu176', 'port': '9325', 'http_auth': 'fzq:es271828'}])
    else:
        return Elasticsearch(
            hosts=[{"host": 'esnode8.cubees.com', 'port': '9200', 'http_auth': 'esjava:esjava123abc'}],
            use_ssl=True, verify_certs=False, ssl_show_warn=False
        )


def migrate_data_to_test_es(esids, json_file=None, to_upload=None, only_load=False):
    """Migrate specific data from online ES host to test ES host, using json file as the intermediary dump."""
    json_path = Path(json_file) if not isinstance(json_file, Path) and json_file is not None else json_file
    if json_path is not None and not json_path.exists():
        es = get_es('online')
        res = get_from_esids(es, esids)
        with json_path.open('w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        print(f"{len(esids)} data fetched")
    else:
        if not to_upload:
            with json_path.open('r') as f:
                res = json.load(f)
        else:
            res = to_upload
        if not only_load:
            es = get_es('test')
            index_bulk(es, res)
            logger.info("{} data uploaded to es test.", len(esids))
    return res


def index_bulk(es, to_upload):
    body = []
    for d in deepcopy(to_upload):
        esid = d.pop('esid')
        body.append({'index': {'_index': index, '_id': esid}})
        body.append(d)
    bulk_result = es.bulk(body, refresh=True)
    assert not bulk_result['errors'], 'Index bulk error!'


def update_bulk(es, to_upload, raise_error=True):
    body = []
    for d in deepcopy(to_upload):
        esid = d.pop('esid')
        body.append({'update': {'_index': index, '_id': esid}})
        body.append({'doc': d})
    bulk_result = es.bulk(body, refresh=True)
    if raise_error:
        assert not bulk_result['errors'], 'Update bulk error!'
    return bulk_result


def get_from_esids(es: Elasticsearch, esids, fields: List[str] = None):
    body = {"query": {"ids": {"values": esids}}}
    if fields:
        body['fields'] = fields
        body['_source'] = False
    raw = es.search(body=body, index=index, _source_includes=fields)
    res = [{'esid': r['_id'], **r['_source']} for r in raw['hits']['hits']]
    return res


def delete_bulk(es: Elasticsearch, esids):
    body = [
        {"delete": {"_index": index, "_id": esid}}
        for esid in esids
    ]
    bulk_res = es.bulk(body=body, refresh=True)
    assert not bulk_res['errors'], 'Delete bulk error!'


def scroll_search(es: Elasticsearch, es_index, body=None, batch_size=1000, desc='Fetching', _source_includes=None):
    results = []
    raw = es.search(body=body, index=es_index, scroll='1m', size=batch_size, _source_includes=_source_includes)
    total = raw['hits']['total']['value']
    bar = tqdm(total=total, desc=desc)
    batch = [{'esid': r['_id'], **r['_source']} for r in raw['hits']['hits']]
    results.extend(batch)
    bar.update(len(batch))
    while len(results) < total:
        raw = es.scroll(scroll_id=raw['_scroll_id'], scroll='1m')
        batch = [{'esid': r['_id'], **r['_source']} for r in raw['hits']['hits']]
        results.extend(batch)
        bar.update(len(batch))
    es.clear_scroll(scroll_id=raw['_scroll_id'])
    bar.close()
    return results


def refine_entity(text, raw_entity, na_value='', separator=None):
    """Refine NER entities from t5 model according to the edit-distance of input and output texts."""
    na_value = na_value or ''
    if separator and (separator in raw_entity):
        raws = raw_entity.split(separator)
        result = raw_entity
        for raw in raws:
            res = refine_entity(text, raw, na_value=na_value, separator=separator)
            result.replace(raw, res)
        return result
    if raw_entity == na_value or raw_entity in text:
        return raw_entity
    df = pd.DataFrame(Levenshtein.editops(text, raw_entity))
    deleted_window = df[df[0] == 'delete'].rolling(window=2)[1].apply(lambda s: s.iloc[1] - s.iloc[0]).fillna(1)
    is_delete_continue = deleted_window.eq(1)
    if is_delete_continue.all():
        # raw_entity doesn't appear at text at all
        return na_value
    else:
        start_loc = df.loc[is_delete_continue.idxmin() - 1, 1] + 1
        is_gap_delete = deleted_window.gt(1)
        # given the entity locates finally, end_loc is not needed
        is_end_deleted = df.loc[df[0] == 'delete', 1].iloc[-1] == len(text) - 1
        end_loc = df.loc[is_gap_delete.index[is_gap_delete][-1], 1] if is_gap_delete.any() and is_end_deleted else None
        result = text[start_loc:end_loc]
        return result if len(result) < len(raw_entity) * 2 else na_value

if __name__ == '__main__':
    esids = [
        '62e7ccb2f9b8a48153d6ef514a7f211d', 'b4970381ff1204a60ee9fa6fe21988b6', 'b0ed65eadbe807c8c44292d8026dc347',
        '0f704a6f3469110526da25afccb30ced', 'b1875bed9123d61e861a38687947448e',
        '66169964f84ea812d2c5e64f21ac514f', '54a6e648f10706839d1af86b71295fd8'
    ]
    migrate_data_to_test_es(esids, json_file='results/data0906.json')
