import json
import os
import pickle
import re
import sys
from itertools import count
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from mart.es_util.ESUtils7 import get_page, Query, QueryType
from mart.ner_util.rule_ner import RuleNER
from mart.prepro_util.prepro import Prepro
from mart.sm_util.sm_util import ner2xlsx, ner_predict
from pharm_ai.perk.utils import remove_illegal_char, classify_char

logger.remove()
logger.add(sys.stderr, filter=lambda record: record["extra"].get("task") == "dt")
dt_logger = logger.bind(task="dt")


class PreprocessorBase:
    root_path = Path(__file__).parent
    data_file = root_path / 'data.h5'
    version = None
    random_state = 1221
    es_index = "drug_earth"
    stopwords_json = 'en_stopwords.json'
    location_info_file = Path('raw_data/ht_location.csv')

    @classmethod
    def _get_h5_key(cls, data_category):
        ver = cls.version.replace('.', '-')
        key = '%s/%s' % (data_category, ver)
        return key

    @classmethod
    def save_to_h5(cls, df: pd.DataFrame, data_category='raw'):
        key = cls._get_h5_key(data_category)
        df.to_hdf(cls.data_file, key)
        dt_logger.success('{} data saved to "{}" with key "{}"', data_category, cls.data_file, key)

    @classmethod
    def get_from_h5(cls, data_category='raw') -> pd.DataFrame:
        key = cls._get_h5_key(data_category)
        res = pd.read_hdf(cls.data_file, key)
        dt_logger.success('{} data loaded from "{}" with key "{}"', data_category, cls.data_file, key)
        return res

    @classmethod
    def _pretty_print(cls, head, *contents):
        print('-' * 15 + head + '-' * 15, end='\n')
        for cont in contents:
            print(cont, end='\n')

    @classmethod
    def export_train_eval_dataset(cls, saving_excel, sample_n_train=None, sample_n_eval=None):
        train_df = cls.get_from_h5('train')
        if sample_n_train:
            train_df = train_df.sample(sample_n_train, random_state=cls.random_state)

        train_df = train_df.applymap(remove_illegal_char)
        eval_df = cls.get_from_h5('eval')
        if sample_n_eval:
            eval_df = eval_df.sample(sample_n_eval, random_state=cls.random_state)
        eval_df = eval_df.applymap(remove_illegal_char)
        with pd.ExcelWriter(saving_excel) as writer:
            train_df.to_excel(writer, sheet_name='train', index=False)
            eval_df.to_excel(writer, sheet_name='eval', index=False)
        dt_logger.info('Train and eval datasets saved to "{}".', saving_excel)

    def preprocess_train_eval_dataset(self, save_h5=True, eval_size=0.1):
        df_raw = self.get_from_h5()
        train_df, eval_df = train_test_split(df_raw, test_size=eval_size, random_state=self.random_state)
        if save_h5:
            self.save_to_h5(train_df, 'train')
            self.save_to_h5(eval_df, 'eval')
        return train_df, eval_df

    @classmethod
    def get_pharm_dict(cls, online=True, offline_dict="pharmacy_en_names_glossary_FZQ_20201207.xlsx",
                       pickle_file='pharm_list.pkl'):
        if online:
            res = get_page(index=cls.es_index,
                           queries=Query(QueryType.EQ, "is_delete", "否"),
                           page_size=-1,
                           show_fields=['title', 'name_short', 'name_en', 'name_spec', 'is_delete'])
            res_list = [r['name_en'] for r in res]
        else:
            res = pd.read_excel(offline_dict)
            res1 = [ss for s in res['name_spec'] for ss in cls.recover_string_list(s) if cls.classify_char(ss) == 'en']
            res2 = [each_item.strip() for _, col in res[['title', 'name_short', 'name_en']].items() for each_item in col
                    if cls.classify_char(each_item) == 'en']
            res_set = set(res1 + res2)
            res_set.remove('')
            res_list = sorted(list(res_set))
        res_list = cls.remove_stopwords(res_list)
        with open(pickle_file, 'wb') as f:
            pickle.dump(res_list, f)
        print(f"Drug English names saved to '{pickle_file}'.")

    @classmethod
    def remove_stopwords(cls, input_list: list):
        with open(cls.stopwords_json, 'r') as f:
            dic = json.load(f)
        stopwords = dic['words']
        res = [s for s in input_list if s.lower() not in stopwords and not re.fullmatch(r'[\-+=\.*&/]+', s)]
        return res

    @classmethod
    def prepare_location_dict(cls):
        df_raw = pd.read_csv(cls.location_info_file)
        res_dic = {}
        for row_id, row in df_raw.iterrows():
            if row['level'] > 1:
                path = [int(p) for p in row['path'].strip(',').split(',')]
                matched_country = df_raw[df_raw['id'] == path[1]].iloc[0]
                match_res = {
                    'standard_contry': matched_country['name_en'],
                    'country_alias': matched_country[['name', 'name_pinyin', 'code']].dropna().to_dict()
                }
                res_dic[row['name']] = match_res
                res_dic[row['name_en']] = match_res
        path_json = cls.location_info_file.parent / (cls.location_info_file.stem + '.json')
        with path_json.open('w') as f:
            json.dump(res_dic, f, indent=4, ensure_ascii=False)
        logger.info('ht_location saved to {}.', path_json)


class PreprocessorT5Base(PreprocessorBase):
    T5_COLS = ['prefix', 'input_text', 'target_text']
    CLASSIFICATION_TASK = []
    NER_TASK = ['drug', 'disease', 'model', 'nation']
    TO_CLS_COLS = ['esid', 'input_text', 'target_text']
    TO_CLS_MAPPING = {'input_text': 'text', 'target_text': 'labels'}
    seperator = '|'

    @classmethod
    def upsampling_t5_classification_dataset(cls, df):
        """Balance data amount of each label.
        df: A classification dataset with only one prefix."""
        max_label_size = df['target_text'].value_counts().max()
        res_df = pd.concat(
            resample(d, n_samples=max_label_size, random_state=cls.random_state) if d.shape[0] < max_label_size else d
            for label, d in df.groupby('target_text')
        )
        return res_df

    @classmethod
    def balance_t5_prefix(cls, df, max_size=None):
        """Upsampling or downsampling dataset of each prefix."""
        default_max_size = df['prefix'].value_counts().max()
        max_sample_size = min(default_max_size, max_size) if max_size else default_max_size

        res_df = pd.concat(
            resample(d, n_samples=max_sample_size, random_state=cls.random_state)
            if d.shape[0] != max_sample_size else d
            for prefix, d in df.groupby('prefix')
        )
        return res_df

    @classmethod
    def describe_dataset(cls, data_category='raw', classification_prefix=None,
                         null_target: str = None):
        df_ = cls.get_from_h5(data_category)
        cls_tasks = classification_prefix or cls.CLASSIFICATION_TASK

        def describe_df(unique=False):
            if unique:
                df = df_.drop_duplicates(subset=cls.T5_COLS)
                head = f'{data_category} dataset (unique)'
            else:
                df = df_
                head = f'{data_category} dataset (duplicated)'
            if not null_target:
                cls._pretty_print(head, f'size={df.shape[0]}', 'prefix counts:',
                                  df['prefix'].value_counts())
            else:
                pos_df = df[df['target_text'] != null_target]
                neg_df = df[df['target_text'] == null_target]
                cls._pretty_print(head, f'size={df.shape[0]}', 'positive prefix counts:',
                                  pos_df['prefix'].value_counts(), 'negtive prefix counts:',
                                  neg_df['prefix'].value_counts())
            if cls_tasks:
                for prefix in cls_tasks:
                    df_cls = df[df['prefix'] == prefix]
                    head_cls = f'{data_category}-{prefix} dataset {"(unique)" if unique else "(duplicated)"}'
                    cls._pretty_print(
                        head_cls, f'size={df_cls.shape[0]}', 'label counts:', df_cls['target_text'].value_counts()
                    )

        describe_df(unique=False)
        describe_df(unique=True)

    @classmethod
    def to_cls_data(cls, prefix, col_mapping=None, columns=None):
        """Convert to bert classification dataset style."""
        raw_df = cls.get_from_h5()
        columns = columns or cls.TO_CLS_COLS
        df = raw_df[raw_df['prefix'] == prefix][columns].rename(columns=cls.TO_CLS_MAPPING)
        if col_mapping:
            df['labels'] = df['labels'].map(col_mapping)
        return df

    @classmethod
    def check_cls_contradict_labels(cls, df: pd.DataFrame, prefix=None):
        prefix = prefix or cls.CLASSIFICATION_TASK
        df = df[df['prefix'].isin(prefix)]

        def check_one_label(df_: pd.DataFrame):
            return df_.groupby('input_text').nunique().pipe(
                lambda d: df_[df_['input_text'].isin(d[d['target_text'] > 1].index)]
            ).sort_values(by='input_text')

        res = df.groupby('prefix').apply(check_one_label)
        dt_logger.warning('{} text with contradicted labels found.', res['input_text'].nunique())
        return res

    @classmethod
    def _get_h5_key(cls, term):
        key_ = f'ner/{cls.version.replace(".", "-")}/{term}'
        return key_


class PreprocessorClassificationBase(PreprocessorBase):
    dataframe_columns = ['text', 'labels']
    reverse_type_mapping = {0: 'CEA/CUA', 1: 'Other', 2: 'CBA', 3: 'CMA', 4: 'CCA', 5: 'BIA'}
    type_mapping = {'CEA/CUA': 0, 'Other': 1, 'CBA': 2, 'CMA': 3, 'CCA': 4, 'BIA': 5}
    reverse_result_mapping = {1: 'cost-effective', 2: 'not cost-effective', 3: 'dominate', 4: 'dominated',
                              0: 'not stated'}
    result_mapping = {'cost-effective': 1, 'not cost-effective': 2, 'dominate': 3, 'dominated': 4,
                      'not stated': 0}

    @classmethod
    def balance_label(cls, df: pd.DataFrame, max_size=None, shuffle_result=True):
        max_sample_size = max_size or df['labels'].value_counts().max()
        res_df = pd.concat(
            resample(d, n_samples=max_sample_size, random_state=cls.random_state)
            if d.shape[0] != max_sample_size else d
            for label_, d in df.groupby('labels')
        )
        return res_df.sample(frac=1, random_state=cls.random_state) if shuffle_result else res_df

    def preprocess_train_eval_dataset(self):
        df_raw = self.get_from_h5()
        train_df, eval_df = train_test_split(df_raw, test_size=0.1, random_state=self.random_state)
        train_sampled = self.balance_label(train_df)
        self.save_to_h5(train_sampled, data_category='train')
        self.save_to_h5(eval_df, data_category='eval')

    @classmethod
    def describe_dataset(cls, data_category='raw'):
        df_ = cls.get_from_h5(data_category)

        def describe_df(unique=False):
            if unique:
                df = df_.drop_duplicates(subset=cls.dataframe_columns)
                head = f'{data_category} dataset (unique)'
            else:
                df = df_
                head = f'{data_category} dataset (duplicated)'
            cls._pretty_print(
                head,
                f'size={df.shape[0]}',
                'label counts:',
                df['labels'].value_counts()
            )

        describe_df(unique=False)
        describe_df(unique=True)

    @classmethod
    def check_contradict_labels(cls, df: pd.DataFrame) -> pd.DataFrame:
        res = df.groupby('text').nunique().pipe(
            lambda d: df[df['text'].isin(d[d['labels'] > 1].index)]
        ).sort_values(by='text')
        dt_logger.warning('{} text with contradicted labels found.', res['text'].nunique())
        return res

    @classmethod
    def _get_h5_key(cls, data_category, append=None):
        ver = cls.version.replace('.', '-')
        key = 'classification/%s/%s' % (ver, data_category)
        if append:
            key = key + '/' + append
        return key

    @classmethod
    def prepare_to_label(cls, dump_json='results/prepare_v3_label_classification.json'):
        """Pull from es to prepare classification dataset."""
        dump_json = Path(dump_json)
        dump_json.parent.mkdir(exist_ok=True)
        if dump_json.exists():
            with dump_json.open('r') as f:
                ls = json.load(f)
            logger.info('{} papers loaded from "{}".', len(ls), dump_json)
        else:
            q = Query(QueryType.EQ, 'tac_result', 'cost-effective')
            ls = get_page('pharmacoeconomics_paper', queries=q, page_size=-1,
                          show_fields=['pm_id', 'title', 'abstract_info_nolabel', 'tac_type', 'tac_result'])
            with dump_json.open('w') as f:
                json.dump(ls, f, ensure_ascii=False, indent=4)
            logger.info('{} papers dumped to "{}".', len(ls), dump_json)
        df = pd.DataFrame.from_records(ls).rename(columns={
            'tac_result': 'predicted_result', 'tac_type': 'predicted_type', 'abstract_info_nolabel': 'abstract'})
        res_df = df.loc[:, ['esid', 'pm_id', 'title', 'abstract', 'predicted_type', 'predicted_result']]
        res_df = res_df.assign(revised_type=None, revised_result=None)
        dump_excel = dump_json.parent / (dump_json.stem + '.xlsx')
        res_df.to_excel(dump_excel, index=False)
        dt_logger.info('Prepared classification dataframe saved to "{}".', dump_excel)


class PreprocessorSeq2SeqBase(PreprocessorBase):
    reverse_type_mapping = {0: 'CEA/CUA', 1: 'Other', 2: 'CBA', 3: 'CMA', 4: 'CCA', 5: 'BIA'}
    type_mapping = {'CEA/CUA': 0, 'Other': 1, 'CBA': 2, 'CMA': 3, 'CCA': 4, 'BIA': 5}
    tasks = ['type', 'result']
    reverse_result_mapping = {1: 'cost-effective', 2: 'not cost-effective', 3: 'dominate', 4: 'dominated',
                              0: 'not stated'}
    result_mapping = {'cost-effective': 1, 'not cost-effective': 2, 'dominate': 3, 'dominated': 4,
                      'not stated': 0}
    SEQ_COLS = ['input_text', 'target_text']

    @classmethod
    def describe_dataset(cls, data_category='raw'):
        df_ = cls.get_from_h5(data_category)
        refined_df = pd.DataFrame(
            cls.seq_decode(df_['target_text'].tolist(), recover_encode=True),
            columns=cls.tasks,
            index=df_.index
        )

        def describe_df(task, unique=False):
            if unique:
                df = df_.drop_duplicates(subset=cls.SEQ_COLS)
                head = f'{data_category} {task} dataset (unique)'
            else:
                df = df_
                head = f'{data_category} {task} dataset (duplicated)'
            cls._pretty_print(head, f'size={df.shape[0]}',
                              'prefix counts:', refined_df[task].value_counts())

        for task in cls.tasks:
            describe_df(task, unique=False)
            describe_df(task, unique=True)

    @classmethod
    def _get_h5_key(cls, data_category):
        ver = cls.version.replace('.', '-')
        key = 'seq/%s/%s' % (ver, data_category)
        return key

    @classmethod
    def seq_decode(cls, raw_texts: List[str], return_raw_results=False, recover_encode=False):
        if not recover_encode:
            raw_results = []
            results = []
            for raw in raw_texts:
                r = re.match(r'([\w\s\-/]+), ([\w\s\-/]+)', raw)
                if r:
                    raw_res = list(r.groups())
                    raw_res_type = raw_res[0].replace(' ', '').upper() \
                        if raw_res[0].replace(' ', '') in ['cea/cua', 'cba', 'cma', 'cca', 'bia'] \
                        else raw_res[0].capitalize()
                    cur_res_type = raw_res_type if raw_res_type in cls.type_mapping else ''
                    raw_res_result = raw_res[1].replace(' ', '') if raw_res[1].find('-') > -1 else raw_res[1]
                    cur_res_result = raw_res_result if raw_res_result in cls.result_mapping else ''
                    cur_res = [cur_res_type, cur_res_result]
                else:
                    raw_res = ''
                    cur_res = ['', '']
                raw_results.append(raw_res)
                results.append(cur_res)
            if return_raw_results:
                return raw_results, results
            else:
                return results
        else:
            results = [text.split(', ') for text in raw_texts]
            return results


class PerkRuleNer(PreprocessorBase):
    """A RuleNer preprocessor & predictor."""
    version = 'v1.0'
    models_list = ['Decision tree model', 'Markov model', 'Partitioned survival model',
                   'Discrete events simulation model', 'DES', 'Dynamic transmission models']

    def __init__(self, disease_pickle='disease_list.pkl',
                 pharm_pickle='pharm_list.pkl', country_pickle='countries_list.pkl'):
        self.disease_pickle = disease_pickle
        with open(disease_pickle, 'rb') as f:
            disease_list = pickle.load(f)
        self.pharm_pickle = pharm_pickle
        with open(pharm_pickle, 'rb') as f:
            pharm_list = pickle.load(f)
        self.country_pickle = country_pickle
        with open(country_pickle, 'rb') as f:
            nation_list = pickle.load(f)
        dic = {"disease": disease_list, "drug": pharm_list, "nation": nation_list, "model": self.models_list}
        self.rule = RuleNER(dic, 'perk_dictionary_0106.pkl')
        self.prepro = Prepro()

    def preprocess_raw(self, xlsx_file="raw_data/药经文献实体识别数据.xlsx"):
        df_raw = pd.read_excel(xlsx_file)
        df_raw.rename(columns={'标题': 'title', '摘要': 'abstract'}, inplace=True)
        to_pred = df_raw['abstract'].tolist()
        res = self.ner_predict(to_pred)
        ner2xlsx(res, df_raw['pm_id'].tolist(), 'perk_rule_ner_result_1205.xlsx',
                 'rule', ['labels'])

    def ner_predict(self, to_predict: list):
        """
        :param list[str] to_predict: List of paragraph text
        :return: Result list of rule NER prediction.
        """
        res = [ner_predict(tp, ..., self.prepro, self.rule, 1) for tp in to_predict]
        return res

    def preprocess_disease_dict(self, xlsx_file="online_entities-20200922.xlsx"):
        df = pd.read_excel(xlsx_file, sheet_name="disease")
        df['language'] = df['disease'].map(classify_char)
        res = df[df['language'] == 'en']['disease'].tolist()
        res = self.remove_stopwords(res)
        with open(self.disease_pickle, 'wb') as f:
            pickle.dump(res, f)

    def preprocess_nation_dict(self, input_file='countries.csv'):
        df = pd.read_csv(input_file)
        arr = df.drop(columns='num_code').values
        res = arr.reshape(arr.size, ).tolist()
        res = [r for r in res if r == r]
        res = self.remove_stopwords(res)
        with open(self.country_pickle, 'wb') as f:
            pickle.dump(res, f)


class PerkPreprocessorV1_1(PreprocessorClassificationBase):
    """
        v1.1: initial dataset (sentence level)
    """
    version = 'v1.1'
    task_map = {0: 'type', 1: 'result'}

    def __init__(self, task=0):
        self.task = task
        self.task_str = self.task_map.get(task)

    def preprocess_raw(self, excel_file='raw_data/药经文本分类typeresult.xlsx'):
        df1 = pd.read_excel(excel_file)
        df1.rename(columns={'Sentence': 'text', 'Unnamed: 2': 'labels_sent', 'Type': 'labels_type'}, inplace=True)
        df1['text'] = df1['text'].astype(str)
        df1['labels_type'] = df1['labels_type'].map(self.type_mapping)
        df2 = pd.read_excel(excel_file, sheet_name=1)
        df2.rename(columns={'Sentence': 'text', 'Unnamed: 2': 'labels_sent', 'result': 'labels_result'}, inplace=True)
        df2['text'] = df2['text'].astype(str)
        df2['labels_result'] = df2['labels_result'].map(self.result_mapping)
        res1 = df1.drop(columns='labels_sent').groupby('PMID').agg({
            'text': lambda s: '. '.join(s.tolist()),
            'labels_type': lambda s: s.unique()[0]
        })
        res2 = df2.drop(columns='labels_sent').groupby('PMID').agg({
            'text': lambda s: '. '.join(s.tolist()),
            'labels_result': lambda s: s.unique()[0]
        })
        res = res1.join(res2.drop(columns='text'), how='outer')
        self.save_to_h5(res)

    def preprocess_train_eval_dataset(self, eval_size=0.15):
        df = self.get_from_h5()
        if self.task == 0:
            df = df.drop(columns='labels_result').rename(columns={'labels_type': 'labels'})
        elif self.task == 1:
            df = df.drop(columns='labels_type').rename(columns={'labels_result': 'labels'})
        else:
            raise ValueError('task not in [0, 1].')
        df_grs = df.groupby('labels')
        resample_size = df_grs['text'].count().max()
        df_train = []
        df_eval = []
        for _, df_each in df_grs:
            if df_each.shape[0] < 10:
                cur_train = resample(df_each, n_samples=resample_size, stratify=df_each['text'],
                                     random_state=self.random_state)
                df_train.append(cur_train)
            else:
                df_train_each, df_eval_each = train_test_split(df_each, test_size=eval_size,
                                                               random_state=self.random_state)
                cur_train = resample(df_train_each, n_samples=resample_size, stratify=df_train_each['text'],
                                     random_state=self.random_state)
                df_train.append(cur_train)
                df_eval.append(df_eval_each)
        res_train = pd.concat(df_train).sample(frac=1, random_state=self.random_state)
        res_eval = pd.concat(df_eval)
        self.save_to_h5(res_train, 'train')
        self.save_to_h5(res_eval, 'eval')

    def _get_h5_key(self, data_category):
        if data_category == 'raw':
            return super()._get_h5_key(data_category=data_category)
        else:
            return super()._get_h5_key(data_category, self.task_str)

    def get_from_h5(self, data_category='raw') -> pd.DataFrame:
        key = self._get_h5_key(data_category)
        res = pd.read_hdf(self.data_file, key)
        dt_logger.success('{} data loaded from "{}" with key "{}"', data_category, self.data_file, key)
        return res

    def describe_dataset(self, data_category='raw'):
        if data_category == 'raw':
            raise RuntimeError('describe raw dataset is not support!')
        df_ = self.get_from_h5(data_category=data_category)

        def describe_df(unique=False):
            if unique:
                df = df_.drop_duplicates(subset=self.dataframe_columns)
                head = f'{data_category} dataset (unique)'
            else:
                df = df_
                head = f'{data_category} dataset (duplicated)'
            self._pretty_print(
                head,
                f'size={df.shape[0]}',
                'label counts:',
                df['labels'].value_counts()
            )

        describe_df(unique=False)
        describe_df(unique=True)

    def save_to_h5(self, df: pd.DataFrame, data_category='raw'):
        key = self._get_h5_key(data_category)
        df.to_hdf(self.data_file, key)
        dt_logger.success('{} data saved to "{}" with key "{}"', data_category, self.data_file, key)


class PerkPreprocessorV1_2(PerkPreprocessorV1_1):
    """
    new dataset (fulltext level). classification model format.
    """
    version = 'v1.2'
    task_map = {0: 'type', 1: 'result'}

    def __init__(self, task=0):
        self.task = task
        self.task_str = self.task_map.get(task)

    def preprocess_raw(self, excel="raw_data/文本分类数据 -20201224.xlsx"):
        df = pd.read_excel(excel)
        df.columns = ['pm_id', 'text', 'labels_type', 'labels_result']
        df['labels_type'] = df['labels_type'].map(self.type_mapping)
        df['labels_result'] = df['labels_result'].map(self.result_mapping)
        df.set_index('pm_id', inplace=True)
        self.save_to_h5(df)

    def preprocess_train_eval_dataset(self, eval_size=0.15):
        df = self.get_from_h5()
        if self.task == 0:
            df = df.drop(columns='labels_result').rename(columns={'labels_type': 'labels'})
            df = df[df['labels'].isin(self.type_mapping.values())]
            res_train, res_eval = train_test_split(df, test_size=eval_size, random_state=self.random_state)
        elif self.task == 1:
            df = df.drop(columns='labels_type').rename(columns={'labels_result': 'labels'})
            df_train, res_eval = train_test_split(df, test_size=eval_size, random_state=self.random_state)
            res_train = self.balance_label(df_train, shuffle_result=False)
        else:
            raise ValueError('task not in [0, 1].')
        self.save_to_h5(res_train, 'train')
        self.save_to_h5(res_eval, 'eval')


class PerkPreprocessorV1_3(PreprocessorSeq2SeqBase):
    """
    new dataset (fulltext level). seq2seq model format.
    """
    version = 'v1.3'

    def preprocess_raw(self, xlsx="raw_data/文本分类数据 -20201224.xlsx"):
        df = pd.read_excel(xlsx)
        df.columns = ['pm_id', 'text', 'labels_type', 'labels_result']
        df = df[df['labels_type'].isin(['CEA/CUA', 'Other'])]
        df_ = pd.DataFrame({
            'pm_id': df['pm_id'],
            'input_text': df['text'],
            'target_text': df[['labels_type', 'labels_result']].apply(
                lambda s: ', '.join(s.tolist()),
                axis=1
            )
        })
        df_ = df_.dropna()
        df_.set_index('pm_id', inplace=True)
        self.save_to_h5(df_)

    def preprocess_train_eval_dataset(self, save_h5=True):
        super().preprocess_train_eval_dataset(save_h5=save_h5, eval_size=0.15)


class PerkPreprocessorV2_1(PerkPreprocessorV1_2):
    """
    Add manual-labeled data, for classification model.
    """
    version = 'v2.1'

    def preprocess_raw(self):
        super().preprocess_raw("raw_data/药经文献文本分类.xlsx")

    def preprocess_train_eval_dataset(self, eval_size=0.15):
        assert self.task in [0, 1]
        df = self.get_from_h5()
        train_pre = PerkPreprocessorV1_2.get_from_h5('train')
        eval_pre = PerkPreprocessorV1_2.get_from_h5('eval')
        df = pd.concat([df, train_pre, eval_pre])
        df_train, res_eval = train_test_split(df, test_size=eval_size, random_state=self.random_state)
        res_train = self.balance_label(df_train)
        self.save_to_h5(res_train, 'train')
        self.save_to_h5(res_eval, 'eval')


class PerkPreprocessorV2_2(PerkPreprocessorV1_3):
    """
    add manual-labeled data. seq2seq model format.
    """
    version = 'v2.2'

    def preprocess_raw(self, excel="raw_data/文本分类数据 -20201224.xlsx"):
        df = pd.read_excel(excel)
        df.columns = ['pm_id', 'text', 'labels_type', 'labels_result']
        df_ = pd.DataFrame({
            'pm_id': df['pm_id'],
            'input_text': df['text'],
            'target_text': df[['labels_type', 'labels_result']].apply(
                lambda s: ', '.join(s.tolist()),
                axis=1
            )
        })
        df_ = df_.dropna()
        df_.set_index('pm_id', inplace=True)
        self.save_to_h5(df_)


class PerkPreprocessorV2_0(PreprocessorT5Base):
    """NER dataset from open Microsoft."""
    random_state = 413
    seperator = '; '

    def preprocess_raw(self):
        df_raw = pd.concat(pd.read_csv(os.path.join('raw_data', 'save_dt', csv_file), compression='gzip',
                                       usecols=['entities', 'categories', 'sentence_id', 'sentences', 'marks'])
                           for csv_file in os.listdir('raw_data/save_dt/')
                           if csv_file.endswith('csv.gz'))
        df_raw_drug = df_raw[df_raw['categories'] == 'MedicationName']
        df_raw_drug = df_raw_drug.rename(columns={'categories': 'prefix', 'sentences': 'input_text',
                                                  'entities': 'target_text'})
        res_df = df_raw_drug.groupby(['prefix', 'input_text'])['target_text'].apply(
            self._concat_ner_entities).reset_index()
        self.save_to_h5(res_df)

    def _concat_ner_entities(self, terms):
        term_str = [t if isinstance(t, str) else str(t) for t in terms]
        res = self.seperator.join(term_str)
        return res


class PerkPreprocessorV2_3(PerkPreprocessorV2_0):
    """NER dataset labeled by Pharmcube."""
    version = 'v2.3'

    def preprocess_raw(self, excel_file="raw_data/实体识别结果.xls"):
        raw_dfs = pd.read_excel(excel_file, sheet_name=None)
        prefixes = self.NER_TASK
        for (item, raw_df_item), prefix in zip(raw_dfs.items(), prefixes):
            raw_df_item.columns = ['esid', 'input_text', 'target_text']
            raw_df_item['prefix'] = prefix
        res_df = pd.concat(raw_dfs.values())
        res_df['target_text'] = res_df['target_text'].fillna("").drop_duplicates()
        self.save_to_h5(res_df)

    def preprocess_train_eval_dataset(self, save_h5=True):
        df_train, df_eval = super().preprocess_train_eval_dataset(save_h5=False)
        df_train = self.sampling_ner_dataset(df_train)
        if save_h5:
            self.save_to_h5(df_train, 'train')
            self.save_to_h5(df_eval, 'eval')

    def sampling_ner_dataset(self, df):
        is_empty_df = df.assign(is_empty=df['target_text'].eq(""))
        stat_df = is_empty_df.groupby(['prefix', 'is_empty'])['esid'].count()
        stat_mean = stat_df.mean()
        ratios = (stat_df / stat_mean).where(stat_df / stat_mean > 1, stat_mean / stat_df)
        if (ratios < 5).all():
            target_amount = stat_df.max()
        else:
            target_amount = int(stat_mean)
        df_res = pd.concat(resample(d, n_samples=target_amount, random_state=self.random_state)
                           for gr, d in is_empty_df.groupby(['prefix', 'is_empty'])).sample(frac=1)
        return df_res

class PerkPreprocessorV3_0(PerkPreprocessorV2_1):
    """Classification dataset preprocessor, add more data on minority class."""
    version = 'v3.0'

    def preprocess_raw(self, excel_file='raw_data/20210930_NLP文本分类.xlsx'):
        df = pd.read_excel(excel_file,
                           names=['pm_id', 'title', 'abstract', 'predicted_type', 'predicted_result', 'labels_type',
                                  'labels_result'],
                           usecols='B:H', dtype={'pm_id': 'str'},
                           na_values='/')
        pmid_counter = ('Unknown_pmid_%04d' % c for c in count(1))
        df['pm_id'].update(
            pd.Series([
                id_ for id_, _ in zip(pmid_counter, range(df['pm_id'].isna().sum()))],
                index=df.index[df['pm_id'].isna()]
            )
        )
        df = df.set_index('pm_id')
        df['labels_type'] = df['labels_type'].where(df['labels_type'].notna(), df['predicted_type'])
        df['labels_type'].update(df['labels_type'].map({'other': 'Other'}))
        df['labels_type'] = df['labels_type'].map(self.type_mapping)
        df['labels_result'] = df['labels_result'].where(df['labels_result'].notna(), df['predicted_result'])
        df['labels_result'].update(df['labels_result'].map({
            'not cost-effetcive': 'not cost-effective',
            'Not cost-effetcive': 'not cost-effective',
            'Not cost-effective': 'not cost-effective',
            'not state': 'not stated', 'Not stated': 'not stated',
            'Dominate': 'dominate'
        }))
        df['labels_result'] = df['labels_result'].map(self.result_mapping)
        df['text'] = df['title'] + ' ' + df['abstract']

        p1 = PerkPreprocessorV1_2()
        df_pre1 = p1.get_from_h5()
        p2 = PerkPreprocessorV2_1()
        df_pre2 = p2.get_from_h5()

        result_df = pd.concat([
            df[['text', 'labels_type', 'labels_result']],
            df_pre1, df_pre2
        ]).drop_duplicates()

        assert not result_df['labels_type'].isna().any()
        assert not result_df['labels_result'].isna().any()
        self.save_to_h5(result_df)

    def preprocess_train_eval_dataset(self, eval_size=0.15):
        df = self.get_from_h5()
        if self.task == 0:
            df = df.drop(columns='labels_result').rename(columns={'labels_type': 'labels'})
        elif self.task == 1:
            df = df.drop(columns='labels_type').rename(columns={'labels_result': 'labels'})
        else:
            raise ValueError('task not in [0, 1].')
        df_train, res_eval = train_test_split(df, test_size=eval_size, random_state=self.random_state)
        res_train = self.balance_label(df_train)
        self.save_to_h5(res_train, 'train')
        self.save_to_h5(res_eval, 'eval')

class PerkPreprocessor:
    CLASSES = [PerkPreprocessorV1_1, PerkPreprocessorV1_2, PerkPreprocessorV1_3,
               PerkPreprocessorV2_0, PerkPreprocessorV2_1, PerkPreprocessorV2_2,
               PerkPreprocessorV2_3, PerkPreprocessorV3_0]
    preprocessor_versions = {c.version: c for c in CLASSES}

    @classmethod
    def get_preprocessor_class(cls, version):
        return cls.preprocessor_versions.get(version)


if __name__ == '__main__':
    for task in [0, 1]:
        p = PerkPreprocessorV3_0(task=task)
        p.preprocess_train_eval_dataset()
        p.describe_dataset('train')
        p.describe_dataset('eval')
