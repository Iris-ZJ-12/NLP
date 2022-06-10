import warnings

import re
import sys
from functools import partial, cached_property
from itertools import combinations, chain
from pathlib import Path

import json
import pandas as pd
from loguru import logger
from scipy.stats import gmean
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
from copy import copy

from mart.es_util.ESUtils7 import get_page
from pharm_ai.nose.utils import remove_illegal_char, Indication

logid = logger.add(sys.stderr, filter=lambda record: record["extra"].get("task") == "dt")
dt_logger = logger.bind(task="dt")


class PreprocessorBase:
    root_path = Path(__file__).parent
    data_file = root_path / 'data.h5'
    version = None
    random_state = 111
    es_index = 'discover_indication'
    es_host = {'test': ('test155.cubees.com',)}

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

    def preprocess_train_eval_dataset(self, save_h5=True, balance_train_label=True):
        df_raw = self.get_from_h5()
        train_df, eval_df = train_test_split(df_raw, test_size=0.1, random_state=self.random_state)
        if save_h5:
            self.save_to_h5(train_df, 'train')
            self.save_to_h5(eval_df, 'eval')
        return train_df, eval_df

    @classmethod
    def pull_dictionary(cls, saving_json='data/es_dict_1101.json', es_host='176'):
        host = cls.es_host.get(es_host)
        raw = get_page(cls.es_index, page_size=-1, host=host)
        with open(saving_json, 'w') as sf:
            json.dump(raw, sf, ensure_ascii=False, indent=2)
        dt_logger.success('{} dictionary pulled from {}, saved to {}', len(raw), es_host, saving_json)




class PreprocessorT5Base(PreprocessorBase):
    T5_COLS = ['prefix', 'input_text', 'target_text']
    CLASSIFICATION_TASK = []
    NER_TASK = []
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
                         null_target: str = None, count_multiple=False):
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
            if count_multiple:
                contain_sep = df['target_text'].str.contains(cls.seperator, regex=False)
                cls._pretty_print(head + ' target_text', f'single target: {df[~contain_sep].shape[0]}',
                                  f'multiple targets: {df[contain_sep].shape[0]}')
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


class PreprocessorClassificationBase(PreprocessorBase):
    dataframe_columns = ['text', 'labels']

    @classmethod
    def balance_label(cls, df: pd.DataFrame, max_size=None, shuffle_result=True, auto_limit_size=False):
        if auto_limit_size:
            max_sample_size = int(gmean(df['labels'].value_counts()))
        else:
            max_sample_size = max_size or df['labels'].value_counts().max()
        res_df = pd.concat(
            resample(d, n_samples=max_sample_size, random_state=cls.random_state)
            if d.shape[0] != max_sample_size else d
            for label_, d in df.groupby('labels')
        )
        return res_df.sample(frac=1, random_state=cls.random_state) if shuffle_result else res_df

    def preprocess_train_eval_dataset(self, auto_limit_size=False, return_dataframe=False):
        df_raw = self.get_from_h5()
        train_df, eval_df = train_test_split(df_raw, test_size=0.1, random_state=self.random_state)
        train_sampled = self.balance_label(train_df, auto_limit_size=auto_limit_size)
        if return_dataframe:
            return train_sampled, eval_df
        else:
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


class PreprocessorEmbeddingBase(PreprocessorBase):
    pass


class NosePreprocessorV1_0(PreprocessorClassificationBase):
    version = 'v1.0'
    dataframe_columns = ['text_a', 'text_b', 'labels']
    synonym_fields = ['name', 'name_en', 'name_short', 'name_synonyms']
    data_fields = ['esid', 'text', 'indication']
    indication_refiner = [
        re.compile(r'(?<=[\[,])\s*(.*?)(?=[,\]])').findall,
        re.compile(r'\s*,\s*').split
    ]

    def __init__(self, dict_json="data/es_dict_1101.json", use_cached_indications: str = None):
        self.dict_json = dict_json
        with open(dict_json) as f:
            self.raw_dict = json.load(f)
        self.indications = [t['name'] for t in self.raw_dict]
        Indication.from_dict(self.raw_dict, cache_file=use_cached_indications)

    def pull_dictionary(self, es_host='176'):
        super().pull_dictionary(saving_json=self.dict_json, es_host=es_host)


    def generate_positive_pairs(self):
        for item in self.raw_dict:
            esid = item['esid']
            synonyms = []
            for field in self.synonym_fields:
                item_field = item.get(field)
                if item_field:
                    if isinstance(item_field, str):
                        synonyms.append(item_field)
                    elif isinstance(item_field, list):
                        for f in item_field:
                            synonyms.append(f)
            for synonym in combinations(synonyms, 2):
                yield {'esid': esid, 'text': synonym, 'labels': 1}

    def preprocess_raw(self, excel_data='data/适应症NLP标注数据2021-10-18.xlsx'):
        raw_df = self._read_raw_excel(excel_data)
        raw_df = pd.concat(df.explode('indication') for df in raw_df)
        raw_df = raw_df[(~raw_df['indication'].isna()) & (
            raw_df['indication'].isin(self.indications))].drop_duplicates(self.data_fields[-2:])
        res_df = pd.DataFrame(self.generate_text_labels(raw_df))
        self.save_to_h5(res_df)

    @classmethod
    def get_production_dataset(cls):
        raw_df = cls.get_from_h5()
        eval_df = cls.get_from_h5('eval')
        raw_df = raw_df[raw_df['text_a'].isin(eval_df.loc[eval_df['labels']==1, 'text_a'])]
        res_df = raw_df.loc[raw_df['labels'] == 1, ['esid', 'text_a', 'text_b']].rename(
            columns={'text_a': 'description', 'text_b': 'indication'})
        return res_df.groupby('description').agg({'esid': lambda s: s.iloc[0], 'indication': list}).reset_index()

    def _read_raw_excel(self, excel_data):
        df = []
        with pd.ExcelFile(excel_data) as excel_f:
            df.append(pd.read_excel(excel_f, sheet_name='Sheet1', names=self.data_fields[-2:]))
            df.append(pd.read_excel(excel_f, sheet_name='临床Chi',
                  names=self.data_fields, usecols='A:C',
                  converters={'indication': self._convert_indication}))
            df.append(
                pd.read_excel(excel_f, sheet_name='临床Chi',
                  names=self.data_fields, usecols='A:C',
                  converters={'indication': self._convert_indication})
            )
            df.append(
                pd.read_excel(excel_f, sheet_name='clinical trial',
                  names=self.data_fields[-2:],
                  converters={'indication': partial(self._convert_indication, use_re=True)})
            )
            df.append(
                pd.read_excel(excel_f, sheet_name='Sheet7',
                  names=self.data_fields,
                  converters={'indication': self._convert_indication})
            )
        return df


    def _convert_indication(self, x, use_re=False, re_pattern=0):
        if x:
            if use_re:
                list_x = self.indication_refiner[re_pattern](x)
            else:
                list_x = json.loads(x)
            if list_x:
                return tuple(list_x)
            else:
                return None
        else:
            return None



    def generate_text_labels(self, from_raw_df: pd.DataFrame):
        for row_ind, row in tqdm(from_raw_df.iterrows(), total=from_raw_df.shape[0]):
            same_texts = from_raw_df[from_raw_df['text'] == row['text']]
            esids = same_texts['esid'].dropna()
            esid = esids.iloc[0] if not esids.empty else str(row_ind)
            yield {'esid': esid, 'text_a': row['text'], 'text_b': row['indication'], 'labels': 1}
            for indication in self.indications:
                if indication not in same_texts['indication']:
                    yield {'esid': esid, 'text_a': row['text'], 'text_b': indication, 'labels': 0}

    def preprocess_train_eval_dataset(self):
        train_sampled, eval_df = super().preprocess_train_eval_dataset(auto_limit_size=True, return_dataframe=True)
        eval_df = self.balance_label(eval_df, auto_limit_size=True).drop_duplicates()
        self.save_to_h5(train_sampled, 'train')
        self.save_to_h5(eval_df, 'eval')


class NosePreprocessorV1_1(NosePreprocessorV1_0):
    version = 'v1.1'

    def generate_indication_synonyms(self, indication_name):
        if indication_name in self.indications:
            dict_item = next(d for d in self.raw_dict if d['name'] == indication_name)
            for field in self.synonym_fields:
                name = dict_item.get(field)
                if field == 'name_synonyms' and name:
                    for res in name:
                        if res:
                            yield res
                else:
                    if name:
                        yield name


    def generate_text_labels(self, from_raw_df: pd.DataFrame):
        for row_ind, row in tqdm(from_raw_df.iterrows(), total=from_raw_df.shape[0]):
            same_texts = from_raw_df[from_raw_df['text'] == row['text']]
            esids = same_texts['esid'].dropna()
            esid = esids.iloc[0] if not esids.empty else str(row_ind)
            for synonyms in self.generate_indication_synonyms(row['indication']):
                yield {'esid': esid, 'text_a': row['text'], 'text_b': synonyms, 'labels': 1}
            for indication in self.indications:
                if indication not in same_texts['indication']:
                    yield {'esid': esid, 'text_a': row['text'], 'text_b': indication, 'labels': 0}



class NosePreprocessorV1_2(NosePreprocessorV1_0):
    version = 'v1.2'

    def generate_text_labels(self, from_raw_df: pd.DataFrame):
        for row_ind, row in tqdm(from_raw_df.iterrows(), total=from_raw_df.shape[0]):
            same_texts = from_raw_df[from_raw_df['text'] == row['text']]
            esids = same_texts['esid'].dropna()
            esid = esids.iloc[0] if not esids.empty else str(row_ind)
            indication: Indication = Indication.get_object(name=row['indication'])
            for synonyms in indication.iter_synonyms():
                yield {'esid': esid, 'text_a': row['text'], 'text_b': synonyms, 'labels': 1}
            for neg_indication in chain(indication.iter_descendants(), indication.iter_siblings_of_ancestors()):
                for neg_synonyms in neg_indication.iter_synonyms():
                    yield {'esid': esid, 'text_a': row['text'], 'text_b': neg_synonyms, 'labels': 0}

    def preprocess_train_eval_dataset(self):
        train_sampled, eval_df = PreprocessorClassificationBase.preprocess_train_eval_dataset(
            self, return_dataframe=True)
        eval_df = self.balance_label(eval_df, auto_limit_size=True).drop_duplicates()
        self.save_to_h5(train_sampled, 'train')
        self.save_to_h5(eval_df, 'eval')


class NosePreprocessorV1_3(NosePreprocessorV1_1, NosePreprocessorV1_2):
    version = 'v1.3'

    def preprocess_raw(self):
        warnings.warn('No need preprocess raw dataset, reusing v1.1 and v1.2.')


    def preprocess_train_eval_dataset(self):
        train_df1 = NosePreprocessorV1_1.get_from_h5('train')
        train_df2 = NosePreprocessorV1_2.get_from_h5('train')
        eval_df1 = NosePreprocessorV1_2.get_from_h5('eval')
        eval_df2 = NosePreprocessorV1_2.get_from_h5('eval')
        train_df = pd.concat([train_df1, train_df2]).drop_duplicates()
        eval_df = pd.concat([eval_df1, eval_df2]).drop_duplicates()
        self.save_to_h5(train_df, 'train')
        self.save_to_h5(eval_df, 'eval')

    @classmethod
    def _get_h5_key(cls, data_category):
        if data_category == 'raw':
            dt_logger.warning('get {} from version v1.1', data_category)
            return NosePreprocessorV1_1._get_h5_key(data_category)
        else:
            return super()._get_h5_key(data_category)

    @classmethod
    def get_production_dataset(cls):
        return NosePreprocessorV1_0.get_production_dataset()

class NosePreprocessorV1_4(PreprocessorT5Base, NosePreprocessorV1_0):
    version = 'v1.4'
    TASK = 'normalize'
    NER_TASK = [TASK]


    def preprocess_raw(self, excel_data='data/适应症NLP标注数据2021-10-18.xlsx'):
        raw_df = self._read_raw_excel(excel_data)
        raw_df = pd.concat(df.explode('indication') for df in raw_df)
        raw_df = raw_df[(~raw_df['indication'].isna()) & (
            raw_df['indication'].isin(self.indications))].drop_duplicates(self.data_fields[-2:])
        res_df = pd.DataFrame(self.generate_input_target_texts(raw_df))
        self.save_to_h5(res_df)

    def generate_input_target_texts(self, from_raw_df: pd.DataFrame):
        for input_text, df_grp in tqdm(from_raw_df.groupby('text')):
            cur_esids = df_grp['esid'].dropna()
            esid = cur_esids.iloc[0] if not cur_esids.empty else ''
            yield {'esid': esid, 'prefix': self.TASK, 'input_text': input_text,
                   'target_text': self.seperator.join(df_grp['indication'])}

    def preprocess_train_eval_dataset(self):
        PreprocessorBase.preprocess_train_eval_dataset(self)

    @classmethod
    def get_production_dataset(cls):
        return NosePreprocessorV1_0.get_production_dataset()


class NosePreprocessorV1_5(NosePreprocessorV1_4):
    version = 'v1.5'

    def generate_input_target_texts(self, from_raw_df: pd.DataFrame):
        for input_text, df_grp in tqdm(from_raw_df.groupby('text')):
            cur_esids = df_grp['esid'].dropna()
            esid = cur_esids.iloc[0] if not cur_esids.empty else ''
            target_texts = df_grp[self.data_fields[-2:]].apply(
                lambda row: Indication.get_object(name=row['indication']).match_synonym(row['text']),
                axis=1
            )
            yield {'esid': esid, 'prefix': self.TASK, 'input_text': input_text,
                   'target_text': self.seperator.join(target_texts)}


class NosePreprocessorV1_6(NosePreprocessorV1_5):
    version = 'v1.6'

    @classmethod
    def _get_h5_key(cls, data_category):
        ver = NosePreprocessorV1_5.version.replace('.', '-') if data_category == 'raw' else cls.version.replace('.', '-')
        key = '%s/%s' % (data_category, ver)
        return key

    def preprocess_train_eval_dataset(self):
        train_df, eval_df = PreprocessorBase.preprocess_train_eval_dataset(self, save_h5=False)
        is_multiple_targets = train_df['target_text'].str.contains(self.seperator, regex=False)
        sample_size = (~is_multiple_targets).sum()
        train_resampled = pd.concat([
            resample(train_df[is_multiple_targets], n_samples=sample_size, random_state=self.random_state),
            train_df[~is_multiple_targets]
        ]).sample(frac=1, random_state=self.random_state)
        self.save_to_h5(train_resampled, 'train')
        self.save_to_h5(eval_df, 'eval')


class NosePreprocessorV1_7(NosePreprocessorV1_4):
    version = 'v1.7'
    seperator = ','

    def generate_input_target_texts(self, from_raw_df: pd.DataFrame):
        """Generate indication esid as target text."""
        for input_text, df_grp in tqdm(from_raw_df.groupby('text')):
            cur_esids = df_grp['esid'].dropna()
            esid = cur_esids.iloc[0] if not cur_esids.empty else ''
            target_texts = df_grp[self.data_fields[-2:]].apply(
                lambda row: Indication.get_object(name=row['indication']).esid,
                axis=1
            )
            yield {'esid': esid, 'prefix': self.TASK, 'input_text': input_text,
                   'target_text': self.seperator.join(target_texts)}


class NosePreprocessorV2_0(PreprocessorEmbeddingBase):
    """Embedding model dataset"""
    version = 'v2.0'

    def __init__(self, use_cached_indications: str = None):
        self.previous_preprocessor = NosePreprocessorV1_0(use_cached_indications=use_cached_indications)


    def get_train_data(self):
        train_data = []
        train_df = self.previous_preprocessor.get_from_h5('train')
        for indication in Indication.objects:
            for name in indication.iter_synonyms():
                train_data.append(InputExample(texts=[name, indication.name]))
                if indication.name != name:
                    train_data.append(InputExample(texts=[indication.name, name]))
        for _, row in train_df[train_df['labels'] == 1].drop_duplicates().iterrows():
            train_data.append(InputExample(texts=[row['text_a'], row['text_b']]))
        return train_data

    def get_eval_data(self):
        """
        Get evaluator data for TripleEvaluator.
        :return: Anchor data, Positive data, Negtive data.
        """
        eval_df = self.previous_preprocessor.get_from_h5()
        select_eval_df = eval_df[eval_df['labels'] == 1].copy()
        negtives = select_eval_df['text_b'].apply(NosePreprocessorV2_0._get_one_parent)
        return (
            select_eval_df[negtives.notna()]['text_b'].tolist(),
            select_eval_df[negtives.notna()]['text_a'].tolist(),
            negtives.dropna().tolist()
        )

    @staticmethod
    def _get_one_parent(name):
        obj = Indication.get_object(name=name)
        for parent in obj.parent:
            return parent.name
        return None

    def get_production_dataset(self):
        return self.previous_preprocessor.get_production_dataset()

class NosePreprocessor:
    CLASSES = [NosePreprocessorV1_0, NosePreprocessorV1_1, NosePreprocessorV1_2,
               NosePreprocessorV1_3, NosePreprocessorV1_4, NosePreprocessorV1_5,
               NosePreprocessorV1_6, NosePreprocessorV1_7, NosePreprocessorV2_0]
    preprocessor_versions = {c.version: c for c in CLASSES}

    @classmethod
    def get_preprocessor_class(cls, version):
        return cls.preprocessor_versions.get(version)


if __name__ == '__main__':
    PreprocessorBase.pull_dictionary('data/es_dict_1209.json')
else:
    logger.remove(logid)