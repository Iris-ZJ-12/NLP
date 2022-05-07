import re
import sys
from copy import deepcopy
from itertools import chain
from math import gcd
from pathlib import Path

import ijson
import json
import numpy as np
import pandas as pd
from loguru import logger
from lxml import etree
from mart.es_util.ESUtils7 import get_page
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from pharm_ai.ak.utils import remove_illegal_char, get_edit_distance

logger.remove()
logger.add(sys.stderr, filter=lambda record: record["extra"].get("task") == "dt")
dt_logger = logger.bind(task="dt")


class PreprocessorBase:
    root_path = Path(__file__).parent
    data_file = root_path / 'data.h5'
    version = None
    random_state = 1014

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
    def get_from_h5(cls, data_categroy='raw') -> pd.DataFrame:
        key = cls._get_h5_key(data_categroy)
        res = pd.read_hdf(cls.data_file, key)
        dt_logger.success('{} data loaded from "{}" with key "{}"', data_categroy, cls.data_file, key)
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

    def preprocess_train_eval_dataset(self, save_h5=True):
        df_raw = self.get_from_h5()
        train_df, eval_df = train_test_split(df_raw, test_size=0.1, random_state=self.random_state)
        if save_h5:
            self.save_to_h5(train_df, 'train')
            self.save_to_h5(eval_df, 'eval')
        return train_df, eval_df


class HTMLProcessor:
    blank_remover = re.compile(r'\s+(?=<.+?>)')
    blank_striper = re.compile(r'(?<=^)[\s\n]+|[\s\n]+(?=$)')
    illegal_element_remover = re.compile(r'<!--.*?-->')
    table_xpath_refiner = re.compile(r'(?<=table)(.*/)span\[\d+\]')

    def __init__(self, html):
        self.html = self.illegal_element_remover.sub('', html)
        self.html_tree: etree._Element = etree.HTML(self.html)
        self.tree = etree.ElementTree(self.html_tree)

    def get_most_close_parent(self, child_xpath1, child_xpath2):
        element1, element2 = [self.html_tree.xpath(p)[0] for p in (child_xpath1, child_xpath2)]
        parent_element = next(e1 for e1 in element1.iterancestors() for e2 in element2.iterancestors() if e1 == e2)
        return parent_element

    def get_root_distance(self, element):
        return len(list(element.iterancestors()))

    def xpath_to_element(self, xpaths, ref_texts: list = None):
        ref_texts = ref_texts or [''] * len(xpaths)
        results = []
        for xp, ref in zip(xpaths, ref_texts):
            if ':' in xp:
                element = []
            elif self.table_xpath_refiner.findall(xp):
                # process elements in table
                xp_new = self.table_xpath_refiner.sub(r'\1', xp)
                text_list = self.html_tree.xpath(xp_new)
                element = [t.getparent() for t in text_list if self.blank_striper.sub('', t)]
            else:
                element = self.html_tree.xpath(xp.split('/text')[0])

            # if failed to use xpath, use reference text to locate element
            if not element:
                try:
                    select_element = next(ele for ele in self.html_tree.iterdescendants()
                                          if ele.text and (ele.text == ref or ref in ele.text))
                    dt_logger.warning('xpath {} not exist, using reference text "{}" to locate.', xp, ref)
                    results.append([select_element])
                except:
                    dt_logger.warning('xpath {} not exist, return None.', xp)
                    results.append([])
            else:
                results.append(element)
        return results

    @classmethod
    def remove_html_attributes(cls, element):
        ele_copy = deepcopy(element)
        for el in chain([ele_copy], ele_copy.iterdescendants()):
            for k in el.keys():
                el.attrib.pop(k)
        res = etree.tostring(
            ele_copy, encoding='UTF-8', method='html'
        ).decode('UTF-8').replace('\n', '').replace('\t', '')
        return cls.blank_remover.sub('', res)

    def remove_tags(self, element=None, remove_style_element=False, len_threshold=None):
        ele = deepcopy(element) if element else self.html_tree
        results = []
        for item in chain([ele], ele.iterdescendants()):
            if item.text and (not remove_style_element or item.tag != 'style'):
                strip_res = self.blank_striper.sub('', item.text)
                if strip_res and (len_threshold is None or len(strip_res) <= len_threshold):
                    results.append(strip_res)
        return results

    @classmethod
    def concat_element_text(cls, *element: etree._Element):
        return ''.join(cls.blank_striper.sub('', ele.text) if ele.text else '' for ele in element)

    def locate_element(self, element_text):
        return [self.tree.getpath(ele) + '/text()' for ele in self.tree.iter()
                if ele.text == element_text]

    def to_string(self):
        return etree.tostring(self.tree, encoding='utf-8').decode('utf-8')


class LabelstudioData:
    xpath_refiner = re.compile(r'tbody\[\d+\]')
    es_host = {'test': ('test155.cubees.com',),
               '176': ("esnode8.cubees.com",)}
    es_index = 'gov_purchase'

    def __init__(self, result_json):
        self.result_json = result_json

    def generate_item(self, ner_type, generate_esid=False):
        """Generate HTML and annotation data.
        Args:
            ner_type: NER types in annotation, eg: ['head', 'product']
            generate_esid: bool. Whether yield esid
        yield:
            - html
            - ner_xpath_texts: Dict[ner_type:str -> Tuple[xpath: str, text: str]].
        """
        with open(self.result_json, 'r') as f:
            for raw_json in ijson.items(f, 'item'):
                html = raw_json['data']['content']
                ner_xpath_texts = {}
                for t in ner_type:
                    xpath_text = []
                    for anno in raw_json['annotations']:
                        for res in anno['result']:
                            if t in res['value']['htmllabels']:
                                xpath = '/' + self.xpath_refiner.sub('', res['value']['start'])
                                text = res['value']['text']
                                if all(xpath != xp for xp, txt in xpath_text):
                                    xpath_text.append((xpath, text))
                    ner_xpath_texts[t] = xpath_text
                if not generate_esid:
                    yield html, ner_xpath_texts
                else:
                    esid = raw_json['data']['esid']
                    yield esid, html, ner_xpath_texts

    @classmethod
    def pull_es_data(cls, saving_json, es_host='test', exclude_num=0, size=1000):
        """
        Args:
            es_host: (test, 176)
            exclude_num: (int) exclude the first N data.
        """
        page_index = exclude_num // gcd(exclude_num, size) + 1
        to_label = get_page(cls.es_index, page_size=size, show_fields=['content'],
                            page_index=page_index,
                            host=cls.es_host.get(es_host))
        contents = [{'data': {'content': t['content'], 'esid': t['esid']}} for t in to_label if t.get('content')]
        with open(saving_json, 'w') as f:
            json.dump(contents, f, ensure_ascii=False, indent=2)
        dt_logger.info('Fetched {} data starting from {}, saved {} data to {}',
                       len(to_label), exclude_num + 1, len(contents), saving_json)


class PreprocessorT5Base(PreprocessorBase):
    T5_COLS = ['prefix', 'input_text', 'target_text']
    CLASSIFICATION_TASK = None
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


class AkPreprocessorV1_0(PreprocessorT5Base):
    version = 'v1.0'
    ner_labels = ['head', 'product']
    task = 'product'
    CLASSIFICATION_TASK = []
    NER_TASK = [task]

    def preprocess_raw(self, label_result_json='data/labeled-2021-10-08-03-30-6961fb0e.json'):
        df = pd.DataFrame({
                              'prefix': self.task, **input_target
                          } for input_target in self.generate_input_target_text(label_result_json))
        self.save_to_h5(df)

    def generate_input_target_text(self, label_result_json):
        parser = LabelstudioData(label_result_json)
        for html, ner_xpath_texts in parser.generate_item(self.ner_labels):
            html_processor = HTMLProcessor(html)
            product_xpath_texts = ner_xpath_texts['product']
            if product_xpath_texts:
                product_xpaths, product_texts = zip(*product_xpath_texts)
                yield {
                    'input_text': html_processor.remove_html_attributes(html_processor.html_tree),
                    'target_text': self.seperator.join(product_texts)
                }

    def get_closest_parents(self, html_processor, head_xpaths, product_xpaths):
        """Return parent elements for each product element (vs head elements)."""
        parent_elements = np.array(
            [[html_processor.get_most_close_parent(head_p, product_p) for head_p in head_xpaths] for product_p in
             product_xpaths], dtype=object)
        root_distances = np.vectorize(html_processor.get_root_distance)(parent_elements)
        result_parent_element = parent_elements[np.arange(parent_elements.shape[0]), root_distances.argmax(axis=1)]
        return result_parent_element


class PreprocessorClassificationBase(PreprocessorBase):
    dataframe_columns = ['text', 'labels']

    @classmethod
    def balance_label(cls, df: pd.DataFrame, max_size=None, shuffle_result=True):
        max_sample_size = max_size or df['labels'].value_counts().max()
        res_df = pd.concat(
            resample(d, n_samples=max_sample_size, random_state=cls.random_state)
            if d.shape[0] != max_sample_size else d
            for label_, d in df.groupby('labels')
        )
        return res_df.sample(frac=1, random_state=cls.random_state) if shuffle_result else res_df

    def prepare_train_eval_dataset(self):
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


class AkPreprocessorV1_1(AkPreprocessorV1_0):
    version = 'v1.1'

    def generate_input_target_text(self, label_result_json, remove_style_element=False,
                                   len_threshold=None):
        parser = LabelstudioData(label_result_json)
        for html, ner_xpath_texts in parser.generate_item(self.ner_labels):
            html_processor = HTMLProcessor(html)
            product_xpath_texts = ner_xpath_texts['product']
            if product_xpath_texts:
                product_xpaths, product_texts = zip(*product_xpath_texts)
                yield {
                    'input_text': self.seperator.join(html_processor.remove_tags(
                        remove_style_element=remove_style_element, len_threshold=len_threshold)),
                    'target_text': self.seperator.join(product_texts)
                }


class AkPreprocessorV1_2(AkPreprocessorV1_1):
    version = 'v1.2'

    def generate_input_target_text(self, label_result_json, **kwargs):
        yield from super().generate_input_target_text(label_result_json, remove_style_element=True, len_threshold=100)


class AkPreprocessorV1_3(PreprocessorClassificationBase):
    version = 'v1.3'
    labelstudio_labels = ['head', 'product']
    blank_striper = HTMLProcessor.blank_striper

    def generate_text_labels(self, label_result_json):
        parser = LabelstudioData(label_result_json)
        for esid, html, ner_xpath_texts in parser.generate_item(self.labelstudio_labels, True):
            html_processor = HTMLProcessor(html)
            product_xpath_texts = ner_xpath_texts['product']
            if product_xpath_texts:
                product_xpaths, product_texts = zip(*product_xpath_texts)
                product_elements = html_processor.xpath_to_element(product_xpaths, product_texts)
                raw_element_texts = [html_processor.concat_element_text(*pe) for pe in product_elements]
                for element in html_processor.html_tree.iterdescendants():
                    text = ''.join(html_processor.remove_tags(element))
                    ele_text = html_processor.blank_striper.sub('', element.text) if element.text else ''
                    if text in raw_element_texts and text:
                        yield {'esid': esid, 'text': text, 'labels': 1}
                    elif ele_text:
                        if any(ele_text in p for p in product_texts):
                            yield {'esid': esid, 'text': next(p for p in product_texts if ele_text in p),
                                   'labels': 1}
                        else:
                            yield {'esid': esid, 'text': ele_text, 'labels': int(ele_text in raw_element_texts)}

    def preprocess_raw(self, label_result_json='data/labeled-2021-10-08-03-30-6961fb0e.json',
                       export_contradicts: str = None):
        df = pd.DataFrame(self.generate_text_labels(label_result_json))
        wrong = self.check_contradict_labels(df)
        wrong_text = wrong['text'].drop_duplicates()
        df['labels'] = df['labels'].where(~df['text'].isin(wrong_text), 1)
        dt_logger.info('All contradict labels set to 1.')
        if export_contradicts:
            wrong = self.check_contradict_labels(df)
            if wrong.empty:
                dt_logger.info('No contradict labels found.')
            else:
                wrong.to_excel(export_contradicts, index=False)
                dt_logger.info('Dataframe with contradict labels exported to "{}".', export_contradicts)
        self.save_to_h5(df)

    def preprocess_train_eval_dataset(self):
        train_df, eval_df = super().preprocess_train_eval_dataset(save_h5=False)
        train_df = self.balance_label(train_df)
        self.save_to_h5(train_df, 'train')
        self.save_to_h5(eval_df, 'eval')


class AkPreprocessorV1_4(PreprocessorT5Base):
    """Classification + T5"""
    version = 'v1.4'
    ner_labels = ['head', 'product']
    CLASSIFICATION_TASK = ['filter']
    NER_TASK = ['product']

    def preprocess_raw(self, label_result_json='data/labeled-2021-10-08-03-30-6961fb0e.json',
                       export_contradicts=None):
        df = pd.DataFrame(self.generate_input_target_text(label_result_json))
        wrong = self.check_cls_contradict_labels(df)
        wrong_text = wrong['input_text'].drop_duplicates()
        df['target_text'] = df['target_text'].where(
            df['prefix'].isin(self.NER_TASK) | ~df['input_text'].isin(wrong_text), '1')
        dt_logger.info('All contradict labels set to 1.')

        if export_contradicts:
            wrong = self.check_cls_contradict_labels(df)
            if wrong.empty:
                dt_logger.info('No contradict labels found.')
            else:
                wrong.to_excel(export_contradicts, index=False)
                dt_logger.info('Dataframe with contradict labels exported to "{}"', export_contradicts)
        self.save_to_h5(df)

    def generate_input_target_text(self, label_result_json):
        cls_task, ner_task = self.CLASSIFICATION_TASK[0], self.NER_TASK[0]
        parser = LabelstudioData(label_result_json)
        for esid, html, ner_xpath_texts in parser.generate_item(self.ner_labels, True):
            html_processor = HTMLProcessor(html)
            product_xpath_texts = ner_xpath_texts['product']
            if product_xpath_texts:
                product_xpaths, product_texts = zip(*product_xpath_texts)
                product_elements = html_processor.xpath_to_element(product_xpaths, product_texts)
                raw_element_texts = [html_processor.concat_element_text(*pe) for pe in product_elements]
                for element in html_processor.html_tree.iterdescendants():
                    text = ''.join(html_processor.remove_tags(element))
                    ele_text = html_processor.blank_striper.sub('', element.text) if element.text else ''
                    if text in raw_element_texts and text:
                        yield {'esid': esid, 'prefix': cls_task, 'input_text': text, 'target_text': '1'}
                        yield {'esid': esid, 'prefix': ner_task, 'input_text': text,
                               'target_text': product_texts[raw_element_texts.index(text)]}
                    elif ele_text:
                        if any(ele_text in p for p in product_texts):
                            sel_ind, sel_product_text = next((i, p) for i, p in enumerate(product_texts)
                                                             if ele_text in p)
                            yield {'esid': esid, 'prefix': cls_task, 'input_text': sel_product_text, 'target_text': '1'}
                            yield {'esid': esid, 'prefix': ner_task, 'input_text': sel_product_text,
                                   'target_text': product_texts[sel_ind]}
                        else:
                            isin = int(ele_text in raw_element_texts)
                            yield {'esid': esid, 'prefix': cls_task, 'input_text': ele_text, 'target_text': str(isin)}
                            if isin:
                                product_dists = [get_edit_distance(ele_text, t) for t in product_texts]
                                max_dist = max(product_dists)
                                yield {'esid': esid, 'prefix': ner_task, 'input_text': ele_text,
                                       'target_text': next(
                                           t for t, d in zip(product_texts, product_dists) if d == max_dist)}

    def preprocess_train_eval_dataset(self):
        train_df, eval_df = super().preprocess_train_eval_dataset(False)
        df_cls = self.upsampling_t5_classification_dataset(
            train_df[train_df['prefix'].isin(self.CLASSIFICATION_TASK)])
        df_ner = train_df[train_df['prefix'].isin(self.NER_TASK)]
        train_df_result = self.balance_t5_prefix(pd.concat([df_cls, df_ner]))
        self.save_to_h5(train_df_result, 'train')
        self.save_to_h5(eval_df, 'eval')


class AkPreprocessor:
    CLASSES = [AkPreprocessorV1_0, AkPreprocessorV1_1, AkPreprocessorV1_2, AkPreprocessorV1_3,
               AkPreprocessorV1_4]
    preprocessor_versions = {c.version: c for c in CLASSES}

    @classmethod
    def get_preprocessor_class(cls, version):
        return cls.preprocessor_versions.get(version)


if __name__ == '__main__':
    # p = AkPreprocessorV1_4()
    # p.preprocess_raw(export_contradicts='results/ak_v1.4_raw_contradict.xlsx')
    # p.preprocess_train_eval_dataset()
    # p.export_train_eval_dataset('results/ak_v1.4_train_eval_samples.xlsx', sample_n_train=1000)
    # p.describe_dataset('train')
    # p.describe_dataset('eval')
    LabelstudioData.pull_es_data('data/to_labels_1027.json', exclude_num=1000, size=2000)
