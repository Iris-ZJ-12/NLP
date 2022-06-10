import json
import pickle
import sys
import warnings
from pathlib import Path

import langid
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm

from mart.prepro_util.utils import remove_html_tags
from pharm_ai.prophet.ira.prepro import IraPreprocessor
from pharm_ai.prophet.utils import remove_illegal_char, scroll_search, get_es

logger.add(sys.stderr, level='DEBUG', filter=lambda record: record["extra"].get("task")=="dt")
dt_logger = logger.bind(task="dt")

class PreprocessorBase:
    root_path = Path(__file__).parent
    data_file = root_path/'data.h5'
    version= None
    random_state = 713


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

        train_df=train_df.applymap(remove_illegal_char)
        eval_df = cls.get_from_h5('eval')
        if sample_n_eval:
            eval_df = eval_df.sample(sample_n_eval, random_state=cls.random_state)
        eval_df = eval_df.applymap(remove_illegal_char)
        with pd.ExcelWriter(saving_excel) as writer:
            train_df.to_excel(writer, sheet_name='train', index=False)
            eval_df.to_excel(writer, sheet_name='eval', index=False)
        dt_logger.info('Train and eval datasets saved to "{}".', saving_excel)

    @classmethod
    def concat_title_fulltext(cls, title, fulltext):
        lang_ = langid.classify(fulltext)[0]
        if lang_ == 'zh':
            res = f'标题：{title}。全文：{fulltext}'
        else:
            res = f'Title: {title}. Fulltext: {fulltext}'
        return res

    @classmethod
    def add_date_to_paragraph(cls, date_, para):
        """Add a sentence about publish date before the paragraph text."""
        lang_ = langid.classify(para)[0]
        if lang_ == 'zh':
            res_para = f'本文发布日期为{date_.year}年{date_.month}月{date_.day}日。{para}'
        else:
            res_para = f'This article was published on {date_.strftime("%B, %d, %Y")}. {para}'
        return res_para


class PreprocessorT5Base(PreprocessorBase):
    es_host = {'online': None, 'test': ('test155.cubees.com',)}
    es_index = 'invest_news'

    T5_COLS = ['prefix', 'input_text', 'target_text']
    CLASSIFICATION_TASK = None
    TO_CLS_COLS = ['esid', 'input_text', 'target_text']
    TO_CLS_MAPPING = {'input_text': 'text', 'target_text': 'labels'}

    def get_all_from_es(self, show_fields, host='online', *, titles=None):
        is_query = titles is not None
        es = get_es(host)
        if is_query:
            body = []
            for title in titles:
                body.append({})
                body.append({"query": {"match": {"title.raw": title}}, "_source": {"includes": show_fields}})
            raw = es.msearch(body, self.es_index)
            results = [{'esid': hit['_id'], **hit['_source']} for r in raw['responses'] for hit in r['hits']['hits']]
        else:
            results = scroll_search(es, self.es_index, _source_includes=show_fields)
        return results

    def _tidy_paragraphs(self, html_content):
        soup = BeautifulSoup(html_content, features='html.parser')
        paras = [ss for s in soup.contents if hasattr(s, 'text') and s.text for ss in s.text.split('\n') if ss]
        return paras

    @classmethod
    def upsampling_t5_classification_dataset(cls, df):
        """Balance data amount of each label.
        df: A classification dataset with only one prefix."""
        max_label_size = df['target_text'].value_counts().max()
        res_df = pd.concat(
            resample(d, n_samples=max_label_size, random_state=123) if d.shape[0] < max_label_size else d
            for label, d in df.groupby('target_text')
        )
        return res_df

    @classmethod
    def balance_t5_prefix(cls, df, max_size=None):
        """Upsampling or downsampling dataset of each prefix."""
        default_max_size = df['prefix'].value_counts().max()
        max_sample_size = min(default_max_size, max_size) if max_size else default_max_size

        res_df = pd.concat(
            resample(d, n_samples=max_sample_size, random_state=cls.random_state) if d.shape[0] != max_sample_size else d
            for prefix, d in df.groupby('prefix')
        )
        return res_df

    @classmethod
    def describe_dataset(cls, data_category='raw', classification_prefix=None,
                         null_target: str=None):
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
                pos_df = df[df['target_text']!=null_target]
                neg_df = df[df['target_text']==null_target]
                cls._pretty_print(head, f'size={df.shape[0]}', 'positive prefix counts:',
                                  pos_df['prefix'].value_counts(), 'negtive prefix counts:',
                                  neg_df['prefix'].value_counts())
            if cls_tasks:
                for prefix in cls_tasks:
                    df_cls = df[df['prefix']==prefix]
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
        df = raw_df[raw_df['prefix']==prefix][columns].rename(columns=cls.TO_CLS_MAPPING)
        if col_mapping:
            df['labels'] = df['labels'].map(col_mapping)
        return df


class ProphetT5PreprocessorV3_1(PreprocessorT5Base):
    """merge historical news_filter, ira, org_filter, org_ner."""
    version = 'v3.1'
    data_file = PreprocessorT5Base.root_path / 'data_t5.h5'
    classification_to_t5_column_mapper = {'text': 'input_text', 'labels': 'target_text'}

    def preprocess_raw(self, fill_empty_paragraph=True, ignore_org_ner=False):
        res_news_filter = self._preprocess_news_filter_raw(fill_empty_paragraph)

        # get ira data
        ira_prepro = IraPreprocessor(version='v2.5')
        res_ira = ira_prepro.get_from_h5()

        # get org_filter data
        org_filter_raw = pd.read_hdf('org_filter/data.h5', 'v1-1/raw')
        res_org_filter = org_filter_raw.rename(
            columns = self.classification_to_t5_column_mapper)[['input_text', 'target_text']].reset_index(drop=True)
        res_org_filter['target_text'] = res_org_filter['target_text'].astype(str)
        res_org_filter = res_org_filter.assign(prefix='org_filter')

        if not ignore_org_ner:
            # get org_ner data
            res_org_ner = self._preprocess_org_ner()

            # join all parts of data
            res = pd.concat([res_news_filter, res_ira, res_org_filter, res_org_ner]).reset_index(drop=True)
        else:
            res = pd.concat([res_news_filter, res_ira, res_org_filter]).reset_index(drop=True)

        return res

    def _preprocess_org_ner(self):
        org_ner_h5 = 'org_ner/data.h5'
        org_ner_raws = [pd.read_hdf(org_ner_h5, k)
                        for k in ['dt0921/train', 'dt0921/test', 'dt1204/raw']]
        # TODO: recover ner entities to t5 format

        return pd.concat(org_ner_raws)

    def _preprocess_news_filter_raw(self, fill_empty_paragraph=True):
        # get news_filter data
        news_filter_h5 = 'news_filter/data.h5'
        news_filter_raw = [pd.read_hdf(news_filter_h5, 'v6-1/raw'),
                           pd.read_hdf(news_filter_h5, 'v6-2/raw')[['ESID', 'title', 'text', 'labels']]]
        news_filter_raw[0] = news_filter_raw[0][news_filter_raw[0]['labels'].isin([0, 1, 2])]
        label_mapper = {0: '医药', 1: '非医药', 2: '非相关'}
        news_filter_raw[0]['labels'] = news_filter_raw[0]['labels'].map(label_mapper)
        news_filter_raw[0].rename(columns={'text': 'titles'}, inplace=True)
        if fill_empty_paragraph:
            temp_pickle_file = 'raw_data/news_filter_fulltexts_0303.pkl'
            paras_df = pd.DataFrame(self.get_fulltext_from_title(news_filter_raw[0]['titles'].tolist(), temp_pickle_file),
                                    columns=['titles', 'text'])
            news_filter_raw[0] = news_filter_raw[0].merge(paras_df, 'left')
            na_rows = (news_filter_raw[0]['text'].isna() | news_filter_raw[0]['text'].eq(''))
            dt_logger.debug('# empty texts: {}', na_rows.sum())
            news_filter_raw[0] = news_filter_raw[0][~na_rows]
        else:
            news_filter_raw[0]['text'] = news_filter_raw[0]['titles']
        news_filter_raw[1].rename(columns={'ESID': 'esid', 'title': 'titles'}, inplace=True)
        news_filter_raw[1]['labels'] = news_filter_raw[1]['labels'].map(label_mapper)
        news_filter_df = pd.concat(news_filter_raw).reset_index(drop=True)
        res_news_filter = news_filter_df.assign(prefix='news_filter').rename(
            columns=self.classification_to_t5_column_mapper)
        return res_news_filter

    def recover_paragraph_from_words(self, words:list):
        previous = [" "] + words[:-1]
        space_pattern = r"[\w']+"
        # prev_pattern = r"[\w,.?!]+"
        # separated_words = [w if re.fullmatch(w, space_pattern)
        #                    for w, p in zip(words, previous)]

    def get_fulltext_from_title(self, titles, pickle_file=None, return_paragraphs=False):
        """Input: titles
        Output: titles and each paragraphs."""
        if pickle_file and Path(pickle_file).exists():
            with open(pickle_file, 'rb') as f:
                contents = pickle.load(f)
            dt_logger.info('raw contents loaded from "{}"', pickle_file)
        else:
            contents = self.get_all_from_es(['title', 'content'], titles=titles)
            with open(pickle_file, 'wb') as f:
                pickle.dump(contents, f)
            dt_logger.info('raw contents written to "{}"', pickle_file)
        res_list = [(content['title'], self._tidy_paragraphs(content['content']))
                    for content in contents if content['content']]
        if return_paragraphs:
            return res_list
        else:
            res = [[l[0], ' '.join([l[0]]+l[1])] for l in res_list]
            return res


class ProphetT5PreprocessorV3_2(PreprocessorT5Base):
    version = 'v3.2'
    data_file = PreprocessorT5Base.root_path / 'data_t5.h5'
    invest_map = {'领投': 'c1', '跟投': 'c2', '其他': 'c3', '其它': 'c3', '投资顾问': 'f'}
    ira_org_fields = ['content', 'company', 'round', 'amount', 'invester', 'unmatch_investers', 'financing_record',
                      'prophet_intermediary_output', 'publish_date']
    seperator = ';'
    CLASSIFICATION_TASK = ['news_filter', 'org_filter', 'ira_filter']
    NER_TASK = ['investee', 'round', 'amount', 'org_ner.c1', 'org_ner.c2', 'org_ner.c3', 'org_ner.f']
    TASK = CLASSIFICATION_TASK + NER_TASK

    def __init__(self, filter_data_file='results/news_filter_data_v3.2.json',
                 ira_org_data_file='results/ira_org_data_v3.2.json',
                 company_name_file='results/company_dictionary.json',
                 invester_list_file='results/invester_list.json'):
        self.filter_data_file=self.root_path/filter_data_file
        self.filter_data_file.parent.mkdir(parents=True, exist_ok=True)
        self.ira_org_data_file = self.root_path/ira_org_data_file
        self.ira_org_data_file.parent.mkdir(parents=True, exist_ok=True)
        self.company_dic = self.get_company_dictionary(company_name_file)
        self.invester_list = self.get_invester_list(invester_list_file)


    def get_news_filter_data(self, host='online'):
        selects = ['title', 'content', 'label']
        body = {"query": {"term": {"is_new": '0'}}}
        es = get_es(host)
        filter_data = scroll_search(es, self.es_index, body, _source_includes=selects)
        with open(self.filter_data_file, 'w') as f:
            json.dump(filter_data, f, ensure_ascii=False, indent=4)
        dt_logger.success('Fetched {} news_filter data, saved to "{}".', len(filter_data), self.filter_data_file)
        return filter_data

    def get_ira_org_data(self, host='online'):
        body = {"query": {"term": {"is_publish": '1'}}}
        es = get_es(host)
        fetched = scroll_search(es, self.es_index, body, _source_includes=self.ira_org_fields)
        with open(self.ira_org_data_file, 'w') as f:
            json.dump(fetched, f, ensure_ascii=False, indent=4)
        dt_logger.success('Fetched {} ira&org data, saved to "{}".', len(fetched), self.ira_org_data_file)
        return fetched

    def preprocess_raw_not_check(self, cache_news_filter_df='results/news_filter_df_v3.2.pkl',
                                 cache_ira_org_df='results/ira_org_df_v3.2.pkl'):
        path_news_filter_df = Path(cache_news_filter_df)
        if path_news_filter_df.exists():
            with path_news_filter_df.open('rb') as f:
                df1 = pickle.load(f)
                dt_logger.info('News_filter dataset loaded from "{}".', cache_news_filter_df)
        else:
            news_filter_df = pd.DataFrame(tqdm(self.preprocess_news_filter_data()))
            df1 = pd.DataFrame({
                'esid': news_filter_df['esid'],
                'prefix': 'news_filter',
                'titles': news_filter_df['title'],
                'input_text': news_filter_df[['title','fulltext']].apply(
                    lambda row: self.concat_title_fulltext(row['title'], row['fulltext']), axis=1),
                'target_text': news_filter_df['label']
            })
            with path_news_filter_df.open('wb') as f:
                pickle.dump(df1, f)
                dt_logger.info('News_filter dataset saved to "{}".', cache_news_filter_df)
        path_ira_org_df = Path(cache_ira_org_df)
        if path_ira_org_df.exists():
            with path_ira_org_df.open('rb') as f:
                df2 = pickle.load(f)
                dt_logger.info('IRA&org dataset loaded from "{}".', cache_ira_org_df)
        else:
            df2 = pd.DataFrame(tqdm(self.preprocess_ira_org_data()))
            df2['publish_date'] = pd.to_datetime(df2['publish_date'], unit='ms')
            with path_ira_org_df.open('wb') as f:
                pickle.dump(df2, f)
                dt_logger.info('Ira&Org dataset saved to "{}".', cache_ira_org_df)
        res_df = pd.concat([df1, df2], ignore_index=True)
        return res_df


    def preprocess_news_filter_data(self):
        with open(self.filter_data_file, 'r') as f:
            raw = json.load(f)
        for t in raw:
            content = t.get('content')
            fulltext = self._get_fulltext_from_content(content)
            yield {
                'esid': t.get('esid'),
                'title': t.get('title'),
                'fulltext': fulltext,
                'label': t.get('label')
            }

    def preprocess_ira_org_data(self, filter_start=None, filter_end=None):
        with self.ira_org_data_file.open('r') as f:
            raw = json.load(f)
        if filter_start or filter_end:
            raw = raw[filter_start:filter_end]
        for t in raw:
            esid = t.get('esid')
            intermediary_str = t.get('prophet_intermediary_output')
            company = t.get('company')
            investee = self.company_id_to_name(company)
            round = t.get('round')
            amount = t.get('amount')
            publish_date = t.get('publish_date')
            if intermediary_str:
                intermediary_output = json.loads(intermediary_str)
                ira_filer_labels = intermediary_output.get('ira_filter_labels')
                org_filter_labels = intermediary_output.get('org_filter_labels')
                if not org_filter_labels and ira_filer_labels:
                    dt_logger.warning('No org_filter data in {}.', esid)
                    org_filter_labels = [None]*len(ira_filer_labels)
                elif not ira_filer_labels and org_filter_labels:
                    dt_logger.warning('No ira_filter data in {}.', esid)
                    ira_filer_labels = [[0,""]]*len(org_filter_labels)
                elif not ira_filer_labels and not org_filter_labels:
                    ira_filer_labels, org_filter_labels = [[0, ""]], [None]
                if not isinstance(ira_filer_labels[0], (tuple, list)):
                    # new intermediary output format
                    ira_filer_labels = list(zip(ira_filer_labels, intermediary_output.get('paragraphs')))
                for (ira_label, para), org_label in zip(ira_filer_labels, org_filter_labels):
                    # ira filter data
                    if para:
                        yield {'esid': esid, 'publish_date': publish_date,
                               'prefix': 'ira_filter', 'input_text': para, 'target_text': str(ira_label)}
                        # org filter data
                        yield {'esid': esid, 'publish_date': publish_date,
                               'prefix': 'org_filter', 'input_text':para, 'target_text': str(org_label)}
                        if org_label:
                            # org ner data
                            yield from self._process_org_ner(t, para, esid)
                        # ira
                        if ira_label==1:
                            if investee:
                                yield {'esid': esid, 'prefix': 'investee', 'input_text': para, 'target_text': investee}
                            else:
                                dt_logger.warning('{}: No investee matched company "{}".', esid, company)
                            yield {'esid': esid, 'prefix': 'round', 'input_text': para, 'target_text': round}
                            yield {'esid': esid, 'prefix': 'amount', 'input_text': para, 'target_text': amount}
            else:
                dt_logger.warning('{} use fulltext as ira&org_ner paragraph.', esid)
                content = t.get('content')
                fulltext = self._get_fulltext_from_content(content)
                if fulltext:
                    # ira
                    if investee:
                        yield {'esid': esid, 'prefix': 'investee', 'input_text': fulltext, 'target_text': investee}
                    yield {'esid': esid, 'prefix': 'round', 'input_text': fulltext, 'target_text': round}
                    yield {'esid': esid, 'prefix': 'amount', 'input_text': fulltext, 'target_text': amount}
                    # org
                    yield from self._process_org_ner(t, fulltext, esid)

    def _process_org_ner(self, item_dict, para, esid, process_financial_advisor=False, yield_null=False):
        investers_str = item_dict.get('invester')
        investers = [] if not investers_str or investers_str == '未披露' else investers_str.split(self.seperator)
        financing_records = item_dict.get('financing_record')
        unmatched = item_dict.get('unmatch_investers')
        financial_advisor_esids = item_dict.get('financial_advisor')
        if unmatched is None:
            unmatched = []
        invester_types = dict()
        if financing_records and investers:
            for r, name_ in zip(financing_records, investers):
                bind_id = r.get('bind_capital_id')
                invester_name = self.invest_id_to_name(bind_id, investers) if bind_id else name_
                if invester_name is None:
                    invester_name = name_
                invester_types[invester_name] = self.invest_map.get(r.get('behavior'))
        # prefix: [org_ner.c1, org_ner.c2, org_ner.c3]
        yield_types = set(invester_types.values())
        for t in ['c1', 'c2', 'c3']:
            org_ner_prefix = f'org_ner.{t}'
            if t in yield_types:
                if t:
                    yield {'esid': esid, 'prefix': org_ner_prefix, 'input_text': para,
                           'target_text': self.seperator.join(k for k, v in invester_types.items() if v==t)}
                else:
                    unmatched+=[k for k,v in invester_types.items() if v is None]
            elif yield_null:
                yield {'esid': esid, 'prefix': org_ner_prefix, 'input_text': para,
                       'target_text': getattr(self, 'ORG_NULL')}

        # prefix: org_ner.f
        if financial_advisor_esids and process_financial_advisor:
            finalcial_advisor = [self.invest_id_to_name(fid) for fid in financial_advisor_esids]
            if finalcial_advisor:
                yield {'esid': esid, 'prefix': 'org_ner.f', 'input_text': para,
                       'target_text': self.seperator.join(filter(bool, finalcial_advisor))}
        elif yield_null and process_financial_advisor:
            yield {'esid': esid, 'prefix': 'org_ner.f', 'input_text': para,
                   'target_text': getattr(self, 'ORG_NULL')}
        # prefix: org_ner
        if unmatched:
            yield {'esid': esid, 'prefix': 'org_ner', 'input_text': para, 'target_text': self.seperator.join(unmatched)}

    def _get_fulltext_from_content(self, content):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            fulltext = remove_html_tags(content) if content else ""
        return fulltext

    def company_id_to_name(self, company_id: str):
        if isinstance(company_id, int):
            company_id = str(company_id)
        if not company_id:
            return None
        elif company_id.isdigit():
            return self.company_dic.get(company_id) or company_id
        else:
            return company_id

    def invest_id_to_name(self, invest_id, ref_names=None):
        if invest_id:
            invest_names = self.invester_list.get(invest_id)
            if invest_names:
                alternative_res = list(invest_names.values())
                res = [r for r in alternative_res if r in ref_names] if ref_names is not None else alternative_res
                return res[0] if res else None
        else:
            return None

    @classmethod
    def get_company_dictionary(cls, company_name_file = 'results/company_dictionary.json'):
        company_names_path = cls.root_path/company_name_file
        if company_names_path.exists():
            with company_names_path.open('r') as f:
                res = json.load(f)
            dt_logger.info('{} company names loaded from "{}".', len(res), company_name_file)
        else:
            es = get_es('online')
            r = scroll_search(es, 'base_company', _source_includes=["id", "short_name"])
            res = dict((rr['id'], rr['short_name']) for rr in r if rr.get('short_name') and rr.get('id'))
            with company_names_path.open('w') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)
            dt_logger.success('{} company names saved to "{}".', len(res), company_name_file)
        return res

    @classmethod
    def get_invester_list(cls, invester_list_file = 'results/invester_list.json'):
        invester_list_path = cls.root_path/invester_list_file
        if invester_list_path.exists():
            with invester_list_path.open('r') as f:
                res = json.load(f)
            dt_logger.info('{} investers loaded from "{}".', len(res), invester_list_file)
        else:
            es = get_es('online')
            r = scroll_search(es, 'invest_capital', _source_includes=['name_short_en', 'name_short'])
            esids = [rr.pop('esid') for rr in r]
            res = dict(zip(esids, r))
            with invester_list_path.open('w') as f:
                json.dump(res, f, indent=4, ensure_ascii=False)
            dt_logger.success('{} investers saved to "{}".', len(res), invester_list_file)
        return res


    def prepare_to_check_dataset(self, saving_excel='results/prophet_tocheck_v3_20210706.xlsx',
                                 not_check_pickle_path='results/not_to_check_v3.pkl'):
        """Prepare ira_filter, org_filter dataset to check."""

        raw_df = self.preprocess_raw_not_check()
        last_prepro = ProphetT5PreprocessorV3_1()
        raw_last = last_prepro.preprocess_raw(fill_empty_paragraph=False,
                                              ignore_org_ner=True)
        not_check = dict()

        # ira_filter data: to-check
        ira_filter_df = raw_df[raw_df['prefix'] == 'ira_filter']
        ira_filter_to_check = ira_filter_df[['esid', 'publish_date', 'input_text', 'target_text']].rename(
            columns={'input_text': 'paragraphs', 'target_text': 'predict_label'}
        )
        ira_filter_to_check['predict_label'] = ira_filter_to_check['predict_label'].astype(int)
        ira_filter_to_check['true_label'] = None

        # org_filter data: to-check
        org_filter_to_check = raw_df[raw_df['prefix']=='org_filter']
        org_filter_to_check_df = org_filter_to_check[['esid', 'publish_date', 'input_text', 'target_text']].rename(
            columns={'input_text': 'paragraphs', 'target_text': 'predicted_label'}
        )
        org_filter_to_check_df['predicted_label'] = org_filter_to_check_df['predicted_label'].astype(int)
        org_filter_to_check_df['true_label'] = None

        # org_filter data: not-to-check
        org_filter_last = raw_last[raw_last['prefix'] == 'org_filter']
        pattern = r'(?:本文发布日期为(?P<date_cn>\d+年\d+月\d+日)。(?P<para_cn>.*))|(?:This article was published on (?P<date_en>\d+-\d+-\d+). (?P<para_en>.*))'
        org_filter_last_ex = org_filter_last['input_text'].str.extract(pattern)
        org_filter_last_ex['date_cn'] = pd.to_datetime(org_filter_last_ex['date_cn'], format="%Y年%m月%d日")
        org_filter_last_ex['date_en'] = pd.to_datetime(org_filter_last_ex['date_en'], format="%Y-%m-%d")
        org_filter_last['input_text'] = org_filter_last_ex['para_cn'].where(
            org_filter_last_ex['para_cn'].notna(), org_filter_last_ex['para_en'])
        org_filter_last['publish_date'] = org_filter_last_ex['date_cn'].where(
            org_filter_last_ex['date_cn'].notna(), org_filter_last_ex['date_en'])
        not_check['org_filter'] = org_filter_last

        # write to excel and pkl
        with pd.ExcelWriter(saving_excel) as writer:
            ira_filter_to_check.to_excel(writer, sheet_name='ira_filter', index=False)
            org_filter_to_check_df.to_excel(writer, sheet_name='org_filter', index=False)
        dt_logger.info('To-check data saved to {}', saving_excel)
        with open(not_check_pickle_path, 'wb') as f:
            pickle.dump(not_check, f)
        dt_logger.info('Not-to-check data dumped to {}', not_check_pickle_path)


    def preprocess_raw(self, checked_xlsx = "raw_data/prophet_tocheck_v3_20210706已核对.xlsx",
                       not_check_pickle_path='results/not_to_check_v3.pkl'):
        """Update checked ira_filter, org_filter datasets."""

        raw_df = self.preprocess_raw_not_check()
        with open(not_check_pickle_path, 'rb') as f:
            not_check = pickle.load(f)
        org_filter_not_check = not_check['org_filter']
        org_filter_not_check['input_text'] = org_filter_not_check.apply(
            lambda row: self.add_date_to_paragraph(row['publish_date'], row['input_text']),
            axis=1
        )

        # read checked data
        with pd.ExcelFile(checked_xlsx) as f:
            checked_ira_filter = pd.read_excel(f, sheet_name='ira_filter',
                                               dtype={'predict_label': 'str', 'true_label': 'str'})
            checked_org_filter = pd.read_excel(f, sheet_name='org_filter',
                                               dtype={'predict_label': 'str', 'true_label': 'str'})

        # tidy data
        column_mapper = {'true_label': 'target_text'}
        checked_ira_filter = checked_ira_filter.rename(columns=column_mapper)
        checked_ira_filter['prefix'] = 'ira_filter'
        checked_ira_filter['publish_date'] = checked_ira_filter['publish_date'].dt.tz_localize('utc').dt.tz_convert(
            'Asia/Shanghai')
        checked_ira_filter['input_text'] = checked_ira_filter.apply(
            lambda row: self.add_date_to_paragraph(row['publish_date'], row['paragraphs']),
            axis=1
        )
        checked_org_filter = checked_org_filter.rename(columns=column_mapper)
        checked_org_filter['prefix'] = 'org_filter'
        checked_org_filter['publish_date'] = checked_org_filter['publish_date'].dt.tz_localize('utc').dt.tz_convert(
            'Asia/Shanghai')
        checked_org_filter['input_text'] = checked_org_filter.apply(
            lambda row: self.add_date_to_paragraph(row['publish_date'], row['paragraphs']),
            axis=1
        )

        # prepare total data
        not_check_t5_data = raw_df[raw_df['prefix'].isin([
            'news_filter', 'investee', 'round', 'amount']) | raw_df['prefix'].str.startswith('org_ner.')]
        total_raw_data = pd.concat([not_check_t5_data, checked_ira_filter, checked_org_filter, org_filter_not_check])
        total_raw_data = total_raw_data[
            total_raw_data['prefix'].ne('news_filter') | total_raw_data['target_text'].isin(['医药', '非医药', '非相关'])
        ] # filter only relevant labels

        # save
        self.save_to_h5(total_raw_data)


    def prepare_train_eval_dataset(self):
        df_raw = self.get_from_h5()
        train_df, eval_df = train_test_split(df_raw[['esid', 'prefix', 'input_text', 'target_text']],
                                             test_size=0.1, random_state=self.random_state)

        # split to upsample
        df_groups = {pref: d for pref, d in train_df.groupby('prefix')}

        # upsample classification tasks
        cls_upsampled = pd.concat(self.upsampling_t5_classification_dataset(d)
                                  for pref, d in df_groups.items() if pref in self.CLASSIFICATION_TASK)
        train_cls_upsampled = pd.concat([cls_upsampled] + [d for pref, d in df_groups.items() if pref in self.NER_TASK])

        # upsample over prefix
        train_upsampled = self.balance_t5_prefix(train_cls_upsampled, max_size=100000).sample(
            frac=1, random_state=self.random_state)

        # saving
        self.save_to_h5(train_upsampled, 'train')
        self.save_to_h5(eval_df, 'eval')


class ClassificationProcessorBase(PreprocessorBase):
    dataframe_columns = ['text', 'labels']

    @classmethod
    def balance_label(cls, df: pd.DataFrame, max_size=None):
        max_sample_size = max_size or df['labels'].value_counts().max()
        res_df = pd.concat(
            resample(d, n_samples=max_sample_size, random_state=cls.random_state)
            if d.shape[0] != max_sample_size else d
            for label_, d in df.groupby('labels')
        )
        return res_df

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

class NewsFilterPreprocessorV3_3(ClassificationProcessorBase):
    version = 'v3.3.0'
    prefix = 'news_filter'
    col_mapping = {'医药': 0, '非医药': 1, '非相关': 2}

    def preprocess_raw(self):
        rely_preprocessor = ProphetT5PreprocessorV3_2()
        df = rely_preprocessor.to_cls_data(self.prefix, self.col_mapping)
        self.save_to_h5(df)


class IraFilterPreprocessorV3_3(ClassificationProcessorBase):
    version = 'v3.3.1'
    prefix = 'ira_filter'

    def preprocess_raw(self):
        rely_preprocessor = ProphetT5PreprocessorV3_2()
        df = rely_preprocessor.to_cls_data(self.prefix)
        df['labels'] = df['labels'].astype(int, copy=True)
        self.save_to_h5(df)

class OrgFilterPreprocessorV3_3(ClassificationProcessorBase):
    version = 'v3.3.2'
    prefix = 'org_filter'

    def preprocess_raw(self):
        rely_preprocessor = ProphetT5PreprocessorV3_2()
        df = rely_preprocessor.to_cls_data(self.prefix)
        df['labels'] = df['labels'].astype(int, copy=True)
        self.save_to_h5(df)


class ProphetT5PreporcessorV3_3_3(ProphetT5PreprocessorV3_2):
    """Drop classification tasks."""
    version = 'v3.3.3'
    CLASSIFICATION_TASK = []

    def preprocess_raw(self, **kwargs):
        df_raw = ProphetT5PreprocessorV3_2.get_from_h5()
        df = df_raw[~df_raw['prefix'].isin(ProphetT5PreprocessorV3_2.CLASSIFICATION_TASK)]
        self.save_to_h5(df)

    def prepare_train_eval_dataset(self):
        df_raw = self.get_from_h5()
        train_df, eval_df = train_test_split(df_raw[['esid', *self.T5_COLS]],
                                             test_size=0.1, random_state=self.random_state)
        train_upsampled = self.balance_t5_prefix(train_df).sample(
            frac=1, random_state=self.random_state)
        self.save_to_h5(train_upsampled, 'train')
        self.save_to_h5(eval_df, 'eval')


class ProphetT5PreprocessorV3_4(ProphetT5PreprocessorV3_2):
    """Split org_ner into two tasks:
    org_ner.c1 + org_ner.c2 + org_ner.c3 + org_ner.f -> org_ner + org_cls"""

    version = 'v3.4'
    ORG_CLS = 'org_cls'
    CLASSIFICATION_TASK = ProphetT5PreprocessorV3_2.CLASSIFICATION_TASK + [ORG_CLS]
    NER_TASK = ['investee', 'round', 'amount', 'org_ner']

    def preprocess_raw(self, **kwargs):
        raw_df = ProphetT5PreprocessorV3_2.get_from_h5()
        cond = raw_df['prefix'].str.startswith('org_ner')
        org_ner_df = raw_df[cond].copy()
        gen = self.seperate_org_ner_task(org_ner_df)
        res_org_ner_df = pd.concat(gen, ignore_index=True)
        res_df = pd.concat([raw_df[~cond], res_org_ner_df], ignore_index=True)
        self.save_to_h5(res_df)

    @classmethod
    def seperate_org_ner_task(cls, raw_org_ner_df):
        org_cls = raw_org_ner_df['prefix'].str.split('.', expand=True)
        raw_org_ner_df['cls'] = org_cls.loc[:, 1]
        for input_text, d in raw_org_ner_df.groupby('input_text'):
            d1 = pd.DataFrame([{'esid': d['esid'].iloc[0], 'prefix': 'org_ner', 'input_text': input_text,
                                'target_text': cls.seperator.join(d['target_text'])}])
            yield d1
            d['target_text'] = d['target_text'].str.split(cls.seperator)
            d2 = d.explode('target_text')[['esid', 'cls', 'target_text']].rename(
                columns={'target_text': 'input_text', 'cls': 'target_text'}).assign(prefix=cls.ORG_CLS)
            yield d2

class ProphetT5PreprocessorV3_5(ProphetT5PreprocessorV3_2):
    version = 'v3.5'
    # add financial_advisor data
    ira_org_fields = ProphetT5PreprocessorV3_2.ira_org_fields + ['financial_advisor']

    def __init__(self, ira_org_data_file='results/ira_org_data_v3.5.json'):
        super(ProphetT5PreprocessorV3_5, self).__init__(ira_org_data_file=ira_org_data_file)

    def preprocess_raw_not_check(self, cache_ira_org_df='results/ira_org_df_v3.5.pkl', **kwargs):
        return super(ProphetT5PreprocessorV3_5, self).preprocess_raw_not_check(
            cache_ira_org_df=cache_ira_org_df, **kwargs
        )

    def _process_org_ner(self, *args, **kwargs):
        return super(ProphetT5PreprocessorV3_5, self)._process_org_ner(*args, **kwargs, process_financial_advisor=True)

    def prepare_train_eval_dataset(self):
        df_raw = self.get_from_h5()
        # drop cls dataset
        train_df, eval_df = train_test_split(
            df_raw.loc[
                ~df_raw['prefix'].isin(self.CLASSIFICATION_TASK),
                ['esid', *self.T5_COLS]
            ],
            test_size=0.1, random_state=self.random_state
        )
        train_upsampled = self.balance_t5_prefix(train_df).sample(
            frac=1, random_state=self.random_state
        )
        self.save_to_h5(train_upsampled, 'train')
        self.save_to_h5(eval_df, 'eval')

class ProphetT5PreprocessorV3_6(ProphetT5PreprocessorV3_5):
    version = 'v3.6'
    ORG_NULL = '--'

    def preprocess_raw_not_check(self, cache_ira_org_df='results/ira_org_df_v3.6.pkl', **kwargs):
        return super(ProphetT5PreprocessorV3_6, self).preprocess_raw_not_check(
            cache_ira_org_df=cache_ira_org_df, **kwargs
        )

    def _process_org_ner(self, *args, **kwargs):
        # set yield_null, also generate org_ner even if not exist
        return super(ProphetT5PreprocessorV3_6, self)._process_org_ner(*args, **kwargs, yield_null=True)

    def describe_dataset(cls, data_category='raw'):
        super().describe_dataset(data_category=data_category, null_target=cls.ORG_NULL)


class ProphetT5PreprocessorV3_7(ProphetT5PreprocessorV3_6, ProphetT5PreporcessorV3_3_3):
    version = 'v3.7'
    CLASSIFICATION_TASK = ProphetT5PreporcessorV3_3_3.CLASSIFICATION_TASK

    def preprocess_raw(self, **kwargs):
        raw1 = ProphetT5PreporcessorV3_3_3.get_from_h5()
        raw2 = ProphetT5PreprocessorV3_6.get_from_h5()
        raw2 = raw2.loc[raw2['prefix'].isin(self.NER_TASK)]
        raw = pd.concat([raw1, raw2]).drop_duplicates(
            subset=['esid', 'prefix', 'input_text', 'target_text'])
        self.save_to_h5(raw)


class ProphetPreproessor:
    CLASSES = [ProphetT5PreprocessorV3_1, ProphetT5PreprocessorV3_2, NewsFilterPreprocessorV3_3,
               IraFilterPreprocessorV3_3, OrgFilterPreprocessorV3_3, ProphetT5PreporcessorV3_3_3,
               ProphetT5PreprocessorV3_4, ProphetT5PreprocessorV3_5, ProphetT5PreprocessorV3_6]
    preprocessor_versions = {c.version: c for c in CLASSES}

    @classmethod
    def get_preprocessor_class(cls, version):
        return cls.preprocessor_versions.get(version)


if __name__ == '__main__':
    p = ProphetT5PreprocessorV3_7()
    p.preprocess_raw()
    p.prepare_train_eval_dataset()
    p.export_train_eval_dataset('results/prophet_v3.7_train_eval_samples.xlsx', sample_n_train=3000,
                                sample_n_eval=200)
    p.describe_dataset('train')
    p.describe_dataset('eval')
else:
    logger.remove()