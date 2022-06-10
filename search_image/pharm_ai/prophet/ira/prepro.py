import pandas as pd
from loguru import logger
from mart.es_util.ESUtils7 import Query, QueryType, get_page
from sklearn.model_selection import train_test_split
import re
import os
import pickle
from bs4 import BeautifulSoup
import numpy as np
import langid
from pathlib import Path

class IraPreprocessor:
    """
    version:
        - v2.0: One prefix format, for Seq2Seq model.
        - v2.1: One prefix format, for mt5_base model.
        - v2.2: Multiple prefix format, for mt5_base model.
        - v2.3: Use v2.1 data, for mt5_small model.
        - v2.4: Use v2.2 data, for mt5_small model.

        v2.0-v2.4 raw data was loaded from "IRA generative data 1224 FZQ 已校验.xlsx"

        - v2.5: Second batch raw data, loaded from "投融资资讯-ira训练20210219.xls", multiple prefix format.
        - v2.6: use v2.5 data, separate seq2seq model for 3 tasks.
        - v2.7: Data augment of "amount" in addition to v2.6 dataset.
    """
    root_path = Path(__file__).parent
    def __init__(self, version='v2.0'):
        self.version = version
        self.random_state = 14

    def save_to_h5(self, df: pd.DataFrame, data_category='raw', suffix=None, excel_path=None, append_excel_sheet=False):
        data_file = self.root_path/'data.h5'
        ver = self.version.replace('.','-')
        key = f'{data_category}/{ver}'
        if suffix:
            key = f'{key}/{suffix}'
        if not excel_path:
            df.to_hdf(data_file, key)
            logger.info('Data has been saved to "{}" with key "{}"', data_file, key)
        else:
            mode_ = 'A' if append_excel_sheet else 'w'
            with pd.ExcelWriter(excel_path, mode=mode_) as writer:
                sheet_name = data_category if not suffix else data_category+'_'+suffix
                df.to_excel(writer, sheet_name)
            logger.info('{} data saved to "{}" in sheet "{}"', data_category, excel_path, sheet_name)

    def get_from_h5(self, data_category='raw', use_other_version=None, suffix=None) -> pd.DataFrame:
        data_file = self.root_path/'data.h5'
        from_version = self.version if use_other_version is None else use_other_version
        ver = from_version.replace('.', '-')
        key = f'{data_category}/{ver}'
        if suffix:
            key = f'{key}/{suffix}'
        res = pd.read_hdf(data_file, key)
        logger.info("Read data from '{}' with key {}", data_file, key)
        return res

    def preprocess_raw_datasets(self, excel_path=None, saving_excel_path=None):
        if self.version in ['v2.0', 'v2.1', 'v2.2', 'v2.5', 'v2.6']:
            if self.version in ['v2.5', 'v2.6']:
                excel_path = "raw_data/投融资资讯-ira训练20210219.xls"
                fulltext_cache_file = "raw_data/ira_fulltexts_0225.pkl"
                df_raw = pd.read_excel(excel_path)
                df_raw = df_raw.rename(
                    columns={'公司': 'investee', '轮次': 'round', '融资金额': 'amount', '发布时间': 'publish_date'})
                fulltexts = self.get_fulltext_from_esid(df_raw['ESID'].tolist(), fulltext_cache_file)
                fulltext_df = pd.DataFrame(fulltexts.items(), columns=['ESID','input_text'])
                df_raw = fulltext_df.merge(df_raw, on='ESID')
                df_raw = df_raw[df_raw['input_text'].ne('')]
                res_ira = df_raw[['input_text', 'investee', 'round', 'amount']]
            else:
                excel_path = "raw_data/IRA generative data 1224 FZQ 已校验.xlsx"
                df_raw = pd.read_excel(excel_path)
                res_ira = df_raw[df_raw['校验']==1][['input_text', 'investee', 'round', 'amount']]
            if self.version in ['v2.0', 'v2.1']:
                # prepare seq2seq dataset formate
                res_in_out = self.encoding(res_ira.dropna())
                if self.version=='v2.1':
                    res_in_out.insert(0, 'prefix', 'generate IRA')
            elif self.version=='v2.6':
                # prepare seq2seq dataset of 3 tasks seperately
                res_in_out = dict()
                for task in ['investee', 'round', 'amount']:
                    res_in_out[task] = res_ira[['input_text', task]].rename(columns={task: 'target_text'})
            else:
                # prepare t5 dataset format
                res_in_out = res_ira.dropna().melt(
                    id_vars=['input_text'],
                    value_vars=['investee', 'round', 'amount'],
                    var_name='prefix', value_name='target_text'
                )[['prefix','input_text','target_text']]
            if saving_excel_path:
                if self.version not in ['v2.1', 'v2.2']:
                    res = res_ira.join(res_in_out['target_text']).sort_values('target_text')
                    res.to_excel(saving_excel_path)
                elif self.version=='v2.6':
                    with pd.ExcelWriter(saving_excel_path) as writer:
                        for task in ['investee', 'round', 'amount']:
                            res_in_out[task].to_excel(writer, sheet_name=task)
                else:
                    res_in_out.to_excel(saving_excel_path)
            else:
                if self.version=='v2.6':
                    for task in ['investee', 'round', 'amount']:
                        self.save_to_h5(res_in_out[task], suffix=task)
                else:
                    self.save_to_h5(res_in_out)

    def encoding(self, df):
        outs = df[['investee','round','amount']].apply(
            lambda s: f"investee:{s['investee']}||round:{s['round']}||amount:{s['amount']}",
            axis = 1
        )
        res = pd.DataFrame({
            "input_text": df['input_text'],
            "target_text": outs
        })
        return res

    def decoding(self, to_decodes: list, recover_encoding=True, keep_case=False,
                 handle_round_expression=False):
        if self.version=='v2.6':
            result = [raw.replace(' ','') for raw in to_decodes]
            if handle_round_expression:
                result = [each.upper().replace('PRE','Pre') for each in result]
        else:
            if not recover_encoding:
                pattern = r'investee:([^|]*)\|\|?(?:round:([^|]*)\|\|?)?(?:amount:([^|]*))?'
                to_decodes = [self.remove_blank(raw) for raw in to_decodes]
            else:
                pattern = r'investee:([^|]*)\|\|round:([^|]*)\|\|amount:([^|]*)'
            result = []
            for raw in to_decodes:
                r = re.match(pattern, raw)
                if r:
                    cur_res = r.groups()
                    cur_i = (cur_res[0].title() if not keep_case else cur_res[0]) if cur_res[0] else ''
                    cur_r = (cur_res[1].upper() if not keep_case else cur_res[1]) if cur_res[1] else ''
                    cur_a = cur_res[2] if cur_res[2] else ''
                    result.append([cur_i, cur_r, cur_a])
                else:
                    result.append(['', '', ''])
        return result

    def preprocess_train_eval(self, eval_size=0.1, excel_path=None):
        if self.version=='v2.6':
            for task in ['investee', 'round', 'amount']:
                df_raw = self.get_from_h5(suffix=task)
                train_df, eval_df = train_test_split(df_raw, test_size=eval_size, random_state=self.random_state)
                self.save_to_h5(train_df, 'train', task, excel_path, False)
                self.save_to_h5(eval_df, 'eval', task, excel_path, True)
        else:
            df_raw = self.get_from_h5()
            train_df, eval_df = train_test_split(df_raw, test_size=eval_size, random_state=self.random_state)
            self.save_to_h5(train_df, 'train', excel_path=excel_path, append_excel_sheet=False)
            self.save_to_h5(eval_df, 'eval', excel_path=excel_path, append_excel_sheet=True)

    def get_train_eval_dataset(self, sample_train=None, task=None):
        get_version = None
        if self.version=='v2.3':
            get_version = 'v2.1'
        elif self.version=='v2.4':
            get_version = 'v2.2'
        train_df = self.get_from_h5('train', use_other_version=get_version, suffix=task)
        eval_df = self.get_from_h5('eval', use_other_version=get_version, suffix=task)
        if sample_train:
            if sample_train<=train_df.shape[0]:
                train_df = train_df.sample(sample_train, random_state=self.random_state)
                logger.info('Sampled {} from training dataset.', sample_train)
            else:
                logger.error('"sample_train" is larger than the size of training dataset {}', train_df.shape[0])
        return train_df, eval_df

    def remove_blank(self, raw_string: str):
        raw_splitted = re.split(r'\b', raw_string)
        splitted = [each_.replace(' ','') for each_ in raw_splitted if not re.fullmatch(r'\s*', each_)]
        nexts = splitted[1:]+['.']
        res_list = []
        for word, next_word in zip(splitted, nexts):
            if re.fullmatch(r'[a-zA-Z]+', word) and re.fullmatch(r'[a-zA-Z]+', next_word):
                res_list.append(word+' ')
            else:
                res_list.append(word)
        result_string = ''.join(res_list)
        return result_string

    def get_fulltext_from_esid(self, esids, pickle_file=None):
        if pickle_file and os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                res = pickle.load(f)
            logger.info('Fulltexts loaded from {}', pickle_file)
        else:
            contents = []
            for batch_start in range(0, len(esids), 1000): # number of max queries is 1024
                batch_end = min(len(esids), batch_start+1000)
                q = Query.queries(*[Query(QueryType.EQ, "_id", esid)
                                    for esid in esids[batch_start:batch_end]], and_perator=False)
                # cont = get_page("invest_news", queries=q, page_size=-1, show_fields='content',
                #                 host=('172.17.108.77','172.17.108.78','172.17.108.79'))
                cont = get_page("invest_news", queries=q, page_size=-1, show_fields='content')
                contents.extend(cont)
            res_list = [(content['esid'], self._tidy_fulltext(content['content']))
                        for content in contents if content['content']]
            res = dict(res_list)
            with open(pickle_file, 'wb') as f:
                pickle.dump(res, f)
            logger.info('Fulltexts written to {}', pickle_file)
        return res

    def _tidy_fulltext(self, html_content):
        soup = BeautifulSoup(html_content, features = 'html.parser')
        paras = [ss for s in soup.contents if hasattr(s, 'text') and s.text
                 for ss in s.text.split('\n') if ss]
        res = ' '.join(paras)
        return res

    def describe_dataset(self, item=None, suffix=None):
        if item:
            multi_task_versions = ['v2.6']
            if (self.version in multi_task_versions and suffix) or self.version not in multi_task_versions:
                df = self.get_from_h5(item, suffix=suffix)
                df = df.drop_duplicates(subset=['prefix', 'input_text'])
                logger.info('{} dataset: total={}', item, df['input_text'].drop_duplicates().shape[0])
                value_counts = df['prefix'].value_counts()
                logger.info('{} dataset: value counts: \n{}', item, value_counts.to_frame().to_dict())
            else:
                for suffix_ in ['investee', 'round', 'amount']:
                    self.describe_dataset(item=item, suffix=suffix_)
        else:
            for item_ in ['raw', 'train', 'eval']:
                self.describe_dataset(item=item_, suffix=suffix)

    def augment_amount_dataset(self, method=None):
        df_raw = self.get_from_h5('eval', 'v2.5')
        df = df_raw[df_raw['prefix']=='amount']
        if method=='jio':
            import jionlp
            # augment Chinese data
            df['money'] = df['input_text'].map(jionlp.extract_money)
            def split_num_unit(s):
                r = re.match(r'(\d*(?:\.\d+)?)([^\d\.]+)', s)
                if r:
                    return r.groups()
                else:
                    return None, None
            def is_extracted_match(s1, s2, only_num=False):
                ent1 = split_num_unit(s1)
                ent2 = split_num_unit(s2)
                if ent1[0] and ent2[0]:
                    res = (ent1[0]==ent2[0]) and (ent2[1] in ent1[1])
                elif not ent1[0] and not ent2[0] and not only_num:
                    res = (ent2[1] in ent1[1])
                else:
                    res = False
                return res
            def select_money(s1, s2s, only_num=False):
                res = None
                for s2 in s2s:
                    if is_extracted_match(s1, s2, only_num=only_num):
                        res = s2
                        break
                return res
            def random_nums_like(s: str, out_len: int):
                assert re.fullmatch(r'\d*(?:\.\d+)?', '.1'), 'input string is not number'
                if s.isdigit():
                    i = int(s)
                    k = int(np.log10(i))
                    r = np.power(10, [max(k-2, 0), k+2])
                    res_i = np.random.randint(r[0], r[1], out_len)
                    res = res_i.astype(str).tolist()
                else:
                    k = float(s)
                    res_i = np.random.normal(k, k/2, out_len)
                    res = ['%.2f'%r for r in abs(res_i)]
                return res
            def generate_random_money(s, t, out_len):
                """
                :param s: to be replaced entity.
                :param t: target entity.
                """
                num_, unit_ = split_num_unit(s)
                nums = random_nums_like(num_, out_len=out_len)
                res = {n + unit_: t.replace(num_, n) for n in nums}
                return res
            df['entity'] = [select_money(label, money_, True) if money_ else None
                            for label, money_ in zip(df['target_text'], df['money'])]
            df['replace_entity'] = [generate_random_money(ent, t, 5) if ent and t else {}
                                    for ent, t in zip(df['entity'], df['target_text'])]
            res_df = pd.DataFrame(
                ((row['prefix'], row['input_text'].replace(row['entity'], replace_), tar)
                    for _, row in df.iterrows() if row['replace_entity']
                    for replace_, tar in row['replace_entity'].items()),
                columns=['prefix', 'input_text', 'target_text']
            )
        elif method=='duckling':
            from fb_duckling import Duckling
            # stack exec duckling-example-exe
            duckling_en = Duckling(locale='en_US')
            duckling_cn = Duckling(locale='zh_CN')
            # TODO
        else:
            # TODO
            pass
        return res_df


if __name__ == '__main__':
    prepro = IraPreprocessor(version='v2.7')
    prepro.augment_amount_dataset(method='duckling')