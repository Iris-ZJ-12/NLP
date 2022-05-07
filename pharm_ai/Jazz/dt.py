import pandas as pd
import numpy as np
import os.path
from pharm_ai.util.pdf_util.extractor_v2 import Extractor
import re
from statistics import mean

class PdfProcessor:
    def __init__(self):
        self.h5 = 'data.h5'
        self.zip_file = 'raw_data/jazz_pdf.zip'
        self.sentence_pattern = r'。([^。]*?再審査期間[^。]*?。)'
        self.table_pattern = r'[。]?([^。]*再審査期間[^。]*[。]?)'
        self.limit_pattern = r'再審査期間.*$'
        self.year_pattern = r'\D(\d{1,2}年)(?!\d{1,2}月)'

    def preprocess_pdf(self):
        """Extract fulltexts from zip of pdf files, and save to h5 """
        xlsx_file = 'raw_data/drug_jp.xlsx'
        df_raw = pd.read_excel(xlsx_file)
        df_raw['pdf_name'] = df_raw['review.review_url'].map(lambda p: os.path.split(p)[1], na_action='ignore')
        e = Extractor()
        extract_text, extract_table = e.extract(self.zip_file, self.zip_file)
        df_raw['pdf_content'] = df_raw['pdf_name'].map(lambda name: extract_text[name], na_action='ignore')
        df_raw['pdf_tables'] = df_raw['pdf_name'].map(lambda name: extract_table[name], na_action='ignore')
        df_raw.replace({'pdf_content': {'': None}}, inplace=True)
        df_raw['pdf_tables'] = df_raw['pdf_tables'].mask(df_raw['pdf_tables'].map(len, na_action='ignore').eq(0)).replace({np.NaN: None})
        df_raw.to_hdf(self.h5, 'raw')

    def extract_recensor_sentences(self, fulltext, use_pattern=1):
        """use_pattern : 1 or 2.
        1: use `self.fulltext_pattern`, 2: use `self.table_pattern`."""
        fulltext = fulltext.replace('\n','').replace(' ','')
        if use_pattern==1:
            r = re.search(self.sentence_pattern, fulltext)
        elif use_pattern==2:
            r = re.search(self.table_pattern, fulltext)
        if r:
            res = list(r.groups())
            return res


    def extract(self, df: pd.DataFrame = None, result_file:str=None,
                extra_columns = None):
        """df: [pandas.DataFrame]: Data containing extracted fulltext for each file.
                The input `df` should include following columns:
                - `pdf_name`: PDF filename;
                - `pdf_content`: extracted full text;
                - `pdf_tables`: extracted tables.
        extra_columns: [list]: Columns in `df` that will be included in result dataframe.
        result_file: [str] The excel filename for saving results.
                    If None, the result dataframe will be returned rather than saved."""
        if df is None:
            df: pd.DataFrame = pd.read_hdf(self.h5, 'raw')
        fulltext_sentences = df['pdf_content'].map(self.extract_recensor_sentences, na_action='ignore')
        fulltext_sentences = fulltext_sentences.map(lambda x:x[0], na_action='ignore')
        table_sentences = df['pdf_tables'].map(self.extract_recensor_table, na_action='ignore')
        table_matched = table_sentences.map(lambda res: res[1][0], na_action='ignore')
        table_sentences = table_sentences.map(lambda res: res[0][0], na_action='ignore')
        res_match = pd.DataFrame({'res_text': fulltext_sentences, 'res_tab': table_matched})
        probab_from_table = res_match.dropna().apply(lambda res: self.table_sentence_in_fulltext(res['res_tab'], res['res_text']), axis=1)
        resource = pd.Series(None, index=res_match.index, dtype='str')
        resource[fulltext_sentences.notna()] = 'text'
        resource[probab_from_table[probab_from_table>0].index] = 'table'
        result_extract = fulltext_sentences.where(resource != 'table', table_sentences)
        res_year = result_extract.dropna().apply(self.extract_year, separation='、')
        extra_columns = ['pdf_name'] if extra_columns is None else ['pdf_name'] + extra_columns
        result = df[extra_columns].assign(
            format = resource.replace({'text': 'textual', 'table': 'tabular'}),
            recensor_sentence = result_extract,
            extension_year = res_year
        )
        if result_file is None:
            return result
        else:
            result.to_excel(result_file, engine='xlsxwriter')

    def table_sentence_in_fulltext(self, table_list, fulltext):
        table_sentence = filter(bool, self.flatten_nested_list(table_list))
        is_in = lambda sentence: sentence.replace('\n','').replace(' ','') in fulltext
        res = list(map(is_in, table_sentence))
        return mean(res)


    def extract_recensor_table(self, table_lists, return_matched=True):
        extract_one_sentence = lambda sent: self.extract_recensor_sentences(sent, use_pattern=2)
        results = filter(lambda res: bool(res[2]),
                         ((table, row, extract_one_sentence(sent))
                          for table in table_lists if table for row in table if row for sent in row if sent))
        try:
            matched, matched_sentences, res_sentences = zip(*results)
            matched_sentences = [' '.join(map(lambda s: self.remove_space(s, True), sents))
                                 for sents in matched_sentences]
        except:
            matched, matched_sentences, res_sentences = [], [], []
        if res_sentences:
            if not return_matched:
                return self.flatten_nested_list(matched_sentences)
            else:
                return self.flatten_nested_list(matched_sentences), list(matched)

    def remove_space(self, str_: str, keep_return=False):
        if not keep_return:
            return str_.replace('\n','').replace(' ', '')
        else:
            return str_.replace(' ', '').replace('\n', ' ')


    def flatten_nested_list(self, li):
        return sum(([x] if not isinstance(x, list) else self.flatten_nested_list(x) for x in li), [])

    def extract_year(self, sentence, separation:str=None):
        limit_sentence = re.search(self.limit_pattern, sentence)
        if limit_sentence:
            limit_sentence = limit_sentence.group()
            r = re.findall(self.year_pattern, limit_sentence)
            if separation is None:
                return r
            else:
                return separation.join(r)

    def extract_single_pdf(self, pdf_name):
        e = Extractor()
        try:
            extract_text = e.extract_raw_full_text(pdf_name)
            extract_table = e.extract_raw_table(pdf_name)
            fulltext_sentence = self.extract_recensor_sentences(extract_text)[0]
            table_sentence, table_matched = self.extract_recensor_table(extract_table)
            table_sentence, table_matched = table_sentence[0], table_matched[0]
            probab_from_table = self.table_sentence_in_fulltext(table_matched, fulltext_sentence)
            if probab_from_table>0:
                resource = 'tabular'
                res_extract = table_sentence
            else:
                resource = 'textual'
                res_extract = fulltext_sentence
            res_year = self.extract_year(res_extract)
            result = {
                'format': resource,
                'recensor_sentence': res_extract,
                'extension_year': res_year
            }
            return result
        except Exception as ex:
            return {
                'format': None,
                'recensor_sentence': None,
                'extension_year': None
            }



if __name__=='__main__':
    x=PdfProcessor()
    # x.preprocess_pdf()
    x.extract(result_file='drug_jp_by_FZQ.xlsx',
              extra_columns=['esid', '批准时间', 'review.review_name', 'review.review_url'])