# -*- coding: UTF-8 -*-
"""
Description : 
"""
from pharm_ai.util.ESUtils6 import Query, QueryType, get_page
from pharm_ai.util.utils import Utilfuncs
from pharm_ai.util.prepro import Prepro
import pandas as pd
import re


def dt_drug():
    res = get_page("drug_wct", show_fields=['esid', 'nct_id', 'criteria', 'official_title'],
                   page_size=500, host=("172.17.108.77", "172.17.108.78", "172.17.108.79")
                   )  # 'criteria', 'official_title'
    print(res)
    df = pd.DataFrame(res)
    df.to_excel('./drug_wct_0316_raw.xlsx')


def dt_0316():
    df = pd.read_excel('./data/v2.1/drug_wct_0316_raw.xlsx')  # type:pd.DataFrame
    Utilfuncs.to_excel(df, './data/v2.1/v2_1_0316.xlsx', sheet_name='raw')
    li = []
    p = Prepro()
    for _, row in df.iterrows():
        sentence_id = 0
        if not pd.isnull(row['official_title']):
            texts = p.cut_sentences(row['official_title'].strip())
            for i in texts:  # type:str
                if i[0].isalnum() and i[-1] == '.' and len(i) <= 4:
                    pass
                else:
                    li.append({'esid': row['esid'], 'NCT': row['nct_id'], 'type': 'official_title',
                               'sentence_id': sentence_id, 'sentence': row['official_title']})
                    sentence_id += 1

        if not pd.isnull(row['criteria']):
            sentences = str(row['criteria']).split('\r<br/>\r<br/>')
            for sen in sentences:  # type:str

                if '<br/>' in sen:
                    sen = sen.replace('<br/>', ' ')
                if '\r<br/>' in sen:
                    sen = sen.replace('\r<br/>', ' ')

                sen = Utilfuncs.remove_html_tags(sen)

                sen = sen.replace('\r', ' ')
                sen = sen.replace('/ ', '/')
                if 'e.g. ' in sen:
                    sen = sen.replace('e.g. ', 'e.g., ')
                sen = re.sub(' +', ' ', sen)
                texts = p.cut_sentences(sen.strip())
                for i in texts:  # type:str
                    if i[0].isalnum() and i[-1] == '.' and len(i) <= 4:
                        pass
                    else:
                        li.append({'esid': row['esid'], 'NCT': row['nct_id'], 'type': 'criteria',
                                   'sentence_id': sentence_id, 'sentence': i})
                        sentence_id += 1

    df_ = pd.DataFrame(li)
    df_.drop_duplicates(subset=['sentence'], inplace=True)
    Utilfuncs.to_excel(df_, './data/v2.1/v2_1_0316.xlsx', sheet_name='to_labeled')

    # sen = re.sub('[\d][.] ', '', sen)
    # print(sen)

    # rows_list = []
    # for i in res:
    #     sentence_id = 0
    #     es_id = i['esid']
    #     if 'official_title' in i.keys():
    #         official_title = i['official_title']
    #         rows_list.append(
    #             {'esid': es_id, 'sentence_id': sentence_id, 'text_type': 'official_title', 'text': official_title})
    #         sentence_id += 1
    #
    #     if 'criteria' in i.keys():
    #         if i['criteria'] != '':
    #             criteria = i['criteria']
    #
    #             criteria = Utilfuncs.remove_html_tags(criteria)
    #             criteria = ' '.join(criteria.split())
    #             criteria = criteria.split('\n')
    #
    #             for j in criteria:
    #                 texts = Prepro.cut_sentences(j.strip())
    #                 for text in texts:
    #                     if text.startswith('- '):
    #                         text = text[2:]
    #                     rows_list.append(
    #                         {'esid': es_id, 'sentence_id': sentence_id, 'text_type': 'criteria', 'text': text})
    #                     sentence_id += 1
    #
    # df = pd.DataFrame(rows_list)
    # df.to_excel('./data/drug_wct.xlsx')
    # df.to_hdf('./data/to_pre_labeled.h5', key='drug_wct')


if __name__ == '__main__':
    dt_0316()
    # df = pd.read_excel('./data/v2.1/v2_1_0316.xlsx', sheet_name='raw4')
    # s = df.loc[0,'criteria']
    # d = repr(s)
    # print(d)
    # print(s.split())

    # d = '8. Positive for Hepatitis B and C.'
    # print(Prepro().cut_sentences(d))
    # df = pd.read_excel("/home/zyl/PharmAI_test/pharm_ai/word/word项目数据.xlsx")
    # print(len(set(df['NCT'].to_list())))
