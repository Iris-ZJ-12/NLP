# encoding: utf-8
'''
@author: zyl
@file: upload_uniprot_db.py
@time: 2021/9/7 10:35
@desc:
'''
import gc
import os
import time

import pandas as pd

from pharm_ai.util.ESUtils7 import get_page, bulk, OpType, RefreshPolicy


class UploadDT:
    def __init__(self):
        # self.test_es_host = '172.17.108.112'
        # self.test_es_host = 'test112.cubees.com'
        self.test_es_host = '39.96.48.206'
        self.test_es_port = '9200'
        self.index = 'drug_patent_uniportkb'
        self.test_file = './data/test.xlsx'

    def run(self):
        self.up_load_0916()
        # self.test()

    def test(self):
        s = get_page(self.index, page_size=20)
        print(s)
        print(len(s))

    @staticmethod
    def upload_data(file):
        df = pd.read_csv(file, compression='gzip')

        df = df[['esid', 'accession', 'accession_secondary', 'dataset', 'dataset_create_time', 'dataset_modify_time',
                 'organism', 'gene', 'protein_name_recom', 'protein_name_recom_short', 'protein_name_alter',
                 'protein_name_alter_short', 'protein_name_alter_cd', 'protein_name_submit', 'evidence_level',
                 'all_name']]

        df['protein_name_recom_short'] = df['protein_name_recom_short'].apply(lambda x: eval(x))
        df['organism'] = df['organism'].apply(lambda x: eval(x))
        df['gene'] = df['gene'].apply(lambda x: eval(x))
        df['protein_name_alter_cd'] = df['protein_name_alter_cd'].apply(lambda x: eval(x))
        df['accession_secondary'] = df['accession_secondary'].apply(lambda x: eval(x))
        df['protein_name_submit'] = df['protein_name_submit'].apply(lambda x: eval(x))
        df['protein_name_alter'] = df['protein_name_alter'].apply(lambda x: eval(x))
        df['protein_name_alter_short'] = df['protein_name_alter_short'].apply(lambda x: eval(x))
        df['all_name'] = df['all_name'].apply(lambda x: eval(x))
        df['protein_name_recom'] = df['protein_name_recom'].apply(lambda x: None if pd.isna(x) else x)

        t = int(round(time.time() * 1000))
        df['create_time'] = t
        df['modify_time'] = t

        data = list(df.to_dict(orient='records'))
        df.drop(index=df.index, inplace=True)
        del df
        gc.collect()
        print('start upload...')
        t1 = time.time()
        bulk(index='drug_patent_uniportkb', op_type=OpType.UPDATE, coll=data, refresh=RefreshPolicy.NONE,
             upsert=True)
        t2 = time.time()

        print(f'spend time:{t2 - t1}')
        data.clear()
        del data
        gc.collect()

    def up_load_0916(self):
        dt_dir = "/home/zyl/disk/Pharm_AI/trembl_dt/"
        all_files = os.listdir(dt_dir)
        for f in all_files:
            print(f'start : {f}')
            UploadDT.upload_data(dt_dir + f)
            print(f'done : {f}')


if __name__ == '__main__':
    UploadDT().run()
