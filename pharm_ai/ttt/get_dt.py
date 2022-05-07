# encoding: utf-8
'''
@author: zyl
@file: get_dt.py
@time: 2021/7/26 下午11:16
@desc:
'''
import pandas as pd
from tqdm import tqdm

from pharm_ai.util.ESUtils7 import Query, QueryType, get_page,get_count


class GetDT:
    def __init__(self):
        pass

    def run(self):
        # self.dt_1109_1()
        self.dt_1109_2()
        # self.test()

    def test_index(self,index):
        print(f"count:{get_count(index)}")
        res_1 = get_page(index, page_size=500)
        df = pd.DataFrame(res_1)
        df.to_excel(f'./data/test_{index}.xlsx')

    def test(self):
        self.test_index('clinical_indication_expansion')
        self.test_index('drug_wct')

    def get_nct_ids(self):
        # 先从clinical_indication_expansion这个索引去除nct_ids
        fields_1 = ['esid', 'therapy_labels', 'nct_ids', 'therapy_labels_en']
        res_1 = get_page("clinical_indication_expansion", show_fields=fields_1, page_size=-1)
        res_df_1 = pd.DataFrame(res_1)
        res_df_1 = res_df_1.explode('nct_ids')
        res_df_1.rename(columns={'nct_ids': 'nct_id', 'esid': 'esid_cie'}, inplace=True)
        res_df_1.dropna(inplace=True)
        return res_df_1

    def get_dt_from_nct_ids(self,nct_ids):
        # 根据nct_id从drug_wct里面取数据
        fields_2 = ['esid', 'criteria', 'study_title', 'nct_id', 'state',
                    'manual_check']
        li_2 = []
        for i in tqdm(range(len(nct_ids))):
            q_2 = Query(QueryType.EQ, "nct_id", nct_ids[i])
            r = get_page("drug_wct", queries=q_2, show_fields=fields_2)

            if r:
                if len(r) > 1:
                    print(i)
                    print(len(r))
                li_2.append(r[0])
        res_df_2 = pd.DataFrame(li_2)
        return res_df_2


    def dt_1109_1(self):
        res_df_1 = self.get_nct_ids()
        nct_ids = res_df_1['nct_id'].tolist()
        print(len(nct_ids))
        print(len(res_df_1))

        res_df_2 = self.get_dt_from_nct_ids(nct_ids)
        print(len(res_df_2))
        res_df = pd.merge(res_df_1, res_df_2, on='nct_id', how='inner')
        print(len(res_df))
        res_df.to_excel('./data/v1/raw_1109.xlsx')

    def dt_1109_2(self):
        res_df_1 = pd.read_excel("/home/zyl/PharmAI_test/pharm_ai/ttt/data/v1/ClinicalTrials.xls")[0:2350]
        res_df_1.rename(columns={"NCT": "nct_id"}, inplace=True)
        print(len(res_df_1))
        nct_ids = res_df_1['nct_id'].tolist()

        res_df_2 = self.get_dt_from_nct_ids(nct_ids)
        print(len(res_df_2))
        res_df = pd.merge(res_df_1, res_df_2, on='nct_id', how='inner')
        print(len(res_df))
        res_df.to_excel('./data/v1/raw_1109_2.xlsx')

if __name__ == '__main__':
    GetDT().run()
