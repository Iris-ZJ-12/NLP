# encoding: utf-8
'''
@author: zyl
@file: dt.py
@time: 2021/10/21 11:41
@desc:
'''
import pandas as pd


class DT:
    def __init__(self):
        pass

    def run(self):
        # self.analyze_dt()
        # self.get_dt_1021()
        # self.get_dt_1025()
        self.get_pdf()
        # self.deal_with_1021()
        pass

    def get_dt_1021(self):
        from pharm_ai.util.ESUtils7 import get_page,get_count,Query,QueryType
        q = Query(QueryType.EQ, 'esid', 'd4b987766f263c9205d34ebafe59e7d0')
        s = get_page('gov_purchase', queries=q,page_size=-1)
        print(s)
        # df = pd.DataFrame(s)
        # df.to_json('./data/all_dt_1021.json.gz', orient='records', compression='gzip')
        # a= get_count('gov_purchase')
        # print(a)



    def get_dt_1025(self):
        from pharm_ai.util.ESUtils7 import get_page,get_count
        s = get_page('bidding_announcement', page_size=-1)
        print(len(s))
        # df = pd.DataFrame(s)
        # df.to_excel('./data/to_label/test_bidding_announcement.xlsx')
        # a= get_count('gov_purchase')
        # print(a)

    def deal_with_1021(self):
        df = pd.read_json('./data/all_dt_1021.json.gz', orient='records', compression='gzip')
        df.to_excel("./data/test.xlsx")
        df1 = df[df['esid']=="036cc693f0eadb97792d7f691c563a04"]
        print(df1)
        df2 = df[df['esid']=="04a2c9cd26d94a8afadeef9b7c4e1ddb"]
        print(df2)

    def analyze_dt(self):
        df = pd.read_json('./data/all_dt_1021.json.gz', orient='records', compression='gzip')[0:10]
        df.to_excel('./data/to_label/test.xlsx')

    def get_pdf(self):
        pdf_url = "http://spider.pharmcube.com/fb877021bd0b4a56533f37f81567d140.pdf"
        import requests
        pdf_response = requests.get(pdf_url)
        print(pdf_response)
        with open(f"./data/to_label/test_pdf.pdf", "wb") as code:
            code.write(pdf_response.content)


if __name__ == '__main__':

    DT().run()