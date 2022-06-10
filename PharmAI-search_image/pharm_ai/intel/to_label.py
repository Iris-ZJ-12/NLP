# encoding: utf-8
'''
@author: zyl
@file: to_label.py
@time: 2021/10/25 10:46
@desc: 准备要标注的数据
'''
import pandas as pd
import requests
from tqdm import tqdm
import re
from pharm_ai.util.pdf_util.extractor_v3 import PDFTextExtractor, FieldTextExtractor, PDFTableExtractor
import ntpath
import re
from itertools import chain

import fitz
import pdfplumber
import pdftotext
from loguru import logger
from tqdm import tqdm

from pharm_ai.util.utils import Utilfuncs
from pharm_ai.intel.get_text import get_ocr_result
import datetime


class TOLabel:
    def __init__(self):
        pass

    def run(self):
        # self.analyze_dt()
        # self.get_dt_1021()
        # self.get_dt_1025()
        # self.get_pdf()
        # self.deal_with_1021()
        # self.get_pdfs()
        # d = self.predict_pdfs()
        # df = pd.DataFrame(d)
        # df.to_json('./data/all_dt_1021.json.bz2', orient='records', force_ascii=False, compression='bz2')

        # df.to_excel("/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/test_pdf.xlsx")
        self.dt()
        pass

    def get_dt_1021(self):
        from pharm_ai.util.ESUtils7 import get_page, Query, QueryType
        q = Query(QueryType.EQ, 'esid', 'd4b987766f263c9205d34ebafe59e7d0')
        s = get_page('gov_purchase', queries=q, page_size=-1)  # 这两个index不相关
        print(s)
        # df = pd.DataFrame(s)
        # df.to_json('./data/all_dt_1021.json.gz', orient='records', compression='gzip')
        # a= get_count('gov_purchase')
        # print(a)

    def get_dt_1025(self):
        from pharm_ai.util.ESUtils7 import get_page
        s = get_page('bidding_announcement', show_fields=('title', 'esid', 'pdf_url'), page_size=-1)
        print(len(s))
        df = pd.DataFrame(s)
        df.to_json('./data/to_label/dt_1025.json.bz2', orient='records', force_ascii=False,
                   compression='bz2')

    def deal_with_1021(self):
        df = pd.read_json('./data/all_dt_1021.json.gz', orient='records', compression='gzip')
        df.to_excel("./data/test.xlsx")
        df1 = df[df['esid'] == "036cc693f0eadb97792d7f691c563a04"]
        print(df1)
        df2 = df[df['esid'] == "04a2c9cd26d94a8afadeef9b7c4e1ddb"]
        print(df2)

    def analyze_dt(self):
        df = pd.read_json('./data/all_dt_1021.json.gz', orient='records', compression='gzip')[0:10]
        df.to_excel('./data/to_label/test.xlsx')

    @staticmethod
    def write_pdf(pdf_url: str):
        pdf_name = pdf_url.split('.pdf')[0].split('/')[-1]
        pdf_response = requests.get(pdf_url)
        with open(f"./data/to_label/pdfs/{pdf_name}.pdf", "wb") as f:
            f.write(pdf_response.content)

    def get_pdfs(self):
        df = pd.read_json('./data/to_label/dt_1025.json.bz2', orient='records',
                          compression='bz2')
        all_pdf_urls = df['pdf_url'].to_list()
        for p in tqdm(all_pdf_urls[0:1000]):
            TOLabel.write_pdf(p)

    @staticmethod
    def get_pdf_text(pdf):
        raw_text = PDFTextExtractor.extract_raw_text_from_pdf(pdf, method='pdftotext')
        if '扫描全能王' in raw_text:
            can_be_read = 'wrong'
            raw_text = get_ocr_result(pdf)
        else:
            can_be_read = 'right'

        # raw_text = re.sub(' +', ' ', raw_text)
        return raw_text

    @staticmethod
    def get_table_from_pdf(pdf):
        # import tabula
        # df = tabula.read_pdf(pdf, pages='all',multiple_tables = True)
        # print(len(df))
        # tabula.convert_into(pdf, "./data/to_label/test_pdf.csv")

        tb = PDFTableExtractor.extract_raw_tables_from_pdf(pdf, return_format='df')
        print(tb)
        # tb = PDFTableExtractor.get_tables_dict2(tb)
        # print(tb)

    def predict_pdfs(self, pdfs_dir="/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/pdfs/"):
        from pharm_ai.intel.try_extractor import IntelExtractor
        fields_texts = IntelExtractor().extract_texts_from_pdf_dir(pdfs_dir)
        return fields_texts  # type:dict

    def dt(self):
        import time
        # df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/test_pdf.xlsx")
        df = pd.read_json('./data/all_dt_1021.json.bz2', orient='records', compression='bz2')

        df = df[['raw_text', 'pdf_name', 'company_name_rule', 'project_name_rule', 'tenderee_rule',
                 'product_name_rule', 'bid_amount_rule']]

        df['company_name_rule'] = df['company_name_rule'].apply(lambda x: '' if not eval(str(x)) else '|'.join(eval(str(x))))
        df['project_name_rule'] = df['project_name_rule'].apply(lambda x: '' if not eval(str(x)) else '|'.join(eval(str(x))))
        df['tenderee_rule'] = df['tenderee_rule'].apply(lambda x: '' if not eval(str(x)) else '|'.join(eval(str(x))))
        df['product_name_rule'] = df['product_name_rule'].apply(lambda x: '' if not eval(str(x)) else '|'.join(eval(str(x))))
        df['bid_amount_rule'] = df['bid_amount_rule'].apply(lambda x: '' if not eval(str(x)) else '|'.join(eval(str(x))))

        from pharm_ai.intel.prepare_labeled_dt import Labeler
        a = df.to_dict(orient='records')
        print(len(a))

        map_dict = {'company_name_rule': '企业名称', 'project_name_rule': '项目名称', 'tenderee_rule': '招标人',
                    'product_name_rule': '产品名称', 'bid_amount_rule': '中标金额'}

        res2 = []
        predictions = []
        id = 0

        for i in a:
            id += 1
            text = i.get('raw_text')
            # print(text)
            if not text:
                text = ''
            data = {'ner': text, 'pdf_name': i.get('pdf_name'), "id": id}

            predictions_res = []

            for k in list(map_dict.keys()):
                s = i.get(k)
                if s == '':
                    continue
                else:
                    s = s.split('|')
                    while '' in s:
                        s.remove('')
                    for _ in s:
                        try:
                            res = Labeler.find_location(_, text)
                            # print(res)
                            for r in res:
                                predictions_res.append(
                                    {
                                        'from_name': 'label',
                                        'to_name': 'text',
                                        'type': 'labels',
                                        'value': {
                                            'start': r.get('start'),
                                            'end': r.get('end'),
                                            'text': _,
                                            'labels': [map_dict.get(k)]}}
                                )
                        except Exception:
                            pass

            completed_by = {
                               "email": "1137379695@qq.com",
                               "first_name": "Z",
                               "last_name": "yl"
                           },
            predictions.append({'model_version': 'v0', 'result': predictions_res, "task": int(id)+len(a),
                                "completed_by": completed_by,
                                "created_at": str(datetime.datetime.now()),
                                "updated_at": str(datetime.datetime.now()),
                                "id": int(id)+len(a),
                                })
            #     predictions = a['company_name_rule']
            res2.append({'data': data, 'annotations': predictions,'id':id})
            predictions = []
        # print(res2)

        # print(a)
        # predictions = [{'model_version': model_name, 'result': spans}]
        # a = [{'data': i,'predictions':predictions} for i in a]
        import json
        with open("./data/to_label/to_label_1028.json", 'w') as t:
            json.dump(res2, t)


if __name__ == '__main__':
    TOLabel().run()
    # pdf = "/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/pdfs/a6cc17f52ca2b48c7b333904503306ae.pdf"
    # pdf = "/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/pdfs/982432712c99c12b5401089e8a479cf4.pdf"
    # TOLabel.get_pdf_text(pdf)
    # TOLabel.get_table_from_pdf(
    #     "/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/pdfs/b64d94e916b7794b878f185a9998a920.pdf")
    pass
