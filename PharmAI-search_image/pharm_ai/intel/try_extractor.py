# encoding: utf-8
'''
@author: zyl
@file: try_extractor.py
@time: 2021/9/22 9:32
@desc:
'''

import ntpath
import re
import time

import pandas as pd
from loguru import logger
from tqdm import tqdm

from pharm_ai.intel.get_text import get_ocr_result
from pharm_ai.util.pdf_util.extractor_v3 import PDFTextExtractor, FieldTextExtractor, PDFTableExtractor
from pharm_ai.util.utils import Utilfuncs
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# # Use in Transformers
# tokenizer = AutoTokenizer.from_pretrained("/large_files/5T/zyl_tmp_model/chinese_pretrain_mrc_roberta_wwm_ext_large/")
# model = AutoModelForQuestionAnswering.from_pretrained(
#     "/large_files/5T/zyl_tmp_model/chinese_pretrain_mrc_roberta_wwm_ext_large/")
from transformers import pipeline
#
# nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)


class IntelExtractor:
    def __init__(self):
        self.fields_re = {
            'project_name': [
                r'项目名称[：:]((?:.|\n)*?)\n',
                r'中标人信息[：:]\n((?:.|\n)*?)\n',
                r'一、中标人信息\n((?:.|\n)*?)\n中标人',
                r'\n((?:.|\n)*?)（项目名称）',
            ],

            'tenderee': [
                r'招\s*\n*\s*标\s*\n*\s*人[：:]((?:.|\n)*?)\n',
                r'招标人名称[：:]((?:.|\n)*?)\n',
                r'采购人信息((?:.|\n)*?)采购代理机构信息',
                r'采购单位信息((?:.|\n)*?)代理机构信息',
                r'招标人（代建单位）：((?:.|\n)*?)\n'
            ],
            'product_name': [
                r'产品名称：((?:.|\n)*?)\n',
                r'招标范围：((?:.|\n)*?)\n',
                r'货物名称：((?:.|\n)*?)\n',
                r'采购内容：((?:.|\n)*?)\n',
                r'服务名称：((?:.|\n)*?)\n',
                r'工程名称：((?:.|\n)*?)\n',
                r'目类名称：((?:.|\n)*?)\n',
                r'品目名称：((?:.|\n)*?)\n',
            ],
            'bid_amount': [
                r'中标金额[：:]((?:.|\n)*?)\n',
                r'中标价格[：:]((?:.|\n)*?)\n',
                r'中标（成交）金额[：:]((?:.|\n)*?)\n',
                r'中标价[：:]((?:.|\n)*?)\n',
                r'成交金额((?:.|\n)*?)\n',
                r'总报价[：:]((?:.|\n)*?)\n',
                r'中标金额[：:]\n*((?:.|\n)*?)\n'
            ],

            'company_name': [
                r'中标人: ((?:.|\n)*?)中标价格',
                r'中标人：((?:.|\n)*?)\n',
                r'供应商名称：((?:.|\n)*?)\n',
                r'供应商((?:.|\n)*?)成交金额',
                r'中标人 ((?:.|\n)*?)\n',
                r'中标人：\n((?:.|\n)*?)\n',
                r'中标单位：\n((?:.|\n)*?)\n'

            ]
        }

    def run(self):
        # dir = "/home/zyl/disk/PharmAI/pharm_ai/intel/data/dt_20210922/2021-09-22招标pdf/"
        # dt = self.extract_texts_from_pdf_dir(dir)
        # df = pd.DataFrame(dt)
        # df.to_excel('./data/test.xlsx')
        self.test()

    def test(self):
        # pdf = "/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/pdfs/982432712c99c12b5401089e8a479cf4.pdf"
        pdf = "/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/pdfs/79eb16cb79362a7f19e9a07c4f1c1277.pdf"
        pdf = "/home/zyl/disk/PharmAI/pharm_ai/intel/data/to_label/pdfs/4b80543d0d47bd1134817dd38700f222.pdf"
        print(self.extract_fields_text_from_pdf(pdf))

    def use_mrc(self, pdf):
        tokenizer = AutoTokenizer.from_pretrained(
            "/large_files/5T/zyl_tmp_model/chinese_pretrain_mrc_roberta_wwm_ext_large/")
        model = AutoModelForQuestionAnswering.from_pretrained(
            "/large_files/5T/zyl_tmp_model/chinese_pretrain_mrc_roberta_wwm_ext_large/")
        from transformers import pipeline

        nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

        raw_text = PDFTextExtractor.extract_raw_text_from_pdf(pdf)
        if '扫描全能王' in raw_text:
            raw_text = get_ocr_result(pdf)
        raw_text = self.handle_raw_text(raw_text)
        return {
            'project_name_model': nlp(question="项目名称是什么?", context=raw_text).get('answer'),
            'tenderee_model': nlp(question="招标人或采购人是谁?", context=raw_text).get('answer'),
            'product_name_model': nlp(question="产品名称是什么?", context=raw_text).get('answer'),
            'bid_amount_model': nlp(question="中标价格或金额是多少?", context=raw_text).get('answer'),
            'company_name_model': nlp(question="中标人是谁?", context=raw_text).get('answer'),
        }

    def extract_texts_from_pdf_dir(self, pdfs_dir):
        pdf_paths = Utilfuncs.list_all_files(pdfs_dir, 'pdf') # list all pdf files
        logger.info('pdfs num: ' + str(len(pdf_paths)))
        fields_texts = []
        for pdf in tqdm(pdf_paths):
            field_text_dict = self.extract_fields_text_from_pdf(pdf)

            # model_dict = self.use_mrc(pdf)

            pdf_name = ntpath.basename(pdf)  # pdf file name
            field_text_dict.update({'pdf_name': pdf_name})
            # field_text_dict.update(model_dict)
            fields_texts.append(field_text_dict)
        return fields_texts  # type:dict

    def extract_fields_text_from_pdf(self, pdf):
        # print(pdf)
        raw_text = PDFTextExtractor.extract_raw_text_from_pdf(pdf,method='pdftotext')
        if '扫描全能王' in raw_text:
            can_be_read = 'wrong'
            raw_text = get_ocr_result(pdf)
        else:
            can_be_read = 'right'
        print(raw_text)
        raw_raw_text = raw_text
        raw_text = self.handle_raw_text(raw_text)

        # 项目名称
        project_name = FieldTextExtractor.extract_field_text_by_re(raw_text, self.fields_re['project_name'])
        # print(project_name)
        # if project_name:
        #     project_name = [p_n.split('\n')[-1] if '\n' in p_n else '' for p_n in project_name]

        # 招标人
        tenderee = FieldTextExtractor.extract_field_text_by_re(raw_text, self.fields_re['tenderee'])
        if tenderee:
            if '名称：' in tenderee[0]:
                tenderee = FieldTextExtractor.extract_field_text_by_re(tenderee[0], [r'名称：((?:.|\n)*?)\n'])
            if '名 称：' in tenderee[0]:
                tenderee = FieldTextExtractor.extract_field_text_by_re(tenderee[0], [r'名 称：((?:.|\n)*?)\n'])

        # 产品名称
        product_name = FieldTextExtractor.extract_field_text_by_re(raw_text, self.fields_re['product_name'])

        # 金额
        bid_amount = FieldTextExtractor.extract_field_text_by_re(raw_text, self.fields_re['bid_amount'])

        # 中标人
        company_name = FieldTextExtractor.extract_field_text_by_re(raw_text, self.fields_re['company_name'])

        tb = PDFTableExtractor.extract_raw_tables_from_pdf(pdf, return_format='raw')
        tb = PDFTableExtractor.get_tables_dict2(tb)
        if len(tb) <= 2:
            tb = PDFTableExtractor.extract_raw_tables_from_pdf(pdf)

        if tb:
            project_name_re2 = ['项目名称']
            for p_n_r in project_name_re2:
                if not project_name:
                    project_name = FieldTextExtractor.extract_field_text_from_table(field_key_re=p_n_r, tables_dict=tb)

            tenderee_re2 = ['招标人', ]
            for t_r in tenderee_re2:
                if not tenderee:
                    tenderee = FieldTextExtractor.extract_field_text_from_table(field_key_re=t_r, tables_dict=tb)

            product_name_re2 = ['产品名称', '工程地点']
            for p_n_r in product_name_re2:
                if not product_name:
                    product_name = FieldTextExtractor.extract_field_text_from_table(field_key_re=p_n_r, tables_dict=tb)

            bid_amount_re2 = ['中标价（万元|%）', '中标价(元)']
            for b_a_r in bid_amount_re2:
                if not bid_amount:
                    bid_amount = FieldTextExtractor.extract_field_text_from_table(field_key_re=b_a_r, tables_dict=tb)

            company_name_re2 = ['中标人', '中标单位名称']
            for c_n_r in company_name_re2:
                if not company_name:
                    company_name = FieldTextExtractor.extract_field_text_from_table(field_key_re=c_n_r, tables_dict=tb)

        # print(repr(raw_text))
        # print(tb)
        # print(f'项目名称: {project_name}')
        # print(f'招标人: {tenderee}')
        # print(f'产品名称: {product_name}')
        # print(f'金额: {bid_amount}')
        # print(f'中标人: {company_name}')

        return {'can_be_read': can_be_read,
                'raw_text': raw_raw_text,
                'project_name_rule': project_name,
                'tenderee_rule': tenderee,
                'product_name_rule': product_name,
                'bid_amount_rule': bid_amount,
                'company_name_rule': company_name,
                }  # dict,{field_name:field_text}

    def handle_raw_text(self, raw_text: str):
        text = re.sub(('  +'), '', raw_text)
        return text

    @staticmethod
    def handle_field_text(field_text: str):
        # ...
        text = field_text
        return text


if __name__ == '__main__':
    IntelExtractor().run()
