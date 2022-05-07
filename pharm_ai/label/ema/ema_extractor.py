# -*- coding: UTF-8 -*-
"""
Description : extract field-text from European pdfs
"""
import ntpath
import random
import re

import pandas as pd
from loguru import logger
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from tqdm import tqdm

from pharm_ai.util.pdf_util.extractor_v3 import PDFTextExtractor, FieldTextExtractor
from pharm_ai.util.utils import Utilfuncs
import itertools


class EMAExtractor:
    def __init__(self):
        self.fields_re = {
            "NAME OF THE MEDICINAL PRODUCT": [
                r'1.\s*NAME OF THE MEDICINAL PRODUCT((?:.|\n)*?)2.\s*QUALITATIVE AND QUANTITATIVE COMPOSITION'],
            "QUALITATIVE AND QUANTITATIVE COMPOSITION": [
                '2.\s*QUALITATIVE AND QUANTITATIVE COMPOSITION((?:.|\n)*?)3.\s*PHARMACEUTICAL FORM'],
            "PHARMACEUTICAL FORM": [r'3.\s*PHARMACEUTICAL FORM((?:.|\n)*?)4.\s*CLINICAL PARTICULARS'],
            "Therapeutic indications": [
                r'4.1\s*Therapeutic indications((?:.|\n)*?)4.2\s*Posology and method of administration',
                r'4.1\s*Therapeutic indications((?:.|\n)*?)4.2\s*al\n\s*Posology and method of administration',
            ],
            "Posology and method of administration": [
                r'4.2\s*Posology and method of administration((?:.|\n)*?)4.3\s*Contraindications',
                r'4.2\s*Posology and method of administration((?:.|\n)*?)4.3\s*al\n\s*Contraindications',
            ],
            "Shelf life": [r'6.3\s*Shelf[\s-]life((?:.|\n)*?)6.4\s*Special precautions for storage'],
            "Special precautions for storage": [
                r'6.4\s*Special precautions for storage((?:.|\n)*?)6.5\s*Nature and contents of container'],
        }

    def test(self):
        # pdf = "./data/pdf/spider_24c9090d2978268f4f472b8e2cc1b4b1.pdf"
        # raw_text = PDFTextExtractor.extract_raw_text_from_pdf(pdf)
        # print(repr(raw_text))
        # if 'ANNEX II' in raw_text:
        #     raw_text = raw_text.split('ANNEX II')[0]
        # raw_text = EMAExtractor.handle_raw_text(raw_text)
        # res = self.extract_field_text_for_ema(raw_text)
        # print(res)
        #
        # df = pd.DataFrame(res)
        # df['pdf'] = pdf
        #
        # df.to_excel('./data/test.xlsx')
        self.refine_and_save()

    def refine_and_save(self):

        # f = "/home/zyl/disk/PharmAI/pharm_ai/label/ema/data/pdf/"
        # t = self.extract_texts_from_pdfs_dir(f)
        # df = pd.DataFrame(t)
        # df.to_excel('./data/ema_raw.xlsx')

        df = pd.read_excel('./data/ema_raw.xlsx')  # type:pd.DataFrame
        df = df.astype('str')
        df['NAME OF THE MEDICINAL PRODUCT'] = df['NAME OF THE MEDICINAL PRODUCT'].apply(
            lambda x: EMAExtractor.handle_field_text(str(x)))
        df['QUALITATIVE AND QUANTITATIVE COMPOSITION'] = df['QUALITATIVE AND QUANTITATIVE COMPOSITION'].apply(
            lambda x: EMAExtractor.handle_field_text(str(x)))
        df['PHARMACEUTICAL FORM'] = df['PHARMACEUTICAL FORM'].apply(
            lambda x: EMAExtractor.handle_field_text(str(x)))
        df['Therapeutic indications'] = df['Therapeutic indications'].apply(
            lambda x: EMAExtractor.handle_field_text(str(x)))
        df['Posology and method of administration'] = df['Posology and method of administration'].apply(
            lambda x: EMAExtractor.handle_field_text(str(x)))
        df['Shelf life'] = df['Shelf life'].apply(
            lambda x: EMAExtractor.handle_field_text(str(x)))
        df['Special precautions for storage'] = df['Special precautions for storage'].apply(
            lambda x: EMAExtractor.handle_field_text(str(x)))
        df['label'] = ''
        df.to_excel('./data/test_ema.xlsx')

    def extract_texts_from_pdfs_dir(self, pdfs_dir):
        pdf_paths = Utilfuncs.list_all_files(pdfs_dir, 'pdf')  # list all pdf files
        # pdf_paths = random.sample(pdf_paths, 20)
        logger.info('pdfs num: ' + str(len(pdf_paths)))
        fields_texts = []
        for pdf in tqdm(pdf_paths):
            raw_text = PDFTextExtractor.extract_raw_text_from_pdf(pdf)
            if 'ANNEX II' in raw_text:
                raw_text = raw_text.split('ANNEX II')[0]
            raw_text = EMAExtractor.handle_raw_text(raw_text)
            fields_text = self.extract_field_text_for_ema(raw_text)  # type:list[dict]
            pdf_name = ntpath.basename(pdf)  # pdf file name
            for i in fields_text:
                i.update({'id': pdf_name})
                fields_texts.append(i)

        return fields_texts  # type:list[dict]

    def extract_field_text_for_ema(self, raw_text):
        r = {}
        for field in self.fields_re:
            try:
                r[field] = FieldTextExtractor.extract_field_text_by_re(raw_text, fields_re=self.fields_re[field])
            except Exception:
                r[field] = []

        keys = list(r.keys())
        res = []
        for f0, f1, f2, f3, f4, f5, f6 in itertools.zip_longest(*r.values()):
            res.append(
                {
                    keys[0]: f0, keys[1]: f1, keys[2]: f2, keys[3]: f3, keys[4]: f4, keys[5]: f5, keys[6]: f6,
                }
            )
        return res

    @staticmethod
    def handle_raw_text(text: str):
        text = EMAExtractor.remove_page_header_footer(text)
        text = ILLEGAL_CHARACTERS_RE.sub(r'', str(text))
        text = EMAExtractor.change_number_font(text)
        if '  ' in text:
            text = re.sub(' +', ' ', text)
        return text

    @staticmethod
    def remove_page_header_footer(text, page_header_footer_pattern=r'\n\s*\d{1,3}\s*\n'):
        text = re.sub(page_header_footer_pattern, ' \n ', text)
        return text

    @staticmethod
    def change_number_font(num_chrs: str):
        for ch in num_chrs:
            if 65296 <= ord(ch) <= 65305:
                num_chrs = num_chrs.replace(ch, chr(ord(ch) - 65248))
        return num_chrs

    @staticmethod
    def handle_field_text(text):
        text = text.strip()
        char_list = ['\xad', '\uf0b7', '\uf8e7', '\xa0', '\ue05d', '\uf0b0']  # '\n', '\t',
        for c in char_list:
            if c in text:
                text = text.replace(c, ' ')
        text = re.sub(' +', ' ', text)

        while re.findall(r'^[\s:\n\t]', text) != []:
            text = text.strip()
            text = text.strip(':')
            text = text.strip('\n')
            text = text.strip('\t')
        return text



class TestModel():
    def __init__(self):
        pass

    def test(self):
        from pprint import pprint
        from sklearn.metrics import classification_report
        from sklearn.utils import resample
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/label/ema/data/ema-200 校验.xlsx")

        # extract success rate
        dict = {
            "num": [200, 200],
            '| NAME OF MEDICINAL PRODUCT': [1 - 0 / 200,1 - 2 / (200 - 0)],
            '| QUALITATIVE AND QUANTITATIVE COMPOSITION': [1 - 0 / 200,1 - 1 / (200 - 0)],
            '| PHARMACEUTICAL FORM': [1 - 0 / 200, 1 - 1 / (200 - 0)],
            '| Therapeutic Indications': [1 - 0 / 200, 1 - 1 / (200 - 0)],
            '| Posology And Method Of Administration': [1 - 0 / 200, 1 - 1 / (200 - 0)],
            '| Shelf Life': [1 - 0 / 200, 1 - 2 / (200 - 0)],
            '| Special precautions for storage': [1 - 0 / 200, 1 - 4 / (200 - 0)],
        }
        indexs = ['successfully extracted', 'correctly extracted']
        cols = ['num', '| NAME OF MEDICINAL PRODUCT', '| QUALITATIVE AND QUANTITATIVE COMPOSITION',
                '| PHARMACEUTICAL FORM','| Therapeutic Indications','| Posology And Method Of Administration',
                '| Shelf Life','| Special precautions for storage',
            ]
        t_df = pd.DataFrame(dict, index=indexs, columns=cols)
        pd.set_option('precision', 4)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 60)
        pd.set_option('display.max_columns', None)
        pd.set_option('mode.chained_assignment', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)  # 将模糊字符宽度设置为2
        pd.set_option('display.unicode.east_asian_width', True)  # 检查东亚字符宽度属性

        pd.set_option("colheader_justify", "center")
        print('field extracted report:')
        pprint(t_df)



if __name__ == '__main__':
    # EMAExtractor().test()
    TestModel().test()