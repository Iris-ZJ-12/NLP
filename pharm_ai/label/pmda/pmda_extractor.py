# -*- coding: UTF-8 -*-
"""
Description : extract field-text from Japanese pdfs
"""
import ntpath
import random
import re
from itertools import chain

import pandas as pd
from loguru import logger
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from tqdm import tqdm
from pharm_ai.label.pmda.predict import Predictor, PMDADT
from pharm_ai.util.pdf_util.extractor_v3 import PDFTextExtractor, FieldTextExtractor, PDFTableExtractor, Extractor
from pharm_ai.util.utils import Utilfuncs
from pharm_ai.label.old.JPN.if_fields_extractor.pdf_extractor import PdfExtractor

old_e = PdfExtractor()


class PMDAExtractor:
    def __init__(self):
        self.fields_table_re = {
            "剤形": r"剤形",
            "規格・含量": r"規格・含量",
            "用法及び用量": r'用法及び用量',
            "小児": r'(?:\d{1,2}[．.]|)小児(?:等への投与|への投与)',
            '年月日': r'製造販売承認年月日薬価基準収載・発売年月日',
        }
        self.field_re = {
            "剤形": [r"\n剤\n*[形型]((?:.|\n)*?)\n製\n*剤\n*の\n*規",
                   r"\n剤\n*[形型]((?:.|\n)*?)\n規\n*格",

                   ],
            "規格・含量": [r"\n規\n*格\n*[・･]\n*含\n*量((?:.|\n)*?)\n*一\n*般\n*名",
                      r"\n規\n*格\n*含\n*量((?:.|\n)*?)\n*一\n*般\n*名",
                      ],
            "用法及び用量": [
                r'\n\d{1,2}\n*[．.]\n*(?:用法及び用量・用途|法及び用量·|用法及び用量|用法・用量|用法および用量)\n((?:.|\n)*?)'
                r'\n\d{1,2}\n*[．.]\n*(?:臨床成績|臨床)',
                r'\n\d{1,2}\n*[．.]\n*(?:用法及び用量・用途|法及び用量·|用法及び用量|用法・用量|用法および用量)((?:.|\n)*?)'
                r'\n\d{1,2}\n*[．.]\n*(?:臨床成績|臨床)',
                r'\d{1,2}\n*[．.]\n*(?:用法及び用量・用途|用法及び用量|用法及び用量|用法・用量|用法および用量)\n((?:.|\n)*?)'
                r'\d{1,2}\n*[．.]\n*(?:臨床成績|臨床)',
            ],
            "小児": [
                r'\n\d{1,2}\n*[．.]\n*小児(?:等への投与·|等への投与|等への使用|等)\n((?:.|\n)*?)'
                r'\n\d{1,2}\n*[．.]\n*(?:臨床検査結果に|高齢者|臨床試験)',
                r'\d{1,2}\n*[．.]\n*小児(?:等への投与·|等への投与|等への使用|等)\n((?:.|\n)*?)'
                r'\d{1,2}\n*[．.]\n*(?:臨床検査結果に|高齢者|臨床試験)',
                r'\d{1,2}\n*[．.]\n*小児(?:等への投与·|等への投与|等への使用|等)((?:.|\n)*?)'
                r'\d{1,2}\n*[．.]\n*(?:臨床検査結果に|高齢者|臨床試験)',
                r'\n[(（]\d{1,2}[)）]\n*小児(?:等への投与·|等への投与|等への使用|等)\n((?:.|\n)*?)'
                r'\n[(（]\d{1,2}[)）]\n*(?:高齢者|臨床検査結果に|臨床試験)',
                r'\n[(（]\d{1,2}[)）]\n*小児(?:等への投与·|等への投与|等への使用|等)((?:.|\n)*?)'
                r'\n[(（]\d{1,2}[)）]\n*(?:高齢者|臨床検査結果に|臨床試験)',
                r'\n小児等((?:.|\n)*?)\n高齢者',
            ],
            "有効期間": [
                r'\n\d{1,2}\n*[．.]\n*有効(?:期限又は使用期限|期間又は\n*使用期\n*限·|期間又は\n*使用期\n*限|期間または使用期限|期間)\n((?:.|\n)*?)'
                r'\n\d{1,2}\n*[．.]\n*(?:貯法・保存条件|包装状態での貯法)',
                r'\d{1,2}\n*[．.]\n*有効(?:期限又は使用期限|期間又は\n*使用期\n*限·|期間又は\n*使用期\n*限|期間または使用期限|期間)\n((?:.|\n)*?)'
                r'\d{1,2}\n*[．.]\n*(?:貯法・保存条件|包装状態での貯法)',
                r'\n\d{1,2}\n*[．.]\n*有効(?:期限又は使用期限|期間又は\n*使用期\n*限·|期間又は\n*使用\n*期\n*限|期間または使用\n*期限|期間)((?:.|\n)*?)'
                r'\n\d{1,2}\n*[．.]\n*(?:貯法・保存条件|包装状態での貯法)',
            ],
            "貯法": [
                r'\n\d{1,2}\n*[．.]\n*(?:包装状態での貯法|貯法・保存条件·|貯法・保存条件)\n((?:.|\n)*?)'
                r'\n\d{1,2}[．.]\n*(?:薬剤取扱い上の\n*注\n*意|薬剤取り扱い上の注意|薬剤取扱い上の|取扱い上の注意|取扱上の注意)',
                r'\d{1,2}\n*[．.]\n*(?:包装状態での貯法|貯法・保存条件·|貯法・保存条件)\n((?:.|\n)*?)'
                r'\d{1,2}[．.]\n*(?:薬剤取扱い上の\n*注\n*意|薬剤取り扱い上の注意|薬剤取扱い上の|取扱い上の注意|取扱上の注意)',
                r'\n\d{1,2}\n*[．.]\n*(?:包装状態での貯法|貯法・保存条件·|貯法・保存条件)((?:.|\n)*?)'
                r'\n\d{1,2}[．.]\n*(?:薬剤取扱い上の\n*注\n*意|薬剤取り扱い上の注意|薬剤取扱い上の|取扱い上の注意|取扱上の注意)',
            ],
            "indication": [r'\D\d[\.． ](?:効能又は効果|効能・効果)\n*([\D\n]?(?:.|\n)*?)(?:\d[\.．]\D|.*効能又は効果|.*効能・効果)', ],
        }
        self.p = Predictor()

    def test(self):
        pdf = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-001-pdf/100f4fb32fc72cf0264357416c583b9c.pdf"
        s = self.extract_field_text_for_pmda(pdf)
        print(s)
        # # ############################
        # f = "./data/pmda-003-pdf-2/"
        # to_save_sheet = "pmda-003-pdf-2"
        # self.refine_and_save(f, to_save_sheet)
        ###############################

    def run(self):
        # self.test()
        self.extract_data()
        # pdf ="/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-002-pdf-2/a4dfd6a65cfd7e4c909a8dbdcd0d3886.pdf"
        # s = self.extract_indication(pdf)
        # print(s)

        # pdfs_dir= "./data/pmda-001-pdf-2/"
        # pdf_paths = Utilfuncs.list_all_files(pdfs_dir, 'pdf')  # list all pdf files
        # pdf_paths = random.sample(pdf_paths, 20)
        # res = []
        # fs = []
        # fields_texts = []
        # for pdf in tqdm(pdf_paths):
        #     res.append(self.extract_indication(pdf))
        #     pdf_name = ntpath.basename(pdf)  # pdf file name
        #     fs.append(pdf_name)
        # d = pd.DataFrame()
        # d['pdf'] = fs
        # d['indications'] = res
        # d.to_excel('./data/test.xlsx')
        # # return fields_texts  # type:list[dict]

    def refine_and_save(self, pdf_file_dir, to_save_sheet):
        res = self.extract_texts_from_pdf_dir(pdf_file_dir, pdf_num=0)

        df = pd.DataFrame(res)
        Utilfuncs.to_excel(df, f'./data/{to_save_sheet}_1.xlsx', to_save_sheet)
        t1 = df['用法及び用量'].tolist()
        t1 = [PMDADT.handle_text(i) for i in t1]
        r1 = self.p.predict1(t1)
        df['用法及び用量_nlp'] = r1

        # t2 = df['小児'].tolist()
        # t2 = [PMDADT.handle_text(i) for i in t2]
        # r2 = self.p.predict2(t2)
        # df['小児_nlp'] = r2
        Utilfuncs.to_excel(df, f'./data/{to_save_sheet}.xlsx', to_save_sheet)

    def extract_texts_from_pdf_dir(self, pdfs_dir, pdf_num=0):
        pdf_paths = Utilfuncs.list_all_files(pdfs_dir, 'pdf')  # list all pdf files
        if pdf_num != 0:
            pdf_paths = random.sample(pdf_paths, pdf_num)
        logger.info('pdfs num: ' + str(len(pdf_paths)))
        fields_texts = []
        for pdf in tqdm(pdf_paths):
            field_text = self.extract_field_text_for_pmda(pdf)
            pdf_name = ntpath.basename(pdf)  # pdf file name
            field_text.update({'pdf name': pdf_name})
            fields_texts.append(field_text)

        return fields_texts  # type:list[dict]

    def extract_field_text_for_pmda(self, pdf):
        res = dict()
        raw_text_1 = PDFTextExtractor.extract_raw_text_from_pdf(pdf, method='pdftotext', end_page=2)
        # print(repr(raw_text_1))
        raw_text_1 = PMDAExtractor.handle_raw_text(raw_text_1)

        raw_tables_dict_1 = PDFTableExtractor.extract_raw_tables_from_pdf(pdf, end_page=2, return_format='dict')
        # print(repr(raw_text_1))
        # print(raw_tables_dict_1)
        for field in ["剤形", "規格・含量"]:
            try:
                field_text = []
                res[field] = ''
                if raw_tables_dict_1 != dict():
                    field_text = FieldTextExtractor.extract_field_text_from_table(self.fields_table_re[field],
                                                                                  tables_dict=raw_tables_dict_1)
                while None in field_text:
                    field_text.remove(None)
                while '' in field_text:
                    field_text.remove('')
                if not field_text:
                    # print('zzz')
                    field_text = FieldTextExtractor.extract_field_text_by_re(raw_text_1, fields_re=self.field_re[field])
                    # print(field_text)
                # print(field_text)

                while '' in field_text:
                    field_text.remove('')

                for i in field_text:
                    i = PMDAExtractor.handle_field_text(i)
                    if re.findall('\D', i):
                        res[field] = i
                        break
            except Exception:
                res[field] = ''
        raw_text_2 = PDFTextExtractor.extract_raw_text_from_pdf(pdf, method='pdftotext', start_page=6)
        # print(repr(raw_text_2))
        raw_text_2 = PMDAExtractor.handle_raw_text(raw_text_2)
        # print(repr(raw_text_2))
        raw_tabels_dict_2 = PDFTableExtractor.extract_raw_tables_from_pdf(pdf, start_page=6, return_format='dict')
        # print(raw_tabels_dict_2)
        # print(repr(raw_text_2))
        for field in ["用法及び用量", "小児"]:
            try:
                res[field] = ''
                field_text = FieldTextExtractor.extract_field_text_by_re(raw_text_2, fields_re=self.field_re[field])
                if not field_text:
                    field_text = FieldTextExtractor.extract_field_text_from_table(self.fields_table_re[field],
                                                                                  tables_dict=raw_tabels_dict_2)
                while '' in field_text:
                    field_text.remove('')

                for i in field_text:
                    t = PMDAExtractor.handle_field_text(i)
                    if re.findall('\D', t):
                        res[field] = t
                        break
            except Exception:
                res[field] = ''

        for field in ["有効期間", "貯法"]:
            try:
                res[field] = ''
                field_text = FieldTextExtractor.extract_field_text_by_re(raw_text_2, fields_re=self.field_re[field])
                # print(repr(field_text))
                while '' in field_text:
                    field_text.remove('')
                for i in field_text:
                    i = PMDAExtractor.handle_field_text(i)
                    if re.findall('\D', i):
                        res[field] = i
                        break

            except Exception:
                res[field] = ''
        return res  # type:dict

    @staticmethod
    def handle_raw_text(text: str):
        text = ILLEGAL_CHARACTERS_RE.sub(r'', str(text))
        text = PMDAExtractor.remove_page_header_footer(text)
        text = Extractor.change_number_font(text)
        if ' ' in text:
            text = re.sub(' ', '', text)

        double_char_list = ['\n', '·', '・', '-', '�', '…']
        for c in double_char_list:
            if c + c in text:
                text = re.sub(c + r'+', c, text)
        if '..' in text:
            text = re.sub(r'\.+', r'.', text)
        illegal_char_list = ['\xad', '\uf0b7', '\uf8e7', '\xa0', '\ue05d']
        for c in illegal_char_list:
            if c in text:
                text = re.sub(c, '', text)
        return text

    @staticmethod
    def remove_page_header_footer(fulltext, page_header_footer_pattern=r'\s*[-－]\s*\d{1,3}\s*[-－]\s*'):
        lines_raw = fulltext.split('\n')
        footers_list = [(footer_ind, line) for footer_ind, line in enumerate(lines_raw) if
                        re.fullmatch(page_header_footer_pattern, line)]
        if footers_list:
            footer_pos, footer_words = zip(*footers_list)
            first_lines_list = [(ind + 1, next_.strip()) for (ind, line), next_ in
                                zip(enumerate(lines_raw), lines_raw[1:] + [""]) if
                                ind in footer_pos and ind < len(lines_raw) - 1]
            first_lines_ind, first_lines = zip(*first_lines_list)
            common_lines = [l for l in set(first_lines) if first_lines.count(l) > 1]
            header_pos = [ind_ for ind_, l in zip(first_lines_ind, first_lines) if l in common_lines]
            lines_res = [line for ind, line in enumerate(lines_raw) if ind not in list(footer_pos) + header_pos]
            result = '\n'.join(lines_res)
        else:
            result = fulltext
        return result

    @staticmethod
    def handle_field_text(text):
        num_chars = ['Ⅹ', 'Ⅷ', 'Ⅴ']
        # print(repr(text))
        for n_c in num_chars:
            if text.endswith(n_c + '‐'):
                if len(text) > 2:
                    text = text[0:-2]

        # print(repr(text))
        if '··' in text:
            text = re.sub('·+', '·', text)
        if '--' in text:
            text = re.sub('-+', '-', text)
        if '・・' in text:
            text = re.sub('・+', '・', text)
        char_list = ['\xad', '\uf0b7', '\uf8e7', '\xa0', '\ue05d', '小児等への投与', '又は使用期限', '又は使用期', 'への投与',
                     '又は 使用期限']  # '\n', '\t',
        if '..' in text:
            text = re.sub(r'\.+', r'.', text)
        for c in char_list:
            if c in text:
                text = text.replace(c, '')
        text = re.sub(' +', ' ', text)
        while re.findall(r'^[\s:\n\t]', text) != []:
            text = text.strip()
            text = text.strip(':')
            text = text.strip('\n')
            text = text.strip('\t')

        text = ILLEGAL_CHARACTERS_RE.sub(r'', str(text))
        return text

    def extract_indication(self, pdfs_dir, to_save='test'):
        pdf_paths = Utilfuncs.list_all_files(pdfs_dir, 'pdf')  # list all pdf files
        # pdf_paths = random.sample(pdf_paths, 10)
        res = []
        fs = []
        for pdf in tqdm(pdf_paths):
            res.append(old_e.preprocess_pdf(pdf)['indication'])
            fs.append(ntpath.basename(pdf))  # pdf file name
        d = pd.DataFrame()
        d['pdf'] = fs
        d['indications'] = res
        d.to_excel(f"/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/indications/{to_save}.xlsx")

    def extract_data(self):
        def get_text(pdf):
            raw_tables_dict_1 = PDFTableExtractor.extract_raw_tables_from_pdf(pdf, end_page=2, return_format='table')
            # print(raw_tables_dict_1)
            if not raw_tables_dict_1:
                return []
            raw_tables = list(chain(*[table for table in raw_tables_dict_1 if table]))
            tmp = 0
            r = []
            for row in raw_tables:
                if tmp == 0:
                    if row[0] != None:
                        field_key = re.sub('\s', ' ', row[0])
                        field_key = re.sub('\d', ' ', field_key)
                        field_key = field_key.replace(' ', '')
                        if field_key.startswith('製造販売承認年月日'):
                            r.append(row[1:])
                            tmp = 1
                else:
                    if row[0] == None:
                        r.append(row[1:])
                    else:
                        tmp = 0
                        break
            if len(r) == 1:
                try:
                    a = r[0][0]
                except Exception:
                    a = ''
                return a
            else:
                res = ''
                for i in r:
                    res = res + str(i) + '\n'
                return res[0:-1]

        all_dirs = ['./data/pmda-001-pdf/', './data/pmda-001-pdf-2/', './data/pmda-002-pdf/', './data/pmda-002-pdf-2/'
            , './data/pmda-003-pdf/', './data/pmda-003-pdf-2/', './data/pmda-004-pdf/']
        all_pdfs = []
        for i in all_dirs:
            all_pdfs.extend(Utilfuncs.list_all_files(i, 'pdf'))
        logger.info('pdfs num: ' + str(len(all_pdfs)))

        fields_texts = []
        pdfs = []
        for pdf in tqdm(all_pdfs):
            field_text = get_text(pdf)
            pdfs.append(pdf.split('data/')[-1])  # pdf file name
            fields_texts.append(field_text)

        df = pd.DataFrame()
        df['pdf'] = pdfs
        df['年月日'] = fields_texts
        df.to_excel('./data/年月日.xlsx')


if __name__ == '__main__':
    # extractor = PMDAExtractor().test()
    PMDAExtractor().run()
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-001-pdf-2/"
    # PMDAExtractor().extract_indication(d, to_save='pmda-001-pdf-2')
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-004-pdf/"
    # PMDAExtractor().extract_indication(d, to_save='pmda-004-pdf')
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-003-pdf/"
    # PMDAExtractor().extract_indication(d, to_save='pmda-003-pdf')
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-002-pdf-2/"
    # PMDAExtractor().extract_indication(d, to_save='pmda-002-pdf-2')
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-002-pdf/"
    # PMDAExtractor().extract_indication(d, to_save='pmda-002-pdf')
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-001-pdf/"
    # PMDAExtractor().extract_indication(d, to_save='pmda-001-pdf')
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-003-pdf-2/"
    # PMDAExtractor().extract_indication(d, to_save='pmda-003-pdf-2')
