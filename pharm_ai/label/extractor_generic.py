# encoding: utf-8
import pickle
import zipfile
import os
import ntpath
from pprint import pprint
# install pdfplumber-0.5.14 on ubuntu 18.04
import pdfplumber
import pdftotext
from pharm_ai.util.utils import Utilfuncs as u
import pandas as pd
from pharm_ai.config import ConfigFilePaths as cfp
from time import time
import json
from loguru import logger
logger.add("time.log", retention="10 days")


class Extractor:
    def __init__(self, fields_dic=None, starts_ends=None):
        try:
            f = open(fields_dic, 'r')
            self.fields = json.load(f)
            self.starts_ends = starts_ends
        except Exception as e:
            logger.error(e)

    def prep_fields_dic(self, excel_path, sheet_name='1', json_file_name=''):
        df = pd.read_excel(excel_path, sheet_name).dropna()
        grs = df.groupby('field_types')
        res = dict()
        for n, g in grs:
            fns = g['field_names'].tolist()
            res[n] = fns
        f = open(json_file_name, 'w')
        json.dump(res, f, indent=4, ensure_ascii=False)

    def extract_raw_full_text(self, pdf, method='pdftotext'):
        raw_text = ''
        try:
            if method == 'pdfplumber':
                pdf = pdfplumber.open(pdf)
                for page in pdf.pages:
                    raw_text += page.extract_text()
            else:
                file = open(pdf, "rb")
                raw_text = ''.join(pdftotext.PDF(file))
        except Exception as e:
            logger.error(pdf)
            logger.error(e)
            raw_text = ''
        return raw_text

    def extract_raw_full_text_dir(self, pdf_dir,
                                  method='pdfplumber'):
        pdf_paths = u.list_all_files(pdf_dir, 'pdf')
        logger.info('path num: '+str(len(pdf_paths)))
        raw_texts = dict()
        for pdf in pdf_paths:
            raw_text = self.extract_raw_full_text(pdf, method)
            pdf = ntpath.basename(pdf)
            raw_texts[pdf] = raw_text
        return raw_texts

    def extract_text(self, pdf, fields, method='pdftotext'):
        raw_text = self.extract_raw_full_text(pdf, method)
        res = self.extract_all_field_texts(raw_text, fields)
        return res

    def extract_all_field_texts(self, text=None, fields=None):

        res = dict()
        for start_end in self.starts_ends:
            wanted_field = self.extract_one_field_text(text, fields, start_end)
            res.update(wanted_field)
        return res

    def extract_one_field_text(self, text, fields, start_end):
        start = start_end[0]
        end = start_end[1]
        start_fields = fields[start]
        end_fields = fields[end]
        res = dict()
        for sf in start_fields:
            for ef in end_fields:
                if sf in text and ef in text:
                    try:
                        field = text.split(sf)[1].split(ef)[0]
                        field = u.clean_space_en(field)
                        res[start] = field
                    except Exception as e:
                        logger.error(e)
                        res[start] = ''
        return res

    def extract_fileds(self, texts_json, save_json_path, err_json_path):
        f = open(texts_json, 'r')
        dt = json.load(f)
        res = dict()
        err_pdfs = []
        t1 = time()
        c = 0
        for pdf_name, pdf_text in dt.items():
            try:
                field_texts = self.extract_all_field_texts(pdf_text, self.fields)
                res[pdf_name] = field_texts
            except Exception as e:
                logger.error(pdf_name)
                logger.error(e)
                err_pdfs.append(pdf_name)
        t2 = time()
        print(t2 - t1)
        f1 = open(save_json_path, 'w')
        json.dump(res, f1, indent=4, ensure_ascii=False)

        err_pdfs = {'errors': err_pdfs}
        f2 = open(err_json_path, 'w')
        json.dump(err_pdfs, f2, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # e = Extractor()
    # excel_path = 'field_names-IF20201010.xlsx'
    # sheet_name = '1'
    json_file_name = 'field_names-PMDA.json'
    # e.prep_fields_dic(excel_path, sheet_name, json_file_name)

    starts_ends = [['INDICATIONS AND DOSAGE', 'Clinical results'],
                   ['Children', 'Senior citizens or Clinical research']]

    e = Extractor(json_file_name, starts_ends)
    f1 = 'pmda_if-all-20201009-.json'
    f2 = 'pmda-if-all-fields.json'
    f3 = 'pmda-if-error-pdfs.json'
    e.extract_fileds(f1, f2, f3)

    def dt1010_pmda():
        f = open('pmda-if-all-fields.json', 'r')
        dt = json.load(f)
        pdfs = []
        ius = []
        chs = []
        for pdf_name, fields in dt.items():
            pdfs.append(pdf_name)
            iu = fields.get('INDICATIONS AND DOSAGE', '')
            if iu:
                iu = u.remove_illegal_chars(iu)
            ius.append(iu)

            ch = fields.get('Children', '')
            if ch:
                ch = u.remove_illegal_chars(ch)
            chs.append(ch)
        dt = {
            'pdf_name': pdfs,
            'INDICATIONS AND DOSAGE': ius,
              'Children': chs
              }
        df = pd.DataFrame(dt)
        u.to_excel(df, 'pmda-if-all-fields-20201010.xlsx', 'result')

    dt1010_pmda()