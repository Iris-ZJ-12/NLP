# encoding: utf-8
import pickle
import zipfile
import os
import ntpath
from pprint import pprint
# install pdfplumber-0.5.14 on ubuntu 18.04
import pdfplumber
import pdftotext
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
import string
from pathlib import Path
from time import time
from random import randint
import pandas as pd
import shutil
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.config import ConfigFilePaths as cfp
from pathlib import Path
import zipfile
import requests
import json
from loguru import logger
from itertools import chain
from tqdm import tqdm


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

    def extract(self, file_name, file):
        random = randint(0, 1000)
        cwd = os.getcwd()
        cwd_name = os.path.basename(cwd)
        sub_dir = '/' + str(cwd_name) + '/' + file_name[:-4] + '_' + str(random) + '/'
        unzip_path = cfp.project_dir + sub_dir + 'unzipped/'
        unzip_path = Path(unzip_path)
        unzip_path.mkdir(parents=True)
        result_text, result_table = dict(), dict()
        if file_name.endswith('.zip'):
            self.unzip(file, unzip_path)
            p = str(unzip_path) + '/'
            pbar = tqdm(os.listdir(p))
            for raw_f in pbar:
                pbar.set_description("Extracting '{}'".format(raw_f))
                raw_f_name = os.path.basename(raw_f)
                try:
                    if self.check_if_pdf(raw_f):
                        raw_f = str(unzip_path) + '/' + raw_f
                        text = self.extract_raw_full_text(raw_f, method='pdftotext', start_page_n=0)
                        result_text[raw_f_name] = text
                    elif raw_f.endswith('.doc'):
                        self.conv_doc2pdf(raw_f)
                        raw_f = str(unzip_path) + '/' + raw_f[:-4] + '.pdf'
                        text = self.extract_raw_full_text(raw_f, method='pdftotext', start_page_n=0)
                        result_text[raw_f_name] = text
                except Exception as e:
                    logger.error(e)
                    result_text[raw_f_name] = ''
                try:
                    if self.check_if_pdf(raw_f):
                        table = self.extract_raw_table(raw_f)
                        result_table[raw_f_name] = table
                    elif raw_f.endswith('.doc'):
                        self.conv_doc2pdf(raw_f)
                        raw_f = str(unzip_path) + '/' + raw_f[:-4] + '.pdf'
                        table = self.extract_raw_table(raw_f)
                        result_table[raw_f_name] = table
                except Exception as e:
                    logger.error(e)
                    result_table[raw_f_name] = ''
        else:
            shutil.rmtree(cfp.project_dir + sub_dir)
            return '请上传 *.pdf / *.PDF / *.doc 的 *.zip 压缩包 !'
        shutil.rmtree(cfp.project_dir + sub_dir)
        return result_text, result_table

    def extract_tables(self, file_name, file):
        random = randint(0, 1000)
        cwd = os.getcwd()
        cwd_name = os.path.basename(cwd)
        sub_dir = '/' + str(cwd_name) + '/' + file_name[:-4] + '_' + str(random) + '/'
        unzip_path = cfp.project_dir + sub_dir + 'unzipped/'
        unzip_path = Path(unzip_path)
        unzip_path.mkdir(parents=True)
        result = dict()
        if file_name.endswith('.zip'):
            self.unzip(file, unzip_path)
            p = str(unzip_path) + '/'
            for raw_f in os.listdir(p):
                raw_f_name = os.path.basename(raw_f)
                try:
                    if self.check_if_pdf(raw_f):
                        raw_f = str(unzip_path) + '/' + raw_f
                        text = self.extract_raw_full_text(raw_f, method='pdftotext', start_page_n=0)
                        result[raw_f_name] = text
                    elif raw_f.endswith('.doc'):
                        self.conv_doc2pdf(raw_f)
                        raw_f = str(unzip_path) + '/' + raw_f[:-4] + '.pdf'
                        text = self.extract_raw_full_text(raw_f, method='pdftotext', start_page_n=0)
                        result[raw_f_name] = text
                except Exception as e:
                    logger.error(e)
                    result[raw_f_name] = ''
        else:
            shutil.rmtree(cfp.project_dir + sub_dir)
            return '请上传 *.pdf / *.PDF / *.doc 的 *.zip 压缩包 !'
        shutil.rmtree(cfp.project_dir + sub_dir)
        return result

    def check_if_pdf(self, file_name):
        if file_name.endswith('.pdf') or file_name.endswith('.PDF'):
            return True
        else:
            return False

    def extract_raw_full_text(self, pdf,
                              method='pdftotext', start_page_n=0,
                              end_page_n: int = None):
        raw_text = ''
        try:
            page_n = 0
            if method == 'pdfplumber':
                pdf = pdfplumber.open(pdf)
                pages = pdf.pages if end_page_n is None else pdf.pages[:end_page_n+1]
                for page in pages:
                    if page_n < start_page_n:
                        page_n += 1
                        continue
                    else:
                        raw_text += page.extract_text()
            else:
                with open(pdf, "rb") as file:
                    li = list(pdftotext.PDF(file))
                raw_text = ''.join(li[start_page_n:])
        except Exception as e:
            logger.error(pdf)
            logger.error(e)
            raw_text = ''
        return raw_text

    def extract_raw_full_text_dir(self, pdf_dir,
                                  method='pdfplumber', start_page_n=0,
                                  file_extention='PDF'):
        pdf_paths = u.list_all_files(pdf_dir, file_extention)
        logger.info('path num: '+str(len(pdf_paths)))
        raw_texts = dict()
        c = 0
        e = 0
        for pdf in pdf_paths:
            raw_text = self.extract_raw_full_text(pdf, method, start_page_n)
            if not raw_text:
                e += 1
            else:
                c += 1
            pdf = ntpath.basename(pdf)
            raw_texts[pdf] = raw_text
            if c % 500 == 0:
                logger.info(str(c)+' pdf full raw texts were extracted.')
            # if e % 10 == 0:
            #     logger.info(str(e)+' pdf raw texts cannot be extracted.')
        return raw_texts

    def extract_text(self, pdf, field_names_dic, h5, method='pdftotext'):
        raw_text = self.extract_raw_full_text(pdf, method)
        res = self.extract_fields(raw_text, field_names_dic, h5)
        return res

    def extract_fields(self, text, field_names_dic,
                       h5=cfp.project_dir + '/package_insert_extractors/fields.h5'):
        df = pd.read_hdf(h5)
        field_names = df['field_names'].tolist()
        field_names_all = ['\n' + fn.lstrip('\\n') for fn in field_names]
        res = dict()
        for k, v in field_names_dic.items():
            res.update(self.extract_field_text(k, v, field_names_all, text))
        return res

    def extract_field_text(self, field_type, field_names, field_names_all, text):
        res = dict()
        for field_name in field_names:
            start = text.find(field_name)
            try:
                if start >= 0:
                    ends = []
                    for fn in field_names_all:

                        if fn != field_name and fn in text:
                            end = text.find(fn) - 1
                            if end < 2:
                                continue
                            ends.append(end)
                    ends = sorted(ends)
                    end = [end for end in ends if end > start][0]
                    field_text = text[start:end].lstrip(field_name)
                    field_text = ' '.join(field_text.split())
                    # field_text = u.clean_space(field_text)
                    # if not field_text.endswith('。'):
                    #     field_text += '。'
                    res[field_type] = field_text
                    # if field_type == '适应症':
                    #     indications_ner = self.extract_indications(field_text)
                    #     if indications_ner:
                    #         res['适应症-实体'] = indications_ner
                    break

            except Exception as e:
                logger.error(e)
                continue
        return res

    def prep_dic(self, dic_xlsx_path, save_h5_path, save_pkl_path):
        df = pd.read_excel(dic_xlsx_path, '1').dropna()
        df.to_hdf(save_h5_path, '1')
        grs = df.groupby('field_types')
        dic = dict()
        for i, g in grs:
            field_names = g['field_names'].tolist()
            dic[i] = field_names
        f = open(save_pkl_path, 'wb')
        pickle.dump(dic, f)

    def conv_doc2pdf(self, file):
        cmd = 'lowriter --convert-to pdf ' + str(file)
        os.system(cmd)

    def unzip(self, file, unzip_path=None):
        zip_file_contents = zipfile.ZipFile(file, 'r')
        for file in zip_file_contents.namelist():
            try:
                if '__MACOSX' in file:
                    continue
                filename = file.encode('cp437').decode('utf-8')
                zip_file_contents.extract(file, unzip_path)  # unzip
                os.rename(os.path.join(unzip_path, file),
                          os.path.join(unzip_path, filename))  # rename file
            except Exception as e:
                logger.error(file)
                logger.error(e)
                continue

    def extract_raw_table(self, file, end_page=None, start_page=0):
        """
        :param str file: PDF File path.
        :param int start_page: the first page to begin extract.
        :param int end_page: the last page to extract (not include).
        :return: List of table list.
        :rtype: list
        """
        try:
            with pdfplumber.open(file) as pdf:
                pages = pdf.pages[start_page:] if end_page is None else pdf.pages[start_page:end_page]
                table_list = [page.extract_tables() for page in pages]
            table_list = list(chain(*[table for table in table_list if table]))
            return table_list
        except:
            return []

if __name__ == '__main__':
    e = Extractor()
    field_names_dic = {'不良反应': ['\n不良反应:', '\n[不良反应]', '\n【 不 良 反 应 】', '\n【不良反应】'],
                       '适应症': ['\n适应症:', '\n[适应症]', '\n【 适 应 症 】', '\n【适应症】', '\n功能主治:'],
                       '药理毒理': ['\n药理毒理:', '\n[药理毒理]', '\n【药 理 毒 理】', '\n【药理毒理】'],
                       '用法用量': ['\n用法用量:', '\n[用法用量]', '\n【 用 法 用 量】', '\n【用法用量】']}
    # r = e.extract_text(f, field_names_dic)
    # print(r)
    f = '/Users/gwl/Projects/PharmAI/pharm_ai/package_insert_extractors/spider_454ed5c0b1f7daf37e7e25e36bbcea20.pdf'
    print(repr(e.extract_raw_full_text(f)))