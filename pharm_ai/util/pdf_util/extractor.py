# encoding: utf-8
import zipfile
import os
from pprint import pprint
# install pdfplumber-0.5.14 on ubuntu 18.04
import pdfplumber
import string
from pathlib import Path
from time import time
import ntpath
from random import randint
import random
import pandas as pd
import shutil
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.config import ConfigFilePaths as cfp
from pathlib import Path
import zipfile
import requests
import json
from loguru import logger

logger.add("time.log", retention="10 days")

random.seed(123)


class Extractor:
    def __init__(self):
        pass

    def extract(self, file_name, file):
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
                        text = self.extract_text(raw_f)
                        result[raw_f_name] = text
                    elif raw_f.endswith('.doc'):
                        self.conv_doc2pdf(raw_f)
                        raw_f = str(unzip_path) + '/' + raw_f[:-4] + '.pdf'

                        text = self.extract_text(raw_f)
                        result[raw_f_name] = text
                except Exception as e:
                    print(e)
                    logger.error(e)
                    print('except')
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

    def extract_text(self, pdf_path):
        raw_text = ''
        pdf = pdfplumber.open(pdf_path)
        for page in pdf.pages:
            if not page.extract_text():
                logger.info('no-text-extracted page #: ' + str(page.page_number))
                logger.info('no-text-extracted page #: ' + ntpath.basename(pdf_path))
                continue
            raw_text += page.extract_text()
        return raw_text

    def extract_paragraphs(self, text):
        paras = text.split('。 \n')
        res = []
        for p in paras:
            p = u.clean_space(p)
            res.append(p)
        return res

    def make_pairs(self, pdf, titles):
        if pdf.endswith('pdf'):
            k = ntpath.basename(pdf).rstrip('.pdf')
        else:
            k = ntpath.basename(pdf).rstrip('.PDF')
        try:
            k = int(k)
        except:
            k = k
        title = titles[k]
        paragraphs = self.extract_paragraphs(e.extract_text(pdf))
        res = []
        if paragraphs:
            for p in paragraphs:
                res.append([title, p])

        tables = self.extract_tables(pdf)
        if tables:
            for t in tables:
                res.append([title, t])
        return res

    def extract_tables(self, pdf):
        res = []
        pdf = pdfplumber.open(pdf)
        for page in pdf.pages:
            for table in page.extract_tables():
                df = pd.DataFrame(table).applymap(str)
                s = ''
                for (col_name, col_vals) in df.iteritems():
                    s = s + str(col_name) + 'ㅎ'
                    cv = ''
                    for v in list(col_vals):
                        cv = cv + str(v) + 'ㅂ'
                    s += cv
                res.append(s)
        return res

    def conv_doc2pdf(self, file):
        cmd = 'lowriter --convert-to pdf ' + str(file)

    def unzip(self, file, unzip_path=None):
        zip_file_contents = zipfile.ZipFile(file, 'r')
        for file in zip_file_contents.namelist():
            try:
                if '__MACOSX' in file:
                    continue
                filename = file.encode('cp437').decode('utf-8')
                zip_file_contents.extract(file, unzip_path)  # unzip
                os.chdir(unzip_path)  # switch to target dir
                os.rename(file, filename)  # rename file
            except Exception as e:
                logger.error(file)
                logger.error(e)
                continue


if __name__ == '__main__':
    e = Extractor()
    