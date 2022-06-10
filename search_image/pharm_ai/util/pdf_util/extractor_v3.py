# -*- coding: UTF-8 -*-
"""
Description : extractor pdf text/table/field_text
"""

import ntpath
import re
from itertools import chain

import fitz
import pdfplumber
import pdftotext
from loguru import logger
from tqdm import tqdm

from pharm_ai.util.utils import Utilfuncs
import pandas as pd

class PDFTextExtractor:
    """the util to extract text from pdf

    give a pdf file path,and then extract text from it,even extract many pdfs texts
    """

    def __init__(self):
        pass

    @staticmethod
    def extract_raw_texts_from_pdf_dir(pdf_dir: str, method: str = 'pymupdf',
                                       start_page: int = 0, end_page: int = None,
                                       return_format='dict'):
        """extract pdfs texts by using a pdf file folder,default extract all pdfs

        Args:
            pdf_dir: a path including many pdfs,like: "./data/pmda-002-pdf-2/"
            method: the method to extract text : 'pdftotext' or 'pdfplumber',pymupdf
            start_page: the start page of pdf to extract
            end_page: the end page of pdf to extract
            pdf_num: the pdf num that you want to extract

        Returns:
            raw_texts: the dict including pdfs and texts,
                    like: [{"spider_c6442014e038d44591b02553736f5578.pdf":"pdf text"},...]
        """

        pdfs = Utilfuncs.list_all_files(pdf_dir, 'pdf')  # list all pdfs
        if not pdfs:
            pdfs = Utilfuncs.list_all_files(pdf_dir, 'PDF')
        if return_format == 'dict':
            raw_texts = dict()
        else:
            raw_texts = []
        for pdf in tqdm(pdfs):
            raw_text = PDFTextExtractor.extract_raw_text_from_pdf(pdf, method, start_page, end_page)
            pdf_name = ntpath.basename(pdf)  # pdf file name
            if return_format == 'dict':
                raw_texts[pdf_name] = raw_text  # {pdf name: pdf text, pdf name: pdf text...}
            else:
                raw_texts.append({'pdf_name': pdf_name, 'raw_text': raw_text})  # [{pdf name: pdf text}...]
        return raw_texts

    @staticmethod
    def extract_raw_text_from_pdf(pdf: str, method: str = 'pymupdf', start_page: int = 0, end_page: int = None):
        """extract one pdf by pdf path and specify the number of pages to fetch when use 'pdfplumber',return a text

        Args:
            pdf: a pdf file, like:'./data/pdf/spider_10b6507fcc301ae2aec9bd38c8c2b6b9.pdf'
            method: pdftotext or pdfplumber,or pymupdf
            start_page: the start page to extract,default 0,means the first page of pdf
            end_page: the start page to extract,default None,means the last page of pdf

        Returns:
            raw_text: the extracted raw text
        """
        try:
            if method == 'pdfplumber':
                with pdfplumber.open(pdf) as pdf_object:
                    pdf_pages = pdf_object.pages[start_page:] if end_page is None else pdf_object.pages[
                                                                                       start_page:end_page + 1]
                    raw_text = ''.join([pdf_page.extract_text() for pdf_page in pdf_pages])
            elif method == 'pdftotext':
                with open(pdf, "rb") as pdf_object:
                    pdf_pages = list(pdftotext.PDF(pdf_object))
                end_page = len(pdf_pages) if end_page is None else end_page + 1
                raw_text = ''.join(pdf_pages[start_page:end_page])
            else:
                pdf_doc = fitz.Document(pdf)
                pdf_pages = pdf_doc.page_count
                end_page = pdf_pages if end_page is None else end_page + 1
                raw_text = ''
                for pg in range(start_page, end_page):
                    pdf_page = pdf_doc.load_page(pg)
                    pdf_page.get_textpage()
                    clip = fitz.Rect(0, 0, pdf_page.rect.width, pdf_page.rect.height)
                    raw_text += pdf_page.get_textpage(clip=clip).extractText()
        except Exception as failed_e:
            logger.error(f'{pdf} : {failed_e}')
            raw_text = 'failed extracted pdf'
        return raw_text  # type:str

    @staticmethod
    def extract_text_in_a_double_column_page(pdf_page=None, pdf: str = '', page_num: int = 0):
        """use PyMuPDF,extract the double column text of one pdf page

        # example,if not pdf_page,you should specify the pdf and page_num
        # >>>pdf_doc = fitz.Document('./data/pdf/spider_10b6507fcc301ae2aec9bd38c8c2b6b9.pdf')
        # >>>pdf_page = pdf_doc.load_page(0)
        # >>>l_t, r_t = PDFTextExtractor.extract_text_in_a_double_column_page(pdf_page=pdf_page)

        Args:
            pdf: pdf, like: "xxx.pdf"
            page_num: page_num, like: 0
            pdf_page: pdf_page--by: pdf_page = pdf_doc.load_page(page_num)

        Returns:
            left_text:str, right_right:str
        """
        try:
            if pdf_page == None:
                pdf_doc = fitz.Document(pdf)  # pdf like: "xxx.pdf"
                pdf_page = pdf_doc.load_page(page_num)  # page_num(int) like: 0

            # left column
            left_clip = fitz.Rect(pdf_page.rect.width * 0.00, pdf_page.rect.height * 0.00,
                                  pdf_page.rect.width * 0.5, pdf_page.rect.height * 1.0)
            left_text = pdf_page.getText(clip=left_clip)

            # right column
            right_clip = fitz.Rect(pdf_page.rect.width * 0.5, pdf_page.rect.height * 0.00,
                                   pdf_page.rect.width * 1.00, pdf_page.rect.height * 1.00)
            right_text = pdf_page.getText(clip=right_clip)
        except Exception as failed_e:
            logger.error(f'{pdf} : {failed_e}')
            left_text, right_text = '', ''
        return left_text, right_text


class PDFTableExtractor:
    """extract tables from pdf

    """

    def __init__(self):
        pass

    @staticmethod
    def extract_raw_tables_from_pdf(pdf: str, start_page: int = 0, end_page: int = None, return_format='dict'):
        """extract tables from a pdf

        Args:
            pdf: PDF File
            start_page: the first page to begin extract,from 0 to start
            end_page: the last page to extract
            return_format: if dict

        Returns:
            table_list : list/dict
        """
        try:
            with pdfplumber.open(pdf) as pdf_object:
                pdf_pages = pdf_object.pages[start_page:] if end_page is None else pdf_object.pages[
                                                                                   start_page:end_page + 1]
                table_list = [pdf_page.extract_tables() for pdf_page in pdf_pages]
                if return_format == 'df':
                    res = []
                    for page_tables in table_list:
                        for every_table in page_tables:
                            res.append(pd.DataFrame(every_table[1:], columns=every_table[0]))
                    return res
            # print(table_list)
            raw_tables = list(chain(*[table for table in table_list if table]))

            if return_format == 'dict':
                raw_tables = PDFTableExtractor.get_tables_dict(raw_tables)
            return raw_tables
        except Exception:
            logger.error(f'{pdf} : {Exception}')
            if return_format == 'dict':
                return dict()
            else:
                return []

    @staticmethod
    def get_tables_dict(extracted_raw_tables: list):
        """format the extracted raw tables to dict,first column is the key ,and other columns are values

        Args:
            extracted_raw_tables: extracted_raw_tables

        Returns:
            tables_dict: {field_key:field_value},like: {'f1':['cc',''],'f2':['sff'],'f3':[]}
        """
        tables_dict = dict()
        for table in extracted_raw_tables:
            for row in table:
                field_key = row[0]
                if field_key:
                    field_key = re.sub('\s', ' ', field_key)
                    field_key = re.sub('\d', ' ', field_key)
                    field_key = field_key.replace(' ', '')
                    if len(row) >= 1:
                        field_value = row[1:]
                    else:
                        field_value = []
                    tables_dict[field_key] = field_value  # type:list
        return tables_dict  # type:dict


    @staticmethod
    def get_tables_dict2(extracted_raw_tables: list):
        """format the extracted raw tables to dict,first column is the key ,and other columns are values

        Args:
            extracted_raw_tables: extracted_raw_tables

        Returns:
            tables_dict: {field_key:field_value},like: {'f1':['cc',''],'f2':['sff'],'f3':[]}
        """

        keys = []
        values = []
        for table in extracted_raw_tables[0:1]:
            for row_id in range(len(table)):
                if row_id ==0:
                    for field_key in table[0]:
                        if field_key:
                            field_key = re.sub('\s', ' ', field_key)
                            field_key = re.sub('\d', ' ', field_key)
                            field_key = field_key.replace(' ', '')
                            keys.append(field_key)
                            values.append([])
                        else:
                            keys.append('None')
                            values.append([])
                    continue

                for field_value_id in range(len(table[row_id])):
                    if table[row_id][field_value_id]:
                        values[field_value_id].append(table[row_id][field_value_id])
        tables_dict = dict(zip(keys,values))
        return tables_dict  # type:dict

class FieldTextExtractor:
    """ extract the field text from raw_text

    """

    def __init__(self):
        pass

    @staticmethod
    def extract_field_text_by_split(text: str, fields: dict, start_end: list, field_location=1):
        """extract one field text by spliting start field and end field

        Args:
            text: str, text
            fields: dict,keys include start fields and end fields, {field_name: {fields_other_names}},
                    like:{'INDICATIONS AND DOSAGE': ['\nINDICATIONS AND DOSAGE:', '\n[INDICATIONS AND DOSAGE]',
                     '\n【 INDICATIONS AND DOSAGE 】'], 'Clinical results':[...]}
            start_end: start field and end field , like: ['INDICATIONS AND DOSAGE', 'Clinical results'],
            field_location: which field text

        Returns:
            dict, {start field name: field_text}
        """
        start_field = start_end[0]
        end_field = start_end[1]
        start_fields = fields[start_field]
        end_fields = fields[end_field]
        res = dict()
        for sf in start_fields:
            for ef in end_fields:
                if sf in text and ef in text:
                    try:
                        field_text = text.split(sf)[field_location].split(ef)[0]
                        # field_text = Utilfuncs.clean_space_en(field_text)
                        res[start_field] = field_text
                    except Exception as e:
                        logger.error(e)
                        res[start_field] = ''
        return res  # type:dict, {start_field:field_text}

    @staticmethod
    def extract_field_text_by_re(text: str, fields_re: list):
        """ extract_field_text_by_re
        Args:
            fields_re: list,the list of field_re
                    like:[r's((?:.|\n)*?)q','ger'...]
                    note: the re list should be in order from longest to shortest
            text: extracted pdf text, like: 'sfqsqgdsgffq'

        Returns:
            r_field :list,match a lot of text,['f', '', 'gff']
        """
        field_text_list = []
        for f_r in fields_re:
            field_text_list = re.findall(f_r, text)
            if field_text_list:
                if '' in field_text_list:
                    field_text_list.remove('')
            if field_text_list != []:  # [] means: not find
                break
        return field_text_list  # type:list

    @staticmethod
    def extract_field_text_from_table(field_key_re: str, tables_dict=None, raw_tables: list = None):
        """ extract_field_text_from_extracted_table

        Args:
            field_key_re: str,like: 'f1'
            raw_tables: extracted_tables
            tables_dict: field_tables_dict,like: {'f1':['cc',''],'f2':['sdfsdf'],'f3':[]}

        Returns:
            tables_dict[key]: list,like: ['cc','']
        """
        if not tables_dict:
            tables_dict = PDFTableExtractor.get_tables_dict(raw_tables)
        all_keys = tables_dict.keys()
        if field_key_re in all_keys:
            return tables_dict[field_key_re]
        else:
            for key in all_keys:
                if re.search(field_key_re, key):
                    return tables_dict[key]
            return []


class Extractor:
    def __init__(self):
        self.fields_re = {'f1': [r'sdf', r'sdsf'],
                          'f2': [r'sdf', r'sdsf'], }

    def extract_texts_from_pdf_dir(self, pdfs_dir):
        pdf_paths = Utilfuncs.list_all_files(pdfs_dir, 'pdf')  # list all pdf files
        logger.info('pdfs num: ' + str(len(pdf_paths)))
        fields_texts = []
        for pdf in tqdm(pdf_paths):
            field_text_dict = self.extract_fields_text_from_pdf(pdf)
            pdf_name = ntpath.basename(pdf)  # pdf file name
            field_text_dict.update({'pdf_name': pdf_name})
            fields_texts.append(field_text_dict)
        return fields_texts  # type:dict

    def extract_fields_text_from_pdf(self, pdf):
        # ...
        # raw_text = self.extract_raw_text_from_pdf(pdf)
        # raw_text = Extractor.handle_raw_text(raw_text)
        # field_text = self.extract_field_text_by_re(raw_text,self.fields)
        # text = self.handle_raw_text(field_text)
        # ...
        text_dict = dict()
        return text_dict  # dict,{field_name:field_text}

    @staticmethod
    def handle_raw_text(raw_text: str):
        # ...
        raw_text = Extractor.remove_page_header_footer(raw_text)
        text = raw_text
        return text

    @staticmethod
    def remove_page_header_footer(raw_text, page_header_footer_pattern=r'\n\s*\d{1,3}\s*\n'):
        # ...
        text = re.sub(page_header_footer_pattern, ' \n ', raw_text)
        return text

    @staticmethod
    def handle_field_text(field_text: str):
        # ...
        text = field_text
        return text

    @staticmethod
    def change_number_font(num_chrs: str):
        """
        example:
        # >>>num_chrs = '２０１９'
        # >>>Extractor().change_number_font(num_chrs) # '2019'
        """
        for ch in num_chrs:
            if 65296 <= ord(ch) <= 65305:
                num_chrs = num_chrs.replace(ch, chr(ord(ch) - 65248))
        return num_chrs


if __name__ == '__main__':
    pdf = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pdf/"
    # s = PDFTextExtractor.extract_raw_text_from_pdf("/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pdf/CN106317224B.PDF")
    print(PDFTextExtractor.extract_raw_texts_from_pdf_dir(pdf, method='pdftotext'))
    pass
