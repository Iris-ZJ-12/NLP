# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: sample_recognize_content.py

DESCRIPTION:
    This sample demonstrates how to extract text, selection marks, and content information from a document
    given through a file.

    Note that selection marks returned from begin_recognize_content() do not return the text associated with
    the checkbox. For the API to return this information, train a custom model to recognize the checkbox and its text.
    See sample_train_model_with_labels.py for more information.

USAGE:
    python sample_recognize_content.py

    Set the environment variables with your own values before running the sample:
    1) AZURE_FORM_RECOGNIZER_ENDPOINT - the endpoint to your Cognitive Services resource.
    2) AZURE_FORM_RECOGNIZER_KEY - your Form Recognizer API key
"""

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.exceptions import HttpResponseError
import sys
import numpy as np
import os
import pandas as pd
import pickle
from loguru import logger
from pdfrw import PdfReader, PdfWriter, PageMerge
import signal
import time

def set_timeout(num):
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError(f'Time out {num}s')
        def to_do(*args, **kwargs):
            signal.signal(signal.SIGALRM, handle)
            signal.alarm(num)
            res = func(*args, **kwargs)
            signal.alarm(0)
            return res
        return to_do
    return wrap


class RecognizeContentSample(object):
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.results = ContentResult(pdf_path)

    def recognize_content(self, result_txt=None, cache_file: str=None, use_cache_result=False, form_obj=None):
        if result_txt:
            sys.stdout = open(result_txt, 'a')
        # [START recognize_content]

        if form_obj:
            form_pages = form_obj
        else:
            form_pages = self.get_form_page_obj(cache_file, use_cache_result)

        print("----Recognizing {}-----".format(self.results.name))

        for idx, content in enumerate(form_pages):
            print("----Recognizing content from page #{}----".format(idx+1))
            page_info = (idx+1, content.width * 2.54, content.height * 2.54, 'cm' if content.unit == 'inch' else content.unit)
            self.results.save_page_result(page_info)
            print("Page has width: {} and height: {}, measured with unit: {}".format(*page_info[1:]))
            tables = self.sort_tables(content)
            for table_idx, table in enumerate(tables):
                table_info = (idx+1, table_idx, table.row_count, table.column_count)
                self.results.save_table_result(table_info=table_info)
                print("Table # {} has {} rows and {} columns".format(*table_info[1:]))
                if 'bounding_box' in table.__dict__:
                    table_bounding_box_info = (idx+1, table_idx, table.bounding_box)
                    self.results.save_table_result(bounding_box_info=table_bounding_box_info)
                    print("Table # {} location on page: {}".format(table_idx, self.format_bounding_box(table.bounding_box)))
                cells = self.sort_cells(table)
                for cell_idx, cell in enumerate(cells):
                    cell_bounding_box_info = (idx+1, table_idx, cell_idx, cell.text, cell.bounding_box)
                    self.results.save_table_result(cell=cell_bounding_box_info)
                    print("...Cell[{}][{}] has text '{}' within bounding box '{}'".format(
                        cell.row_index,
                        cell.column_index,
                        cell.text,
                        self.format_bounding_box(cell.bounding_box)
                    ))
            if content.lines:
                for line_idx, line in enumerate(content.lines):
                    line_info = (idx+1, line_idx, len(line.words), line.text, line.bounding_box)
                    self.results.save_line_result(line=line_info)
                    print("Line # {} has word count '{}' and text '{}' within bounding box '{}'".format(
                        line_idx,
                        len(line.words),
                        line.text,
                        self.format_bounding_box(line.bounding_box)
                    ))
                    for word in line.words:
                        word_info = (idx+1, line_idx, word.text, word.confidence)
                        self.results.save_line_result(word=word_info)
                        print("...Word '{}' has a confidence of {}".format(word.text, word.confidence))

            # for selection_mark in content.selection_marks:
            #     print("Selection mark is '{}' within bounding box '{}' and has a confidence of {}".format(
            #         selection_mark.state,
            #         format_bounding_box(selection_mark.bounding_box),
            #         selection_mark.confidence
            #     ))
            print("----------------------------------------")

        # [END recognize_content]

    @set_timeout(600)
    def get_form_page_obj(self, cache_file:str=None, use_cache_result=False, return_obj=True):
        if not use_cache_result or not cache_file:
            to_ocr = True
        elif not os.path.exists(cache_file):
            to_ocr = True
        else:
            to_ocr = False
        if to_ocr:
            form_pages = self.ocr(cache_file)
        else:
            with open(cache_file, 'rb') as cache_f:
                form_pages = pickle.load(cache_f)
            logger.info('Form_page object was loaded from cache file "{}".', cache_name)
        if return_obj:
            return form_pages

    def ocr(self, cache_file):
        endpoint = 'https://sss.cognitiveservices.azure.com/'
        key = '72932751ac914646b17f46f64d8bd76b'
        form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        try:
            with open(self.pdf_path, "rb") as f:
                poller = form_recognizer_client.begin_recognize_content(form=f)
        except HttpResponseError as ex:
            logger.error(ex.message)
            pdf_processor = PdfProcessor(self.pdf_path)
            pdf_processor.adjust_page_size()
            self.pdf_path = pdf_processor.pdf_path
            with open(self.pdf_path, 'rb') as f:
                poller = form_recognizer_client.begin_recognize_content(form=f)
        form_pages = poller.result()
        if cache_file:
            with open(cache_file, 'wb') as cache_f:
                pickle.dump(form_pages, cache_f)
            logger.info('Form_page object has been saved to cache file "{}".', cache_name)
        return form_pages

    def format_bounding_box(self, bounding_box):
        if not bounding_box:
            return "N/A"
        return ", ".join(["[{}, {}]".format(p.x, p.y) for p in bounding_box])

    def sort_tables(self, content):
        tables = content.tables
        if tables:
            bounding_boxes = [self.get_bounding_box(t) for t in tables]
            bounding_boxes = np.array(bounding_boxes)
            table_min_y = bounding_boxes[:,:,1].min(axis=1)
            res_ids = table_min_y.argsort()
            res = [tables[i] for i in res_ids]
        else:
            res = []
        return res

    def get_bounding_box(self, table):
        """Get bounding box of all cells in a table."""
        if 'bounding_box' in table.__dict__:
            res = table.bounding_box
        else:
            x_points, y_points = zip(*[(box.x, box.y) for cell in table.cells for box in cell.bounding_box])
            x_min, x_max = min(x_points), max(x_points)
            y_min, y_max = min(y_points), max(y_points)
            res = [(x_min, y_min),(x_max, y_min),(x_max, y_max),(x_min, y_max)]
        return res

    def sort_cells(self, table):
        cells = table.cells
        boxes = [[[b.x, b.y] for b in c.bounding_box] for c in table.cells]
        bounding_boxes = np.array(boxes)
        ind = np.lexsort((bounding_boxes[:,0,0], bounding_boxes[:,0,1]))
        res = [cells[i] for i in ind]
        return res

class ContentResult:
    def __init__(self, pdf_path):
        self.name = os.path.basename(pdf_path)

        # page_result: [page_index, width, height, unit]
        self.page_result = []

        # table_info: [page_index, table_index, row_count, column_count, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]
        self.table_info_result = []

        # cell_info: [page_index, table_index, cell_index, cell_text, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]
        self.cell_info = []

        # line_info: [page_index, line_index, word_count, text, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]
        self.line_info = []

        # word_info: [page_index, line_index, word_text, word_confidence]
        self.word_info = []


    def save_page_result(self, new_result: tuple):
        """Append [page_index, width, height, unit]"""
        self.page_result.append(new_result)

    def save_table_result(self, *, table_info: tuple=None, cell: tuple=None, bounding_box_info:tuple=None):
        """table_info: Tuple[page_index, table_index, row_count, column_count],
        cell: Tuple[page_index, table_index, cell_index, bounding_box: cell.bounding_box],
        bounding_box_info: Tuple[page_index, table_index: int, bounding_box: table.bounding_box]"""
        if table_info:
            self.table_info_result.append(list(table_info)+[None]*8)
        if bounding_box_info:
            for ind, res in enumerate(self.table_info_result):
                if res[0] == bounding_box_info[0] and res[1] == bounding_box_info[1]:
                    res[-8:] = [xy*2.54 for p in bounding_box_info[-1] for xy in [p.x, p.y]]
        if cell:
            new_cell_info = list(cell[:4]) + [xy*2.54 for p in cell[-1] for xy in [p.x, p.y]]
            self.cell_info.append(new_cell_info)

    def save_line_result(self, *, line: tuple = None, word: tuple = None):
        """line: Tuple[page_index, line_index, word_count, line.bounding_box],
        word: Tuple[page_index, line_index, word_text, word_confidence]"""
        if line:
            self.line_info.append(list(line[:4]) + [xy*2.54 for p in line[4] for xy in [p.x, p.y]])
        if word:
            self.word_info.append(word)

    def get_page_result(self, add_name=False):
        df = pd.DataFrame(self.page_result, columns=['page_index', 'width', 'height', 'unit'])
        if add_name:
            df = self.add_name(df)
        return df

    def get_table_info(self, add_name=False):
        df = pd.DataFrame(self.table_info_result, columns=[
            'page_index', 'table_index', 'row_count', 'column_count', 'p1x', 'p1y', 'p2x', 'p2y', 'p3x', 'p3y', 'p4x', 'p4y'])
        if add_name:
            df = self.add_name(df)
        return df

    def add_name(self, df: pd.DataFrame):
        res = df.insert(0, 'pdf_name', self.name)
        return res

    def get_cell_info(self, add_name=False):
        df = pd.DataFrame(self.cell_info,
                          columns=['page_index', 'table_index', 'cell_index', 'cell_text', 'p1x', 'p1y', 'p2x', 'p2y',
                                   'p3x', 'p3y', 'p4x', 'p4y'])
        if add_name:
            df = self.add_name(df)
        return df

    def get_line_info(self, add_name=False):
        df = pd.DataFrame(self.line_info,
                          columns=['page_index', 'line_index', 'word_count', 'text', 'p1x', 'p1y', 'p2x', 'p2y', 'p3x',
                                   'p3y', 'p4x', 'p4y'])
        if add_name:
            df = self.add_name(df)
        return df

    def get_word_info(self, add_name=False):
        df = pd.DataFrame(self.word_info,
                          columns=['page_index', 'line_index', 'word_text', 'word_confidence'])
        if add_name:
            df = self.add_name(df)
        return df

    def save_to_excel(self, excel_name:str=None):
        page_res = self.get_page_result()
        table_res = self.get_table_info()
        cell_res = self.get_cell_info()
        line_res = self.get_line_info()
        word_res = self.get_word_info()

        if not excel_name:
            excel_name = 'results/result_for_'+self.name.split('.')[0]+'.xlsx'
        check_dir(excel_name)
        with pd.ExcelWriter(excel_name) as writer:
            page_res.to_excel(writer, sheet_name='Page')
            table_res.to_excel(writer, sheet_name='Table')
            cell_res.to_excel(writer, sheet_name='Cell')
            line_res.to_excel(writer, sheet_name='Line')
            word_res.to_excel(writer, sheet_name='Word')

class PdfProcessor:
    def __init__(self, pdf_path):
        self.set_(pdf_path)

    def set_(self, pdf_path):
        self.pdf_path = pdf_path
        self.name = os.path.basename(self.pdf_path)
        self.reader = PdfReader(self.pdf_path)
        self.pages = self.reader.pages

    def get_pages_size(self):
        result = np.array([(PageMerge()+p).xobj_box[2:] for p in self.pages])/72 # in inch
        return result

    def adjust_page_size(self, max_size=(17, 17)):
        out_f = os.path.join(os.path.dirname(self.pdf_path), 'adjusted_'+self.name)
        writer = PdfWriter(out_f)
        sizes = self.get_pages_size()
        is_adjust, max_sizes = np.broadcast_arrays((sizes>np.array(max_size)).any(axis=1)[:,np.newaxis], np.array(max_size))
        scales=(np.reciprocal(sizes)*max_sizes).min(axis=1)
        for scale, page in zip(scales, self.pages):
            srcpage = PageMerge()+page
            if scale<1:
                srcpage[0].scale(scale)
            res_page = srcpage.render()
            writer.addpage(res_page)
        writer.write()
        logger.info("'{}' Adjusted to '{}'", self.name, out_f)
        self.set_(out_f)

def check_dir(file_path):
    dir_ = os.path.dirname(file_path)
    if not os.path.exists(dir_):
        logger.warning('Folder "{}" not exist, now creating.')
        os.makedirs(dir_, exist_ok=True)


if __name__ == '__main__':
    pdfs = ['raw_data/PDF0205/'+f for f in os.listdir('raw_data/PDF0205/') if f.endswith('.pdf')]
    for i, pdf in enumerate(pdfs):
        sample = RecognizeContentSample(pdf)
        cache_name = 'cache/PDF0205/' + sample.results.name.split('.')[0]+'.txt'
        # sample.recognize_content(cache_file=cache_name, use_cache_result=True)
        # sample.results.save_to_excel('results/PDF0205/result_for_' + sample.results.name.split('.')[0]+'.xlsx')
        sample.get_form_page_obj(cache_name, True)