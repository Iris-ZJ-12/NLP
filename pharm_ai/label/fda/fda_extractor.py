# -*- coding: UTF-8 -*-
"""
Description : extract field-text from American pdfs
"""
import ntpath
import re

import fitz
import pandas as pd
from loguru import logger
import random
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
# from sklearn.utils import resample
from tqdm import tqdm
from pharm_ai.label.fda.predict import FDADT,Predictor
from pharm_ai.util.pdf_util.extractor_v3 import PDFTextExtractor, FieldTextExtractor
from pharm_ai.util.utils import Utilfuncs


class FDAExtractor:
    def __init__(self):
        self.two_column_pdf_fields_re = {
            'INDICATIONS AND USAGE': [r'\n\s*INDICATIONS AND USAGE\s*\n((?:.|\n)*?)\n\s*DOSAGE AND ADMINISTRATION',
                                      r'INDICATIONS AND USAGE((?:.|\n)*?)DOSAGE AND ADMINISTRATION'],
            'DOSAGE AND ADMINISTRATION': [r'\n\s*DOSAGE AND ADMINISTRATION\s*\n((?:.|\n)*?)\n\s*DOSAGE',
                                          r'DOSAGE AND ADMINISTRATION((?:.|\n)*?)DOSAGE FORMS AND STRENGTHS'],
            'Pediatric Use': [
                r'\n\s*(?:8.3|8.4.|8 . 4|8.4|8.5.|8.5)\s*\n*\s*(?:Pediatric Use|Pediatric Us e)\s*\n((?:.|\n)*?)'
                r'\n\s*(?:8.4|8.5.|8 . 5|8.5|8.6.|8.6|11)\s*\n*\s*(?:Geriatric Use|Geriatric Us e|Renal Impairment|DESCRIPTION|Hepatic Impairment)',
                r'\n\s*(?:8.4.|8.4|8.5.|8.5)\s*\n*\s*Pediatric Use((?:.|\n)*?)'
                r'\n\s*(?:8.5.|8.5|8.6.|8.6|11)\s*\n*\s*(?:Geriatric Use|Geriatric Us e|Renal Impairment|DESCRIPTION)',
                r'\n\s*(?:Pediatric Use.|• Pediatric use|Pediatric Use|Pediatric use|H. PEDIATRIC USE)((?:.|\n)*?)'
                r'\n\s*(?:Geriatric Use.|• Geriatric Use|Geriatric Use|See 17 for PATIENT COUNSELING INFORMATION|Renal Impairment|DESCRIPTION|ADVERSE REACTIONS|Use in the Elderly|Post-Marketing Experience|/I. GERIATRIC USE:|Animal Studies.)',


                # r'\n\s*(?:8.3|8.4)\s*\n*\s*Pediatric (?:Use|use)\s*\n((?:.|\n)*?)\n\s*(?:8.4|8.5|8.6)\s*\n*\s*(?:Geriatric Use|Renal Impairment)',
                #
                # r'\n\s*Pediatric (?:Use|use)\s*\n((?:.|\n)*?)\n\s*(?:Geriatric Use|ADVERSE REACTIONS)',
                # r'Pediatric Use\s*\n((?:.|\n)*?)Geriatric Use',
                # r'\n\s*Pediatric Use\s*\n((?:.|\n)*?)\n\s*Geriatric Use',
                # r'\n\s*\d{1,2}\s*\n*\s*[．.]\d{1,2}\s*\n*\s*Pediatric Use\s*\n((?:.|\n)*?)\n\s*\d{1,2}\s*\n*\s*[．.]\d{1,2}\s*\n*\s*(?:Geriatric Use|Renal Impairment)',
                # r'\n\s*\d{1,2}\s*\n*\s*[．.]\d{1,2}\s*\n*\s*Pediatric Use\s*\n((?:.|\n)*?)\n\s*\d{1,2}\s*\n*\s*OVERDOSAGE',
                # r'8.4\s*\n*\s*Pediatric Use((?:.|\n)*?)8.5\s*\n*\s*Geriatric Use'
            ],
            'HOW SUPPLIED': [
                r'\n\s*(?:16.|16|17)\s*\n*\s*(?:HOW SUPPLIED/STORAGE AND HANDLING|HOW SUPPLIED/ STORAGE AND HANDLING|HOW SUPPLIED / STORAGE AND HANDLING|HOW SUPPLIED/STORAGE HANDLING)\s*\n((?:.|\n)*?)'
                r'\n\s*(?:17.|17|18)\s*\n*\s*(?:PATIENT COUNSELING INFORMATION)',
                r'(?:HOW SUPPLIED/STORAGE AND HANDLING|HOW SUPPLIED/ STORAGE AND HANDLING)\s*\n((?:.|\n)*?)'
                r'(?:PATIENT COUNSELING INFORMATION|Storage|$)',

                # r'\n\s*\d{1,2}.*\s*\n*\s*HOW SUPPLIED/STORAGE AND HANDLING\s*\n((?:.|\n)*?)\n\s*d{1,2}.*\s*\n*\s*PATIENT COUNSELING INFORMATION',
                # r'16.*\s*\n*\s*HOW SUPPLIED/STORAGE AND HANDLING((?:.|\n)*?)17.*\s*\n*\s*PATIENT COUNSELING INFORMATION',
                # r'16.*\s*\n*\s*HOW SUPPLIED/STORAGE AND HANDLING((?:.|\n)*?)$'
            ],
        }

        self.one_column_pdf_fields_re = {
            'INDICATIONS AND USAGE': [
                r'\n\s*(?:INDICATIONS AND USAGE|INDICATIONS|INDICATION AND USAGE|Indications and Usage|Indications|INDICATION)\s*\n*\s*:*\s*\n((?:.|\n)*?)'
                r'\n\s*(?:CONTRAINDICATIONS|CONTRAINDICATION|Contraindications|WARNINGS)',
                r'\n\s*(?:INDICATIONS AND USAGE|INDICATIONS|INDICATION AND USAGE|Indications and Usage|Indications|INDICATION)\s*\n*\s*:*\s*((?:.|\n)*?)'
                r'\n\s*(?:CONTRAINDICATIONS|CONTRAINDICATION|Contraindications|WARNINGS)',

                # r'\n\s*INDICATIONS AND USAGE\s*\n*:*\s*\n((?:.|\n)*?)\n\s*CONTRAINDICATIONS',
                #                       r'\n\s*Indications and Usage\s*\n*:*\s*\n((?:.|\n)*?)\n\s*Contraindications',
                #                       r'\n\s*INDICATIONS\s*\n((?:.|\n)*?)\n\s*CONTRAINDICATIONS',
                #                       r'INDICATIONS AND USAGE((?:.|\n)*?)CONTRAINDICATIONS',
                #                       r'Indications and Usage((?:.|\n)*?)Contraindications',
                #                       r'INDICATIONS((?:.|\n)*?)CONTRAINDICATIONS',
            ],
            'DOSAGE AND ADMINISTRATION': [
                r'\n\s*(?:DOSAGE AND ADMINISTRATION \(See WARNINGS\)|DOSAGE AND ADMINISTRATION|IX. DRUG DOSAGE & ADMINISTRATION|Dosage and Administration)\s*\n*\s*:*\s*\n((?:.|\n)*?)'
                r'\n\s*(?:HOW SUPPLIED|How Supplied|X. UVA RADIATION SOURCE SPECIFICATIONS & INFORMATION|IMURAN|OVERDOSAGE)',
                r'\n\s*(?:DOSAGE AND ADMINISTRATION \(See WARNINGS\)|DOSAGE AND ADMINISTRATION|IX. DRUG DOSAGE & ADMINISTRATION|Dosage and Administration)\s*\n*\s*:*\s*((?:.|\n)*?)'
                r'\n\s*(?:HOW SUPPLIED|How Supplied|X. UVA RADIATION SOURCE SPECIFICATIONS & INFORMATION|IMURAN|OVERDOSAGE)',

                # r'\n\s*DOSAGE AND ADMINISTRATION\s*\n((?:.|\n)*?)\n\s*HOW SUPPLIED',
                #                           r'\n\s*IX. DRUG DOSAGE & ADMINISTRATION\s*\n((?:.|\n)*?)\n\s*X. UVA RADIATION SOURCE SPECIFICATIONS & INFORMATION',
                #                           r'DOSAGE AND ADMINISTRATION((?:.|\n)*?)HOW SUPPLIED',
                #                           r'\n\s*DOSAGE AND ADMINISTRATION((?:.|\n)*?)\n\s*How Supplied',
            ],
            'Pediatric Use': [
                r'\n\s*(?:H. Pediatric Use|9. Pediatric Use|13. Pediatric Use|15.Pediatric Use|10 Pediatric Use|17  Pediatric Use|Pediatric Use|Pediatric use|H. PEDIATRIC USE|14. PEDIATRIC USE|12. PEDIATRIC USE|Pediatric Patients)\s*\n*\s*:*\s*\n((?:.|\n)*?)'
                r'\n\s*(?:I. Geriatric Use|10. Geriatric Use|14. Geriatric Us|16.Geriatric Use|11 Geriatric Use|Geriatric Use|ADVERSE REACTIONS|Use in the Elderly|Post-Marketing Experience|/I. GERIATRIC USE:|INFORMATION|Geriatric Patients|DRUG INTERACTIONS)',
                r'\n\s*(?:Pediatric Use.|Pediatric Use|Pediatric use|H. PEDIATRIC USE|PEDIATRIC USE)((?:.|\n)*?)'
                r'\n\s*(?:Geriatric Use|ADVERSE REACTIONS|Use in the Elderly|Post-Marketing Experience|/I. GERIATRIC USE:|Animal Studies.)',
                # r'\n\s*Pediatric (?:Use|use)\s*\n((?:.|\n)*?)\n\s*(?:Geriatric Use|ADVERSE REACTIONS)',
                #               r'\n\s*Pediatric Use((?:.|\n)*?)\n\s*(?:Geriatric Use|ADVERSE REACTIONS)',
                #               r'\n\s*(?:8.3|8.4)*\s*\n*\s*Pediatric Use\s*\n((?:.|\n)*?)\n\s*(?:8.4|8.5|)*Geriatric Use',
                #               r'Pediatric Use:*\s*\n((?:.|\n)*?)Geriatric Use/ADVERSE REACTIONS',
                #               r'Pediatric Use:*\s*\n((?:.|\n)*?)(?:Geriatric Use/ADVERSE REACTIONS|Geriatric Use)',
                #               r'Pediatric Use\s*\n((?:.|\n)*?)(?:Geriatric Use|Use in the Elderly|Post-Marketing Experience)',
                #               r'Pediatric [U|u]se\s*\n((?:.|\n)*?)ADVERSE REACTIONS',
                #               r'Pediatric Use\s*\n((?:.|\n)*?)$',
                #               r'Pediatric Use((?:.|\n)*?)\n\s*Geriatric Use',
            ],
            'HOW SUPPLIED': [
                r'\n\s*(?:HOW SUPPLIED/STORAGE AND HANDLING|HOW SUPPLIED|How Supplied)\s*:*\s*\n((?:.|\n)*?)'
                r'\n\s*(?:REFERENCE|Medication Guide|PATIENT INFORMATION|ANIMAL PHARMACOLOGY|MEDICATION GUIDE|PATIENT|Directions for Use|References|$)',
                r'\n\s*(?:HOW SUPPLIED/STORAGE AND HANDLING|HOW SUPPLIED|How Supplied)\s*:*\s*((?:.|\n)*?)'
                r'\n\s*(?:REFERENCE|Medication Guide|PATIENT INFORMATION|ANIMAL PHARMACOLOGY|MEDICATION GUIDE|PATIENT|Directions for Use|References|$)',

                # r'(?:HOW SUPPLIED/STORAGE AND HANDLING|HOW SUPPLIED)((?:.|\n)*?)(?:REFERENCE|Medication Guide|PATIENT INFORMATION|ANIMAL PHARMACOLOGY)',
                #              r'(?:HOW SUPPLIED|How Supplied|HOW SUPPLIED/STORAGE AND HANDLING)((?:.|\n)*?)$',
            ],
        }
        self.p = Predictor()

    def test(self):
        # f = "./data/pdf/f4759f954f31c003ab1225d0405d08dc.pdf"
        # raw_text = FDAExtractor.extract_raw_text_for_fda(f)
        # raw_text = FDAExtractor.handle_raw_text(raw_text)
        # print(repr(raw_text))
        # field_text = FDAExtractor().extract_field_for_fda(raw_text)
        # print(field_text)
        # #############################
        self.refine_and_save()
        pass

    def refine_and_save(self):
        f = '/home/zyl/disk/PharmAI/pharm_ai/label/fda/data/pdf/'
        t = FDAExtractor.extract_texts_from_pdfs_dir(f, pdf_num=0)
        df = pd.DataFrame(t)
        t1 = df['Pediatric Use'].tolist()
        t1 = [FDADT.handle_text(i) for i in t1]
        # print(t1)
        r1 = self.p.predict(t1)
        df['Pediatric Use_nlp'] = r1

        df.to_excel('./data/pdfs_raw_text.xlsx')
        ###############################################
        # df = pd.read_excel('./data/pdfs_raw_text.xlsx')
        # df['INDICATIONS AND USAGE'] = df['INDICATIONS AND USAGE'].apply(
        #     lambda x: FDAExtractor.handle_field_text(str(x)) if len(str(x)) > 1 else str(x))
        # df['DOSAGE AND ADMINISTRATION'] = df['DOSAGE AND ADMINISTRATION'].apply(
        #     lambda x: FDAExtractor.handle_field_text(str(x)) if len(str(x)) > 1 else str(x))
        # df['Pediatric Use'] = df['Pediatric Use'].apply(
        #     lambda x: FDAExtractor.handle_field_text(str(x)) if len(str(x)) > 1 else str(x))
        # df['HOW SUPPLIED'] = df['HOW SUPPLIED'].apply(
        #     lambda x: FDAExtractor.handle_field_text(str(x)) if len(str(x)) > 1 else str(x))
        # df.to_excel('./data/pdfs_text.xlsx')

        # # df = pd.read_excel('./data/pdfs_text.xlsx')  # type:pd.DataFrame
        # df = df.astype('str')
        # test_df = resample(df, replace=False, n_samples=200)
        # test_df = test_df[
        #     ['id', 'type', 'INDICATIONS AND USAGE', 'DOSAGE AND ADMINISTRATION', 'Pediatric Use', 'HOW SUPPLIED']]
        # test_df['label'] = ''
        # test_df.to_excel('./data/test_fda.xlsx')

    @staticmethod
    def extract_texts_from_pdfs_dir(pdfs_dir, pdf_num=0):
        # extract pdf texts by using pdfs_dir,return a list like [{file_name:file_text}]
        pdf_paths = Utilfuncs.list_all_files(pdfs_dir, 'pdf')  # list all pdf files
        if pdf_num != 0:
            pdf_paths = random.sample(pdf_paths, pdf_num)
        logger.info('pdfs num: ' + str(len(pdf_paths)))
        fields_texts = []
        for pdf in tqdm(pdf_paths):
            raw_text = FDAExtractor.extract_raw_text_for_fda(pdf)
            raw_text = FDAExtractor.handle_raw_text(raw_text)
            if raw_text != 'error pdf':
                field_text = FDAExtractor().extract_field_for_fda(raw_text)
                pdf_name = ntpath.basename(pdf)  # pdf file name
                field_text.update({'pdf_name': pdf_name})
                fields_texts.append(field_text)
            else:
                pdf_name = ntpath.basename(pdf)  # pdf file name
                fields_texts.append({'pdf_name': pdf_name,
                                     'type': 'error pdf',
                                     'INDICATIONS AND USAGE': '',
                                     'DOSAGE AND ADMINISTRATION': '',
                                     'Pediatric Use': '',
                                     'HOW SUPPLIED': ''})

        return fields_texts  # type:list[dict]

    @staticmethod
    def extract_raw_text_for_fda(pdf):
        raw_text = ''
        try:
            pdf_doc = fitz.Document(pdf)
            two_column = 0
            pages = pdf_doc.page_count
            for pg in range(0, pages):
                pdf_page = pdf_doc.load_page(pg)
                clip = fitz.Rect(0, 0, pdf_page.rect.width, pdf_page.rect.height)
                text = pdf_page.get_textpage(clip=clip).extractText()
                if ('HIGHLIGHTS OF PRESCRIBING INFORMATION' in text) or ('Highlights Of Prescribing Information' in text):
                    two_column = 1
                elif 'HIGHLIGHTS OF PRESCRIBING' in text:
                    two_column = 1
                if two_column == 1:
                    l_t, r_t = PDFTextExtractor.extract_text_in_a_double_column_page(pdf_page=pdf_page)
                    # for l_t
                    if ('FULL PRESCRIBING INFORMATION: CONTENTS' in l_t) or ('FULL PRESCRIBING INFORMATION' in l_t):
                        l_t = l_t.split('FULL PRESCRIBING INFORMATION')[0]
                        two_column = 0

                    # for r_t
                    if 'Revised:' in r_t:
                        r_t = r_t.split('Revised')[0]
                        two_column = 0

                    text = l_t + r_t
                raw_text += text
        except Exception as failed_e:
            logger.error(f'{pdf} : {failed_e}')
            return 'error pdf'
        return raw_text  # type:str

    @staticmethod
    def handle_raw_text(text: str):
        text = ILLEGAL_CHARACTERS_RE.sub(r'', str(text))
        double_chars = [' ',  '_', '∙', '�','-','—']
        for i in double_chars:
            if i + i in text:
                text = re.sub((i+i) + r'+', ' ', text)

        if '..' in text:
            text = re.sub('\.+', ' ', text)
        if ' \n \n' in text:
            text = re.sub('( \n)+', '\n', text)
        if '\n\n' in text:
            text = re.sub('\n+', ' \n ', text)
        if '- ' in text:
            text = re.sub('(- )+', '- ', text)
        if '\t' in text:
            text = re.sub('\t', ' ', text)
        char_list = ['\xad', '\uf0b7', '\uf8e7', '\xa0', '\ue05d', '\uf0d4', '\uf0b7', '\xa0', '\uf020',
                     '\uf6da']  # '\n', '\t',
        for c in char_list:
            if c in text:
                text = text.replace(c, ' ')
        text = FDAExtractor.remove_footer(text)
        return text

    def extract_field_for_fda(self, raw_text: str):
        res = dict()
        if ('HIGHLIGHTS OF PRESCRIBING' in raw_text) or ('Highlights Of Prescribing Information' in raw_text):
            res.update({'type': 'two_column'})
            for field_name, field_re in self.two_column_pdf_fields_re.items():
                if field_name in ('INDICATIONS AND USAGE', 'DOSAGE AND ADMINISTRATION'):
                    r = FieldTextExtractor.extract_field_text_by_re(raw_text, field_re)

                    while '' in r:
                        r.remove('')
                    if not r:
                        res.update({field_name: ''})
                    else:

                        for i in r:
                            i = FDAExtractor.handle_field_text(i)
                            # print(i)
                            if re.findall('\D', i):
                                res.update({field_name: i})
                                break
                        # res.update({field_name: r[0]})
                else:
                    r = FieldTextExtractor.extract_field_text_by_re(raw_text, field_re)
                    while '' in r:
                        r.remove('')
                    if not r:
                        res.update({field_name: ''})
                    else:
                        for i in r:
                            i = FDAExtractor.handle_field_text(i)
                            if field_name == 'Pediatric Use':
                                for r in [r'Pediatric Use']:
                                    if re.search(r, i):
                                        i = i.split(r)[-1]
                            if field_name == 'HOW SUPPLIED':
                                for r in [r'HOW SUPPLIED/STORAGE AND HANDLING']:
                                    if re.search(r, i):
                                        i = i.split(r)[-1]

                            if re.findall('\D', i):
                                res.update({field_name: i})
                                # break
                        # res.update({field_name: r[-1]})
        else:
            for field_name, field_re in self.one_column_pdf_fields_re.items():
                r = FieldTextExtractor.extract_field_text_by_re(raw_text, field_re)
                while '' in r:
                    r.remove('')
                if not r:  # []==False
                    res.update({field_name: ''})
                else:
                    for i in r:
                        i = FDAExtractor.handle_field_text(i)
                        if re.findall('\D', i):
                            res.update({field_name: i})
                            break
                    # res.update({field_name: r[0]})
            if res['INDICATIONS AND USAGE'] == '' and res['DOSAGE AND ADMINISTRATION'] == '' and res[
                'Pediatric Use'] == '' and res['HOW SUPPLIED'] == '':
                res.update({'type': 'problematic pdf'})
            else:
                res.update({'type': 'other'})
        return res  # dict,{field_name:field_text}

    @staticmethod
    def remove_footer(text):
        footer_re = [r'\n \d* \n Reference ID: \d{7}',
                     r'Page \d* of \d* \n* Reference ID: \d{7}',
                     r'\n Reference ID: \d{7}']
        for i in footer_re:
            text = re.sub(i, '', text)
        return text

    @staticmethod
    def handle_field_text(text):
        text = text.strip()
        if text != '' and text[0] == '-':
            text = text[1:]
        if text != '' and text[-1] == '-':
            text = text[0:-1]
        text = re.sub(' +', ' ', text)
        while re.findall(r'^[\s:\n)]', text) != []:
            text = text.strip()
            text = text.strip(':')
            text = text.strip('\n')
            text = text.strip(')')
        # re_replace = [r'Pediatric Use']
        # for r in re_replace:
        #     print()
        #     if re.search(r,text):
        #         print(r)
        #         print('True')
        #         text = text.split(r)[-1]
        #         print(text)
        return text


if __name__ == '__main__':
    FDAExtractor().test()
