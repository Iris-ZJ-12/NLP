import os
from pharm_ai.util.pdf_util.extractor_v2 import Extractor
import pandas as pd
from tqdm import tqdm
import random
from loguru import logger
import re
# from pharm_ai.if_fields_extractor.predictor import Predictor
from tempfile import TemporaryDirectory

class PdfExtractor:
    def __init__(self, version='v1.0', cuda_device=-1):
        self.pdf_path = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-001-pdf-2/"
        self.pdfs = os.listdir(self.pdf_path)
        self.pdf_extractor = Extractor()
        self.fields = ['formulation']
        self.random_state = 1130
        self.patterns = {
            "version": r'(\d{4}年\d{1,2}月[\(（]?(?:改訂|作成)[（\(]?.*?[\)）])',
            "product_no": r'日本標準商品分類番号[：:]?(\d+)',
            "inn_jp": r'和名[：:](.*)',
            "inn_en": r'洋名[：:](.*)',
            "application_key": r'開発・製造販売[（\(]輸入[）\)]',
            "application": r"(?:製造)?(?:販|発)売(?:元)?[：:](?:\n)?(.*)",
            "indication": r'\D\d[\.． ](?:効能又は効果|効能・効果)\n*([\D\n]?(?:.|\n)*?)(?:\d[\.．]\D|.*効能又は効果|.*効能・効果)',
            "approval_no": r'\d{3,}[A-Z]{3}\d{3,}',
            "geriatric_use": [r'\d{1,2}[．.]高齢者への(?:投与|使用)((?:.|\n)*?)\d{1,2}[．.]妊婦[、，]産婦[、，]授乳妊?婦',
                              r'(\(\d{1,2}\)高齢者((?:.|\n)*?)\d{1,2}[．.]\D)'],
            "pregnant_use": [r'(?:\d{1,2}[．.]妊婦[、，]産婦[、，]授乳婦等への投与((?:.|\n)*?)\d{1,2}[．.]小児等への(?:投与|使用))',
                             r'(?:\d{1,2}[．.]妊婦[、，]産婦[、，]授乳婦等への投((?:.|\n)*?)\d{1,2}[．.]小児等への(?:投与|使用))',
                             r'(?:\d{1,2}[．.]妊婦[、，]産婦[、，]授乳婦等への((?:.|\n)*?)\d{1,2}[．.]小児等への(?:投与|使用))',
                             r'(?:\d{1,2}[．.]妊婦[、，]産婦[、，]授乳婦等へ((?:.|\n)*?)\d{1,2}[．.]小児等への(?:投与|使用))',
                             r'(?:\d{1,2}[．.]妊婦[、，]産婦[、，]授乳婦等((?:.|\n)*?)\d{1,2}[．.]小児等への(?:投与|使用))',
                             r'(?:\d{1,2}[．.]妊婦[、，]産婦[、，]授乳婦((?:.|\n)*?)\d{1,2}[．.]小児等への(?:投与|使用))',
                             r'((\(\d{1,2}\)妊婦(?:.|\n)*?)\(\d{1,2}\)小児等)',
                             r'(?:\d{1,2}[．.]妊婦[、，]産婦[、，]授乳((?:.|\n)*?)\d{1,2}[．.]小児等への(?:投与|使用))',],
            "geriatric_use_refine": r'\d{1,2}[．.]高齢者への(?:投与|使用)',
            "pregnant_use_refine": r'(?:\d{1,2}[．.]妊婦[、，]産婦[、，]授乳婦(?:等への投与|等への投|等への|等へ|等)?)'
        }
        self.page_number_pattern=r'\s*[-－]\s*\d{1,3}\s*[-－]\s*'
        # self.drug_category_predictor = Predictor(version=version, cuda_device=cuda_device)
        random.seed(self.random_state)
        logger.add("extractor.log")

    def preprocess_pdf(self, pdf_file):
        """
        :param str pdf_file: PDF file path.
        :return:

        Preprocess single PDF.
        """
        raw_first_texts = self.pdf_extractor.extract_raw_full_text(pdf_file, end_page_n=2)
        raw_content_texts = self.pdf_extractor.extract_raw_full_text(pdf_file, start_page_n=6) # avoid contents pages
        raw_content_texts = self.remove_page_header_footer(raw_content_texts)
        raw_tables = self.pdf_extractor.extract_raw_table(pdf_file, end_page=2)
        raw_content_tables = self.pdf_extractor.extract_raw_table(pdf_file, start_page=6)
        formulation = self.extract_formulation(raw_tables)
        version, product_no = self.extract_version_productNo(raw_first_texts)
        drug_category = self.extract_drug_categrory(raw_first_texts, raw_tables)
        otc = self.extract_otc(raw_tables)
        inn_jp, inn_en = self.extract_inn(raw_tables)
        application = self.extract_application(raw_tables)
        indication, geriatric_use, pregnant_use = self.extract_indication_use(raw_content_texts, raw_content_tables)
        approval_no = self.extract_approval_no(raw_content_texts)
        # return {'res1': geriatric_use, 'res2':pregnant_use}
        return {'formulation': formulation,
                'version': version,
                'product_no': product_no,
                'drug_category': drug_category,
                'otc': otc,
                'inn_jp': inn_jp,
                'inn_en': inn_en,
                'application': application,
                'indication': indication,
                'approval_no': approval_no,
                'geriatric_use': geriatric_use,
                'pregnant_use': pregnant_use}

    def extract_formulation(self, raw_tables):
        # find formulation from the first 2 tables
        try:
            first_tables = [[column.replace(' ','') if column else None for column in row] for table in raw_tables[:2] for row in table]
            raw_formulation = [row[1] for row in first_tables if row[0]=='剤形']
            formulation = raw_formulation[0] if raw_formulation else None
            return formulation
        except Exception as ex:
            logger.error(ex)
            return ""

    def extract_version_productNo(self, raw_fulltext:str):
        try:
            first_line = ''.join(raw_fulltext.split('\n')[:3])
            first_line_short = self.change_number_font(first_line.replace(' ',''))
            re_version = re.search(self.patterns['version'], first_line_short)
            re_productNo = re.search(self.patterns['product_no'], first_line_short)
            version = re_version.groups()[0] if re_version else ""
            product_no = re_productNo.groups()[0] if re_productNo else ""
            return version, product_no
        except Exception as ex:
            logger.error(ex)
            return "",""

    def extract_drug_categrory(self, raw_fulltest, raw_tables):
        try:
            texts = self.get_fulltext_before_table(raw_fulltest, raw_tables)
            lines = []
            for line in texts:
                line = self.change_number_font(line)
                if re.search(r'\d',line):
                    continue
                if re.fullmatch(r'[a-z A-Z]+',line):
                    continue
                if '処方箋医薬品' in line or '日本薬局方' in line:
                    continue
                if '医薬品' in line or '分類番号' in line:
                    continue
                lines.append(line)
            if len(lines)==0:
                res = ""
            elif len(lines)==1:
                res = lines[0]
            else:
                res = self.drug_category_predictor.predict(lines)
            return res
        except Exception as ex:
            return ""

    def preproces_pdfs(self, to_extract_pdfs: list=None, sample_size: int=None, excel_name=None):
        if to_extract_pdfs is None:
            pdfs = random.choices(self.pdfs, k=sample_size) if sample_size is not None else self.pdfs
        else:
            pdfs = to_extract_pdfs
        pbar = tqdm(pdfs)
        results = []
        for pdf in pbar:
            pbar.set_description("Extracting: {}".format(pdf))
            sample_pdf = os.path.join(self.pdf_path, pdf)
            res = self.preprocess_pdf(sample_pdf)
            res.update({"PDF_name": pdf})
            results.append(res)
        res_df = pd.DataFrame(results)
        if excel_name is None:
            return res_df
        else:
            res_df[['inn_jp', 'inn_en', 'application', 'approval_no']] = res_df[
                ['inn_jp', 'inn_en', 'application', 'approval_no']].applymap(
                lambda list_: '\n'.join(list_) if isinstance(list_, list) else list_)
            res_df.to_excel(excel_name)

    def preprocess_training_data(self, df:pd.DataFrame = None, excel_name=None, output=None):
        """
        :param pandas.DataFrame df: input result dataframe
        :param excel_name: input result excel path.
        :param str output: output result saved excel path.
        :return: result dataframe if no output path specified, else result saved to excel name.
        """
        if excel_name and not df:
            df = pd.read_excel(excel_name)
            df['drug_category_content'] = df['drug_category_content'].map(lambda s:[s_.strip("''") for s_ in s.strip('[]').split(', ')])
        df_ = df[['PDF_name', 'drug_category_content']]
        content_len = df_['drug_category_content'].map(len)
        drug_category = df_['drug_category_content'].map(lambda s:s[0]).pipe(lambda s:s.mask((s=='')|(content_len!=1)))
        df_['drug_category'] = drug_category
        res = df[['PDF_name']].join(df_['drug_category_content'].pipe(lambda s:s.mask(content_len<2))).dropna()
        res = res.explode('drug_category_content')
        res['labels'] = None
        if output is not None:
            res.to_excel(output)
        else:
            return res


    def change_number_font(self, num_chrs:str):
        """
        example:
        # >>>num_chrs = '２０１９'
        # >>>extractor = PdfExtractor()
        # >>>extractor.change_number_font(num_chrs)
        '2019'
        """
        for ch in num_chrs:
            if 65296<=ord(ch)<=65305:
                num_chrs = num_chrs.replace(ch, chr(ord(ch)-65248))
        return num_chrs

    def get_fulltext_before_table(self, raw_fulltext, raw_tables):
        lines_text = [t.replace(' ','') for t in raw_fulltext.split('\n')]
        lines_table = [''.join(w.replace(' ','') for w in line if w) for line in raw_tables[0]]
        res_list = []
        for i, line in enumerate(lines_text):
            if line.find(lines_table[0])>-1 or lines_table[0].find(line)>-1:
                if i>1:
                    res_list = lines_text[1:i]
                    break
        return res_list

    def extract_otc(self, raw_table):
        res = self.get_table_value_by_key(raw_table, '製剤の規制区分')
        return res

    def get_table_value_by_key(self, raw_tables, key, use_re = False):
        # first_tables = [[column.replace(' ', '') if column else None for column in row] for table in raw_tables[:2] for
        #                 row in table]
        # raw_formulation = [row[1] for row in first_tables if row[0] == '剤形']
        # formulation = raw_formulation[0] if raw_formulation else None
        first_tables = [[column.replace(' ','') if column else None for column in row]
                        for table in raw_tables[:2] for row in table]
        if not use_re:
            extracts = [row[1] if len(row)>1 else None for row in first_tables if row[0] == key]
            res = extracts[0] if extracts else None
        else:
            res = None
            for row in first_tables:
                for ind, cell in enumerate(row):
                    if cell:
                        r = re.match(key, cell.replace('\n',''))
                        if r:
                            res = row[ind+1] if ind+1<len(row) else None
                            break
                    else:
                        continue
        return res

    def extract_inn(self, raw_tables):
        inn_raw = self.get_table_value_by_key(raw_tables, '一般名')
        if inn_raw:
            inn_list = inn_raw.split('\n')
            res_jp, res_en = [], []
            for line in inn_list:
                r1 = re.search(self.patterns['inn_jp'], line)
                r2 = re.search(self.patterns['inn_en'], line)
                if not r1 and not r2:
                    res_jp.append(line)
                    res_en.append(line)
                    continue
                if r1:
                    res_jp.append(r1.groups()[0])
                if r2:
                    res_en.append(r2.groups()[0])
            return res_jp, res_en
        else:
            return [],[]

    def extract_application(self, raw_tables):
        extracts = self.get_table_value_by_key(raw_tables, self.patterns['application_key'],
                                               use_re=True)
        if extracts:
            re_res = re.findall(self.patterns['application'], extracts)
            res = [r_ for r_ in re_res if r_]
        else:
            res = None
        return res

    def extract_indication_use(self, raw_fulltext, raw_tables):
        fulltext = raw_fulltext.replace(' ','')
        fulltext = self.change_number_font(fulltext)
        r_indication = re.search(self.patterns['indication'], fulltext)
        r_geriatric = None
        for gpattern in self.patterns['geriatric_use']:
            r_geriatric = re.search(gpattern, fulltext)
            if r_geriatric:
                break
        r_pregnant = None
        for ppattern in self.patterns['pregnant_use']:
            r_pregnant = re.search(ppattern, fulltext)
            if r_pregnant:
                break
        res_indication = r_indication.groups()[0].strip('\n') if r_indication else None
        if r_geriatric:
            extract_geriatric = [r_ for r_ in r_geriatric.groups() if r_]
            res_geriatric_extract = extract_geriatric[0].strip('\n') if extract_geriatric else None
            if res_geriatric_extract:
                geriatric_fintuned = re.split(self.patterns['geriatric_use_refine'], res_geriatric_extract)
                res_geriatric = geriatric_fintuned[-1].strip('\n')
            else:
                res_geriatric = None
        else:
            res_geriatric = self.get_table_value_by_key(raw_tables, '高齢者への投与')
        if r_pregnant:
            extract_pregnant = [r_ for r_ in r_pregnant.groups() if r_]
            res_pregnant_extract = extract_pregnant[0].strip('\n') if extract_pregnant else None
            if res_pregnant_extract:
                pregnant_fintuned = re.split(self.patterns['pregnant_use_refine'], res_pregnant_extract)
                res_pregnant = pregnant_fintuned[-1].strip('\n')
            else:
                res_pregnant = None
        else:
            res_pregnant = self.get_table_value_by_key(raw_tables, '妊婦、産婦、授乳婦')
        return res_indication, res_geriatric, res_pregnant

    def extract_approval_no(self, raw_fulltext):
        fulltext = self.change_number_font(raw_fulltext.replace(' ',''))
        res = re.findall(self.patterns['approval_no'], fulltext)
        return res

    def extract_from_zip(self, zip_file, saved_excel=None):
        with TemporaryDirectory(prefix='extract_', dir='raw_data') as tp:
            self.pdf_extractor.unzip(zip_file, tp)
            pdfs = os.listdir(tp)
            if saved_excel is None:
                result = self.preproces_pdfs(pdfs)
                return result
            else:
                self.preproces_pdfs(pdfs, excel_name=saved_excel)

    def remove_page_header_footer(self, fulltext):
        lines_raw = fulltext.split('\n')
        footers_list = [(footer_ind, line) for footer_ind, line in enumerate(lines_raw) if
                 re.fullmatch(self.page_number_pattern, line)]
        if footers_list:
            footer_pos, footer_words = zip(*footers_list)
            first_lines_list = [(ind + 1, next_.strip()) for (ind, line), next_ in
                           zip(enumerate(lines_raw), lines_raw[1:] + [""]) if
                           ind in footer_pos and ind < len(lines_raw) - 1]
            first_lines_ind, first_lines = zip(*first_lines_list)
            common_lines = [l for l in set(first_lines) if first_lines.count(l)>1]
            header_pos = [ind_ for ind_, l in zip(first_lines_ind, first_lines) if l in common_lines]
            lines_res = [line for ind, line in enumerate(lines_raw) if ind not in list(footer_pos)+header_pos]
            result = '\n'.join(lines_res)
        else:
            result = fulltext
        return result
from pharm_ai.util.utils import Utilfuncs
import ntpath
extractor = PdfExtractor()

def run_indication(pdfs_dir,to_save='test'):
    pdf_paths = Utilfuncs.list_all_files(pdfs_dir, 'pdf')  # list all pdf files
    # pdf_paths = random.sample(pdf_paths, 10)
    res = []
    fs = []
    for pdf in tqdm(pdf_paths):
        res.append(extractor.preprocess_pdf(pdf)['indication'])
        fs.append(ntpath.basename(pdf))  # pdf file name
    d = pd.DataFrame()
    d['pdf'] = fs
    d['indications'] = res
    d.to_excel(f"/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/indications/{to_save}.xlsx")
    d.to_excel('./test.xlsx')
    # return fields_texts  # type:list[dict]


if __name__ == '__main__':
    d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-001-pdf-2/"
    run_indication(d,to_save='pmda-001-pdf-2')
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-004-pdf/"
    # run_indication(d, to_save='pmda-004-pdf')
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-003-pdf/"
    # run_indication(d, to_save='pmda-003-pdf')
    # d = "/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/pmda-002-pdf-2/"
    # run_indication(d, to_save='pmda-002-pdf-2')

