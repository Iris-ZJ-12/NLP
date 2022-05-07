import pdfplumber
from pharm_ai.util.utils import Utilfuncs as u
import pandas as pd
from pharm_ai.config import ConfigFilePaths as cfp
import os
import shutil
from loguru import logger
from pathlib import Path
from os import path


class Extractor:
    def __init__(self):
        self.d = cfp.project_dir + '/pet/temp/'
        if not path.exists(self.d):
           # self.d = Path(self.d)
           os.mkdir(self.d)

    def extract_tables(self, file, file_name, excel_path):

        p = self.d + file_name

        with open(p, "wb") as buffer:
            shutil.copyfileobj(file, buffer)

        pdf = pdfplumber.open(p)
        count = 1
        for page in pdf.pages:
            for table in page.extract_tables():
                df = pd.DataFrame(table)
                df = df.applymap(str)
                u.to_excel(df, excel_path, str(count))
                count += 1

    def rm(self, path):
        shutil.rmtree(path)
        logger.info(f'{path} was deleted.')


if __name__ == '__main__':
    e = Extractor()
    dir = cfp.project_dir + '/pet/temp'
    for f in os.listdir(dir):
        if f.endswith('.pdf') or f.endswith('.PDF'):
            p = os.path.join(dir, f)
            print(p)
            excel = p[:-4] + '.xlsx'
            print(excel)
            e.extract_tables(p, excel)