# -*- coding: UTF-8 -*-
"""
Description : 
"""

import pandas as pd
from pharm_ai.util.utils import Utilfuncs
import re

class FDADT:
    def __init__(self):
        pass

    @staticmethod
    def handle_text(text):
        if pd.isna(text):
            return ''
        text = Utilfuncs.remove_illegal_chars(text)
        char_list = ['\n','\u200c','\uf09e']
        for i in char_list:
            if i in text:
                text = re.sub(i,'',text)
        if '  ' in text:
            text = re.sub(' +',' ',text)
        if '.' in text:
            text = '.'.join(text.split('.')[0:-1])+'.'
        else:
            text = text+'.'
        # print(repr(text))
        return text

    def dt_0622(self):
        df = pd.read_excel("./data/pdfs_raw_text.xlsx")
        t1 = df['INDICATIONS AND USAGE'].tolist()
        t1 = df['DOSAGE AND ADMINISTRATION'].tolist()
        t1 = df['Pediatric Use'].tolist()
        t1 = [FDADT.handle_text(i) for i in t1]
        # for i in t1:
        #     print(repr(i))
        # print(t1)

if __name__ == '__main__':
    FDADT().dt_0622()