# -*- coding: UTF-8 -*-
"""
Description : 
"""
import pandas as pd
from pharm_ai.util.utils import Utilfuncs
import re


class PMDADT:
    def __init__(self):
        pass

    @staticmethod
    def handle_text(text):
        if pd.isna(text):
            return ''
        text = Utilfuncs.remove_illegal_chars(text)
        char_list = ['\n', '\u200c', ]
        for i in char_list:
            if i in text:
                text = re.sub(i, '', text)
        return text

    def dt_0622(self):
        df = pd.read_excel("./data/pmda-002-pdf-2.xlsx", "raw_text")
        t1 = df['用法及び用量'].tolist()
        t1 = [PMDADT.handle_text(i) for i in t1]
        print(t1)


if __name__ == '__main__':
    PMDADT().dt_0622()
