# encoding: utf-8
'''
@author: zyl
@file: analyze_dt.py
@time: 2021/11/9 17:47
@desc:
'''
import numpy as np
import pandas as pd
from transformers.models.t5 import T5Tokenizer
import matplotlib.pyplot as plt
class AnalyzeDT:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('/large_files/pretrained_pytorch/mt5-base/', truncate=True)

    def run(self):
        pass

    def get_tokens_len(self,text):
        tokens = self.tokenizer.tokenize(str(text))
        return len(tokens)

    def test(self):
        pass

    def draw_pie(self,df,column):
        s = df[column].apply(lambda x: 'None' if pd.isna(x) else x).value_counts().to_dict()
        labels = []
        x = []
        for k,v in s.items():
            labels.append(k)
            x.append(v)

        plt.subplot()
        plt.pie(x=x,labels=labels)
        plt.show()

    def draw_hist(self,df,column):
        plt.subplot()
        plt.hist(df[column], bins=500)
        plt.show()

    def draw_box(self,df,column):
        plt.subplot()
        plt.boxplot(df[column])
        plt.show()

    def get_base_info(self,df):
        desc = df.describe(percentiles=[.0, .25, .5, .75, 1])
        info = df.info()
        print(desc)
        print(info)


class TTTDT:
    def __init__(self):
        self.analyzer = AnalyzeDT()

    def run(self):
        self.dt_1110()
        pass

    def dt_1110(self):
        df = pd.read_excel('./data/v1/processed_1109.xlsx', 'all')
        df['criteria_token_len'] = df['criteria'].apply(self.analyzer.get_tokens_len)
        df['title_token_len'] = df['study_title'].apply(self.analyzer.get_tokens_len)

        self.analyzer.get_base_info(df)
        self.analyzer.draw_hist(df,column='criteria_token_len')
        self.analyzer.draw_box(df, column='criteria_token_len')
        self.analyzer.draw_pie(df,column='therapy_labels')

        self.analyzer.draw_hist(df, column='title_token_len')
        self.analyzer.draw_box(df, column='title_token_len')


if __name__ == '__main__':

    TTTDT().run()

