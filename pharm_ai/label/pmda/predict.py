# -*- coding: UTF-8 -*-
"""
Description : 
"""
from scipy.special import softmax
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel
import torch
import pandas as pd
# from pharm_ai.package_insert_extractors.ped_extractors.pmda_if.dt import train_test_20201020
import numpy as np
from pprint import pprint
from pharm_ai.config import ConfigFilePaths as cfp
import requests
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.label.pmda.pmda_dt import PMDADT


class Predictor:
    def __init__(self):
        f = '/home/zyl/disk/PharmAI/pharm_ai/label/pmda/'
        cd1 = f + 'cache/20201020/1/'
        bm1 = f + 'best_model/20201020/1/'
        self.model1 = ClassificationModel('bert', bm1, num_labels=2, use_cuda=True,
                                          args={'reprocess_input_data': True,
                                                'use_cached_eval_features': True,
                                                'overwrite_output_dir': True,
                                                'n_gpu': 1,
                                                'fp16': False,
                                                'use_multiprocessing': False,
                                                'output_dir': bm1, 'cache_dir': cd1,
                                                'best_model_dir': bm1})
        cd2 = f + 'cache/20201020/2/'
        bm2 = f + 'best_model/20201020/2/'
        self.model2 = ClassificationModel('bert', bm2, num_labels=2, use_cuda=True,
                                          args={'reprocess_input_data': True,
                                                'use_cached_eval_features': True,
                                                'overwrite_output_dir': True,
                                                'n_gpu': 1,
                                                'fp16': False,
                                                'use_multiprocessing': False,
                                                'cache_dir': cd2,
                                                'best_model_dir': bm2})

    def predict1(self, texts):
        predicted_labels, raw_outputs = self.model1.predict(texts)
        return predicted_labels

    def predict2(self, texts):
        predicted_labels, raw_outputs = self.model2.predict(texts)
        return predicted_labels

    # text1: INDICATIONS AND DOSAGE
    # text2: Children
    def predict(self, text1, text2):
        res = {'INDICATIONS AND DOSAGE': -1, 'Children': -1}
        if not text1 or text1 == np.nan:
            if text2 and text2 != np.nan:
                pred2 = self.predict2([text2])
                res['Children'] = pred2[0]
        else:
            pred1 = self.predict1([text1])
            if pred1[0] == 1:
                res['INDICATIONS AND DOSAGE'] = 1
            else:
                res['INDICATIONS AND DOSAGE'] = 0
                if text2 and text2 != np.nan:
                    pred2 = self.predict2([text2])
                    res['Children'] = pred2[0]
        return res

    def eval(self, true_labels, predicted_labels):
        print(classification_report(true_labels, predicted_labels, digits=4))
        pprint(classification_report(true_labels, predicted_labels, output_dict=True))
        print('*' * 100)

    def eval20201020(self):
        h5 = 'train_test_20201020.h5'
        train1 = pd.read_hdf(h5, 'train1')
        test1 = pd.read_hdf(h5, 'test1')
        test1.to_excel('./data/tset2.xlsx')
        train2 = pd.read_hdf(h5, 'train2')
        test2 = pd.read_hdf(h5, 'test2')
        test2.to_excel('./data/tset.xlsx')
        # # train1, test1, train2, test2 = train_test_20201020()
        # trues1 = test1['labels'].tolist()
        # preds1 = self.predict1(test1['text'].tolist())
        # self.eval(trues1, preds1)
        #
        # trues2 = test2['labels'].tolist()
        #
        # preds2 = self.predict2(test2['text'].tolist())
        # self.eval(trues2, preds2)

    def test(self):
        df = pd.read_excel("./data/pmda-002-pdf-2.xlsx", "raw_text")
        t1 = df['用法及び用量'].tolist()
        t1 = [PMDADT.handle_text(i) for i in t1]
        r1 = self.predict1(t1)
        df['用法及び用量_nlp'] = r1

        t2 = df['小児'].tolist()
        t2 = [PMDADT.handle_text(i) for i in t2]
        print(t2)
        r2 = self.predict2(t2)
        df['小児_nlp'] = r2
        df.to_excel('./data/test.xlsx')


class TestModel:
    def __init__(self):
        pass

    @staticmethod
    def test_0707():
        from pprint import pprint
        from sklearn.metrics import classification_report
        from sklearn.utils import resample
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/label/pmda/data/test/pmda-001-pdf.xlsx")
        print(len(df))
        # df = df[['剤形', '規格・含量', '用法及び用量', '小児', '有効期間', '貯法', 'pdf name', 'indications' ]]
        # df.dropna(inplace=True)
        # print(len(df))
        # to_df = resample(df, replace=False, n_samples=200)
        # to_df.to_excel('./data/test/to_check.xlsx')


        # extract success rate
        dict = {
            "data_number":[len(df),200],
            '剤形':[1-df['剤形'].isnull().sum().sum()/len(df),1-2/200],
            '  規格・含量': [1-df['規格・含量'].isnull().sum().sum()/len(df), 1-2/200],
            '   用法及び用量': [1-df['用法及び用量'].isnull().sum().sum()/len(df), 1-(36+9)/len(df)],
            '      小児': [1-df['小児'].isnull().sum().sum()/len(df), 1-10/len(df)],
            '       有効期間': [1-df['有効期間'].isnull().sum().sum()/len(df), 1-3/len(df)],
            '         貯法': [1-df['貯法'].isnull().sum().sum()/len(df), 1-3/len(df)],
            '          効能・効果': [1-df['indications'].isnull().sum().sum()/len(df), 1-3/200],
        }
        indexs = ['successfully extracted', 'correctly extracted']
        cols = ['data_number', '剤形','  規格・含量','   用法及び用量','      小児','       有効期間','         貯法','          効能・効果']
        t_df = pd.DataFrame(dict,index=indexs,columns=cols)
        pd.set_option('precision', 4)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 60)
        pd.set_option('display.max_columns', None)
        pd.set_option('mode.chained_assignment', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)  # 将模糊字符宽度设置为2
        pd.set_option('display.unicode.east_asian_width', True)  # 检查东亚字符宽度属性

        pd.set_option("colheader_justify", "center")
        print('field extracted report:')
        pprint(t_df)

        # model test
        df = df[['用法及び用量','用法及び用量_nlp','用法及び用量_nlp_ss']]  # type:pd.DataFrame
        df.dropna(subset=['用法及び用量'],inplace=True)
        print(len(df))
        df.dropna(subset=['用法及び用量_nlp','用法及び用量_nlp_ss'],inplace=True)
        print(len(df))
        df['用法及び用量_nlp_ss'] = df['用法及び用量_nlp_ss'].map({'a':0,'b':0,1:1,0:0})
        true = df['用法及び用量_nlp_ss'].tolist()
        true = [int(i) for i in true]
        pred = df['用法及び用量_nlp'].tolist()
        pred = [int(i) for i in pred]
        print('model predict (用法及び用量) report:')
        c = classification_report(true,pred)
        print(c)


if __name__ == "__main__":
    TestModel.test_0707()
    # p = Predictor()
    # p.test()

    # p.eval20201020()
    # h5 = 'train_test_20201020.h5'
    # train1 = pd.read_hdf(h5, 'train1')
    # test1 = pd.read_hdf(h5, 'test1')
    # train2 = pd.read_hdf(h5, 'train2')
    # test2 = pd.read_hdf(h5, 'test2')
