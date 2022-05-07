# -*- coding: UTF-8 -*-
"""
Description : 
"""
from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
import logging
# from pharm_ai.package_insert_extractors.ped_extractors.dt import fda_train_test_20201012
from time import time
from pharm_ai.config import ConfigFilePaths as cfp
import torch
from pharm_ai.util.utils import Utilfuncs as u
from sklearn.metrics import classification_report
import numpy as np
from loguru import logger
from pharm_ai.label.fda.dt import FDADT

f = "/home/zyl/disk/PharmAI/pharm_ai/label/fda/"
# f = cfp.project_dir + '/package_insert_extractors/ped_extractors/fda/'
od = f + 'outputs/20201012/'
cd = f + 'cache/20201012/'
bm = f + 'best_model/20201012/'

torch.cuda.empty_cache()


# train, test = fda_train_test_20201012()

class Predictor:
    def __init__(self):
        self.model = ClassificationModel('bert', bm,
                                         num_labels=2, use_cuda=True, cuda_device=1,
                                         args={'reprocess_input_data': True,
                                               'use_cached_eval_features': True,
                                               'overwrite_output_dir': True,
                                               'fp16': True,
                                               'n_gpu': 1,
                                               'use_multiprocessing': False,
                                               'output_dir': od, 'cache_dir': cd,
                                               'best_model_dir': bm})

    def predict(self, texts):
        predicted_labels, raw_outputs = self.model.predict(texts)
        predicted_labels = predicted_labels.tolist()
        return predicted_labels

    def eval(self, test_df, excel_path=None, sheet_name='default'):
        texts = test_df['text'].tolist()
        trues = test_df['labels'].tolist()
        preds = self.predict(texts)

        print(classification_report(trues, preds, digits=4))

        if excel_path:
            df = pd.DataFrame({'texts': texts,
                               'predicted_labels': preds,
                               'actual_labels': trues})
            u.to_excel(df, excel_path, sheet_name)

    def eval_20201013(self):
        h5 = 'fda_all_fields_2_20201012.h5'
        df = pd.read_hdf(h5)
        pu = df['PEDIATRIC USE'].tolist()
        labels = []
        for text in pu:
            try:
                label = self.predict([text])[0]
                labels.append(label)
            except Exception as e:
                logger.warning(e)
                labels.append(text)
        df['labels'] = labels
        x = 'fda-all-fields-2-20201013.xlsx'
        u.to_excel(df, x, 'result')

    def test(self):
        df = pd.read_excel("./data/pdfs_raw_text.xlsx")
        t1 = df['Pediatric Use'].tolist()
        t1 = [FDADT.handle_text(i) for i in t1]
        r1 = self.predict(t1)
        df['Pediatric Use_nlp'] = r1

        t2 = df['INDICATIONS AND USAGE'].tolist()
        t2 = [FDADT.handle_text(i) for i in t2]
        r2 = self.predict(t2)
        df['INDICATIONS AND USAGE_nlp'] = r2

        t3 = df['DOSAGE AND ADMINISTRATION'].tolist()
        t3 = [FDADT.handle_text(i) for i in t3]
        # print(t2)
        r3 = self.predict(t3)
        df['DOSAGE AND ADMINISTRATION_nlp'] = r3
        df.to_excel('./data/test.xlsx')


def fda_train_test_20201012():
    h5 = "/home/zyl/disk/PharmAI/pharm_ai/label/fda/fda_train_test-20201012.h5"
    train = pd.read_hdf(h5, 'train')
    print(train)
    test = pd.read_hdf(h5, 'test')
    return train, test


class TestModel():
    def __init__(self):
        pass

    def test(self):
        from pprint import pprint
        from sklearn.metrics import classification_report
        from sklearn.utils import resample
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/label/fda/data/fda-200 校验.xlsx")

        # extract success rate
        dict = {
            "data_number": [216, 216],
            '| INDICATIONS AND USAGE': [1 - 3 / 216,
                                      1 - 1 / (216 - 3)],
            '| DOSAGE AND ADMINISTRATION': [1 - 2 / 216,
                                             1 - 3 / (216 - 2)],
            '| Pediatric Use': [1 - 15 / 216, 1 - 5 / (216 - 15)],
            '| HOW SUPPLIED': [1 - 4 / 216, 1 - 10 / (216 - 4)],
        }
        indexs = ['successfully extracted', 'correctly extracted']
        cols = ['data_number', '| INDICATIONS AND USAGE', '| DOSAGE AND ADMINISTRATION', '| Pediatric Use',
                '| HOW SUPPLIED']
        t_df = pd.DataFrame(dict, index=indexs, columns=cols)
        pd.set_option('precision', 4)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 60)
        pd.set_option('display.max_columns', None)
        pd.set_option('mode.chained_assignment', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)  # 将模糊字符宽度设置为2
        pd.set_option('display.unicode.east_asian_width', True)  # 检查东亚字符宽度属性

        pd.set_option("colheader_justify", "center")
        print('field extracted report:')
        print(f'data_num : {len(df)}')
        print(f'error pdfs : {round((4 + 32) / len(df)*100,2)}% , two column pdfs : {round(144 / len(df)*100,2)}% ,'
              f' other pdfs:{round(72 / len(df)*100,2)}% ')
        pprint(t_df)

        # model test
        df = df[df['type'].isin(['two_column', 'other'])]
        df.dropna(subset=['Pediatric Use'], inplace=True)
        print(len(df))
        df.dropna(subset=['Pediatric Use_nlp', 'Pediatric Use_nlp_ss'], inplace=True)
        print(len(df))

        true = df['Pediatric Use_nlp'].tolist()
        true = [int(i) for i in true]
        pred = df['Pediatric Use_nlp_ss'].tolist()
        pred = [int(i) for i in pred]
        print('model predict (Pediatric Use) report:')
        c = classification_report(true, pred)
        print(c)


if __name__ == '__main__':
    # fda_train_test_20201012()
    # p = Predictor().test()
    TestModel().test()
    # p.eval_20201013()

    # h5 = 'fda_if_ped_unlabeled-20201009.h5'
    # df = pd.read_hdf(h5)
    # texts = df['text'].tolist()
    # preds = p.predict(texts)
    # df = pd.DataFrame({'text': texts, 'predicted_labels': preds})
    # f= 'fda_if_ped_test_result-20201009.xlsx'
    # u.to_excel(df, f, '6274')
