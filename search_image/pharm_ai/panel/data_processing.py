# encoding: utf-8
"""
@author: zyl
@file: data_processing.py
@time: 2021/11/25 11:17
@desc:
"""
import pandas as pd

from zyl_utils.data_utils.nlp_utils import DTUtils
from zyl_utils.utils.utils import Utils


class Processing:
    def __init__(self):
        pass

    def run(self):
        self.process_v4_2()
        pass

    @staticmethod
    def process_v4_2():
        def get_train_eval(sub_dt):
            sub_dt = sub_dt[sub_dt['input_text_tokens'] < 512]
            sub_dt = sub_dt[sub_dt['max_entity_length'] < 28]

            sub_dt = sub_dt[['id', 'text_type', 'input_text', 'target_text', 'prefix']]
            train, eval = DTUtils.cut_train_eval(sub_dt)  # (pd.DataFrame,pd.DataFrame)
            train = DTUtils.transfomer_data_format_from_t5_to_ner(train, keep_addition_info=('id', 'text_type'))
            eval = DTUtils.transfomer_data_format_from_t5_to_ner(eval, keep_addition_info=('id', 'text_type'))
            return train, eval

        analysis_dt = pd.read_excel("./data/v4/analysis_v4_2_t5.xlsx")  # type:pd.DataFrame
        disease = analysis_dt[analysis_dt['prefix'] == 'disease']
        target = analysis_dt[analysis_dt['prefix'] == 'target']
        disease_train, disease_eval = get_train_eval(disease)
        target_train, target_eval = get_train_eval(target)

        disease_train.to_hdf('./data/v4/processing_v4_2_bio.h5', 'disease_train')
        disease_eval.to_hdf('./data/v4/processing_v4_2_bio.h5', 'disease_eval')
        target_train.to_hdf('./data/v4/processing_v4_2_bio.h5', 'target_train')
        target_eval.to_hdf('./data/v4/processing_v4_2_bio.h5', 'target_eval')


if __name__ == '__main__':
    Processing().run()
