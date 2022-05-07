# -*- coding: UTF-8 -*-
"""
Description : 
"""
import pandas as pd


class WORDUtils:
    def __init__(self):
        self.indication_dict = WORDUtils.get_dict(is_saved=True)
        self.indication_id_dict = self.get_num_dict(is_saved=True)

    def test(self):
        self.get_num_dict(is_saved=True)
        # WORDUtils.get_dict(is_saved=True)
        # WORDUtils.

    @staticmethod
    def get_dict(is_saved=True):
        if is_saved:
            df = pd.read_excel('./data/dict/indication_dict_0624.xlsx')
            return dict(zip(df['细分适应证'].tolist(), df['indication_EN'].tolist()))
        df = pd.read_excel('./data/v2.3/word数据标注.xlsx', '字典')  # type:pd.DataFrame
        keys = list(set(df['indication_EN'].tolist()))
        sub_df = pd.DataFrame()
        sub_df['indication_EN'] = keys
        sub_df['细分适应证'] = keys
        sub_df['疾病'] = [df[df['indication_EN'] == k]['疾病'].tolist()[0] for k in keys]
        combine_df = pd.concat([df, sub_df], ignore_index=True)
        s = pd.Series({'疾病': 'other', '细分适应证': 'other', 'indication_EN': 'other'})
        combine_df = combine_df.append(s, ignore_index=True)
        combine_df.to_excel('./data/dict/indication_dict_0624.xlsx')

    @staticmethod
    def create_fake_dt_by_pos_dt(dicts, sentences, input_texts):
        print('create negative data by using random method:...')
        negative_data = pd.DataFrame(columns=['prefix', 'input_text', 'target_text'])

        for i in dicts:
            for j in sentences:
                if str(j) + '|' + str(i) not in input_texts:
                    negative_data = negative_data.append([{'prefix': 'word', 'input_text':
                        'sentence: ' + str(j) + ' | indications: ' + str(i), 'target_text': 0}], ignore_index=True)
        negative_data.drop_duplicates(inplace=True)
        print('Done creating negative data:...')
        return negative_data

    def get_num_dict(self, is_saved=True):
        if is_saved:
            df = pd.read_excel('./data/dict/indication_id_dict_0624.xlsx')
            return dict(zip(df['indication'].tolist(), df['indication_id'].tolist()))
        all_indications = list(set(self.indication_dict.values()))
        df = pd.DataFrame()
        df['indication'] = all_indications
        df['indication_id'] = list(range(1, len(df) + 1))
        df = df.append(pd.Series({'indication': 'other', 'indication_id': '0'}), ignore_index=True)
        df.to_excel('./data/dict/indication_id_dict_0624.xlsx')


if __name__ == '__main__':
    WORDUtils().test()
