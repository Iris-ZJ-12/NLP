# -*- coding: UTF-8 -*-
"""
Description : 
"""
import pandas as pd
import langid
import time
from pharm_ai.util.utils import Utilfuncs as u
from sklearn.utils import resample
from pharm_ai.util.prepro import Prepro

# def get_dict2():
#     df = pd.read_excel("/home/zyl/disk_a/PharmAI/pharm_ai/word/data/v2.0/word项目数据.xlsx", sheet_name="Sheet1")
#     nd = df[df['word'] != '、']
#     nd = nd[nd['word'] != '']
#     nd.dropna(subset=['sentence'], inplace=True)
#     nd['word'] = nd['word'].apply(lambda x: str(x).strip())
#     dicts = list(set(nd['word'].values))
#     dicts.remove('nan')
#     dicts.remove('')
#     dicts.sort()
#
#     dict_df = pd.DataFrame()
#     dict_df['word'] = dicts
#     project_ids = list(range(1, len(dicts) + 1))
#     project_ids = [str(i) for i in project_ids]
#     dict_df['project_id'] = project_ids
#     dict_df = dict_df.append({'word': 'nan', 'project_id': '0'}, ignore_index=True)
#     u.to_excel(dict_df, '/home/zyl/disk_a/PharmAI/pharm_ai/word/data/v2.0/v2_0_2_0309.xlsx', sheet_name='dict')
#
#     dt = pd.merge(nd, dict_df, how='left', on=['word'])
#     # dt['project_id'] = dt['project_id'].apply(lambda x: '0' if str(x) == 'nan' else x)
#     # dt['word'] = dt['word'].apply(lambda x: 'nan' if str(x) == 'nan' else x)
#     dt['project_id'] = dt['project_id'].apply(lambda x: '0' if str(x) == 'nan' else x)
#     u.to_excel(dt, '/home/zyl/disk_a/PharmAI/pharm_ai/word/data/v2.0/v2_0_2_0309.xlsx', sheet_name='raw')
#
#     li = []
#     for sentence, s_df in dt.groupby('sentence', sort=False):
#         target_texts = s_df['project_id'].to_list()
#         print(target_texts)
#         if len(target_texts) == 0:
#             target_text = target_texts[0]
#         else:
#             target_text = ','.join(sorted(list(set(target_texts))))
#             li.append({"input_text": str(sentence).strip(), "target_text": target_text})
#     df_ = pd.DataFrame(li)
#     df_['prefix'] = 'dictionary_match'
#     df_ = df_[['prefix', 'input_text', 'target_text']]
#     u.to_excel(df_, '/home/zyl/disk_a/PharmAI/pharm_ai/word/data/v2.0/v2_0_2_0309.xlsx', sheet_name='t5_dt')
#
#     raw_df = resample(df_, random_state=342, replace=False)
#     cut_point = int(0.8 * len(raw_df))
#     train_df = raw_df[0:cut_point]
#     eval_df = raw_df[cut_point:]
#     u.to_excel(train_df, '/home/zyl/disk_a/PharmAI/pharm_ai/word/data/v2.0/v2_0_2_0309.xlsx', sheet_name='train')
#     u.to_excel(eval_df, '/home/zyl/disk_a/PharmAI/pharm_ai/word/data/v2.0/v2_0_2_0309.xlsx', sheet_name='eval')

# p = Prepro()

# def get_token_num(text):
#     lang = langid.classify(text)[0]
#     if lang == 'zh':
#         sents = p.ht.cut_sentences(text)
#         sents = p.tokenize_hybrid_text
#     t = Prepro().tokenize_hybrid_text_generic(text=text)
#     token_length = 0
#     for i in t:
#         token_length += len(i)
#     else:
#         pass
#     return token_length

class WordDT:
    def __init__(self, version, new_xlsx_data):
        self.ver = version
        self.nd = pd.read_excel(new_xlsx_data, sheet_name='Sheet1')
        self.tx = './data/' + '.'.join(version.split('.')[0:-1]) + '/' + '_'.join(version.split('.')) + '_' + \
                  time.strftime("%m%d_%H%M%S", time.localtime()) + '.xlsx'

    def deal_with_new_data(self, nd: pd.DataFrame):
        nd = nd[nd['word'] != '、']
        nd.dropna(subset=['sentence'], inplace=True)
        sentences = set(nd['sentence'].tolist())

        nd.dropna(subset=['word'], inplace=True)
        nd['word'] = nd['word'].apply(lambda x: x.strip())
        dicts = set(nd['word'].values)

        nd['sentence'] = nd['sentence'].apply(lambda x: 'sentence: ' + str(x))
        nd['input_text'] = nd['sentence'].str.cat(nd['word'], sep='|indications: ')
        nd['target_text'] = '1'
        nd['prefix'] = 'word'
        nd = nd[['prefix', 'input_text', 'target_text']]

        return dicts, sentences, nd

    def create_fake_dt(self, dicts, sentences, input_texts):
        print('create negative data by using random method:...')
        negative_data = pd.DataFrame(columns=['prefix', 'input_text', 'target_text'])

        for i in dicts:
            for j in sentences:
                if str(j) + '|' + str(i) not in input_texts:
                    negative_data = negative_data.append([{'prefix': 'word', 'input_text':
                        'sentence: ' + str(j) + ' | indications: ' + str(i), 'target_text': '0'}], ignore_index=True)
        negative_data.drop_duplicates(inplace=True)
        print('Done creating negative data:...')
        return negative_data

    @staticmethod
    def cut_train_eval(df):
        raw_df = resample(df, random_state=342, replace=False)
        cut_point = int(0.8 * len(raw_df))
        train_df = raw_df[0:cut_point]
        eval_df = raw_df[cut_point:]
        return train_df, eval_df

    def visual_dt(self, xlsx_file, sheet_name='Sheet1'):
        df = pd.read_excel(xlsx_file, sheet_name)
        print(xlsx_file)
        print('name: ', sheet_name)
        print('length: ', df.shape[0])
        vc = df['target_text'].value_counts()
        print(vc)

    def get_text_token(self, text):
        pass


class V2_0(WordDT):
    def __init__(self, version):
        super(V2_0, self).__init__(version=version, new_xlsx_data="./data/v2.0/word项目数据.xlsx")
        # dicts, sentences, nd = self.deal_with_new_data(self.nd)
        # neg_df = self.create_fake_dt(dicts, sentences, nd['input_text'].values)
        #
        # u.to_excel(nd, self.tx, 'positive_dt')
        # u.to_excel(neg_df, self.tx, 'negative_dt')
        #
        # neg_df = resample(neg_df, n_samples=6000, random_state=442, replace=False)
        # raw_df = pd.concat([nd, neg_df], ignore_index=True)
        # train, eval = self.cut_train_eval(raw_df)
        # u.to_excel(raw_df, self.tx, 'raw')
        # u.to_excel(train, self.tx, 'train')
        # u.to_excel(eval, self.tx, 'eval')

        # self.visual_dt("./data/v2.0/processed_0302.xlsx", sheet_name='raw')
        # self.visual_dt("./data/v2.0/processed_0302.xlsx", sheet_name='train')
        # self.visual_dt("./data/v2.0/processed_0302.xlsx", sheet_name='eval')
        # self.visual_dt("./data/v2.0/processed_0302.xlsx", sheet_name='positive_dt')

    def V2_0_3(self):
        raw = pd.read_excel("/home/zyl/disk_a/PharmAI/pharm_ai/word/data/v2.0/v2_0_2_0309.xlsx", sheet_name='raw')
        raw['word'] = raw['word'].apply(lambda x: '|' if str(x) == 'nan' else '|' + str(x).strip())
        li = []
        for sentence, s_df in raw.groupby('sentence', sort=False):
            target_texts = s_df['word'].to_list()
            if len(target_texts) == 0:
                target_text = target_texts[0]
            else:
                target_text = ''.join(sorted(list(set(target_texts))))
            li.append({"input_text": str(sentence).strip(), "target_text": target_text})

        df_ = pd.DataFrame(li)
        df_['prefix'] = 'dictionary_match'
        df_ = df_[['prefix', 'input_text', 'target_text']]
        u.to_excel(df_, self.tx, sheet_name='raw')
        raw_df = resample(df_, replace=False)
        cut_point = int(0.8 * len(raw_df))
        train_df = raw_df[0:cut_point]
        eval_df = raw_df[cut_point:]
        u.to_excel(train_df, self.tx, sheet_name='train')
        u.to_excel(eval_df, self.tx, sheet_name='eval')


from pharm_ai.word.word_utils import WORDUtils


class V2_3():
    def __init__(self):
        self.start_time = '2021-06-24'
        self.indication_dict = WORDUtils().indication_dict
        self.indication_id_dict = WORDUtils().indication_id_dict
        pass

    def test(self):
        self.dt_0624()



    def dt_0624(self):
        df_1 = pd.read_excel("./data/v2.3/prepared_dt_0531-2021.6.10.xlsx")  # type:pd.DataFrame
        pos_df_1 = df_1[df_1['disease'].notna()]

        pos_df_1 = pos_df_1[pos_df_1['细分适应症'].notna()]
        neg_list_1 = df_1[~df_1['sentence'].isin(pos_df_1['sentence'].tolist())]
        o_n = 1
        pos_df_1['indication'] = pos_df_1['细分适应症'].apply(
            lambda x: self.indication_dict.get(x) if self.indication_dict.get(x) else x)
        pos_df_1['indication'] = pos_df_1['indication'].apply(
            lambda x: self.indication_id_dict.get(x) if self.indication_id_dict.get(x) else x)

        print(pos_df_1['indication'].tolist())
        # print(neg_list_1)

        df_2 = pd.read_excel("./data/v2.3/word数据标注.xlsx", 'to_labeled')  # type:pd.DataFrame
        pos_df_2 = df_2[df_2['label_EN'].notna()]
        neg_list_2 = df_2[~df_2['sentence'].isin(pos_df_2['sentence'].tolist())]
        pos_df_2['indication'] = pos_df_2['label_EN'].apply(
            lambda x: self.indication_dict.get(x) if self.indication_dict.get(x) else x)
        print(pos_df_2['indication'].tolist())
        # print(neg_list_2)


if __name__ == '__main__':
    V2_3().test()
    # V2_0(version='v2.0.5').V2_0_3()
    pass
