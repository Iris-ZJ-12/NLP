# encoding: utf-8
'''
@author: zyl
@file: dt.py
@time: 2021/8/12 10:33
@desc:
'''

import pandas as pd
from tqdm import tqdm

from pharm_ai.util.utils import Utilfuncs


#
# class DT:
#     def __init__(self):
#         self.e = Evaluator()
#         pass
#
#     def run(self):
#         # self.dt_0811()
#         # self.up_train()
#         # self.dt_0812()
#         # self.dt_0813()
#         # self.dt_0816()
#         # self.dt_0816_2()
#         # self.dt_0816_3()
#         self.dt_0817()
#         pass
#
#     def prepare_neg_dt(self, df, neg_smaple_size=30, method='random', score_threshold=0.32):
#         entities = list(set(df['entry'].tolist()))
#         print(len(entities))
#         li = []
#         for _, sub_df in tqdm(df.iterrows()):
#             if method == 'predict':
#                 e_s = self.e.predict([sub_df['entity']], top_k=100, score_threshold=score_threshold)
#
#                 if (e_s == [[]]) | (e_s == []):
#                     e_s = random.sample(entities, neg_smaple_size)
#                 else:
#                     e_s = e_s[0][0:neg_smaple_size]
#             else:
#                 e_s = random.sample(entities, neg_smaple_size)
#             # print(len(e_s))
#             for e_ in e_s:
#                 if df[(df['entry'] == e_) & (df['entity'] == sub_df['entity'])].empty:
#                     li.append({'entry': e_, 'entity': sub_df['entity'], 'label': 0})
#         neg_df = pd.DataFrame(li)
#         return neg_df
#
#     @staticmethod
#     def format_dt(df):
#         df['entry'] = df['target_text'].apply(lambda x: ModelUtils.revise_target_text(x, '|', 'list'))
#         df['label'] = df['entry'].apply(lambda x: 1 / len(x))
#         df = df.explode('entry')
#         df.rename(columns={"input_text": 'entity'}, inplace=True)
#         df = df[['entity', 'entry', 'label']]  # type:pd.DataFrame
#         df.drop_duplicates(inplace=True)
#         return df
#
#     def dt_0811(self):
#         project = 'panel_entry_match'
#         df1 = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/entry_dict_0508.xlsx",
#                             "disease_dict")
#         df1['label'] = 1
#         df1 = df1[['entity', 'entry', 'label']]
#         # Utilfuncs.to_excel(df1, './data/em_0811.xlsx', 'dict')
#
#         df2 = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/processed_entry_match_0508.xlsx", 'train')
#         df2 = df2[df2['prefix'] == 'disease_em']
#         df2 = DT.format_dt(df2)
#         # Utilfuncs.to_excel(df2, './data/em_0811.xlsx', 'old_train')
#
#         df3 = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/processed_entry_match_0508.xlsx", 'eval')
#         df3 = df3[df3['prefix'] == 'disease_em']
#         df3 = DT.format_dt(df3)
#         Utilfuncs.to_excel(df3, './data/em_0811.xlsx', 'old_eval')
#
#         train_pos_df = pd.concat([df1, df2], ignore_index=True)
#         train_pos_df.drop_duplicates(inplace=True)
#         Utilfuncs.to_excel(train_pos_df, './data/em_0811.xlsx', 'train_pos')
#
#         train_neg_df = self.prepare_neg_dt(train_pos_df, neg_smaple_size=10)
#         Utilfuncs.to_excel(train_neg_df, './data/em_0811.xlsx', 'train_neg')
#
#         train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)
#         train_df = resample(train_df, replace=False)
#         Utilfuncs.to_excel(train_df, './data/em_0811.xlsx', 'train')
#
#     @staticmethod
#     def prepare_dt(df):
#         dt = []
#         for _, sub_df in df.iterrows():
#             dt.append(InputExample(texts=[sub_df['entry'], sub_df['entity']], label=sub_df['label']))
#         return dt
#
#     def up_train(self):
#         pos_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0811.xlsx", 'train_pos')
#         neg_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0811.xlsx", 'train_neg')
#         up_pos_df = resample(pos_df, replace=True, n_samples=int(len(neg_df) - len(pos_df)))
#         train_up = pd.concat([pos_df, neg_df, up_pos_df], ignore_index=True)
#         train_up = resample(train_up, replace=False)
#         Utilfuncs.to_excel(train_up, './data/em_0812.xlsx', 'train_up')
#
#     def dt_0813(self):
#         train_pos_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0811.xlsx",
#                                      'train_pos')
#         train_neg_df = self.prepare_neg_dt(train_pos_df, neg_smaple_size=10, method='predict')
#         Utilfuncs.to_excel(train_neg_df, './data/em_0812_2.xlsx', 'train_neg')
#
#         train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)
#         train_df = resample(train_df, replace=False)
#         Utilfuncs.to_excel(train_df, './data/em_0812_2.xlsx', 'train')
#
#         up_pos_df = resample(train_pos_df, replace=True, n_samples=int(len(train_neg_df) - len(train_pos_df)))
#         train_up = pd.concat([train_pos_df, train_neg_df, up_pos_df], ignore_index=True)
#         train_up = resample(train_up, replace=False)
#         Utilfuncs.to_excel(train_up, './data/em_0812_2.xlsx', 'train_up')
#
#     def dt_0816(self):
#         train_pos_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0811.xlsx",
#                                      'train_pos')
#         train_neg_df = self.prepare_neg_dt(train_pos_df, neg_smaple_size=10, method='predict')
#         Utilfuncs.to_excel(train_neg_df, './data/em_0816.xlsx', 'train_neg')
#
#         train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)
#         train_df = resample(train_df, replace=False)
#         Utilfuncs.to_excel(train_df, './data/em_0816.xlsx', 'train')
#
#         up_pos_df = resample(train_pos_df, replace=True, n_samples=int(len(train_neg_df) - len(train_pos_df)))
#         train_up = pd.concat([train_pos_df, train_neg_df, up_pos_df], ignore_index=True)
#         train_up = resample(train_up, replace=False)
#         Utilfuncs.to_excel(train_up, './data/em_0816.xlsx', 'train_up')
#
#     def dt_0816_2(self):
#         train_pos_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0811.xlsx",
#                                      'train_pos')
#         train_neg_df = self.prepare_neg_dt(train_pos_df, neg_smaple_size=10, method='predict', score_threshold=1.1)
#         Utilfuncs.to_excel(train_neg_df, './data/em_0816_2.xlsx', 'train_neg')
#
#         train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)
#         train_df = resample(train_df, replace=False)
#         Utilfuncs.to_excel(train_df, './data/em_0816_2.xlsx', 'train')
#
#         up_pos_df = resample(train_pos_df, replace=True, n_samples=int(len(train_neg_df) - len(train_pos_df)))
#         train_up = pd.concat([train_pos_df, train_neg_df, up_pos_df], ignore_index=True)
#         train_up = resample(train_up, replace=False)
#         Utilfuncs.to_excel(train_up, './data/em_0816_2.xlsx', 'train_up')
#
#     def dt_0816_3(self):
#         train_pos_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0816.xlsx",
#                                      'train')
#         train_pos_df['label'] = train_pos_df['label'].apply(lambda x: 1 if x != 0 else 0)
#         Utilfuncs.to_excel(train_pos_df, './data/em_0816_3.xlsx', 'train')
#
#         eval = pd.read_excel('./data/em_0811.xlsx', 'old_eval')
#         eval['label'] = eval['label'].apply(lambda x: 1 if x != 0 else 0)
#         Utilfuncs.to_excel(eval, './data/em_0816_3.xlsx', 'eval')
#
#         train_up = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0816.xlsx",
#                                  'train_up')
#         train_up['label'] = train_up['label'].apply(lambda x: 1 if x != 0 else 0)
#         Utilfuncs.to_excel(train_up, './data/em_0816_3.xlsx', 'train_up')
#
#     def dt_0817(self):
#         # train_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0816_3.xlsx",
#         #                          'train')
#         #
#         # train_df.rename(columns={"entity": 'text_a', 'entry': 'text_b', 'label': 'labels'}, inplace=True)
#         # Utilfuncs.to_excel(train_df, './data/em_0817.xlsx', 'train')
#         #
#         # eval_df_pos = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0816_3.xlsx",
#         #                             'eval')
#         # eval_df_neg = self.prepare_neg_dt(eval_df_pos, neg_smaple_size=10, method='predict', score_threshold=1.1)
#         # eval_df = pd.concat([eval_df_pos, eval_df_neg], ignore_index=True)
#         # eval_df = resample(eval_df, replace=False)
#         # eval_df.rename(columns={"entity": 'text_a', 'entry': 'text_b', 'label': 'labels'}, inplace=True)
#         # Utilfuncs.to_excel(eval_df, './data/em_0817.xlsx', 'eval')
#
#         train_df_up = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0816_3.xlsx",
#                                     'train_up')
#
#         train_df_up.rename(columns={"entity": 'text_a', 'entry': 'text_b', 'label': 'labels'}, inplace=True)
#         # Utilfuncs.to_excel(train_df_up, './data/em_0817.xlsx', 'train_up')
#
#         pass
#
#
# from pharm_ai.panel.my_utils import ModelUtils, MyModel, DTUtils
#
#
# class DTV2():
#     di_dict_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/entry_dict_0508.xlsx",
#                                "disease_dict")
#     di_dict = dict(zip(di_dict_df['entry'].tolist(), di_dict_df['esid'].tolist()))
#
#     def __init__(self):
#         'test BatchSemiHardTripletLoss and BatchHardSoftMarginTripletLoss'
#         pass
#
#     def run(self):
#         # self.dt_0827()
#         self.dt_0830()
#         pass
#
#     @staticmethod
#     def format_dt(text):
#         if DTV2.di_dict.get(text):
#             return DTV2.di_dict.get(text)
#         else:
#             return 0
#
#     def dt_0827(self):
#         train_pos = pd.read_excel('./data/em_0811.xlsx', 'train_pos')
#         train_pos['label'] = train_pos['entry'].apply(DTV2.format_dt)
#         train_pos['texts'] = train_pos['entity']
#         train_pos = train_pos[['texts', 'label']]
#         Utilfuncs.to_excel(train_pos, './data/em_0827.xlsx', 'train_pos')
#
#         old_eval = pd.read_excel('./data/em_0811.xlsx', 'old_eval')
#
#         old_eval['label'] = old_eval['entry'].apply(DTV2.format_dt)
#         old_eval['texts'] = old_eval['entity']
#         old_eval = old_eval[['texts', 'label']]
#
#         Utilfuncs.to_excel(old_eval, './data/em_0827.xlsx', 'old_eval')
#
#     @staticmethod
#     def prepare_t5_data(df):
#         input_texts = []
#         target_texts = []
#         for a, b, l in zip(df['text_a'].tolist(), df['text_b'].tolist(), df['labels'].tolist()):
#             input_texts.append(str(a) + ' | ' + str(b))
#             target_texts.append(str(l))
#         df['input_text'] = input_texts
#         df['target_text'] = target_texts
#         df['prefix'] = 'disease'
#         return df[['prefix', 'input_text', 'target_text']]
#
#     def dt_0830(self):
#         df_t = pd.read_excel('./data/em_0817.xlsx', 'train')
#         df_t = DTV2.prepare_t5_data(df_t)
#         Utilfuncs.to_excel(df_t, './data/em_0830.xlsx', 'train')
#
#         df_e = pd.read_excel('./data/em_0817.xlsx', 'eval')
#         df_e = DTV2.prepare_t5_data(df_e)
#         Utilfuncs.to_excel(df_e, './data/em_0830.xlsx', 'eval')
#
#         df3 = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/processed_entry_match_0508.xlsx", 'eval')
#         df3 = df3[df3['prefix'] == 'disease_em']
#
#         Utilfuncs.to_excel(df3, './data/em_0830.xlsx', 'eval_old')
#
#
# df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0811.xlsx",
#                    'train')
# e_1 = df['entity'].tolist()
# e_2 = df['entry'].tolist()
# e_1.extend(e_2)
# all_entities = list(set(e_1))
#
#
# class DTV4():
#     def __init__(self):
#
#         pass
#
#     def run(self):
#         # self.dt_0901()
#         self.dt_0901_2()
#
#     @staticmethod
#     def tripletloss_apply(df):
#         positive_df = df[df['label'] == 1]
#         positive = list(set(positive_df['entity'].tolist()))
#         p_s = []
#         n_s = []
#         for p in positive:
#             negative = random.sample(all_entities, 10)
#             for n in negative:
#                 if n not in positive:
#                     p_s.append(p)
#                     n_s.append(n)
#         df_ = pd.DataFrame()
#         df_['positive'] = p_s
#         df_['negative'] = n_s
#         df_['anchor'] = df['entry'].tolist()[0]
#         return df_
#
#     def dt_0901(self):
#         # TripletLoss三元组
#         train = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0811.xlsx", 'train_pos')  # type:pd.DataFrame
#         train = train.groupby('entry').apply(DTV4.tripletloss_apply)
#         train = train[['anchor', 'positive', 'negative']]
#         # print(train)
#         Utilfuncs.to_excel(train, './data/em_0901.xlsx', 'train')
#
#         eval = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/processed_entry_match_0508.xlsx", 'eval')
#         eval = eval[eval['prefix'] == 'disease_em']
#         eval = eval[['input_text', 'target_text']]
#
#         Utilfuncs.to_excel(eval, './data/em_0901.xlsx', 'eval')
#
#     def dt_0901_2(self):
#         # nli
#         pass

class DTV5:
    def __init__(self):
        # 用第一第二批的疾病数据
        di_dict = pd.read_excel("../data/em_v2/entries_dict_0902.xlsx", 'disease')
        self.all_entries = list(set(di_dict['entry'].tolist()))
        self.all_entries.append("没有疾病")
        from predict_retrieval import Predictor
        self.p = Predictor()
        pass

    def run(self):
        # self.dt_0930()
        # self.dt_0930_2()
        # self.dt_1009()
        # self.dt_1012()
        self.dt_1013()

    def prepare_dt(self, df, neg_smaple_size=30,mode='random'):
        import random
        res_li = []
        for _, sub_df in tqdm(df.iterrows()):
            entity = sub_df['entity']
            entries = sub_df['entry'].split(',')
            if mode=='predict':
                e_s = self.p.predict([entity], top_k=100, score_threshold=0.0)
                if (e_s == [[]]) | (e_s == []):
                    candidate_entries = random.sample(self.all_entries, neg_smaple_size)
                else:
                    candidate_entries = e_s[0][0:neg_smaple_size]
            else:
                candidate_entries = random.sample(self.all_entries, neg_smaple_size)
            for c_e in candidate_entries:
                if c_e not in entries:
                    res_li.append({'entity': entity, 'entry': c_e, 'label': 0})
            for e in entries:
                res_li.append({'entity': entity, 'entry': e, 'label': 1 / len(entries)})

        res_df = pd.DataFrame(res_li)
        return res_df

    def dt_0930(self):
        from pharm_ai.panel.my_utils import DTUtils
        disease_df = pd.read_excel("../data/em_v2/process_0906.xlsx", 'disease')
        disease_df.rename(columns={'modified_entry': 'entry'}, inplace=True)
        disease_df = disease_df[['entity', 'entry']]

        def revise_entry(entry: str):
            if entry == '|':
                return "没有疾病"
            if '|' in entry:
                entries = entry.split('|')
                while '' in entries:
                    entries.remove('')
                return ','.join(entries)
            return entry

        disease_df['entry'] = disease_df['entry'].apply(revise_entry)
        disease_df.drop_duplicates(inplace=True)
        Utilfuncs.to_excel(disease_df, "./data/v2/dt_0930.xlsx", 'disease_dt')

        di_dict = pd.read_excel("../data/em_v2/entries_dict_0902.xlsx", 'disease')
        di_dict = di_dict[['entity', 'entry']]
        di_dict.drop_duplicates(inplace=True)
        Utilfuncs.to_excel(di_dict, "./data/v2/dt_0930.xlsx", 'di_dict')

        train, eval = DTUtils.cut_train_eval(disease_df)
        train = pd.concat([train, di_dict], ignore_index=True)
        train.drop_duplicates(inplace=True)
        Utilfuncs.to_excel(train, "./data/v2/dt_0930.xlsx", 'train')
        Utilfuncs.to_excel(eval, "./data/v2/dt_0930.xlsx", 'eval')

    def dt_0930_2(self):
        # train_df = pd.read_excel("./data/v2/dt_0930.xlsx","train")
        # res_df = self.prepare_dt(train_df, neg_smaple_size=30)  # type:pd.DataFrame
        # res_df.to_csv("./data/v2/train.csv.gz",compression='gzip',sep='|')


        # eval_df = pd.read_excel("./data/v2/dt_0930.xlsx","eval")
        # res_df = self.prepare_dt(eval_df, neg_smaple_size=30)  # type:pd.DataFrame
        # res_df.to_csv("./data/v2/eval.csv.gz",compression='gzip',sep='|')

        s = pd.read_csv("./data/v2/eval.csv.gz", compression='gzip', sep='|')
        s.to_excel('./data/v2/t2.xlsx')

    def dt_1009(self):
        train_df = pd.read_excel("./data/v2/dt_0930.xlsx","train")
        res_df = self.prepare_dt(train_df, neg_smaple_size=100,mode='predict')  # type:pd.DataFrame
        res_df.to_csv("./data/v2/train_4.csv.gz",compression='gzip',sep='|')

        eval_df = pd.read_excel("./data/v2/dt_0930.xlsx","eval")
        res_df = self.prepare_dt(eval_df, neg_smaple_size=30,mode='predict')  # type:pd.DataFrame
        res_df.to_csv("./data/v2/eval_4.csv.gz",compression='gzip',sep='|')

        # s = pd.read_csv("./data/v2/eval.csv.gz", compression='gzip', sep='|')
        # s.to_excel('./data/v2/t2.xlsx')

    def dt_1012(self):
        df = pd.read_excel("./data/v2/dt_0930.xlsx","di_dict")
        entries = sorted(list(set(df['entry'].tolist())))
        print(len(entries))
        dict = [{'entry':'没有疾病','label_num':0}]

        for i,e in enumerate(entries):
            dict.append({'entry':e,'label_num':i+1})
        label_dict= pd.DataFrame(dict)
        label_dict.sort_values("label_num",inplace=True)
        label_dict.to_excel("./data/v2/label_dict.xlsx")


    def dt_1013(self):
        df = pd.read_excel("./data/v2/dt_0930.xlsx","di_dict")
        entries = sorted(list(set(df['entry'].tolist())))
        print(len(entries))
        dict = [{'entry':'没有疾病','label_num':0}]

        for i,e in enumerate(entries):
            dict.append({'entry':e,'label_num':i+1})
        label_dict= pd.DataFrame(dict)
        label_dict.sort_values("label_num",inplace=True)
        label_dict.to_excel("./data/v2/label_dict.xlsx")



if __name__ == '__main__':
    # DT().run()
    # DTV2().run()
    # DTV4().run()
    DTV5().run()
