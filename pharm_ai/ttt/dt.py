# encoding: utf-8
'''
@author: zyl
@file: dt.py
@time: 2021/7/26 上午12:33
@desc:
'''
import pandas as pd
import re
from tqdm import tqdm
import html
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from pharm_ai.util.utils import Utilfuncs
import ast
from sklearn.utils import resample
from pharm_ai.util.mt5_ner import DTUtils

# AllExcludsionCriteria = ['Key Exclusion Criteria:\r<br/>', 'Key Exclusion criteria:<br/>', 'EXCLUSION\r',
#                          'Exclusion criteria COHORT B\r<br/>', 'EXCLUSION CRITERIA:\r<br/>', 'EXCLUSION<br/>',
#                          'Main exclusion criteria:\r<br/>', 'Main exclusion Criteria:\r', 'Exclusion:\r',
#                          'Exclusion criteria:<br/>', '[Exclusion criteria]\r', 'Exclusion criteria<br/>',
#                          'Exlusion criteria:\r<br/>', 'Exclusion criteria:\r', 'exclusion criteria：<br/>',
#                          'EXCLUSION CRITERIA FOR PRE-SCREENING\r<br/>', 'Exclusion Criteria:\r<br/>', 'Exclusion:<br/>',
#                          'Major exclusion criteria<br/>', 'Key exclusion Criteria:\r<br/>', 'Exclusions:\r<br/>',
#                          'Exclusion criteria (Core Phase):<br/>', 'EXCLUSION CRITERIA<br/>',
#                          'Exclusion criteria\r<br/>', 'Key exclusion criteria:\r<br/>', 'EXCLUSION CRITERIA\r<br/>',
#                          'Exclusion criteria<br/>', 'Main exclusion criteria for all patients:\r<br/>',
#                          'Exclusion criteria :\r<br/>', 'Exclusion Criteria\r', 'Exclusion criteria:',
#                          'EXCLUSION CRITERIA:', 'Exclusion Criteria', 'fail-1']

AllExcludsionCriteria = ["Exclusion criteria:\n", "Key Exclusion Criteria:\n", 'Exclusion:\n',
                         "Exclusion Criteria:\n", "Exclusion Criteria\n", "Exclusion criteria:",
                         "Exclusion Criteria:", "EXCLUSION CRITERIA:\n ", "Exclusion Criteria",
                         "Exclusion criteria", "Key Exclusion Criteria :\n", "Exclusion\n", "EXCLUSION CRITERIA\n",
                         "KEY EXCLUSION CRITERIA", "Main exclusion criteria for all patients:\n",
                         "Main exclusion Criteria:\n", "Major exclusion criteria\n", "Exclusions:\n",
                         "Main exclusion criteria:\n",
                         "EXCLUSION CRITERIA", "Key exclusion Criteria:\n", "Exclusion :\n ",
                         "EXCLUSION", "Key exclusion criteria:\n", "Exlusion criteria:\n",
                         ]

# class Therapy:
#     def __init__(self,en_name,zh_name,label_num):
#         self.en_name = en_name
#         self.zh_name = zh_name
#         self.label_num = label_num
#
# class Therapies:
#     def __init__(self):
#         self.
#
#         pass
ClassificationDictionary = {
    '其他治疗': 0, '一线治疗': 1, '二线治疗': 2, '辅助治疗': 3, '新辅助治疗': 4, '二线及以上治疗': 5, '末线治疗': 6,
    '三线治疗': 7, '维持治疗': 8, '初始治疗': 9, '巩固治疗': 10, '后续治疗': 11, '姑息治疗': 12, '诱导治疗': 13,
    '复发治疗': 14, '辅助内分泌治疗': 15, '挽救治疗': 16, '预防性治疗': 17, '没有治疗': 18,
}

ClassificationListEN = [
    'Other therapy', 'First line', 'Second line', 'Adjuvant therapy', 'Neoadjuvant therapy',
    'Second and later lines therapy', 'Last line therapy',
    'Third line', 'Maintenance therapy', 'Primary therapy', 'Consolidation therapy', 'Subsequent therapy',
    'Palliative care', 'Induction therapy',
    'Treatment for recurrence', 'Adjuvant endocrine therapy', 'Salvage treatment', 'Preventive therapy',
    'No therapy'
]  # Salvage treatment挽救治疗

ClassificationListZH = [
    '其他治疗', '一线治疗', '二线治疗', '辅助治疗', '新辅助治疗', '二线及以上治疗', '末线治疗',
    '三线治疗', '维持治疗', '初始治疗', '巩固治疗', '后续治疗', '姑息治疗', '诱导治疗',
    '复发治疗', '辅助内分泌治疗', '挽救治疗', '预防性治疗', '没有治疗',
]
ClassificationListNum = list(range(len(ClassificationListEN)))
therapy_zh_en_dict = dict(zip(ClassificationListZH, ClassificationListEN))
print(dict(zip(ClassificationListEN[0:-1], ClassificationListZH[0:-1])))
therapy_en_num_dict = dict(zip(ClassificationListEN, ClassificationListNum))
therapy_num_en_dict = dict(zip(ClassificationListNum,ClassificationListEN))

from bs4 import BeautifulSoup


class TTTDT:
    def __init__(self):
        pass

    @staticmethod
    def get_target_text_num(text):
        if text == ',':
            return text
        texts = text.split(',')
        r = []
        for t in texts:
            r.append(str(therapy_en_num_dict.get(t)))
            # r.append(str(ClassificationDictionary[t]))
        return ','.join(r)

    @staticmethod
    def get_target_text_en(text):
        if text == ',':
            return text
        texts = text.split(',')
        r = []
        for t in texts:
            r.append(therapy_zh_en_dict[t])
        return ','.join(r)

    @staticmethod
    def get_inclusion_criteria(text):
        text = ILLEGAL_CHARACTERS_RE.sub(r'', str(text))
        text = text.replace('\r<br/>\r<br/>', '\n')
        text = text.replace('\r<br/>', ' ')
        text = BeautifulSoup(text, 'html.parser').get_text()  # 解析获取所有文本
        text = html.unescape(text)  # 去除html转义字符
        text = re.sub(r' +', ' ', text)

        for char in AllExcludsionCriteria:
            if char not in text:
                continue
            else:
                text = text.split(char)[0]
        text = text.strip()
        return text

    @staticmethod
    def deal_with_labels(labels):
        if pd.isna(labels):  # 空
            return ''
        else:
            if (str(labels) == '[None]') | (str(labels) == "['']"):
                return ''
            else:
                all_labels = ast.literal_eval(str(labels))
                all_labels = [therapy_zh_en_dict.get(i) for i in all_labels]
                return ','.join(all_labels)

    @staticmethod
    def deal_with_labels2(labels):
        if pd.isna(labels):  # 空
            return ''
        else:
            if '｜' in labels:
                all_labels = labels.split('｜')
            else:
                all_labels = labels.split('|')

            res = []
            for i in all_labels:
                if i == '二线及以上':
                    res.append("Second and later lines therapy")
                else:
                    res.append(therapy_zh_en_dict.get(i.strip()))
            return ','.join(res)

    @staticmethod
    def cut_train_eval(df):
        raw_df = resample(df, replace=False)
        cut_point = min(8000, int(0.2 * len(raw_df)))
        eval_df = raw_df[0:cut_point]
        train_df = raw_df[cut_point:]
        return train_df, eval_df

    @staticmethod
    def up_sampling(df):
        neg_df = df[df['labels'] == 0]
        pos_df = df[df['labels'] == 1]
        print(f'neg_df_len: {len(neg_df)} , pos_df_len: {len(pos_df)} ')
        max_length = max(len(neg_df), len(pos_df))
        if len(neg_df) < max_length:
            up_df = resample(neg_df, n_samples=int(max_length - len(neg_df)), replace=True, random_state=123)
        elif len(pos_df) < max_length:
            up_df = resample(pos_df, n_samples=int(max_length - len(pos_df)), replace=True, random_state=123)
        else:
            up_df = pd.DataFrame()
        new_up_df = pd.concat([pos_df, neg_df, up_df], ignore_index=True)
        return resample(new_up_df, replace=False, n_samples=len(new_up_df), random_state=3242)


class DTV1(TTTDT):
    def __init__(self):
        super(DTV1, self).__init__()

    def run(self):
        # self.test()
        # self.process_dt_0727()
        # self.dt_0729()
        # self.dt_0805()
        # self.dt_0806()
        # self.dt_0809()
        # self.dt_1109()
        # self.dt_1109_2()
        # self.dt_1109_3()
        # self.dt_1110()
        # self.dt_1111()
        self.train__()

    def process_dt_0727(self):
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/ttt/data/raw_0727.xlsx")  # type:pd.DataFrame
        df.dropna(inplace=True)
        df = df[~df['therapy_labels'].isin(["['']"])]
        df['criteria'] = df['criteria'].apply(TTTDT.get_inclusion_criteria)
        df['therapy_labels'] = df['therapy_labels'].apply(
            lambda x: ','.join(ast.literal_eval(x) if x != '[None]' else ','))
        Utilfuncs.to_excel(df, './data/dt_0728.xlsx', 'raw')

        df['therapy_labels_en'] = df['therapy_labels'].apply(TTTDT.get_target_text_en)
        df['therapy_labels_num'] = df['therapy_labels'].apply(TTTDT.get_target_text_num)
        df['input_text'] = 'Title: ' + df['study_title'] + ' Criteria: ' + df['criteria']
        df = df[['nct_id', 'input_text', 'therapy_labels', 'therapy_labels_en', 'therapy_labels_num']]
        train_df, eval_df = TTTDT.cut_train_eval(df)
        Utilfuncs.to_excel(df, './data/dt_0728.xlsx', 'all')
        Utilfuncs.to_excel(train_df, './data/dt_0728.xlsx', 'train')
        Utilfuncs.to_excel(eval_df, './data/dt_0728.xlsx', 'eval')

    def dt_0729(self):
        train_df = pd.read_excel('./data/dt_0728.xlsx', 'train')
        train_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
        train_df['prefix'] = 'classification'
        df = DTUtils.up_sampling_one_prefix(train_df, delimiter=',')
        Utilfuncs.to_excel(df, './data/dt_0728.xlsx', 'train_up')

    @staticmethod
    def turn_df_to_sentence_pair(df):
        def apply_text(text: str):
            return text.replace(' Criteria: ', ' | Criteria: ', 1)

        df['text_a'] = df['input_text'].apply(apply_text)
        ta = df['text_a'].tolist()
        tb = df['therapy_labels_en'].tolist()
        df_li = []
        for a, b in zip(ta, tb):

            if b == ',':
                therapy = ['No therapy']
            elif ',' in b:
                therapy = b.split(',')
            else:
                therapy = [b]
            for c in ClassificationListEN:
                if c in therapy:
                    df_li.append({'text_a': a, 'text_b': 'Therapy Labels: ' + c, 'labels': 1})
                else:
                    df_li.append({'text_a': a, 'text_b': 'Therapy Labels: ' + c, 'labels': 0})

        df = pd.DataFrame(df_li)
        return df

    def dt_0805(self):
        train_df = pd.read_excel("./data/dt_0728.xlsx", 'train')
        train_df = DTV1.turn_df_to_sentence_pair(train_df)
        Utilfuncs.to_excel(train_df, './data/dt_0806.xlsx', 'train')

        eval_df = pd.read_excel("./data/dt_0728.xlsx", 'eval')
        eval_df = DTV1.turn_df_to_sentence_pair(eval_df)
        Utilfuncs.to_excel(eval_df, './data/dt_0806.xlsx', 'eval')

        train_up = TTTDT.up_sampling(train_df)
        Utilfuncs.to_excel(train_up, './data/dt_0806.xlsx', 'train_up')

    @staticmethod
    def turn_df_mutil_label(df):
        def apply_texts(text: str):
            return text.replace(' Criteria: ', ' | Criteria: ', 1)

        df['text'] = df['input_text'].apply(apply_texts)

        def apply_labels(labels: str):
            if labels == ',':
                res = [0] * (int(len(ClassificationListEN) - 1))
                # res.append(1)
                return res
            labels = labels.split(',')
            res = []
            all_ids = list(range(len(ClassificationListEN) - 1))

            for i in all_ids:
                if str(i) in labels:
                    res.append(1)
                else:
                    res.append(0)
            return res

        df['labels'] = df['therapy_labels_num'].apply(apply_labels)
        df = df[['text', 'labels']]
        return df

    def dt_0806(self):
        train_df = pd.read_excel("./data/dt_0728.xlsx", 'train')
        train_df = DTV1.turn_df_mutil_label(train_df)
        Utilfuncs.to_excel(train_df, './data/dt_0809.xlsx', 'train')

        eval_df = pd.read_excel("./data/dt_0728.xlsx", 'eval')
        eval_df = DTV1.turn_df_mutil_label(eval_df)
        Utilfuncs.to_excel(eval_df, './data/dt_0809.xlsx', 'eval')

    def dt_0809(self):
        def apply_texts(text: str):
            return text.replace(' Criteria: ', ' | Criteria: ', 1)

        train_df = pd.read_excel("./data/dt_0728.xlsx", 'train')
        train_df['input_text'] = train_df['input_text'].apply(apply_texts)
        Utilfuncs.to_excel(train_df, './data/dt_0809_2.xlsx', 'train')

        eval_df = pd.read_excel("./data/dt_0728.xlsx", 'eval')
        eval_df['input_text'] = eval_df['input_text'].apply(apply_texts)
        Utilfuncs.to_excel(eval_df, './data/dt_0809_2.xlsx', 'eval')

    def test(self):
        # df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/ttt/data/v1/raw_1109.xlsx")[1800:]
        # df['criteria'] = df['criteria'].apply(TTTDT.get_inclusion_criteria)
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/ttt/data/v1/process_1109.xlsx")
        c = df['criteria'].tolist()[0:100]
        for i in c:
            print(repr(i))

    def dt_1109(self):
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/ttt/data/v1/raw_1109.xlsx")
        df = df[['nct_id', 'therapy_labels', 'esid_cie', 'esid', 'study_title', 'criteria']]
        df['criteria'] = df['criteria'].apply(TTTDT.get_inclusion_criteria)  # type: pd.DataFrame
        df['therapy_labels'] = df['therapy_labels'].apply(TTTDT.deal_with_labels)
        Utilfuncs.to_excel(df, './data/v1/process_1109.xlsx', 'raw')  # 预处理的数据

    def dt_1109_2(self):
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/ttt/data/v1/raw_1109_2.xlsx")
        df = df[['esid', 'nct_id', '治疗类型', 'study_title', 'criteria']]
        df['criteria'] = df['criteria'].apply(TTTDT.get_inclusion_criteria)  # type: pd.DataFrame

        df['therapy_labels'] = df['治疗类型'].apply(TTTDT.deal_with_labels2)
        Utilfuncs.to_excel(df, './data/v1/process_1109.xlsx', 'raw2')

    def dt_1109_3(self):
        # 将上面两个合并起来
        df1 = pd.read_excel('./data/v1/process_1109.xlsx', 'raw')
        df1 = df1[['nct_id', 'study_title', 'criteria', 'therapy_labels']]
        df2 = pd.read_excel('./data/v1/process_1109.xlsx', 'raw2')
        df2 = df2[['nct_id', 'study_title', 'criteria', 'therapy_labels']]

        df = pd.concat([df1, df2], ignore_index=True)
        train_df, eval_df = TTTDT.cut_train_eval(df)
        Utilfuncs.to_excel(df, './data/v1/processed_1109.xlsx', 'all')
        Utilfuncs.to_excel(train_df, './data/v1/processed_1109.xlsx', 'train')
        Utilfuncs.to_excel(eval_df, './data/v1/processed_1109.xlsx', 'eval')

    def dt_1110(self):
        # 把上面数据变成mt5格式
        df = pd.read_excel( './data/v1/processed_1109.xlsx', 'train') # type:pd.DataFrame
        df.fillna(',',inplace=True)
        df['therapy_labels_num'] = df['therapy_labels'].apply(TTTDT.get_target_text_num)
        df['input_text'] = 'Title: ' + df['study_title'] + '\n | Criteria: ' + df['criteria']
        df = df[['nct_id', 'input_text', 'therapy_labels', 'therapy_labels_num']]
        Utilfuncs.to_excel(df, './data/v1/train_1110.xlsx', 'train_mt5')


        df = pd.read_excel('./data/v1/processed_1109.xlsx', 'eval')  # type:pd.DataFrame
        df.fillna(',', inplace=True)
        df['therapy_labels_num'] = df['therapy_labels'].apply(TTTDT.get_target_text_num)
        df['input_text'] = 'Title: ' + df['study_title'] + '\n | Criteria: ' + df['criteria']
        df = df[['nct_id', 'input_text', 'therapy_labels', 'therapy_labels_num']]
        Utilfuncs.to_excel(df, './data/v1/train_1110.xlsx', 'eval_mt5')

        # df['therapy_labels_num'] = df['therapy_labels'].apply(TTTDT.get_target_text_num)
        # df['input_text'] = 'Title: ' + df['study_title'] + ' Criteria: ' + df['criteria']
        #
        #
        # df['input_text'] = 'Title: ' + df['study_title'] + ' Criteria: ' + df['criteria']
        # df = df[['nct_id', 'input_text', 'therapy_labels', 'therapy_labels_en', 'therapy_labels_num']]

    def dt_1110_2(self):
        # 把上面数据变成sentence-pair格式
        df = pd.read_excel('./data/v1/processed_1109.xlsx', 'train')

        df['therapy_labels_num'] = df['therapy_labels'].apply(TTTDT.get_target_text_num)
        df['input_text'] = 'Title: ' + df['study_title'] + ' | Criteria: ' + df['criteria']
        df = df[['nct_id', 'input_text', 'therapy_labels', 'therapy_labels_num']]


    def dt_1111(self):
        # 把上面数据变成sentence-pair格式
        df = pd.read_excel('./data/v1/train_1110.xlsx', 'train_mt5')
        all_therapies = ClassificationListEN[0:-1]
        li = []
        for _,every_sample in tqdm(df.iterrows()):
            if every_sample['therapy_labels']==',':
                for j in all_therapies:
                    li.append({'text_a':every_sample['input_text'],'text_b':j,'labels':0})
            else:
                true_labels = every_sample['therapy_labels'].split(',')
                if len(true_labels)==1:
                    t_label = 1 # ==
                else:
                    t_label = 2 # in
                for j in all_therapies:
                    if j in true_labels:
                        li.append({'text_a': every_sample['input_text'], 'text_b': j, 'labels': t_label})
                    else:
                        li.append({'text_a': every_sample['input_text'], 'text_b': j, 'labels': 0})
        print(len(li))
        df = pd.DataFrame(li)
        df.to_json(f'./data/v1/train_1111.json.gz', compression='gzip')

    def train__(self):
        df = pd.read_json(f'./data/v1/train_1111.json.gz', compression='gzip')
        df['labels'] = df['labels'].apply(lambda x: 0 if x == 0 else 1)
        df = TTTDT.up_sampling(df)
        print(len(df))
        df.to_json(f'./data/v1/train_1111_up.json.gz', compression='gzip')

if __name__ == '__main__':
    # DTV1().run()
    pass
