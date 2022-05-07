# -*- coding: UTF-8 -*-
"""
Description : 
"""

url = "http://101.201.249.176:9103/predict/"
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_excel("/home/zyl/PharmAI/pharm_ai/zz/data/v7/政策库(1).xls")
# df = df[~df['是否中标资讯'].isna()]
# df = df[~df['是否中标资讯(nlp)'].isna()]
# true_labels = df['是否中标资讯'].tolist()
# predicted_labels = df['是否中标资讯(nlp)'].tolist()
# report_table = classification_report(true_labels, predicted_labels, digits=4)
# print(report_table)

# bu
df.dropna(subset=['中标项目(魔方)', '是否中标资讯(nlp)'], inplace=True)
l1 = df['中标项目(魔方)'].tolist()
l2 = df['中标项目(nlp)'].tolist()
all_list = set(l1)
all_list.update(set(l2))
all_list = list(all_list)
true_labels = []
predicted_labels = []
for i, j in zip(l1, l2):
    if i == j:
        true_labels.append('1')
        predicted_labels.append('1')
    else:
        true_labels.append('1')
        predicted_labels.append('0')

        true_labels.append('0')
        predicted_labels.append('1')

# for i in range(10000):
#     true_labels.append('0')
#     predicted_labels.append('0')
report_table = classification_report(true_labels, predicted_labels, digits=4)
print(report_table)

# titles =

# df = pd.read_excel("./data/v2_5/processed_dt_0430.xlsx", sheet_name='raw')  # type:pd.DataFrame
# df.drop_duplicates(subset=['input_text'],inplace=True)
# c = df['id'].value_counts()
# ids = c[c>6].index.tolist()
# df = df[df['id'].isin(ids)]
# df.to_excel('./data/v2.6/test_article.xlsx')
# print(len(ids))

# df = pd.read_excel('./data/v2.6/test_article.xlsx')
# # df.rename(columns={'target_text': 'target_text_label'}, inplace=True)
# # df['to_predict'] = df['prefix'].str.cat(df['input_text'], sep=': ')  # type:pd.DataFrame
# to_predict = df['input_text'].to_list()  # type:list

# to_predict =[
#     ["贵州", "关于取消四川康福来药业集团有限公司等企业部分中标药品中标（挂网）资格的通知", "2020-06-11"],
#     ["云南", "关于药品生产企业名称变更的公示", "2020-09-21"],
#     ["广西", "关于公示2020年度非免疫规划疫苗集中采购目录的通知", "2020-04-27"],
#     ["浙江", "关于拟恢复部分药品配送企业网上药品配送资格的公示", "2020-11-26"],
#     ["甘肃", "关于公示2020年度甘肃省非免疫规划疫苗阳光挂网复议结果的通知", "2020-09-18"]
# ]
# t1 = time.time()
# res=requests.post(url=url, json={"texts": to_predict}).json()
# print(res)
# t2 = time.time()
# print(f'time:{(t2-t1)/len(to_predict)}s * {len(to_predict)} = {t2-t1}',t2-t1)
# # print(10000/24)
