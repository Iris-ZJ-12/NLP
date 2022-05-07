import pickle
from pharm_ai.util.ESUtils6 import get_page
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.config import ConfigFilePaths as cfp
import os

from loguru import logger
import json
import simplejson
import bz2
import langid
from math import factorial
import h5py
excel = '20200424-claim标注数据-v2.xlsx'
h5 = '/home/gwl/disk_a/Projects/PharmAI/pharm_ai/patent_claims/20200424-claim标注数据-v2.h5'


def dt0423():
    # df = pd.read_excel(excel, 'original').dropna(how='all')
    # df.to_hdf(h5, 'original')
    df = pd.read_hdf(h5)
    groups = df.groupby('patent_code')
    claims_non_claims_data = []
    claim_types_data = []
    parents_data = []
    for n, g in groups:
        # print(n)
        # print(' ')
        current_claims = []
        for i, r in g.iterrows():
            claim_num = r['claim_num']
            if type(r['claim_text']) != str:
                print(type(r['claim_text']))
            claim_text = ' '.join(r['claim_text'].split())
            claim_type_pre = r['claim_type_pre']
            claim_class_pre = r['claim_class_pre']
            if claim_class_pre == 'claim':
                # print(claim_text)
                # print(claim_type_pre)
                claims_non_claims_dt_one = [claim_text, 'claim']
                claim_types_dt_one = [claim_text, claim_type_pre]
                claim_types_data.append(claim_types_dt_one)
                claims_non_claims_data.append(claims_non_claims_dt_one)
                # print('~'*50)
            else:
                # print(claim_text)
                try:
                    parents_claims = str(r['claim_ref_pre'])
                    parents_claims = [int(i) for i in parents_claims.split('/')]
                except:
                    print('Oops ...')
                    continue
                # print(claim_ref_pre)
                # print(claim_num)
                current_claims = list(range(1, claim_num))
                non_parents_claims = list(set(parents_claims).symmetric_difference(set(current_claims)))

                for parent_claim in parents_claims:
                    parents_dt_one = [claim_text, str(parent_claim), 1]
                    parents_data.append(parents_dt_one)
                for non_parent_claim in non_parents_claims:
                    parents_dt_one = [claim_text, str(non_parent_claim), 0]
                    parents_data.append(parents_dt_one)
                claims_non_claims_dt_one = [claim_text, 'claim-dependent']
                claims_non_claims_data.append(claims_non_claims_dt_one)
        #         print('@'*50)
        #     print('-'*50)
        #     print(' ')
        # print('*' * 50)
        # print('*' * 50)

    claims_non_claims_data = pd.DataFrame(claims_non_claims_data)
    # u.to_excel(claims_non_claims_data, excel, 'all_claims_non_claims_0426')

    groups = claims_non_claims_data.groupby(1)
    print('len(claims_non_claims_data): ' + str(len(claims_non_claims_data)))
    for n, g in groups:
        print(n)
        print(len(g))
        print(' ')
    print('*' * 50)

    claim_types_data = pd.DataFrame(claim_types_data)
    # u.to_excel(claim_types_data, excel, 'all_claim_types_data_0426')
    groups = claim_types_data.groupby(1)
    print('len(claim_types_data): ' + str(len(claim_types_data)))
    for n, g in groups:
        print(n)
        print(len(g))
        print(' ')
    print('*' * 50)

    parents_data = pd.DataFrame(parents_data)
    # u.to_excel(parents_data, excel, 'all_parents_data_0426')
    groups = parents_data.groupby(2)
    print('len(parents_data): ' + str(len(parents_data)))
    for n, g in groups:
        print(n)
        print(len(g))
        print(' ')
    print('*' * 50)


def train_test_claims():
    all_claims_non_claims_0426 = pd.read_hdf(h5, 'all_claims_non_claims_0426')
    all_claims_non_claims_0426[1] = \
        all_claims_non_claims_0426[1].map({'claim': 1, 'claim-dependent': 0})
    groups = all_claims_non_claims_0426.groupby(1)
    claim = groups.get_group(1).sample(frac=1, random_state=123)
    claim_test = claim.iloc[:50]
    claim_train = claim.iloc[50:]
    claim_train = resample(claim_train, replace=True,
                           n_samples=4655, random_state=123)
    claim_dependent = groups.get_group(0).sample(frac=1, random_state=123)
    claim_dependent_test = claim_dependent.iloc[:50]
    claim_test = pd.concat([claim_test, claim_dependent_test])\
        .sample(frac=1, random_state=123)
    claim_dependent_train = claim_dependent.iloc[50:]
    claim_train = pd.concat([claim_train, claim_dependent_train])\
        .sample(frac=1, random_state=123)
    return claim_train, claim_test


def train_test_claim_type():
    all_claim_types_data_0426 = pd.read_hdf(h5, 'all_claim_types_data_0426')\
        .sample(frac=1, random_state=123)
    claim_type_mapper = {'分析方法': 0, '制剂': 1, '制备方法': 2, '包材': 3,
                         '化合物': 4, '医疗器械': 5, '医药用途': 6, '晶型': 7,
                         '盐': 8, '组合物': 9, '给药装置': 10}
    all_claim_types_data_0426[1] = all_claim_types_data_0426[1].map(claim_type_mapper)
    groups = all_claim_types_data_0426.groupby(1)
    claim_type_test = []
    claim_type_train = []
    for n, g in groups:
        g_test = g.iloc[:8]
        g_train = g.iloc[8:]
        if n != 6:
            g_train = resample(g_train, replace=True,
                               n_samples=310, random_state=123)
        claim_type_test.append(g_test)
        claim_type_train.append(g_train)
    claim_type_test = pd.concat(claim_type_test).sample(frac=1, random_state=123)
    claim_type_train = pd.concat(claim_type_train).sample(frac=1, random_state=123)
    return claim_type_train, claim_type_test


def train_test_parents():
    all_parents_data_0426 = pd.read_hdf(h5, 'all_parents_data_0426')\
        .sample(frac=1, random_state=123)
    all_parents_data_0426.rename(columns={0: 'text_a', 1: 'text_b', 2: 'labels'},
                                 inplace=True)
    all_parents_data_0426['text_b'] = all_parents_data_0426['text_b'].astype(str)
    all_parents_data_0426['labels'] = all_parents_data_0426['labels'].astype(int)
    groups = all_parents_data_0426.groupby('labels')
    parents = groups.get_group(1)
    parents_test = parents.iloc[:50]
    parents_train = parents.iloc[50:]
    parents_train = resample(parents_train, replace=True,
                             n_samples=99844, random_state=123)
    non_parents = groups.get_group(0)
    non_parents_test = non_parents.iloc[:50]
    non_parents_train = non_parents.iloc[50:]
    parents_test = pd.concat([parents_test, non_parents_test])\
        .sample(frac=1, random_state=123)
    parents_train = pd.concat([parents_train, non_parents_train])\
        .sample(frac=1, random_state=123)
    return parents_train, parents_test


def dt0618():
    p = 'drug_patent_claim.json'
    dt = json.load(open(p, 'r'))
    dt_new = dict()
    for k, v in dt.items():
        v_new = []
        for ele in v:
            txt = u.remove_html_tags(ele[1])
            v_new.append([ele[0], txt])
        dt_new[k] = v_new
    p = 'patent_claims-batch-1.json'
    f = open(p, 'w', encoding='utf-8')
    json.dump(dt_new, f, indent=4, ensure_ascii=False)

    print(len(dt))
    res = []
    # d1 = dict(list(data.items())[:10000])
    # res.append(d1)
    # d2 = dict(list(data.items())[10000:20000])
    # res.append(d2)
    d3 = dict(list(dt.items())[20000:30000])
    res.append(d3)
    d4 = dict(list(dt.items())[30000:40000])
    res.append(d4)
    d5 = dict(list(dt.items())[40000:50000])
    res.append(d5)
    d6 = dict(list(dt.items())[50000:60000])
    res.append(d6)
    d7 = dict(list(dt.items())[60000:70000])
    res.append(d7)
    d8 = dict(list(dt.items())[70000:80000])
    res.append(d8)
    d9 = dict(list(dt.items())[80000:90000])
    res.append(d9)
    d10 = dict(list(dt.items())[90000:100000])
    res.append(d10)
    d11 = dict(list(dt.items())[100000:])
    res.append(d11)
    return res


def dt0622():
    # p = 'exception0619.json'
    # data = json.load(open(p, 'r'))
    # res = dict()
    # for k, v in data.items():
    #     l = len(v)
    #     if l >= 500:
    #         res[k] = v
    # p = 'exception-huge.json'
    # f = open(p, 'w', encoding='utf-8')
    # json.dump(res, f, indent=4, ensure_ascii=False)

    # p = 'exception0619.json'
    # data = json.load(open(p, 'r'))
    # li = []
    # for k, v in data.items():
    #     v_new = []
    #     for ele in v:
    #         txt = u.remove_html_tags(ele[1])
    #         v_new.append([ele[0], txt])
    #     li.append({k: v_new})
    p = 'exception0622.p'
    # f = open(p, 'wb')
    # pickle.dump(li, f)
    f = open(p, 'rb')
    dt = pickle.load(f)
    return dt

def dt1104():
    p = "drug_patent_claim-20201104.json"
    with open(p, 'r') as json_file:
        dt = json.load(json_file)
    dt_new={k:[[ele[0], u.remove_html_tags(ele[1])] for ele in v] for k,v in dt.items()}
    res = [dict(list(dt_new.items())[i:min(i+10000, len(dt_new))])
           for i in range(0, len(dt_new), 10000)]
    return res

import signal
import time


def mytest():

    def timeout_handler(signum, frame):  # Custom signal handler
        raise Exception

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    for i in range(5):
        # Start the timer. Once 5 seconds are over, a SIGALRM signal is sent.
        signal.alarm(2)
        # This try/except loop ensures that
        #   you'll catch TimeoutException when it's sent.
        try:
            if i == 2 or i == 4:
                time.sleep(4)
        except:
            print(i)
            continue  # continue the for loop if function A takes more than 2 second
        else:
            # Reset the alarm
            signal.alarm(0)


def dt0623():
    p = 'res20200618-1.json'
    dt1 = json.load(open(p, 'r'))
    print(len(dt1))
    p = 'exception0622.p'
    f = open(p, 'rb')
    dt = pickle.load(f)
    dt2 = dict()
    for d in dt:
        dt2.update(d)
    failed = np.setdiff1d(list(dt2.keys()), list(dt1.keys()))
    print(failed)
    print(len(failed))
    dt = {k: v for k, v in dt2.items() if k in failed}
    p = 'failed.json'
    f = open(p, 'w', encoding='utf-8')
    json.dump(dt, f, indent=4, ensure_ascii=False)


def dt0623_final():
    p = 'patent_claims-batch-1.json'
    dt = json.load(open(p, 'r'))
    print(len(dt))
    res = []
    d3 = dict(list(dt.items())[20000:30000])
    res.append(d3)
    # d4 = dict(list(data.items())[30000:40000])
    # res.append(d4)
    d5 = dict(list(dt.items())[40000:50000])
    res.append(d5)
    d6 = dict(list(dt.items())[50000:60000])
    res.append(d6)
    d7 = dict(list(dt.items())[60000:70000])
    res.append(d7)
    d8 = dict(list(dt.items())[70000:80000])
    res.append(d8)
    d9 = dict(list(dt.items())[80000:90000])
    res.append(d9)
    d10 = dict(list(dt.items())[90000:100000])
    res.append(d10)
    d11 = dict(list(dt.items())[100000:])
    res.append(d11)
    return res


def dt0812():
    p = 'drug_patent_claim-20200812.json'
    dt = json.load(open(p, 'r'))
    print(len(dt))
    d = dict(list(dt.items())[:500])
    return d


def dt0811():
    p = 'patent_claims-batch-1.json'
    dt = json.load(open(p, 'r'))
    d3 = dict(list(dt.items())[:500])
    return d3

def dt0701():
    p = 'res20200618-10.json'
    dt = simplejson.load(open(p, 'r'))
    print(len(dt))


def dt0819_2():
    f = 'a-2020-08-19 17.41.10.xlsx'
    df1 = pd.read_excel(f, '1').dropna()

    df1a = df1['claim_class_act']
    df1p = df1['claim_class_pre']
    print(classification_report(df1a, df1p, digits=4))

    df2 = pd.read_excel(f, '2').dropna()
    df2a = df2['claim_type_act']
    df2p = df2['claim_type_pre']
    print(classification_report(df2a, df2p, digits=4))

    df3 = pd.read_excel(f, '3').dropna()
    df3a = df3['claim_ref_act']
    df3p = df3['claim_ref_pre']
    print(classification_report(df3a, df3p, digits=4))


def dt0806():
    f = 'patent_claims_data-20200804.xlsx'
    h5 = 'train_test_20200806-2.h5'
    df1 = pd.read_excel(f, '1').dropna().\
        sample(frac=1, random_state=123)
    df1[1] = df1[1].\
        map({'independent-claim': 1, 'dependent-claim': 0})

    train1 = df1.iloc[:-200]
    grs1 = train1.groupby(1)
    t1 = []
    for n, g in grs1:
        if n == 1:
            g = resample(g, replace=True,
                         n_samples=20000, random_state=123)
        t1.append(g)
    train1 = pd.concat(t1).sample(frac=1, random_state=123)
    train1.to_hdf(h5, 'train1')
    test1 = df1.iloc[-200:]
    test1.to_hdf(h5, 'test1')

    df2 = pd.read_excel(f, '2').dropna().sample(frac=1, random_state=123)
    claim_type_mapper = {'包材':0,'分析方法':1,'给药装置':2,'化合物':3,'晶型':4,
                         '其他':5,'前药':6,'溶剂化物':7,'细胞':8,'序列':9,'盐':10,
                         '医疗器械':11,'医药用途':12,'医药中间体':13,'载体':14,
                         '诊断试剂':15,'酯':16,'制备方法':17,'制剂':18,
                         '制药设备':19,'组合物':20, '#': 21}
    df2[1] = df2[1].map(claim_type_mapper)
    train2 = df2.iloc[:-100]
    grs2 = train2.groupby(1)
    t2 = []
    for n, g in grs2:
        if n != 12:
            g = resample(g, replace=True,
                         n_samples=710, random_state=123)
        t2.append(g)
    train2 = pd.concat(t2).sample(frac=1, random_state=123)
    train2.to_hdf(h5, 'train2')
    test2 = df2.iloc[-200:]
    test2.to_hdf(h5, 'test2')

    df3 = pd.read_excel(f, '3').dropna().sample(frac=1, random_state=123)
    train3 = df3.iloc[:-200]
    test3 = df3.iloc[-200:]
    grs3 = train3.groupby(1)
    t3 = []
    for n, g in grs3:
        if n == 1:
            g = resample(g, replace=True,
                         n_samples=390000, random_state=123)
        t3.append(g)
    train3 = pd.concat(t3).sample(frac=1, random_state=123)
    train3.to_hdf(h5, 'train3')
    test3.to_hdf(h5, 'test3')


def train_test():
    h5 = 'train_test_20200806-2.h5'
    train1 = pd.read_hdf(h5, 'train1')
    test1 = pd.read_hdf(h5, 'test1')
    train2 = pd.read_hdf(h5, 'train2')
    test2 = pd.read_hdf(h5, 'test2')
    train3 = pd.read_hdf(h5, 'train3')
    train3[0] = train3[0].astype(str)
    train3[1] = train3[1].astype(str)
    train3.columns = ['text_a', 'text_b', 'labels']
    test3 = pd.read_hdf(h5, 'test3')
    test3[0] = test3[0].astype(str)
    test3[1] = test3[1].astype(str)
    test3.columns = ['text_a', 'text_b', 'labels']
    return train1, test1, train2, test2, train3, test3


def dt0819():
    p = 'test-20200812.json'
    dt = json.load(open(p, 'r'))
    dt = list(dict(list(dt.items())).keys())
    df = pd.DataFrame({'esid': dt})
    f = '500esid-test-20200812.xlsx'
    u.to_excel(df, f, 'esid')


def dt0824():
    f = 'a-2020-08-19 17.41.10.xlsx'
    # df1 = pd.read_excel(f, '1').dropna()
    # df1[1] = df1[1].\
    #     map({'independent-claim': 1, 'dependent-claim': 0})
    #
    # df2 = pd.read_excel(f, '2').dropna()
    # ff = '20200819-独权数据.xlsx'
    # df2_2 = pd.read_excel(ff, '2')
    claim_type_mapper = {'包材':0,'分析方法':1,'给药装置':2,'化合物':3,'晶型':4,
                         '其他':5,'前药':6,'溶剂化物':7,'细胞':8,'序列':9,'盐':10,
                         '医疗器械':11,'医药用途':12,'医药中间体':13,'载体':14,
                         '诊断试剂':15,'酯':16,'制备方法':17,'制剂':18,
                         '制药设备':19,'组合物':20, '#': 21}
    # df2[1] = df2[1].map(claim_type_mapper)
    # df2_2[1] = df2_2[1].map(claim_type_mapper)

    df3 = pd.read_excel(f, '3').dropna()

    train1, test1, train2, test2, train3, test3 = train_test()
    # train1 = pd.concat([train1, df1])
    # train1.drop_duplicates(inplace=True)
    # 37023
    # print(len(train1))
    f = 'patent-claims-data-20200824.xlsx'
    # u.to_excel(train1, f, 'train1')

    # 16087
    # train2 = pd.concat([train2, df2, df2_2])
    # train2.drop_duplicates(inplace=True)
    # print(len(train2))
    # u.to_excel(train2, f, 'train2')

    # 552186
    train3 = pd.concat([train3, df3])
    train3.drop_duplicates(inplace=True)
    # print(len(train3))
    # u.to_excel(train3, f, 'train3')

    # grs1 = train1.groupby(1)
    # t1 = []
    # for n, g in grs1:
    #     print(len(g))
    #     if n == 1:
    #         g = resample(g, replace=True,
    #                      n_samples=21448, random_state=123)
    #     t1.append(g)
    # train1 = pd.concat(t1).sample(frac=1, random_state=123)
    # h5 = 'train-test-20200824.h5'
    # train1.to_hdf(h5, 'train1')

    # grs2 = train2.groupby(1)
    # t2 = []
    # for n, g in grs2:
    #     # print(n)
    #     # print(len(g))
    #     # print('*'*50)
    #     if n != 21:
    #         g = resample(g, replace=True,
    #                      n_samples=6764, random_state=123)
    #     t2.append(g)
    # train2 = pd.concat(t2).sample(frac=1, random_state=123)
    h5 = 'train-test-20200824.h5'
    # train2.to_hdf(h5, 'train2')

    grs3 = train3.groupby('labels')
    t3 = []
    for n, g in grs3:
        # print(n)
        # print(len(g))
        # print('*'*50)
        if n == 1:
            g = resample(g, replace=True,
                         n_samples=499885, random_state=123)
        t3.append(g)
    train3 = pd.concat(t3).sample(frac=1, random_state=123)
    train3.to_hdf(h5, 'train3')

    test1.to_hdf(h5, 'test1')
    test2.to_hdf(h5, 'test2')
    test3.to_hdf(h5, 'test3')


def train_test0824():
    h5 = cfp.project_dir + '/patent_claims/train-test-20200824.h5'
    train1 = pd.read_hdf(h5, 'train1')
    test1 = pd.read_hdf(h5, 'test1')

    train2 = pd.read_hdf(h5, 'train2')
    train2.drop_duplicates(inplace=True)
    test2 = pd.read_hdf(h5, 'test2')
    dt2 = pd.concat([train2, test2]).sample(frac=1, random_state=123)
    grs2 = dt2.groupby(1)
    dt2.drop(grs2.get_group(21).index)
    # cancel = pd.read_excel('cancel.xlsx', '1').dropna()
    # cancel.to_hdf('cancel.h5', '1')
    cancel = pd.read_hdf(cfp.project_dir + '/patent_claims/cancel.h5')
    dt2 = pd.concat([dt2, cancel]).sample(frac=1, random_state=123)
    grs2 = dt2.groupby(1)
    train2_ = []
    test2_ = []

    for n, g in grs2:
        # print(n)
        # print(len(g))
        # print('*'*50)
        if len(g) < 100:
            train2_.append(g.iloc[:-2])
            test2_.append(g.iloc[-2:])
        else:
            train2_.append(g.iloc[:-25])
            test2_.append(g.iloc[-25:])

    train2 = pd.concat(train2_).sample(frac=1, random_state=123)
    grs2 = train2.groupby(1)
    t2 = []
    for n, g in grs2:
        # print(n)
        # print(len(g))
        # print('*'*50)
        if len(g) < 500:
            g = resample(g, replace=True,
                         n_samples=500, random_state=123)
        t2.append(g)
    train2 = pd.concat(t2).sample(frac=1, random_state=123)

    test2 = pd.concat(test2_).sample(frac=1, random_state=123)

    train3 = pd.read_hdf(h5, 'train3')
    train3['text_a'] = train3['text_a'].astype(str)
    train3['text_b'] = train3['text_b'].astype(str)
    test3 = pd.read_hdf(h5, 'test3')
    test3['text_a'] = test3['text_a'].astype(str)
    test3['text_b'] = test3['text_b'].astype(str)
    print(len(train1))
    print(len(test1))
    print(len(train2))
    print(len(test2))
    print(len(train3))
    print(len(test3))
    return train1, test1, train2, test2, train3, test3

def dt0121():
    """Export data with low accuracy. by FZQ"""
    from sklearn.metrics import accuracy_score
    with open('PC_data_0120.json','r') as f:
        raw = json.load(f)
    raw_df = pd.DataFrame.from_records(raw).dropna()
    minor_types=['其他', '诊断试剂', '包材', '前药', '医药中间体', '溶剂化物', '酯', '制药设备']
    select_df = raw_df[raw_df['claim_type_act'].isin(minor_types)].sort_values(
        by=['publication_docdb_comb','claim_num'])
    select_df.to_excel('patent_claims 待校验小量类型 by FZQ.xlsx', index=False)


class Preprocessor:
    def __init__(self, version):
        self.h5 = 'data.h5'
        self.version=version

    def get_h5_key(self, term='raw', version=None):
        ver = self.version.replace('.','-') if not version else version.replace('.','-')
        res = f'{ver}/{term}'
        return res

    def save_to_h5(self, df, term='raw'):
        key = self.get_h5_key(term=term)
        df.to_hdf(self.h5, key)
        logger.info('Datasets have been saved to {} with key "{}"', self.h5, key)

    def get_from_h5(self, term='raw', version=None):
        """
        :param term: ['raw', 'train', 'eval']
        :param version: possible other version
        :return: Dataframe
        """
        key = self.get_h5_key(term=term, version=version)
        df = pd.read_hdf(self.h5, key)
        logger.info('Read "{}"', key)
        return df

    def get_train_eval_datasets(self, train_sampling=1, eval_sampling=1, to_excel_path=None):
        train_df: pd.DataFrame = self.get_from_h5('train')
        eval_df: pd.DataFrame = self.get_from_h5('eval')
        if 0<train_sampling<=1:
            train_df = train_df.sample(frac=train_sampling, random_state=self.random_state)
        elif train_sampling>1:
            train_df = train_df.sample(n=train_sampling, random_state=self.random_state)
        if 0<eval_sampling<=1:
            eval_df = eval_df.sample(frac=eval_sampling, random_state=self.random_state)
        elif eval_sampling>1:
            eval_df = eval_df.sample(n=eval_sampling, random_state=self.random_state)
        if not to_excel_path:
            return train_df, eval_df
        else:
            with pd.ExcelWriter(to_excel_path) as writer:
                if train_df.shape[0]<50000:
                    train_df.to_excel(writer, 'train', index=False)
                else:
                    train_df.head(50000).to_excel(writer, 'train', index=False)
                    logger.warning('Training size: {}, saving 50000', train_df.shape[0])
                eval_df.to_excel(writer, 'eval', index=False)
            logger.info('Training and eval datasets saved to "{}"', to_excel_path)


class GenerativeDataProcessor(Preprocessor):
    def __init__(self, version='v2.2'):
        """
        :param version:
        -v2/v3:
            - v2.1: Reuse older dataset.
            - v2.2: Generative mt5 dataset, multiple tasks on one prefix.
            - v2.3: Each prefix refers to individual task.
            - v2.4: Add task of classifying corporation person, in addition to v2.3.
                Task 1-4: (300000, 300000, 300000, 30000)
            - v2.6: Use v2.4 dataset but task 1-4: (300000, 300000, 300000, 300000). (note task4 use more)
                and balance label=1 and label=0.
        - v4:
            - v4.0: task 4 add new classes. task1-4: (600000, 600000, 600000, 600000)
        """
        self.version = version
        self.h5 = 'data.h5'
        self.h5_copy = 'data2.h5'
        self.random_state = 120
        self.is_claim_mapper = {'independent-claim': 1, 'dependent-claim': 0}
        self.claim_type_mapper = {'包材': 0, '分析方法': 1, '给药装置': 2, '化合物': 3, '晶型': 4, '其他': 5, '前药': 6, '溶剂化物': 7,
                                  '细胞': 8, '序列': 9, '盐': 10, '医疗器械': 11, '医药用途': 12, '医药中间体': 13, '载体': 14, '诊断试剂': 15,
                                  '酯': 16, '制备方法': 17, '制剂': 18, '制药设备': 19, '组合物': 20, '#': 21}
        self.minor_type_mapper = {'法人':['企业', '学校/研究机构', '政府机构', '医院'], '自然人':['个人']}

    def get_from_h5(self, term='raw', version=None, return_multiple_parts=False):
        """
        :param return_multiple_parts: version>v2.4 might return two parts
        """
        if self.version in ['v2.4', 'v2.6', 'v4.0'] and term == 'raw' and return_multiple_parts:
            df1 = self.get_from_h5(version='v2.3')
            df2_ver = 'v2.4' if self.version in ['v2.4', 'v2.6'] else None
            df2 = self.get_from_h5(version=df2_ver)
            return df1, df2
        else:
            if self.version=='v2.6' and term == 'raw':
                version='v2.4'
            key = self.get_h5_key(term=term, version=version)
            version = self.version if not version else version
            from_h5 = self.h5_copy if version and version.startswith('v4') else self.h5
            df = pd.read_hdf(from_h5, key)
            logger.info('Dataset loaded from "{}" with key "{}"',from_h5, key)
            return df


    def save_to_h5(self, df, term='raw'):
        key = self.get_h5_key(term=term)
        to_h5 = self.h5_copy if self.version.startswith('v4') else self.h5
        df.to_hdf(to_h5, key)
        # with pd.HDFStore(to_h5, 'a') as hstore:
        #     hstore.put(key, df, format='table')
        logger.info('Datasets have been saved to {} with key "{}"', to_h5, key)

    def preprocess_raw_dataset(self, reuse_data=False, json_f='raw_data/PC_data_0120.json', sample_smaller: int = None,
                               saved_sampled_train=None, train_sample_size=1000, update_checked_data=True):
        if reuse_data:
            train_df1, eval_df1, train_df2, eval_df2, train_df3, eval_df3 = train_test0824()
            df1 = pd.concat([train_df1, eval_df1])
            df2 = pd.concat([train_df2, eval_df2])
            df3 = pd.concat([train_df3, eval_df3])
            raw_df = self.join_sub_datasets(df1, df2, df3)
            # amount too small
            target_text = raw_df['labels1']+'||'+raw_df['labels2']+'||'+raw_df['labels3'].map(
                lambda s:','.join(str(ss) for ss in s))
            res_df = pd.DataFrame({
                'prefix': 'claim',
                'input_text': raw_df['text'],
                'target_text': target_text
            })
        else:
            if os.path.exists(json_f):
                with open(json_f, 'r') as f:
                    raw = json.load(f)
            else:
                raw = get_page('drug_patent_claim', page_size=-1,
                               show_fields=['publication_docdb_comb', 'claim_num', 'claim_text',
                                            'claim_class_act','claim_type_act', 'claim_ref_act'])
                with open(json_f, 'w') as f:
                    json.dump(raw, f, indent=4)
            raw_df = pd.DataFrame.from_records(raw).dropna()
            raw_df.drop_duplicates(subset=raw_df.columns.drop('esid'), inplace=True)

            # update human-checked data
            if update_checked_data:
                checked_xlsx = "raw_data/patent_claims 待校验小量类型 by FZQ--返稿--20210125.xlsx"
                checked_raw_df = pd.read_excel(checked_xlsx)
                checked_raw_df['claim_type'] = checked_raw_df['人工校对claim_type_act'].where(
                    ~checked_raw_df['人工校对claim_type_act'].isna(), checked_raw_df['claim_type_act'])
                raw_df = raw_df.merge(checked_raw_df[['esid','claim_type']], 'left', 'esid')
                raw_df['claim_type_act'] = raw_df['claim_type_act'].where(raw_df['claim_type'].isna(), raw_df['claim_type'])
                raw_df.drop(columns='claim_type', inplace=True)
                logger.info('Updated data from "{}"', checked_xlsx)

                checked_xlsx2 = "raw_data/patent_claims 校验数据 20210122 by FZQ-任务分配（claims三分类）-汇总v3.xlsx"
                checked_raw_df2 = pd.read_excel(checked_xlsx2)
                checked_raw_df2['claim_ref'] = checked_raw_df2['校对claim_ref_act'].where(
                    ~checked_raw_df2['校对claim_ref_act'].isna(), checked_raw_df2['claim_ref_act']
                ).astype(str)
                checked_raw_df2.rename(columns={'校对claim_class_act': 'claim_class', '校对claim_type_act': 'claim_type'},
                                       inplace=True)
                raw_df = raw_df.merge(checked_raw_df2[['esid', 'claim_class', 'claim_type', 'claim_ref']],
                                      'left', 'esid')
                raw_df['claim_class_act'] = raw_df['claim_class_act'].where(raw_df['claim_class'].isna(), raw_df['claim_class'])
                raw_df['claim_type_act'] = raw_df['claim_type_act'].where(raw_df['claim_type'].isna(), raw_df['claim_type'])
                raw_df['claim_ref_act'] = raw_df['claim_ref_act'].where(raw_df['claim_ref'].isna(), raw_df['claim_ref'])
                raw_df.drop(columns=['claim_class', 'claim_type', 'claim_ref'], inplace=True)
                logger.info('Updated data from "{}', checked_xlsx2)

            # sample smaller groups
            if sample_smaller is not None:
                sampled_patent = raw_df['publication_docdb_comb'].drop_duplicates().sample(
                    sample_smaller, random_state=self.random_state)
                raw_df = raw_df[raw_df['publication_docdb_comb'].isin(sampled_patent)]

            # remove problem data of duplicates
            is_uniques = raw_df.groupby('publication_docdb_comb')['claim_num'].apply(
                lambda s:s.value_counts().eq(1))
            raw_df = raw_df[~raw_df['publication_docdb_comb'].isin(is_uniques[~is_uniques].index)]

            # remove problem data of: 1) class=D and ref is empty and claim_num<100
            # 2) class=I and text not contains 'canceled' and type='#'
            wrongs = (raw_df['claim_class_act'] == 'D') & (raw_df['claim_ref_act'].isin(['','nan'])) & (
                raw_df['claim_num']<100)
            wrongs = wrongs | ((raw_df['claim_class_act']=='I') & (raw_df['claim_type_act'].eq('#')) & (
                ~raw_df['claim_text'].str.contains('canceled')))

            # remove problem data of claim num malposition in text, fix no nums in text (include claim_num in claim_text)
            nums_in_text = raw_df['claim_text'].str.extract(r'(\d*)\D.*')[0]
            no_nums_in_text = nums_in_text[(nums_in_text.eq(''))|(nums_in_text.isna())]
            nums_in_text = nums_in_text[~nums_in_text.index.isin(no_nums_in_text.index)].astype(int)
            raw_df['claim_text'] = raw_df['claim_text'].where(
                ~raw_df.index.isin(no_nums_in_text.index),raw_df['claim_num'].astype(str) + '. ' + raw_df['claim_text'])
            wrongs = wrongs | (raw_df['claim_num'].ne(nums_in_text) & (~raw_df.index.isin(no_nums_in_text.index)))
            raw_df = raw_df[~wrongs]

            mapper = dict(zip(['I','D'], self.is_claim_mapper.keys()))
            if self.version=='v2.2':
                # multiple task in one prefix: 'claim'
                target_text = (raw_df['claim_class_act'].map(mapper) + '||' + raw_df['claim_type_act'] + '||' +
                               raw_df['claim_ref_act'].str.replace('.',','))
                res_df = pd.DataFrame({
                    'prefix': 'claim',
                    'input_text': raw_df['claim_text'],
                    'target_text':target_text
                })
            else:
                # multiple prefix: claim_class, claim_type, claim_ref
                to_melt = raw_df.rename(columns={'claim_class_act': 'claim_class', 'claim_type_act': 'claim_type',
                                                'claim_ref_act': 'claim_ref'})
                to_melt['claim_class'] = to_melt['claim_class'].map(mapper)
                to_melt['claim_ref'] = to_melt['claim_ref'].str.replace('.', ',').map(self.multiple_ref_values_to_ranges)
                res_df = to_melt.melt(id_vars=['claim_text', 'publication_docdb_comb', 'claim_num', 'esid'],
                                      value_vars=['claim_ref', 'claim_class', 'claim_type'], var_name='prefix',
                                      value_name='target_text').rename(columns={'claim_text': 'input_text'})

                # filter datasets for task2 and task3
                drop_condition = (res_df['prefix']=='claim_type') & (~res_df['target_text'].str.contains('canceled')) & (
                    res_df['target_text']=='#')
                drop_condition |= (res_df['prefix']=='claim_type') & (res_df['target_text']=='')
                drop_condition |= (res_df['prefix']=='claim_ref') & (res_df['target_text']=='*')
                res_df = res_df[~drop_condition]
                self.save_to_h5(res_df)

        if saved_sampled_train is not None:
            sampled_train_patent = raw_df['publication_docdb_comb'].drop_duplicates().sample(
                n=train_sample_size, random_state=self.random_state)
            sampled_train_df = raw_df[raw_df['publication_docdb_comb'].isin(sampled_train_patent)]
            sampled_train_df=sampled_train_df.rename(columns={'publication_docdb_comb':'patent_code'}).sort_values(
                by=['patent_code','claim_num'])
            logger.info('Size of sampled train dataset to check: {}', sampled_train_df.shape)
            sampled_train_df.to_excel(saved_sampled_train, index=False)

    def preprocess_train_eval_dataset(self, sample_class=None, sample_type=None, sample_ref=None, sample_person=None,
                                      eval_size=10000, sample_keep_minor_type=True):
        """
        :param sample_class: Sentence level sample number. v2.4 and v2.6: 300000, v4.0: total.
        :param sample_type: Sentence level sample number. v2.4 and v2.6: 300000, v4.0: total.
        :param sample_ref: Sentence level sample number. v2.4 and v2.6: 300000, v4.0: total.
        :param sample_person: Sentence level sample number. v2.4: 30000, v2.6: 300000, v4.0: total.
        :param eval_size: eval size of task 1-3.
        :param sample_keep_minor_type: Used when sample type.
        :return:
        """
        person_ver = ['v2.4', 'v2.6', 'v4.0']
        if self.version in person_ver:
            raw_df, raw_person = self.get_from_h5(return_multiple_parts=True)
        else:
            raw_df = self.get_from_h5()
        # split train and eval dataset
        train_df, eval_df = train_test_split(raw_df, test_size=eval_size, random_state=self.random_state,
                                             stratify=raw_df['prefix'])
        if self.version in person_ver:
            train_person, eval_person = train_test_split(raw_person, test_size=0.1, random_state=self.random_state,
                                                         stratify=raw_person['target_text'])
            if self.version in ['v2.6', 'v4.0']:
                upsample_size_person = train_person['target_text'].value_counts().max()
                train_person = pd.concat(
                    resample(d_label, n_samples=upsample_size_person, random_state=self.random_state)
                    if d_label.shape[0]<upsample_size_person else d_label
                    for label_, d_label in train_person.groupby('target_text')
                )
                logger.debug('Person training dataset label-balanced.')

        # upsample and downsample training dataset
        class_df = train_df[train_df['prefix']=='claim_class']
        if sample_class:
            class_df = resample(class_df, n_samples=sample_class, random_state=self.random_state,
                                stratify=class_df['target_text'])

        if sample_type:
            if sample_keep_minor_type:
                minor_types=['其他', '诊断试剂', '包材', '前药', '医药中间体', '溶剂化物', '酯', '制药设备', '衍生物', '盐']
                to_upsample = train_df[(train_df['prefix']=='claim_type') & train_df['target_text'].isin(minor_types)]
                upsample_size = to_upsample['target_text'].value_counts().max()
                upsampled = pd.concat(resample(d, n_samples=upsample_size, random_state=self.random_state)
                                      for _, d in to_upsample.groupby('target_text'))
                to_downsample = train_df[(train_df['prefix'] == 'claim_type') & (~train_df['target_text'].isin(minor_types))]
                downsample_size = sample_type - upsampled.shape[0]
                if downsample_size>0:
                    downsampled = resample(to_downsample, n_samples=downsample_size, random_state=self.random_state)
                else:
                    downsampled = to_downsample
                type_df = pd.concat([upsampled, downsampled])
            else:
                type_df = resample(train_df[train_df['prefix']=='claim_type'], n_samples=sample_type,
                                   random_state=self.random_state)
        else:
            type_df = train_df[train_df['prefix']=='claim_type']

        ref_df = train_df[train_df['prefix']=='claim_ref']
        if sample_ref:
            ref_df = resample(ref_df, n_samples=sample_ref, random_state=self.random_state)

        # concat train and eval dataset
        if self.version in person_ver:
            if sample_person:
                person_df = resample(train_person, n_samples=sample_person, random_state=self.random_state)
            else:
                person_df = train_person
            train_df = pd.concat([class_df, type_df, ref_df, person_df]).sort_index().sample(
                frac=1, random_state=self.random_state)
            eval_df = pd.concat([eval_df, eval_person]).sort_index()
        else:
            train_df = pd.concat([class_df, type_df, ref_df]).sort_index().sample(
                frac=1, random_state=self.random_state)

        self.save_to_h5(train_df, 'train')
        self.save_to_h5(eval_df, 'eval')

    def preprocess_person(self):
        """Preprocess classify dataset for task 4."""
        if self.version=='v2.4':
            excel_file = 'raw_data/20210223-专利申请人类型标注（法人&自然人）.xlsx'
            df = pd.read_excel(excel_file, names=['input_text', 'target_text'], dtype=str)
        elif self.version=='v4.0':
            excel_file = "raw_data/202105-申请人类型细分.xlsx"
            df = pd.read_excel(excel_file, names=['input_text', 'applicant_type', 'target_text'], dtype=str)
            df['target_text'] = df['target_text'].where(~df['target_text'].str.contains('政府机构'), '政府机构')
        df.insert(0, 'prefix', 'person')
        self.save_to_h5(df.dropna())

    def reshape_parents_claim_datasets(self, df):
        filtered = df[df['labels']==1][['text_a','text_b']]
        res = filtered.groupby('text_a')['text_b'].apply(
            lambda s:s.astype('int').sort_values().tolist()
        ).reset_index()
        return res

    def join_sub_datasets(self, df1, df2, df3):
        res1 = df1.drop_duplicates().astype({1:int})
        res1.columns=['text', 'labels1']
        mapper1=dict(zip(self.is_claim_mapper.values(), self.is_claim_mapper.keys()))
        res1['labels1'] = res1['labels1'].map(mapper1)
        res2 = df2.drop_duplicates().astype({1:int})
        res2.columns=['text', 'labels2']
        mapper2=dict(zip(self.claim_type_mapper.values(),self.claim_type_mapper.keys()))
        res2['labels2'] = res2['labels2'].map(mapper2)
        res3 = self.reshape_parents_claim_datasets(df3.drop_duplicates())
        res3.columns=['text', 'labels3']
        res = res1.merge(res2, on='text').merge(res3, on='text')
        return res

    def refine_result(self, to_refine: pd.Series):
        pattern = r'(.*)\|\|(.*)\|\|(.*)'
        res = to_refine.str.extract(pattern)
        res.columns=['claim_class', 'claim_type', 'claim_ref']
        return res

    def multiple_ref_values_to_ranges(self, input_str: str):
        """input example: 1,2,4,6,7,8,9,10
        output example: 1-2, 4, 6, 7-10"""
        from numpy import nan
        value_strs = input_str.split(',')
        if all(x.isdigit() for x in value_strs):
            values=sorted(int(x) for x in value_strs)
            lasts = [nan] + values[:-1]
            nexts = values[1:]+[nan]
            result_values = []
            for val, last_val, next_val in zip(values, lasts, nexts):
                if val != last_val + 1 and val+1== next_val:
                    from_=val
                elif last_val+1==val and val+1 != next_val:
                    result_values.append(f'{from_}-{val}')
                elif last_val+1 != val and val+1 != next_val:
                    result_values.append(str(val))
            result_str = ','.join(result_values)
        else:
            result_str = input_str
        return result_str

    def ref_ranges_to_values(self, input_str: str, return_str = True):
        """input example: 1-2, 4, 6, 7-10,
        output_example: 1,2,4,6,7,8,9,10"""
        ranges = input_str.split(',')
        values=[]
        for range_ in ranges:
            cur_values = range_.split('-')
            if len(cur_values) == 2:
                values.extend(list(range(int(cur_values[0]), int(cur_values[1])+1)))
            elif len(cur_values) ==1 and cur_values[0]:
                values.append(int(cur_values[0]))
        if return_str:
            res_str = ','.join(str(x) for x in sorted(values))
            return res_str
        else:
            return values

    def describe_dataset(self, return_counts=False):
        counts = []
        for item in ['train', 'eval']:
            df: pd.DataFrame = self.get_from_h5(item)
            print(f'df.size for {item} dataset:', df.shape[0])
            import uuid
            df['esid'] = df['esid'].fillna(uuid.uuid1())
            print(f'Total sentences for {item} dataset:', df['input_text'].nunique())
            print(f'Total number of claims for {item} dataset:', df['publication_docdb_comb'].nunique())
            print('-'*50)

            count_sent = df.drop_duplicates().groupby('prefix')['esid'].count()
            print(f'Sentence counts for {item} dataset: \n', count_sent)
            print('-'*50)

            claim_num = df.groupby('prefix')['publication_docdb_comb'].nunique()
            print(f'Claims counts for {item} dataset: \n', claim_num)
            print('-'*50)

            class_type_df = df[df['prefix'].isin(['claim_class', 'claim_type'])].drop_duplicates()
            class_type_count = class_type_df.groupby('target_text')['esid'].count()
            if return_counts:
                counts.append(class_type_count)
            print(f'Sentence counts for {item} dataset: \n', class_type_count)
            print('-'*50)

            if self.version in ['v2.4', 'v2.6', 'v4.0']:
                df_person = df[df['prefix']=='person'].drop_duplicates()
                count_person = df_person['target_text'].value_counts()
                print(f'Person type counts for {item} dataset:\n', count_person)
                print('-'*50)
        if return_counts:
            return counts

class ClassificationProcessor(Preprocessor):
    def __init__(self, version='v2.5'):
        """
        :param version:
            - v2.5: classification of corporation person
        """
        self.version = version
        self.h5 = 'data.h5'
        self.random_state = 324

    def preprocess_raw(self):
        excel_file = 'raw_data/20210223-专利申请人类型标注（法人&自然人）.xlsx'
        df = pd.read_excel(excel_file, names=['text', 'labels'])
        self.save_to_h5(df)

    def preprocess_train_eval_dataset(self, eval_size=0.1):
        df_raw = self.get_from_h5()
        train_raw, eval_df = train_test_split(df_raw, test_size=eval_size, random_state=self.random_state,
                                              stratify=df_raw['labels'])
        # upsampling
        upsample_size = train_raw['labels'].value_counts().max()
        train_df = pd.concat(
            resample(d_label, n_samples=upsample_size, random_state=self.random_state)
            if d_label.shape[0]<upsample_size else d_label
            for label_, d_label in train_raw.groupby('labels')
        )
        self.save_to_h5(train_df, 'train')
        self.save_to_h5(eval_df, 'eval')

    def describe_dataset(self):
        for item in ['train', 'eval']:
            df = self.get_from_h5(item)
            if item=='train':
                print(f'Total size for {item} dataset (upsampled):', df.shape[0])
                print('-' * 50)
                counts = df['labels'].value_counts()
                print(f'# each labels for {item} dataset (upsampled):\n', counts)
                print('-'*50)

            df = df.drop_duplicates()
            print(f'Total size for {item} dataset (duplicates_dropped):', df.shape[0])
            print('-'*50)

            counts = df['labels'].value_counts()
            print(f'# each labels for {item} dataset (duplicates_dropped):\n', counts)
            print('-'*50)

if __name__ == '__main__':
    x=GenerativeDataProcessor('v4.0')
    x.get_train_eval_datasets(to_excel_path='results/train_eval_dataset_v4.0.xlsx')