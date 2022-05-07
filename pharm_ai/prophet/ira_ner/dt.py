import pandas as pd
from pharm_ai.util.prepro import Prepro
import random
from pharm_ai.util.sm_util import SMUtil
from pharm_ai.util.ESUtils7 import Query, QueryType, get_page

random.seed(123)


def dt20201112():
    x = '../ira_filter/prophet-2-核心摘要段标注-20200927交付版.xlsx'
    df = pd.read_excel(x, '1').dropna()
    df = df.groupby('labels').get_group(1)
    texts = df['text'].tolist()
    p = Prepro()
    x = 'prophet-ira_filter-ner-unlabeled-20201112.xlsx'
    p.prep_ner_train_dt(texts, x, shuffle=True)


def prep_train_test_ner(df=None, dev_ratio=0.1, h5_path=None):
    grs = df.groupby('sentence_id')
    grs = [g for _, g in grs]
    pos = []
    neg = []
    for g in grs:
        labels = g['labels'].tolist()
        temp = [True for label in labels if label.startswith('b-')]
        if temp:
            pos.append(g)
        else:
            neg.append(g)
    random.shuffle(pos)
    sep = round(dev_ratio*len(pos))
    train_pos = pos[sep:]
    test_pos = pos[:sep]
    print(len(test_pos))

    train_neg = neg[sep:]
    test_neg = neg[:sep]
    print(len(test_neg))

    train = train_pos + train_neg
    test = test_pos + test_neg

    print(str(len(train))+' sentences in train set.')
    print(str(len(test)) + ' sentences in test set.')

    random.shuffle(train)
    random.shuffle(test)

    train = pd.concat(train).sample(frac=1, random_state=123)
    test = pd.concat(test).sample(frac=1, random_state=123)

    train.to_hdf(h5_path, 'train')
    print('train saved')
    test.to_hdf(h5_path, 'test')
    print('test saved')


def dt20201120_1():
    # x = '三要素接口新数据202011119.xlsx'
    # df = pd.read_excel(x, '1').dropna()
    h5 = 'ner-unlabeled-20201120.h5'
    # df.to_hdf(h5, '1')
    df = pd.read_hdf(h5)
    df['text'] = df['text'].astype(str)
    texts = df['text'].tolist()
    article_ids = df['article_ids'].tolist()
    return texts, article_ids

def dt20201201():
    xlsx = "raw_data/ira_ner-preds-1529条预跑模型 20201130-校验.xlsx"
    h5 = 'train_test-20201118.h5'
    df = pd.read_excel(xlsx)
    df = df.rename(columns={'标注':'labels'})[['words','labels','sentence_id']]
    df['words'] = df['words'].astype(str)
    return df


def train_test():
    h5 = 'train_test-20201118.h5'
    h5_v2 = 'train_test_20201201.h5'
    train_v1 = pd.read_hdf(h5, 'train')
    train_v2 = dt20201201()
    train_v2_upsampling = SMUtil.ner_upsampling(train_v2)
    train = pd.concat([train_v1, train_v2_upsampling])
    train.to_hdf(h5_v2, 'train')
    test = pd.read_hdf(h5, 'test')
    test.to_hdf(h5_v2, 'test')

def get_sentences_1203():
    h5 = 'train_test_20201201.h5'
    train_ner_df, test_ner_df = pd.read_hdf(h5, 'train'), pd.read_hdf(h5, 'test')
    train_sents = train_ner_df['sentence_id'].drop_duplicates()
    res = []
    for sent in train_sents:
        q = Query(QueryType.EQ, "esid", sent)
        r = get_page("invest_news_nlp", queries=q, page_size=-1)
        res.append(r)
    return res


if __name__ == '__main__':
    get_sentences_1203()