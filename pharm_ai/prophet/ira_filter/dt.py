import langid
import pandas as pd
from sklearn.utils import resample

from mart.utils import to_excel, conv_date2en


def dt1110():
    x = 'prophet-2-核心摘要段标注-20200927交付版.xlsx'
    df = pd.read_excel(x, '1').dropna().sample(frac=1,
                                               random_state=123)
    grs = df.groupby('labels')
    g1 = grs.get_group(1)
    g0 = grs.get_group(0)

    train1 = g1.iloc[100:]
    test1 = g1.iloc[:100]
    train0 = g0.iloc[100:]
    test0 = g0.iloc[:100]

    train1 = resample(train1, replace=True,
                      n_samples=len(train0),
                      random_state=123)
    train = pd.concat([train1, train0]).sample(frac=1, random_state=123)
    print(len(train))
    test = pd.concat([test1, test0]).sample(frac=1, random_state=123)
    print(len(test))

    h5 = 'train_test-20201110.h5'
    train.to_hdf(h5, 'train')
    test.to_hdf(h5, 'test')


def dt20201119():
    x = '核心摘要段标注-20201119 加时间加编数据(1).xlsx'
    df = pd.read_excel(x, '1').dropna()
    pub_dates = df['publish_date'].astype(str).tolist()
    print(pub_dates[0])
    paras = df['paragraph'].astype(str).tolist()
    labels = df['labels'].tolist()
    text = []
    for d, p in zip(pub_dates, paras):
        txt = d + '. ' + p
        text.append(txt)
    df = pd.DataFrame({'text': text,
                       'labels': labels}).\
        sample(frac=1, random_state=123)
    train = df.iloc[200:]
    print('# of train samples: '+str(len(train)))
    test = df.iloc[:200]
    print('# of test samples: ' + str(len(test)))

    # h5 = 'train_test-20201119.h5'
    # train.to_hdf(h5, 'train')
    # test.to_hdf(h5, 'test')


def dt20201120():
    x = '三要素接口补充负样本20201120.xlsx'
    df = pd.read_excel(x, '1').dropna().\
        sample(frac=1, random_state=123)
    df = df.iloc[:3000]
    x = 'prophet-ira_filter-negs-3000-20201120.xlsx'
    to_excel(df, x, 'unchecked')


def train_test():
    h5 = 'train_test-20201123.h5'
    train = pd.read_hdf(h5, 'train')
    test = pd.read_hdf(h5, 'test')
    return train, test


def dt20201123():
    x = 'prophet-ira-negs-3000-20201120负样本校验+时间-校验.xlsx'
    df = pd.read_excel(x, '1').dropna()
    pub_dates = df['publish_date'].tolist()
    print(pub_dates[0])
    paras = df['paragraph'].tolist()
    labels = df['labels'].tolist()
    c = 0
    paras_with_dates = []
    for date, para in zip(pub_dates, paras):
        m = date.month
        d = date.day
        y = date.year
        para = str(para)
        lang = langid.classify(para)[0]
        para = para.strip()
        if lang == 'zh':
            if not para.endswith('。'):
                para = para + '。'
            para = f'本文发布日期为{y}年{m}月{d}日。{para}'
        else:
            if not para.endswith('.'):
                para = para + '.'
            para = f'This article was published on {conv_date2en(m, d, y)}. {para}'
        paras_with_dates.append(para)
    df = pd.DataFrame({'text': paras_with_dates, 'labels': labels}).sample(frac=1, random_state=123)
    train = df.iloc[300:]
    test = df.iloc[:300]
    grs_train = train.groupby('labels')
    print(len(grs_train.get_group(1)))
    print(len(grs_train.get_group(0)))
    grs_test = test.groupby('labels')
    print(len(grs_test.get_group(1)))
    print(len(grs_test.get_group(0)))
    grs = train.groupby('labels')
    train1 = grs.get_group(1)
    train0 = grs.get_group(0).sample(frac=1, random_state=123)
    train1 = resample(train1, replace=True, n_samples=len(train0), random_state=123)
    train = pd.concat([train1, train0]).sample(frac=1, random_state=123)
    h5 = 'train_test-20201123.h5'
    # train.to_hdf(h5, 'train')
    # test.to_hdf(h5, 'test')


if __name__ == '__main__':
    dt20201123()