import pandas as pd
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split

h5 = "data.h5"
random_state=2333
test_size=400
test_size_model2=200
mapping = {'医药-金融':0,'医药-非金融':2,'非医药-金融':1,'非医药-非金融':2}
mapping_2 = {'非相关': 2, '医药': 0, '非医药': 1, 0.5:3}

def train_test_v4():
    # read previous datasets
    train_df1: pd.DataFrame = pd.read_hdf(h5, 'dt0928/train')
    train_df1.rename(columns={'esid':'ESID'},inplace=True)
    train_df1=train_df1.drop_duplicates().sort_index()
    eval_df: pd.DataFrame=pd.read_hdf(h5, 'dt0928/eval')
    eval_df.rename(columns={'esid':'ESID'},inplace=True)
    eval_df.sort_index(inplace=True)
    eval_df.drop_duplicates(ignore_index=True,inplace=True)
    eval_df.to_hdf(h5, 'dt1015/eval')

    # append new training dataset
    xlsx = "raw_data/投资资讯按段落标注-20201013补充数据交付版.xlsx"
    dt = pd.read_excel(xlsx)
    dt.rename(columns={'标题':'title','段落':'paragraph','标注':'labels'},inplace=True)
    dt.labels = dt['labels'].map({1:0, 2:2, 3:1})
    dt['text']=dt['title']+dt['paragraph'].astype('str')
    dt = pd.concat([dt[['ESID','text','labels']],train_df1]).reset_index(drop=True)
    train_grs = dt.groupby('labels')
    resample_size = train_grs.apply(len).max()
    train_df = pd.concat([resample(gr, n_samples=resample_size, random_state=random_state) for g_id, gr in train_grs])
    train_df = train_df.sample(frac=1, random_state=random_state)
    train_df.to_hdf(h5, 'dt1015/train')

    # use previous test dataset
    test_df:pd.DataFrame = pd.read_hdf(h5, 'dt0928/test')
    test_df.to_hdf(h5, 'dt1015/test')

def preprocess(model=0):
    """
    Preprocess datasets.
    :param model: This project consists of two models, model=1 or 2. If model=0, preprocessing for both model.
    :return: Results saved to 'data.h5'
    """
    # preprocess data for model 1
    if model==0 or model==1:
        xlsx = "raw_data/投融资标题分类 - 交付版20201110.xls"
        df_raw = pd.read_excel(xlsx)
        df_raw.rename(columns={'标题':'text', 'label':'labels'}, inplace=True)
        df_raw['labels'] = df_raw['labels'].map(mapping_2)
        df_raw.to_hdf(h5, 'v6-1/raw')

    # preprocess data for model 2
    if model==0 or model==2:
        xlsx = "raw_data/逐段标注医药非医药 20201123-交付.xlsx"
        df_raw = pd.read_excel(xlsx)
        df_raw['text'] = df_raw['title'].astype('str')+' '+df_raw['content'].astype('str')
        df_raw.rename(columns={'label':'labels'},inplace=True)
        df_raw['labels'] = df_raw['labels'].map({1:0,2:1})
        df_raw.to_hdf(h5, 'v6-2/raw')

    if model==0 or model==3:
        xlsx = "raw_data/逐段标注医药非医药 20201123-交付.xlsx"
        df_raw = pd.read_excel(xlsx)
        df_res = (df_raw.groupby('ESID')[['title', 'content', 'label']]
            .aggregate({
            'title': lambda x: x.unique()[0],
            'content': lambda s: ' '.join(s.astype(str)),
            'label': lambda s: s.min()
        }))
        df_res['text']=df_res['title']+' '+df_res['content']
        df_res = df_res.reset_index()
        df_res.to_hdf(h5, 'v6-3/raw')

def train_test(model=0):
    if model==0 or model==1:
        df_raw = pd.read_hdf(h5, 'v6-1/raw')
        df_train, df_test = train_test_split(df_raw, test_size=test_size, random_state=random_state)
        df_train.to_hdf(h5, 'v6-1/train')
        df_test.to_hdf(h5, 'v6-1/test')
    if model==0 or model==2:
        df_raw = pd.read_hdf(h5, 'v6-2/raw')
        train_grs = df_raw.groupby('ESID')
        train_list = shuffle([gr for gr_id, gr in train_grs], random_state=random_state)
        df_train, df_test = pd.concat(train_list[:-test_size_model2]), pd.concat(train_list[-test_size_model2:])
        resample_size = df_train.groupby('labels')['text'].count().max()
        df_train = pd.concat([resample(gr, n_samples=resample_size, random_state=random_state)
                              for gr_id, gr in df_train.groupby('labels')])
        df_train = df_train.sample(frac=1, random_state=random_state)
        df_train.to_hdf(h5, 'v6-2/train')
        df_test.to_hdf(h5, 'v6-2/test')
    if model==0 or model==3:
        df_raw = pd.read_hdf(h5, 'v6-3/raw')
        df_train, df_test = train_test_split(df_raw, test_size=test_size_model2, random_state=random_state)
        df_train.to_hdf(h5, 'v6-3/train')
        df_test.to_hdf(h5, 'v6-3/test')

if __name__=="__main__":
    preprocess(model=3)
    train_test(model=3)