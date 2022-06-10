import pandas as pd
import random
from pharm_ai.util.ESUtils7 import Query, QueryType, get_page
from pharm_ai.util.sm_util import SMUtil

h5 = 'data.h5'
random_state=1204

def dt0921():
    xlsx_file = "raw_data/invest_news_nlp_deal2.xlsx"
    dt = pd.read_excel(xlsx_file)
    dt = dt.iloc[:,[3,2,1]].rename(columns={'labels_act':'labels'})
    dt.to_hdf(h5, 'dt0921/raw')

def get_es_labeled_data(number:int=20000):
    # pre是预测的，act是标注完的，segment是分段，num是顺序，main_id是对应资讯esid，type是字段名，token是具体的单词
    res_dict = get_page(index="invest_news_nlp", page_size=number)
    res_tb = pd.DataFrame.from_records(res_dict)
    # re-get data to keep full words of the same article exist
    main_ids = res_tb['main_id'].unique()
    res_dict_full = [get_page(index="invest_news_nlp", page_size=-1,
                               queries=Query(QueryType.EQ, "main_id", id))
                     for id in main_ids]
    res_dict_full = [item for res in res_dict_full for item in res]
    res_tb_full = pd.DataFrame.from_records(res_dict_full)
    return res_tb_full

def dt1204():
    df = get_es_labeled_data(number=80000)
    df['sentence_id'] = df['main_id'] + '_' + df['segment'].astype(str)
    df.rename(columns={'act': 'labels', 'token':'words'}, inplace=True)
    df.to_hdf(h5, 'dt1204/raw')

def train_test_v1_2():
    input_h5 = "raw_data/train_test_20200722.h5"
    train = pd.read_hdf(input_h5, 'train')
    test = pd.read_hdf(input_h5, 'test')
    dt = pd.read_hdf(h5, 'dt0921/raw')
    grs = [df for _, df in dt.groupby('sentence_id')]
    random.shuffle(grs)
    train_new = pd.concat(grs[:-150])
    test_new = pd.concat(grs[-150:])
    train = pd.concat([train, train_new])
    test = pd.concat([test, test_new])
    train.to_hdf(h5, 'dt0921/train')
    test.to_hdf(h5, 'dt0921/test')

def train_test():
    h5 = 'data.h5'
    train_previous, test_previous = pd.read_hdf(h5, 'dt0921/train'), pd.read_hdf(h5, 'dt0921/test')
    df_previous = pd.concat([train_previous, test_previous])
    df_raw = pd.read_hdf(h5, 'dt1204/raw')
    df = df_raw[~df_raw['main_id'].isin(df_previous['sentence_id'])][['sentence_id', 'words', 'labels']]
    grs = [d for _,d in df.groupby('sentence_id')]
    random.seed(random_state)
    random.shuffle(grs)
    train_new = pd.concat([train_previous, *grs[:-150]])
    train_df = SMUtil.ner_upsampling(train_new, random_state=random_state)
    train_df.to_hdf(h5, 'dt1204/train')
    test_previous.to_hdf(h5, 'dt1204/test')

if __name__=='__main__':
    # dt1204()
    train_test()