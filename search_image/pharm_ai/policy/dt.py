import os

import numpy as np
import pandas as pd
from sklearn.utils import resample, shuffle

da = pd.read_excel("/home/ryz/work/from_zj/政策库v1.1分类数据_summary-删了一部分.xlsx", sheet_name="Sheet1")
da = da.replace(["\s", "\n", "_x000D_", "<br>", "&nbsp"], "", regex=True)
'''preprocess data to get the full train data and eval data'''


def policy_classfication_rawdata():
    # set all other minor class to 其他
    others = pd.read_csv("/home/ryz/work/from_zj/other.label.txt")
    da["new_label"] = da["政策分类"].apply(lambda x: "其他" if x in others["政策分类"].tolist() else x)

    # convert chinese labels to corresponding number labels
    label_dict = {}
    for i, j in enumerate(da["new_label"].value_counts().index):
        label_dict[j] = i
    da["labels"] = da["new_label"].apply(lambda x: label_dict[x])

    ## As for sample balance，set 100 as the threshold for downsampling label class which have samples great than threshold and upsampling according label class which have samples less than threshold .
    # get the evaluation dataset,
    '''
    get 1 sample for labels number <=20, 
    2 sample for labels number >20 & <= 50;
    4 sample for labels number >50 & <= 100;
    5% sample for labels number >100
    '''
    resamples = {}
    labelNum = da["new_label"].value_counts()
    for i in labelNum.index:
        if labelNum[i] <= 20:
            resamples[i] = 1
        elif 20 < labelNum[i] <= 50:
            resamples[i] = 2
        elif 50 < labelNum[i] <= 100:
            resamples[i] = 4
        else:
            resamples[i] = 0.05

    eval_da = pd.DataFrame()
    for i in resamples:  # key is class label
        if resamples[i] < 1:
            tmp = da[da["new_label"] == i].sample(frac=resamples[i], random_state=23)
        else:
            tmp = da[da["new_label"] == i].sample(resamples[i], random_state=23)
        eval_da = pd.concat([eval_da, tmp])
    eval_da.to_hdf("/home/ryz/work/from_zj/data/multi_eval_data.h5", "eval", mode='w')
    train_da = da[~da.index.isin(eval_da.index)]
    return train_da, eval_da


def save_to_hdf(dirname, prefix, train=None, eval=None):
    if not train is None:
        train.to_hdf(os.path.join(dirname, prefix + "_train.h5"), "train", mode='w')
    if not eval is None:
        eval.to_hdf(os.path.join(dirname, prefix + "_eval.h5"), "eval", mode='w')


'''format the train and eval data to meet the requirement of multi_classfication model'''


def policy_multi_classfication(train_da, eval_da):
    #### balance the train dataset
    label_grs = train_da.groupby('labels')
    ## TODO: for major class, should replace = False
    train_df = label_grs.apply(resample, n_samples=150, random_state=2333).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=2333)

    train_df1 = train_df[["标题", "labels"]]
    train_df1.columns = ["text", "labels"]
    eval_df1 = eval_da[["标题", "labels"]]
    eval_df1.columns = ["text", "labels"]
    save_to_hdf("/home/ryz/work/from_zj/data", "multi_clas", train=train_df1, eval=eval_df1)
    print("/home/ryz/work/from_zj/data/multi_clas* processed!")


'''format the train and eval data to meet the requirement of sentence pair classification model'''
def policy_senpair_classfication(train_da, eval_da):
    train_new = train_da[["标题", "政策分类"]].copy()
    train_new["labels"] = 1
    train_new.columns = ["text_a", "text_b", "labels"]
    total_class = train_new["text_b"].unique().shape[0]
    eval_new = eval_da[["标题", "政策分类"]].copy()
    eval_new["labels"] = 1
    eval_new.columns = ["text_a", "text_b", "labels"]

    def get_negative_sample(x, num):
        # np.random.seed(2)
        target_label = np.argwhere(train_new["text_b"].unique() == x["text_b"])[0][0]
        other_label = np.delete(train_new["text_b"].unique(), target_label)
        neg = np.random.choice(other_label, num, replace=False)
        dict1 = {"text_a": x["text_a"], "text_b": neg}
        return pd.DataFrame(dict1, index=range(len(neg)))

    ######## make negative sample in train data  ##########
    negs = []
    for i in range(0, train_new.shape[0]):
        negs.append(get_negative_sample(train_new.iloc[i,], 10))
    negs_da = pd.concat(negs, axis=0).reset_index(drop=True)
    negs_da["labels"] = 0
    train_data = pd.concat([train_new, negs_da]).reset_index(drop=True)
    train_data = shuffle(train_data)
    ######## make negative sample in evaluation data  #################
    negs = []
    for i in range(0, eval_new.shape[0]):
        negs.append(get_negative_sample(eval_new.iloc[i,], 10))
    negs_da = pd.concat(negs, axis=0).reset_index(drop=True)
    negs_da["labels"] = 0
    eval_data = pd.concat([eval_new, negs_da]).reset_index(drop=True)
    eval_data = shuffle(eval_data)
    save_to_hdf("/home/ryz/work/from_zj/data", "multi_clas_sp", train=train_data, eval=eval_data)
    train_data.to_csv("/home/ryz/work/from_zj/data/multi_clas_sp_train.csv", sep="\t", index=False)
    eval_data.to_csv("/home/ryz/work/from_zj/data/multi_clas_sp_eval.csv", sep="\t", index=False)
    print("/home/ryz/work/from_zj/data/multi_clas_sp* processed!")


def getda(df):
    return df.sample(frac=0.05, random_state=2333)


''' add new negative samples.
preprocess data to get the full train data and eval data for binary classfication'''


def policy_binary_classfication():
    newda = pd.read_excel("/home/ryz/work/from_zj/新分类_负样本_esid.xlsx", sheet_name="Sheet2")
    da_0 = newda[newda["分类"] == 0]
    da_1 = newda[newda["分类"] != 0]

    da_tmp = da[["标题"]]
    da_1_tmp = da_1[["标题"]]
    pos_da = pd.concat([da_tmp, da_1_tmp], axis=0)
    pos_da["labels"] = 1
    pos_da.columns = ["text", "labels"]

    neg_da = da_0[["标题", "分类"]].copy()
    neg_da.columns = ["text", "labels"]

    total_da = pd.concat([pos_da, neg_da], ignore_index=True)
    total_da = shuffle(total_da)

    eval_da = total_da.groupby("labels").apply(getda).droplevel(0)
    train_da = total_da[~total_da.index.isin(eval_da.index)]
    train_da = shuffle(train_da)

    save_to_hdf("/home/ryz/work/from_zj/data", "binary_clas", train=train_da, eval=eval_da)
    print("/home/ryz/work/from_zj/data/binary_clas* processed!")


# format data for sentence transformers BatchSemiHardTripletLoss function
def get_SentenceTrans_data(da):
    # format
    # da = pd.read_excel("/home/ryz/work/from_zj/政策库v1.1分类数据_summary-删了一部分.xlsx", sheet_name="Sheet1")
    # da = da.replace(["\s", "\n", "_x000D_", "<br>", "&nbsp"], "", regex=True)
    classes = da["政策分类"].value_counts()
    print(classes.shape)
    classes = classes[classes > 5]
    print(classes.shape)
    da_filter = da[da["政策分类"].isin(classes.index)].copy()
    label_dict = {}
    for i, j in enumerate(classes.index):
        label_dict[j] = i
    da_filter["labels"] = da_filter["政策分类"].apply(lambda x: label_dict[x])

    # set resample ratio
    resamples = {}
    labelNum = da_filter["labels"].value_counts()
    for i in labelNum.index:
        if labelNum[i] <= 20:
            resamples[i] = 2
        elif 20 < labelNum[i] <= 100:
            resamples[i] = 0.08
        else:
            resamples[i] = 0.05
    # do sampling
    eval_da = pd.DataFrame()
    for i in resamples:  # key is class label
        if resamples[i] < 1:
            tmp = da_filter[da_filter["labels"] == i].sample(frac=resamples[i], random_state=23)
        else:
            tmp = da_filter[da_filter["labels"] == i].sample(resamples[i], random_state=23)
        eval_da = pd.concat([eval_da, tmp])

    train_da = da_filter[~da_filter.index.isin(eval_da.index)]
    save_to_hdf("/home/ryz/work/from_zj/data", "st_semihard", train=train_da, eval=eval_da)
    print("/home/ryz/work/from_zj/data/st_semihard* processed!")
    # return train_da, eval_da


if __name__ == "__main__":
    # train_da, eval_da = policy_classfication_rawdata()
    # policy_multi_classfication(train_da, eval_da)
    # policy_senpair_classfication(train_da, eval_da)  # eval data is same with eval_da
    # policy_binary_classfication()
    get_SentenceTrans_data(da)
