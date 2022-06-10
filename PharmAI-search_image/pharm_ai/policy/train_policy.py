import os

import pandas as pd
from simpletransformers.classification import ClassificationArgs
from simpletransformers.classification import DDPClassificationModel
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.utils import resample


############### resample
def resample_df(data, m):
    # resample all class with 150
    label_grs = data.groupby('labels')
    # resample_size = label_grs.count()['text'].max()
    train_df = label_grs.apply(resample, n_samples=m, random_state=2333).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=2333)
    # train_df1 = train_df[["标题","labels"]]
    # train_df1.columns = ["text", "labels"]
    return train_df


############# init model
def multiclassfication_model_config():
    output_dir = "/home/ryz/work/from_zj/output/"
    cache_dir = "/home/ryz/work/from_zj/output_cache/"
    best_model_dir = "/home/ryz/work/from_zj/best_model/"
    version = "v1"
    model_args = ClassificationArgs(
        wandb_project="policy_classify", n_gpu=4,
        reprocess_input_data=True, use_cached_eval_features=True,
        overwrite_output_dir=True, fp16=True,
        num_train_epochs=9, learning_rate=1e-4,
        logging_steps=12, train_batch_size=300,
        eval_batch_size=100, save_eval_checkpoints=False,
        save_model_every_epoch=False, evaluate_during_training=True,
        evaluate_during_training_steps=10, evaluate_during_training_verbose=True,
        output_dir=output_dir, cache_dir=cache_dir,
        best_model_dir=best_model_dir, wandb_kwargs={'tags': [version]})
    return model_args


def multiclassfication_data():
    # the data has been resampled and was ready for training
    train_da = pd.read_hdf("/home/ryz/work/from_zj/data/multi_clas_train.h5", "train")
    eval_da = pd.read_hdf("/home/ryz/work/from_zj/data/multi_clas_eval.h5", "eval")
    return train_da, eval_da


def sp_data():
    # the data has been resampled and was ready for training
    train_da = pd.read_hdf("/home/ryz/work/from_zj/data/multi_clas_sp_train.h5", "train")
    eval_da = pd.read_hdf("/home/ryz/work/from_zj/data/multi_clas_sp_eval.h5", "eval")
    return train_da, eval_da


def sp_model_config():
    basedir = "/home/ryz/work/from_zj/multicalss_policy/"
    output_dir = basedir + "output/"
    cache_dir = basedir + "output_cache/"
    best_model_dir = basedir + "best_model/"
    version = "v1"
    model_args = ClassificationArgs(
        wandb_project="policy_classify", n_gpu=6,
        reprocess_input_data=True, use_cached_eval_features=True,
        overwrite_output_dir=True, fp16=True,
        num_train_epochs=3, learning_rate=5e-5,
        logging_steps=20, train_batch_size=300,
        eval_batch_size=300,
        save_eval_checkpoints=False, save_model_every_epoch=False,
        evaluate_during_training=True, evaluate_during_training_steps=10,
        evaluate_during_training_verbose=True, best_model_dir=best_model_dir,
        output_dir=output_dir, cache_dir=cache_dir,
        wandb_kwargs={'tags': [version]},
        # lazy_loading=True,
        # lazy_text_a_column=0, lazy_text_b_column=1,
        # lazy_labels_column=2
    )
    return model_args


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


################################# for policy identification #############################################
def binary_model_config():
    basedir = "/home/ryz/work/from_zj/binary_policy/"
    output_dir = basedir + "output/"
    cache_dir = basedir + "output_cache/"
    best_model_dir = basedir + "best_model/"
    version = "v1"
    model_args = ClassificationArgs(
        wandb_project="policy_binary", n_gpu=4,
        reprocess_input_data=True, use_cached_eval_features=True,
        overwrite_output_dir=True, fp16=True,
        num_train_epochs=6, learning_rate=5e-5,
        logging_steps=20, train_batch_size=400,
        save_eval_checkpoints=False, save_model_every_epoch=False,
        evaluate_during_training=True, evaluate_during_training_steps=10,
        evaluate_during_training_verbose=True, best_model_dir=best_model_dir,
        output_dir=output_dir, cache_dir=cache_dir,
        wandb_kwargs={'tags': [version]})
    return model_args


def binary_class_data():
    train_da = pd.read_hdf("/home/ryz/work/from_zj/data/binary_clas_train.h5", "train")
    eval_da = pd.read_hdf("/home/ryz/work/from_zj/data/binary_clas_eval.h5", "eval")
    ## upsample positive sample equal to negative sample number
    train_pos = resample(train_da[train_da["labels"] == 1], n_samples=train_da["labels"].value_counts()[0],
                         random_state=23).reset_index(drop=True)
    train_data = pd.concat([train_pos, train_da[train_da["labels"] != 1]])
    return train_data, eval_da


def cmd_args():
    from argparse import ArgumentParser
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-r', default=150, help='resample sample of each class.')
    arg_parser.add_argument('-m', default="mc", choices=["mc", "sp"],
                            help='model type in multi-classification and sentence-pair')
    arg_parser.add_argument('-t', required=True, default="mc", choices=["mc", "bc"],
                            help='task type in multi-classification and binary classification for policy headline')
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,7,8'
    args = cmd_args()
    if args.t == "bc":
        train_data, eval_da = binary_class_data()
        model_args = binary_model_config()
        # weight = [1,2] # the ratio of label 0 and 1 number is about 2:1
        model = DDPClassificationModel('bert', "bert-base-chinese", args=model_args, cuda_device=-1)
        model.train_model(train_data, eval_df=eval_da, classification_report=classification_report)
    else:
        if args.m == "sp":
            model_args = sp_model_config()
            train_da, eval_da = sp_data()
            pretrain = "/large_files/pretrained_pytorch/chinese-roberta-wwm-ext"
            model = DDPClassificationModel('bert', pretrain, args=model_args, cuda_device=-1)
            tda = "/home/ryz/work/from_zj/data/multi_clas_sp_train.csv"
            eda = "/home/ryz/work/from_zj/data/multi_clas_sp_eval.csv"
            model.train_model(train_da, eval_df=eval_da)
            # model.train_model(tda, eval_df=eda,weight=[1,10])
        else:

            if args.r:  # default
                # train_data = resample_df(train_df1, int(args.r))
                train_da, eval_da = multiclassfication_data()
            else:
                # train_data = train_df1
                pass
            num_labels = 81
            weight = (1.1 - train_da["labels"].value_counts() / train_da["labels"].value_counts().max()).tolist()
            model_args = multiclassfication_model_config()
            model = DDPClassificationModel('bert', "bert-base-cased", num_labels=num_labels, args=model_args,
                                           cuda_device=-1)
            model.train_model(train_da, eval_df=eval_da, classification_report=classification_report)
