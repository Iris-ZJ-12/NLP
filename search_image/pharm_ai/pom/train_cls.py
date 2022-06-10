import os
import warnings

import pandas as pd
import numpy as np
from datetime import datetime
from dt import Task, read_pickle
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from pharm_ai.util.utils import Utilfuncs as U
from pharm_ai.config import ConfigFilePaths as cfp
from simpletransformers.classification import ClassificationArgs, ClassificationModel


def get_data(test_size=0.15):
    t5task = Task(read_pickle('tasks/is_medical_news.pkl'))
    df = t5task.df
    df = df.drop(columns=['prefix'])
    df = df.rename(columns={'input_text': 'text', 'target_text': 'labels'})
    df['labels'] = df['labels'].map(int)
    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=999)
    return train_df, eval_df


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def upsampling(train_df):
    labels = list(set(train_df['labels'].tolist()))
    dfs = []
    for label in labels:
        dfs.append(train_df[train_df['labels'] == label])
    up_sampling_num = max([len(df) for df in dfs])
    up_dfs = []
    for df in dfs:
        if len(df) < up_sampling_num:
            up_sampling_df = resample(
                df,
                replace=True,
                n_samples=(up_sampling_num - len(df)),
                random_state=9999
            )
        else:
            up_sampling_df = pd.DataFrame()
        up_dfs.append(up_sampling_df)
    dfs.extend(up_dfs)
    train_df = pd.concat(dfs, ignore_index=True).sample(frac=1)
    return train_df


def train(n_gpu):
    f = cfp.project_dir + '/pom/'
    output_dir = os.path.join(f, 'outputs', 'cls')
    cache_dir = os.path.join(f, 'cache', 'cls')
    if os.path.exists(cache_dir):
        cmd = 'rm -rf ' + cache_dir
        os.system(cmd)
    best_model_dir = os.path.join(f, 'best_model', 'cls')
    args = ClassificationArgs(
        n_gpu=n_gpu,
        reprocess_input_data=True,
        use_cached_eval_features=True,
        overwrite_output_dir=True,
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False,
        fp16=False,
        num_train_epochs=3,
        # learning_rate=3e-4,
        logging_steps=2,
        train_batch_size=64,
        eval_batch_size=64,
        save_eval_checkpoints=False,
        save_model_every_epoch=False,
        evaluate_during_training=True,
        evaluate_during_training_steps=10,
        evaluate_during_training_verbose=True,
        evaluate_during_training_silent=False,
        output_dir=output_dir,
        cache_dir=cache_dir,
        best_model_dir=best_model_dir,
        early_stopping_metric='f1',
        early_stopping_metric_minimize=False
    )
    train_df, eval_df = get_data()
    train_df = upsampling(train_df)
    eval_df.to_excel('eval_df_cls.xlsx')
    U.fix_torch_multiprocessing()
    pre_path = '/large_files/pretrained_pytorch/chinese-macbert-base/'
    # pre_path = '/large_files/pretrained_pytorch/chinese-roberta-wwm-ext/'
    # pre_path = cfp.bert_dir_remote
    model = ClassificationModel('bert', pre_path, num_labels=2, args=args)
    model.train_model(train_df, eval_df=eval_df, f1=macro_f1)


def evaluate():
    f = cfp.project_dir + '/pom/'
    best_model_dir = os.path.join(f, 'best_model', 'cls')
    args = ClassificationArgs(
        use_multiprocessing_for_evaluation=False,
        eval_batch_size=128,
    )
    eval_df = pd.read_excel('eval_df_cls.xlsx')
    model = ClassificationModel('bert', best_model_dir, num_labels=2, args=args)
    result, outputs, wrongs = model.eval_model(eval_df)
    preds = np.argmax(outputs, axis=1)
    labels = eval_df['labels'].to_list()
    print(classification_report(y_true=labels, y_pred=preds))
    return eval_df, result, outputs, wrongs, labels


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.simplefilter(action='ignore', category=Warning)
    now = datetime.now()
    train(n_gpu=1)
    print("Training finished with time:", datetime.now() - now)
    eval_df, result, outputs, wrongs, labels = evaluate()
