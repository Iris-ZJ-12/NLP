import os
import pandas as pd

import warnings
from datetime import datetime
from sklearn.metrics import f1_score
from pharm_ai.util.utils import Utilfuncs as U
from pharm_ai.config import ConfigFilePaths as cfp
from simpletransformers.classification import ClassificationArgs, ClassificationModel
from simpletransformers.classification import DDPClassificationModel


def get_data(path='cls_data.h5'):
    train_df = pd.read_hdf(path, 'train')
    eval_df = pd.read_hdf(path, 'valid')
    return train_df, eval_df


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def train(n_gpu, ddp):
    f = cfp.project_dir + '/ddp/'
    output_dir = os.path.join(f, 'outputs', 'cls')
    cache_dir = os.path.join(f, 'cache', 'cls')
    best_model_dir = os.path.join(f, 'best_model', 'cls')
    args = ClassificationArgs(
        wandb_project="noun",
        n_gpu=n_gpu,
        reprocess_input_data=True,
        use_cached_eval_features=True,
        overwrite_output_dir=True,
        use_multiprocessing=False,
        fp16=False,
        num_train_epochs=2,
        learning_rate=3e-4,
        logging_steps=2,
        train_batch_size=200,
        eval_batch_size=400,
        save_eval_checkpoints=False,
        save_model_every_epoch=False,
        evaluate_during_training=True,
        evaluate_during_training_steps=10,
        evaluate_during_training_verbose=True,
        evaluate_during_training_silent=False,
        output_dir=output_dir,
        cache_dir=cache_dir,
        best_model_dir=best_model_dir,
        wandb_kwargs={'tags': ['4.3']},
        r_drop=True
    )
    train_df, eval_df = get_data()
    U.fix_torch_multiprocessing()
    if ddp:
        model = DDPClassificationModel('bert', cfp.bert_dir_remote, num_labels=6, args=args)
    else:
        model = ClassificationModel('bert', cfp.bert_dir_remote, num_labels=6, args=args)
    model.train_model(train_df, eval_df=eval_df)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,5'
    warnings.simplefilter(action='ignore', category=Warning)
    now = datetime.now()
    train(n_gpu=4, ddp=True)
    print("Training finished with time:", datetime.now() - now)
