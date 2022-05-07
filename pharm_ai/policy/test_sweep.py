import logging
import os

import pandas as pd
import wandb
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import classification_report
from sklearn.utils import resample

'''
This script is used for testing hyperparameter search by wandb sweep.
'''


def binary_class_data():
    train_da = pd.read_hdf("/home/ryz/work/from_zj/data/binary_clas_train.h5", "train")
    eval_da = pd.read_hdf("/home/ryz/work/from_zj/data/binary_clas_eval.h5", "eval")
    ## upsample positive sample equal to negative sample number
    train_pos = resample(train_da[train_da["labels"] == 1], n_samples=train_da["labels"].value_counts()[0],
                         random_state=23).reset_index(drop=True)
    train_data = pd.concat([train_pos, train_da[train_da["labels"] != 1]])
    train_data = train_data.sample(2000)
    eval_da = eval_da.sample(500)
    return train_data, eval_da


def binary_model_config():
    basedir = "/home/ryz/work/from_zj/binary_policy_test/"
    output_dir = basedir + "output/"
    cache_dir = basedir + "output_cache/"
    best_model_dir = basedir + "best_model/"
    version = "v1"
    model_args = ClassificationArgs(
        wandb_project="policy_binary_test", n_gpu=4,
        reprocess_input_data=True, use_cached_eval_features=True,
        overwrite_output_dir=True, fp16=True,
        num_train_epochs=6,
        logging_steps=20, train_batch_size=300,
        save_eval_checkpoints=False, save_model_every_epoch=False,
        evaluate_during_training=True, evaluate_during_training_steps=10,
        evaluate_during_training_verbose=True, best_model_dir=best_model_dir,
        output_dir=output_dir, cache_dir=cache_dir,
        wandb_kwargs={'tags': [version]})
    return model_args


sweep_config = {
    "name": "test2",
    "method": "bayes",  # grid, random
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "num_train_epochs": {"values": [3, 5, 7]},
        "learning_rate": {"min": 5e-5, "max": 4e-4},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 6, },
}

# return an id used in wandb.agent
sweep_id = wandb.sweep(sweep_config, project="Sweep Hyperparameter Optimization")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data, eval_da = binary_class_data()
model_args = binary_model_config()


# weight = [1,2] # the ratio of label 0 and 1 number is about 2:1
# model = DDPClassificationModel('bert', "bert-base-chinese", args=model_args, cuda_device=-1)
# model.train_model(train_data, eval_df=eval_da, classification_report=classification_report)


def train():
    wandb.init(group="test")
    model = ClassificationModel("bert", "bert-base-chinese", use_cuda=True, args=model_args,
                                sweep_config=wandb.config, )
    # Train the model
    model.train_model(train_data, eval_df=eval_da, classification_report=classification_report)
    # Evaluate the model
    # model.eval_model(eval_da)
    wandb.finish()  # same with finish


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,7'
    wandb.agent(sweep_id, train, count=3)
