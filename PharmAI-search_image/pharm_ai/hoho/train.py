import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
warnings.simplefilter(action='ignore', category=Warning)

from simpletransformers.classification import ClassificationArgs
from pharm_ai.hoho.dt import HohoDT
from pharm_ai.hoho.model import HohoModel


def train():
    args = ClassificationArgs(
        use_early_stopping=False,
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False,
        save_model_every_epoch=False,
        save_eval_checkpoints=False,
        learning_rate=4e-5,
        train_batch_size=16,
        eval_batch_size=16,
        logging_steps=30,
        evaluate_during_training=True,
        evaluate_during_training_steps=20,
        evaluate_during_training_silent=False,
        num_train_epochs=3
    )
    dt = HohoDT()
    train_df, valid_df = dt.get_training_data()
    model = HohoModel(args).get_model()
    model.train_model(train_df=train_df, eval_df=valid_df)


if __name__ == '__main__':
    train()
