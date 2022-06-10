# encoding: utf-8
"""
@author: zyl
@file: model_train.py
@time: 2021/11/25 13:51
@desc:
"""
from pharm_ai.panel.bio_model import NERBIO

class Trainer(NERBIO):
    def __init__(self):
        super(Trainer, self).__init__()
        self.wandb_proj = 'panel_entity_recognition'

    def run(self):
        self.train_1125()

    def train_1125(self):
        train_file = './data/v4/processing_v4_2_bio.h5 -- disease_train'
        eval_file = './data/v4/processing_v4_2_bio.h5 -- disease_eval'
        train_df = pd.read_hdf('./data/v4/processing_v4_2_bio.h5', 'disease_train')  # type:pd.DataFrame
        eval_df = pd.read_hdf('./data/v4/processing_v4_2_bio.h5', 'disease_eval')  # type:pd.DataFrame
        print(len(train_df))
        print(len(eval_df))
        self.save_dir = './'
        self.model_version = 'erv4.2.0.5'
        self.model_type = 'bert'
        self.pretrained_model = 'bert-base-multilingual-cased'  # 预训练模型位置 model_name
        self.use_cuda = True
        self.cuda_device = 0
        self.labels = ["O", "B-DISEASE", "I-DISEASE"]

        self.model_args = self.my_config()
        self.model_args.update(
            {
                'train_file': train_file,
                'eval_file': eval_file,
                'num_train_epochs': 3,
                'learning_rate': 3e-4,
                'train_batch_size': 24,  # 28
                'gradient_accumulation_steps': 16,
                'eval_batch_size': 16,
                'max_seq_length': 512,
            }
        )

        self.labels = ["O", "B-DISEASE", "I-DISEASE"]
        self.train(train_df, eval_df)

if __name__ == '__main__':
    Trainer().run()
