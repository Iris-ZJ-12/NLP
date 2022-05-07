# encoding: utf-8
"""
@author: zyl
@file: eval.py
@time: 2021/11/29 10:00
@desc:
"""
import pandas as pd
from pharm_ai.panel.bio_train import NERBIO

class Evaluator(NERBIO):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.wandb_proj = 'panel_entity_recognition'

    def run(self):
        self.eval_1125()

    def eval_1125(self):
        eval_data = pd.read_hdf('./data/v4/processing_v4_2_bio.h5', 'disease_eval')  # type:pd.DataFrame
        eval_file = './data/v4/processing_v4_2_bio.h5-----disease_eval'

        self.save_dir = './'
        self.model_version = 'erv4.2.0.4'
        self.model_type = 'bert'
        self.use_cuda = True
        self.cuda_device = 3

        self.model_args = self.my_config()
        self.model_args.update(
            {
                'eval_file': eval_file,
                'eval_batch_size': 128,
                # 'max_seq_length': 512,
            }
        )
        self.eval(eval_data, use_t5_matric=True)

if __name__ == '__main__':
    Evaluator().run()


