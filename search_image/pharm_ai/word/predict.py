# -*- coding: UTF-8 -*-
"""
Description : 
"""
import pandas as pd
import time
from pharm_ai.word.model import WordModelT5
import wandb
from pharm_ai.util.sm_util import SMUtil


class WordPredictorT5:
    def __init__(self, version, cuda_device, n_gpu):
        self.args = WordModelT5(version=version, cuda_device=cuda_device, n_gpu=n_gpu)
        self.args.hyper_args.eval_batch_size = 32
        self.args.hyper_args.num_beams = 2
        self.args.hyper_args.max_length = 20
        self.args.hyper_args.max_seq_length = 144
        self.args.hyper_args.length_penalty = 2

        self.model = self.get_model()
        wandb.init(project=self.args.pn, config=self.args.hyper_args,
                   name=version + time.strftime("_P_%m%d_%H:%M:%S", time.localtime()),
                   tags=['.'.join(version.split('.')[0:-1])+ '_predict'],)

    def get_model(self):
        return self.args.get_predict_model()

    def predict(self, to_predict_texts: list):
        start_time = time.time()
        preds = self.model.predict(to_predict_texts)
        end_time = time.time()

        need_time = round((end_time - start_time) / len(to_predict_texts), 5)
        print('predict time: {}s * {} = {}s'.format(need_time, len(to_predict_texts),
                                                    round(need_time * len(to_predict_texts), 5)))
        return preds

    def eval(self, eval_df: pd.DataFrame, delimiter='|'):
        start_time = time.time()
        res = SMUtil.eval_ner_v2(eval_df, self.model, metrics_style='loose', check_in_text=False, delimiter=delimiter)
        end_time = time.time()
        need_time = round((end_time - start_time) / eval_df.shape[0], 5)
        print('results:', res)
        print('eval time: {}s * {} = {}s'.format(need_time, eval_df.shape[0], round(need_time * eval_df.shape[0], 5)))
        wandb.log(
            {
                "res": res,
                "time(s)": round((end_time - start_time) / eval_df.shape[0], 5)
            }
        )


class V2_0(WordPredictorT5):
    def __init__(self, version='v2.0.0'):
        super(V2_0, self).__init__(version=version, cuda_device=-1, n_gpu=1)

    def do_predict_2(self):
        eval_df = pd.read_excel('./data/v2.0/v2_0_2_0309.xlsx', sheet_name='eval')
        eval_df = eval_df.astype('str')
        eval_df = eval_df[eval_df['target_text'] != '0']
        self.eval(eval_df)

    def do_predict_5(self):
        eval_df = pd.read_excel('./data/v2.0/v2_0_5_0310_160449.xlsx', sheet_name='eval')
        eval_df = eval_df.astype('str')
        eval_df = eval_df[eval_df['target_text'] != '|']
        self.eval(eval_df)


if __name__ == '__main__':
    # V2_0(version='v2.0.4').do_predict_2()
    V2_0(version='v2.0.5').do_predict_5()
