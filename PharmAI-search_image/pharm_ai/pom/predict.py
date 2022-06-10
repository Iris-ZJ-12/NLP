# -*- coding: UTF-8 -*-
"""
Description : 
"""
import os
import time
import pandas as pd

from pharm_ai.pom.model import PomModel
from pharm_ai.util.sm_util import SMUtil


class PomPredictor:

    def __init__(self, version, n_gpu=1, project_name='pom'):
        self.ver = version
        self.args = PomModel(version=version, n_gpu=n_gpu, project_name=project_name)

    def get_model(self):
        return self.args.get_predict_model()

    def eval(self, eval_df):
        self.args.hyper_args.use_cached_eval_features = False
        self.args.hyper_args.use_multiprocessing = False
        model = self.get_model()

        res = SMUtil.eval_ner_v2(eval_df, model, metrics_style='loose', check_in_text=False,
                                 delimiter=',')
        return res

    def predict(self, to_predict_texts: list):
        model = self.get_model()
        start_time = time.time()
        res = model.predict(to_predict_texts)
        end_time = time.time()
        need_time = round((end_time - start_time) / len(to_predict_texts), 5)

        print('predict time: {}s * {} = {}s'.format(need_time, len(to_predict_texts),
                                                    round(need_time * len(to_predict_texts), 4)))
        return res


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '8'
    eval_df = pd.read_excel('eval_df.xlsx')
    print(PomPredictor('v1.3').eval(eval_df))
