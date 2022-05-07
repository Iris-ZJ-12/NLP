# encoding: utf-8
'''
@author: zyl
@file: predict_2.py
@time: 2021/8/17 10:37
@desc:
'''

import time

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import classification_report
from pharm_ai.zz.es_utils import ESObject
from simpletransformers.classification import ClassificationModel
# from pharm_ai.zz.m2.model import ZZModel2
# from pharm_ai.zz.m2.train import config_usual_args
from pharm_ai.panel.panel_utils import ModelUtils,DTUtils
import wandb
from pharm_ai.panel.entry_match.train_2 import EMT2
from pharm_ai.panel.entry_match.eval import Evaluator
import os
class EMP2(EMT2):
    def __init__(self):
        super(EMP2, self).__init__()
        self.e = Evaluator()

    def run(self):
        self.eval_0817()

    def get_predict_model(self):
        self.args.use_cached_eval_features = False
        return ClassificationModel(model_type=self.model_type, model_name=self.args.best_model_dir,
                                   use_cuda=self.use_cuda, cuda_device=self.cuda_device,
                                   args=self.args)

    def predict(self,model,to_predict:list):
        res = []
        for i in to_predict:
            r = self.e.predict([i], top_k=10, score_threshold=1.1)
            if r:
                to_pre = r[0]
                # to_predict_model=[[i,j] for j in to_pre]
                to_predict_model = [str(i)+' | '+str(j) for j in to_pre]
                preds, model_outputs = model.predict(to_predict_model)
                preds = preds.tolist()
                if 1 not in preds:
                    res.append({to_pre[0]})
                else:
                    r_= set()
                    for p,q in zip(to_pre,preds):
                        if q==1:
                            r_.add(p)
                    res.append(r_)
            else:
                res.append(set())
        return res

    def predict2(self,to_predict:list):
        res = []
        for i in to_predict:
            r = self.e.predict([i], top_k=10, score_threshold=1.1)
            if r:
                to_pre = r[0]
                res.append({to_pre[0]})
            else:
                res.append(set())
        return res

    def eval_0817(self):
        self.wandb_proj = 'panel_entry_match'
        self.model_version = 'emv2.1.0.6'
        self.args = EMT2.set_model_parameter(model_version=self.model_version)
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        self.cuda_device = 0
        self.args.n_gpu = 1
        self.args.eval_batch_size = 128

        eval_df = pd.read_excel('./data/em_0817.xlsx', 'eval')
        eval_df = eval_df[eval_df['labels'] == 1]
        to_predict = eval_df['text_a'].tolist()

        model = self.get_predict_model()
        res = self.predict(model,to_predict)
        # res = self.predict2( to_predict)
        labels = eval_df['text_b'].tolist()
        l = ModelUtils.revise_target_texts(labels,input_texts=[], check_in_input_text=False, delimiter='|')
        ModelUtils.entity_recognition_v2(l, res)

# def test():
    # def predict_proj(self, model, to_predict_texts, query_max_size=100, return_format='project_name'):
    #     res = []
    #     for text_a in to_predict_texts:
    #         e_s = self.e.predict([sub_df['entity']], top_k=100, score_threshold=score_threshold)
    #
    #         query_result = self.es.fuzzy_match(query_filed_name='text_b', query_filed_value=text_a, get_filed='text_b',
    #                                            query_max_size=query_max_size)
    #         one_all_combinations = [[text_a, text_b] for text_b in query_result]
    #         one_re = model.predict(one_all_combinations)[1]
    #         one_re_index = np.argmax(one_re, axis=0)[1]
    #         one_res = one_all_combinations[one_re_index]  # ['text_a','text_b']
    #         pro_dict = self.es.accurate_match(query_filed_name='text_b', query_filed_value=one_res[1])
    #         pro_dict.update({'text_a': text_a})
    #         res.append(pro_dict)
    #     res_df = pd.DataFrame(res)  # columns:['project_name','text_b','text_a','es_id','province']
    #
    #     if return_format == 'esid':
    #         final_res = res_df['es_id']
    #     elif return_format == 'project_name':
    #         final_res = res_df['project_name']
    #     elif return_format == 'text_b':
    #         final_res = res_df['text_b']
    #     else:
    #         final_res = res_df
    #     return final_res
    #
    # def predict(self, to_predict_texts, query_max_size=100, return_format='project_name'):
    #     model = self.get_predict_model()
    #     final_res = self.predict_proj(model, to_predict_texts=to_predict_texts, query_max_size=query_max_size,
    #                                   return_format=return_format)
    #     return final_res
    #
    # def eval(self, eval_df):
    #     # deal with dt
    #     eval_df = eval_df[['text_a', 'text_b', 'labels']]
    #     eval_df['text_a'] = eval_df['text_a'].astype(str)
    #     eval_df['text_b'] = eval_df['text_b'].astype(str)
    #     true_labels = eval_df['labels'].tolist()
    #     to_predict = eval_df[['text_a', 'text_b']].values.tolist()
    #
    #     eval_length = eval_df.shape[0]
    #
    #     # wand_b
    #     wandb.init(project=self.wandb_proj, config=self.args,
    #                name=self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
    #                tags=[self.model_version, 'eval', 'm2'])
    #
    #     # model
    #     model = self.get_predict_model()
    #     print(f'eval length: {eval_length}')
    #
    #     try:
    #         start_time = time.time()
    #         predicted_labels, _ = model.predict(to_predict)
    #         report_table = classification_report(true_labels, predicted_labels, digits=4)
    #         print(report_table)
    #         report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
    #         logger.info('eval finished!!!')
    #         end_time = time.time()
    #         need_time = round((end_time - start_time) / eval_length, 5)
    #         eval_time = round(need_time * eval_length, 4)
    #         print(f'eval time: {need_time} s * {eval_length} = {eval_time} s')
    #         wandb.log(
    #             {"eval_length": eval_length, '0_f1': report_dict['0']['f1-score'], '1_f1': report_dict['1']['f1-score'],
    #              'sum_f1': report_dict['macro avg']['f1-score']})
    #     except Exception as error:
    #         logger.error(f'eval failed!!! ERROR:{error}')
    #     finally:
    #         wandb.finish()
    #
    # def eval_0713(self):
    #     self.model_version = 'v1.7.0.2'
    #     self.args = ZZTrainerM2.set_model_parameter(model_version=self.model_version)
    #     # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
    #     self.cuda_device = 4
    #     self.args.n_gpu = 1
    #     self.args.eval_batch_size = 300
    #     self.args.quantized_model = True
    #     self.use_cuda = False
    #     self.args.onnx =True
    #
    #     eval_df = pd.read_excel("../data/v1.7/processed_dt0708.xlsx", 'm2_eval')
    #     self.eval(eval_df)
if __name__ == '__main__':
    EMP2().run()
