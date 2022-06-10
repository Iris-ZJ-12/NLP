# encoding: utf-8
"""
@author: zyl
@file: eval.py
@time: 2021/11/12 16:09
@desc:
"""

# from pharm_ai.panel.entry_match.train_reranker import RerankerTrainer


# class RerankerPredictor:
#     def __init__(self):
#         # self.model_path = "./best_model/v2/v2.2.1/"
#         self.model_path = "./best_model/v1.4.0.0/"
#         self.cuda_device = '1'
#         self.model_dim = 768
#         self.eval_batch_size = 24
#         self.label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
#         self.int2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
#         self.train_num_labels = len(set(self.label2int.values()))
#         self.set_model()
#
#     def run(self):
#         # to_predicts = [["DMAIC", "分枝杆菌病"],
#         #                ["tooth loss", "牙齿脱落"],
#         #                ["CORD", "没有疾病"],
#         #                ["实体恶性肿瘤", "实体瘤"],
#         #                ["viral, parasitic or bacterial diseases", "细菌感染"]]
#         # self.predict(to_predicts)
#         self.eval()
#         pass
#
#     def set_model(self):
#         from sentence_transformers.cross_encoder import CrossEncoder
#         self.model = CrossEncoder(self.model_path, device=f'cuda:{str(self.cuda_device)}',
#                                   num_labels=self.train_num_labels)
#
#     def predict(self, to_predicts: list):
#         import numpy as np
#
#         pred_scores = self.model.predict(to_predicts, convert_to_numpy=True, show_progress_bar=False)
#         # print(pred_scores)
#         pred_labels = np.argmax(pred_scores, axis=1)
#         # relationships = list(map(lambda x: self.int2label.get(x), pred_labels))
#
#         # for t_p, r in zip(to_predicts, relationships):
#         #     print(f'entity:{t_p[0]} --- {r} --- entry:{t_p[1]}')
#         return pred_labels
#
#     def eval(self):
#         eval_df = pd.read_excel('./data/v1/train_1110.xlsx', 'eval_mt5')
#         input_text = eval_df['input_text'].tolist()
#         from dt import ClassificationListEN
#         from tqdm import tqdm
#         all_therapies = ClassificationListEN[0:-1]
#
#         res = []
#         for text_a in tqdm(input_text):
#             to_predicts = [[text_a, t] for t in all_therapies]
#
#             predictions = self.predict(to_predicts)
#             # print(predictions)
#             # predictions.argmax()
#             r = set()
#             for i, j in zip(all_therapies, predictions):
#                 if j != 0:
#                     r.add(i)
#             res.append(r)
#
#         from zyl_utils.model_utils.ner_utils import NERUtils
#         labels = eval_df['therapy_labels'].tolist()
#         labels = NERUtils.revise_target_texts(labels, input_texts=[], delimiter=',')
#
#         res_df = NERUtils.entity_recognition_v2(y_true=labels, y_pred=res)
#         res_dict = {'sum': res_df}
#
#         res = dict()
#         for k, v in res_dict.items():
#             res[k] = {'pos_score': v.iloc[0, -1], 'neg_score': v.iloc[1, -1], 'sum_score': v.iloc[2, -1],
#                       'weighted_score': v.iloc[3, -1], }
#         return res

import pandas as pd
from tqdm import tqdm

from dt import ClassificationListEN
from zyl_utils.model_utils.ner_utils import NERUtils


class EvalModel:
    def __init__(self):
        self.all_therapies = ClassificationListEN[0:-1]

    def run(self):
        self.eval_v1()
        pass

    def get_eval_data(self):
        eval_df = pd.read_excel('./data/v1/train_1110.xlsx', 'eval_mt5')
        to_predict_texts = eval_df['input_text'].tolist()
        labels = eval_df['therapy_labels'].tolist()
        labels = NERUtils.revise_target_texts(labels, input_texts=[], check_in_input_text=False, delimiter=',')
        return to_predict_texts, labels

    def eval_sentence_pair(self, model):
        to_predict_texts, labels = self.get_eval_data()

        res = []
        for text_a in tqdm(to_predict_texts):
            to_predicts = [[text_a, t] for t in self.all_therapies]
            predictions = model.predict(to_predicts, batch_size= 32,
               show_progress_bar = None,
               # num_workers= 2,
               activation_fct = None,
               apply_softmax = False,
             )
            print(predictions)
            # print(predictions)
            # predictions.argmax()

            r = set()
            for i, j in zip(self.all_therapies, predictions):
                if j != 0:
                    r.add(i)
            res.append(r)

        res_df = NERUtils.entity_recognition_v2(y_true=labels, y_pred=res)

        res_dict = {'sum': res_df}

        res = dict()
        for k, v in res_dict.items():
            res[k] = {'pos_score': v.iloc[0, -1], 'neg_score': v.iloc[1, -1], 'sum_score': v.iloc[2, -1],
                      'weighted_score': v.iloc[3, -1]}
        return res

    def get_sbert_model(self, model_path, use_cross_encoder_model=True, cuda_device=None, num_labels=None,
                        max_seq_length=None):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.cross_encoder import CrossEncoder
        if not cuda_device:
            from pharm_ai.ttt.sentence_pair import RerankerTrainer
            cuda_device = RerankerTrainer.get_auto_device()
        if use_cross_encoder_model:
            model = CrossEncoder(model_path, device=f'cuda:1,3',
                                 num_labels=num_labels, max_length=max_seq_length)
            from torch import nn
            model.model = nn.DataParallel(model.model, ...)
        else:
            model = SentenceTransformer(model_path, device=f'cuda:{str(cuda_device)}')
        print(f'use_model: {model_path}')
        return model

    def eval_v1(self):
        model = self.get_sbert_model("cross-encoder/stsb-roberta-base", use_cross_encoder_model=True)
        self.eval_sentence_pair(model)
        pass


if __name__ == '__main__':
    EvalModel().run()
