# encoding: utf-8
'''
@author: zyl
@file: predict.py
@time: 2021/7/26 上午12:33
@desc:
'''
import os
import warnings
from tqdm import tqdm
import pandas as pd
# from sklearn.utils import resample
from pharm_ai.util.mt5_ner import MT5Utils
from pharm_ai.util.utils import Utilfuncs

Utilfuncs.fix_torch_multiprocessing()
warnings.filterwarnings("ignore")


# class TTTPredictor(TTTTrainer):
#     def __init__(self):
#         super(TTTPredictor, self).__init__()
#         self.wandb_proj = 'ttt'
#
#         self.truncating_size = 400
#         self.overlapping_size = 200
#         self.use_truncation = False
#         self.check_in_input_text = False
#         self.delimiter = ','
#
#     def get_predict_model(self, method='mt5'):
#         self.args.use_cached_eval_features = False
#         self.args.update_from_dict(
#             {'truncating_size': self.truncating_size,
#              'overlapping_size': self.overlapping_size,
#              'use_truncation': self.use_truncation,
#              'check_in_input_text': self.check_in_input_text,
#              'delimiter': self.delimiter,
#              }
#         )
#
#         if method == 'sentence_pair':
#             return ClassificationModel(model_type=self.model_type, model_name=self.args.best_model_dir,
#                                        use_cuda=self.use_cuda, cuda_device=self.cuda_device,
#                                        args=self.args)
#         elif method == 'multi_label_classification':
#             return MultiLabelClassificationModel(model_type=self.model_type, model_name=self.args.best_model_dir,
#                                                  num_labels=self.num_labels,
#                                                  use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args,
#                                                  )
#
#         else:
#             return T5Model(model_type=self.model_type, model_name=self.args.best_model_dir,
#                            use_cuda=self.use_cuda, cuda_device=self.cuda_device,
#                            args=self.args)
#
#     def er_predict(self, prefixes, input_texts):
#         model = self.get_predict_model()
#         pred_target_texts = MT5Utils.predict_entity_recognition(model, prefixes, input_texts,
#                                                                 use_truncation=self.use_truncation,
#                                                                 truncating_size=self.truncating_size,
#                                                                 overlapping_size=self.overlapping_size,
#                                                                 delimiter=self.delimiter)
#         pred_target_texts = MT5Utils.revise_target_texts(target_texts=pred_target_texts, input_texts=input_texts,
#                                                          check_in_input_text=self.check_in_input_text,
#                                                          delimiter=self.delimiter)
#         return pred_target_texts  # type:list[set]
#
#     @MT5Utils.eval_decoration
#     def er_eval(self, eval_df, pos_neg_ratio=None):
#         model = self.get_predict_model()
#         eval_res_dict = MT5Utils.eval_entity_recognition(model, eval_df, pos_neg_ratio=pos_neg_ratio,
#                                                          delimiter=self.delimiter,
#                                                          check_in_input_text=self.check_in_input_text,
#                                                          use_truncation=self.use_truncation,
#                                                          truncating_size=self.truncating_size,
#                                                          overlapping_size=self.overlapping_size)
#         res = {}
#         for k, v in eval_res_dict.items():
#             res[k] = {'pos_score': v.iloc[0, -1], 'neg_score': v.iloc[1, -1], 'sum_score': v.iloc[2, -1],
#                       'weighted_score': v.iloc[3, -1], }
#         return res
#
#     @staticmethod
#     def predict2(model, text_as, text_bs):
#         assert len(text_as) == len(text_bs)
#         to_predict = [[a, b] for a, b in zip(text_as, text_bs)]
#         predictions, raw_outputs = model.predict(to_predict)
#         return predictions
#
#     @MT5Utils.eval_decoration
#     def eval2(self, eval_df):
#         model = self.get_predict_model(method='sentence_pair')
#         text_as = eval_df['text_a'].tolist()
#         text_bs = eval_df['text_b'].tolist()
#         preds = TTTPredictor.predict2(model, text_as, text_bs)
#
#         eval_df['predicts'] = preds
#
#         def turn_entities(df):
#             p_r = set()
#             l_r = set()
#             for p, l, t in zip(df['predicts'].tolist(), df['labels'].tolist(), df['text_b'].tolist()):
#                 if p == 1:
#                     if t != 'Therapy Labels: No therapy':
#                         p_r.add(t)
#                 if l == 1:
#                     if t != 'Therapy Labels: No therapy':
#                         l_r.add(t)
#             return pd.Series([p_r, l_r], index=['pred', 'labels'])
#
#         eval_df = eval_df.groupby('text_a').apply(turn_entities)
#         y_true = eval_df['labels'].tolist()
#         y_pred = eval_df['pred'].tolist()
#         res_df = MT5Utils.entity_recognition_v2(y_true=y_true, y_pred=y_pred)
#         res_dict = {'sum': res_df}
#
#         res = dict()
#         for k, v in res_dict.items():
#             res[k] = {'pos_score': v.iloc[0, -1], 'neg_score': v.iloc[1, -1], 'sum_score': v.iloc[2, -1],
#                       'weighted_score': v.iloc[3, -1], }
#         return res
#
#     @MT5Utils.eval_decoration
#     def eval3(self, eval_df):
#         model = self.get_predict_model(method='multi_label_classification')
#
#         eval_df = eval_df[['text', 'labels']]
#         eval_df['labels'] = eval_df['labels'].apply(lambda x: eval(x))
#
#         result, model_outputs, wrong_predictions = model.eval_model(
#             eval_df
#         )
#         print('#' * 30)
#         print(f'r:{result}')
#         to_preds = eval_df['text'].tolist()
#         predictions, _ = model.predict(to_preds)
#         print(predictions)
#         y_pred = []
#         for i in predictions:
#             r = set()
#             for j in range(len(i) - 1):
#                 if i[j] == 1:
#                     r.add(j)
#             y_pred.append(r)
#
#         labels = eval_df['labels'].tolist()
#         y_true = []
#         for i in labels:
#             r = set()
#             for j in range(len(i) - 1):
#                 if i[j] == 1:
#                     r.add(j)
#             y_true.append(r)
#         print(
#             y_true
#         )
#         print(y_pred)
#         res_df = MT5Utils.entity_recognition_v2(y_true=y_true, y_pred=y_pred)
#         res_dict = {'sum': res_df}
#
#         res = dict()
#         for k, v in res_dict.items():
#             res[k] = {'pos_score': v.iloc[0, -1], 'neg_score': v.iloc[1, -1], 'sum_score': v.iloc[2, -1],
#                       'weighted_score': v.iloc[3, -1], }
#         return res


# class PredictorV1(TTTPredictor):
#     def __init__(self):
#         super(PredictorV1, self).__init__()
#         self.start_time = '2021-07-29'
#
#     def run(self):
#         # self.eval_0729()
#         # self.eval_0806()
#         # self.eval_0807()
#         self.eval_0809()
#
#     def eval_0729(self):
#         self.method = 'mt5'
#         self.model_version = 'v1.0.0.4'
#         self.truncating_size = 400
#         self.overlapping_size = 200
#         self.use_truncation = False
#
#         self.args = TTTTrainer.set_model_parameter(model_version=self.model_version)
#         # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#         self.cuda_device = 0
#
#         self.args.n_gpu = 4
#         self.args.eval_batch_size = 128
#         self.args.quantized_model = False
#
#         eval_df = pd.read_excel('./data/dt_0728.xlsx', 'eval')
#         eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
#         eval_df['prefix'] = 'classification'
#         self.er_eval(eval_df)
#
#     def eval_0806(self):
#         self.method = 'sentence_pair'
#         self.model_type = 'bert'
#         self.model_version = 'v1.1.0.0'
#
#         self.args = TTTTrainer.set_model_parameter(model_version=self.model_version,method=self.method)
#         os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#         self.cuda_device = 0
#
#         self.args.n_gpu = 1
#         self.args.eval_batch_size = 128
#         self.args.quantized_model = False
#
#         eval_df = pd.read_excel('./data/dt_0806.xlsx', 'eval')
#         eval_df = eval_df[['text_a', 'text_b', 'labels']]
#         eval_df = eval_df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})
#         self.eval2(eval_df)
#
#     def eval_0807(self):
#         self.method = 'multi_label_classification'
#         self.model_type = 'xlnet'
#         self.model_version = 'v1.2.0.6'
#
#         self.num_labels = 18
#
#         self.args = TTTTrainer.set_model_parameter(model_version=self.model_version, method=self.method)
#         os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#         self.cuda_device = 0
#
#         self.args.n_gpu = 1
#         self.args.eval_batch_size = 64
#         self.args.quantized_model = False
#         self.args.max_seq_length = 512
#
#         # eval_df = pd.read_excel('./data/dt_0807.xlsx', 'eval')
#         eval_df = pd.read_excel('./data/dt_0809.xlsx', 'eval')
#         self.eval3(eval_df)
#
#     def eval_0809(self):
#         self.method = 'mt5'
#         self.model_version = 'v1.3.0.0'
#         self.truncating_size = 400
#         self.overlapping_size = 200
#         self.use_truncation = False
#
#         self.args = TTTTrainer.set_model_parameter(model_version=self.model_version)
#         # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#         self.cuda_device = 0
#
#         self.args.n_gpu = 4
#         self.args.eval_batch_size = 64
#         self.args.max_seq_length = 512
#         self.args.quantized_model = False
#
#         # eval_df = pd.read_excel('./data/dt_0809_2.xlsx', 'eval')
#         # eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
#         # eval_df['prefix'] = 'classification'
#
#         eval_df = pd.read_excel('./data/v1/train_1110.xlsx', 'eval_mt5')
#
#         eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
#         eval_df['prefix'] = 'classification'
#
#         self.er_eval(eval_df)

from zyl_utils.model_utils.models.my_model import MyModel
from zyl_utils.model_utils.ner_utils import NERUtils


class PredictorV1(MyModel):
    def __init__(self):
        super(PredictorV1, self).__init__()
        self.start_time = '2021-07-29'

        self.wandb_proj = 'ttt'
        self.model_version = 'v1.0.0.0'  # to save model or best model

        self.use_model = 'sentence_pair'  # mt5 /classification/sentence_pair
        self.model_type = 'mt5'
        self.pretrained_model = 'roberta-base'  # 预训练模型位置
        self.use_cuda = True
        self.args = MyModel.set_model_parameter(model_version=self.model_version,
                                                args=self._set_args(),
                                                save_dir="/home/zyl/disk/PharmAI/pharm_ai/ttt/")

    def run(self):
        self.eval_1112()
        # self.eval_1112_2()
        # self.eval_0729()
        # self.eval_0806()
        # self.eval_0807()
        # self.eval_0809()

    @MyModel.eval_decoration
    def eval_mt5(self, eval_df):
        model = self.get_predict_model()

        eval_res_dict = NERUtils.eval_entity_recognition(model, eval_df,
                                                         check_in_input_text=False,
                                                         delimiter=',', tokenizer=None,
                                                         use_sliding_window=False,
                                                         use_multi_gpus=None)
        res = {}
        for k, v in eval_res_dict.items():
            res[k] = {'pos_score': v.iloc[0, -1], 'neg_score': v.iloc[1, -1], 'sum_score': v.iloc[2, -1],
                      'weighted_score': v.iloc[3, -1]}
        return res

    # @MyModel.eval_decoration
    def eval_sentence_pair(self, eval_df):
        self.args.silent = True
        model = self.get_predict_model()
        input_text = eval_df['input_text'].tolist()
        from dt import ClassificationListEN
        all_therapies = ClassificationListEN[0:-1]

        res = []
        for text_a in tqdm(input_text):
            to_predicts = [[text_a, t] for t in all_therapies]

            predictions, raw_outputs = model.predict(to_predicts)
            predictions.argmax()
            r = set()
            for i, j in zip(all_therapies, predictions):
                if j == 1:
                    r.add(i)

            res.append(r)

        labels = eval_df['therapy_labels'].tolist()
        labels = NERUtils.revise_target_texts(labels, input_texts=[], delimiter=',')

        res_df = MT5Utils.entity_recognition_v2(y_true=labels, y_pred=res)
        res_dict = {'sum': res_df}

        res = dict()
        for k, v in res_dict.items():
            res[k] = {'pos_score': v.iloc[0, -1], 'neg_score': v.iloc[1, -1], 'sum_score': v.iloc[2, -1],
                      'weighted_score': v.iloc[3, -1], }
        return res

    # print('1')

    # text_as = eval_df['text_a'].tolist()
    # text_bs = eval_df['text_b'].tolist()
    # preds = TTTPredictor.predict2(model, text_as, text_bs)

    # eval_df['predicts'] = preds
    #
    # def turn_entities(df):
    #     p_r = set()
    #     l_r = set()
    #     for p, l, t in zip(df['predicts'].tolist(), df['labels'].tolist(), df['text_b'].tolist()):
    #         if p == 1:
    #             if t != 'Therapy Labels: No therapy':
    #                 p_r.add(t)
    #         if l == 1:
    #             if t != 'Therapy Labels: No therapy':
    #                 l_r.add(t)
    #     return pd.Series([p_r, l_r], index=['pred', 'labels'])
    #
    # eval_df = eval_df.groupby('text_a').apply(turn_entities)
    # y_true = eval_df['labels'].tolist()
    # y_pred = eval_df['pred'].tolist()
    #
    # model.predict()
    #
    # eval_res_dict = NERUtils.eval_entity_recognition(model, eval_df,
    #                                                  check_in_input_text=False,
    #                                                  delimiter=',', tokenizer=None,
    #                                                  use_sliding_window=False,
    #                                                  use_multi_gpus=None)
    # res = {}
    # for k, v in eval_res_dict.items():
    #     res[k] = {'pos_score': v.iloc[0, -1], 'neg_score': v.iloc[1, -1], 'sum_score': v.iloc[2, -1],
    #               'weighted_score': v.iloc[3, -1]}
    # return res

    def eval_1112_2(self):
        self.use_model = 'sentence_pair'
        self.model_type = 'roberta'
        self.model_version = 'v1.1.1.3'

        # self.model_version = "/home/zyl/disk/PharmAI/pharm_ai/"
        # self.use_cuda = True

        self.args = MyModel.set_model_parameter(model_version=self.model_version, args=self._set_args(),
                                                save_dir="/home/zyl/disk/PharmAI/pharm_ai/ttt/")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.args.n_gpu = 1
        self.cuda_device = -1

        self.args.eval_batch_size = 256  # 256
        self.args.max_seq_length = 512

        eval_df = pd.read_excel('./data/v1/train_1110.xlsx', 'eval_mt5')

        eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
        eval_df['prefix'] = 'classification'
        self.eval_sentence_pair(eval_df)

    def eval_1112(self):
        self.use_model = 'mt5'
        self.model_type = 'mt5'
        self.model_version = 'v1.3.0.5'

        self.args = MyModel.set_model_parameter(model_version=self.model_version, args=self._set_args(),
                                                save_dir="/home/zyl/disk/PharmAI/pharm_ai/ttt/")
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        self.use_cuda = True
        self.args.n_gpu = 1
        self.cuda_device = 0

        self.args.eval_batch_size = 64  # 256
        self.args.max_seq_length = 512

        eval_df = pd.read_excel('./data/v1/train_1110.xlsx', 'eval_mt5')

        eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
        eval_df['prefix'] = 'classification'
        self.eval_mt5(eval_df)

        # eval_df = pd.read_json(f'./data/v1/eval_1111.json.gz', compression='gzip')
        # eval_df = eval_df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})
        # eval_df['labels'] = eval_df['labels'].apply(lambda x: 0 if x == 0 else 1)

    # def eval_0729(self):
    #     self.method = 'mt5'
    #     self.model_version = 'v1.0.0.4'
    #     self.truncating_size = 400
    #     self.overlapping_size = 200
    #     self.use_truncation = False
    #
    #     self.args = TTTTrainer.set_model_parameter(model_version=self.model_version)
    #     # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    #     self.cuda_device = 0
    #
    #     self.args.n_gpu = 4
    #     self.args.eval_batch_size = 128
    #     self.args.quantized_model = False
    #
    #     eval_df = pd.read_excel('./data/dt_0728.xlsx', 'eval')
    #     eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
    #     eval_df['prefix'] = 'classification'
    #     self.er_eval(eval_df)
    #
    # def eval_0806(self):
    #     self.method = 'sentence_pair'
    #     self.model_type = 'bert'
    #     self.model_version = 'v1.1.0.0'
    #
    #     self.args = TTTTrainer.set_model_parameter(model_version=self.model_version, method=self.method)
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    #     self.cuda_device = 0
    #
    #     self.args.n_gpu = 1
    #     self.args.eval_batch_size = 128
    #     self.args.quantized_model = False
    #
    #     eval_df = pd.read_excel('./data/dt_0806.xlsx', 'eval')
    #     eval_df = eval_df[['text_a', 'text_b', 'labels']]
    #     eval_df = eval_df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})
    #     self.eval2(eval_df)
    #
    # def eval_0807(self):
    #     self.method = 'multi_label_classification'
    #     self.model_type = 'xlnet'
    #     self.model_version = 'v1.2.0.6'
    #
    #     self.num_labels = 18
    #
    #     self.args = TTTTrainer.set_model_parameter(model_version=self.model_version, method=self.method)
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    #     self.cuda_device = 0
    #
    #     self.args.n_gpu = 1
    #     self.args.eval_batch_size = 64
    #     self.args.quantized_model = False
    #     self.args.max_seq_length = 512
    #
    #     # eval_df = pd.read_excel('./data/dt_0807.xlsx', 'eval')
    #     eval_df = pd.read_excel('./data/dt_0809.xlsx', 'eval')
    #     self.eval3(eval_df)
    #
    # def eval_0809(self):
    #     self.method = 'mt5'
    #     self.model_version = 'v1.3.0.0'
    #     self.truncating_size = 400
    #     self.overlapping_size = 200
    #     self.use_truncation = False
    #
    #     self.args = TTTTrainer.set_model_parameter(model_version=self.model_version)
    #     # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    #     self.cuda_device = 0
    #
    #     self.args.n_gpu = 4
    #     self.args.eval_batch_size = 64
    #     self.args.max_seq_length = 512
    #     self.args.quantized_model = False
    #
    #     # eval_df = pd.read_excel('./data/dt_0809_2.xlsx', 'eval')
    #     # eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
    #     # eval_df['prefix'] = 'classification'
    #
    #     eval_df = pd.read_excel('./data/v1/train_1110.xlsx', 'eval_mt5')
    #
    #     eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
    #     eval_df['prefix'] = 'classification'
    #
    #     self.er_eval(eval_df)


if __name__ == '__main__':
    PredictorV1().run()
