import time

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import classification_report
from pharm_ai.zz.es_utils import ESObject
from simpletransformers.classification import ClassificationModel
# from pharm_ai.zz.m2.model import ZZModel2
# from pharm_ai.zz.m2.train import config_usual_args
import wandb
from pharm_ai.zz.m2.train import ZZTrainerM2


class ZZPredictorM2(ZZTrainerM2):
    def __init__(self):
        super(ZZPredictorM2, self).__init__()
        self.es = ESObject(index_name='zz_projects_0708', index_type='_doc', hosts='101.201.249.176',
                           user_name='elastic', user_password='Zyl123123', port=9325)
        # self.es = ESObject(index_name='zz_projects_0708', index_type='_doc', hosts='0.0.0.0',
        #                    user_name='elastic', user_password='Zyl123123', port=9325)

    def run(self):
        self.eval_0713()

    def get_predict_model(self):
        self.args.use_cached_eval_features = False
        return ClassificationModel(model_type=self.model_type, model_name=self.args.best_model_dir,
                                   use_cuda=self.use_cuda, cuda_device=self.cuda_device,
                                   args=self.args)

    def predict_proj(self, model, to_predict_texts, query_max_size=100, return_format='project_name'):
        res = []
        for text_a in to_predict_texts:
            query_result = self.es.fuzzy_match(query_filed_name='text_b', query_filed_value=text_a, get_filed='text_b',
                                               query_max_size=query_max_size)
            one_all_combinations = [[text_a, text_b] for text_b in query_result]
            one_re = model.predict(one_all_combinations)[1]
            one_re_index = np.argmax(one_re, axis=0)[1]
            one_res = one_all_combinations[one_re_index]  # ['text_a','text_b']
            pro_dict = self.es.accurate_match(query_filed_name='text_b', query_filed_value=one_res[1])
            pro_dict.update({'text_a': text_a})
            res.append(pro_dict)
        res_df = pd.DataFrame(res)  # columns:['project_name','text_b','text_a','es_id','province']

        if return_format == 'esid':
            final_res = res_df['es_id']
        elif return_format == 'project_name':
            final_res = res_df['project_name']
        elif return_format == 'text_b':
            final_res = res_df['text_b']
        else:
            final_res = res_df
        return final_res

    def predict(self, to_predict_texts, query_max_size=100, return_format='project_name'):
        model = self.get_predict_model()
        final_res = self.predict_proj(model, to_predict_texts=to_predict_texts, query_max_size=query_max_size,
                                      return_format=return_format)
        return final_res

    def eval(self, eval_df):
        # deal with dt
        eval_df = eval_df[['text_a', 'text_b', 'labels']]
        eval_df['text_a'] = eval_df['text_a'].astype(str)
        eval_df['text_b'] = eval_df['text_b'].astype(str)
        true_labels = eval_df['labels'].tolist()
        to_predict = eval_df[['text_a', 'text_b']].values.tolist()

        eval_length = eval_df.shape[0]

        # wand_b
        wandb.init(project=self.wandb_proj, config=self.args,
                   name=self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
                   tags=[self.model_version, 'eval', 'm2'])

        # model
        model = self.get_predict_model()
        print(f'eval length: {eval_length}')

        try:
            start_time = time.time()
            predicted_labels, _ = model.predict(to_predict)
            report_table = classification_report(true_labels, predicted_labels, digits=4)
            print(report_table)
            report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
            logger.info('eval finished!!!')
            end_time = time.time()
            need_time = round((end_time - start_time) / eval_length, 5)
            eval_time = round(need_time * eval_length, 4)
            print(f'eval time: {need_time} s * {eval_length} = {eval_time} s')
            wandb.log(
                {"eval_length": eval_length, '0_f1': report_dict['0']['f1-score'], '1_f1': report_dict['1']['f1-score'],
                 'sum_f1': report_dict['macro avg']['f1-score']})
        except Exception as error:
            logger.error(f'eval failed!!! ERROR:{error}')
        finally:
            wandb.finish()

    def eval_0713(self):
        self.model_version = 'v1.7.0.2'
        self.args = ZZTrainerM2.set_model_parameter(model_version=self.model_version)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        self.cuda_device = 4
        self.args.n_gpu = 1
        self.args.eval_batch_size = 300
        self.args.quantized_model = True
        self.use_cuda = False
        self.args.onnx =True

        eval_df = pd.read_excel("../data/v1.7/processed_dt0708.xlsx", 'm2_eval')
        self.eval(eval_df)


# class ZZPredictorM2:
#     def __init__(self):
#         self.start_time = ''
#         self.end_time = ''
#         self.wandb_proj = 'zz'
#         self.es = ESObject(index_name='zz_projects_0708', index_type='_doc', hosts='101.201.249.176',
#                            user_name='elastic', user_password='Zyl123123', port=9325)
#
#         self.use_cuda = True
#         self.cuda_device = 0
#
#         self.model_version = 'test'
#         self.args = self.get_args()
#         self.args.n_gpu = 1
#
#     def get_args(self):
#         return config_usual_args(model_version=self.model_version)
#
#     def get_model(self):
#         self.args.use_cached_eval_features = False
#         return ClassificationModel(model_type='bert', model_name=self.args.best_model_dir,
#                                    use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args)
#
#     def eval(self, eval_df: pd.DataFrame):
#         # deal with dt
#         eval_df = eval_df[['text_a', 'text_b', 'labels']]
#         eval_df['text_a'] = eval_df['text_a'].astype(str)
#         eval_df['text_b'] = eval_df['text_b'].astype(str)
#         true_labels = eval_df['labels'].tolist()
#         to_predict = eval_df[['text_a', 'text_b']].values.tolist()
#
#         eval_length = eval_df.shape[0]
#
#         # wand_b
#         wandb.init(project=self.wandb_proj, config=self.args,
#                    name=self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
#                    tags=[self.model_version, 'eval', 'm2'])
#
#         # model
#         model = self.get_model()
#         print(f'eval length: {eval_length}')
#
#         try:
#             start_time = time.time()
#             predicted_labels, _ = model.predict(to_predict)
#             report_table = classification_report(true_labels, predicted_labels, digits=4)
#             print(report_table)
#             report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
#             logger.info('eval finished!!!')
#             end_time = time.time()
#             need_time = round((end_time - start_time) / eval_length, 5)
#             eval_time = round(need_time * eval_length, 4)
#             print(f'version: {self.model_version}')
#             print(f'eval length: {eval_length}')
#             print(f'eval time: {need_time} s * {eval_length} = {eval_time} s')
#             wandb.log(
#                 {"eval_length": eval_length, '0_f1': report_dict['0']['f1-score'], '1_f1': report_dict['1']['f1-score'],
#                  'sum_f1': report_dict['macro avg']['f1-score']})
#         except Exception as error:
#             logger.error(f'eval failed!!! ERROR:{error}')
#         finally:
#             wandb.finish()
#
#     def predict(self, to_predict_texts: list, return_format='esid', query_max_size=100):
#         start_time = time.time()
#         model = self.get_model()
#         res = []
#         for text_a in to_predict_texts:
#             query_result = self.es.fuzzy_match(query_filed_name='text_b', query_filed_value=text_a, get_filed='text_b',
#                                                query_max_size=query_max_size)
#             one_all_combinations = [[text_a, text_b] for text_b in query_result]
#
#             one_re = model.predict(one_all_combinations)[1]
#             one_re_index = np.argmax(one_re, axis=0)[1]
#             one_res = one_all_combinations[one_re_index]  # ['text_a','text_b']
#
#             pro_dict = self.es.accurate_match(query_filed_name='text_b', query_filed_value=one_res[1])
#             pro_dict.update({'text_a': text_a})
#             res.append(pro_dict)
#         res_df = pd.DataFrame(res)  # columns:['project_name','text_b','text_a','es_id','province']
#         if return_format == 'esid':
#             final_res = res_df['es_id']
#         elif return_format == 'project_name':
#             final_res = res_df['project_name']
#         elif return_format == 'text_b':
#             final_res = res_df['text_b']
#         else:
#             final_res = res_df
#         end_time = time.time()
#         print('m2 predict time: {}s * {}'.format((end_time - start_time) / len(to_predict_texts),
#                                                  len(to_predict_texts)))
#         return final_res
#
#
# class ZZ2V1_7(ZZPredictorM2):
#     def __init__(self):
#         super(ZZ2V1_7, self).__init__()
#
#     def run(self):
#         self.eval_0713()
#
#     def eval_0713(self):
#         self.time = '2021-07-13'
#         self.model_version = 'v1.7.0.5'
#         self.args = self.get_args()
#         # import os
#         # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
#         self.cuda_device = 3
#         self.args.n_gpu = 1
#
#         self.args.eval_batch_size = 148  #
#         # self.args.hyper_args.max_seq_length = 128
#
#         eval_df = pd.read_excel("../data/v1.7/processed_dt0708.xlsx", 'm2_eval')
#         self.eval(eval_df)


# class ZZPredictorM2:
#     def __init__(self, version, h5='../data/zz.h5', index_name='zz_projects_0125'):
#         self.h5 = h5
#         self.ver = version
#         # self.es = ESObject(index_name=index_name, index_type='_doc', hosts='0.0.0.0', user_name='elastic',
#         #                    user_password='Zyl123123', port=9200)
#         self.es = ESObject(index_name=index_name, index_type='_doc', hosts='101.201.249.176', user_name='elastic',
#                            user_password='Zyl123123', port=9200)
#
#         self.m2 = ZZModelM2(version=version)
#         self.m2.hyper_args.eval_batch_size = 10
#         self.model = self.get_model()
#
#     def get_model(self):
#         return self.m2.get_predict_model()
#
#     def predict(self, to_predict_texts: list, return_format='esid', query_max_size=100):
#         start_time = time.time()
#         res = []
#         for text_a in to_predict_texts:
#             query_result = self.es.fuzzy_match(query_filed_name='text_b', query_filed_value=text_a, get_filed='text_b',
#                                                query_max_size=query_max_size)
#             one_all_combinations = [[text_a, text_b] for text_b in query_result]
#             one_re = self.model.predict(one_all_combinations)[1]
#             one_re_index = np.argmax(one_re, axis=0)[1]
#             one_res = one_all_combinations[one_re_index]  # ['text_a','text_b']
#             pro_dict = self.es.accurate_match(query_filed_name='text_b', query_filed_value=one_res[1])
#             pro_dict.update({'text_a': text_a})
#             res.append(pro_dict)
#         res_df = pd.DataFrame(res)  # columns:['project_name','text_b','text_a','es_id','province']
#
#         if return_format == 'esid':
#             final_res = res_df['es_id']
#         elif return_format == 'project_name':
#             final_res = res_df['project_name']
#         elif return_format == 'text_b':
#             final_res = res_df['text_b']
#         else:
#             final_res = res_df
#         end_time = time.time()
#         print('m2 predict time: {}s * {}'.format((end_time - start_time) / len(to_predict_texts),
#                                                  len(to_predict_texts)))
#         return final_res

#     def eval(self, eval_df):
#         start_time = time.time()
#         result, model_outputs, wrong_predictions = self.model.eval_model(eval_df=eval_df)
#         end_time = time.time()
#         print('m2 eval time: {}s * {}'.format((end_time - start_time) / eval_df.shape[0], eval_df.shape[0]))
#         return result, model_outputs, wrong_predictions
#
#     def do_eval(self, metric='metric1'):
#         eval_df = pd.read_hdf(self.h5, self.ver + '/m2_eval')
#         eval_df = eval_df[0:100]
#
#         if metric == 'metric1':
#             result, model_outputs, wrong_predictions = self.eval(eval_df=eval_df)
#             print('result:', result)
#             print('model_outputs:', model_outputs)
#             print('wrong_predictions:', wrong_predictions)
#         elif metric == 'metric2':
#             true_projs = eval_df['text_b'].tolist()
#             predicted_projs = self.predict(eval_df['text_a'].tolist(), return_format='project_name', query_max_size=100)
#             acc_probability = 0
#             for i, j in zip(true_projs, predicted_projs):
#                 if i == j:
#                     acc_probability += (1 / len(true_projs))
#             print('acc probability:', acc_probability)
#         else:
#             eval_df['text_a'] = eval_df['text_a'].astype(str)
#             eval_df['text_b'] = eval_df['text_b'].astype(str)
#             true_labels = eval_df['labels'].tolist()
#
#             predicted_labels, _ = self.model.predict(eval_df[['text_a', 'text_b']].values.tolist())
#             report_table = classification_report(true_labels, predicted_labels, digits=4)
#             report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
#
#             logger.info("Eval result for classifying label 0/1: {}", report_dict)
#             print(report_table)
#
#     def do_predict(self):
#         pass


# class ZZPredictorM2:
#     def __init__(self, model_version, cuda_device=-1, n_gpu=1):
#         self.es = ESObject(index_name='zz_projects_0708', index_type='_doc', hosts='101.201.249.176',
#                            user_name='elastic', user_password='Zyl123123', port=9325)
#         self.args = self.get_args(model_version=model_version, cuda_device=cuda_device, n_gpu=n_gpu)
#
#     def get_args(self, model_version, cuda_device, n_gpu, project_name='zz'):
#         return ZZModel2(model_version=model_version, cuda_device=cuda_device, n_gpu=n_gpu, project_name=project_name)
#
#     def get_model(self):
#         return self.args.get_predict_model()
#
#     def predict(self, model, to_predict_texts: list, return_format='esid', query_max_size=100):
#         start_time = time.time()
#         res = []
#         for text_a in to_predict_texts:
#             query_result = self.es.fuzzy_match(query_filed_name='text_b', query_filed_value=text_a, get_filed='text_b',
#                                                query_max_size=query_max_size)
#             one_all_combinations = [[text_a, text_b] for text_b in query_result]
#             one_re = model.predict(one_all_combinations)[1]
#             one_re_index = np.argmax(one_re, axis=0)[1]
#             one_res = one_all_combinations[one_re_index]  # ['text_a','text_b']
#             pro_dict = self.es.accurate_match(query_filed_name='text_b', query_filed_value=one_res[1])
#             pro_dict.update({'text_a': text_a})
#             res.append(pro_dict)
#         res_df = pd.DataFrame(res)  # columns:['project_name','text_b','text_a','es_id','province']
#
#         if return_format == 'esid':
#             final_res = res_df['es_id']
#         elif return_format == 'project_name':
#             final_res = res_df['project_name']
#         elif return_format == 'text_b':
#             final_res = res_df['text_b']
#         else:
#             final_res = res_df
#         end_time = time.time()
#         print('m2 predict time: {}s * {}'.format((end_time - start_time) / len(to_predict_texts),
#                                                  len(to_predict_texts)))
#         return final_res

# def do_eval(self, metric='metric1'):
#     eval_df = pd.read_hdf(self.h5, self.ver + '/m2_eval')
#     eval_df = eval_df[0:100]
#
#     if metric == 'metric1':
#         result, model_outputs, wrong_predictions = self.eval(eval_df=eval_df)
#         print('result:', result)
#         print('model_outputs:', model_outputs)
#         print('wrong_predictions:', wrong_predictions)
#     elif metric == 'metric2':
#         true_projs = eval_df['text_b'].tolist()
#         predicted_projs = self.predict(eval_df['text_a'].tolist(), return_format='project_name', query_max_size=100)
#         acc_probability = 0
#         for i, j in zip(true_projs, predicted_projs):
#             if i == j:
#                 acc_probability += (1 / len(true_projs))
#         print('acc probability:', acc_probability)
#     else:
#         eval_df['text_a'] = eval_df['text_a'].astype(str)
#         eval_df['text_b'] = eval_df['text_b'].astype(str)
#         true_labels = eval_df['labels'].tolist()
#
#         predicted_labels, _ = self.model.predict(eval_df[['text_a', 'text_b']].values.tolist())
#         report_table = classification_report(true_labels, predicted_labels, digits=4)
#         report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
#
#         logger.info("Eval result for classifying label 0/1: {}", report_dict)
#         print(report_table)
#
# def do_predict(self):
#     pass

# def eval_0713(self):
#     self.args.model_version = 'v1.7.0.2'
#     self.args.hyper_args.wandb_project = 'zz'
#     # self.args.hyper_args.wandb_kwargs = {'pred'}
#     self.args.cuda_device = 4
#     self.args.hyper_args.n_gpu = 1
#     self.args.hyper_args.eval_batch_size = 400
#     model = self.get_model()
#     eval_df = pd.read_excel("../data/v1.7/processed_dt0708.xlsx", 'm2_eval')
#     eval_df['text_a'] = eval_df['text_a'].astype(str)
#     eval_df['text_b'] = eval_df['text_b'].astype(str)
#     true_labels = eval_df['labels'].tolist()
#
#     predicted_labels, _ = model.predict(eval_df[['text_a', 'text_b']].values.tolist())
#     report_table = classification_report(true_labels, predicted_labels, digits=4)
#     report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
#
#     logger.info("Eval result for classifying label 0/1: {}", report_dict)
#     print(report_dict)
#     print(report_table)


if __name__ == '__main__':
    ZZPredictorM2().run()
    # ZZ2V1_7().run()
    # ZZPredictorM2(model_version='s').eval_0713()
    # h5 = '../data/zz.h5'
    # eval_df = pd.read_hdf('../data/zz.h5', 'v6_0128' + '/m2_eval')
    #
    # eval_df = eval_df[0:100]
    #
    # s = ZZPredictorM2('v6_0128').get_model().eval_model(eval_df)
    # print(s)
