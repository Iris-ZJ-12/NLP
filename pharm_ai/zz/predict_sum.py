import numpy as np
import pandas as pd
from itertools import product
from loguru import logger
from sklearn.metrics import classification_report
from pharm_ai.zz.es_utils import ESObject
import time
from pharm_ai.zz.m1.predict import ZZPredictorM1
from pharm_ai.zz.m2.predict import ZZPredictorM2


class ZZPredictorSum:
    def __init__(self, version, h5='./data/data.h5'):
        self.h5 = h5
        self.ver = version

        self.m1 = ZZPredictorM1()
        self.m1.model_version = 'v7_0708'
        self.m1.args = self.m1.set_model_parameter(model_version=self.m1.model_version)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        self.m1.cuda_device = 4
        self.m1.args.n_gpu = 1
        self.m1.args.eval_batch_size = 108

        self.m2 = ZZPredictorM2()
        self.m2.model_version = 'v1.7.0.2'
        self.m2.args = self.m2.set_model_parameter(model_version=self.m2.model_version)
        self.m2.cuda_device = 4
        self.m2.args.n_gpu = 1
        self.m2.args.eval_batch_size = 300

        # self.projects_list = pd.read_hdf('./data/projs.h5', 'projects1214')['ESID'].to_list()
        self.projects_list = ESObject(index_name='zz_projects_0708', index_type='_doc', hosts='101.201.249.176',
                                      user_name='elastic',
                                      user_password='Zyl123123', port=9325).get_all_values_about_filed('es_id')

    def generate_true_pred(self, dt: pd.DataFrame):
        joined_texts = dt.drop(columns=['labels', 'project_names', 'project_id']).apply(
            lambda strs: ' '.join(strs), axis=1)
        pred_res1 = self.m1.predict(joined_texts.values.tolist())
        to_predict2 = joined_texts[np.array(pred_res1, dtype=np.bool)]
        pred_res2 = self.m2.predict(to_predict2.values, return_format='esid', query_max_size=100)  # text_bs
        pred_res2 = pd.Series(pred_res2.values, index=to_predict2.index, name='project_id_predicted')
        dt['labels_predicted'] = pred_res1
        dt = dt.join(pred_res2)
        dt['project_id_predicted'] = dt['project_id_predicted'].replace({np.NaN: None})
        dt = self.convert_to_sentense_pair_format(dt)
        return dt

    def convert_to_sentense_pair_format(self, dt):
        expand_df = dt[dt.labels == dt.labels_predicted].reset_index()
        expand_df = expand_df.drop(columns='project_names').rename(columns={'index': 'match_id'})
        expand_df_grs = expand_df.groupby('labels')
        expand_df_pos = expand_df_grs.get_group(1)
        expanded_ids = pd.DataFrame(product(expand_df_pos.match_id.to_list(), self.projects_list),
                                    columns=['match_id', 'project_id'])
        expanded_ids_true = expanded_ids.merge(expand_df_pos[['match_id', 'title', 'province', 'date']], "left")
        expand_df_pos_true = expanded_ids_true.merge(
            expand_df_pos.drop(columns=['labels', 'labels_predicted', 'project_id_predicted']).assign(labels_true=1),
            "left").fillna({'labels_true': 0})
        expanded_ids_pred = expanded_ids.rename(columns={'project_id': 'project_id_predicted'}).merge(
            expand_df_pos[['match_id', 'title', 'province', 'date']], "left")
        expand_df_pos_pred = expanded_ids_pred.merge(
            expand_df_pos.drop(columns=['labels', 'labels_predicted', 'project_id']).assign(labels_predicted=1),
            "left").fillna({'labels_predicted': 0})
        expand_df_pos_true_pred = expand_df_pos_true.merge(
            expand_df_pos_pred.rename(columns={'project_id_predicted': 'project_id'}))
        expand_df_pos_true_pred = expand_df_pos_true_pred.astype({'labels_true': 'int', 'labels_predicted': 'int'})
        return expand_df_pos_true_pred

    def predict(self, df):
        dt_true_pred = self.generate_true_pred(df)
        trues = dt_true_pred.labels_true.to_list()
        preds = dt_true_pred.labels_predicted.to_list()
        print(len(preds))
        report_table = classification_report(trues, preds, digits=4)
        print(report_table)
        report_dict = classification_report(trues, preds, output_dict=True)
        logger.info("Test result for entire zz project:\n{}", report_dict)

    def do_predict0129(self):
        # dt = pd.read_hdf(h5, 'dt1214/sum_test')[-20:-1]
        dt = pd.read_excel('./data/v6/processed_dt0128.xlsx', 'sum_eval')
        dt.rename(columns={'province_x': 'province', 'ESID': 'project_id', 'project_name': 'project_names'},
                  inplace=True)
        dt = dt[['title', 'province', 'date', 'labels', 'project_names', 'project_id']]
        dt['title'] = dt['title'].astype(str)
        dt['province'] = dt['province'].astype(str)
        dt['date'] = dt['date'].astype(str)
        dt.fillna('None', inplace=True)

        start_time = time.time()
        self.predict(dt)
        end_time = time.time()
        print('sum predict time: {}s * {}'.format((end_time - start_time) / dt.shape[0], dt.shape[0]))

    def eval_0713(self):
        dt = pd.read_excel("./data/v1.7/processed_dt0708.xlsx", 'sum_eval')
        dt.rename(columns={'province_x': 'province', 'ESID': 'project_id', 'project_name': 'project_names'},
                  inplace=True)
        dt = dt[['title', 'province', 'date', 'labels', 'project_names', 'project_id']]
        dt['title'] = dt['title'].astype(str)
        dt['province'] = dt['province'].astype(str)
        dt['date'] = dt['date'].astype(str)
        dt.fillna('None', inplace=True)

        start_time = time.time()
        self.predict(dt)
        end_time = time.time()
        print('sum predict time: {}s * {}'.format((end_time - start_time) / dt.shape[0], dt.shape[0]))


if __name__ == '__main__':
    # ZZPredictorSum('v6_0128').do_predict0129()
    ZZPredictorSum('0713').eval_0713()
