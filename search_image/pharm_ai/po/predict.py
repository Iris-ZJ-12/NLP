# encoding: utf-8
'''
@author: zyl
@file: predict.py
@time: 2021/8/4 下午10:29
@desc:
'''

import pandas as pd
from sklearn.metrics import classification_report

from pharm_ai.config import ConfigFilePaths
from pharm_ai.po.my_utils import MyModel, DTUtils
from simpletransformers.classification import ClassificationArgs


class PoPredictor(MyModel):
    def __init__(self):
        super(PoPredictor, self).__init__()
        self.start_time = '2021-08'
        self.end_time = '2021-08-25'
        self.bm_version = 'v4.2.0.4'

        self.wandb_proj = 'po'
        self.use_model = 'sentence_pair'  # mt5 /classification
        self.model_type = 'bert'
        self.pretrained_model = ConfigFilePaths.bert_dir_remote

    @MyModel.eval_decoration
    def eval(self, eval_df):
        model = self.get_predict_model()
        eval_df = DTUtils.deal_with_df(eval_df, use_model=self.use_model)
        to_predict_texts = eval_df[['text_a', 'text_b']].values.tolist()
        predicted_labels, raw_outputs = model.predict(to_predict_texts)

        true_labels = eval_df['labels'].tolist()
        report_table = classification_report(true_labels, predicted_labels, digits=4)
        print(report_table)

        report_dict = classification_report(true_labels, predicted_labels, output_dict=True)

        wandb_log_res = {'0_f1': report_dict['0']['f1-score'], '1_f1': report_dict['1']['f1-score'],
                         'sum_f1': report_dict['macro avg']['f1-score']}
        return wandb_log_res

    def predict(self, to_predict_texts):
        model = self.get_predict_model()
        predicted_labels, raw_outputs = model.predict(to_predict_texts)
        return predicted_labels

    @staticmethod
    def api_clean_dt(request_text):
        to_predict_texts = []
        for t in request_text:
            # t = DTUtils.clean_text(t[0])
            if ('No abstract available' in t[1]) or (t[1] == ''):
                to_predict_texts.append([t[0], t[0]])
            else:
                to_predict_texts.append([t[0], DTUtils.clean_text(t[1])])
        return to_predict_texts


class V4Predictor(PoPredictor):
    def __init__(self):
        super(V4Predictor, self).__init__()

    def run(self):
        self.eval_0825()

    def eval_0825(self):
        self.model_version = 'v4.2.0.4'
        self.args = MyModel.set_model_parameter(model_version=self.model_version, args=ClassificationArgs(),
                                                save_dir='po')

        # self.use_cuda = True
        # self.args.quantized_model = True
        # self.args.onnx = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        self.cuda_device = 4
        self.args.n_gpu = 1

        self.args.eval_batch_size = 256
        self.args.max_seq_length = 512

        eval_df = pd.read_excel('./data/v4/processed_0825.xlsx', 'eval')
        self.eval(eval_df)
        eval_df2 = pd.read_excel('./data/v4/processed_0825.xlsx', 'test')
        self.eval(eval_df2)


if __name__ == '__main__':
    V4Predictor().run()
    pass
