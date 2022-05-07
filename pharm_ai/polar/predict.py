# encoding: utf-8
'''
@author: zyl
@file: predict.py
@time: 2021/7/21 下午11:55
@desc:
'''
import time

import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from pharm_ai.config import ConfigFilePaths
from pharm_ai.polar.train import PolarTrainer
import wandb


class PolarPredictor(PolarTrainer):
    def __init__(self):
        super(PolarPredictor, self).__init__()

    def run(self):
        self.eval_0723()

    def get_predict_model(self):
        self.args.use_cached_eval_features = False
        return ClassificationModel(model_type=self.model_type, model_name=self.args.best_model_dir,
                                   use_cuda=self.use_cuda, cuda_device=self.cuda_device,
                                   args=self.args)

    def predict(self, to_predict_texts):
        model = self.get_predict_model()
        predicted_labels, raw_outputs = model.predict(to_predict_texts)
        return predicted_labels

    def eval(self, eval_df):
        # deal with eval df
        eval_df = eval_df[['text', 'labels']]
        eval_length = eval_df.shape[0]

        # wand_b
        wandb.init(project=self.wandb_proj, config=self.args,
                   name=self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
                   tags=[self.model_version, 'eval'])

        try:
            start_time = time.time()
            to_predict_texts = eval_df['text'].tolist()
            predicted_labels = self.predict(to_predict_texts)
            true_labels = eval_df['labels'].tolist()
            report_table = classification_report(true_labels, predicted_labels, digits=4)
            report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
            logger.info('eval finished!!!')
            logger.info("Eval result for classifying label 0/1: {}", report_dict)
            print(report_table)
            end_time = time.time()
            need_time = round((end_time - start_time) / eval_length, 5)
            eval_time = round(need_time * eval_length, 4)
            print(f'eval length: {eval_length}')
            print(f'eval time: {need_time} s * {eval_length} = {eval_time} s')
            wandb.log(
                {"eval_length": eval_length, '0_f1': report_dict['0']['f1-score'], '1_f1': report_dict['1']['f1-score'],
                 'sum_f1': report_dict['macro avg']['f1-score']})
        except Exception as error:
            logger.error(f'eval failed!!! ERROR:{error}')
        finally:
            wandb.finish()

    def eval_0723(self):
        self.model_version = 'v0.0.0.1'
        self.args = PolarTrainer.set_model_parameter(model_version=self.model_version)

        self.args.quantized_model = True
        self.args.onnx = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        self.cuda_device = 0
        self.args.n_gpu = 1
        self.use_cuda = False

        self.args.eval_batch_size = 256
        eval_df = pd.read_excel('./data/processed_dt_0722.xlsx', 'eval')
        self.eval(eval_df)


if __name__ == '__main__':
    PolarPredictor().run()
    pass
