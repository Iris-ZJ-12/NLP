import time

import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from pharm_ai.config import ConfigFilePaths
from pharm_ai.ddp.ddpclassification_model import DDPClassificationModel
from pharm_ai.zz.m1.model import ZZModelM1
from pharm_ai.zz.m1.train import ZZTrainerM1
import wandb


class ZZPredictorM1(ZZTrainerM1):
    def __init__(self):
        super(ZZPredictorM1, self).__init__()

    def run(self):
        self.eval_0708()

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
                   tags=[self.model_version, 'eval', 'm1'])

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
            wandb.log({"eval_length": eval_length,'0_f1': report_dict['0']['f1-score'], '1_f1': report_dict['1']['f1-score'],
                 'sum_f1': report_dict['macro avg']['f1-score']})
        except Exception as error:
            logger.error(f'eval failed!!! ERROR:{error}')
        finally:
            wandb.finish()

    def eval_0708(self):
        self.model_version = 'v7_0708'
        self.args = ZZTrainerM1.set_model_parameter(model_version=self.model_version)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        self.cuda_device = 2
        self.args.n_gpu = 1
        self.args.quantized_model = True
        self.use_cuda = False
        self.args.onnx = True
        self.args.eval_batch_size = 108
        eval_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/processed_dt0708.xlsx", 'm1_eval')
        self.eval(eval_df)

# class ZZPredictorM1:
#     def __init__(self, version, h5='../data/zz.h5'):
#         self.h5 = h5
#         self.ver = version
#         self.m1 = ZZModelM1(version=version)
#         self.m1.hyper_args.eval_batch_size = 108
#         self.model = self.get_model()
#
#     def get_model(self):
#         return self.m1.get_predict_model()
#
#     def predict(self, to_predict_texts: list):
#         start_time = time.time()
#         predicted_labels, raw_outputs = self.model.predict(to_predict_texts)
#         end_time = time.time()
#         print('m1 predict time: {}s * {}'.format((end_time - start_time) / len(to_predict_texts), len(to_predict_texts)))
#         return predicted_labels
#
#     def eval(self, eval_df):
#         start_time = time.time()
#         result, model_outputs, wrong_predictions = self.model.eval_model(eval_df=eval_df)
#         end_time = time.time()
#         print('m1 eval time: {}s * {}'.format((end_time - start_time) / eval_df.shape[0], eval_df.shape[0]))
#         return result, model_outputs, wrong_predictions
#
#     def do_eval(self):
#         eval_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/processed_dt0708.xlsx", 'm1_eval')
#         true_labels = eval_df['labels'].tolist()
#         predicted_labels = self.predict(eval_df['text'].values.tolist())
#         report_table = classification_report(true_labels, predicted_labels, digits=4)
#         report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
#         logger.info("Eval result for classifying label 0/1: {}", report_dict)
#         print(report_table)
#
#     def do_predict(self):
#         pass


if __name__ == '__main__':
    # ZZPredictorM1('v7_0708').do_eval()
    ZZPredictorM1().run()
