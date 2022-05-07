# encoding: utf-8
'''
@author: zyl
@file: train_2.py
@time: 2021/8/16 15:56
@desc:
'''
import time
import warnings

import pandas as pd
import wandb
from loguru import logger
# from sklearn.utils import resample
# import os

from simpletransformers.classification import ClassificationModel, ClassificationArgs, DDPClassificationModel

from pharm_ai.config import ConfigFilePaths
# from pharm_ai.ddp.ddpclassification_model import DDPClassificationModel
from pharm_ai.util.utils import Utilfuncs

Utilfuncs.fix_torch_multiprocessing()
warnings.filterwarnings("ignore")
import os
from scipy.stats import pearsonr, spearmanr


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]

def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])


class EMT2:
    def __init__(self):
        self.start_time = ''
        self.end_time = ''

        self.wandb_proj = 'panel_entry_match'
        self.model_version = 'test'

        self.model_type = 'bert'
        self.pretrained_model = ConfigFilePaths.bert_dir_remote
        self.use_cuda = True
        self.cuda_device = 0
        self.args = EMT2.set_model_parameter(model_version=self.model_version)

    def run(self):
        self.train_0817()

    @staticmethod
    def set_model_parameter(model_version, args=ClassificationArgs()):
        # multiprocess
        args.use_multiprocessing = False
        args.use_multiprocessing_for_evaluation = False

        # base config
        args.reprocess_input_data = True
        args.fp16 = False
        args.manual_seed = 234
        # args.gradient_accumulation_steps = 8  # ==increase batch size,Use time for memory,

        # save
        args.no_save = False
        args.save_eval_checkpoints = False
        args.save_model_every_epoch = False
        args.save_optimizer_and_scheduler = True
        args.save_steps = -1

        # eval
        args.evaluate_during_training = True
        args.evaluate_during_training_verbose = True

        args.no_cache = False
        args.use_early_stopping = False
        args.encoding = None
        args.do_lower_case = False
        args.dynamic_quantize = False
        args.quantized_model = False
        args.silent = False

        args.overwrite_output_dir = True
        saved_dirs = '/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/'
        args.output_dir = saved_dirs + 'outputs/' + model_version + '/'
        args.cache_dir = saved_dirs + 'cache/' + model_version + '/'
        args.best_model_dir = saved_dirs + 'best_model/' + model_version + '/'
        args.tensorboard_dir = saved_dirs + 'runs/' + model_version + '/' + time.strftime("%Y%m%d_%H%M%S",
                                                                                          time.localtime()) + '/'
        return args

    def get_train_model(self):
        self.args.use_cached_eval_features = False
        if self.args.n_gpu <= 1:
            return ClassificationModel(model_type=self.model_type, model_name=self.pretrained_model,
                                       use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args)
        else:
            return DDPClassificationModel(model_type=self.model_type, model_name=self.pretrained_model, use_cuda=True,
                                          cuda_device=-1, args=self.args)

    def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame):
        # deal with dt
        train_df = train_df[['text_a', 'text_b', 'labels']]
        eval_df = eval_df[['text_a', 'text_b', 'labels']]
        train_df['text_a'] = train_df['text_a'].astype(str)
        train_df['text_b'] = train_df['text_b'].astype(str)
        train_df['labels'] = train_df['labels'].astype(int)
        train_df['text'] = train_df['text_a'].str.cat(train_df['text_b'], sep=' | ')
        train_df = train_df[['text', 'labels']]
        eval_df['text_a'] = eval_df['text_a'].astype(str)
        eval_df['text_b'] = eval_df['text_b'].astype(str)
        eval_df['labels'] = eval_df['labels'].astype(int)
        eval_df['text'] = eval_df['text_a'].str.cat(eval_df['text_b'], sep=' | ')
        eval_df = eval_df[['text', 'labels']]

        # config some parameters
        train_size = train_df.shape[0]
        all_steps = train_size / self.args.train_batch_size
        self.args.logging_steps = int(all_steps / 20)
        self.args.evaluate_during_training_steps = int(all_steps / 5 / self.args.gradient_accumulation_steps)

        self.args.wandb_project = self.wandb_proj
        self.args.wandb_kwargs = {
            'name': self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
            'tags': [self.model_version, 'train']}

        # get model
        model = self.get_train_model()
        print(f'train length: {train_size}')
        model.args.update_from_dict({'train_length': train_size})

        # train
        try:
            start_time = time.time()
            model.train_model(train_df=train_df, eval_df=eval_df)
            logger.info('train finished!!!')
            end_time = time.time()
            need_time = round((end_time - start_time) / train_size, 4)
            training_time = round(need_time * train_size, 4)
            print(f'train time: {need_time} s * {train_size} = {training_time} s')
        except Exception as error:
            logger.error(f'train failed!!! ERROR:{error}')
        finally:
            wandb.finish()

    def train_0817(self):
        self.wandb_proj = 'panel_entry_match'
        self.model_version = 'emv2.1.0.7'

        self.args = EMT2.set_model_parameter(model_version=self.model_version)
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        self.cuda_device = 0
        self.args.n_gpu = 1

        self.args.num_train_epochs = 5
        self.args.learning_rate = 1e-6
        self.args.train_batch_size = 128
        self.args.eval_batch_size = 64
        self.args.gradient_accumulation_steps = 4
        # self.args.hyper_args.max_seq_length = 128
        train_df = pd.read_excel('./data/em_0817.xlsx', 'train_up')
        eval_df = pd.read_excel('./data/em_0817.xlsx', 'eval')
        self.train(train_df, eval_df)

    def train_0830(self):
        self.wandb_proj = 'panel_entry_match'
        self.model_version = 'emv2.1.0.8'

        self.args = EMT2.set_model_parameter(model_version=self.model_version)
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        self.cuda_device = 0
        self.args.n_gpu = 1

        self.args.num_train_epochs = 5
        self.args.learning_rate = 1e-6
        self.args.train_batch_size = 128
        self.args.eval_batch_size = 64
        self.args.gradient_accumulation_steps = 4
        # self.args.hyper_args.max_seq_length = 128
        train_df = pd.read_excel('./data/em_0817.xlsx', 'train_up')
        eval_df = pd.read_excel('./data/em_0817.xlsx', 'eval')
        self.train(train_df, eval_df)


from pharm_ai.panel.my_utils import MyModel, DTUtils, ModelUtils
from simpletransformers.t5 import T5Args
from pharm_ai.panel.panel_utils import ModelUtils


class V2Trainer(MyModel):
    def __init__(self):
        super(V2Trainer, self).__init__()
        self.start_time = '2021-08'
        self.end_time = '2021-0~'
        self.bm_version = '~'

        self.wandb_proj = 'panel_entry_match'
        self.use_model = 'mt5'  # mt5 /classification
        self.model_type = 'mt5'
        self.pretrained_model = ConfigFilePaths.mt5_base_remote

    def run(self):
        # self.train_0830()
        self.eval_0830()
        pass

    def train_0830(self):
        self.model_version = 'emv2.2.0.4'
        self.pretrained_model = '/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/emv2.2.0.2/'
        self.args = MyModel.set_model_parameter(model_version=self.model_version, args=T5Args(),
                                                save_dir='panel/entry_match')
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
        self.cuda_device = 0
        self.args.n_gpu = 5

        self.args.num_train_epochs = 1
        self.args.learning_rate = 1e-4
        self.args.train_batch_size = 42  # 512
        self.args.eval_batch_size = 32  # 256
        # self.args.max_seq_length = 128
        self.args.gradient_accumulation_steps = 1  # 256

        train_df = pd.read_excel('./data/em_0830.xlsx', 'train')
        from sklearn.utils import resample

        train_neg = train_df[train_df['target_text']==0][0:12000]
        train_pos = train_df[train_df['target_text']==1]
        train_df = pd.concat([train_neg,train_pos],ignore_index=True)
        train_df = resample(train_df, replace=False)

        eval_df = pd.read_excel('./data/em_0830.xlsx', 'eval')
        eval_neg = eval_df[eval_df['target_text'] == 0][0:1000]
        eval_pos = eval_df[eval_df['target_text'] == 1]
        eval_df = pd.concat([eval_neg, eval_pos], ignore_index=True)
        eval_df = resample(eval_df, replace=False)
        self.train(train_df=train_df, eval_df=eval_df)

    @staticmethod
    def apply_df(df):
        r = set()
        r2 = set()
        for i, j, z in zip(df['entries'].tolist(), df['target_text'].tolist(), df['predicted_labels'].tolist()):
            if j == '1':
                r.add(i)
            if z == '1':
                r2.add(i)

        return pd.Series({'true': r, 'pred': r2})

    @MyModel.eval_decoration
    def eval(self, eval_df):
        model = self.get_predict_model()
        eval_df = DTUtils.deal_with_df(eval_df, use_model=self.use_model)  # type:pd.DataFrame
        to_predict_texts = eval_df['input_text'].tolist()
        from pharm_ai.panel.entry_match.eval import Evaluator
        e = Evaluator()
        preds = []
        df22 = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/entry_dict_0508.xlsx",
                             "disease_dict")
        all_entities = list(set(df22['entry'].tolist()))
        to_p = []
        e_s_ = []
        for t in to_predict_texts:
            e_s = e.predict([t], top_k=100, score_threshold=1.1)
            import random
            if (e_s == [[]]) | (e_s == []):
                e_s = random.sample(all_entities, 10)
            else:
                e_s = e_s[0][0:10]
            e_s_.extend(e_s)
            to_pred = ['disease: '+str(t)+' | '+str(j) for j in e_s]
            # print(to_pred)
            to_p.extend(to_pred)
        predicted_labels = model.predict(to_p)
        # print(predicted_labels)

        for i in range(0,len(predicted_labels),10):
            r = set()
            for p, q in zip(predicted_labels[i:i+10], e_s_[i:i+10]):
                if p == '1' or p == 1:
                    r.add(str(q))

            preds.append(r)
        # print(preds)

        true = ModelUtils.revise_target_texts(eval_df['target_text'].tolist(),[],False, delimiter='|')
        # print(true)

        res = ModelUtils.entity_recognition_v2(true,preds)
        print(res)

        # entities = []
        # entries = []
        # for i, j in zip(eval_df['prefix'].tolist(), eval_df['input_text'].tolist()):
        #     to_predict_texts.append(i + ': ' + j)
        #     entities.append(j.split(' | ')[0])
        #     entries.append(j.split(' | ')[1])
        # predicted_labels = model.predict(to_predict_texts)
        # # print(predicted_labels)
        # eval_df['predicted_labels'] = predicted_labels
        # eval_df['entities'] = entities
        # eval_df['entries'] = entries
        #
        # df = eval_df.groupby('entities').apply(V2Trainer.apply_df)
        # # print(df)
        # res_df = ModelUtils.entity_recognition_v2(df['true'].tolist(), df['pred'].tolist())

        # true_labels = eval_df['labels'].tolist()
        # report_table = classification_report(true_labels, predicted_labels, digits=4)
        # print(report_table)
        #
        # report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
        #
        wandb_log_res = {'res': 0}
        return wandb_log_res

    def eval_0830(self):
        self.model_version = 'emv2.2.0.4'
        self.args = MyModel.set_model_parameter(model_version=self.model_version, args=T5Args(),
                                                save_dir='panel/entry_match')

        # self.use_cuda = True
        # self.args.quantized_model = True
        # self.args.onnx = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        self.cuda_device = 4
        self.args.n_gpu = 1

        self.args.eval_batch_size = 10

        eval_df = pd.read_excel('./data/em_0830.xlsx', 'eval_old')
        self.eval(eval_df)



if __name__ == '__main__':
    # EMT2().run()
    V2Trainer().run()
    pass
