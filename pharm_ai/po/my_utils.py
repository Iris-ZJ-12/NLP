# encoding: utf-8
'''
@author: zyl
@file: my_utils.py
@time: ~~
@desc: zyl utils
'''
import os
import time

import pandas as pd
import wandb
from loguru import logger

from pharm_ai.config import ConfigFilePaths
from pharm_ai.panel.t_t5 import T5Model
from pharm_ai.util.utils import Utilfuncs
from simpletransformers.classification import ClassificationModel, ClassificationArgs, DDPClassificationModel
from simpletransformers.t5 import DDPT5Model


class DTUtils:
    def __init__(self):
        pass

    @staticmethod
    def df_clean_language(df, column_name, language_list=('en', 'zh')):
        # dataframe过滤出某一列文本的语言
        import langid
        df['language'] = df[column_name].apply(lambda x: langid.classify(str(x))[0])
        df = df[df['language'].isin(language_list)]
        df = df.drop(['language'], axis=1)
        return df

    @staticmethod
    def clean_text(text):
        import re
        text = re.sub('<[^<]+?>', '', text).replace('\n', '').strip()  # 去html中的<>标签
        text = ' '.join(text.split()).strip()
        return text

    @staticmethod
    def cut_train_eval(all_df):
        from sklearn.utils import resample
        raw_df = resample(all_df, replace=False)
        cut_point = min(8000, int(0.2 * len(raw_df)))
        eval_df = raw_df[0:cut_point]
        train_df = raw_df[cut_point:]
        return train_df, eval_df

    @staticmethod
    def two_classification_up_sampling(train_df, column='labels', pos_label=1):
        import pandas as pd
        from sklearn.utils import resample
        negative_df = train_df[train_df[column] != pos_label]
        neg_len = negative_df.shape[0]
        positive_df = train_df[train_df[column] == pos_label]
        pos_len = positive_df.shape[0]
        if neg_len > pos_len:
            up_sampling_df = resample(positive_df, replace=True, n_samples=(neg_len - pos_len), random_state=242)
            return pd.concat([train_df, up_sampling_df], ignore_index=True)
        elif neg_len < pos_len:
            up_sampling_df = resample(negative_df, replace=True, n_samples=(pos_len - neg_len), random_state=242)
            return pd.concat([train_df, up_sampling_df], ignore_index=True)
        else:
            return train_df

    @staticmethod
    def deal_with_df(df, use_model='cls'):
        if use_model == 't5' or use_model == 'mt5':
            df = df[['prefix', 'input_text', 'target_text']]
            df = df.astype('str')
        elif use_model == 'sentence_pair':
            df = df[['text_a', 'text_b', 'labels']]
            df = df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})
        else:
            df = df[['text', 'labels']]
            df = df.astype({'text': 'str', 'labels': 'int'})
        return df


class ModelUtils:
    def __init__(self):
        pass

    @staticmethod
    def send_to_me(message):
        sender_email = "pharm_ai_group@163.com"
        sender_password = "SYPZFDNDNIAWQJBL"  # This is authorization password, actual password: pharm_ai163
        sender_smtp_server = "smtp.163.com"
        send_to = "1137379695@qq.com"
        Utilfuncs.send_email_notification(sender_email, sender_password, sender_smtp_server,
                                          send_to, message)

    @staticmethod
    def remove_some_model_files(args):
        import os
        if os.path.isdir(args.output_dir):
            cmd = 'rm -rf ' + args.output_dir.split('outputs')[0] + 'outputs/'
            os.system(cmd)
        if os.path.isdir(args.output_dir.split('outputs')[0] + '__pycache__/'):
            cmd = 'rm -rf ' + args.output_dir.split('outputs')[0] + '__pycache__/'
            os.system(cmd)
        if os.path.isdir(args.output_dir.split('outputs')[0] + 'cache/'):
            cmd = 'rm -rf ' + args.output_dir.split('outputs')[0] + 'cache/'
            os.system(cmd)


class MyModel:
    def __init__(self):
        self.start_time = '...'
        self.end_time = '...'

        self.wandb_proj = 'po'
        self.model_version = 'test'  # to save model

        self.use_model = 'classification'  # mt5 /classification
        self.model_type = 'bert'
        self.pretrained_model = ConfigFilePaths.bert_dir_remote
        self.use_cuda = True
        self.cuda_device = 0
        self.num_labels = 2
        self.args = MyModel.set_model_parameter(model_version=self.model_version)

    @staticmethod
    def set_model_parameter(model_version='test', args=ClassificationArgs(), save_dir='test'):
        # multiprocess
        args.use_multiprocessing = False
        args.use_multiprocessing_for_evaluation = False

        # base config
        args.reprocess_input_data = True
        args.use_cached_eval_features = False
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
        saved_dirs = ConfigFilePaths.project_dir + '/' + save_dir + '/'  # PharmAI/pharm_ai+ '/po/'
        args.output_dir = saved_dirs + 'outputs/' + model_version + '/'
        args.cache_dir = saved_dirs + 'cache/' + model_version + '/'
        args.best_model_dir = saved_dirs + 'best_model/' + model_version + '/'
        args.tensorboard_dir = saved_dirs + 'runs/' + model_version + '/' + time.strftime("%Y%m%d_%H%M%S",
                                                                                          time.localtime()) + '/'
        return args

    def get_train_model(self):
        if self.args.n_gpu <= 1:
            if self.use_model == 't5' or self.use_model == 'mt5':
                self.args.use_multiprocessed_decoding = False
                return T5Model(model_type=self.model_type, model_name=self.pretrained_model,
                               use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args)
            else:
                return ClassificationModel(model_type=self.model_type, model_name=self.pretrained_model,
                                           use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args,
                                           num_labels=self.num_labels)
        else:
            if self.use_model == 't5' or self.use_model == 'mt5':
                self.args.use_multiprocessed_decoding = False
                return DDPT5Model(model_type=self.model_type, model_name=self.pretrained_model, use_cuda=True,
                                  cuda_device=-1, args=self.args)
            elif self.use_model == 'sentence_pair':
                return ClassificationModel(model_type=self.model_type, model_name=self.pretrained_model,
                                           use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args,
                                           num_labels=self.num_labels)
            else:
                return DDPClassificationModel(model_type=self.model_type, model_name=self.pretrained_model,
                                              use_cuda=True, args=self.args, num_labels=self.num_labels)

    def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, if_send_message=False):
        # deal with dt
        train_df = DTUtils.deal_with_df(train_df, use_model=self.use_model)
        eval_df = DTUtils.deal_with_df(eval_df, use_model=self.use_model)

        # config some parameters
        train_size = train_df.shape[0]
        self.args.update_from_dict({'train_length': train_size})
        all_steps = train_size / self.args.train_batch_size
        self.args.logging_steps = int(max(all_steps / 20 / self.args.gradient_accumulation_steps, 1))
        self.args.evaluate_during_training_steps = int(
            max(all_steps / 6 / self.args.gradient_accumulation_steps, 1))

        self.args.wandb_project = self.wandb_proj
        self.args.wandb_kwargs = {
            'name': self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
            'tags': [self.model_version, 'train']}

        # get model
        model = self.get_train_model()

        # train
        try:
            start_time = time.time()
            logger.info(f'start training: model_version---{self.model_version},train length---{train_size}')
            if self.use_model == 't5' or self.use_model == 'mt5':
                model.train_model(train_data=train_df, eval_data=eval_df)
            else:
                model.train_model(train_df=train_df, eval_df=eval_df)
            logger.info('training finished!!!')
            end_time = time.time()
            logger.info(f'train time: {round(end_time - start_time, 4)} s')
        except Exception as error:
            logger.error(f'train failed!!! ERROR:{error}')
            if if_send_message:
                ModelUtils.send_to_me(f'train failed!!! ERROR:{error}')
        finally:
            wandb.finish()
            ModelUtils.remove_some_model_files(model.args)

    def get_predict_model(self):
        if self.args.n_gpu <= 1:
            if self.use_model == 't5' or self.use_model == 'mt5':
                self.args.use_multiprocessed_decoding = False
                return T5Model(model_type=self.model_type, model_name=self.args.best_model_dir,
                               use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args)
            else:
                return ClassificationModel(model_type=self.model_type, model_name=self.args.best_model_dir,
                                           use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args,
                                           num_labels=self.num_labels)
        else:
            if self.use_model == 't5' or self.use_model == 'mt5':
                self.args.use_multiprocessed_decoding = False
                return DDPT5Model(model_type=self.model_type, model_name=self.args.best_model_dir, use_cuda=True,
                                  cuda_device=-1, args=self.args)
            elif self.use_model == 'sentence_pair':
                return ClassificationModel(model_type=self.model_type, model_name=self.args.best_model_dir,
                                           use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args,
                                           num_labels=self.num_labels)
            else:
                return DDPClassificationModel(model_type=self.model_type, model_name=self.args.best_model_dir,
                                              use_cuda=True, args=self.args, num_labels=self.num_labels)

    @staticmethod
    def eval_decoration(eval_func):
        # #############################################################
        # examples: should set : self.wand_b_pro , self.ver , self.args.hyper_args
        # >>> @eval_decoration
        # >>> def eval(eval_df,a,b):
        # >>>     eval_res = func... a,b
        # >>>     return eval_res
        # ############################################################
        def eval_method(self, eval_df, *args, **kwargs):
            # # deal with eval df
            # eval_df = eval_df[['prefix', 'input_text', 'target_text']]
            # eval_df = eval_df.astype('str')
            eval_length = eval_df.shape[0]

            # wand_b
            wandb.init(project=self.wandb_proj, config=self.args,
                       name=self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
                       tags=[self.model_version, 'eval'])
            try:
                start_time = time.time()
                logger.info(f'start eval: model_version---{self.model_version},eval length---{eval_length}')
                eval_res = eval_func(self, eval_df, *args, **kwargs)
                logger.info('eval finished!!!')
                end_time = time.time()
                need_time = round((end_time - start_time) / eval_length, 5)
                eval_time = round(need_time * eval_length, 4)
                print(f'eval results: {eval_res}')
                logger.info(f'eval time: {need_time} s * {eval_length} = {eval_time} s')
                assert isinstance(eval_res, dict) == True
                eval_res.update({"eval_length": eval_length})
                wandb.log(eval_res)
            except Exception as error:
                logger.error(f'eval failed!!! ERROR:{error}')
                eval_res = None
            finally:
                wandb.finish()
            return eval_res

        return eval_method


if __name__ == '__main__':
    class Porject(MyModel):
        def __init__(self):
            super(Porject, self).__init__()
            self.start_time = '...'
            self.end_time = '...'

            self.wandb_proj = 'test'
            self.use_model = 'classification'  # mt5 /classification
            self.model_type = 'bert'
            self.pretrained_model = ConfigFilePaths.bert_dir_remote

        def run(self):
            self.train_test()

        def train_test(self):
            self.model_version = 'vtest'
            self.pretrained_model = '/home/zyl/disk/PharmAI/pharm_ai/po/best_model/v4.2.0.4/'
            self.args = MyModel.set_model_parameter(model_version=self.model_version,
                                                    args=ClassificationArgs(),
                                                    save_dir='po')
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
            self.cuda_device = 0
            self.args.n_gpu = 3

            self.args.num_train_epochs = 1
            self.args.learning_rate = 5e-5
            self.args.train_batch_size = 64  # 512
            self.args.eval_batch_size = 32  # 256
            self.args.max_seq_length = 512
            self.args.gradient_accumulation_steps = 8  # 256

            train_df = pd.read_excel('./data/processed_0825.xlsx', 'train')
            eval_df = pd.read_excel('./data/processed_0825.xlsx', 'test')
            self.train(train_df=train_df, eval_df=eval_df)


    pass
