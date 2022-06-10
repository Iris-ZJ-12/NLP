# encoding: utf-8
'''
@author: zyl
@file: train.py
@time: 2021/7/26 上午12:33
@desc:
'''
import warnings

import pandas as pd

from pharm_ai.util.utils import Utilfuncs

# from sklearn.utils import resample
from simpletransformers.t5 import T5Args
from simpletransformers.classification import ClassificationArgs

Utilfuncs.fix_torch_multiprocessing()
warnings.filterwarnings("ignore")


# class TTTTrainer:
#
#     def __init__(self):
#         self.start_time = '2021-07-28'
#         self.end_time = ''
#         self.wandb_proj = 'ttt'
#         self.model_version = 'test'
#         self.model_type = 't5'
#         self.pretrained_model = 't5-base'
#         self.use_cuda = True
#         self.cuda_device = 0
#         self.method = 'mt5'  # sentence_pair, multi_label_classification
#         self.args = TTTTrainer.set_model_parameter(model_version=self.model_version, method=self.method)
#         self.num_labels = 18
#
#     @staticmethod
#     def set_model_parameter(model_version, method='mt5', proj_name='ttt'):
#         if method == 'mt5':
#             args = T5Args()
#         elif method == 'sentence_pair':
#             args = ClassificationArgs()
#         elif method == 'multi_label_classification':
#             args = MultiLabelClassificationArgs()
#         else:
#             args = T5Args()
#
#         # multiprocess
#         args.use_multiprocessing = False
#         args.use_multiprocessing_for_evaluation = False
#
#         # base config
#         args.reprocess_input_data = True
#         args.fp16 = False
#         args.manual_seed = 234
#         # args.gradient_accumulation_steps = 8  # ==increase batch size,Use time for memory,
#
#         # save
#         args.no_save = False
#         args.save_eval_checkpoints = False
#         args.save_model_every_epoch = False
#         args.save_optimizer_and_scheduler = True
#         args.save_steps = -1
#
#         # eval
#         args.evaluate_during_training = True
#         args.evaluate_during_training_verbose = True
#
#         args.no_cache = False
#         args.use_early_stopping = False
#         args.encoding = None
#         args.do_lower_case = False
#         args.dynamic_quantize = False
#         args.quantized_model = False
#         args.silent = False
#
#         args.overwrite_output_dir = True
#         saved_dirs = ConfigFilePaths.project_dir + '/' + proj_name + '/'
#         args.output_dir = saved_dirs + 'outputs/' + model_version + '/'
#         args.cache_dir = saved_dirs + 'cache/' + model_version + '/'
#         args.best_model_dir = saved_dirs + 'best_model/' + model_version + '/'
#         args.tensorboard_dir = saved_dirs + 'runs/' + model_version + '/' + time.strftime("%Y%m%d_%H%M%S",
#                                                                                           time.localtime()) + '/'
#         return args
#
#     def get_train_model(self, method='mt5'):
#         self.args.use_cached_eval_features = False
#
#         if method == 'sentence_pair':
#             if self.args.n_gpu <= 1:
#                 return ClassificationModel(model_type=self.model_type, model_name=self.pretrained_model,
#                                            use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args)
#             else:
#                 return DDPClassificationModel(model_type=self.model_type, model_name=self.pretrained_model,
#                                               use_cuda=True,
#                                               cuda_device=-1, args=self.args)
#         elif method == 'multi_label_classification':
#             return MultiLabelClassificationModel(model_type=self.model_type, model_name=self.pretrained_model,
#                                                  num_labels=self.num_labels,
#                                                  use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args,
#                                                  )
#         else:
#             self.args.use_multiprocessed_decoding = False
#             if self.args.n_gpu <= 1:
#                 return T5Model(model_type=self.model_type, model_name=self.pretrained_model,
#                                use_cuda=self.use_cuda, cuda_device=self.cuda_device, args=self.args)
#             else:
#                 return DDPT5Model(model_type=self.model_type, model_name=self.pretrained_model, use_cuda=True,
#                                   cuda_device=-1, args=self.args)
#
#     def train(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, method='mt5'):
#         if method == 'sentence_pair':
#             train_df = train_df[['text_a', 'text_b', 'labels']]
#             eval_df = eval_df[['text_a', 'text_b', 'labels']]
#             train_df = train_df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})
#             eval_df = eval_df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})
#             train_df['labels'] = train_df['labels'].apply(lambda x: 0 if x==0 else 1)
#             eval_df['labels'] = eval_df['labels'].apply(lambda x: 0 if x == 0 else 1)
#
#         elif method == 'multi_label_classification':
#             train_df = train_df[['text', 'labels']]
#             eval_df = eval_df[['text', 'labels']]
#             train_df['labels'] = train_df['labels'].apply(lambda x: eval(x))
#             eval_df['labels'] = eval_df['labels'].apply(lambda x: eval(x))
#         else:
#             # deal with dt
#             train_df = train_df[['prefix', 'input_text', 'target_text']]
#             eval_df = eval_df[['prefix', 'input_text', 'target_text']]
#             train_df = train_df.astype('str')
#             eval_df = eval_df.astype('str')
#
#         # config some parameters
#         train_size = train_df.shape[0]
#         all_steps = train_size / self.args.train_batch_size
#         self.args.logging_steps = int(all_steps / 20)
#         self.args.evaluate_during_training_steps = int(all_steps / 6)
#         self.args.wandb_kwargs = {
#             'name': self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
#             'tags': [self.model_version, 'train']}
#         self.args.wandb_project = self.wandb_proj
#
#         # get model
#         model = self.get_train_model(method=method)
#         print(f'train length: {train_size}')
#         model.args.update_from_dict({"train_length": train_df.shape[0]})
#         # train
#         try:
#             start_time = time.time()
#             if method == 'mt5':
#                 model.train_model(train_data=train_df, eval_data=eval_df)
#             else:
#                 model.train_model(train_df=train_df, eval_df=eval_df)
#             logger.info('train finished!!!')
#             end_time = time.time()
#             need_time = round((end_time - start_time) / train_size, 4)
#             training_time = round(need_time * train_size, 4)
#             model.args.update_from_dict({"train_time": training_time})
#             print(f'train time: {need_time} s * {train_size} = {training_time} s')
#         except Exception as error:
#             logger.error(f'train failed!!! ERROR:{error}')
#             # DTUtils.send_to_me(f'train failed!!! ERROR:{error}')
#         finally:
#             wandb.finish()
#             if os.path.isdir(self.args.output_dir):
#                 cmd = 'rm -rf ' + self.args.output_dir.split('outputs')[0] + 'outputs/'
#                 os.system(cmd)
#             if os.path.isdir(self.args.output_dir.split('outputs')[0] + '__pycache__/'):
#                 cmd = 'rm -rf ' + self.args.output_dir.split('outputs')[0] + '__pycache__/'
#                 os.system(cmd)
#

# class TrainerV1(TTTTrainer):
#     def __init__(self):
#         super(TrainerV1, self).__init__()
#         self.start_time = '2021-07-28'
#         self.end_time = ''
#
#     def run(self):
#         # self.train_0728()
#         self.train_0805()
#         # self.train_0806()
#         # self.train_0809()
#         # self.train_1110()
#
#     def train_0728(self):
#         self.model_version = 'v1.0.0.4'
#         self.args = TrainerV1.set_model_parameter(model_version=self.model_version)
#         os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
#         self.cuda_device = 0
#
#         self.args.n_gpu = 4
#         self.args.num_train_epochs = 10
#         self.args.learning_rate = 1e-4
#         self.args.train_batch_size = 128
#         self.args.eval_batch_size = 128
#         # train_df = pd.read_excel('./data/dt_0728.xlsx', 'train')
#         train_df = pd.read_excel('./data/dt_0728.xlsx', 'train_up')
#         train_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
#         train_df['prefix'] = 'classification'
#         eval_df = pd.read_excel('./data/dt_0728.xlsx', 'eval')
#         eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
#         eval_df['prefix'] = 'classification'
#         self.train(train_df, eval_df)
#
#     def train_0805(self):
#         self.method = 'sentence_pair'
#         self.model_type = 'roberta'
#         self.pretrained_model = "roberta-base"
#         # self.pretrained_model = ConfigFilePaths.bert_dir_remote
#         self.model_version = 'v1.1.0.4'
#         self.args = TrainerV1.set_model_parameter(model_version=self.model_version, method=self.method)
#         os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#         self.cuda_device = 0
#
#         self.args.n_gpu = 1
#         self.args.num_train_epochs = 5
#         self.args.learning_rate = 2e-4
#         self.args.train_batch_size = 16
#         self.args.eval_batch_size = 8
#         self.args.max_seq_length = 512
#         # train_df = pd.read_excel('./data/dt_0806.xlsx', 'train_up')
#         # eval_df = pd.read_excel('./data/dt_0806.xlsx', 'eval')
#
#         train_df = pd.read_json(f'./data/v1/train_1111.json.gz', compression='gzip')
#         eval_df = pd.read_json(f'./data/v1/eval_1111.json.gz', compression='gzip')
#
#         self.train(train_df, eval_df, method=self.method)
#
#     def train_0806(self):
#         self.method = 'multi_label_classification'
#         self.model_type = 'xlnet'
#         self.num_labels = 18
#         self.pretrained_model = 'xlnet-base-cased'
#         self.model_version = 'v1.2.0.6'
#         self.args = TrainerV1.set_model_parameter(model_version=self.model_version, method=self.method)
#         os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
#         self.cuda_device = 1
#
#         self.args.n_gpu = 1
#         self.args.num_train_epochs = 5
#         self.args.learning_rate = 1e-3
#         self.args.train_batch_size = 16
#         self.args.eval_batch_size = 32
#         self.args.reprocess_input_data = False
#         self.args.use_cached_eval_features = True
#
#         self.args.gradient_accumulation_steps = 2
#         self.args.max_seq_length = 512
#
#         # train_df = pd.read_excel('./data/dt_0807.xlsx', 'train')
#         # eval_df = pd.read_excel('./data/dt_0807.xlsx', 'eval')
#
#         train_df = pd.read_excel('./data/dt_0809.xlsx', 'train')
#         eval_df = pd.read_excel('./data/dt_0809.xlsx', 'eval')
#         self.train(train_df, eval_df, method=self.method)
#
#     def train_0809(self):
#         self.model_version = 'v1.0.1.0'
#         self.args = TrainerV1.set_model_parameter(model_version=self.model_version, method=self.method)
#         os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
#         self.cuda_device = 1
#
#         self.args.n_gpu = 4
#         self.args.max_seq_length = 512
#
#         self.args.num_train_epochs = 5
#         self.args.learning_rate = 1e-4
#         self.args.train_batch_size = 16
#         self.args.gradient_accumulation_steps = 4
#         self.args.eval_batch_size = 8
#         self.args.reprocess_input_data = True
#         self.args.use_cached_eval_features = False
#
#         train_df = pd.read_excel('./data/dt_0809_2.xlsx', 'train')
#         train_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
#         train_df['prefix'] = 'classification'
#
#         eval_df = pd.read_excel('./data/dt_0809_2.xlsx', 'eval')
#         eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
#         eval_df['prefix'] = 'classification'
#
#         self.train(train_df, eval_df, method=self.method)
#
#     def train_1110(self):
#         self.model_version = 'v1.3.0.0'
#         self.args = TrainerV1.set_model_parameter(model_version=self.model_version, method=self.method)
#         os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
#         self.cuda_device = 1
#
#         self.args.n_gpu = 3
#         self.args.max_seq_length = 512
#
#         self.args.num_train_epochs = 5
#         self.args.learning_rate = 2e-4
#         self.args.train_batch_size = 16
#         self.args.gradient_accumulation_steps = 8
#         self.args.eval_batch_size = 8
#         self.args.reprocess_input_data = True
#         self.args.use_cached_eval_features = False
#
#         train_df = pd.read_excel('./data/v1/train_1110.xlsx', 'train_mt5')
#
#         train_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
#         train_df['prefix'] = 'classification'
#
#         eval_df = pd.read_excel('./data/v1/train_1110.xlsx', 'eval_mt5')
#
#         eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
#         eval_df['prefix'] = 'classification'
#
#         self.train(train_df, eval_df, method=self.method)



from zyl_utils.model_utils.models.my_model import MyModel


class TTTTrainer(MyModel):
    def __init__(self):
        super(TTTTrainer, self).__init__()
        self.wandb_proj = 'ttt'
        self.model_version = 'v1.0.0.0'  # to save model or best model

        self.use_model = 'sentence_pair'  # mt5 /classification/sentence_pair
        self.model_type = 'roberta'
        self.pretrained_model = 'roberta-base'  # 预训练模型位置
        self.use_cuda = True
        self.args = MyModel.set_model_parameter(model_version=self.model_version,
                                                args=self._set_args(),
                                                save_dir="/home/zyl/disk/PharmAI/pharm_ai/ttt/")
    def run(self):
        self.train_1111()
        # self.train_1115()

    def train_1111(self):
        self.model_version = 'v1.1.1.4'
        self.use_model = 'sentence_pair'
        self.num_labels = 2
        self.model_type = "mpnet"
        self.pretrained_model = 'MPNet'

        # self.pretrained_model = 'roberta-base'
        # self.pretrained_model = '/home/zyl/disk/PharmAI/pharm_ai/ttt/best_model/v1.1.1.2/'
        import os
        if os.path.isfile(self.pretrained_model + 'scheduler.pt'):
            cmd = 'rm ' + self.pretrained_model + 'scheduler.pt'
            os.system(cmd)

        self.args = MyModel.set_model_parameter(model_version=self.model_version,
                                                args=ClassificationArgs(),
                                                save_dir="/home/zyl/disk/PharmAI/pharm_ai/ttt/")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        self.args.n_gpu = 1
        self.cuda_device = 0
        self.use_cuda = True

        self.args.num_train_epochs = 3
        self.args.learning_rate = 2e-5
        self.args.train_batch_size = 16  # 512
        self.args.eval_batch_size = 16  # 256
        self.args.max_seq_length = 512
        self.args.gradient_accumulation_steps = 16  # 256
        self.args.no_cache = False
        # self.args.lazy_loading = True

        train_df = pd.read_json(f'./data/v1/train_1111_up.json.gz', compression='gzip')
        train_df = train_df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})
        train_df.rename(columns={'text_a':'text_b','text_b':'text_a'},inplace=True)
        # train_df = pd.read_json(f'./data/v1/train_1111.json.gz', compression='gzip')
        # train_df['labels'] = train_df['labels'].apply(lambda x: 0 if x == 0 else 1)

        eval_df = pd.read_json(f'./data/v1/eval_1111.json.gz', compression='gzip')
        eval_df = eval_df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})
        eval_df['labels'] = eval_df['labels'].apply(lambda x: 0 if x == 0 else 1)
        eval_df.rename(columns={'text_a': 'text_b', 'text_b': 'text_a'}, inplace=True)

        self.train(train_df, eval_df)

    def train_1115(self):
        self.model_version = 'v1.3.0.2'
        self.use_model = 't5'
        self.model_type = 't5'
        # self.pretrained_model = 'roberta-base'
        # self.pretrained_model = 't5-base'
        self.pretrained_model = '/home/zyl/disk/PharmAI/pharm_ai/ttt/best_model/v1.3.0.1/'
        import os
        if os.path.isfile(self.pretrained_model+'scheduler.pt'):
            cmd = 'rm ' + self.pretrained_model+'scheduler.pt'
            os.system(cmd)
        self.args = MyModel.set_model_parameter(model_version=self.model_version,
                                                args=T5Args(),
                                                save_dir="/home/zyl/disk/PharmAI/pharm_ai/ttt/")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        self.args.n_gpu = 3
        self.cuda_device = 0
        self.use_cuda = True

        self.args.num_train_epochs = 3
        self.args.learning_rate = 8e-5
        self.args.train_batch_size = 16
        self.args.eval_batch_size = 8
        self.args.max_seq_length = 512
        self.args.gradient_accumulation_steps = 16  # 256
        # self.args.lazy_loading = True

        train_df = pd.read_excel('./data/v1/train_1110.xlsx', 'train_mt5')

        train_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
        train_df['input_text'] = train_df['input_text'].apply(lambda x:str(x).replace('\n',''))
        train_df['prefix'] = 'classification'

        eval_df = pd.read_excel('./data/v1/train_1110.xlsx', 'eval_mt5')

        eval_df.rename(columns={"therapy_labels_num": "target_text"}, inplace=True)
        eval_df['input_text'] = eval_df['input_text'].apply(lambda x:str(x).replace('\n',''))
        eval_df['prefix'] = 'classification'

        self.train(train_df, eval_df)

if __name__ == '__main__':
    # TrainerV1().run()
    TTTTrainer().run()
