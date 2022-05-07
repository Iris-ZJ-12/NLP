# encoding: utf-8
'''
@author: zyl
@file: train_reranker.py
@time: 2021/9/30 9:37
@desc:
'''
import json
import math

import pandas as pd

log_file = './model_log_reranking.json'

from torch.utils.data import Dataset
import random
from sentence_transformers import InputExample


class CLSDataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']
        qid = query['qid']

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)  # Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)
        else:  # We only have negatives, use two negs
            pos_id = query['neg'].pop(0)  # Pop negative and add at end
            pos_text = self.corpus[pos_id]
            query['neg'].append(pos_id)

        # Get a negative passage
        neg_id = query['neg'].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        pos_score = self.ce_scores[qid][pos_id]
        neg_score = self.ce_scores[qid][neg_id]

        return InputExample(texts=[query_text, pos_text, neg_text], label=pos_score - neg_score)

    def __len__(self):
        return len(self.queries)


class RerankerTrainer:
    def __init__(self):
        self.model_path = "distiluse-base-multilingual-cased-v1"
        self.dimensions = 512
        self.cuda_device = 1
        self.max_seqence_length = 128
        self.use_st_model = True
        self.train_batch_size = 16
        self.epoch = 5
        self.learning_rate = 1e-5
        self.all_scores = []
        self.best_score = 0
        self.label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.train_num_labels = len(set(self.label2int.values()))
        pass

    @staticmethod
    def get_auto_device():
        import pynvml
        import numpy as np
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        deviceMemory = []
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            deviceMemory.append(mem_info.free / 1024 / 1024)  # M

        deviceMemory = np.array(deviceMemory, dtype=np.int64)
        best_device_index = np.argmax(deviceMemory)

        if deviceMemory[best_device_index] > 20000:
            print(f'use cuda_device:{best_device_index}')
            return best_device_index
        else:
            print(f'warning! no cuda_device:{best_device_index}')
            return -1

    def get_train_objectives(self, train_df, model, loss_func=None, top_k=30):
        from sentence_transformers import InputExample, SentencesDataset
        from torch.utils.data import DataLoader
        from sklearn.utils import resample
        import torch
        train_df = resample(train_df, replace=False)
        train_examples = []
        self.loss_func = loss_func

        for _, sub_df in train_df.iterrows():
            if sub_df['label'] == 1:
                for j in range(top_k):
                    train_examples.append(
                        InputExample(texts=[sub_df['entity'], sub_df['entry']], label=self.label2int['neutral']))
            elif sub_df['label'] > 0:
                for j in range(top_k):
                    train_examples.append(
                        InputExample(texts=[sub_df['entity'], sub_df['entry']], label=self.label2int['entailment']))
            else:
                train_examples.append(
                    InputExample(texts=[sub_df['entity'], sub_df['entry']], label=self.label2int['contradiction']))

        print(f'train_length:{len(train_examples)}')
        self.train_length = len(train_examples)
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.train_batch_size)

        if loss_func == 'CrossEntropyLoss':  # 多分类
            train_loss = torch.nn.CrossEntropyLoss()
            self.loss_func = 'CrossEntropyLoss'
        elif loss_func == 'BCEWithLogitsLoss':  # 二分类
            train_loss = torch.nn.BCEWithLogitsLoss()
            self.loss_func = 'BCEWithLogitsLoss'
        else:
            train_loss = None
            # For binary tasks and tasks with continious scores (like STS), we set num_labels=1.
            # For classification tasks, we set it to the number of labels we have.
            # nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        return train_dataloader, train_loss

    def get_model(self):
        from sentence_transformers import models
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.cross_encoder import CrossEncoder
        if self.use_st_model:
            model = CrossEncoder(self.model_path, device=f'cuda:{str(self.cuda_device)}',
                                 num_labels=self.train_num_labels)
        else:
            from torch import nn
            word_embedding_model = models.Transformer(self.model_path, max_seq_length=self.max_seqence_length)
            # word_embedding_model = T5(self.model_path,max_seq_length=self.max_seqence_length)
            # dense_model = models.Dense(in_features=word_embedding_model.get_word_embedding_dimension(),
            #                            out_features=word_embedding_model.get_word_embedding_dimension(),
            #                            activation_function=nn.Tanh())
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False,
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_mean_sqrt_len_tokens=False, )

            model = SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                        device=f'cuda:{str(self.cuda_device)}')
        # self.dimensions = model.get_sentence_embedding_dimension()
        self.max_seqence_length = model.max_length
        self.tokenizer = model.tokenizer
        print(f'use_pred_model: {self.model_path}')
        return model

    def get_evaluator(self, dev_df, evaluator_func='MyEvaluator2', collection='t1'):
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
        from sklearn.utils import resample

        self.evaluator_func = evaluator_func
        dev_df = resample(dev_df, replace=False)

        if evaluator_func == 'MyEvaluator':
            from pharm_ai.panel.entry_match.revise_evaluator import MyEvaluator
            from sentence_transformers import InputExample
            dev_df = dev_df[dev_df['label'] != 0.0]  # type:pd.DataFrame
            dev_df = dev_df.groupby('entity').apply(lambda x: x['entry'].tolist())
            scores = dev_df.index.tolist()
            eval_examples = []
            dev_samples = []
            for t, r in zip(dev_df.index.tolist(), dev_df.tolist()):
                eval_examples.append(InputExample(texts=[t, r]))
            evaluator = MyEvaluator.from_input_examples(eval_examples, name='sts-eval', collection=collection)

        elif evaluator_func == 'EmbeddingSimilarityEvaluator':
            sentences_1 = []
            sentences_2 = []
            scores = []
            dev_samples = []
            for _, sub_df in dev_df.iterrows():
                if sub_df['label'] != 0.0:
                    sentences_1.append(sub_df['entity'])
                    sentences_2.append(sub_df['entry'])
                    scores.append(sub_df['label'])

            evaluator = EmbeddingSimilarityEvaluator(sentences_1, sentences_2, scores)
        else:
            from sentence_transformers import InputExample
            from pharm_ai.panel.entry_match.revise_evaluator import MyEvaluator2
            dev_samples = []
            for _, sub_df in dev_df.iterrows():
                if sub_df['label'] == 1:
                    dev_samples.append(
                        InputExample(texts=[sub_df['entity'], sub_df['entry']], label=self.label2int['neutral']))
                elif sub_df['label'] > 0:
                    dev_samples.append(
                        InputExample(texts=[sub_df['entity'], sub_df['entry']], label=self.label2int['entailment']))
                else:
                    dev_samples.append(
                        InputExample(texts=[sub_df['entity'], sub_df['entry']], label=self.label2int['contradiction']))
            evaluator = MyEvaluator2.from_input_examples(dev_samples, name='AllNLI-dev')

        print(f'dev_length:{len(dev_samples)}')
        self.dev_length = len(dev_samples)
        return evaluator

    @staticmethod
    def save_parameters(para_obj, save_model='./test.json'):
        """
        存储一个对象的参数，对象参数可以是模型参数或超参数
        Args:
            para_obj: 要存储的参数的对象
            save_model: 保存路径

        Returns:

        """
        para_list = para_obj.__dir__()
        # save_para_list = ['best_score','device','max_seq_length','tokenizer']
        para = {}
        for p in para_list:
            if not p.startswith('_'):
                # if p in save_para_list:
                r = getattr(para_obj, p)
                if isinstance(r, int) or isinstance(r, str) or isinstance(r, float) or isinstance(r, list) \
                        or isinstance(r, bool):
                    para[p] = r

        with open(save_model, "w", encoding='utf-8') as f:
            # indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
            # f.write(json.dumps(para,indent=4))
            json.dump(para, f, indent=4)  # 传入文件描述符，和dumps一样的结果

        para.pop("all_scores")
        with open(log_file, "a", encoding='utf-8') as f:
            json.dump(para, f, indent=4)
            f.write('\n')

    def train(self, train_df, dev_df, save_model="./best_model/test/", loss_func='SoftmaxLoss',
              evaluator_func='MyEvaluator2', collection='t1', top_k=30):
        self.save_model = save_model
        model = self.get_model()

        train_dataloader, train_loss = self.get_train_objectives(train_df, model, loss_func=loss_func, top_k=top_k)

        evaluator = self.get_evaluator(dev_df, evaluator_func=evaluator_func, collection=collection)

        warmup_steps = math.ceil(len(train_dataloader) * self.epoch * 0.1)  # 10% of train data for warm-up
        evaluation_steps = math.ceil(len(train_dataloader) * 0.1)

        print('start train...')
        # Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        model.fit(train_dataloader=train_dataloader, epochs=self.epoch, warmup_steps=warmup_steps,
                  evaluator=evaluator, save_best_model=True,
                  output_path=save_model,
                  evaluation_steps=evaluation_steps,
                  callback=self.call_back,
                  loss_fct=train_loss,
                  optimizer_params={'lr': self.learning_rate})

        df = pd.DataFrame(self.all_scores)
        df.to_excel(save_model + 'my_score.xlsx')
        RerankerTrainer.save_parameters(self, save_model=f'{save_model}parameters.json')

    def call_back(self, score, epoch, steps):
        self.all_scores.append({str(epoch) + '-' + str(steps): score})
        if score > self.best_score:
            self.best_score = score
        print(f'epoch:{epoch}: score:{score} ')


class TrainerV1(RerankerTrainer):
    def __init__(self):
        super(TrainerV1, self).__init__()

    def run(self):
        self.train_1011()

    def train_1011(self):
        self.train_file = "./data/v2/train_2.csv.gz"
        train_df = pd.read_csv(self.train_file, compression='gzip', sep='|')
        self.dev_file = "./data/v2/eval.csv.gz"
        dev_df = pd.read_csv(self.dev_file, compression='gzip', sep='|')

        # self.model_path = "sentence-transformers/distiluse-base-multilingual-cased-v1"
        self.model_path = "./best_model/v2/v2.2.0/"

        # self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
        # self.model_path = '/large_files/pretrained_pytorch/mt5_zh_en/'

        # self.model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # self.model_path = "./best_model/v2/v2.2.1/"

        # self.model_path = "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking"

        self.dimensions = 768
        self.cuda_device = RerankerTrainer.get_auto_device()
        self.max_seqence_length = 128
        self.use_st_model = True
        self.train_batch_size = 32
        self.epoch = 3
        self.learning_rate = 2e-5
        self.train(train_df, dev_df, save_model="./best_model/v2/v2.2.2/",
                   loss_func='CrossEntropyLoss',  # CrossEntropyLoss，BCEWithLogitsLoss，nli
                   evaluator_func="MyEvaluator2",
                   collection='t2',
                   top_k=30)

    def train_cross_model(self):
        self.train_file = "./data/v2/train_2.csv.gz"
        train_df = pd.read_csv(self.train_file, compression='gzip', sep='|')
        self.dev_file = "./data/v2/eval.csv.gz"
        dev_df = pd.read_csv(self.dev_file, compression='gzip', sep='|')

        # self.model_path = "sentence-transformers/distiluse-base-multilingual-cased-v1"
        self.model_path = "./best_model/v2/v2.2.0/"

        # self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
        # self.model_path = '/large_files/pretrained_pytorch/mt5_zh_en/'

        # self.model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # self.model_path = "./best_model/v2/v2.2.1/"

        # self.model_path = "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking"

        self.dimensions = 768
        self.cuda_device = RerankerTrainer.get_auto_device()
        self.max_seqence_length = 128
        self.use_st_model = True
        self.train_batch_size = 32
        self.epoch = 3
        self.learning_rate = 2e-5
        self.train(train_df, dev_df, save_model="./best_model/v2/v2.2.2/",
                   loss_func='CrossEntropyLoss',  # CrossEntropyLoss，BCEWithLogitsLoss，nli
                   evaluator_func="MyEvaluator2",
                   collection='t2',
                   top_k=30)


if __name__ == '__main__':
    # FineTurn().run()s
    TrainerV1().run()
    pass
