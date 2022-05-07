# encoding: utf-8
'''
@author: zyl
@file: sentence_pair.py
@time: 2021/11/12 10:50
@desc:
'''
import json
import math

import pandas as pd

log_file = './model_log_reranking.json'

# class CLSDataset(Dataset):
#     def __init__(self, queries, corpus):
#         self.queries = queries
#         self.queries_ids = list(queries.keys())
#         self.corpus = corpus
#         self.ce_scores = ce_scores
#
#         for qid in self.queries:
#             self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
#             self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
#             random.shuffle(self.queries[qid]['neg'])
#
#     def __getitem__(self, item):
#         query = self.queries[self.queries_ids[item]]
#         query_text = query['query']
#         qid = query['qid']
#
#         if len(query['pos']) > 0:
#             pos_id = query['pos'].pop(0)  # Pop positive and add at end
#             pos_text = self.corpus[pos_id]
#             query['pos'].append(pos_id)
#         else:  # We only have negatives, use two negs
#             pos_id = query['neg'].pop(0)  # Pop negative and add at end
#             pos_text = self.corpus[pos_id]
#             query['neg'].append(pos_id)
#
#         # Get a negative passage
#         neg_id = query['neg'].pop(0)  # Pop negative and add at end
#         neg_text = self.corpus[neg_id]
#         query['neg'].append(neg_id)
#
#         pos_score = self.ce_scores[qid][pos_id]
#         neg_score = self.ce_scores[qid][neg_id]
#
#         return InputExample(texts=[query_text, pos_text, neg_text], label=pos_score - neg_score)
#
#     def __len__(self):
#         return len(self.queries)

import numpy as np
from sklearn.metrics import classification_report, f1_score
import logging
import os
import csv
# from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
# from scipy.stats import pearsonr, spearmanr

from typing import List
from sentence_transformers.readers import InputExample

logger = logging.getLogger(__name__)


class MyEvaluator2:
    """nli自然语义推理，改进版--f1
    This evaluator can be used with the CrossEncoder class.

    It is designed for CrossEncoders with 2 or more outputs. It measure the
    accuracy of the predict class vs. the gold labels.
    """

    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], name: str = '', write_csv: bool = True):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.name = name

        self.csv_file = "CESoftmaxAccuracyEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CESoftmaxAccuracyEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
        print(pred_scores)
        pred_labels = np.argmax(pred_scores, axis=1)

        assert len(pred_labels) == len(self.labels)

        # acc = np.sum(pred_labels == self.labels) / len(self.labels)

        print(classification_report(self.labels, pred_labels))
        print('\n')
        acc = f1_score(self.labels, pred_labels, average='macro')

        logger.info("Accuracy: {:.2f}".format(acc * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc])

        return acc


class RerankerTrainer:
    def __init__(self):
        # self.model_path = "distiluse-base-multilingual-cased-v1"
        self.model_path = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        self.dimensions = 512
        self.cuda_device = 1
        self.max_seqence_length = 512
        self.use_st_model = True
        self.train_batch_size = 16
        self.epoch = 5
        self.learning_rate = 1e-5
        self.all_scores = []
        self.best_score = 0
        # self.label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.label2int = {"contradiction": 0, "entailment": 1}
        # self.train_num_labels = len(set(self.label2int.values()))
        self.train_num_labels = 1
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
            if sub_df['labels'] >= 1:
                for j in range(top_k):
                    train_examples.append(
                        InputExample(texts=[ sub_df['text_b'],sub_df['text_a']], label=1))
            # elif sub_df['labels'] > 1:
            #     for j in range(top_k):
            #         train_examples.append(
            #             InputExample(texts=[sub_df['text_a'], sub_df['text_b']], label=1))
            else:
                train_examples.append(
                    InputExample(texts=[sub_df['text_b'], sub_df['text_a']], label=0))

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

        if evaluator_func == 'EmbeddingSimilarityEvaluator':
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
        elif evaluator_func == 'CEBinaryClassificationEvaluator':
            from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
            from sentence_transformers import InputExample
            # from pharm_ai.panel.entry_match.revise_evaluator import MyEvaluator2
            dev_samples = []
            for _, sub_df in dev_df.iterrows():
                if sub_df['labels'] >= 1:
                    dev_samples.append(
                        InputExample(texts=[sub_df['text_a'], sub_df['text_b']], label=1))
                # elif sub_df['labels'] > 1:
                #     dev_samples.append(
                #         InputExample(texts=[sub_df['text_a'], sub_df['text_b']], label=1))
                else:
                    dev_samples.append(
                        InputExample(texts=[sub_df['text_a'], sub_df['text_b']], label=0))
            evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='Quora-dev')

        else:
            from sentence_transformers import InputExample
            # from pharm_ai.panel.entry_match.revise_evaluator import MyEvaluator2
            dev_samples = []
            for _, sub_df in dev_df.iterrows():
                if sub_df['labels'] >= 1:
                    dev_samples.append(
                        InputExample(texts=[sub_df['text_a'], sub_df['text_b']], label=1))
                # elif sub_df['labels'] > 1:
                #     dev_samples.append(
                #         InputExample(texts=[sub_df['text_a'], sub_df['text_b']], label=1))
                else:
                    dev_samples.append(
                        InputExample(texts=[sub_df['text_a'], sub_df['text_b']], label=0))
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
        # Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if
        # self.config.num_labels == 1 else nn.CrossEntropyLoss()
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
        # train_df = pd.read_json(f'./data/v1/train_1111_up.json.gz', compression='gzip')
        # train_df = train_df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})
        # train_df = pd.read_json(f'./data/v1/train_1111.json.gz', compression='gzip')
        # train_df['labels'] = train_df['labels'].apply(lambda x: 0 if x == 0 else 1)

        # eval_df = pd.read_json(f'./data/v1/eval_1111.json.gz', compression='gzip')
        # eval_df = eval_df.astype({'text_a': 'str', 'text_b': 'str', 'labels': 'int'})

        self.train_file = './data/v1/train_1111.json.gz'
        train_df = pd.read_json(self.train_file, compression='gzip')

        self.dev_file = './data/v1/eval_1111.json.gz'
        dev_df = pd.read_json(self.dev_file, compression='gzip')

        # self.model_path = "sentence-transformers/distiluse-base-multilingual-cased-v1"
        # self.model_path = "./best_model/v1.4.0.0/"
        # self.model_path = "cross-encoder/ms-marco-MiniLM-L-12-v2"

        # self.model_path = "cross-encoder/nli-roberta-base"

        # self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
        # self.model_path = '/large_files/pretrained_pytorch/mt5_zh_en/'

        # self.model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # self.model_path = "./best_model/v2/v2.2.1/"
        self.model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # self.model_path = "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking"

        self.dimensions = 768
        self.cuda_device = RerankerTrainer.get_auto_device()
        self.max_seqence_length = 512
        self.use_st_model = True
        self.train_batch_size = 24
        self.epoch = 3
        self.learning_rate = 2e-4
        self.train(train_df, dev_df, save_model="./best_model/v1.4.0.5/",
                   loss_func='BCEWithLogitsLoss',  # CrossEntropyLoss，BCEWithLogitsLoss，nli
                   evaluator_func="CEBinaryClassificationEvaluator", # MyEvaluator2,CEBinaryClassificationEvaluator
                   collection='t2',
                   top_k=15)

# from pharm_ai.panel.entry_match.train_reranker import RerankerTrainer


# class RerankerPredictor:
#     def __init__(self):
#         # self.model_path = "./best_model/v2/v2.2.1/"
#         self.model_path = "./best_model/v1.4.0.0/"
#         self.cuda_device = '1'
#         self.model_dim = 768
#         self.eval_batch_size = 24
#         self.label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
#         self.int2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
#         self.train_num_labels = len(set(self.label2int.values()))
#         self.set_model()
#
#     def run(self):
#         # to_predicts = [["DMAIC", "分枝杆菌病"],
#         #                ["tooth loss", "牙齿脱落"],
#         #                ["CORD", "没有疾病"],
#         #                ["实体恶性肿瘤", "实体瘤"],
#         #                ["viral, parasitic or bacterial diseases", "细菌感染"]]
#         # self.predict(to_predicts)
#         self.eval()
#         pass
#
#     def set_model(self):
#         from sentence_transformers.cross_encoder import CrossEncoder
#         self.model = CrossEncoder(self.model_path, device=f'cuda:{str(self.cuda_device)}',
#                                   num_labels=self.train_num_labels)
#
#     def predict(self, to_predicts: list):
#         import numpy as np
#
#         pred_scores = self.model.predict(to_predicts, convert_to_numpy=True, show_progress_bar=False)
#         # print(pred_scores)
#         pred_labels = np.argmax(pred_scores, axis=1)
#         # relationships = list(map(lambda x: self.int2label.get(x), pred_labels))
#
#         # for t_p, r in zip(to_predicts, relationships):
#         #     print(f'entity:{t_p[0]} --- {r} --- entry:{t_p[1]}')
#         return pred_labels
#
#     def eval(self):
#         eval_df = pd.read_excel('./data/v1/train_1110.xlsx', 'eval_mt5')
#         input_text = eval_df['input_text'].tolist()
#         from dt import ClassificationListEN
#         from tqdm import tqdm
#         all_therapies = ClassificationListEN[0:-1]
#
#         res = []
#         for text_a in tqdm(input_text):
#             to_predicts = [[text_a, t] for t in all_therapies]
#
#             predictions = self.predict(to_predicts)
#             # print(predictions)
#             # predictions.argmax()
#             r = set()
#             for i, j in zip(all_therapies, predictions):
#                 if j != 0:
#                     r.add(i)
#             res.append(r)
#
#         from zyl_utils.model_utils.ner_utils import NERUtils
#         labels = eval_df['therapy_labels'].tolist()
#         labels = NERUtils.revise_target_texts(labels, input_texts=[], delimiter=',')
#
#         res_df = NERUtils.entity_recognition_v2(y_true=labels, y_pred=res)
#         res_dict = {'sum': res_df}
#
#         res = dict()
#         for k, v in res_dict.items():
#             res[k] = {'pos_score': v.iloc[0, -1], 'neg_score': v.iloc[1, -1], 'sum_score': v.iloc[2, -1],
#                       'weighted_score': v.iloc[3, -1], }
#         return res

if __name__ == '__main__':
    TrainerV1().run()
    # RerankerPredictor().run()



# if __name__ == '__main__':
#     # FineTurn().run()s
#
#     pass
