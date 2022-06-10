# encoding: utf-8
'''
@author: zyl
@file: train_retrieval.py
@time: 2021/9/30 9:37
@desc:
'''
import json
import math

import pandas as pd

log_file = './model_log_retrieval.json'
from collections import defaultdict


class TrainRetrieval:
    def __init__(self):
        self.model_path = "distiluse-base-multilingual-cased-v1"
        self.output_dimension = 768
        self.cuda_device = 1
        self.max_seqence_length = 128
        self.use_st_model = True
        self.train_batch_size = 16
        self.epoch = 5
        self.learning_rate = 1e-5
        self.all_scores = []
        self.best_score = 0
        self.data_top_k = 30
        self.corpus = self.get_corpus()
        pass

    @staticmethod
    def triplets_from_labeled_dataset(input_examples):
        import random
        from sentence_transformers.readers import InputExample
        # Create triplets for a [(label, sentence), (label, sentence)...] dataset
        # by using each example as an anchor and selecting randomly a
        # positive instance with the same label and a negative instance with a different label
        triplets = []
        label2sentence = defaultdict(list)
        for inp_example in input_examples:
            label2sentence[inp_example.label].append(inp_example)

        for inp_example in input_examples:
            anchor = inp_example

            if len(label2sentence[inp_example.label]) < 2:  # We need at least 2 examples per label to create a triplet
                continue

            positive = None
            while positive is None or positive.guid == anchor.guid:
                positive = random.choice(label2sentence[inp_example.label])

            negative = None
            while negative is None or negative.label == anchor.label:
                negative = random.choice(input_examples)

            triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

        return triplets

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

    def get_train_objectives(self, train_df, model, loss_func='CosineSimilarityLoss'):
        from sentence_transformers import InputExample, SentencesDataset, losses
        from torch.utils.data import DataLoader
        from sklearn.utils import resample
        train_df = resample(train_df, replace=False)
        train_examples = []
        self.loss_func = loss_func
        if loss_func == 'MultipleNegativesRankingLoss':
            for _, sub_df in train_df.iterrows():
                if sub_df['label'] != 0:
                    train_examples.append(InputExample(texts=[sub_df['entity'], sub_df['entry']], label=1))
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
        elif loss_func == 'OnlineContrastiveLoss':
            train_df = train_df[train_df['label'] != 0.0]  # type:pd.DataFrame

            dev_df = train_df.groupby('entity').apply(lambda x: x['entry'].tolist())

            scores = dev_df.index.tolist()
            eval_examples = []
            for t, r in zip(dev_df.index.tolist(), dev_df.tolist()):
                eval_examples.append(InputExample(texts=[t, r]))

            for _, sub_df in train_df.iterrows():
                if sub_df['label'] > 0:
                    label = 1
                    train_examples.append(InputExample(texts=[sub_df['entity'], sub_df['entry']], label=label))
                    train_examples.append(InputExample(texts=[sub_df['entry'], sub_df['entity']], label=label))
                else:
                    label = 0
                    train_examples.append(InputExample(texts=[sub_df['entity'], sub_df['entry']], label=label))

            train_loss = losses.OnlineContrastiveLoss(model=model)
        elif loss_func == 'multi-task':
            train_samples_MultipleNegativesRankingLoss = []
            train_samples_ConstrativeLoss = []

            for _, sub_df in train_df.iterrows():
                if sub_df['label'] > 0:
                    label = 1
                else:
                    label = 0
                train_samples_ConstrativeLoss.append(
                    InputExample(texts=[sub_df['entity'], sub_df['entry']], label=label))
                if str(label) == '1':
                    for _ in range(int(self.data_top_k / 2)):
                        train_samples_MultipleNegativesRankingLoss.append(
                            InputExample(texts=[sub_df['entity'], sub_df['entry']], label=1))
                        train_samples_MultipleNegativesRankingLoss.append(
                            InputExample(texts=[sub_df['entry'], sub_df['entity']], label=1))

            # Create data loader and loss for MultipleNegativesRankingLoss
            train_dataset_MultipleNegativesRankingLoss = SentencesDataset(train_samples_MultipleNegativesRankingLoss,
                                                                          model=model)
            train_dataloader_MultipleNegativesRankingLoss = DataLoader(train_dataset_MultipleNegativesRankingLoss,
                                                                       shuffle=True, batch_size=self.train_batch_size)
            train_loss_MultipleNegativesRankingLoss = losses.MultipleNegativesRankingLoss(model)

            # Create data loader and loss for OnlineContrastiveLoss
            train_dataset_ConstrativeLoss = SentencesDataset(train_samples_ConstrativeLoss, model=model)
            train_dataloader_ConstrativeLoss = DataLoader(train_dataset_ConstrativeLoss, shuffle=True,
                                                          batch_size=self.train_batch_size)

            # As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
            distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
            # Negative pairs should have a distance of at least 0.5
            margin = 0.5
            train_loss_ConstrativeLoss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric,
                                                                      margin=margin)
            train_object = [(train_dataloader_MultipleNegativesRankingLoss, train_loss_MultipleNegativesRankingLoss),
                            (train_dataloader_ConstrativeLoss, train_loss_ConstrativeLoss)]

            return train_object

        elif loss_func == 'BatchHardSoftMarginTripletLoss':
            ### There are 4 triplet loss variants:
            ### - BatchHardTripletLoss
            ### - BatchHardSoftMarginTripletLoss
            ### - BatchSemiHardTripletLoss
            ### - BatchAllTripletLoss

            from sentence_transformers.datasets.SentenceLabelDataset import SentenceLabelDataset

            guid = 1
            self.label_map_file = "./data/v2/label_dict.xlsx"
            label_map = pd.read_excel(self.label_map_file)
            label_map = dict(zip(label_map['entry'].tolist(), label_map['label_num'].tolist()))
            train_examples = []
            for _, sub_df in train_df.iterrows():
                if sub_df['label'] != 0:
                    train_examples.append(InputExample(guid=str(guid), texts=[sub_df['entity']],
                                                       label=label_map.get(sub_df['entry'])))
                    guid += 1

            print(f'train_length:{len(train_examples)}')
            self.train_length = len(train_examples)

            train_dataset = SentenceLabelDataset(train_examples)
            train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, drop_last=True)
            train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
            return train_dataloader, train_loss
        else:
            for _, sub_df in train_df.iterrows():
                train_examples.append(InputExample(texts=[sub_df['entity'], sub_df['entry']], label=sub_df['label']))
            train_loss = losses.CosineSimilarityLoss(model=model)

        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.train_batch_size)
        train_obj = [(train_dataloader, train_loss)]
        return train_obj

    def get_model(self):
        from sentence_transformers import models

        from sentence_transformers import SentenceTransformer
        if self.use_st_model:
            model = SentenceTransformer(self.model_path,
                                        device=f'cuda:{str(self.cuda_device)}')
        else:
            from torch import nn
            word_embedding_model = models.Transformer(self.model_path, max_seq_length=self.max_seqence_length)
            # from sentence_transformers.models.T5 import T5
            # word_embedding_model = T5(self.model_path,max_seq_length=self.max_seqence_length)
            # dense_model = models.Dense(in_features=word_embedding_model.get_word_embedding_dimension(),
            #                            out_features=word_embedding_model.get_word_embedding_dimension(),
            #                            activation_function=nn.Tanh())
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False,
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_mean_sqrt_len_tokens=False,
                                           )
            dense_layer = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                       out_features=self.output_dimension, activation_function=nn.Tanh())
            normalize_layer = models.Normalize()
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_layer, normalize_layer],
                                        device=f'cuda:{str(self.cuda_device)}')
        self.output_dimension = model.get_sentence_embedding_dimension()
        print(f'output_dimension:{self.output_dimension}')
        self.max_seqence_length = model.max_seq_length
        print(f'max_seqence_length:{self.max_seqence_length}')
        self.tokenizer = model.tokenizer
        print(f'use_pred_model: {self.model_path}')
        return model

    def get_evaluator(self, dev_df, evaluator_func='EmbeddingSimilarityEvaluator', collection='t1',
                      top_k=100, encode_batch_size=128):
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
            for t, r in zip(dev_df.index.tolist(), dev_df.tolist()):
                eval_examples.append(InputExample(texts=[t, r]))
            evaluator = MyEvaluator.from_input_examples(eval_examples, name='sts-eval', collection=collection,
                                                        top_k=top_k, encode_batch_size=encode_batch_size)
        elif evaluator_func == 'seq_evaluator':
            from sentence_transformers import evaluation
            from sentence_transformers import InputExample
            from pharm_ai.panel.entry_match.revise_evaluator import MyEvaluator
            evaluators = []

            sentences_1 = []
            sentences_2 = []
            scores_ = []
            for _, sub_df in dev_df.iterrows():

                sentences_1.append(sub_df['entity'])
                sentences_2.append(sub_df['entry'])
                if sub_df['label'] > 0:
                    scores_.append(1)
                else:
                    scores_.append(0)

            binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(sentences_1, sentences_2, scores_)
            evaluators.append(binary_acc_evaluator)

            dev_df = dev_df[dev_df['label'] != 0.0]  # type:pd.DataFrame
            dev_df = dev_df.groupby('entity').apply(lambda x: x['entry'].tolist())
            # scores = dev_df.index.tolist()
            eval_examples = []
            for t, r in zip(dev_df.index.tolist(), dev_df.tolist()):
                eval_examples.append(InputExample(texts=[t, r]))
            my_evaluator = MyEvaluator.from_input_examples(eval_examples, name='sts-eval', collection=collection,
                                                           top_k=top_k, encode_batch_size=encode_batch_size)

            evaluators.append(my_evaluator)
            seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
            return seq_evaluator

        elif evaluator_func == 'EmbeddingSimilarityEvaluator':
            sentences_1 = []
            sentences_2 = []
            scores = []
            for _, sub_df in dev_df.iterrows():
                if sub_df['label'] != 0.0:
                    sentences_1.append(sub_df['entity'])
                    sentences_2.append(sub_df['entry'])
                    scores.append(sub_df['label'])

            evaluator = EmbeddingSimilarityEvaluator(sentences_1, sentences_2, scores)
        else:
            sentences_1 = []
            sentences_2 = []
            scores = []
            for _, sub_df in dev_df.iterrows():
                if sub_df['label'] != 0.0:
                    sentences_1.append(sub_df['entity'])
                    sentences_2.append(sub_df['entry'])
                    scores.append(sub_df['label'])
            evaluator = EmbeddingSimilarityEvaluator(sentences_1, sentences_2, scores)
        print(f'dev_length:{len(scores)}')
        self.dev_length = len(scores)
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

    def train(self, train_df, dev_df, save_model="./best_model/test/", loss_func='CosineSimilarityLoss',
              evaluator_func='EmbeddingSimilarityEvaluator', collection='t1', top_k=100, encode_batch_size=128):
        self.save_model = save_model

        model = self.get_model()

        train_obj = self.get_train_objectives(train_df, model, loss_func=loss_func)

        self.train_length = 999999999
        for t in train_obj:
            self.train_length = min(len(t[0]), self.train_length)

        print(f'train_length:{self.train_length}')

        evaluator = self.get_evaluator(dev_df, evaluator_func=evaluator_func, collection=collection, top_k=top_k,
                                       encode_batch_size=encode_batch_size)

        print(self.train_length)
        warmup_steps = math.ceil(self.train_length * 1 * 0.1)  # 10% of train data for warm-up
        evaluation_steps = math.ceil(self.train_length * 0.1)

        print('start train...')
        model.fit(train_objectives=train_obj, epochs=self.epoch, warmup_steps=warmup_steps,
                  evaluator=evaluator,
                  save_best_model=True,
                  output_path=save_model,
                  evaluation_steps=evaluation_steps,
                  callback=self.call_back,
                  optimizer_params={'lr': self.learning_rate})

        df = pd.DataFrame(self.all_scores)
        df.to_excel(save_model + 'my_score.xlsx')
        TrainRetrieval.save_parameters(self, save_model=f'{save_model}parameters.json')

    def call_back(self, score, epoch, steps):
        self.all_scores.append({str(epoch) + '-' + str(steps): score})
        if score > self.best_score:
            self.best_score = score
        print(f'epoch:{epoch}: score:{score} ')

    def get_corpus(self):
        self.corpus_file = "./data/v2/label_dict.xlsx"
        corpus = pd.read_excel(self.corpus_file)
        corpus = dict(zip(corpus['entry'].tolist(), corpus['label_num'].tolist()))
        return corpus


class Trainer(TrainRetrieval):
    def __init__(self):
        super(Trainer, self).__init__()

    def run(self):
        # self.train_0930()
        # self.train_1008()
        # self.train_1009()
        self.pretrain_model()

    def train_0930(self):
        train_df = pd.read_csv("./data/v2/train_2.csv.gz", compression='gzip', sep='|')
        dev_df = pd.read_csv("./data/v2/eval.csv.gz", compression='gzip', sep='|')
        # self.model_path = "distiluse-base-multilingual-cased-v1"
        self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/v2/v2.0/"

        # self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"


        self.cuda_device = 3
        self.max_seqence_length = 128
        self.use_st_model = True
        self.train_batch_size = 256
        self.epoch = 5
        self.learning_rate = 4e-5
        self.train(train_df, dev_df, save_model="./best_model/v2/v2.1/", loss_func='CosineSimilarityLoss',
                   evaluator_func='MyEvaluator', collection='t2')

    def train_1008(self):
        self.train_file = "./data/v2/train_2.csv.gz"
        train_df = pd.read_csv(self.train_file, compression='gzip', sep='|')
        self.dev_file = "./data/v2/eval.csv.gz"
        dev_df = pd.read_csv(self.dev_file, compression='gzip', sep='|')
        # self.model_path = "all-mpnet-base-v2"
        self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"

        self.cuda_device = 1
        self.max_seqence_length = 384
        self.use_st_model = True
        self.train_batch_size = 32
        self.epoch = 3
        self.learning_rate = 9e-5
        self.train(train_df, dev_df, save_model="./best_model/v2/v2.9/", loss_func='MultipleNegativesRankingLoss',
                   evaluator_func="MyEvaluator", collection='t3')

    def train_1009(self):
        self.train_file = "./data/v2/train.csv.gz"
        train_df = pd.read_csv(self.train_file, compression='gzip', sep='|')
        self.dev_file = "./data/v2/eval.csv.gz"
        dev_df = pd.read_csv(self.dev_file, compression='gzip', sep='|')

        # self.model_path = "all-mpnet-base-v2"
        # self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
        # self.model_path = '/large_files/pretrained_pytorch/mt5_zh_en/'
        # self.model_path = "paraphrase-multilingual-mpnet-base-v2"
        self.model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # self.model_path = "sentence-transformers/distiluse-base-multilingual-cased-v1"
        # self.model_path = "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking"
        # self.model_path = "./best_model/v2/v2.1.1/"


        self.cuda_device = Trainer.get_auto_device()
        self.max_seqence_length = 128
        self.use_st_model = False
        self.train_batch_size = 64
        self.epoch = 3
        self.learning_rate = 8e-5
        self.output_dimension = 768
        self.train(train_df, dev_df, save_model="./best_model/v2/v2.1.3/",
                   loss_func='MultipleNegativesRankingLoss',  # multi-task
                   # CosineSimilarityLoss,MultipleNegativesRankingLoss,OnlineContrastiveLoss,BatchHardSoftMarginTripletLoss
                   evaluator_func="MyEvaluator",
                   collection=f't_{self.cuda_device}',
                   top_k=100,
                   encode_batch_size=64)



    def pretrain_model(self):
        self.train_file = "./data/v2/train.csv.gz"
        train_df = pd.read_csv(self.train_file, compression='gzip', sep='|')
        self.dev_file = "./data/v2/eval.csv.gz"
        dev_df = pd.read_csv(self.dev_file, compression='gzip', sep='|')

        self.model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.use_st_model = False

        self.cuda_device = Trainer.get_auto_device()
        self.max_seqence_length = 128
        self.output_dimension = 1024
        self.train_batch_size = 72
        self.epoch = 5
        self.learning_rate = 2e-4

        self.train(train_df, dev_df, save_model="./best_model/v2/v2.1.0/",
                   loss_func='MultipleNegativesRankingLoss',  # multi-task
                   evaluator_func="MyEvaluator",
                   collection=f't_{self.cuda_device}',
                   top_k=10,
                   encode_batch_size=64)

    def train_retrieval_model(self):
        self.train_file = "./data/v2/train.csv.gz"
        train_df = pd.read_csv(self.train_file, compression='gzip', sep='|')
        self.dev_file = "./data/v2/eval.csv.gz"
        dev_df = pd.read_csv(self.dev_file, compression='gzip', sep='|')

        self.model_path = "./best_model/v2/v2.1.0/"
        self.use_st_model = True

        self.cuda_device = Trainer.get_auto_device()
        self.max_seqence_length = 128
        self.output_dimension = 1024
        self.train_batch_size = 64
        self.epoch = 5
        self.learning_rate = 4e-5
        self.corpus = self.get_corpus()

        self.train(train_df, dev_df, save_model="./best_model/v2/v2.1.1/",
                   loss_func='OnlineContrastiveLoss',  # multi-task
                   evaluator_func="MyEvaluator",
                   collection=f't_{self.cuda_device}',
                   top_k=10,
                   encode_batch_size=64)

if __name__ == '__main__':
    # get_auto_device()
    # FineTurn().run()
    Trainer().run()
    pass
