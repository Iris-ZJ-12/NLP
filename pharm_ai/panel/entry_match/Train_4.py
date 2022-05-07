# encoding: utf-8
'''
@author: zyl
@file: Train_4.py
@time: 2021/9/1 11:40
@desc:
'''
all_scores = []
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import gzip
import csv
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample
from pharm_ai.panel.entry_match.revise_evaluator import MyEvaluator2
from pharm_ai.panel.panel_utils import ModelUtils


class TrainerV5:
    def __init__(self):
        pass

    def run(self):
        self.train(train_batch_size=128, epoch=5, lr=4e-5, save_model="./best_model/emv2.5.0.1/", cuda_device=0)

    @staticmethod
    def get_dt(df):
        texts = df['texts'].tolist()
        label = df['label'].tolist()
        examples = []
        for t, l in zip(texts, label):
            examples.append(InputExample(texts=[t], label=l))
        return examples

    def get_model(self, cuda_device):
        # word_embedding_model = models.Transformer(
        #     "/home/zyl/disk/PharmAI/pharm_ai/test_jina/train/models/distiluse_base_multilingual_cased/0_Transformer/",
        #     max_seq_length=256)
        # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        # model = CrossEncoder(
        #     "/home/zyl/disk/PharmAI/pharm_ai/test_jina/train/models/distiluse_base_multilingual_cased/0_Transformer/"
        #     , device=f'cuda:{str(cuda_device)}', num_labels=2)

        model = CrossEncoder(
            "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/emv2.5.0.0/"
            , device=f'cuda:{str(cuda_device)}', num_labels=2)

        # model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/emv2_2/"
        # model_dim = 768
        # model = SentenceTransformer(model_path, device=f'cuda:{str(cuda_device)}')
        return model

    def train(self, train_batch_size=128, epoch=10, lr=8e-5, save_model="./best_model/em2/", cuda_device=1):
        model_save_path = save_model
        model = self.get_model(cuda_device)

        train_df = pd.read_excel('./data/em_0817.xlsx', 'train')

        t_a = train_df['text_a'].tolist()
        t_b = train_df['text_b'].tolist()
        l_s = train_df['labels'].tolist()
        train_examples = []
        for a, b, l in zip(t_a, t_b, l_s):
            train_examples.append(InputExample(texts=[a, b], label=l))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)

        test_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/processed_entry_match_0508.xlsx",
                                'eval')
        test_df = test_df[test_df['prefix'] == 'disease_em']
        labels = test_df['target_text'].tolist()
        revised_target_texts = ModelUtils.revise_target_texts(target_texts=labels, input_texts=labels,
                                                              check_in_input_text=False, delimiter='|')
        to_predict = test_df['input_text'].tolist()
        eval_examples = []
        for t, r in zip(to_predict, revised_target_texts):
            eval_examples.append(InputExample(texts=[t, r]))
        evaluator = MyEvaluator2.from_input_examples(eval_examples, name='sts-eval')

        # eval_df = pd.read_excel('./data/em_0817.xlsx', 'eval')
        # eval_df = eval_df[eval_df['target_text'] == 1]
        #
        #
        # t_a = eval_df['text_a'].tolist()
        # t_b = eval_df['text_b'].tolist()
        # l_s = eval_df['labels'].tolist()
        # eval_examples = []
        # for a, b, l in zip(t_a, t_b, l_s):
        #     eval_examples.append(InputExample(texts=[a, b], label=l))
        # evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(eval_examples, name='AllNLI-dev')

        # evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples, name='AllNLI-dev')
        #
        # test_df = pd.read_excel("./data/em_0901.xlsx", 'eval')
        # labels = test_df['target_text'].tolist()
        # revised_target_texts = ModelUtils.revise_target_texts(target_texts=labels, input_texts=labels,
        #                                                       check_in_input_text=False, delimiter='|')
        # to_predict = test_df['input_text'].tolist()
        # eval_examples = []
        # for t, r in zip(to_predict, revised_target_texts):
        #     eval_examples.append(InputExample(texts=[t, r]))
        # evaluator = MyEvaluator.from_input_examples(eval_examples, name='sts-eval')

        warmup_steps = math.ceil(len(train_dataloader) * 1 * 0.1)  # 10% of train data for warm-up

        evaluation_steps_ = int(len(train_dataloader) * 0.1)
        model.fit(train_dataloader=train_dataloader,
                  epochs=epoch, warmup_steps=warmup_steps,
                  evaluator=evaluator,
                  save_best_model=True,
                  output_path=model_save_path,
                  evaluation_steps=evaluation_steps_,
                  callback=TrainerV5.call_back,
                  optimizer_params={'lr': lr})
        global all_scores
        df = pd.DataFrame(all_scores)
        df.to_excel(model_save_path + 'my_score.xlsx')

    @staticmethod
    def call_back(score, epoch, steps):
        global all_scores
        all_scores.append({str(epoch) + '-' + str(steps): score})
        print(f'epoch:{epoch}, score:{score}')


if __name__ == '__main__':
    TrainerV5().run()
