# encoding: utf-8
'''
@author: zyl
@file: train.py
@time: 2021/8/11 13:18
@desc:
'''
import math

import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import losses
from sentence_transformers import evaluation
from torch.utils.data import DataLoader
from pharm_ai.panel.panel_utils import ModelUtils
from pharm_ai.panel.entry_match.dt import DT

all_scores = []

from pharm_ai.panel.entry_match.revise_evaluator import MyEvaluator


class FineTurn:
    def __init__(self):
        pass

    def run(self):
        # self.train(train_batch_size=128, epoch=5, lr=1e-4, save_model="./best_model/em/",cuda_device=1)
        # self.train(train_batch_size=128, epoch=5, lr=1e-5, save_model="./best_model/em2/",cuda_device=1)
        # self.train(train_batch_size=128, epoch=5, lr=1e-6, save_model="./best_model/em3/", cuda_device=0)
        # self.train(train_batch_size=128, epoch=5, lr=8e-5, save_model="./best_model/em6/", cuda_device=1)
        # self.train(train_batch_size=128, epoch=5, lr=5e-5, save_model="./best_model/em7/", cuda_device=1)
        # self.train(train_batch_size=128, epoch=5, lr=3e-5, save_model="./best_model/em8/", cuda_device=0)
        # self.train(train_batch_size=128, epoch=5, lr=9e-5, save_model="./best_model/em9/", cuda_device=2)
        # self.train(train_batch_size=128, epoch=5, lr=5e-5, save_model="./best_model/em10/", cuda_device=1)
        # self.train(train_batch_size=128, epoch=5, lr=9e-5, save_model="./best_model/em11/", cuda_device=2)
        # self.train(train_batch_size=128, epoch=5, lr=9e-5, save_model="./best_model/em12/", cuda_device=0)
        # self.train(train_batch_size=128, epoch=5, lr=1e-5, save_model="./best_model/em13/", cuda_device=1)

        # self.train(train_batch_size=128, epoch=5, lr=2e-4, save_model="./best_model/em14/", cuda_device=1)
        # self.train(train_batch_size=128, epoch=5, lr=9e-5, save_model="./best_model/em15/", cuda_device=0)
        # self.train(train_batch_size=128, epoch=5, lr=6e-4, save_model="./best_model/em16/", cuda_device=2)
        # self.train(train_batch_size=128, epoch=5, lr=1e-5, save_model="./best_model/em17/", cuda_device=1)
        # self.train(train_batch_size=128, epoch=5, lr=1e-6, save_model="./best_model/em18/", cuda_device=3)
        # self.train(train_batch_size=128, epoch=5, lr=1e-6, save_model="./best_model/em19/", cuda_device=0)

        # self.train(train_batch_size=128, epoch=5, lr=1e-4, save_model="./best_model/em20/", cuda_device=0)
        # self.train(train_batch_size=128, epoch=5, lr=1e-5, save_model="./best_model/em21/", cuda_device=0)
        # self.train(train_batch_size=128, epoch=1, lr=1e-6, save_model="./best_model/em22/", cuda_device=4)
        # self.train(train_batch_size=128, epoch=3, lr=5e-5, save_model="./best_model/em23/", cuda_device=4)
        # self.train(train_batch_size=128, epoch=3, lr=5e-5, save_model="./best_model/em24/", cuda_device=4)
        self.train(train_batch_size=128, epoch=3, lr=5e-5, save_model="./best_model/em25/", cuda_device=4)
        pass

    def test(self):
        pass

    def get_dt(self):
        # train_df = pd.read_excel('./data/em_0811.xlsx', 'train')
        # train_df = pd.read_excel('./data/em_0812.xlsx', 'train_up')
        # train_df = pd.read_excel('./data/em_0812_2.xlsx', 'train')
        # train_df = pd.read_excel('./data/em_0812_2.xlsx', 'train_up')
        # train_df = pd.read_excel('./data/em_0811.xlsx', 'train_pos')
        train_df = pd.read_excel('./data/em_0816.xlsx', 'train_up')
        eval_df = pd.read_excel('./data/em_0811.xlsx', 'old_eval')

        # train_df = pd.read_excel('./data/em_0816_3.xlsx', 'train')  # type:pd.DataFrame
        # train_df = train_df.astype({'label':'float64'})
        # eval_df = pd.read_excel('./data/em_0816_3.xlsx', 'eval')
        # eval_df = eval_df.astype({'label': 'float64'})
        # eval_df_neg = DT().prepare_neg_dt(eval_df, neg_smaple_size=3, method='predict')

        train_examples = DT.prepare_dt(train_df)
        eval_examples = DT.prepare_dt(eval_df)
        return train_examples, eval_examples

    def get_model(self, cuda_device):
        # word_embedding_model = models.Transformer(
        #     "/home/zyl/disk/PharmAI/pharm_ai/test_jina/train/models/distiluse_base_multilingual_cased/0_Transformer/",
        #     max_seq_length=256)
        # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        # model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=f'cuda:{str(cuda_device)}')

        model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
        model_dim = 768
        model = SentenceTransformer(model_path, device=f'cuda:{str(cuda_device)}')
        return model

    def train(self, train_batch_size=128, epoch=5, lr=1e-5, save_model="./best_model/em2/", cuda_device=1):
        model_save_path = save_model
        model = self.get_model(cuda_device)

        train_examples, eval_examples = self.get_dt()
        train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=train_batch_size)
        # train_loss = losses.CosineSimilarityLoss(model)
        # train_loss = losses.MultipleNegativesRankingLoss(model)
        # train_loss = losses.ContrastiveLoss(model)
        train_loss =losses.OnlineContrastiveLoss(model)

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

        # evaluator = evaluation.TripletEvaluator.from_input_examples(eval_examples, name='sts-eval')
        # evaluator = evaluation.RerankingEvaluator.mro().from_input_examples(eval_examples, name='sts-eval')
        evaluator = MyEvaluator.from_input_examples(eval_examples, name='sts-eval')
        # evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(train_examples[-300:-1], name='sts-eval')

        warmup_steps = math.ceil(len(train_dataloader) * 1 * 0.1)  # 10% of train data for warm-up

        evaluation_steps_ = int(len(train_dataloader) * 0.1)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epoch, warmup_steps=warmup_steps,
                  evaluator=evaluator,
                  save_best_model=True,
                  output_path=model_save_path,
                  evaluation_steps=evaluation_steps_,
                  callback=FineTurn.call_back,
                  optimizer_params={'lr': lr})
        global all_scores
        df = pd.DataFrame(all_scores)
        df.to_excel(model_save_path + 'my_score.xlsx')

    @staticmethod
    def call_back(score, epoch, steps):
        global all_scores
        all_scores.append({str(epoch) + '-' + str(steps): score})
        print(f'epoch:{epoch}, score:{score} ')


from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample


class Trainer2:
    def __init__(self):
        pass

    def run(self):
        # self.train(train_batch_size=128, epoch=10, lr=5e-5, save_model="./best_model/emv2_1/", cuda_device=1)
        # self.train(train_batch_size=128, epoch=10, lr=1e-5, save_model="./best_model/emv2_2/", cuda_device=1)
        self.train(train_batch_size=128, epoch=5, lr=4e-5, save_model="./best_model/emv2_3/", cuda_device=2)

    @staticmethod
    def get_dt(df):
        texts = df['texts'].tolist()
        label = df['label'].tolist()
        examples = []
        for t, l in zip(texts, label):
            examples.append(InputExample(texts=[t], label=l))
        return examples

    # @staticmethod
    # def deal_with_eval_examples(df):
    #     dt = []
    #     for _, sub_df in df.iterrows():
    #         dt.append(InputExample(texts=[sub_df['entry'], sub_df['entity']], label=sub_df['label']))
    #     eval_df = pd.read_excel('./data/em_0816_3.xlsx', 'eval')
    #     eval_df_neg = DT().prepare_neg_dt(eval_df, neg_smaple_size=3, method='predict')

    def get_model(self, cuda_device):
        # word_embedding_model = models.Transformer(
        #     "/home/zyl/disk/PharmAI/pharm_ai/test_jina/train/models/distiluse_base_multilingual_cased/0_Transformer/",
        #     max_seq_length=256)
        # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        # model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=f'cuda:{str(cuda_device)}')

        model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/emv2_2/"
        model_dim = 768
        model = SentenceTransformer(model_path, device=f'cuda:{str(cuda_device)}')
        return model

    def train(self, train_batch_size=128, epoch=10, lr=8e-5, save_model="./best_model/em2/", cuda_device=1):
        model_save_path = save_model
        model = self.get_model(cuda_device)

        train_df = pd.read_excel('./data/em_0827.xlsx', 'train_pos')

        train_examples = Trainer2.get_dt(train_df)
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
        # train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
        train_loss = losses.BatchSemiHardTripletLoss(model=model)

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
        evaluator = MyEvaluator.from_input_examples(eval_examples, name='sts-eval')

        warmup_steps = math.ceil(len(train_dataloader) * 1 * 0.1)  # 10% of train data for warm-up

        evaluation_steps_ = int(len(train_dataloader) * 0.1)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epoch, warmup_steps=warmup_steps,
                  evaluator=evaluator,
                  save_best_model=True,
                  output_path=model_save_path,
                  evaluation_steps=evaluation_steps_,
                  callback=FineTurn.call_back,
                  optimizer_params={'lr': lr})
        global all_scores
        df = pd.DataFrame(all_scores)
        df.to_excel(model_save_path + 'my_score.xlsx')

    @staticmethod
    def call_back(score, epoch, steps):
        global all_scores
        all_scores.append({str(epoch) + '-' + str(steps): score})
        print(f'epoch:{epoch}, score:{score}')

class TrainerV4:
    def __init__(self):
        pass

    def run(self):
        self.train(train_batch_size=64, epoch=5, lr=1e-5, save_model="./best_model/emv2.4.0.2/", cuda_device=1)

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
        # model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=f'cuda:{str(cuda_device)}')

        model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/emv2.4.0.1/"
        model_dim = 768
        model = SentenceTransformer(model_path, device=f'cuda:{str(cuda_device)}')
        return model

    def train(self, train_batch_size=128, epoch=10, lr=8e-5, save_model="./best_model/em2/", cuda_device=1):
        model_save_path = save_model
        model = self.get_model(cuda_device)

        train_df = pd.read_excel('./data/em_0901.xlsx', 'train')
        anchor_s = train_df['anchor'].tolist()
        positive_s = train_df['positive'].tolist()
        negative_s = train_df['negative'].tolist()
        train_examples = []
        for a,p,n in zip(anchor_s, positive_s,negative_s):
            train_examples.append(InputExample(texts=[a,p,n]))

        train_dataset = SentencesDataset(train_examples, model)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model=model)

        test_df = pd.read_excel("./data/em_0901.xlsx", 'eval')
        labels = test_df['target_text'].tolist()
        revised_target_texts = ModelUtils.revise_target_texts(target_texts=labels, input_texts=labels,
                                                              check_in_input_text=False, delimiter='|')
        to_predict = test_df['input_text'].tolist()
        eval_examples = []
        for t, r in zip(to_predict, revised_target_texts):
            eval_examples.append(InputExample(texts=[t, r]))
        evaluator = MyEvaluator.from_input_examples(eval_examples, name='sts-eval')

        warmup_steps = math.ceil(len(train_dataloader) * 1 * 0.1)  # 10% of train data for warm-up

        evaluation_steps_ = int(len(train_dataloader) * 0.1)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epoch, warmup_steps=warmup_steps,
                  evaluator=evaluator,
                  save_best_model=True,
                  output_path=model_save_path,
                  evaluation_steps=evaluation_steps_,
                  callback=FineTurn.call_back,
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
    # FineTurn().run()
    # Trainer2().run()
    TrainerV4().run()
    pass
