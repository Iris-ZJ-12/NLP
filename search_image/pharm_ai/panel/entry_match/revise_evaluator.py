# encoding: utf-8
'''
@author: zyl
@file: revise_evaluator.py
@time: 2021/8/27 15:39
@desc:
'''

from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
import logging
import os
import csv
# from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
# from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
from sentence_transformers.readers import InputExample
import pandas as pd
from sentence_transformers import SentenceTransformer
from pharm_ai.panel.entry_match.eval import Evaluator
from pharm_ai.panel.panel_utils import ModelUtils
from pharm_ai.panel.entry_match.milvus_util import MilvusHelper

logger = logging.getLogger(__name__)
df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/v2/dt_0930.xlsx", "di_dict")
library = list(set(df['entry'].tolist()))


class MyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, to_predict_texts: List[str], labels: List[str], batch_size: int = 16,
                 main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False,
                 write_csv: bool = True, collection='t1',top_k=100,encode_batch_size=128):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param to_predict_texts:  List with the first sentence in a pair
        :param labels: List with the second sentence in a pair
        :param scores: Similarity score between to_predict_texts[i] and labels[i]
        :param write_csv: Write results to a CSV file
        """
        self.to_predict_texts = to_predict_texts
        self.labels = labels
        self.write_csv = write_csv
        self.top_k = top_k
        self.encode_batch_size =encode_batch_size
        assert len(self.to_predict_texts) == len(self.labels)

        self.collection = collection
        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "score"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        to_predict_texts = []
        labels = []

        for example in examples:
            to_predict_texts.append(example.texts[0])
            labels.append(example.texts[1])
        return cls(to_predict_texts, labels, **kwargs)

    @staticmethod
    def evaluate(y_true, y_pred):
        r = 0
        w = 0
        for t, p in zip(y_true, y_pred):
            if set(t).issubset(set(p)):
                r += 1
            else:
                w += 1
        print(f'right: {r}')
        print(f'wrong:{w}')
        return r / (r + w)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        milvus = MilvusHelper(dimension=model.get_sentence_embedding_dimension(), clear_collection=True,
                              collection=self.collection)
        vecs = model.encode(library, batch_size=self.encode_batch_size, show_progress_bar=False,
                            normalize_embeddings=True)

        milvusids = milvus.insert(vecs)
        milvus.create_index()
        id_dict = {i: j for i, j in zip(milvusids, library)}

        vecs2 = model.encode(self.to_predict_texts, batch_size=self.encode_batch_size, show_progress_bar=False,
                             normalize_embeddings=True)
        res = milvus.search(top_k=self.top_k, query=vecs2.tolist())

        matches = Evaluator.result_filter(res, id_dict, return_method='entry', score_threshold=0)

        score = MyEvaluator.evaluate(self.labels, matches)

        logger.info(f"my score: {score}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, score])

        return score
        # if self.main_similarity == SimilarityFunction.COSINE:
        #     return eval_spearman_cosine
        # elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
        #     return eval_spearman_euclidean
        # elif self.main_similarity == SimilarityFunction.MANHATTAN:
        #     return eval_spearman_manhattan
        # elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
        #     return eval_spearman_dot
        # elif self.main_similarity is None:
        #     return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
        # else:
        #     raise ValueError("Unknown main_similarity value")


import logging
import os
import csv
from typing import List

from sklearn.metrics import accuracy_score, classification_report, f1_score


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


if __name__ == '__main__':
    y_pred = [0, 2, 1, 0]
    y_true = [0, 1, 2, 1]
    print(classification_report(y_true, y_pred))
    print(f1_score(y_true, y_pred, average='macro'))
    print(f1_score(y_true, y_pred, average='micro'))
    print(accuracy_score(y_true, y_pred))  # 0.5
    print(accuracy_score(y_true, y_pred, normalize=False))  # 2

# from sentence_transformers import SentenceTransformer, models
#
# word_embedding_model = models.Transformer(
#     "/home/zyl/disk/PharmAI/pharm_ai/test_jina/train/models/distiluse_base_multilingual_cased/0_Transformer/",
#     max_seq_length=256)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# my_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=f'cuda:{str(0)}')
#
# milvus = MilvusHelper(dimension=768, clear_collection=True, collection='test')
# vecs = my_model.encode(library, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
#
# milvusids = milvus.insert(vecs)
# milvus.create_index()
# id_dict = {i: j for i, j in zip(milvusids, library)}
#
#
# class MyEvaluator2:
#     """
#     This evaluator can be used with the CrossEncoder class.
#
#     It is designed for CrossEncoders with 2 or more outputs. It measure the
#     accuracy of the predict class vs. the gold labels.
#     """
#
#     def __init__(self, to_predict_texts: List, labels: List, name: str = '', write_csv: bool = True):
#         self.to_predict_texts = to_predict_texts
#         self.labels = labels
#         self.write_csv = write_csv
#         self.name = name
#
#         self.csv_file = "CESoftmaxAccuracyEvaluator" + ("_" + name if name else '') + "_results.csv"
#         self.csv_headers = ["epoch", "steps", "Accuracy"]
#         self.write_csv = write_csv
#
#     @classmethod
#     def from_input_examples(cls, examples: List[InputExample], **kwargs):
#         to_predict_texts = []
#         labels = []
#
#         for example in examples:
#             to_predict_texts.append(example.texts[0])
#             labels.append(example.texts[1])
#         return cls(to_predict_texts, labels, **kwargs)
#
#     def e2(self):
#         pass
#
#     # @classmethod
#     # def from_input_examples(cls, examples: List[InputExample], **kwargs):
#     #     sentence_pairs = []
#     #     labels = []
#     #
#     #     for example in examples:
#     #         sentence_pairs.append(example.texts)
#     #         labels.append(example.label)
#     #     return cls(sentence_pairs, labels, **kwargs)
#
#     def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
#         if epoch != -1:
#             if steps == -1:
#                 out_txt = " after epoch {}:".format(epoch)
#             else:
#                 out_txt = " in epoch {} after {} steps:".format(epoch, steps)
#         else:
#             out_txt = ":"
#
#         logger.info("CESoftmaxAccuracyEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
#
#         vecs2 = my_model.encode(self.to_predict_texts, batch_size=128, show_progress_bar=False,
#                                 normalize_embeddings=True)
#         res = milvus.search(top_k=10, query=vecs2.tolist())
#         # print(res)
#         matches = Evaluator.result_filter(res, id_dict, return_method='entry', score_threshold=0)
#
#         sentence_pairs = []
#         for t, r in zip(self.to_predict_texts, matches):
#             for j in r:
#                 sentence_pairs.append([t, j])
#         pred_scores = model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
#         pred_labels = np.argmax(pred_scores, axis=1).tolist()
#
#         p_s = []
#         for k in range(int(len(pred_labels) / 10)):
#             r = set()
#             for i in range(len(pred_labels[k:k + 10])):
#                 if pred_labels[k * 10 + i] == 1:
#                     r.add(matches[k][i])
#             p_s.append(r)
#
#         ModelUtils.entity_recognition_v2(self.labels, p_s)
#
#         acc = np.sum(pred_labels == self.labels) / len(self.labels)
#
#         logger.info("Accuracy: {:.2f}".format(acc * 100))
#
#         if output_path is not None and self.write_csv:
#             csv_path = os.path.join(output_path, self.csv_file)
#             output_file_exists = os.path.isfile(csv_path)
#             with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 if not output_file_exists:
#                     writer.writerow(self.csv_headers)
#
#                 writer.writerow([epoch, steps, acc])
#
#         return acc
