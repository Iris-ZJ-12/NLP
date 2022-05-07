# encoding: utf-8
'''
@author: zyl
@file: predict_retrieval.py
@time: 2021/10/8 9:13
@desc:
'''


import pandas as pd
from sentence_transformers import SentenceTransformer

from pharm_ai.panel.ner_utils import NERUtils
from pharm_ai.panel.entry_match.milvus_util import MilvusHelper


class RetrievalPredictor:
    def __init__(self):
        self.milvus = None
        self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/v2/v2.5/"
        # self.model_path ="/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/v2/v2.1/"
        # self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
        self.model_path  ="distiluse-base-multilingual-cased-v1"
        self.model_path ="/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/v2/v2.1.3/"
        self.cuda_device = 3
        self.top_k = 100
        self.model_dim = 768
        self.set_model()
        self.batch_size = 32
        self.set_milvus()

    # @staticmethod
    # def result_filter(res: list[list], id_dict, score_threshold=0.5, return_method='dict'):
    #     matched = []
    #     if not res:
    #         return matched
    #     for each_matches in res:
    #         m = []
    #         for each_match in each_matches:
    #             score = each_match.distance
    #             if score <= score_threshold:
    #                 break
    #             candidate = id_dict[each_match.id]
    #             m.append((candidate, each_match.distance))
    #         matched.append(m)
    #     if return_method == 'entry':
    #         matched = [[j[0] for j in i] for i in matched]
    #     return matched

    @staticmethod
    def result_filter(res: list[list], id_dict, score_threshold=0.5, return_method='dict'):
        matched = []
        if not res:
            return matched
        for each_matches in res:
            m = []
            for each_match in each_matches:
                score = each_match.distance
                if score <= score_threshold:
                    continue
                candidate = id_dict[each_match.id]
                m.append((candidate, each_match.distance))
            matched.append(m)
        if return_method == 'entry':
            matched = [[j[0] for j in i] for i in matched]
        return matched

    def set_model(self):
        self.model = SentenceTransformer(self.model_path, device=f'cuda:{self.cuda_device}')
        self.model_dim = self.model.get_sentence_embedding_dimension()

    def set_milvus(self,clear_collection=True,collection='test1'):
        self.milvus = MilvusHelper(dimension=self.model.get_sentence_embedding_dimension(), clear_collection=clear_collection, collection=collection)
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/v2/dt_0930.xlsx", "di_dict")
        library = list(set(df['entry'].tolist()))
        vecs = self.model.encode(library, batch_size=self.batch_size, show_progress_bar=True, normalize_embeddings=True)
        milvusids = self.milvus.insert(vecs)
        self.milvus.create_index()
        self.id_dict = {i: j for i, j in zip(milvusids, library)}

    def eval(self):
        # model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em25/"
        # model_path = '/home/zyl/disk/PharmAI/pharm_ai/panel/best_model/paraphrase-multilingual-mpnet-base-v2/'

        eval_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/v2/dt_0930.xlsx", "eval")
        eval_df['prefix'] = 'disease_em'
        labels = eval_df['entry'].tolist()
        revised_target_texts = NERUtils.revise_target_texts(target_texts=labels, input_texts=labels,
                                                              check_in_input_text=False, delimiter='|')
        to_predict = eval_df['entity'].tolist()
        vecs2 = self.model.encode(to_predict, batch_size=self.batch_size, show_progress_bar=True, normalize_embeddings=True)

        res = self.milvus.search(top_k=self.top_k, query=vecs2.tolist())

        matches = RetrievalPredictor.result_filter(res, self.id_dict, return_method='entry', score_threshold=0)

        r = 0
        w = 0
        for t, m in zip(revised_target_texts, matches):
            if set(t).issubset(set(m)):
                r += 1
            else:
                w += 1
        print(f'right:{r},wrong:{w},recall:{r / (r + w)}')

    # def set_milvus(self, cuda_device=0):
    #     model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
    #     model_dim = 768
    #     self.sbert = SentenceTransformer(model_path, device=f'cuda:{cuda_device}')

    #     # milvus
    #     self.milvus = MilvusHelper(dimension=model_dim, clear_collection=True, collection='test')
    #     df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/entry_dict_0508.xlsx", "disease_dict")
    #     library = list(set(df['entry'].tolist()))
    #     vecs = self.sbert.encode(library, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    #     milvusids = self.milvus.insert(vecs)
    #     self.milvus.create_index()
    #     self.id_dict = {i: j for i, j in zip(milvusids, library)}

    def predict(self, to_predict: list, top_k=10, score_threshold=0.0):
        if not self.milvus:
            self.set_milvus()

        vecs2 = self.model.encode(to_predict, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        res = self.milvus.search(top_k=top_k, query=vecs2.tolist())

        res = RetrievalPredictor.result_filter(res, self.id_dict, return_method='entry', score_threshold=score_threshold)

        return res

    def get_res(self,to_predicts,model_path,top_k):
        self.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/v2/v2.1.3/"
        self.cuda_device = 3
        self.top_k = 100
        self.model_dim = 768
        self.set_model()
        self.batch_size = 32
        self.set_milvus()

    # def test(self):
    #     train_pos_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0811.xlsx",
    #                                  'train_pos')
    #     to_predict = list(set(train_pos_df['entity'].tolist()))[0:1000]
    #     for i in to_predict:
    #         r = self.predict([i], top_k=10, score_threshold=0.5)
    #         print(len(r[0]))

if __name__ == '__main__':
    RetrievalPredictor().eval()
    # Evaluator().test()