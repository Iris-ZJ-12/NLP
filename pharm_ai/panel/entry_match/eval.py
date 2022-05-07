# encoding: utf-8
'''
@author: zyl
@file: eval.py
@time: 2021/8/11 13:18
@desc:
'''

import pandas as pd
from sentence_transformers import SentenceTransformer

from pharm_ai.panel.panel_utils import ModelUtils
from pharm_ai.panel.entry_match.milvus_util import MilvusHelper


class Evaluator:
    def __init__(self):
        self.milvus = None

    def get_model(self):
        pass

    def get_eval_dt(self):
        pass

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
                    break
                candidate = id_dict[each_match.id]
                m.append((candidate, each_match.distance))
            matched.append(m)
        if return_method == 'entry':
            matched = [[j[0] for j in i] for i in matched]
        return matched

    @staticmethod
    def result_filter2(res: list[list], id_dict, score_threshold=0.5, return_method='dict'):
        matched = []
        if not res:
            return matched
        for each_matches in res:
            # print(each_matches)
            m = []
            for each_match in each_matches:
                # print(each_match)
                score = each_match.distance
                if score >= score_threshold:
                    continue
                candidate = id_dict[each_match.id]
                m.append((candidate, each_match.distance))
            matched.append(m)
        if return_method == 'entry':
            matched = [[j[0] for j in i] for i in matched]
        return matched

    def eval(self):
        model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
        model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/v2/v2.1/"
        model_path = "distiluse-base-multilingual-cased-v1"
        # model_path = '/home/zyl/disk/PharmAI/pharm_ai/panel/best_model/paraphrase-multilingual-mpnet-base-v2/'
        model_dim = 768
        sbert = SentenceTransformer(model_path, device='cuda:1')

        # milvus
        milvus = MilvusHelper(dimension=model_dim, clear_collection=True, collection='test1')
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/entry_dict_0508.xlsx", "disease_dict")
        library = list(set(df['entry'].tolist()))
        vecs = sbert.encode(library, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        milvusids = milvus.insert(vecs)
        milvus.create_index()
        id_dict = {i: j for i, j in zip(milvusids, library)}
        # print(id_dict)
        # #######################

        test_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/processed_entry_match_0508.xlsx",
                                'eval')
        test_df = test_df[test_df['prefix'] == 'disease_em']
        labels = test_df['target_text'].tolist()
        revised_target_texts = ModelUtils.revise_target_texts(target_texts=labels, input_texts=labels,
                                                              check_in_input_text=False, delimiter='|')

        to_predict = test_df['input_text'].tolist()
        vecs2 = sbert.encode(to_predict, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

        res = milvus.search(top_k=10, query=vecs2.tolist())

        matches = Evaluator.result_filter(res, id_dict, return_method='entry', score_threshold=0)

        def evaluate(y_true, y_pred):
            r = 0
            w = 0
            for t, p in zip(y_true, y_pred):
                if set(t).issubset(set(p)):
                    r += 1
                else:
                    w += 1
            print(r)
            print(w)
            print(f'recall:{r / (r + w)}')

        evaluate(revised_target_texts, matches)

    def set_milvus(self, cuda_device=0):
        model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
        model_dim = 768
        self.sbert = SentenceTransformer(model_path, device=f'cuda:{cuda_device}')

        # milvus
        self.milvus = MilvusHelper(dimension=model_dim, clear_collection=True, collection='test')
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.4/entry_dict_0508.xlsx", "disease_dict")
        library = list(set(df['entry'].tolist()))
        vecs = self.sbert.encode(library, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        milvusids = self.milvus.insert(vecs)
        self.milvus.create_index()
        self.id_dict = {i: j for i, j in zip(milvusids, library)}

    def predict(self, to_predict: list, top_k=10, cuda_device=0, score_threshold=0.5):
        if not self.milvus:
            self.set_milvus(cuda_device=cuda_device)

        # #######################
        vecs2 = self.sbert.encode(to_predict, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        res = self.milvus.search(top_k=top_k, query=vecs2.tolist())

        # res = Evaluator.result_filter(res, self.id_dict, return_method='entry', score_threshold=score_threshold)

        res = Evaluator.result_filter2(res, self.id_dict, return_method='entry', score_threshold=score_threshold)

        return res

    def test(self):
        train_pos_df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/em_0811.xlsx",
                                     'train_pos')
        to_predict = list(set(train_pos_df['entity'].tolist()))[0:1000]
        for i in to_predict:
            r = self.predict([i], top_k=10, score_threshold=0.5)
            print(len(r[0]))

if __name__ == '__main__':
    Evaluator().eval()
    # Evaluator().test()
