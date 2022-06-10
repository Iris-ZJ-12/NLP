# encoding: utf-8
'''
@author: zyl
@file: predict.py
@time: 2021/8/11 13:18
@desc:
'''
import pandas as pd
from pharm_ai.panel.entry_match.milvus_util import MilvusHelper

class Predictor():
    def __init__(self):
        pass

    def run(self):
        self.test()

    def test(self):
        to_predicts = ['中枢神经系统疾病','CLOVES syndrome','hair loss','disease or symptom is regulated by a kinase enzyme'
                       'tumors of the central nervous system (brain, astrocytoma, glioblastoma, glioma)',
                       'lung, colon, pancreatic, gastric, ovarian, cervical, breast and prostate cancer',
                       ]
        labels = ['神经系统疾病','CLOVES syndrome','脱发','没有疾病','脑癌,胶质母细胞瘤,胶质瘤',
                  '肺癌,结肠癌,胰腺癌,胃癌,卵巢癌,宫颈癌,乳腺癌,前列腺癌']
        self.predict(to_predicts)
        pass


    def get_retrieval_model(self):
        from pharm_ai.panel.entry_match.predict_retrieval import RetrievalPredictor
        retrieval_model = RetrievalPredictor()
        retrieval_model.model_path = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/v2/v2.1.5/"
        retrieval_model.cuda_device = 3
        retrieval_model.top_k = 100
        retrieval_model.model_dim = 768
        retrieval_model.set_model()
        retrieval_model.batch_size = 32
        retrieval_model.set_milvus(clear_collection=False,collection='test1')
        return retrieval_model

    def get_reranker_model(self):
        from pharm_ai.panel.entry_match.predict_reranker import RerankerPredictor
        reranker_model = RerankerPredictor()
        reranker_model.model_path = "./best_model/v2/v2.2.1/"
        reranker_model.cuda_device = 3
        reranker_model.model_dim = 768
        reranker_model.eval_batch_size = 24
        reranker_model.label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        reranker_model.int2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        reranker_model.train_num_labels = len(set(reranker_model.label2int.values()))
        reranker_model.set_model()
        
        return reranker_model

    def eval(self):
        pass

    def predict(self,to_predicts:list):
        import numpy as np
        retrieval_model = self.get_retrieval_model()
        res_1 = retrieval_model.predict(to_predicts,top_k=10)
        print(res_1)

        reranker_model= self.get_reranker_model()

        for t_p,r in zip(to_predicts,res_1):
            to_predicts2 = [[t_p,_] for _ in r]
            print(to_predicts2)
            pred_scores = reranker_model.model.predict(to_predicts2, convert_to_numpy=True, show_progress_bar=True)
            pred_labels = np.argmax(pred_scores, axis=1)

            relationships = list(map(lambda x: reranker_model.int2label.get(x), pred_labels))
            for t_p, r in zip(to_predicts2, relationships):
                print(f'entity:{t_p[0]} --- {r} --- entry:{t_p[1]}')
            

if __name__ == '__main__':
    Predictor().run()
