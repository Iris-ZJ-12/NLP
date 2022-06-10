# encoding: utf-8
'''
@author: zyl
@file: predict_reranker.py
@time: 2021/10/12 13:25
@desc:
'''
from pharm_ai.panel.entry_match.train_reranker import RerankerTrainer


class RerankerPredictor:
    def __init__(self):
        self.model_path = "./best_model/v2/v2.2.1/"
        self.cuda_device = RerankerTrainer.get_auto_device()
        self.model_dim = 768
        self.eval_batch_size = 24
        self.label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.int2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.train_num_labels = len(set(self.label2int.values()))
        self.set_model()

    def run(self):
        to_predicts = [["DMAIC", "分枝杆菌病"],
                       ["tooth loss", "牙齿脱落"],
                       ["CORD", "没有疾病"],
                       ["实体恶性肿瘤", "实体瘤"],
                       ["viral, parasitic or bacterial diseases", "细菌感染"]]
        self.predict(to_predicts)
        pass

    def set_model(self):
        from sentence_transformers.cross_encoder import CrossEncoder
        self.model = CrossEncoder(self.model_path, device=f'cuda:{str(self.cuda_device)}',
                                  num_labels=self.train_num_labels)

    def predict(self, to_predicts: list):
        import numpy as np

        pred_scores = self.model.predict(to_predicts, convert_to_numpy=True, show_progress_bar=True)
        pred_labels = np.argmax(pred_scores, axis=1)
        relationships = list(map(lambda x: self.int2label.get(x), pred_labels))

        for t_p, r in zip(to_predicts, relationships):
            print(f'entity:{t_p[0]} --- {r} --- entry:{t_p[1]}')
        return pred_labels


if __name__ == '__main__':
    RerankerPredictor().run()
