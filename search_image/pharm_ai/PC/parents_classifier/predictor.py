from scipy.special import softmax
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel
import torch
import pandas as pd
import numpy as np
from pprint import pprint
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.PC.dt import train_test0824

f = cfp.project_dir + '/patent_claims/parents_classifier/'
od = f + 'outputs/20200824/'
cd = f + 'cache/20200824/'
bm = f + 'best_model/20200824/'


class ParentsClassifier:

    def __init__(self, cuda_device=-1):

        if torch.cuda.is_available():
            self.use_cuda = True
            self.n_gpu = 1
            self.fp16 = True
        else:
            self.use_cuda = False
            self.n_gpu = 1
            self.fp16 = False

        self.model = ClassificationModel('bert', bm, cuda_device=cuda_device,
                                    num_labels=2, use_cuda=self.use_cuda,
                                    args={'reprocess_input_data': True,
                                          'use_cached_eval_features': True,
                                          'overwrite_output_dir': True,
                                          'n_gpu': self.n_gpu,
                                          'fp16': self.fp16,
                                          'use_multiprocessing': False,
                                          'no_cache': True,
                                          'output_dir': bm, 'cache_dir': cd,
                                          'best_model_dir': bm})

    def predict(self, texts):
        predicted_labels, raw_outputs = self.model.predict(texts)
        predicted_labels = predicted_labels.tolist()
        return predicted_labels

    def eval(self):
        train1, test1, train2, test2, train3, test3 = train_test0824()
        trues = test3['labels'].tolist()
        texts = test3[['text_a', 'text_b']].values.tolist()
        preds = self.predict(texts)
        print(classification_report(trues, preds, digits=4))
        pprint(classification_report(trues, preds, output_dict=True))
#
#
# if __name__ == "__main__":
#     pc = ParentsClassifier()
#     pc.eval()