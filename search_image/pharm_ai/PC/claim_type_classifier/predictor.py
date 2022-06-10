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
f = cfp.project_dir + '/patent_claims/claim_type_classifier/'
od = f + 'outputs/20200824/'
cd = f + 'cache/20200824/'
bm = f + 'best_model/20200824/'


class ClaimTypeClassifier:

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
                                    num_labels=22, use_cuda=self.use_cuda,
                                    args={'reprocess_input_data': True,
                                          'use_cached_eval_features': False,
                                          'no_cache':True,
                                          'overwrite_output_dir': True,
                                          'n_gpu': self.n_gpu,
                                          'fp16': self.fp16,
                                          # 'gradient_accumulation_steps': 20,
                                          'learning_rate': 8e-6,
                                          'use_multiprocessing': False,
                                          'output_dir': bm, 'cache_dir': cd,
                                          'best_model_dir': bm})

    def predict(self, texts):
        predicted_labels, raw_outputs = self.model.predict(texts)
        predicted_labels = predicted_labels.tolist()
        # probabilities = softmax(raw_outputs, axis=1)
        # probabilities = np.max(probabilities, axis=1).tolist()
        claim_type_mapper = {'包材': 0, '分析方法': 1, '给药装置': 2, '化合物': 3, '晶型': 4,
                             '其他': 5, '前药': 6, '溶剂化物': 7, '细胞': 8, '序列': 9, '盐': 10,
                             '医疗器械': 11, '医药用途': 12, '医药中间体': 13, '载体': 14,
                             '诊断试剂': 15, '酯': 16, '制备方法': 17, '制剂': 18,
                             '制药设备': 19, '组合物': 20, '#': 21}
        claim_type_mapper = {y: x for x, y in claim_type_mapper.items()}
        # df = pd.DataFrame({'predicted_labels': predicted_labels,
        #                    'probabilities': probabilities})
        df = pd.DataFrame({'predicted_labels': predicted_labels})
        df['predicted_labels'] = df['predicted_labels'].map(claim_type_mapper)
        predicted_labels = df['predicted_labels'].tolist()
        return predicted_labels

    def eval(self):
        train1, test1, train2, test2, train3, test3 = train_test0824()
        texts = test2[0].tolist()
        preds = self.predict(texts)
        claim_type_mapper = {'包材': 0, '分析方法': 1, '给药装置': 2, '化合物': 3, '晶型': 4,
                             '其他': 5, '前药': 6, '溶剂化物': 7, '细胞': 8, '序列': 9, '盐': 10,
                             '医疗器械': 11, '医药用途': 12, '医药中间体': 13, '载体': 14,
                             '诊断试剂': 15, '酯': 16, '制备方法': 17, '制剂': 18,
                             '制药设备': 19, '组合物': 20, '#': 21}
        claim_type_mapper = {y: x for x, y in claim_type_mapper.items()}
        test2[1] = test2[1].map(claim_type_mapper)
        trues = test2[1].tolist()
        f = 'patent-claims-ind._types-test-20200824.xlsx'
        df = pd.DataFrame({'texts': texts, 'predicted_labels': preds, 'actual_labels': trues})
        u.to_excel(df, f, 'result')
        print(classification_report(trues, preds, digits=4))
        pprint(classification_report(trues, preds, output_dict=True))
#
#
# if __name__ == "__main__":
#     ctc = ClaimTypeClassifier()
#     ctc.eval()