# -*- coding: utf-8 -*-
from simpletransformers.ner import NERModel
import torch
from pharm_ai.util.utils import Utilfuncs as u
import pandas as pd
from time import time
from pprint import pprint
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.util.sm_util import SMUtil
from pharm_ai.util.prepro import Prepro
from pharm_ai.prophet.ira_ner.dt import dt20201120_1
torch.cuda.empty_cache()


class IraNER:
    def __init__(self, len_threshold=900, cuda_device=1):
        if torch.cuda.is_available():
            use_cuda = True
        else:
            use_cuda = False

        od = cfp.project_dir + '/prophet/ira_ner/outputs/20201201/'
        cd = cfp.project_dir + '/prophet/ira_ner/cache/20201201/'
        bm = cfp.project_dir + '/prophet/ira_ner/best_model/20201201/'
        args = {'fp16': False, 'reprocess_input_data': True, 'max_seq_length': 190,
                'n_gpu': 1, 'use_multiprocessing': False, 'output_dir': od,
                'cache_dir': cd, 'best_model_dir': bm}
        labels = ['b-financed#', 'i-financed#', 'b-investee', 'i-investee', 'O']
        self.model = NERModel('bert', bm, use_cuda=use_cuda, cuda_device=cuda_device,
                              labels=labels, args=args)
        self.pp = Prepro()
        self.len_threshold = len_threshold

    def predict(self, texts):
        labeled_texts = []
        for text in texts:
            text = str(text)
            labeled_text = SMUtil.ner_predict(text, self.model, self.pp,
                                              len_threhold=self.len_threshold)
            labeled_texts.append(labeled_text)
        return labeled_texts

    def to_xlsx(self):
        texts, article_ids = dt20201120_1()
        raw = self.predict(texts)
        x = 'ira_ner-preds-20201120.xlsx'
        SMUtil.ner2xlsx(raw, article_ids, x, 'result', ['ira_ner'])

    def eval(self):
        h5 = 'train_test_20201201.h5'
        test_df = pd.read_hdf(h5, 'test')
        eval_result, raw_outputs, predicts = self.model.eval_model(test_df)
        pprint(eval_result)


if __name__ == '__main__':

    i = IraNER()
    i.eval()