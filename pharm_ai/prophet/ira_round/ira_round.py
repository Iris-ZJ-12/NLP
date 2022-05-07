import logging
from time import time
import pandas as pd
import numpy as np
from simpletransformers.seq2seq import Seq2SeqModel
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.prophet.ira_round.dt import train_test
from pharm_ai.util.utils import Utilfuncs as u
from sklearn.metrics import accuracy_score


class IraRound:
    def __init__(self, cuda_device=1):
        f = cfp.project_dir + '/prophet/ira_round/'
        od = f + 'outputs/20201124-2/'
        cd = f + 'cache/20201124-2/'
        bm = f + 'best_model/20201124-2/'
        self.labels = ['并购', '增发', 'IPO', 'Pre-IPO', 'R轮', 'I轮', 'H轮',
                       'G轮', 'F轮', 'E7轮', 'E6轮', 'E5轮', 'E4轮', 'E3轮',
                       'E2轮', 'E1轮', 'E+轮', 'Pre-E+轮', 'E轮', 'Pre-E轮',
                       'D7轮', 'D6轮', 'D5轮', 'D4轮', 'D3轮', 'D2轮', 'D1轮',
                       'D+轮', 'Pre-D+轮', 'D轮', 'Pre-D轮', 'C7轮', 'C6轮',
                       'C5轮', 'C4轮', 'C3轮', 'C2轮', 'C1轮', 'C++轮', 'C+轮',
                       'Pre-C+轮', 'C轮', 'Pre-C轮', 'B7轮', 'B6轮', 'B5轮', 'B4轮',
                       'B3轮', 'B2轮', 'Pre-B2轮', 'B1轮', 'B++轮', 'B+轮', 'Pre-B+轮',
                       'B轮', 'Pre-B轮', 'A7轮', 'A6轮', 'A5轮', 'A4轮', 'A3轮', 'A2轮',
                       'A1轮', 'A++轮', 'A+轮', 'Pre-A+轮', 'A轮', 'Pre-A轮', '种子轮',
                       '天使轮', '战略融资', '其他']
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": 180,
            "learning_rate": 4e-5,
            "train_batch_size": 20,
            "num_train_epochs": 20,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_during_training": True,
            "evaluate_generated_text": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": False,
            "max_length": 10,
            "manual_seed": 4,
            'output_dir': od,
            'cache_dir': cd,
            'best_model_dir': bm}

        # Initialize model
        self.model = Seq2SeqModel('bert', cuda_device=cuda_device, encoder_decoder_name=bm, args=model_args)

    def prep_input(self, paras, titles):
        texts = []
        for p, t in zip(paras, titles):
            txt = f'title: {str(t)} paragraph: {str(p)}'
            texts.append(txt)
        return texts

    def predict(self, texts):
        preds = self.model.predict(texts)
        preds_ = []
        for p in preds:
            preds_.append(self.refine_label(p))
        return preds_

    def predict_api(self, paras, titles):
        texts = self.prep_input(paras, titles)
        res = self.predict(texts)
        return res

    # def refine_label(self, label):
    #     label = label.replace(' ', '').upper().replace('PRE-', 'Pre-')
    #     tmps = []
    #     for l in self.labels:
    #         if l in label:
    #             tmps.append(l)
    #     if tmps:
    #         label = max(tmps)
    #     return label

    def refine_label(self, label):
        label = label.replace(' ', '').upper().replace('PRE-', 'Pre-')
        tmps = []
        for i, l in enumerate(self.labels):
            if l in label:
                tmps.append(l)
        i = -1
        if tmps:
            label = max(tmps)
            i = tmps.index(max(tmps))
        return [label, i]

    def eval(self, texts, excel_path=None):

        preds = self.predict(texts)

        df = pd.DataFrame({'input_text': test['input_text'].tolist(),
                           'preds': preds, 'actuals': trues})
        df['preds'] = df['preds'].apply(lambda x: x[0].strip())
        df['actuals'] = df['actuals'].apply(lambda x: x.strip())
        df['if_correct'] = np.where(df['preds'] == df['actuals'], 1, 0)
        print("accuracy: %d/%d=%0.4f"%(df['actuals'].eq(df['preds']).sum(), df.shape[0],
              accuracy_score(df['actuals'], df['preds'])))
        if excel_path:
            u.to_excel(df, excel_path, 'result')
        return df


if __name__ == '__main__':
    p = IraRound()
    train, test = train_test()
    trues = test['target_text'].tolist()
    texts = test['input_text'].tolist()
    x = 'ira_round-test_result-20201124-2.xlsx'
    p.eval(texts, excel_path=None)

