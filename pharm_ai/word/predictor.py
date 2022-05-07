import logging
from time import time
import pandas as pd
import numpy as np
from simpletransformers.seq2seq import Seq2SeqModel
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.word.dt import train_test
from pharm_ai.util.utils import Utilfuncs as u


class Predictor:
    def __init__(self):
        f = cfp.project_dir + '/word/'
        od = f + 'outputs/20201217/'
        cd = f + 'cache/20201217/'
        bm = f + 'best_model/20201217/'

        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": 180,
            "learning_rate": 4e-5,
            "use_multiprocessing": False,
            "max_length": 50,
            "manual_seed": 4,
            'output_dir': od,
            'cache_dir': cd,
            'best_model_dir': bm}

        # Initialize model
        self.model = Seq2SeqModel('bert', encoder_decoder_name=bm, args=model_args)

    def prep_input(self, paras, titles):
        texts = []
        for p, t in zip(paras, titles):
            txt = f'title: {str(t)} paragraph: {str(p)}'
            texts.append(txt)
        return texts

    def predict(self, texts):
        preds = self.model.predict(texts)
        preds_ = []
        for pred in preds:
            if pred == '| |':
                pred = '||'
            preds_.append(pred)
        return preds_

    def eval(self, texts, excel_path=None):
        preds = self.predict(texts)
        df = pd.DataFrame({'input_text': test['input_text'].tolist(),
                           'preds': preds, 'actuals': trues})
        df['preds'] = df['preds'].apply(lambda x: x.strip())
        df['actuals'] = df['actuals'].apply(lambda x: x.strip())
        df['if_correct'] = np.where(df['preds'] == df['actuals'], 1, 0)
        print(df['if_correct'].tolist())
        if excel_path:
            u.to_excel(df, excel_path, 'result')
        return df


if __name__ == '__main__':
    p = Predictor()
    train, test = train_test()
    trues = test['target_text'].tolist()
    texts = test['input_text'].tolist()
    x = 'word-test_result-20201217xlsx'
    p.eval(texts, excel_path=x)

