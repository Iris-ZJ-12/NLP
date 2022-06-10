from datetime import datetime

import langid
import torch
from sklearn.metrics import classification_report

from mart.utils import conv_date2en
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.prophet.ira_filter.dt import train_test
from simpletransformers.classification import ClassificationModel

f = cfp.project_dir + '/prophet/ira_filter/'
od = f + 'outputs/20201123/'
cd = f + 'cache/20201123/'
bm = f + 'best_model/20201123/'

torch.cuda.empty_cache()


class IraFilter:
    def __init__(self, cuda_device=1):
        self.model = ClassificationModel('bert', bm,
                                         num_labels=2, use_cuda=True, cuda_device=cuda_device,
                                         args={'reprocess_input_data': True,
                                               'use_cached_eval_features': True,
                                               'overwrite_output_dir': True,
                                               'fp16': False,
                                               'n_gpu': 1,
                                               'use_multiprocessing': False,
                                               'no_cache': True,
                                               'output_dir': od, 'cache_dir': cd,
                                               'best_model_dir': bm})

    def prepro(self, paras, pub_dates):
        paras_with_dates = []
        for date, para in zip(pub_dates, paras):
            if not isinstance(date, datetime):
                date = datetime.strptime(date, "%Y-%m-%d")
            m = date.month
            d = date.day
            y = date.year
            lang = langid.classify(para)[0]
            para = para.strip()
            if lang == 'zh':
                if not para.endswith('。'):
                    para = para + '。'
                para = f'本文发布日期为{y}年{m}月{d}日。{para}'
            else:
                if not para.endswith('.'):
                    para = para + '.'
                para = f'This article was publish on {conv_date2en(m, d, y)}. {para}'
            paras_with_dates.append(para)
        return paras_with_dates

    def predict_(self, texts):
        if texts:
            preds, _ = self.model.predict(texts)
            return preds
        else:
            return []

    def predict(self, paras, pub_dates):
        paras_with_dates = self.prepro(paras, pub_dates)
        preds = self.predict_(paras_with_dates)
        return preds

    def eval20201123(self):
        train, test = train_test()
        trues = test['labels'].tolist()
        texts = test['text'].tolist()
        preds = self.predict_(texts)
        print(classification_report(trues, preds, digits=4))
        print('*'*100)
        # x = 'prophet-ira_filter-test-20201123.xlsx'
        # df = pd.DataFrame({'text': texts, 'actual_labels': trues, 'predicted_labels': preds})
        # u.to_excel(df, x, 'result')


if __name__ == '__main__':
    p = IraFilter()
    p.eval20201123()