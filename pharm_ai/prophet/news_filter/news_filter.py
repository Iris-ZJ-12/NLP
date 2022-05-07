import os.path

import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import classification_report

from pharm_ai.prophet.news_filter.train import TitleTrainer, UncertainTitleTrainer
from simpletransformers.classification import ClassificationModel

h5 = os.path.join(os.path.dirname(__file__),'data.h5')

logger.add('result.log', filter=lambda record: record["extra"].get("task") == "newsfilter")
newsfilter_logger = logger.bind(task="newsfilter")

class TitleClassfier:
    def __init__(self, cuda_device=1, **kwargs):
        self.n_gpu=1
        if torch.cuda.is_available():
            self.use_cuda = True
            self.fp16=True
        else:
            self.use_cuda = False
            self.fp16 = False

        self.mapping = {0: '医药', 1: '非医药', 2: '非相关', 3:'0.5'}
        trainer = TitleTrainer()
        self.args = trainer.training_args.update(kwargs)
        self.model = ClassificationModel('bert', trainer.bm, cuda_device=cuda_device,
                                         num_labels=4, use_cuda=self.use_cuda,
                                         args=self.args)

    def predict(self, texts, return_label=True):
        predicted_labels, _ = self.model.predict(texts)
        if return_label:
            result = pd.Series(predicted_labels).map(self.mapping).to_list()
        else:
            result = predicted_labels
        return result


    def eval(self):
        df = pd.read_hdf(h5, 'v6-1/test')
        true_labels = df['labels'].map(self.mapping)
        predicted_labels = self.predict(df['text'].tolist())
        report_table = classification_report(true_labels, predicted_labels, digits=4)
        report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
        print(report_table)
        newsfilter_logger.info("Model 1 evaluate result: {}", report_dict)

class UncertainTitleClassifier:
    def __init__(self, cuda_device=1, **kwargs):
        self.mapping={0: '医药', 1: '非医药'}
        trainer = UncertainTitleTrainer()
        self.args = trainer.training_args.update(kwargs)
        self.model = ClassificationModel('bert',trainer.bm, cuda_device=cuda_device, num_labels=2,
                                         use_cuda=True, args=trainer.training_args)

    def predict_paragraph(self, texts, return_label=False):
        """
        Predict paragraph labels (not article labels)
        :param texts: List[text: str]
        :return: List[label: ('医药' or '非医药')] if return_label=True
                 List[label: (0 or 1)] if return_label=False
        """
        predicted_raw, _ = self.model.predict(texts)
        if return_label:
            result = list(map(lambda x: self.mapping.get(x), predicted_raw))
            return result
        else:
            return predicted_raw

    def predict_article_batch(self, to_predict: pd.DataFrame):
        """
        Predict article labels (not paragraph labels)
        :param to_predict: pandas.DataFrame['ESID','text']
        :return: pandas.DataFrame['ESID','labels_predicted':['医药' or '非医药']]
        """
        texts = to_predict['text'].tolist()
        paragraph_predicted = self.predict_paragraph(texts)
        to_predict['labels_predicted'] = paragraph_predicted
        res_df = (to_predict.groupby('ESID')['labels_predicted'].min()
                  .map(lambda x: self.mapping.get(x)).reset_index())
        return res_df

    def predict_article(self, to_predict):
        """
        :param list[str] to_predict: Each paragraph of an article
        :return: Predict result. '医药' or '非医药'.
        :rtype: str.
        """
        paragraph_predicted = self.predict_paragraph(to_predict)
        res = self.mapping.get(min(paragraph_predicted))
        return res

    def eval(self, option='article'):
        """
        Evaluate for paragraph or article, or both.
        :param option: str ('article', 'paragraph' or 'both')
        :return: Print classification report.
        """
        df = pd.read_hdf(h5, 'v6-2/test').rename(columns={'content':'paragraph'})
        if option in ['article','both']:
            article_predicted = self.predict_article(df)
            article_true = df.groupby('ESID')['labels'].min().map(lambda x: self.mapping.get(x)).reset_index()
            res_df = pd.merge(article_true, article_predicted, on='ESID')
            trues = res_df['labels'].tolist()
            predicts = res_df['labels_predicted'].tolist()
            report_table = classification_report(trues, predicts, digits=4)
            report_dict = classification_report(trues, predicts, digits=4, output_dict=True)
            print("Evaluate for articles:\n",report_table)
            newsfilter_logger.info("Model 2 evaluate result (articles): {}", report_dict)
        if option in ['paragraph','both']:
            trues = df['labels'].map(lambda x:self.mapping.get(x)).tolist()
            predicts = self.predict_paragraph(df['text'].values.tolist(),True)
            report_table = classification_report(trues, predicts, digits=4)
            report_dict = classification_report(trues, predicts, digits=4, output_dict=True)
            print("Evaluate for paragraphs:\n", report_table)
            newsfilter_logger.info("Model 2 evaluate result (paragraph): {}", report_dict)

class GroceryClassifier:
    def __init__(self, name="simple_model"):
        from tgrocery import Grocery
        self.model = Grocery(name)
        self.model.load()
        self.eval_df = pd.read_hdf(h5, 'v6-3/test')
        self.mapping = {1:"医药", 2:"非医药"}

    def eval(self):
        eval_df = self.eval_df
        eval_df['label'] = eval_df['label'].map(self.mapping)
        eval_src = eval_df[['label','text']].apply(tuple, axis=1).to_list()
        res = self.model.test(eval_src)
        report_table = classification_report(res.true_y, res.predicted_y, digits=4)
        print(report_table)
        report_dict = classification_report(res.true_y, res.predicted_y, digits=4, output_dict=True)
        newsfilter_logger.info("SVM model evalate result: {}", report_dict)

    def predict(self, to_predicts: list):
        result = [self.model.predict(text).predicted_y for text in to_predicts]
        return result

class NewsFilter:
    def __init__(self, cuda_device=1):
        self.title_classifier = TitleClassfier(cuda_device=cuda_device)
        self.uncertain_classifier = UncertainTitleClassifier(cuda_device=cuda_device)

    def eval(self):
        eval_df = pd.read_hdf(h5, 'v6-2/test')
        trues = eval_df.groupby('ESID')['labels'].min().map(self.title_classifier.mapping).reset_index()
        res = self.predict(eval_df)
        res_df = res.merge(trues, on='ESID')
        # generate report table
        trues_li = res_df['labels'].to_list()
        predicts_li = res_df['predict_labels'].to_list()
        report_table = classification_report(trues_li, predicts_li, digits=4)
        print(report_table)
        report_dict = classification_report(trues_li, predicts_li, digits=4, output_dict=True)
        newsfilter_logger.info("Both model evaluate result: {}", report_dict)

    def predict_batch(self, to_predict: pd.DataFrame):
        """
        Use both model to predict news_filter_labels.
        :param to_predict: pandas.DataFrame[ESID, title, text], each row in paragraph level.
        :return: result: pandas.DataFrame[predict_labels1, predict_labels2, predict_labels], each row in article level.
        """
        # use model1 to predict
        to_predicts1 = to_predict[['ESID', 'title']].drop_duplicates()
        predicts1 = self.title_classifier.predict(to_predicts1['title'].to_list())
        predicts_res1 = pd.Series(predicts1, index=to_predicts1.index, name='predict_labels1')
        res1 = to_predicts1.join(predicts_res1)
        # use model2 to predict
        to_predicts2 = to_predict[to_predict['ESID'].isin(res1[res1['predict_labels1'] == '0.5']['ESID'])][
            ['ESID', 'text']]
        predicts2 = self.uncertain_classifier.predict_article(to_predicts2)
        # merge predicts
        res = res1.merge(predicts2.rename(columns={'labels_predicted': 'predict_labels2'}), on='ESID', how='left')
        res['predict_labels'] = res['predict_labels2'].where(res['predict_labels2'].notna(), res['predict_labels1'])
        return res

    def predict(self, title, paragraphs):
        """
        :param str title: Article title.
        :param list[str] paragraphs: Paragraphs for an article.
        :return: Predict label for an article.
        :rtype: str

        Predict label for one article.
        """
        predict_res1 = self.title_classifier.predict([title])[0]
        if predict_res1 == '0.5':
            to_predict2 = ' '.join(paragraphs)
            predict_res2 = self.uncertain_classifier.predict_article([to_predict2])
            res = predict_res2
        else:
            res = predict_res1
        return res

if __name__ == '__main__':
    x=NewsFilter()
    x.eval()
else:
    logger.remove()