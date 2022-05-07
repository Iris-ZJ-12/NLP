from simpletransformers.classification import ClassificationModel
from pharm_ai.prophet.org_filter.train import OrgFilterTrainer
from pharm_ai.prophet.org_filter.dt import OrgFilterPreprocessor
from pprint import pprint
from sklearn.metrics import classification_report, f1_score
import pandas as pd

class OrgFilter:
    def __init__(self, cuda_device=-1, version='v1', date='20201128'):
        self.preprocessor = OrgFilterPreprocessor(version=version)
        trainer = OrgFilterTrainer(version=version, date=date)
        self.model_args = trainer.modal_args
        self.model_args.update({'use_cached_eval_features': False})
        self.model = ClassificationModel('bert', trainer.bestmodel_dir, num_labels=2, args=self.model_args, cuda_device=cuda_device)

    def predict(self, is_folding_head: bool, texts):
        """
        :param bool is_folding_head: Whether the article is folding head.
        :param list[str] texts: Texts to predict.
        :return: Result labels (0 or 1) of each text.
        :rtype: list[str]
        """
        if is_folding_head and texts:
            res, _ = self.model.predict(texts)
            return res
        else:
            return []

    def eval(self):
        eval_df = pd.read_hdf(self.preprocessor.h5, self.preprocessor.h5_keys['test'])
        eval_result, _, _ = self.model.eval_model(eval_df, f1=f1_score)
        pprint(eval_result)
        preds = self.predict(eval_df['text'].astype(str).tolist())
        trues = eval_df['labels'].tolist()
        report_res = classification_report(trues, preds, digits=4)
        print(report_res)


if __name__ == '__main__':
    x=OrgFilter(cuda_device=0 ,version='v1-1', date='20201204')
    to_predict = ["疼痛领域具有明确的未被满足的临床需求，现有的镇痛药物以阿片类药物和非甾体抗炎药（NSAIDs）为主，但阿片类药物由于严重的成瘾性已引发全球性危机，NSAIDs则镇痛疗效弱，主要用于轻中度疼痛。目前临床亟需非阿片、长效、强效的镇痛产品。",
                  "血液病专科医疗集团陆道培医疗集团已完成超过 1 亿元人民币 B+轮股权融资，本轮融资由约印医疗基金领投，招商证券资本、软银中国资本、众合资本及老股东朴道医疗等跟投。",
                  "The round was led by Hoya Corporate Venture Capital Group, the corporate venture arm of Hoya Group and is part of a larger Series B Financing planned in early 2021 that will fund the company through approval of Juvene IOL pivotal FDA study."]
    # res = x.predict(to_predict)
    # pprint(res)
    x.eval()