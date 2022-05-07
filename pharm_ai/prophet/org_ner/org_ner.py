from pharm_ai.prophet.org_ner.train import Trainer
from simpletransformers.ner import NERModel
from loguru import logger
from pprint import pprint
from seqeval.metrics import classification_report
from mart.sm_util.sm_util import hide_labels_arg, ner_predict, recover_ner_entity
from mart.prepro_util.prepro import Prepro
import sys

logger.add('result.log', filter=lambda record: record["extra"].get("task") == "orgner")
orgner_logger = logger.bind(task="orgner")

class OrgNER:
    def __init__(self, cuda_device=-1):
        self.trainer = Trainer()
        ner_model = NERModel(
            'bert', self.trainer.best_model_dir,
            use_cuda=True,
            cuda_device=cuda_device,
            args=self.trainer.model_args)
        self.model = hide_labels_arg(ner_model, option="predict")
        self.prepro = Prepro()

    def eval(self):
        # report phrase
        eval_df = self.trainer.get_eval_df()
        eval_result, raw_outputs, predicts = self.model.eval_model(eval_df, report=classification_report)
        report_dict = {k: v for k, v in eval_result.items() if k != 'report'}
        orgner_logger.success("Eval result (phrase): {}", report_dict)
        pprint(report_dict)
        print(eval_result['report'])

    def predict_ner(self, to_predict, return_raw=True):
        predict_f =lambda p: ner_predict(p, self.model, self.prepro)
        res = list(map(predict_f, to_predict))
        res_entity = recover_ner_entity(res, True)
        if return_raw:
            return res_entity, res
        else:
            return res_entity

    def predict(self, is_folding_head: bool, news_filter_label, org_filter_labels, paragraphs):
        """
        :param is_folding_head:
        :param org_filter_labels:
        :param paragraphs:
        :return:
        """
        select_org_ner, candidate_org_ner = [], []
        if org_filter_labels and is_folding_head and news_filter_label!='非相关':
            for org_filter_label, para in zip(org_filter_labels, paragraphs):
                if org_filter_label:
                    res1, res2 = self.predict_ner([para])
                    select_org_ner.append(res1[0])
                    candidate_org_ner.append(res2[0])
                else:
                    select_org_ner.append(dict())
                    candidate_org_ner.append(dict())
        return select_org_ner, candidate_org_ner

if __name__=='__main__':
    predictor = OrgNER()
    # to_predict=['LONDON, Nov 21, 2019/PRNewswire / --  Azeria Therapeutics (Azeria), a newly formed pioneer factor drug discovery company, today announced a £ 32 million Series B financing in which Syncona has committed £ 29.5 million alongside existing investor the CRT Pioneer Fund.',
    #             '成都思多科医疗科技有限公司（简称“思多科”）宣布完成数千万人民币的天使轮融资，由紫牛基金独家投资，探针资本担任独家融资顾问。',
    #             '&#10;纳百医疗表示，本次融资将为客户提供更好的全方位贴身服务体验和一站式解决方案，扩大生产经营规模，缓解供不应求的产能压力并满足订单及时交付的要求，进一步落实公司CDMO全产业链的长远规划；秉持业务区域划分和客户服务本地化及时响应原则，密切配合客户解决在早期研发设计、中期生产制造和后期检验检测等环节面临的各种难题，满足“多品种、小批量、高要求”行业需求特点；引进各类高科技人才和智能化设备，在全国范围内将打造华南、华北、华东以及西南生产基地形成“三主一次”市场格局。']
    to_predict=["高瓴资本是什么？高瓴资本、约印医疗基金、浪子韩亚跟投"]
    res, res_raw = predictor.predict_ner(to_predict)
    print(res)
else:
    logger.remove()


