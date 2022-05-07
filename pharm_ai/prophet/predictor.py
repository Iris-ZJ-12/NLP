from __future__ import annotations

from pharm_ai.prophet.utils import *
from simpletransformers.t5 import T5Model
from simpletransformers.classification import ClassificationModel
from pharm_ai.prophet.dt import ProphetPreproessor
from functools import cached_property
from sklearn.metrics import classification_report, f1_score
import pandas as pd
from mart.sm_util.sm_util import eval_ner_v2
from itertools import compress
from datetime import datetime


class ProphetT5Predictor(ProphetT5ModelBase):

    def __init__(self, version, cuda_device=-1, task_id=0, **kwargs):
        self.version = version
        self.cuda_device = cuda_device
        super().__init__(version, [self.cuda_device], task_id=task_id)
        self.model_args.update_from_dict(kwargs)
        self.update_eval_args()
        self.sm_model = T5Model('mt5', self.best_model_dir, args=self.model_args,
                                cuda_device=get_cuda_device(cuda_device))

    @cached_property
    def prepro(self):
        prepro_class = ProphetPreproessor.get_preprocessor_class(self.version)
        return prepro_class()

    def eval(self, eval_df: pd.DataFrame = None, saving_excel=None, compare_eval=True,
             compared_model: T5Model = None, return_dataframe=False):
        to_eval_total = eval_df if eval_df is not None else self.prepro.get_from_h5('eval')
        to_eval_cls = to_eval_total[to_eval_total['prefix'].isin(self.prepro.CLASSIFICATION_TASK)]
        hist_eval_res = {}
        cls_eval_res = {}
        if not to_eval_cls.empty:
            to_eval_cls['predicts'] = self.predict(to_eval_cls['prefix'].tolist(),
                                                   to_eval_cls['input_text'].tolist())
            cls_eval_res = self._eval_cls_tasks(to_eval_cls)
            if compare_eval:
                compared_model = compared_model or self._get_history_t5_model(self.prepro.CLASSIFICATION_TASK)
                to_predict = (to_eval_cls['prefix'] + ': ' + to_eval_cls['input_text']).tolist()
                to_eval_cls['history_predicts'] = compared_model.predict(to_predict)
                cls_hist_eval_res = self._eval_cls_tasks(
                    to_eval_cls[['prefix', 'input_text', 'target_text', 'history_predicts']].rename(
                        columns={'history_predicts': 'predicts'}))
                hist_eval_res = {**cls_hist_eval_res}
        to_eval_ner = to_eval_total[to_eval_total['prefix'].isin(self.prepro.NER_TASK)]
        ner_eval_res = {}
        if not to_eval_ner.empty:
            ner_eval_res = self._eval_ner_tasks(to_eval_ner)
            if compare_eval:
                compared_model = compared_model or self._get_history_t5_model(self.prepro.NER_TASK)
                to_predict = (to_eval_ner['prefix'] + ': ' + to_eval_ner['input_text']).tolist()
                to_eval_ner['history_predicts'] = compared_model.predict(to_predict)
                ner_hist_eval_res = self._eval_ner_tasks(
                    to_eval_ner, compared_model)
                hist_eval_res = {**hist_eval_res, **ner_hist_eval_res}
        if saving_excel:
            self._saving_eval_result(to_eval_cls, to_eval_ner, saving_excel)
        eval_res = {**cls_eval_res, **ner_eval_res}
        if hist_eval_res:
            eval_res = {'current_version_result': eval_res,
                        'history_version_result': hist_eval_res}
        if return_dataframe:
            return self._eval_result_to_dataframe(eval_res, compare_eval)
        return eval_res

    def _eval_result_to_dataframe(self, eval_res: dict, compare_history: bool):
        if compare_history:
            res_df = pd.DataFrame({'version': ver, 'subproject': sub, 'metric': k, 'value': score}
                                  for ver, res_ver in eval_res.items()
                                  for sub, sub_res in res_ver.items()
                                  for k, score in sub_res.items())
            reformat_df = res_df.pivot_table('value', 'subproject', ['version', 'metric'])
            diff_df = reformat_df['current_version_result'] - reformat_df['history_version_result']
            diff_df.columns = diff_df.columns.map(lambda c: ('diff', c)).set_names(['version', 'metric'])
            total_df = pd.concat([reformat_df, diff_df], axis=1)
        else:
            res_dt = pd.DataFrame({'subproject': sub, 'metric': k, 'value': score}
                                  for sub, sub_res in eval_res.items()
                                  for k, score in sub_res.items())
            total_df = res_dt.pivot_table('value', 'subproject', 'metric')
        print(total_df)
        return total_df

    def _get_history_t5_model(self, tasks):
        """v2 PseuT5Model"""
        model = PseudoT5Model(tasks, self.cuda_device)
        model.predict_config['fillna'] = getattr(self.prepro, 'ORG_NULL')
        return model

    def _saving_eval_result(self, to_eval_cls, to_eval_ner, saving_excel):
        to_eval_ner['predicts'] = self.predict(to_eval_ner['prefix'].tolist(),
                                               to_eval_ner['input_text'].tolist())
        with pd.ExcelWriter(saving_excel) as writer:
            for cls_task in self.prepro.CLASSIFICATION_TASK:
                task_df = to_eval_cls[to_eval_cls['prefix'] == cls_task].applymap(remove_illegal_char)
                if not task_df.empty:
                    saving_task_df = task_df.rename(columns={'input_text': 'text', 'target_text': 'trues'}).drop(
                        columns=['prefix'])
                    saving_task_df.to_excel(writer, sheet_name=cls_task, index=False)
            for ner_task in self.prepro.NER_TASK:
                task_df = to_eval_ner[to_eval_ner['prefix'] == ner_task].applymap(remove_illegal_char)
                if not task_df.empty:
                    saving_task_df = task_df.rename(columns={'input_text': 'text', 'target_text': 'trues'}).drop(
                        columns=['prefix'])
                    saving_task_df.to_excel(writer, sheet_name=ner_task, index=False)

    def _eval_cls_tasks(self, to_eval_cls):
        cls_eval_res = {}
        for cls_task in self.prepro.CLASSIFICATION_TASK:
            task_df = to_eval_cls[to_eval_cls['prefix'] == cls_task]
            r = f1_score(task_df['target_text'], task_df['predicts'], average='macro')
            print(cls_task + ':\n',
                  classification_report(task_df['target_text'], task_df['predicts'],
                                        digits=4, zero_division=0))
            cls_eval_res[cls_task] = {'f1': r}
        return cls_eval_res

    def _eval_ner_tasks(self, to_eval_ner, ner_t5model=None):
        """
        :param to_eval_ner: DataFrame [prefix, input_text, target_text].
        :param ner_t5model: T5Model.
        :return: acc, rouge, edit_distance score of each prefix.
        """
        ner_t5model = ner_t5model or self.sm_model
        eval_acc = eval_ner_v2(to_eval_ner, ner_t5model, delimiter=self.prepro.seperator)
        to_predicts = (to_eval_ner['prefix'] + ': ' + to_eval_ner['input_text']).tolist()
        predicts = ner_t5model.predict(to_predicts)
        eval_res = {}
        for k in to_eval_ner['prefix'].unique():
            eval_res[k] = {'acc': eval_acc[k]}
            cond = to_eval_ner['prefix'] == k
            trues = to_eval_ner[cond]['target_text'].tolist()
            preds = list(compress(predicts, cond))
            rouge_score = get_rouge_f1_scores(trues, preds)
            eval_res[k] = {**eval_res[k], **rouge_score}
        return eval_res


class ProphetClassificationPredictor(ProphetClassificationModelBase):
    def __init__(self, version, cuda_device=-1, task_id=0, **kwargs):
        self.version = version
        self.cuda_device = cuda_device
        self.sub_project_name = self.prepro.prefix
        super(ProphetClassificationPredictor, self).__init__(
            version, self.prepro.prefix, cuda_devices=[self.cuda_device],
            task_id=task_id
        )
        self.model_args.update_from_dict(kwargs)
        self.sm_model = ClassificationModel('bert', self.best_model_dir, args=self.model_args,
                                            cuda_device=get_cuda_device(cuda_device))

    @cached_property
    def prepro(self):
        prepro_class = ProphetPreproessor.get_preprocessor_class(self.version)
        return prepro_class()

    def eval(self, saving_excel=None, compare_eval=True):
        eval_df = self.prepro.get_from_h5('eval')
        eval_res, res_eval_df = super().eval(
            eval_df, self.prepro.col_mapping.keys() if hasattr(self.prepro, 'col_mapping') else None)
        if compare_eval:
            print('History eval:')
            eval_hist_res, eval_hist_df = self.eval_hist(eval_df)
            res_eval_df['v2_predict_labels'] = eval_hist_df['predict_labels']
            eval_res = {'current_version_result': eval_res, 'v2_version_result': eval_hist_res}
        if saving_excel:
            res_eval_df = res_eval_df.applymap(remove_illegal_char)
            res_eval_df.to_excel(saving_excel, sheet_name=self.sub_project_name, index=False)
        return eval_res

    def eval_hist(self, eval_df: pd.DataFrame):
        funcs = {'news_filter': self._eval_history_news_filter,
                 'ira_filter': self._eval_history_ira_filter,
                 'org_filter': self._eval_history_org_filter}
        return funcs[self.sub_project_name](eval_df)

    def get_pseudo_t5_model(self):
        return PseudoT5Model([self.sub_project_name], self.cuda_device)

    def _eval_history_news_filter(self, eval_df: pd.DataFrame):
        pseu = self.get_pseudo_t5_model()
        predictor = pseu._news_filter_init()
        true_labels = eval_df['labels'].tolist()
        predict_labels = pseu._news_filter_predict(predictor, eval_df['text'].tolist())
        predict_labels = list(map(self.prepro.col_mapping.get, predict_labels))
        eval_res = f1_score(true_labels, predict_labels, average='macro')
        res_eval_df = eval_df.rename(columns={'labels': 'true_labels'}).assign(predict_labels=predict_labels)
        print(classification_report(true_labels, predict_labels, target_names=list(self.prepro.col_mapping.keys()),
                                    digits=4, zero_division=0))
        return eval_res, res_eval_df

    def _eval_history_ira_filter(self, eval_df: pd.DataFrame):
        pseu = self.get_pseudo_t5_model()
        predictor = pseu._ira_filter_init()
        true_labels = eval_df['labels'].tolist()
        predict_labels = pseu._ira_filter_predict(predictor, eval_df['text'].tolist())
        predict_labels = list(map(int, predict_labels))
        eval_res = f1_score(true_labels, predict_labels, average='macro')
        res_eval_df = eval_df.rename(columns={'labels': 'true_labels'}).assign(predict_labels=predict_labels)
        print(classification_report(true_labels, predict_labels, digits=4, zero_division=0))
        return eval_res, res_eval_df

    def _eval_history_org_filter(self, eval_df: pd.DataFrame):
        pseu = self.get_pseudo_t5_model()
        predictor = pseu._org_filter_init()
        true_labels = eval_df['labels'].tolist()
        predict_labels = pseu._org_filter_predict(predictor, eval_df['text'].tolist())
        predict_labels = list(map(int, predict_labels))
        eval_res = f1_score(true_labels, predict_labels, average='macro')
        res_eval_df = eval_df.rename(columns={'labels': 'true_labels'}).assign(predict_labels=predict_labels)
        print(classification_report(true_labels, predict_labels, digits=4, zero_division=0))
        return eval_res, res_eval_df


class NewsFilterPredictor(ProphetClassificationPredictor):
    def __init__(self, cuda_device):
        super(NewsFilterPredictor, self).__init__(
            'v3.3.0', cuda_device, task_id=0, eval_batch_size=20, silent=True
        )
        self.mapping = reverse_dict_key_value(self.prepro.col_mapping)

    def predict(self, title="", paragraphs: list = None):
        fulltext = ' '.join(paragraphs) if paragraphs else ''
        to_predicts = self.prepro.concat_title_fulltext(title, fulltext)
        res, prob = super(NewsFilterPredictor, self).predict([to_predicts])
        return list(map(self.mapping.get, res))


class IraFilterPredictor(ProphetClassificationPredictor):
    datetime_format = "%Y-%m-%d"

    def __init__(self, cuda_device):
        super(IraFilterPredictor, self).__init__(
            'v3.3.1', cuda_device, task_id=2, eval_batch_size=50, max_seq_length=256,
            silent=True
        )

    def predict(self, date: datetime = None, para: list = None, len_threshod=50):
        para = para or []
        if para:
            drop_short = [(ind, self.prepro.add_date_to_paragraph(date, p) if date else p) for ind, p in enumerate(para)
                          if
                          len(p) > len_threshod]
            if not drop_short:
                return [0] * len(para)
            pred_ind, to_predicts = zip(*drop_short)
            pred_res, prob = super().predict(list(to_predicts))
            pred_ind_res = dict(zip(pred_ind, pred_res))
            res = [pred_ind_res.get(ind, 0) for ind, p in enumerate(para)]
            return res
        else:
            return []


class OrgFilterPredictor(ProphetClassificationPredictor):

    def __init__(self, cuda_device):
        super(OrgFilterPredictor, self).__init__(
            'v3.3.2', cuda_device, task_id=2, eval_batch_size=50, max_seq_length=256,
            silent=True
        )

    def predict(self, date: datetime = None, para: list = None, len_threshod=50):
        para = para or []
        if para:
            drop_short = [(ind, self.prepro.add_date_to_paragraph(date, p) if date else p) for ind, p in enumerate(para)
                          if len(p) > len_threshod]
            if not drop_short:
                return [0] * len(para)
            pred_ind, to_predicts = zip(*drop_short)
            pred_res, prob = super().predict(list(to_predicts))
            pred_ind_res = dict(zip(pred_ind, pred_res))
            res = [pred_ind_res.get(ind, 0) for ind, p in enumerate(para)]
            return res
        else:
            return []


class NerPredictor(ProphetT5Predictor):
    IRA = ['investee', 'round', 'amount']
    ORG = ['org_ner.c1', 'org_ner.c2', 'org_ner.c3', 'org_ner.f']
    CLS = ['c1', 'c2', 'c3', 'f']

    def __init__(self, cuda_device):
        super().__init__(
            'v3.6', cuda_device, task_id=1, eval_batch_size=48, max_seq_length=300, max_length=50,
            silent=True
        )

    def predict_ira(self, ira_filter_labels: list, paragraphs: list):
        if not ira_filter_labels or not paragraphs:
            return None, None, None
        if any(lb == 1 for lb in ira_filter_labels):
            prefixes, to_predicts = zip(
                *[(pref, para) for filter_label, para in zip(ira_filter_labels, paragraphs) if filter_label
                  for pref in self.IRA])
            results = super().predict(list(prefixes)[:3], list(to_predicts)[:3], separator=self.prepro.seperator,
                                      to_refine=False, sliding=False)
        else:
            # extract the first paragraph
            prefixes, to_predicts = self.IRA, [paragraphs[0]] * len(self.IRA)
            results = super(NerPredictor, self).predict(prefixes, to_predicts, separator=self.prepro.seperator,
                                                        to_refine=False, sliding=False)
        results = ['战略投资' if p == 'round' and r == '未披露' else r for p, r in zip(prefixes, results)]
        return tuple(results)

    def predict_org(self, news_filter_label: str, org_filter_labels: list, paragraphs: list):
        results = [{} for _ in paragraphs]
        if news_filter_label != "非相关":
            if not any(lb == 1 for lb in org_filter_labels):
                # extract the first paragraph
                org_filter_labels = org_filter_labels.copy()
                org_filter_labels[0] = 1
            selected_inds, prefixes, to_predicts = zip(
                *[(ind, pref, para) for ind, (filter_label, para) in enumerate(zip(org_filter_labels, paragraphs))
                  if filter_label == 1 for pref in self.ORG]
            )
            selected_results = super().predict(list(prefixes), list(to_predicts),
                                               separator=self.prepro.seperator, na_value=self.prepro.ORG_NULL)
            for ind, pref, res in zip(selected_inds, prefixes, selected_results):
                p = pref.split('.')[-1]
                if results[ind].get(p):
                    results[ind][p].extend(res.split(self.prepro.seperator))
                else:
                    results[ind][p] = res.split(self.prepro.seperator)
            self.remove_duplicated_cls_results(results)
        return results

    def remove_duplicated_cls_results(self, results):
        for each_res in results:
            prev = set()
            for k in self.CLS:
                v = set(each_res.get(k) or [])
                # also remove negative class symbol
                res = list(v - {getattr(self.prepro, 'ORG_NULL')} - prev)
                prev |= v
                if res:
                    each_res[k] = res
                elif v:
                    del each_res[k]


if __name__ == '__main__':
    predictor = NerPredictor(8)
    to_predict = ["8月16日，美柏医药生物（以下简称“美柏生物”）宣布完成数千万元A轮融资，本轮融资由雅惠投资领投，"
                  "腾云大健康以及老股东丹麓资本持续跟投，星汉资本担任独家财务顾问。",
                  "亿欧大健康2月9日讯，缔佳医疗（美立刻）完成数千万元战略融资，本轮由上市公司欧普康视旗下中合欧普医疗健康基金投资，"
                  "WinX Capital凯乘资本担任财务顾问。缔佳医疗由清华大学、北京大学口腔医学团队联合创立，"
                  "致力于为牙齿隐形矫正提供稳妥、高效的全套解决方案，是一家口腔专用3D打印光敏树脂自研自产公司，"
                  "拥有自主生产多层齿科膜片，并建成中国唯一的自动化智能制造基地。凭借关键原材料核心技术以及3D打印等自动化工艺，"
                  "缔佳医疗产品“美立刻隐形矫治器”于2015年面市并迅速占领市场，截至2020年底已为4万余名患者提供了隐形矫治服务。"
                  "本轮融资后，欧普康视将与缔佳医疗在口腔领域展开深度合作，涵盖产业链、终端连锁及市场渠道等。"
                  "此外，经过投资与招商协同，缔佳医疗“隐形矫治器全产业链智能制造+研发项目”已落地合肥高新区，项目计划投资5亿元，"
                  "建设包含隐形矫治器全自动化生产线、高分子齿科膜片生产线、口腔用光敏树脂生产线和口腔数字化产品研发中心的第二总部。"]
    res = predictor.predict_org('医药', [1, 1], to_predict)
    print(res)
