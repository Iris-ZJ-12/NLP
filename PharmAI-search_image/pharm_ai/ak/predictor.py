from functools import cached_property
from itertools import compress

import Levenshtein
import ijson
import json

from tqdm import tqdm

from pharm_ai.ak.dt import AkPreprocessor, HTMLProcessor
from pharm_ai.ak.utils import *
from simpletransformers.classification import ClassificationModel
from simpletransformers.t5 import T5Model


class AkT5Predictor(AkT5ModelBase):

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
        prepro_class = AkPreprocessor.get_preprocessor_class(self.version)
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


class AkClassificationPredictor(AkClassificationModelBase):
    def __init__(self, version, cuda_device=-1, task_id=0, **kwargs):
        self.version = version
        self.cuda_device = cuda_device
        super(AkClassificationPredictor, self).__init__(
            version, getattr(self.prepro, 'prefix', None), cuda_devices=[self.cuda_device],
            task_id=task_id
        )
        self.model_args.update_from_dict(kwargs)
        self.sm_model = ClassificationModel('bert', self.best_model_dir, args=self.model_args,
                                            cuda_device=get_cuda_device(cuda_device))

    @cached_property
    def prepro(self):
        prepro_class = AkPreprocessor.get_preprocessor_class(self.version)
        return prepro_class()

    def eval(self, saving_excel=None, compare_eval=True):
        eval_df = self.prepro.get_from_h5('eval')
        eval_res, res_eval_df = super().eval(
            eval_df, self.prepro.col_mapping.keys() if hasattr(self.prepro, 'col_mapping') else None)
        if saving_excel:
            res_eval_df = res_eval_df.applymap(remove_illegal_char)
            res_eval_df.to_excel(saving_excel, sheet_name=self.sub_project_name or 'eval_res', index=False)
        return eval_res


class T5Predictor(AkT5Predictor):
    init_mapping = {
        'v1.0': {'version': 'v1.4', 'task_id': 0, 'eval_batch_size': 25, 'silent': True,
                 'max_length': 40},
    }

    def __init__(self, version, cuda_device):
        init_paras = self.init_mapping.get(version)
        if not init_paras:
            raise ValueError('Version %s is not supported.' % version)
        super().__init__(cuda_device=cuda_device, **init_paras)

    def predict(self, html, return_location=False):
        html_processor = HTMLProcessor(html)
        filter_task = self.prepro.CLASSIFICATION_TASK[0]
        ner_task = self.prepro.NER_TASK[0]
        paras = html_processor.remove_tags(remove_style_element=True, len_threshold=100)
        filter_res = super().predict([filter_task] * len(paras), paras)
        to_refine_paras = [p for p, r in zip(paras, filter_res) if r == '1']
        results = super().predict([ner_task] * len(to_refine_paras), to_refine_paras)
        results = [self.refine_entity(t, r) for t, r in zip(to_refine_paras, results)]
        if not return_location:
            return results
        else:
            final_res = {'html': html_processor.to_string(), 'results': {}}
            for t in set(results):
                html_text = list(set(p for indp, p in enumerate(to_refine_paras) if results[indp] == t))
                xpaths = [{'xpath': path_, 'raw': txt} for txt in html_text for path_ in html_processor.locate_element(txt)]
                final_res['results'][t] = xpaths
            return final_res

    @staticmethod
    def refine_entity(input_text: str, predict_text: str, score_threshold=0.5):
        if predict_text in input_text:
            return predict_text
        else:
            score = Levenshtein.ratio(input_text, predict_text)
            if score < score_threshold:
                return input_text
            else:
                ops = Levenshtein.editops(input_text, predict_text)
                strip_strs = ''.join(input_text[op[1]] for op in ops if op[0] == 'delete')
                return input_text.strip(strip_strs)

    def close_multiprocessing(self):
        self.sm_model.args.use_multiprocessing_for_evaluation = False
        self.sm_model.args.use_multiprocessing = False
        self.sm_model.args.use_multiprocessed_decoding = False

    def pre_annotate(self, data_json='data/to_labels_1027.json', saving_json='data/pre_annotated_1027.json'):
        pre_anno_data = []
        def refine_xpath_func(xp):
            return ('/' + '/'.join(
                x.split('[')[0] if x.startswith('div') or x.startswith('span') else (x + '/' if x == 'table' else x)
                for ix, x in enumerate(xp.split('/'))
                if ix == 0 or ix > 2) + '[1]')
        with open(data_json) as f:
            for i, t in enumerate(tqdm(ijson.items(f, 'item'))):
                dat = {'data': t['data']}
                res = self.predict(t['data']['content'], return_location=True)
                dat['data']['content'] = res['html']
                dat['predictions'] = [{
                    'model_version': 'v1.0',
                    'result': [
                        {
                            'from_name': 'ner',
                            'to_name': 'text',
                            'value': {
                                'start': refine_xpath_func(each_loc['xpath']),
                                'end': refine_xpath_func(each_loc['xpath']),
                                'text': ent_res,
                                'htmllabels': ['product'],
                                'startOffset': each_loc['raw'].find(ent_res),
                                'endOffset': each_loc['raw'].find(ent_res) + len(ent_res)
                            },
                            'type': 'hypertextlabels'
                        }
                        for ent_res, ent_loc in res['results'].items()
                        for each_loc in ent_loc
                    ]
                }]
                pre_anno_data.append(dat)
        with open(saving_json, 'w') as sf:
            json.dump(pre_anno_data, sf, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    predictor = T5Predictor('v1.0', 0)
    predictor.pre_annotate()
