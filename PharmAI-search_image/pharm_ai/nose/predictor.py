from functools import cached_property
from itertools import compress

from mart.parallel_util.parallize import Parallizer
from tqdm import tqdm

from pharm_ai.nose.dt import NosePreprocessor
from pharm_ai.nose.utils import *
from simpletransformers.classification import ClassificationModel
from simpletransformers.t5 import T5Model
from pharm_ai.nose.utils import Indication


class NoseT5Predictor(NoseT5ModelBase):

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
        prepro_class = NosePreprocessor.get_preprocessor_class(self.version)
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


class NoseClassificationPredictor(NoseClassificationModelBase):
    TEXT_COL = ['text_a', 'text_b']

    def __init__(self, version, cuda_device=-1, task_id=0, **kwargs):
        self.version = version
        self.cuda_device = cuda_device
        super(NoseClassificationPredictor, self).__init__(
            version, getattr(self.prepro, 'prefix', None), cuda_devices=[self.cuda_device],
            task_id=task_id
        )
        weight = kwargs.get('weight') and kwargs.pop('weight')
        self.model_args.update_from_dict(kwargs)
        self.sm_model = ClassificationModel('bert', self.best_model_dir, args=self.model_args, weight=weight,
                                            cuda_device=get_cuda_device(cuda_device))

    @cached_property
    def prepro(self):
        prepro_class = NosePreprocessor.get_preprocessor_class(self.version)
        return prepro_class()

    def eval(self, saving_excel=None, compare_eval=True):
        eval_df = self.prepro.get_from_h5('eval')
        eval_res, res_eval_df = super().eval(
            eval_df, self.prepro.col_mapping.keys() if hasattr(self.prepro, 'col_mapping') else None)
        if saving_excel:
            res_eval_df = res_eval_df.applymap(remove_illegal_char)
            res_eval_df.to_excel(saving_excel, sheet_name=self.sub_project_name or 'eval_res', index=False)
        return eval_res

class ClsPredictor(NoseClassificationPredictor):
    init_mapping = {
        'v1.0': {'version': 'v1.3', 'task_id': 0, 'eval_batch_size': 800, 'silent': True, 'max_seq_length': 32,
                 'weight': [62, 1]}
    }

    def __init__(self, version, cuda_device: int):
        init_paras = self.init_mapping.get(version)
        if not init_paras:
            raise ValueError('Version %s is not supported.' % version)
        super().__init__(cuda_device=cuda_device, **init_paras)
        self.indications = self.prepro.indications
        self.num_indications = len(self.indications)
    

    def predict(self, to_predicts: list, return_probability=False, match_esid=True):
        ind, texts = zip(*[(i, (p, indication)) for i, p in enumerate(to_predicts) for indication in self.indications])
        raw_res, probability = super().predict(list(texts))
        num_predicts = len(to_predicts)
        raw_res = np.array(raw_res).reshape((num_predicts, self.num_indications)).T
        probability = probability.transpose().reshape((2, num_predicts, self.num_indications)).swapaxes(0, 2)
        res = self.refine_predicts(raw_res, probability)
        if match_esid:
            res_esid = [[Indication.get_esid_from_name(name) for name in names] for names in res]
            return {'esid': res_esid, 'indication': res}
        else:
            return (res, probability) if return_probability else res

    def refine_predicts(self, raw_res: np.ndarray, probability: np.ndarray):
        res = []
        for i in np.arange(raw_res.shape[1]):
            cond = (raw_res[:, i] == 1)
            if cond.any():
                cur_res = Indication.drop_parent_indications(self.indication_array[cond].tolist())
            else:
                cur_res = [self.indication_array[probability[:, i, 1].argmax()]]
            res.append(cur_res)
        return res

    def close_multiprocessing(self):
        self.set_multiprocessing(False)

    @cached_property
    def indication_array(self):
        return np.array(self.indications)
    
    def eval(self, saving_excel=None):
        df = self.prepro.get_production_dataset()
        res = {'esid': [], 'indication': []}
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            each_res = self.predict([row['description']])
            res['esid'].append(each_res['esid'][0])
            res['indication'].append(each_res['indication'])
        res = self.predict(df['description'].tolist())
        acc = accuracy(df['indication'].tolist(), res['indication'])
        print(f'accuracy (loose match): {acc}')
        if saving_excel:
            res_eval_df = pd.DataFrame({
                'description': df['description'].map(remove_illegal_char),
                'true_indication': df['indication'],
                'predicts': res['indication'],
                'indication_esid': res['esid']
            })
            res_eval_df.to_excel(saving_excel)
        return acc
    


class T5Predictor(NoseT5Predictor):
    init_config = {
        'v1.0': {'version': 'v1.6', 'task_id': 0, 'eval_batch_size': 40, 'max_length': 20},
        'v1.1': {'version': 'v1.6', 'task_id': 0, 'eval_batch_size': 40, 'max_length': 20,
                 'dict_json': 'data/es_dict_1209.json'}
    }

    def __init__(self, version, cuda_device: int):
        init_paras = self.init_config.get(version)
        if not init_paras:
            raise ValueError('Version %s is not supported.' % version)
        self.dict_json = init_paras.pop('dict_json', None)
        super().__init__(cuda_device=cuda_device, **init_paras)
        self.prefix = self.prepro.TASK


    @cached_property
    def all_indication_synonyms(self):
        iter_synonym = ((ind.esid, ind.name, syn) for ind in Indication.objects for syn in ind.iter_synonyms())
        esids, names, synonyms = zip(*iter_synonym)
        return pd.Series(esids), pd.Series(names), pd.Series(synonyms)

    get_ratio = np.vectorize(Levenshtein.ratio)

    def predict(self, to_predicts, return_dict=True, workers=1):
        df = pd.DataFrame({'prefix': self.prefix, 'to_predicts': to_predicts})

        # rule predict
        if workers == 1:
            rule_df = df.assign(indication=df['to_predicts'].map(Indication.rule_match))
        else:
            rule_df = df.copy()
            p = Parallizer('mp', max_workers=workers)
            rule_df['indication'] = p.parallize(df['to_predicts'].tolist(), Indication.batch_rule_match, 'texts')
        not_empty = rule_df['indication'].map(bool)
        model_df = df[~not_empty].copy()
        rule_df = rule_df[not_empty].copy()
        rule_df['esid'] = rule_df['indication'].map(
            lambda lst: list(map(Indication.get_esid_from_name, lst))
        )

        # model predict
        if not model_df.empty:
            model_df['raw_predicts'] = super().predict(model_df['prefix'].tolist(), model_df['to_predicts'].tolist())
            stacked_predicts = model_df['raw_predicts'].str.split(self.prepro.seperator, expand=True).stack()
            esids, names, all_synonyms = self.all_indication_synonyms
            scores = self.get_ratio(*np.meshgrid(all_synonyms.values, stacked_predicts.values))
            selected_inds = scores.argmax(axis=1)
            series2list = lambda s: s.dropna().tolist()
            model_res_df = pd.DataFrame({
                'predicts': stacked_predicts, 'esid': esids.values[selected_inds],
                'indication': names.values[selected_inds]
            }).stack(level=0).unstack(level=1).apply(series2list, axis=1).unstack()
        else:
            model_res_df = pd.DataFrame(columns=['predicts', 'esid', 'indication'])

        # join results
        res_df = pd.concat([rule_df, model_res_df]).sort_index()
        if return_dict:
            return {col: res_df[col].to_list() for col in ['esid', 'indication']}
        else:
            return res_df

    def close_multiprocessing(self):
        self.set_multiprocessing(False)

    def eval(self, saving_excel=None, workers=1):
        df = self.prepro.get_production_dataset()
        to_predicts = df['description']
        res = self.predict(to_predicts, return_dict=False, workers=workers)
        acc = accuracy(df['indication'], res['indication'])
        print(f'accuracy (loose match): {acc}')
        if saving_excel:
            res_eval_df = pd.DataFrame({
                'description': df['description'].map(remove_illegal_char),
                'true_indication': df['indication'],
                'raw_predicts': res['predicts'],
                'refined_predicts': res['indication'],
                'indication_esid': res['esid']
            })
            res_eval_df.to_excel(saving_excel)
        return acc

    @cached_property
    def prepro(self):
        prepro_class = NosePreprocessor.get_preprocessor_class(self.version)
        return prepro_class() if not self.dict_json else prepro_class(dict_json=self.dict_json)



if __name__ == '__main__':
    # predictor = NoseClassificationPredictor(
        # 'v1.3', 0, task_id=0, max_seq_length=32, eval_batch_size=800, weight=[62, 1])
    # predictor = ClsPredictor('v1.0', 8)
    # predictor = NoseT5Predictor('v1.7', 0, 0)
    # res = predictor.eval(saving_excel='results/eval_v1.7.0.xlsx', compare_eval=False, return_dataframe=True)
    predictor = T5Predictor('v1.1', 8)
    # predictor.eval(saving_excel='results/nose_product_eval_cls_v1.3.xlsx')
    predictor.eval()