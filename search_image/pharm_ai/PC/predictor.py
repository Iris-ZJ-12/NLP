from pharm_ai.PC.dt import GenerativeDataProcessor, ClassificationProcessor
from datetime import datetime
from simpletransformers.t5.t5_model import T5Model, T5Args
from simpletransformers.classification import ClassificationArgs, ClassificationModel
from pharm_ai.PC.claims_non_claims_classifier.predictor import ClaimClassifier
from pharm_ai.PC.claim_type_classifier.predictor import ClaimTypeClassifier
from pharm_ai.PC.parents_classifier.predictor import ParentsClassifier
import pandas as pd
from loguru import logger
from os import path
import os
from pprint import pprint
from progress.bar import Bar
import json
import gc
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pharm_ai.util.utils import Utilfuncs as u
import re
from torch.cuda import empty_cache

class PredictorAPI:
    def __init__(self):
        self.cc = ClaimClassifier()
        self.ctc = ClaimTypeClassifier()
        self.pc = ParentsClassifier()

    def predict_api(self, args: dict):
        """
        return:
            Format: List[result:List[ID:int, result_1:str, result_2:'str', result_3: List[parent_claim_ID: str]]]
                - result_1: `independent-claim` or `dependent-claim`.
                - result_2: Claim type. `#` refers to sentense length >=100, or number of sentenses >=100,
                            or not included in 20 claim type.
                - result_3: List of parent claim IDs.
                    parent_claim_ID: str. `*`: independent claim have no parent claims."""
        res = dict()
        failed = dict()

        # def timeout_handler(signum, frame):  # Custom signal handler
        #     raise Exception
        #
        # # Change the behavior of SIGALRM
        # signal.signal(signal.SIGALRM, timeout_handler)

        with Bar('Predicting...') as bar:
            c = 1

            for k, v in args.items():
                gc.collect()
                bar.next()
                if c % 500 == 0:
                    print(' ')
                    print('*'*100)
                    print(str(c) + ' samples were processed.')
                    print('*' * 100)
                    print(' ')

                # # timeout if one iteration exceeds 7 minutes.
                # signal.alarm(7 * 60)
                try:
                    v.sort(key=lambda t: t[0])
                    nums, txts = map(list, zip(*v))
                    preds_cc = self.cc.predict(txts)
                    preds_cc[0] = 'independent-claim'
                    x = []
                    for i, p in zip(v, preds_cc):
                        one = i + [p]
                        x.append(one)
                    y = []
                    for j in x:
                        if j[2] == 'independent-claim':
                            pred_ctc = self.ctc.predict([j[1]])[0]
                            y.append([pred_ctc, ['*']])
                        else:
                            # sequence # >= 100 !
                            if j[0] >= 100:
                                y.append(['#', []])
                                logger.warning(k)
                                logger.warning('seq.#: '+str(j[0]))
                                continue
                            nn = list(range(1, j[0]))
                            tt = [[j[1], str(n)] for n in nn]

                            # # of sentences >= 100 !
                            if len(tt) >= 100:
                                y.append(['#', []])
                                logger.warning(k)
                                logger.warning('#sens: '+str(len(tt)))
                                continue
                            preds_pc = self.pc.predict(tt)
                            preds_pc = [int(t[1]) for t, p in zip(tt, preds_pc) if p == 1]
                            # preds_pc = str(preds_pc).replace(' ', '').replace\
                            #     (',', '.').replace('[', '').replace(']', '')
                            y.append(['#', preds_pc])
                    z = []
                    for h, q in zip(x, y):
                        del h[1]
                        one = h + q
                        z.append(one)
                    res[k] = z
                    # increment count of successful predictions
                    c += 1
                except Exception as e:
                    logger.warning(e)
                    print(k)
                    logger.warning(k)
                    print(v)
                    print('='*100)
                    failed[k] = v
                    continue
                # else:
                #     # Reset the alarm
                #     signal.alarm(0)

        # save failed samples to file
        if failed:
            t = datetime.now().strftime("%d-%b-%Y")
            p = './failed_logs/failed_' + t + '.json'
            if path.exists(p):
                f = open(p, 'r', encoding='utf-8')
                failed_ = json.load(f)
                f.close()
                failed_.update(failed)
                f = open(p, 'w', encoding='utf-8')
                json.dump(failed_, f, indent=4, ensure_ascii=False)
                f.close()
            else:
                if not path.exists('./failed_logs/'):
                    os.makedirs('./failed_logs/')
                p = './failed_logs/failed_' + t + '.json'
                f = open(p, 'w', encoding='utf-8')
                json.dump(failed, f, indent=4, ensure_ascii=False)
            res.update(failed)
        return res

    def eval(self):
        self.cc.eval()
        self.ctc.eval()
        self.pc.eval()

class T5Predictor:
    def __init__(self, version='v2.2', cuda_device=-1, use_cuda=True, update_args:dict=None):
        self.version=version

        self._init_model(version, cuda_device, use_cuda, update_args)
        self.prepro = GenerativeDataProcessor(self.version)
        self.pc_patterns = [r'according to claim (?P<single>\d{1,3})',
                            r'according to any one of claims from (?P<from>\d{1,2}) to (?P<to>\d{1,3})',
                            r'according to any one of claims (?P<from>\d{1,2})(?:-| to |~)(?P<to>\d{1,3})',
                            r'according to claims (?P<from>\d{1,2})(?:-| to |~)(?P<to>\d{1,2})',
                            r'according to any one of claims (?P<multiple1>\d{1,3})(?: and | or )(?P<multiple2>\d{1,'
                            r'3})(?! to |-)',
                            r'根据权利要求(?P<single>\d{1,3})所述', r'根据权利要求(?P<from>\d{1,3})(?:至|-)(?P<to>\d{1,3})[中]?任一项',
                            r'根据权利要求(?P<from>\d{1,3})～(?P<to>\d{1,3})(?!、)',
                            r'根据权利要求(?P<multiple1>\d{1,3})或(?P<multiple2>\d{1,3})']
        self.random_state = 222
        self.column_mapper = {'input_text': 'claim_text', 'publication_docdb_comb': 'claim_id', 'target_text': 'trues'}
        if version=='v4.0':
            self.person_type_mapper = {'企业': 0, '学校/研究机构': 1, '政府机构': 2, '医院': 3, '个人': 4, '其他': 5}

    def _init_model(self, version, cuda_device=-1, use_cuda=True, update_args:dict=None):
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs', version)
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache', version)
        best_model_dir = os.path.join(os.path.dirname(__file__), 'best_model', version)
        model_args = T5Args(reprocess_input_data=True, overwrite_output_dir=True, save_eval_checkpoints=False,
            save_model_every_epoch=False, evaluate_during_training=True, evaluate_during_training_verbose=True,
            use_multiprocessing=False, use_multiprocessed_decoding=False, fp16=False, logging_steps=10,
            evaluate_during_training_steps=10, eval_batch_size=45, output_dir=output_dir, cache_dir=cache_dir,
            best_model_dir=best_model_dir, wandb_project='PC')
        if update_args:
            model_args.update_from_dict(update_args)
            logger.debug('Update args: {}', update_args)
        select_cuda = cuda_device if cuda_device > -1 or model_args.n_gpu > 1 else u.select_best_cuda()
        self.model = T5Model('mt5', best_model_dir, model_args, use_cuda=use_cuda, cuda_device=select_cuda)

    def model_eval(self):
        eval_df = self._get_eval_df()
        res = self.model.eval_model(eval_df)
        pprint(res)

    def eval(self, save_to_excel=None, eval_ref_metric='accuracy', flatten_ref:bool=None, trim_detail=False,
             drop_minor_predicts=True, **kwargs):
        """
        :param str save_to_excel: Excel path.
        :param str eval_ref_metric: 'accuracy', 'f1'.
        :param bool faltten_ref: F1 metric of claim ref use flattened element of each instead of the whole label.
            if None, output results of both options.
        :param bool trim_detail: Pretty print classification_report for claim ref.
        :param bool drop_minor_predicts: drop zero classes when computing classification_report (class_type).
        """
        eval_df = self._get_eval_df(**kwargs)
        if self.version=='v2.2':
            # only one prefix
            trues = self.prepro.refine_result(eval_df['target_text'])
            predicts_raw = self.model.predict(eval_df['input_text'].tolist())
            predicts_raw = pd.Series(predicts_raw, index=eval_df.index)
            predicts = self.prepro.refine_result(predicts_raw)
        else:
            # multiple prefixes
            to_predicts = eval_df.rename(columns=self.column_mapper)
            predicts_df = self.predict(to_predicts, return_pivot=False, **kwargs)
            res_df = to_predicts.merge(predicts_df)
            res_pivot = res_df.pivot(index=['claim_text', 'claim_id', 'claim_num'],
                                     columns='prefix', values=['trues','predicts']).reset_index()
            trues = res_pivot.loc[:, 'trues']
            predicts = res_pivot.loc[:, 'predicts']

        class_dt = pd.DataFrame({
            'trues': trues['claim_class'],
            'predicts': predicts['claim_class'].mask(
                ~predicts['claim_class'].isin(self.prepro.is_claim_mapper.keys()))}).dropna()
        report_class = classification_report(class_dt['trues'], class_dt['predicts'])
        print('Claim class evaluating result (F1 score):\n', report_class)

        type_dt = pd.DataFrame({
            'trues':trues['claim_type'],
            'predicts': predicts['claim_type'].mask(
                ~predicts['claim_type'].isin(self.prepro.claim_type_mapper.keys()))}).dropna()
        if drop_minor_predicts:
            cond = type_dt['predicts'].isin(type_dt['trues'].drop_duplicates())
            if not cond.all():
                type_dt = type_dt[cond]
                logger.warning('#classes of class_type predicts is larger than trues, dropped those classes.')
        report_type = classification_report(type_dt['trues'], type_dt['predicts'])
        print('Claim type evaluating result (F1 score):\n', report_type)
        if eval_ref_metric=='accuracy':
            if flatten_ref is None:
                self._eval_ref_accuracy(predicts, trues, flatten_ref=False)
                self._eval_ref_accuracy(predicts, trues, flatten_ref=True)
            else:
                self._eval_ref_accuracy(predicts, trues, flatten_ref)
        elif eval_ref_metric=='f1':
            self._eval_ref_f1(trues, predicts, flatten_ref, trim_detail=trim_detail)
        if self.version in ['v2.4', 'v2.6', 'v4.0']:
            # eval person
            to_predict_person_dt = to_predicts[to_predicts['prefix']=='person'][['claim_text','trues']]
            person_dt = pd.DataFrame({
                'trues': to_predict_person_dt['trues'],
                'predicts': self.predict_person(to_predict_person_dt['claim_text'])
            })
            target_names = None if self.version == 'v4.0' else ['自然人', '法人']
            report_person = classification_report(person_dt['trues'], person_dt['predicts'],
                                                  target_names=target_names, digits=4)
            print('Person type evaluating result (F1 score):\n', report_person)

        # save to excel
        if save_to_excel:
            with pd.ExcelWriter(save_to_excel) as writer:
                class_dt.to_excel(writer, sheet_name='claim_class', index=False)
                type_dt.to_excel(writer, sheet_name='claim_type', index=False)
                pd.DataFrame({"trues": trues['claim_ref'],
                    "predicts": predicts['claim_ref']
                }).to_excel(writer, sheet_name='claim_ref', index=False)
                if self.version in ['v2.4', 'v2.6', 'v4.0']:
                    person_dt.to_excel(writer, sheet_name='person', index=False)
            logger.info("Eval results saved to '{}'", save_to_excel)


    def _eval_ref_f1(self, trues, predicts, flatten_ref, trim_detail=False):
        if not flatten_ref:
            true_labels = trues['claim_ref'].dropna()
            predict_labels = predicts['claim_ref'].dropna()
        else:
            predict_labels, true_labels = self._change_ref_label_level(predicts, trues)
        report_ref = classification_report(true_labels.to_list(), predict_labels.to_list(), digits=4)
        if not trim_detail:
            print('Claim ref evaluating result (F1 score):\n', report_ref)
        else:
            report_ref_split = report_ref.split('\n\n')
            report_ref_trunk = '\n\n'.join([report_ref_split[0], report_ref_split[2]])
            lstrip_num = min(len(line) - len(line.lstrip()) for line in report_ref_trunk.split('\n') if line)
            report_ref_lstrip = '\n'.join(
                [line[lstrip_num:] if line else line for line in report_ref_trunk.split('\n')])
            print('Claim ref evaluating result (F1 score):\n', report_ref_lstrip)

    def _change_ref_label_level(self, predicts, trues):
        true_labels_raw = trues['claim_ref'].dropna().map(
            lambda x: self.prepro.ref_ranges_to_values(x, return_str=False))
        predict_labels_raw = predicts['claim_ref'].dropna()
        true_predict_labels = pd.DataFrame({'trues': true_labels_raw, 'predicts': predict_labels_raw})
        true_predict_balanced = true_predict_labels.apply(
            lambda s: self._balance_true_predict_label(s['trues'], s['predicts'], binary=False),
            axis=1, result_type='expand')
        true_labels = true_predict_balanced[0].explode().dropna()
        predict_labels = true_predict_balanced[1].explode().dropna()
        return predict_labels, true_labels

    def _eval_ref_accuracy(self, predicts, trues, flatten_ref=True):
        if not flatten_ref:
            true_labels = [self.prepro.ref_ranges_to_values(label, return_str=False)
                           for label in trues['claim_ref'].dropna()]
            predict_labels = predicts['claim_ref'].dropna().tolist()
            acc_ref = sum(1 for p in predict_labels if p in true_labels)/len(true_labels)
            print('Accuracy of claim_ref evaluating result (article level): ', acc_ref)
        else:
            predicts_, trues_ = self._change_ref_label_level(predicts, trues)
            acc_ref = accuracy_score(trues_.tolist(), predicts_.tolist())
            print('Accuracy of claim_ref evaluating result (sentence level):', acc_ref)


    def _get_eval_df(self, **kwargs):
        eval_df = self.prepro.get_from_h5('eval')
        return eval_df

    def predict(self, to_predict: [dict, pd.DataFrame], multi_step=True, return_pivot=True, **kwargs):
        """
        :param to_predict:
            - Dict[claim_esid -> List[List[claim_num, claim_text]]]
            - pandas.DataFrame[['claim_num', 'claim_text', 'claim_id', (or 'prefix')]]
        :return:
            - Dict[claim_esid -> List[List[claim_num: int,
                claim_class: str(dependent-claim or independent-claim), claim_type: str, claim_ref: List[int or '*']]]]
            - pandas.DataFrame[['claim_id', 'claim_num', 'claim_text', 'claim_class', 'claim_type', 'claim_ref']]
        """
        if multi_step:
            # Predict 'claim_type' and 'claim_ref' following predicting 'claim_class'
            if isinstance(to_predict, dict):
                to_predict_df = pd.concat(
                    pd.DataFrame(v, columns=['claim_num', 'claim_text']).assign(claim_id=k)
                    for k,v in to_predict.items()).reset_index(drop=True)
                to_predict_df['claim_text'] = to_predict_df['claim_text'].where(
                    to_predict_df['claim_text'].str.match(r'\d+'),
                    to_predict_df['claim_num'].astype(str) + '.' + to_predict_df['claim_text'])
            elif isinstance(to_predict, pd.DataFrame):
                to_predict_df = to_predict.drop_duplicates(subset=['claim_num','claim_text','claim_id'])
            res_class_raw = self._predict_class(to_predict_df['claim_text'], **kwargs)
            res_df = to_predict_df.assign(claim_class =res_class_raw)
            res_df['claim_class'] = res_df['claim_class'].where(
                res_df['claim_class'].isin(['independent-claim', 'dependent-claim']), 'dependent-claim')
            res_df['claim_class'] = res_df['claim_class'].where(
                ~res_df['claim_text'].str.fullmatch(r'[\W\d]*cancel(ed)?[\W]*'), 'independent-claim'
            )
            to_predict_type_text = res_df[res_df['claim_class'] == 'independent-claim']['claim_text']
            res_df['claim_type'] = self._predict_type(to_predict_type_text, **kwargs)
            res_df['claim_type'] = res_df['claim_type'].fillna('#')
            res_df['claim_ref'] = self._predict_ref(res_df, **kwargs)
            if isinstance(to_predict, dict):
                result = {claim_id:res.loc[:,['claim_num','claim_class','claim_type','claim_ref']].values.tolist()
                      for claim_id, res in res_df.groupby('claim_id')}
            elif isinstance(to_predict, pd.DataFrame):
                if return_pivot:
                    result = res_df
                else:
                    result = res_df.melt(['claim_id', 'claim_num', 'claim_text'],
                                         ['claim_class', 'claim_type', 'claim_ref'],
                                         'prefix', 'predicts')
        else:
            # Predict 3 task at the same time
            if isinstance(to_predict, dict):
                to_predict_df = pd.concat(
                    pd.DataFrame(v, columns=['claim_num', 'claim_text']).assign(claim_id=k, prefix=prefix) for k, v in
                    to_predict.items() for prefix in ['claim_class', 'claim_type', 'claim_ref']).reset_index(drop=True)
                to_predict_df['claim_text'] = to_predict_df['claim_text'].where(
                    to_predict_df['claim_text'].str.match(r'\d+'),
                    to_predict_df['claim_num'].astype(str) + '.' + to_predict_df['claim_text'])
            elif isinstance(to_predict, pd.DataFrame):
                to_predict_df = to_predict[['claim_num', 'claim_text', 'claim_id', 'prefix']]
            res_raw = self.model.predict((to_predict_df['prefix'] + ': ' + to_predict_df['claim_text']).tolist())
            res_df = to_predict_df.assign(predicts=res_raw)
            res_pivot = res_df.pivot(['claim_id','claim_num','claim_text'], 'prefix', 'predicts').reset_index()
            res_pivot['claim_ref'] = res_pivot['claim_ref'].mask(
                ~res_pivot['claim_ref'].str.fullmatch(r'((\d+|\d+\-\d+),?\s*)+')).dropna().map(
                lambda s:self.prepro.ref_ranges_to_values(s, return_str=False))
            res_pivot['claim_type'] = res_pivot['claim_type'].where(
                res_pivot['claim_class']=='independent-claim', '#')
            res_pivot['claim_ref'] = res_pivot['claim_ref'].where(
                res_pivot['claim_class']=='dependent-claim', ['*'])
            if isinstance(to_predict, dict):
                result = {claim_id:res.loc[:,['claim_num','claim_class','claim_type','claim_ref']].values.tolist()
                          for claim_id, res in res_pivot.groupby('claim_id')}
            elif isinstance(to_predict, pd.DataFrame):
                result = res_pivot
        empty_cache()
        return result

    def _update_model_args(self, key, kwargs):
        if key in kwargs and kwargs[key] != self.model.args.__dict__.get(key):
            self.model.args.eval_batch_size = kwargs[key]
            logger.info('adjusted to {}={}', key, kwargs[key])

    def _update_predicting_args(self, kwargs):
        self._update_model_args('eval_batch_size', kwargs)
        self._update_model_args('n_gpu', kwargs)

    def _predict_class(self, to_predict_text, **kwargs):
        self._update_predicting_args(kwargs)
        res_class_raw = self.model.predict(('claim_class: ' + to_predict_text).tolist())
        return res_class_raw

    def _predict_type(self, to_predict_type_text, **kwargs):
        to_pred_type = 'claim_type: ' + to_predict_type_text
        if not to_pred_type.empty:
            self._update_predicting_args(kwargs)
            res_type_raw = pd.Series(self.model.predict(to_pred_type.tolist()), index=to_pred_type.index).fillna('#')
            res_type_raw = res_type_raw.where(
                (res_type_raw.isin(self.prepro.claim_type_mapper.keys()) &
                 ~to_predict_type_text.str.fullmatch(r'[\W\d]*cancel(ed)?[\W]*')),
                '#')
        else:
            res_type_raw = '#'
        return res_type_raw

    def _predict_ref(self, df, **kwargs):
        """df: pandas.DataFrame[['claim_class', 'claim_text']]"""
        df_ = df.copy()
        to_pred_ref = 'claim_ref: ' + df_[df_['claim_class'] == 'dependent-claim']['claim_text']
        if not to_pred_ref.empty:
            self._update_predicting_args(kwargs)
            claim_ref_res = pd.Series(self.model.predict(to_pred_ref.to_list()), index=to_pred_ref.index)
            range_pattern = r'(\d+\s*(\-\s*\d+\s*)?)|((\d+\s*(\-\s*\d+\s*)?,\s*)+(\d+\s*(\-\s*\d+\s*)?)?)'
            df_['claim_ref'] = claim_ref_res.mask(~claim_ref_res.str.fullmatch(range_pattern)).dropna().map(
                lambda s: self.prepro.ref_ranges_to_values(s, return_str=False))
        else:
            df_['claim_ref'] = None
        res_df_claim_ref1 = df_[df_['claim_class'] == 'dependent-claim']['claim_ref']
        res_df_claim_ref2 = df_[df_['claim_class'] == 'independent-claim']['claim_ref']
        res_df_claim_ref1_notna = res_df_claim_ref1[res_df_claim_ref1.notna()]
        res_df_claim_ref1_na = res_df_claim_ref1[res_df_claim_ref1.isna()]
        res_df_claim_ref1_na = pd.Series(
            self.predict_by_re(df_[df_.index.isin(res_df_claim_ref1_na.index)]['claim_text'].tolist()),
            index=res_df_claim_ref1_na.index).map(lambda s: s if isinstance(s, list) else [])
        res_df_claim_ref1 = res_df_claim_ref1_notna.append(res_df_claim_ref1_na).sort_index()
        res_df_claim_ref2 = res_df_claim_ref2.map(lambda s: ['*'])
        res_ref_raw = res_df_claim_ref1.append(res_df_claim_ref2).sort_index()
        return res_ref_raw

    def predict_person(self, to_predicts, return_int=False):
        if isinstance(to_predicts, pd.Series):
            res_person_raw = pd.Series(self.model.predict(to_predicts.tolist()),
                                       index=to_predicts.index, name='person')
            if self.version == 'v4.0':
                res_person = res_person_raw
                if return_int:
                    res_person = res_person.map(self.person_type_mapper).fillna(4).astype(int)
            else:
                res_person = res_person_raw.where(res_person_raw.isin(['0', '1']), '1')
                if return_int:
                    res_person = res_person.astype(int)
            return res_person
        elif isinstance(to_predicts, list):
            if not all(p.startswith('person: ') for p in to_predicts):
                to_predicts = ['person: '+p for p in to_predicts]
            res_raw = self.model.predict(to_predicts)
            if self.version=='v4.0':
                result = res_raw
                if return_int:
                    result = [self.person_type_mapper.get(r, 4) for r in result]
            else:
                if return_int:
                    result = [int(r) if r in ['0', '1'] else 1 for r in res_raw]
                else:
                    result = [r if r in ['0', '1'] else '1' for r in res_raw]
            return result

    def predict_by_re(self, to_predict: list):
        """
        Predict parent claims by regular expression instead of NLP model.
        :param to_predict: List[claim_text: str].
        :return: List of indexes. Regular expression matching result.
        """
        result = []
        for cur_to_predict_sent in to_predict:
            res = dict()
            for pattern in self.pc_patterns:
                r = re.search(pattern, cur_to_predict_sent)
                if r:
                    res.update(r.groupdict())
            if 'from' in res.keys() and 'to' in res.keys():
                result.append(list(range(int(res['from']), int(res['to']) + 1)))
            elif 'single' in res.keys():
                result.append([int(res['single'])])
            elif any('multiple' in k for k in res.keys()):
                result.append(sorted(int(v) for k,v in res.items() if 'multiple' in k))
            else:
                result.append([])
        return result


    def _balance_true_predict_label(self, trues:list, predicts:list, binary=True):
        t, p = set(trues), set(predicts)
        tp = t.union(p)
        if not binary:
            res_true = [tpp if tpp in t else 'missing' for tpp in tp]
            res_predicts = [tpp if tpp in p else 'missing' for tpp in tp]
        else:
            res_true = [1 if tpp in t else 0 for tpp in tp]
            res_predicts = [1 if tpp in p else 0 for tpp in tp]
        return res_true, res_predicts


class PredictorSum(PredictorAPI, T5Predictor):
    def __init__(self, version=None, *, subtask_versions=('v1', 'v1', 'v1'), cuda_device=(-1, -1, -1),
                 **kwargs):
        if version:
            self.hybrid = False
            self.is_t5 = version.startswith('v2')
            self.version = version
            if not self.is_t5:
                self.version = version
                PredictorAPI.__init__(self)
            elif self.is_t5:
                T5Predictor.__init__(self, version=version, **kwargs)
            self.column_mapper = {'input_text': 'claim_text', 'publication_docdb_comb': 'claim_id', 'target_text': 'trues'}
        else:
            self.hybrid = True
            self.is_t5 = False
            self.subtask_versions = subtask_versions
            if subtask_versions[0]=='v1':
                self.cc = ClaimClassifier(cuda_device[0])
            else:
                T5Predictor.__init__(self, version=subtask_versions[0], cuda_device=cuda_device[0], **kwargs)
            if subtask_versions[1]=='v1':
                self.ctc = ClaimTypeClassifier(cuda_device[1])
            elif not hasattr(self, 'model'):
                T5Predictor.__init__(self, version=subtask_versions[1], cuda_device=cuda_device[1], **kwargs)
            if subtask_versions[2]=='v1':
                self.pc = ParentsClassifier(cuda_device[2])
            elif not hasattr(self, 'model'):
                T5Predictor.__init__(self, version=subtask_versions[2], cuda_device=cuda_device[2], **kwargs)
        self.random_state=222

    def eval(self, eval_data_version=None, use_api:str=None, sample_n=None, **kwargs):
        if not eval_data_version or self.version==eval_data_version:
            if not self.is_t5 and not hasattr(self, 'subtask_versions'):
                PredictorAPI.eval(self)
            else:
                T5Predictor.eval(self, **kwargs)
        else:
            if not self.is_t5 and eval_data_version in ['v2.3', 'v2.4', 'v2.6']:
                self.prepro = GenerativeDataProcessor(eval_data_version)
                T5Predictor.eval(self, version=eval_data_version, sample_n=sample_n,
                                 api_address=use_api, **kwargs)

    def _get_eval_df(self, version=None, sample_n=None, **kwargs):
        if not version:
            version = self.version
        if version.startswith('v2'):
            prepro = GenerativeDataProcessor(version) if not hasattr(self, 'prepro') else self.prepro
            _, eval_df = prepro.get_train_eval_datasets()
        else:
            # TODO
            eval_df = None
        if sample_n:
            logger.debug('Sampling {} data from evaluating data', sample_n)
            return eval_df.sample(n=sample_n, random_state=self.random_state)
        else:
            return eval_df

    def predict(self, to_predict, batch_num=None, **kwargs):
        if self.is_t5 or self.subtask_versions:
            result = T5Predictor.predict(self, to_predict, **kwargs)
        else:
            if isinstance(to_predict, dict):
                self.predict_api(to_predict)
            elif isinstance(to_predict, pd.DataFrame):
                to_predict_df = to_predict[['claim_id','claim_num','claim_text']].drop_duplicates()
                to_predict_df = self._validate_v1_predict_data(to_predict_df)
                to_predict_dict = {claim_id: d.loc[:,['claim_num', 'claim_text']].values.tolist()
                                   for claim_id, d in to_predict_df.groupby('claim_id')}
                if 'api_address' not in kwargs or not kwargs['api_address']:
                    result_dict = self.predict_api(to_predict_dict)
                else:
                    result_dict = self.predict_by_api(to_predict_dict, api_address=kwargs.pop('api_address'),
                                                      batch_num=batch_num, **kwargs)
                result_df_raw = pd.concat(
                    pd.DataFrame(v, columns=['claim_num', 'claim_class', 'claim_type', 'claim_ref']).assign(
                        claim_id=k)
                    for k,v in result_dict.items()).reset_index(drop=True)
                result_df_merge = result_df_raw.merge(
                    to_predict_df, 'right', ['claim_id', 'claim_num'])
                result_df_melt = result_df_merge.melt(['claim_id', 'claim_num', 'claim_text'],
                                                      ['claim_class', 'claim_type', 'claim_ref'], 'prefix', 'predicts')
                result_df = result_df_melt.merge(to_predict)
                return_pivot = bool(kwargs.get('return_pivot'))
                if return_pivot:
                    result = result_df.pivot(result_df.columns.drop(['prefix','predicts']), 'prefix', 'predicts').reset_index()
                else:
                    result = result_df
        return result

    def predict_by_api(self, to_predict_dict, api_address:str, batch_num=None, **kwargs):
        import requests
        if not batch_num:
            response = requests.post(api_address, json={'patent_input': to_predict_dict})
            result_dict = json.loads(response.text)
        else:
            result_dict = dict()
            for batch_start in range(0, len(to_predict_dict), batch_num):
                batch_end = min(batch_start+batch_num, len(to_predict_dict))
                to_predict_batch = {k:v for i,(k,v) in enumerate(to_predict_dict.items())
                                    if batch_start<=i<batch_end}
                response = requests.post(api_address, json={'patent_input': to_predict_batch})
                result_dict.update(json.loads(response.text))
                logger.info('Get API result of batch {}-{}.', batch_start, batch_end)
        if not result_dict:
            raise RuntimeError('No result returned from API.')
        return result_dict

    def _validate_v1_predict_data(self, data_):
        if isinstance(data_, dict):
            # TODO
            pass
        is_valid = self._check_v1_predict_data_validate(data_)
        if not is_valid:
            df = pd.DataFrame(
                [[cid, n, d[d['claim_num'] == n]['claim_text'].values[0]
                if n in d['claim_num'].values else 'missing text'] for cid, d in
                 data_.groupby('claim_id') for n in range(1, d['claim_num'].max() + 1)], columns=data_.columns)
            return df
        else:
            return data_

    def _check_v1_predict_data_validate(self, data_):
        is_valid = all([d['claim_num'].min() == 1 for cid, d in
                        data_[['claim_id', 'claim_num']].groupby('claim_id')])
        return is_valid

    def _predict_class(self, to_predict_text, **kwargs):
        if hasattr(self, 'cc'):
            res = self.cc.predict(to_predict_text.tolist())
        else:
            res = T5Predictor._predict_class(self, to_predict_text, **kwargs)
        return res

    def _predict_type(self, to_predict_type_text, **kwargs):
        if hasattr(self, 'ctc'):
            res_raw = self.ctc.predict(to_predict_type_text.tolist())
            res = pd.Series(res_raw, index=to_predict_type_text.index)
        else:
            res = T5Predictor._predict_type(self, to_predict_type_text, **kwargs)
        return res


class PersonClassifier:
    def __init__(self, version='v2.5', cuda_device=-1, n_gpu=1):
        self.version=version
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs', version)
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache', version)
        best_model_dir = os.path.join(os.path.dirname(__file__), 'best_model', version)
        model_args = ClassificationArgs(
            n_gpu=n_gpu,
            overwrite_output_dir=True,
            reprocess_input_data=False,
            use_cached_eval_features=True,
            save_eval_checkpoints=False,
            evaluate_during_training=True,
            evaluate_during_training_verbose=True,
            evaluate_during_training_steps=5,
            train_batch_size=30,
            eval_batch_size=30,
            learning_rate=4e-5,
            num_train_epochs=1,
            output_dir=output_dir,
            best_model_dir=best_model_dir,
            cache_dir=cache_dir,
            wandb_project='patent_claims'
        )
        self.model = ClassificationModel('bert', best_model_dir, args=model_args, num_labels=2,
                                         cuda_device=cuda_device)

    def eval(self, method='predict', export_excel=None):
        prepro = ClassificationProcessor(self.version)
        eval_df = prepro.get_from_h5('eval')
        if method=='predict':
            trues = eval_df['labels'].tolist()
            to_predicts = eval_df['text'].tolist()
            predicts, model_outputs = self.model.predict(to_predicts)
            report = classification_report(trues, predicts, target_names=['自然人', '法人'], digits=4)
            print(report)
            if export_excel:
                res_df = pd.DataFrame({'trues':trues, 'predicts':predicts})
                res_df['is_true'] = res_df['trues'].eq('predicts')
                res_df.to_excel(export_excel)
                logger.info('Evaluation results saved to "{}"', export_excel)
        elif method=='model':
            report_fun = lambda trues, preds: classification_report(
                trues, preds, target_names=['自然人', '法人'], digits=4)
            eval_res, model_outputs, wrong_preds = self.model.eval_model(
                eval_df, wandb_log=False, f1=f1_score, report=report_fun)
            report = eval_res.pop('report')
            pprint(eval_res)
            print(report)
        else:

            raise ValueError('method should be [predict, model].')

if __name__ == "__main__":
    x=T5Predictor('v4.0', cuda_device=0, update_args={'eval_batch_size': 10})
    # x = PredictorSum(subtask_versions=('v1','v1','v2.6'))
    x.eval(save_to_excel="results/eval_0511.xlsx")