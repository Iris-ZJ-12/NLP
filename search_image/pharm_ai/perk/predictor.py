from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from simpletransformers.t5 import T5Model, T5Args
from pharm_ai.perk.dt import ClassifyPreprocessor, NerPreprocessor
from loguru import logger
from sklearn.metrics import classification_report, accuracy_score
from typing import List
from datetime import datetime
import pandas as pd
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.util.sm_util import SMUtil
import wandb
from dataclasses import asdict
import json
from pathlib import Path
from pprint import pformat
import os

class Classifier:
    def __init__(self, version='v1.0', task=0, cuda_device=-1, use_cuda=True, generative=False):
        """Initialize a TypeClassifier (task=0) or ResultClassifier (task=1) or SeqModel (generative=True)."""
        self.task = task
        self.generative = generative
        self.prepro = ClassifyPreprocessor(version=version, generative=generative)
        if not generative:
            model_args=ClassificationArgs(
                use_multiprocessing=False,
                sliding_window=True,
                n_gpu=1,
                onnx=True,
                quantized_model=True,
            )
            task_str = {0: 'type', 1:'result'}.get(task)
            best_model_dir = f'best_model/{task_str}/{version}'
            num_labels = 2 if version.startswith('v1') else (6 if task==0 else 5)
            self.model = ClassificationModel('bert', best_model_dir, num_labels=num_labels, cuda_device=cuda_device,
                                             args=model_args, use_cuda=use_cuda and not model_args.quantized_model)
            mapping = {0: self.prepro.reverse_type_mapping,
                       1: self.prepro.reverse_result_mapping}
            self.mapping = mapping.get(task)
        else:
            best_model_dir = f'best_model/seq/{version}'
            model_args = Seq2SeqArgs(
                use_multiprocessing=False,
                output_dir=f'outputs/seq/{version}',
                cache_dir=f'cache/seq/{version}',
                best_model_dir=best_model_dir
            )
            self.model = Seq2SeqModel('bert', encoder_decoder_name=best_model_dir, args=model_args,
                                      use_cuda=use_cuda)
        self.rule_classification_mapping = {
            "CBA": ['cost benefit', 'cost-benefit', 'cost/benefit'],
            "CMA": ['cost minimization', 'cost-minimization', 'cost minimisation', 'cost-minimisation'],
            "CCA": ['cost-consequence', 'cost consequence'],
            "BIA": ['budget impact', 'budget-impact']
        }
        logger.add('classification_eval.log')

    def predict(self, to_predicts: list):
        if not self.generative and self.task == 0:
            rule_predicts = self.rule_predict(to_predicts)
            res1 = [(i, pred) for i,pred in enumerate(rule_predicts) if pred]
            to_predict2 = [(i, pred) for i,(r1, pred) in enumerate(zip(rule_predicts, to_predicts)) if not r1]
            if to_predict2:
                inds, to_preds = zip(*to_predict2)
                predicts, _ = self.model.predict(to_preds)
                res2 = list(zip(inds, map(lambda x:self.mapping.get(x), predicts)))
                last_inds, res = zip(*sorted(res2+ res1, key=lambda x:x[0]))
            else:
                last_inds, res = zip(*res1)
        elif not self.generative and self.task == 1:
            predicts, _ = self.model.predict(to_predicts)
            res = list(map(lambda x:self.mapping.get(x), predicts))
        elif self.generative:
            type_rule_predicts = self.rule_predict(to_predicts)
            res_seq = self.seq_predict(to_predicts)
            res = [[t_, r_] if t is None else [t, r_]
                   for t, (t_, r_) in zip(type_rule_predicts, res_seq)]
        return res

    def rule_predict(self, to_predicts: List[str]):
        result = []
        for to_predict in to_predicts:
            res = None
            for label, words in self.rule_classification_mapping.items():
                if any(word in to_predict.lower() for word in words):
                    res = label
                    break
            result.append(res)
        return result

    def seq_predict(self, to_predicts: List[str], return_raw_predicts=False):
        assert self.generative, "Set `generative=True`"
        predicts = self.model.predict(to_predicts)
        res = self.prepro.seq_decode(predicts, return_raw_results=return_raw_predicts)
        return res

    def eval(self, save_seq_result=False, join_result=False, saving_excel=None):
        _, df_eval = self.prepro.get_train_eval_dataset(self.task)
        if not self.generative:
            trues = df_eval['labels'].map(self.mapping).to_list()
            to_predicts = df_eval['text'].to_list()
            predicts = self.predict(to_predicts)
            report = classification_report(trues, predicts, digits=4)
            eval_res = classification_report(trues, predicts, output_dict=True)
            print(report)
            logger.info('Evaluate result: {}', eval_res)
            if saving_excel:
                eval_res_df = pd.DataFrame({
                    'text': to_predicts,
                    'labels_true': trues,
                    'labels_predicted': predicts
                })
                eval_res_df.to_excel(saving_excel)
        else:
            true_labels = df_eval['target_text'].tolist()
            true_types, true_results = zip(*self.prepro.seq_decode(true_labels, recover_encode=True))
            to_predicts = df_eval['input_text'].tolist()
            if not save_seq_result:
                predict_types, predict_results = zip(*self.seq_predict(to_predicts))
            else:
                raw_seq_results, seq_results = self.seq_predict(to_predicts, return_raw_predicts=True)
                raw_predict_types, raw_predict_results = zip(*raw_seq_results)
                predict_types, predict_results = zip(*seq_results)
                saved_eval_df = pd.DataFrame({
                    'input_text': to_predicts, 'target_text': true_labels,
                    'raw_predicted_type': raw_predict_types, 'type': predict_types,
                    'raw_predicted_result': raw_predict_results, 'result': predict_results
                })
                saved_excel_name = f'Seq evaluated result {datetime.today().strftime("%m%d")}.xlsx'
                saved_eval_df.to_excel(saved_excel_name)
                print(f'{saved_excel_name} was saved.')
            if not join_result:
                report_type = classification_report(true_types, predict_types, digits=4)
                print('Type classification evaluating result:\n',report_type)
                report_result = classification_report(true_results, predict_results, digits=4)
                print('Result classification evaluating result:\n', report_result)
            else:
                true_all = true_types + true_results
                predict_all = predict_types + predict_results
                report_all = classification_report(true_all, predict_all, digits=4)
                print('Type & result classification evaluating result: \n', report_all)


class Ner:
    root_path = Path(__file__).parent

    # (version, sub_task) -> model_type: mt5 or t5
    model_type_mapping = {('v2.1', 3): 't5'}

    def __init__(self, version='v2.0', sub_task=None, cuda_device=-1, use_cuda=True):
        self.version = version
        if cuda_device>-1:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        output_dir = self.root_path/'outputs'/version
        cache_dir = self.root_path/'cache'/version
        best_model_dir = self.root_path/'best_model'/'ner'/(f'{version}.{sub_task}' if sub_task else version)
        model_args = T5Args(
            wandb_project='perk',
            output_dir=output_dir.as_posix(),
            cache_dir=cache_dir.as_posix(),
            best_model_dir=best_model_dir.as_posix(),
            n_gpu=1,
            reprocess_input_data=True,
            use_cached_eval_features=False,
            overwrite_output_dir=True,
            use_multiprocessing=False,
            use_multiprocessed_decoding=False,
            use_multiprocessing_for_evaluation=False,
            eval_batch_size=50,
            max_seq_length=64,
            max_length=15,
            quantized_model=True
        )
        select_cuda = 0 if cuda_device>-1 else -1
        model_type = self.model_type_mapping.get((version, sub_task), 'mt5')
        self.model = T5Model(model_type, best_model_dir, model_args,
                             use_cuda=use_cuda and not model_args.quantized_model, cuda_device=select_cuda)
        if self.version=='v2.0':
            self.prefix = ['MedicationName']
        elif self.version=='v2.1':
            self.prefix = ['drug', 'disease', 'model', 'nation']
        self.prepro = NerPreprocessor(version)
        with open('raw_data/ht_location.json', 'r') as f:
            self.location_dic = json.load(f)


    def predict(self, prefix, to_predicts, return_raw=False):
        """
        NER prediction.
        :param list prefix: Prefix.
        :param list to_predicts: List of sentences to extract.
        :return: List of result of each sentence.
        """
        assert len(prefix)==len(to_predicts)
        to_predicts = [p+': '+t for p, t in zip(prefix, to_predicts)]
        raw_results = self.model.predict(to_predicts)
        if not return_raw:
            results = [r.split(self.prepro.seperator) for r in raw_results]
            return results
        else:
            return raw_results

    def eval(self, method='predict', saving_excel=None, word_level=True, return_result=False):
        """
        Eval model.
        :param method: eval method.
            - predict
            - model_eval
        :return: print scores or return eval results.
        """
        df_eval = self.prepro.get_from_h5('eval')
        if method=='predict':
            trues = df_eval['target_text'].to_list()
            to_predicts = df_eval['input_text'].to_list()
            prefixes = df_eval['prefix']
            predicts = self.predict(prefixes, to_predicts, True)
            if not word_level:
                accuracy_ = accuracy_score(trues, predicts)
                logger.info('accuracy (sentense level)=\n{}', pformat(accuracy_))
                eval_res = {'acc': accuracy_}
            else:
                eval_res = SMUtil.eval_ner_v2(df_eval, self.model, delimiter=self.prepro.seperator)
            logger.info('accuracy (entity level)=\n{}', pformat(eval_res))
        elif method == 'model_eval':
            eval_res = self.model.eval_model(df_eval)
            logger.info('eval result: \n{}', pformat(eval_res))
        if saving_excel:
            res_eval = pd.DataFrame({
                'prefix': prefixes,
                'input_text': to_predicts,
                'trues': trues, 'predicts': predicts
            })
            res_eval.to_excel(saving_excel, index=False)
            logger.info('eval results saved to "{}:', saving_excel)
        if return_result:
            return eval_res

    def sweep(self, sweep_name=None, sweep_id=None):
        if not sweep_id:
            sweep_config={
                "method": "bayes",
                "metric": {"goal": "maximize", "name": "eval_score.sum"},
                "parameters": {
                    "max_seq_length": {
                        "min": 40,
                        "max": 150,
                        "distribution": "int_uniform"
                    },
                    "max_length": {
                        "min": 10,
                        "max": 50,
                        "distribution": "int_uniform"
                    }
                }
            }
            if sweep_name:
                sweep_config.update({'name': sweep_name})
            sweep_id = wandb.sweep(sweep_config, project='perk')
            self._watched_model = False

        def eval_fun():
            args = {**asdict(self.model.args)}
            run = wandb.init(project='perk', config=args)
            self.model.args.update_from_dict(dict(run.config))
            if not self._watched_model:
                wandb.watch(self.model.model)
                self._watched_model=True
            eval_result = self.eval(return_result=True)
            wandb.log({'eval_result': eval_result})
            wandb.join()
        wandb.agent(sweep_id, function=eval_fun)


if __name__ == '__main__':
    # for task in [0,1]:
    #     classifier = Classifier('v2.0', n_gpu=2, task=task, generative=False)
    #     task_str={0: 'type', 1: 'result'}.get(task)
    #     classifier.eval(saving_excel=f'results/eval_classification_{task_str[task]}_v2.0.xlsx')
    ner = Ner('v2.1', 3, 0)
    ner.eval(saving_excel='results/eval_ner_v2.1.3.xlsx')

