from mart.sm_util.sm_util import get_edit_distance_ratios, get_rouge

from pharm_ai.prophet.ira.prepro import IraPreprocessor
import os
from pharm_ai.config import ConfigFilePaths as cfp
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from pprint import pprint
from sklearn.metrics import accuracy_score
import pandas as pd
from loguru import logger
from simpletransformers.t5 import T5Model,T5Args
import wandb
from dataclasses import asdict


class Ira:
    def __init__(self, version='v2.0', cuda_device=-1, n_gpu=1, task=None):
        self.version = version
        self.cuda_device = cuda_device if n_gpu==1 else -1
        prefix = os.path.join(cfp.project_dir, 'prophet', 'ira')
        output_dir = os.path.join(prefix, 'outputs', version)
        cache_dir = os.path.join(prefix, 'cache', version)
        best_model_dir = os.path.join(prefix, 'best_model', version)
        self.task = task
        if version=='v2.6' and task:
            output_dir = os.path.join(output_dir, task)
            cache_dir = os.path.join(cache_dir, task)
            best_model_dir = os.path.join(best_model_dir, task)
        if version in ['v2.0', 'v2.6']:
            model_args = Seq2SeqArgs(
                n_gpu=n_gpu,
                max_seq_length=225,
                max_length=10,
                reprocess_input_data=True,
                overwrite_output_dir=True,
                use_multiprocessing=False,
                output_dir=output_dir,
                cache_dir=cache_dir,
                best_model_dir=best_model_dir,
                eval_batch_size=12,
                length_penalty=1.0,
            )
            self.model = Seq2SeqModel('bert', encoder_decoder_name=best_model_dir, args=model_args,
                                      cuda_device=self.cuda_device)
        else:
            model_args = T5Args(
                n_gpu=n_gpu,
                max_seq_length=166,
                max_length=30,
                num_beams=2,
                repetition_penalty=4.2,
                top_k=70,
                top_p=0.98,
                reprocess_input_data=True,
                overwrite_output_dir=True,
                use_multiprocessing=False,
                evaluate_during_training=True,
                evaluate_during_training_verbose=True,
                evaluate_during_training_steps=10,
                fp16=False,
                output_dir=output_dir,
                cache_dir=cache_dir,
                best_model_dir=best_model_dir
            )
            self.model = T5Model('mt5', best_model_dir, model_args, cuda_device=self.cuda_device)

    def model_eval(self, wandb_log=True):
        if wandb_log:
            run = wandb.init(project='prophet')
            logger.debug(dict(run.config))
            self.model.args.update_from_dict(dict(run.config))
            if not self._watched_model:
                # Sometimes no need to call wandb.watch when using sweep (just call once)
                wandb.watch(self.model.model)
                self._watched_model=True
        prepro = IraPreprocessor(self.version)
        eval_df = prepro.get_from_h5('eval', suffix=self.task)
        self.model.args.evaluate_generated_text=True
        def acc_rouge_f(trues, preds):
            if self.version=='v2.6':
                refined = prepro.decoding(preds, handle_round_expression=(self.task=='round'))
                acc_ = accuracy_score(trues, refined)
                rouge = get_edit_distance_ratios(trues, refined)['edit_distances'].mean()
            else:
                acc_ = accuracy_score(trues, preds)
                rouge = get_edit_distance_ratios(trues, preds)['edit_distances'].mean()
            return {'accuracy':acc_, 'rouge': rouge}
        res = self.model.eval_model(
            eval_df, eval_score=acc_rouge_f)
        pprint(res)
        if wandb_log:
            wandb.log(res)

    def eval(self, result_xlsx=None, get_rouge_score=True):
        prepro = IraPreprocessor(self.version)
        eval_df = prepro.get_from_h5('eval', suffix=self.task)
        eval_df = self._validate_eval_df(eval_df)
        one_prefix_versions = ['v2.0', 'v2.1', 'v2.3']
        if self.version in one_prefix_versions:
            # one joined prefix
            trues = prepro.decoding(eval_df['target_text'].tolist())
            trues_i, trues_r, trues_a = zip(*trues)
            predicts_raw = self.model.predict(eval_df['input_text'].tolist())
            predicts = prepro.decoding(predicts_raw, recover_encoding=False, keep_case=self.version in ['v2.1'])
            predicts_i, predicts_r, predicts_a = zip(*predicts)
        elif self.version=='v2.6':
            # no prefix for individual seq2seq model format
            to_pred_list = eval_df['input_text'].tolist()
            trues = eval_df['target_text'].tolist()
            predicts_ = self.model.predict(to_pred_list)
            predicts = prepro.decoding(predicts_, handle_round_expression=(self.task == 'round'))
        else:
            # three types of prefix
            trues_i = eval_df[eval_df['prefix']=='investee']['target_text']
            trues_r = eval_df[eval_df['prefix']=='round']['target_text']
            trues_a = eval_df[eval_df['prefix']=='amount']['target_text']
            to_predicts = eval_df['prefix']+ ': ' + eval_df['input_text']
            eval_df['predicts'] = self.model.predict(to_predicts.tolist())
            predicts_i = eval_df[eval_df['prefix']=='investee']['predicts']
            predicts_r = eval_df[eval_df['prefix']=='round']['predicts']
            predicts_a = eval_df[eval_df['prefix']=='amount']['predicts']
        # metric accuracy
        if self.version=='v2.6':
            accuracy_ = accuracy_score(trues, predicts)
            logger.info('\nAccuracy score:\n{}: {}', self.task, accuracy_)
        else:
            accuracy_i = accuracy_score(trues_i, predicts_i)
            accuracy_r = accuracy_score(trues_r, predicts_r)
            accuracy_a = accuracy_score(trues_a, predicts_a)
            logger.info('\nAccuracy score:\ninvestee: {}\nround: {}\namount: {}', accuracy_i, accuracy_r, accuracy_a)
        # metric rouge
        if get_rouge_score:
            if self.version=='v2.6':
                rouge_ = self.get_rough_score(trues, predicts)
            else:
                rouge_i = self.get_rough_score(trues_i, predicts_i)
                rouge_r = self.get_rough_score(trues_r, predicts_r)
                rough_a = self.get_rough_score(trues_a, predicts_a)
        # saving result to excel
        if result_xlsx:
            if self.version in one_prefix_versions:
                res_df = pd.DataFrame({
                    'input_text': eval_df['input_text'],
                    'investee_true': trues_i, 'investee_pred': predicts_i,
                    'round_true': trues_r, 'round_pred': predicts_r,
                    'amount_true': trues_a, 'amount_pred': predicts_a
                })
            elif self.version=='v2.6':
                res_df = pd.DataFrame({
                    'input_text': eval_df['input_text'],
                    self.task+'_true': trues, self.task+'_pred': predicts
                })
            else:
                eval_df.rename(columns={'target_text':'true', 'predicts':'pred'}, inplace=True)
                res_pivot = pd.pivot(eval_df.drop_duplicates(), 'input_text', 'prefix', ['true', 'pred'])
                res_df = res_pivot.swaplevel(axis=1)[['investee','round','amount']]
                res_df.columns = res_df.columns.to_flat_index().map(lambda x: '_'.join(x))
                res_df = res_df.reset_index()
            with pd.ExcelWriter(result_xlsx) as writer:
                if get_rouge_score:
                    if self.version=='v2.6':
                        res_df.to_excel(writer, sheet_name='eval_result_'+self.task)
                        if rouge_ is not None:
                            rouge_.to_excel(writer, sheet_name=self.task)
                    else:
                        res_df.to_excel(writer, sheet_name='eval_result')
                        if rouge_i is not None:
                            rouge_i.to_excel(writer, sheet_name='investee')
                        if rouge_r is not None:
                            rouge_r.to_excel(writer, sheet_name='round')
                        if rough_a is not None:
                            rough_a.to_excel(writer, sheet_name='amount')
            logger.info('Evaluating results have been saved to "{}"', result_xlsx)

    def predict(self, to_predict: list):
        if self.version not in ['v2.5']:
            raw_res = self.model.predict(to_predict)
            prepro = IraPreprocessor(self.version)
            res = prepro.decoding(raw_res, recover_encoding=False, keep_case=self.version in ['v2.1'])
        else:
            ids, model_to_predicts = zip(*[(i_, prefix + ': ' + raw) for i_, raw in enumerate(to_predict)
                                           for prefix in ['investee', 'round', 'amount']])
            res_raw = self.model.predict(model_to_predicts)
            res = [[r for id_, r in zip(ids, res_raw) if id_ == sent_id] for sent_id in set(ids)]
        return res

    def get_rough_score(self, trues, predicts):
        trues, predicts = zip(*[(t_, p_) for t_, p_ in zip(trues, predicts) if t_ and p_])
        df = pd.DataFrame({'pred': predicts, 'true': trues}).reset_index(drop=True)
        df['if_correct'] = df['pred'].eq(df['true']).astype(int)
        try:
            rouge_df = get_rouge(trues, predicts)
            distance_df = get_edit_distance_ratios(trues, predicts)
            res = pd.concat([df, rouge_df, distance_df], axis=1)
            return res
        except ValueError:
            logger.error('Failed to compute rough score.')
            return None

    def _validate_eval_df(self, df: pd.DataFrame):
        if self.version=='v2.5':
            key_cols = ['input_text', 'prefix']
            res_df = df[df['input_text'].ne('')].drop_duplicates(subset=key_cols)
        else:
            res_df = df
        return res_df

    def sweep(self, sweep_name=None, sweep_id=None):
        if not sweep_id:
            sweep_config={
                "method": "bayes",
                "metric": {"goal": "maximize", "name":"eval_score.accuracy"},
                "parameters": {
                    "max_seq_length":{
                        "min": 100,
                        "max": 350,
                        "distribution": "int_uniform"
                    },
                    "max_length": {
                        "min": 5,
                        "max": 30,
                        "distribution": "int_uniform"
                    },
                    "length_penalty": {
                        "min": 0.3,
                        "max": 5.0,
                        "distribution": "uniform"
                    },
                    "repetition_penalty": {
                        "min": 0.3,
                        "max": 5.0,
                        "distribution": "uniform"
                    },
                    "num_beams": {
                        "min": 1,
                        "max": 10,
                        "distribution": "int_uniform"
                    },
                    "top_k": {
                        "min": 10,
                        "max": 100,
                        "distribution": "int_uniform"
                    },
                    "top_p": {
                        "min": 0.5,
                        "max": 1.0,
                        "distribution": "uniform"
                    }
                }
            }
            if sweep_name:
                sweep_config.update({'name': sweep_name})
            sweep_id=wandb.sweep(sweep_config, project='prophet')

        def eval_fun():
            args = {**asdict(self.model.args)}
            wandb.init(project='prophet', config=args)
            self.model_eval()
            wandb.join()

        self._watched_model=False
        wandb.agent(sweep_id, function=eval_fun)


if __name__ == '__main__':
    x = Ira(version='v2.5', cuda_device=-1, n_gpu=1)
    x.model_eval(False)