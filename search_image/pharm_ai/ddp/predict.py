from simpletransformers.t5 import T5Model, T5Args
from pharm_ai.perk.dt import NerPreprocessor
from sklearn.utils import shuffle
from loguru import logger
from sklearn.metrics import accuracy_score
import pandas as pd
from pharm_ai.util.sm_util import SMUtil
import wandb
from dataclasses import asdict


class Ner:

    def __init__(self, version='v2.1', n_gpu=2):
        self.version = version
        output_dir = f'outputs/{version}/'
        cache_dir = f'cache/{version}'
        best_model_dir = f'best_model/ner/{version}'
        model_args = T5Args(
            wandb_project='perk',
            output_dir=output_dir,
            cache_dir=cache_dir,
            best_model_dir=best_model_dir,
            n_gpu=n_gpu,
            reprocess_input_data=True,
            use_cached_eval_features=False,
            overwrite_output_dir=True,
            use_multiprocessing=False,
            use_multiprocessed_decoding=False,
            use_multiprocessing_for_evaluation=False,
            eval_batch_size=128,
            max_seq_length=64,
            max_length=10,
            quantized_model=False
        )
        self.model = T5Model('mt5', best_model_dir, model_args, use_cuda=True)
        # self.model = T5Model("mt5", "/home/clr/disk/playground/outputs/v2.1", model_args)
        if self.version=='v2.0':
            self.prefix = ['MedicationName']
        elif self.version=='v2.1':
            self.prefix = ['drug', 'disease', 'model', 'nation']
        self.prepro = NerPreprocessor(version)


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
        # df_eval = shuffle(self.prepro.get_from_h5('eval'), random_state=99)[:1000]
        df_eval = self.prepro.get_from_h5('eval')
        print(df_eval.shape)
        if method=='predict':
            trues = df_eval['target_text'].to_list()
            to_predicts = df_eval['input_text'].to_list()
            prefixes = df_eval['prefix']
            if not word_level:
                predicts = self.predict(prefixes, to_predicts, True)
                accuracy_ = accuracy_score(trues, predicts)
                logger.info('accuracy (sentence level)={}', accuracy_)
                eval_res = {'acc': accuracy_}
            else:
                eval_res = SMUtil.eval_ner_v2(df_eval, self.model, delimiter=self.prepro.seperator)
            logger.info('accuracy (entity level)={}', eval_res)
        elif method == 'model_eval':
            eval_res = self.model.eval_model(df_eval)
            logger.info('eval result: {}', eval_res)
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
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    classifier = Ner('v2.1', n_gpu=1)
    classifier.eval(return_result=True)
