from pharm_ai.PC.predictor import T5Predictor
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.util.ESUtils7 import get_page, Query, QueryType, bulk, OpType
import json
import ijson
from loguru import logger
from argparse import ArgumentParser
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import re
from tqdm import tqdm


class BatchPredictorPC:
    """Pull PC data, batch predict offline, and upload results.

    Usage:
        Run `python batch_predictor.py -h` to see command help.

        1. pull_full_data:
            Run on online server (for connecting cube ES server).
            Data saved at results folder (json_f). Copy (eg: scp) them to local machine to do predicting job.
        2. Run `predict` subcommand to predict. Since data too large, data are splited with `start` and `size` parameter.
            Results saved at results folder.
        3. Copy results to online server. Optional:
            Run `upload --cache` to prepare uploading results, and copy them to online server.
        4. Run `upload` subcommand to upload results.
    """

    patent_es_index = "drug_patent_claim"
    applicant_es_index = "drug_patent_applicant"
    log_file = 'batch_predict.log'
    date_format = '%b,%d,%Y %H:%M:%S'

    def __init__(self, init_model=True, batch_size=50, json_f = 'results/PC_data_0301.json'):
        self.json_f = Path(json_f)
        if init_model:
            update_args = {'eval_batch_size': batch_size, 'use_multiprocessing': False,
                           'use_multiprocessed_decoding': False, 'use_multiprocessing_for_evaluation': False}
            self.predictor = T5Predictor(version='v4.0', cuda_device=0, update_args=update_args)


    def pull_patent_data(self, start_spider_time=None):
        ls = get_page(self.patent_es_index, page_size=-1,
                      queries=Query(QueryType.GE, 'spider_wormtime', start_spider_time) if start_spider_time else None,
                      show_fields=['publication_docdb_comb', 'main_id', 'claim_num', 'claim_text', 'claim_class_act',
                                   'claim_type_act', 'claim_ref_act'])
        self.dump_result_to_json(ls)

    def dump_result_to_json(self, raw_result):
        with open(self.json_f, 'w') as f:
            json.dump(raw_result, f, ensure_ascii=False, indent=4)
        logger.success('Pulled full data ({})', len(raw_result))

    def pull_applicant_data(self, start_crete_time=None):
        ls = get_page(self.applicant_es_index, page_size=-1,
                      queries=Query(QueryType.GE, 'create_time', start_crete_time) if start_crete_time else None,
                      show_fields=['applicant_all_name'])
        self.dump_result_to_json(ls)

    def load_and_cache_patent_predicting_data(self, start, len_, print_size=False, keep_to_pred_esid=False, saving_json=True):
        cache_f = self._get_cache_path(start, len_, suffix="_with_esid" if keep_to_pred_esid else "")
        if cache_f.exists():
            with open(cache_f, 'r') as p:
                to_pred = json.load(p)
            logger.info('Predicting data loaded from {}', cache_f)
        else:
            with open(self.json_f, 'r') as f0:
                raw_df = pd.DataFrame.from_records(
                    [t for t in ijson.items(f0, 'item')],
                    columns=['main_id', 'esid', 'claim_num', 'claim_text'])

            if print_size:
                logger.info('Total number of claims: {}', raw_df['main_id'].nunique())
            keep_cols = ['claim_num', 'claim_text'] if not keep_to_pred_esid else ['esid', 'claim_num', 'claim_text']
            to_pred = {mid: d[keep_cols].values.tolist() for i, (mid, d) in
                       enumerate(raw_df.groupby('main_id')) if start <= i < start+len_}
            if saving_json:
                with open(cache_f, 'w') as p:
                    json.dump(to_pred, p, ensure_ascii=False, indent=4)
                logger.info('{} Predicting claims generated and saved to {}', len(to_pred), cache_f)
        end_ = start + len(to_pred)
        return end_, to_pred

    def load_and_cache_applicant_predicting_data(self, start, len_, saving_json=True):
        cache_f = self._get_cache_path(start, len_)
        if cache_f.exists():
            with open(cache_f, 'r') as f:
                to_pred = json.load(f)
            logger.info('Predicting data loaded from {}', cache_f)
        else:
            with open(self.json_f, 'r') as f0:
                to_pred = [t['applicant_all_name'] for i,t in enumerate(ijson.items(f0, 'item')) if start<=i<start+len_]
            if saving_json:
                with open(cache_f, 'w') as p:
                    json.dump(to_pred, p, ensure_ascii=False, indent=4)
                logger.info('{} predicting applicant generated and saved to {}',  len(to_pred), cache_f)
        end_ = start + len(to_pred)
        return end_, to_pred

    def load_and_cache_applicant_uploading_results(self, start, len_, saving_json=True):
        saving_file = self._get_cache_path(start, len_, "_to_upload")
        if saving_file.exists():
            with saving_file.open('r') as f0:
                res_json = json.load(f0)
            return res_json
        else:
            res_json = self._get_predicting_result_path(start, len_)
            with self.json_f.open('r') as f1, res_json.open('r') as f2:
                dt_iter = ijson.items(f1, 'item')
                res_iter = ijson.items(f2, 'item')
                res_upload = [{'esid': t['esid'], 'applicant_nlp_pre': r} for t, r in zip(dt_iter, res_iter)]
            if saving_json:
                with saving_file.open('w') as f3:
                    json.dump(res_upload, f3, ensure_ascii=False, indent=4)
                logger.info('{} uploading applicant results saved to {}', len(res_upload), res_json)
            return res_upload


    def _get_cache_path(self, start, len_, suffix=""):
        cache_f = self.json_f.parent / (self.json_f.stem + '_%d_to_%d' % (start, len_) + suffix + self.json_f.suffix)
        return cache_f

    def predict(self, start=0, len_=10000, predict_applicant=False):
        res_logger = logger.add(self.log_file)
        if predict_applicant:
            end_, to_pred = self.load_and_cache_applicant_predicting_data(start, len_)
        else:
            end_, to_pred = self.load_and_cache_patent_predicting_data(start, len_)
        t1 = datetime.now()
        logger.info('Start predicting at {}', t1.strftime(self.date_format))
        if predict_applicant:
            res = self.predictor.predict_person(to_pred, return_int=True)
        else:
            res = self.predictor.predict(to_pred)
        t2 = datetime.now()
        logger.info('End predicting at {}', t2.strftime(self.date_format))
        saving_f = self._get_predicting_result_path(start, len_)
        with open(saving_f, 'w') as s:
            json.dump(res, s, indent=4, ensure_ascii=False)
        logger.success('Results saved to {}, used {}min', saving_f, (t2 - t1).seconds / 60)
        logger.remove(res_logger)

    def _get_predicting_result_path(self, start, len_):
        return self.json_f.parent / (self.json_f.stem + 'predict_results_%d_to_%d' % (start, len_) + self.json_f.suffix)

    @classmethod
    def chunker(cls, seq, size):
        res = []
        for el in seq:
            res.append(el)
            if len(res) == size:
                yield res
                res = []
        if res:
            yield res


    def prepare_to_upload(self, start=0, len_=10000, return_json=True, saving_json:bool=False):
        saving_file = self._get_cache_path(start, len_, '_to_upload')
        if saving_file.exists():
            with saving_file.open('r') as f0:
                res_json = json.load(f0)
            return res_json
        dt_json = self._get_cache_path(start, len_, "_with_esid")
        if dt_json.exists():
            with open(dt_json, 'r') as f1:
                dt_raw = json.load(f1)
        else:
            _, dt_raw = self.load_and_cache_patent_predicting_data(start, len_, keep_to_pred_esid=True, saving_json=False)
        dt_df = pd.DataFrame(((id_, *row) for id_, r in dt_raw.items() for row in r),
                             columns=['main_id', 'esid', 'claim_num', 'claim_text'])
        res_json = self._get_predicting_result_path(start, len_)
        with open(res_json, 'r') as f2:
            res_raw = json.load(f2)
        res_df = pd.DataFrame(((id_, *row) for id_, r in res_raw.items() for row in r),
                              columns=['main_id', 'claim_num', 'claim_class', 'claim_type', 'claim_ref'])
        res = dt_df.merge(res_df, 'left', ['main_id', 'claim_num'])
        if not return_json:
            return res
        else:
            col_mapper = {'esid': '_id', 'claim_class': 'claim_class_act', 'claim_type': 'claim_type_act',
                          'claim_ref': 'claim_ref_act'}
            res = res.rename(columns=col_mapper)
            res['claim_class_act'] = res['claim_class_act'].map({'independent-claim': 'I', 'dependent-claim': 'D'})
            res['claim_class_pre'] = res['claim_class_act']
            res['claim_type_act'] = res['claim_type_act'].fillna('#')
            res['claim_type_pre'] = res['claim_type_act']
            res['claim_ref_act'] = res['claim_ref_act'].apply(lambda ls: '.'.join(str(xx) for xx in ls))
            res['claim_ref_pre'] = res['claim_ref_act']
            res_json = res.to_dict('records')
            if saving_json:
                with saving_file.open('w') as f:
                    json.dump(res_json, f, ensure_ascii=False, indent=4)
            return res_json

    def upload_batch_result(self, start=0, len_=10000, upload_applicant=False):
        res_logger=logger.add(self.log_file)
        if upload_applicant:
            to_upload = self.load_and_cache_applicant_uploading_results(start, len_, saving_json=False)
            bulk(self.applicant_es_index, OpType.UPDATE, to_upload)
        else:
            to_upload = self.prepare_to_upload(start, len_)
            bulk(self.patent_es_index, OpType.UPDATE, to_upload)
        logger.success('{} data started from {} uploaded to es.', len_, start)
        logger.remove(res_logger)

    def get_predicted_batches(self):
        res = []
        for file in self.json_f.parent.iterdir():
            if re.search(self.json_f.stem+r'predict_results_\d+_to_\d+.json', file.as_posix()):
                nums = re.findall(r'\d+', file.stem)
                res.append((int(nums[-2]), int(nums[-1])))
        return res




def main():
    arg_parser = ArgumentParser(prog='batch-predict-PC')
    subparsers = arg_parser.add_subparsers(help='sub-command help')

    pull_parser = subparsers.add_parser('pull', help='pull full data. Note: Run this command on online server.')
    pull_parser.add_argument('-f', '--json-file', type=str, default='results/PC_data_20210615.json',
                             help='The json file to save pulled data. [results/PC_data_20210615.json]')
    pull_parser.add_argument('--applicant', action='store_true', help='Pull applicant data instead of patent data.')
    pull_parser.add_argument('-t', '--start-spider-time', type=int, default=None,
                             help='Specify the beginnign spider wormtime to pull. eg: 1623754390553')
    def pull_func(args):
        p = BatchPredictorPC(init_model=False, json_f=args.json_file)
        if args.applicant:
            p.pull_applicant_data(start_crete_time=args.start_spider_time)
        else:
            p.pull_patent_data(start_spider_time=args.start_spider_time)
    pull_parser.set_defaults(func=pull_func)

    predict_parser = subparsers.add_parser('predict', help='predict batches of data.')
    predict_parser.add_argument('-f', '--json-file', type=str, default='results/PC_data_20210615.json',
                                help='The json file saving the whole pulled data. [results/PC_data_20210615.json]')
    predict_parser.add_argument('-c', '--cuda-device', type=int, default=0, help='The one GPU device to use. [0]')
    predict_parser.add_argument('-b', '--batch-size', type=int, default=50, help='Batch size. [50]')
    predict_parser.add_argument('--applicant', action='store_true', help='Cache or predict applicant data.')
    predict_parser.add_argument('--start', type=int, default=0, help='Start predicting position. [0]')
    predict_parser.add_argument('--size', type=int, default=10000, help='Batch predicting size. [10000]')
    predict_parser.add_argument('--only-data', action='store_true', help='Only process batch data, not predict.')
    predict_parser.add_argument('--no-notify', action='store_false', help='Do not send email notification.')
    predict_parser.add_argument('--email', type=str, default='fanzuquan@pharmcube.com',
                                help='Email to receive notifications. [fanzuquan@pharmcube.com]')
    def predict_fun(args):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

        if args.only_data:
            p = BatchPredictorPC(init_model=False, batch_size=args.batch_size, json_f=args.json_file)
            cache_func = (p.load_and_cache_patent_predicting_data if not args.applicant
                          else p.load_and_cache_applicant_predicting_data)
            func = (u.email_wrapper(cache_func, args.email) if args.no_notify else cache_func)
        else:
            p = BatchPredictorPC(init_model=True, batch_size=args.batch_size, json_f=args.json_file)
            predict_func = (p.predict if not args.applicant else
                            lambda start, len_ : p.predict(start, len_, predict_applicant=True))
            func = u.email_wrapper(predict_func, args.email) if args.no_notify else predict_func
        func(args.start, args.size)
    predict_parser.set_defaults(func=predict_fun)


    upload_parser = subparsers.add_parser('upload',
                                          help='Upload or prepare predicted results. '
                                               'Note: Run this command on online server if results prepared,'
                                               'else at local when prepare results.')
    upload_parser.add_argument('-f', '--json-file', type=str, default='results/PC_data_20210615.json',
                                help='The json file saving the whole pulled data. [results/PC_data_20210615.json]')
    upload_parser.add_argument('--applicant', action='store_true', help='Upload applicant results instead of patents.')
    upload_parser.add_argument('--cache', action='store_true',
                               help='Only prepare and cache to-upload results, not upload.')
    def upload_func(args):
        p = BatchPredictorPC(init_model=False, json_f=args.json_file)
        result_sizes = p.get_predicted_batches()
        if not args.cache:
            logger.info('To upload {} results.', len(result_sizes))
            pbar = tqdm(result_sizes)
            for start, size in pbar:
                pbar.set_description(f'Uploading {size} claim results from {start}...')
                p.upload_batch_result(start, size, args.applicant)
        else:
            logger.info('To prepare {} results.', len(result_sizes))
            pbar = tqdm(result_sizes)
            for start, size in pbar:
                pbar.set_description(f'Preparing {size} claim results from {start}...')
                if args.applicant:
                    p.load_and_cache_applicant_uploading_results(start, size)
                else:
                    p.prepare_to_upload(start, size, saving_json=True)

    upload_parser.set_defaults(func=upload_func)
    args = arg_parser.parse_args()
    args.func(args)



if __name__ == '__main__':
    main()
