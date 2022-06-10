# -*- coding: UTF-8 -*-
"""
Description : use multi-processing or multi-threading to process data in CPU
"""
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import pandas as pd
import psutil
import torch
from loguru import logger


class Parallizer():
    def __init__(self, method: str, max_workers: int, data_strategy='evenly_split'):
        """

        Args:
            method: "mp" or "mt", "mp" == multi-processing , "mt" == multi-threading
            max_workers: the pointed maximum number of processes or threads
            data_strategy: the strategy for processing data,default:'evenly_split' (split the data evenly according to
                            max_workers)
        """
        self.md = method
        self.mw = self.get_actual_max_workers(max_workers)
        self.ds = data_strategy

    # to get the actual max workers according to the operating system
    def get_actual_max_workers(self, max_workers):
        if self.md == 'mt':
            logger.info('Use multi-threading!')
            sys_max_workers = mp.cpu_count()
            # sys_max_workers = int(int(subprocess.getstatusoutput('ulimit -u')[1])/2)
        else:
            logger.info('Use multi-processing!')
            sys_max_workers = psutil.cpu_count(False)
            # sys_max_workers = mp.cpu_count()
        max_workers = min(sys_max_workers, max_workers)
        if max_workers == sys_max_workers:
            logger.warning('The max workers you can use is {}! Please check out the parameter: max_workers'.format(
                sys_max_workers))
        logger.info('Use max_workers: {}'.format(max_workers))
        return max_workers

    def parallize(self, data, func, data_para='data', **kwargs):
        start = time.time()
        if self.md == 'mp':
            res = self.multi_process(data, func, data_para=data_para, **kwargs)  # type:list
        else:
            res = self.multi_thread(data, func, data_para=data_para, **kwargs)  # type:list
        end = time.time()
        logger.info('Finishing and totally taking {} seconds!!!'.format(round(end - start, 3)))
        return res

    # processing the data by the data strategy
    @staticmethod
    def process_data(data, max_workers, data_strategy='evenly_split'):
        logger.info('Start processing data...')
        dt_length = len(data)
        if data_strategy == 'evenly_split':  # split the data evenly according to max_workers
            step = int(dt_length / max_workers) + 1  # the length of one split data
            if isinstance(data, pd.DataFrame) or isinstance(data, list):  # type:pd.DataFrame/list
                sub_data_sets = [data[i:i + step] for i in range(0, dt_length, step)]
            elif isinstance(data, dict):
                keys_list = list(data.keys())
                sub_data_sets = []
                for i in range(0, dt_length, step):
                    sub_data_sets.append({j: data[j] for j in keys_list[i:i + step]})
            else:
                sub_data_sets = [data]
        else:
            sub_data_sets = [data]
        logger.info('End processing data!')
        return sub_data_sets  # type:list

    def multi_process(self, data, func, data_para='data', **kwargs):
        logger.debug('Start main process id: {}'.format(os.getpid()))
        sub_data_sets = self.process_data(data, data_strategy=self.ds, max_workers=self.mw)

        res = dict()
        with ProcessPoolExecutor(max_workers=self.mw, initializer=self.initializer) as executor:
            if not isinstance(func, list):
                futures = {}
                for dt, n in zip(sub_data_sets, list(range(len(sub_data_sets)))):
                    kwargs[data_para] = dt
                    futures[executor.submit(func, **kwargs)] = n
            else:
                assert len(func) == len(sub_data_sets)
                futures = {executor.submit(func[n], dt, **kwargs): n for dt, n in
                           zip(sub_data_sets, list(range(len(sub_data_sets))))}
            for f in as_completed(futures):  # not block,iterator
                f.dt_id = futures[f]
                try:
                    res.update({f.dt_id: f.result()})
                except Exception:
                    # logger.exception(Exception)
                    res.update({f.dt_id: 'parallel_failed'})
                finally:
                    self.done(f)
                    # f.add_done_callback(self.done)
        res = [res[i] for i in sorted(res.keys())]
        # single process and failed
        if len(res) == 1 and res[0] == 'parallel_failed':
            with open(f'./error_dt_{time.strftime("%Y%m%d", time.localtime())}.txt', 'a+') as f:
                f.write(f'error data:{str(data)}\n')
            return ['parallel_failed'] * len(data)

        # deal with failed process
        for i in range(len(res)):
            if res[i] == 'parallel_failed':
                res[i] = self.multi_process(data=sub_data_sets[i], func=func, data_para=data_para,
                                            **kwargs)  # use multi-processing to deal with failed process

        res = self.flat_nested_list2(res)
        assert len(res) == len(data)
        return res  # type:list

    def initializer(self):
        if self.md == 'mp':
            logger.info('Starting Process id: {}'.format(os.getpid()))
        elif self.md == 'mt':
            logger.info('Starting Process id: {}, threading id: {}'.format(os.getpid(), threading.currentThread()))

    def done(self, fn):
        if self.md == 'mp':
            if fn.cancelled():
                logger.info('Cancelled! Process id: {}, dt_id: {} .'.format(os.getpid(), fn.dt_id))
            elif fn.done():
                error = fn.exception()
                if error:
                    logger.error('ERROR! (Process id: {}, dt_id: {} ) : {}!'.format(os.getpid(), fn.dt_id, error))
                else:
                    logger.info('Success! Process id: {}, dt_id: {} .'.format(os.getpid(), fn.dt_id))
        elif self.md == 'mt':
            if fn.cancelled():
                logger.info('Cancelled! Thread id: {}, dt_id: {} .'.format(threading.currentThread(), fn.dt_id))
            elif fn.done():
                error = fn.exception()
                if error:
                    logger.error(
                        'ERROR! (Thread id: {}, dt_id: {} ) : {}!'.format(threading.currentThread(), fn.dt_id, error))
                else:
                    logger.info('Success! Thread id: {}, dt_id: {} .'.format(threading.currentThread(), fn.dt_id))

    def multi_thread(self, data, func, data_para='data', **kwargs):
        logger.debug('Start main thread id: {}'.format(threading.currentThread()))
        sub_data_sets = self.process_data(data, data_strategy=self.ds, max_workers=self.mw)
        res = dict()
        with ThreadPoolExecutor(max_workers=self.mw, initializer=self.initializer) as executor:
            if not isinstance(func, list):
                futures = {}
                for dt, n in zip(sub_data_sets, list(range(len(sub_data_sets)))):
                    kwargs[data_para] = dt
                    futures[executor.submit(func, **kwargs)] = n
            else:
                assert len(func) == len(sub_data_sets)
                futures = {executor.submit(func[n], dt, **kwargs): n for dt, n in
                           zip(sub_data_sets, list(range(len(sub_data_sets))))}
            for f in as_completed(futures):  # not block,iterator
                f.dt_id = futures[f]
                try:
                    res.update({futures[f]: f.result()})
                except Exception:
                    res.update({futures[f]: 'parallel_failed'})
                finally:
                    f.add_done_callback(self.done)
        res = [res[i] for i in sorted(res.keys())]

        # single process and failed
        if len(res) == 1 and res[0] == 'parallel_failed':
            with open('./error_dt.txt', 'a+') as f:
                f.write('time:{}, '.format(time.strftime("%Y%m%d_%H:%M:%S", time.localtime())))
                f.write('error data:{}\n'.format(str(data)))
            return ['parallel_failed'] * len(data)

        # deal with failed process
        for i in range(len(res)):
            if res[i] == 'parallel_failed':
                res[i] = self.multi_thread(sub_data_sets[i], func)  # use multi-processing to deal with failed process
        res = self.flat_nested_list2(res)
        assert len(res) == len(data)
        return res  # type:list

    def flat_nested_list2(self, res_list: list[list]):
        res = []
        for i in res_list:
            for j in i:
                res.append(j)
        return res

    def flat_nested_list(self, res_list):
        """

        Args:
            res_list: [[2, 80, 150, 12], [392, 36, 252, 80], [36, 392]]

        Returns: [2, 80, 150, 12, 392, 36, 252, 80, 36, 392]

        """
        r_list = []
        a = 0
        for sublist in res_list:
            try:
                for i in sublist:
                    r_list.append(i)
            except TypeError:
                r_list.append(sublist)
        for i in r_list:
            if isinstance(i, list):
                a = + 1
                break
        if a == 1:
            return self.flat_nested_list(r_list)
        if a == 0:
            return r_list

    # use the multi-threading to predict, and the num of threads is the num of gpus
    def parallize_gpu(self, model_class, to_predict_data: list, **model_init_args):
        self.mw = torch.cuda.device_count()
        logger.info(
            'Starting eval by using gpus and multi-threading! And the num of gpus/threads is : {}'.format(self.mw))
        funcs = self.get_models_func(model_class, **model_init_args)
        self.md = 'mt'
        try:
            res = self.parallize(data=to_predict_data, func=funcs, **model_init_args)
        except Exception as e:
            logger.error('ERROR {}: {}'.format(threading.currentThread(), e))
            res = ['parallel_failed'] * len(to_predict_data)
        return res  # type:list

    # get model funcs,and the parameters is 'to_predict_data', you should modify it by considering your model_class
    def get_models_func(self, model_class, **model_init_args: dict):
        funcs = []
        for i in range(self.mw):
            # set you model and model args
            funcs.append(model_class(cuda_device=i, n_gpu=1, **model_init_args).model.predict)
        return funcs


if __name__ == '__main__':

    class A:
        def __init__(self):
            pass

        def test_func(self, para1, para2, a, para3):
            l = []
            for i in a:
                l.append([{i, i * para1 + para2 + para3}])
            return l


    data = list(range(0, 20))
    data.extend(['s', 'sd', 's', 3, 44, 'xx', 7, 1, 2, 4])
    a = A()
    p = Parallizer(method='mp', max_workers=4)
    r2 = p.parallize(data=data, data_para='a', func=a.test_func, para1=2, para2=3, para3=4)
    print(r2)
    print(len(r2))

    #####################################
    # # parallize_gpu
    # class my_parallizer(Parallizer):
    #     def __init__(self, method: str, max_workers: int,):
    #         super(my_parallizer, self).__init__(method=method, max_workers=max_workers)
    #
    #     # get model funcs,and the parameters is 'to_predict_data', you should modify it by considering your model_class
    #     def get_models_func(self, model_class, **model_init_args: dict):
    #         funcs = []
    #         for i in range(self.mw):
    #             # set you model and model args
    #             my_model = model_class()
    #             my_model.args = my_model.get_args(version='v2.6.1.0', cuda_device=i, n_gpu=1)
    #             model = my_model.get_model()
    #             model.args.eval_batch_size = 10
    #             model.args.max_length = 20
    #             model.args.max_seq_length = 100
    #             if model_init_args:
    #                 model.args.update_from_dict(model_init_args)
    #
    #             funcs.append(model.predict)
    #         return funcs
    # from pharm_ai.panel.predict import ERV2_6
    # eval_df = pd.read_excel('/home/zyl/disk/PharmAI/pharm_ai/panel/data/v2.6/eval_dt_0701.xlsx')
    # eval_df = eval_df.astype('str')
    # to_predict_text = list(eval_df['prefix'].map(str) + ': ' + eval_df['input_text'].map(str))
    # p = my_parallizer(method='mt', max_workers=4)
    # res = p.parallize_gpu(model_class=ERV2_6, to_predict_data=to_predict_text)
