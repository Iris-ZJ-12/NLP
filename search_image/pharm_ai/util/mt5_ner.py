# encoding: utf-8
'''
@author: zyl
@file: mt5_ner.py
@time: 2021/7/28 上午2:58
@desc:
'''

import copy
import html
import multiprocessing as mp
import re
import time
from functools import reduce
from itertools import product

import Levenshtein
import langid
import nltk
import pandas as pd
import wandb
from loguru import logger
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from sklearn.utils import resample
from transformers.models.t5 import T5Tokenizer

from pharm_ai.util.utils import Utilfuncs

word_tokenizer = T5Tokenizer.from_pretrained('/large_files/pretrained_pytorch/mt5-base/', truncate=True)


class DTUtils:

    def __init__(self):
        pass

    @staticmethod
    def turn_columns_to_prefixes(raw_df: pd.DataFrame, turn_columns: list, to_prefixes: list,
                                 saved_columns: list, to_column: str = 'target_text'):
        """
        raw_df:  [main_id	text_type	input_text	disease_labels	target_labels]
        # df = PanelUtils.turn_columns_to_prefix(raw_df,columns=['disease_labels','target_labels'],
        #                                       prefixes=['dis','tar'],to_column='target_text',
        #                                       saved_columns=['main_id','text_type','input_text'])
        Returns: [main_id	text_type	input_text	prefix	target_text]
        """
        assert len(turn_columns) == len(to_prefixes)
        all_dfs = []
        for c, p in zip(turn_columns, to_prefixes):
            every_df = raw_df[saved_columns]
            every_df['prefix'] = p
            every_df[to_column] = raw_df[c].tolist()
            all_dfs.append(every_df)
        new_df = pd.concat(all_dfs, ignore_index=True)
        return new_df

    @staticmethod
    def turn_prefixes_to_columns(raw_df: pd.DataFrame, turn_prefixes: list, to_columns: list,
                                 saved_columns: list, prefix_column: str = 'prefix', turn_column='target_text'):
        """
        raw_df: [main_id	text_type	input_text	prefix	target_text]
        # df = PanelUtils.turn_prefixes_to_columns(raw_df, turn_prefixes=['dis', 'tar'], to_columns=['dd', 'tt'],
        #                              saved_columns=['input_text', 'main_id'],prefix_column= 'prefix',turn_column='target_text')

        Returns: [ input_text	main_id	dd	tt ]
        """
        all_dfs = []
        for p, c in zip(turn_prefixes, to_columns):
            every_df = raw_df[raw_df[prefix_column] == p]
            every_df.rename(columns={turn_column: c}, inplace=True)
            all_columns = copy.deepcopy(saved_columns)
            all_columns.append(c)
            every_df = every_df[all_columns]
            all_dfs.append(every_df)
        new_df = reduce(lambda left, right: pd.merge(left, right, on=saved_columns, how='outer'), all_dfs)
        return new_df

    @staticmethod
    def cut_train_eval(df):
        raw_df = resample(df, replace=False)
        cut_point = min(8000, int(0.2 * len(raw_df)))
        eval_df = raw_df[0:cut_point]
        train_df = raw_df[cut_point:]
        return train_df, eval_df

    @staticmethod
    def up_sampling_one_prefix(train_df: pd.DataFrame, delimiter='|'):
        """

        Args:
            train_df: ['prefix', 'input_text', 'target_text']

        Returns:

        """
        negative_df = train_df[train_df['target_text'] == delimiter]
        neg_len = negative_df.shape[0]
        positive_df = train_df[train_df['target_text'] != delimiter]
        pos_len = positive_df.shape[0]
        if neg_len > pos_len:
            up_sampling_df = resample(positive_df, replace=True, n_samples=(neg_len - pos_len), random_state=3242)
            return pd.concat([train_df, up_sampling_df], ignore_index=True)
        elif neg_len < pos_len:
            up_sampling_df = resample(negative_df, replace=True, n_samples=(pos_len - neg_len), random_state=3242)
            return pd.concat([train_df, up_sampling_df], ignore_index=True)
        else:
            return train_df

    @staticmethod
    def up_sampling(train_df: pd.DataFrame):
        """
        Args:
            train_df: ['prefix', 'input_text', 'target_text']
        Returns:
        """
        print('use up sampling!')
        di_df = train_df[train_df['prefix'] == 'disease']
        di_df = DTUtils.up_sampling_one_prefix(di_df)
        tar_df = train_df[train_df['prefix'] == 'target']
        tar_df = DTUtils.up_sampling_one_prefix(tar_df)
        up_sampling_num = max([len(di_df), len(tar_df)])

        if len(di_df) < up_sampling_num:
            up_sampling_di_df = resample(di_df, replace=True, n_samples=(up_sampling_num - len(di_df)),
                                         random_state=3242)
        else:
            up_sampling_di_df = pd.DataFrame()

        if len(tar_df) < up_sampling_num:
            up_sampling_tar_df = resample(tar_df, replace=True, n_samples=(up_sampling_num - len(tar_df)),
                                          random_state=3242)
        else:
            up_sampling_tar_df = pd.DataFrame()

        return pd.concat([di_df, tar_df, up_sampling_di_df, up_sampling_tar_df],
                         ignore_index=True)


    @staticmethod
    def clean_text(text: str):
        text = ILLEGAL_CHARACTERS_RE.sub(r'', str(text))
        text = html.unescape(text)
        replaced_chars = ['\u200b', '\ufeff', '\ue601', '\ue317', '\n', '\t', '\ue000', '\ue005']
        for i in replaced_chars:
            if i in text:
                text = text.replace(i, '')

        text = ' '.join(text.split())
        text = text.strip()
        return text

    @staticmethod
    def truncate_text_df(df, truncated_input_text='truncated_input_text', truncating_size=400, overlapping_size=200,
                         check_in_text=False, truncated_target_text='truncated_target_text', inplace=False,
                         delimiter='|', ):
        df['ids'] = range(df.shape[0])
        df[truncated_input_text] = df['input_text'].apply(MT5Utils.truncate_text,
                                                          args=(truncating_size, overlapping_size))
        df = df.explode(truncated_input_text)

        if check_in_text:
            li = []
            for entities, text in zip(df['target_text'].tolist(), df[truncated_input_text].tolist()):
                entities = set(entities.split(delimiter))
                if '' in entities:
                    entities.remove('')
                if entities != set():
                    s = delimiter
                    for entity in list(entities):
                        if str(entity) in str(text):
                            s += (str(entity) + delimiter)
                    li.append(s)
                else:
                    li.append(delimiter)
            df[truncated_target_text] = li
        if inplace:
            df.drop('input_text', axis=1, inplace=True)  # type:pd.Dataframe
            df.drop('target_text', axis=1, inplace=True)
            df.rename(columns={truncated_input_text: 'input_text', truncated_target_text: 'target_text'}, inplace=True)
        return df

    @staticmethod
    def send_to_me(message):
        sender_email = "pharm_ai_group@163.com"
        sender_password = "SYPZFDNDNIAWQJBL"  # This is authorization password, actual password: pharm_ai163
        sender_smtp_server = "smtp.163.com"
        send_to = "1137379695@qq.com"
        Utilfuncs.send_email_notification(sender_email, sender_password, sender_smtp_server,
                                          send_to, message)

    @staticmethod
    def cut_sentence(paragraph):
        sent_tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')

        language = langid.classify(paragraph)[0]
        if language == 'zh':
            sentences = DTUtils.zh_cut_sentences(paragraph)
        else:
            sentences = sent_tokenizer.tokenize(paragraph)
        return sentences

    @staticmethod
    def zh_cut_sentences(para, drop_empty_line=True, strip=True, deduplicate=False):
        """

        Args:
            para: 输入文本
            drop_empty_line: 是否丢弃空行
            strip:  是否对每一句话做一次strip
            deduplicate: 是否对连续标点去重，帮助对连续标点结尾的句子分句

        Returns:
            sentences: list of str
        """
        if deduplicate:
            para = re.sub(r"([。！？\!\?])\1+", r"\1", para)

        para = re.sub('([。！？\?!])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?!][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        sentences = para.split("\n")
        if strip:
            sentences = [sent.strip() for sent in sentences]
        if drop_empty_line:
            sentences = [sent for sent in sentences if len(sent.strip()) > 0]
        return sentences

    @staticmethod
    def get_tokens(text):
        word_tokenizer = T5Tokenizer.from_pretrained('/large_files/pretrained_pytorch/mt5-base/', truncate=True)
        tokens = word_tokenizer.tokenize(str(text))
        return tokens


class MT5Utils:
    def __init__(self):
        pass

    @staticmethod
    def eval_decoration(eval_func):
        # #############################################################
        # examples: should set : self.wand_b_pro , self.model_version , self.args
        # >>> @eval_decoration
        # >>> def eval(eval_df,a,b):
        # >>>     eval_res = func... a,b
        # >>>     return eval_res
        # ############################################################
        def eval_method(self, eval_df, *args, **kwargs):
            # deal with eval df
            # eval_df = eval_df[['prefix', 'input_text', 'target_text']]
            # eval_df = eval_df.astype('str')
            eval_length = eval_df.shape[0]

            # wand_b
            wandb.init(project=self.wandb_proj, config=self.args,
                       name=self.model_version + time.strftime("_%m%d_%H:%M:%S", time.localtime()),
                       tags=[self.model_version, 'eval'])
            try:
                start_time = time.time()
                eval_res = eval_func(self, eval_df, *args, **kwargs)
                logger.info('eval finished!!!')
                end_time = time.time()
                need_time = round((end_time - start_time) / eval_length, 5)
                eval_time = round(need_time * eval_length, 4)
                print(f'version: {self.model_version}')
                print(f'eval length: {eval_length}')
                print(f'eval results: {eval_res}')
                print(f'eval time: {need_time} s * {eval_length} = {eval_time} s')
                wandb.log({"eval_res": eval_res, "eval_length": eval_length})
            except Exception as error:
                logger.error(f'eval failed!!! ERROR:{error}')
                # DTUtils.send_to_me(f'train failed!!! ERROR:{error}')
                eval_res = None
            finally:
                wandb.finish()
            return eval_res

        return eval_method

    @staticmethod
    def eval_entity_recognition(model, eval_df: pd.DataFrame, check_in_input_text: bool, delimiter='|',
                                use_truncation=True, truncating_size=400, overlapping_size=200, pos_neg_ratio=None):
        """eval entity recognition in mt5 model, version-v2 , reference: https://docs.qq.com/doc/DYXRYQU1YbkVvT3V2

        Args:
            model: a mt5 model
            eval_df: a pd.Dataframe , must have columns ['prefix','input_text','target_text']
            check_in_input_text: if the entities are in input_texts
            delimiter: the delimiter in target_text to split different entities
            use_truncation: if truncate the input text when predict
            truncating_size: truncating_size
            overlapping_size: overlapping_size
            pos_neg_ratio : the ratio of positive and negative sample importance

        Returns:
            show report and res
        """

        prefixes = eval_df['prefix'].to_list()
        input_texts = eval_df['input_text'].tolist()
        target_texts = eval_df['target_text'].tolist()
        revised_target_texts = MT5Utils.revise_target_texts(target_texts=target_texts, input_texts=input_texts,
                                                              check_in_input_text=False,
                                                              delimiter=delimiter)

        pred_target_texts = MT5Utils.predict_entity_recognition(model, prefixes, input_texts, use_truncation,
                                                                  truncating_size, overlapping_size)
        revised_pred_target_texts = MT5Utils.revise_target_texts(target_texts=pred_target_texts,
                                                                   input_texts=input_texts,
                                                                   check_in_input_text=check_in_input_text,
                                                                   delimiter=delimiter)

        eval_df['true_target_text'] = revised_target_texts
        eval_df['pred_target_text'] = revised_pred_target_texts

        eval_res = {}
        for prefix in set(prefixes):
            prefix_df = eval_df[eval_df['prefix'] == prefix]
            y_true = prefix_df['true_target_text'].tolist()
            y_pred = prefix_df['pred_target_text'].tolist()
            print(f'{prefix} report:')
            res_df = MT5Utils.entity_recognition_v2(y_true, y_pred, pos_neg_ratio=pos_neg_ratio)
            eval_res[prefix] = res_df

        print(f'sum report:')
        res_df = MT5Utils.entity_recognition_v2(revised_target_texts, revised_pred_target_texts,
                                                  pos_neg_ratio=pos_neg_ratio)
        eval_res['sum'] = res_df

        return eval_res

    @staticmethod
    def predict_entity_recognition(model, prefixes: list, input_texts: list, use_truncation=True, truncating_size=400,
                                   overlapping_size=200, delimiter='|') -> list:
        """predict entity recognition in mt5 model,

        Args:
            model: a mt5 model
            prefixes: prefixes
            input_texts: input_texts
            use_truncation: if use_truncation
            truncating_size: truncating_size
            overlapping_size: overlapping_size
            delimiter: the delimiter in target_text to split different entities,default: '|'

        Returns:
            pred_target_texts:list,every element in pred_target_texts corresponds a prefix and an input_text
        """
        assert len(prefixes) == len(input_texts)
        if use_truncation:
            t_ids, t_prefixes, t_input_texts = MT5Utils.truncate_texts(input_texts, prefixes,
                                                                         truncating_size=truncating_size,
                                                                         overlapping_size=overlapping_size)
            to_predict_texts = [i + ': ' + j for i, j in zip(t_prefixes, t_input_texts)]
            pred_target_texts = model.predict(to_predict_texts)
            pred_target_texts = MT5Utils.combine_pred_target_texts_by_t_ids(pred_target_texts, t_ids, delimiter)
        else:
            to_predict_texts = [i + ': ' + j for i, j in zip(prefixes, input_texts)]
            pred_target_texts = model.predict(to_predict_texts)
        assert len(pred_target_texts) == len(input_texts)
        return pred_target_texts  # type:list[str]

    @staticmethod
    def truncate_texts(input_texts: list, prefixes: list, truncating_size=900, overlapping_size=400):
        """for every input_text in input_texts, truncate it and record the truncated_ids for combining

        Args:
            input_texts: the list of many input_text
            prefixes: the prefix list of the input_texts list
            truncating_size: truncating_size
            overlapping_size: overlapping_size

        Returns:
            truncated_ids, truncated_prefixes, truncated_input_texts
        """
        assert len(input_texts) == len(prefixes)  # every input_text corresponds a prefix
        input_texts_ids = range(len(input_texts))

        truncated_ids = []
        truncated_prefixes = []
        truncated_input_texts = []

        for i_t_d, p, i_t in zip(input_texts_ids, prefixes, input_texts):
            truncated_input_text = MT5Utils.truncate_text(i_t, truncating_size=truncating_size,
                                                            overlapping_size=overlapping_size)
            for t_i_t in truncated_input_text:
                truncated_ids.append(i_t_d)
                truncated_input_texts.append(t_i_t)
                truncated_prefixes.append(p)
        return truncated_ids, truncated_prefixes, truncated_input_texts  # type:tuple[list[int],list[str],list[str]]

    @staticmethod
    def truncate_text(input_text: str, truncating_size=400, overlapping_size=200) -> list:
        """ truncate a input text

        Args:
            input_text: a str text
            truncating_size: truncating_size:sliding window
            overlapping_size: overlapping_size

        Returns:
            truncated_input_text: the list of truncated_input_text
        """
        if not isinstance(input_text, str):
            input_text = str(input_text)
        step_size = truncating_size - overlapping_size
        if step_size < 1:
            step_size = truncating_size
        steps = int(len(input_text) / step_size)
        truncated_input_text = []
        for i in range(0, steps + 1):
            text_i = input_text[i * step_size:i * step_size + truncating_size]
            if text_i != '':
                truncated_input_text.append(text_i)
        if (len(truncated_input_text) > 1) and (len(truncated_input_text[-1]) < overlapping_size):
            truncated_input_text = truncated_input_text[0:-1]
        return truncated_input_text  # type:list[str]

    @staticmethod
    def truncate_textV2(input_text: str, sliding_window=128, tokenizer=word_tokenizer) -> list:
        """ truncate a input text

        Args:
            input_text: a str text
            truncating_size: truncating_size:sliding window, max_seq_length
            overlapping_size: overlapping_size

        Returns:
            truncated_input_text: the list of truncated_input_text
        """

        if not isinstance(input_text, str):
            input_text = str(input_text)
        tokens = tokenizer.tokenize(input_text)

        if len(tokens) <= sliding_window:
            return [input_text]
        else:
            truncated_input_text = []
            step_size = int(sliding_window * 0.8)
            steps = int(len(input_text) / step_size)
            for i in range(0, steps + 1):
                text_i_tokens = tokens[i * step_size:i * step_size + sliding_window]
                if text_i_tokens != []:
                    text_i = ''.join(text_i_tokens).replace('▁', ' ').strip()
                    truncated_input_text.append(text_i)

            if (len(truncated_input_text) > 1) and (len(truncated_input_text[-1]) < (sliding_window - step_size)):
                truncated_input_text = truncated_input_text[0:-1]
            return truncated_input_text

        # steps = int(len(input_text) / step_size)
        # truncated_input_text = []
        # for i in range(0, steps + 1):
        #     text_i = input_text[i * step_size:i * step_size + truncating_size]
        #     if text_i != '':
        #         truncated_input_text.append(text_i)
        # if (len(truncated_input_text) > 1) and (len(truncated_input_text[-1]) < overlapping_size):
        #     truncated_input_text = truncated_input_text[0:-1]
        # return truncated_input_text  # type:list[str]

    @staticmethod
    def combine_pred_target_texts_by_t_ids(pred_target_texts, t_ids, delimiter: str = '|') -> list:
        """combine truncated_predicted_target_texts by truncated_ids

        Args:
            pred_target_texts: the result of predicting the truncated input_texts
            t_ids: get the truncated_ids when truncating input_texts
            delimiter: the delimiter in target_text to split different entities

        Returns:
            pred_target_texts: predicted target_texts
        """
        ids_target_text_dict = dict()
        for i, j in zip(t_ids, pred_target_texts):
            if not ids_target_text_dict.get(i):
                ids_target_text_dict[i] = j
            else:
                ids_target_text_dict[i] = ids_target_text_dict[i] + delimiter + j

        pred_target_texts = [ids_target_text_dict[k] for k in sorted(ids_target_text_dict.keys())]
        return pred_target_texts  # type:list

    @staticmethod
    def revise_target_texts(target_texts: list, input_texts: list, check_in_input_text: bool, delimiter='|'):
        """revise the target texts,

        Args:
            target_texts: the list of the target_texts
            input_texts:  the list of the input_texts
            check_in_input_text: if check the entities in input_text
            delimiter: the delimiter in target_text to split different entities

        Returns:
            revised_target_texts = list[set]
        """
        revised_target_texts = [MT5Utils.revise_target_text(t_t, return_format='set', delimiter=delimiter) for
                                t_t in target_texts]  # type:list[set,...]
        if check_in_input_text:
            revised_target_texts = MT5Utils.keep_entities_in_input_text(input_texts, revised_target_texts)
        return revised_target_texts  # type:list[set]

    @staticmethod
    def revise_target_text(target_text: str, delimiter: str = '|', return_format='set'):
        """ revise the target text

        Args:
            target_text: str, target_text
            return_format: 'set' means:'every entity is an element in a set', 'str' means: different entities are split
                            by the delimiter
            delimiter: the delimiter in target_text to split different entities

        Returns:
            revised_target_text : set or list
        """
        assert isinstance(target_text, str)
        target_text = target_text.split(delimiter)
        target_text = set([' '.join(e.strip().split()) for e in target_text])
        if '' in target_text:
            target_text.remove('')
        if return_format == 'set':
            revised_target_text = target_text
        elif return_format == 'list':
            revised_target_text = list(target_text)
        else:  # return_format == 'str'
            revised_target_text = '|'
            if target_text != set():
                for entity in list(target_text):
                    revised_target_text += (str(entity) + '|')
        return revised_target_text

    @staticmethod
    def keep_entities_in_input_text(input_texts: list, target_texts: list):
        """for each sample, for every entity ,keep the entities that are in the input text,and remove other entities

        Args:
            input_texts: the list of many input_text,and every input text is a string
            target_texts: the list of many target_text,and evert target text is a set

        Returns:
            revise_target_texts: list[str]
        """
        revised_target_texts = []
        for input_text, target_text in zip(input_texts, target_texts):
            if target_text != set():
                elements = list(target_text)
                for e in elements:
                    if str(e) not in input_text:
                        target_text.remove(e)  # type:set
            revised_target_texts.append(target_text)
        return revised_target_texts  # type:list[set]

    @staticmethod
    def entity_recognition_v2(y_true: list, y_pred: list, pos_neg_ratio: str = None):
        """the metric of entity_recognition, version-v2, reference: https://docs.qq.com/doc/DYXRYQU1YbkVvT3V2

        Args:
            y_true: the list of true target texts,each element is a set
            y_pred: the list of pred target texts,each element is a set
            pos_neg_ratio: the ratio of positive and negative sample importance, default: the ratio of positive and
                           negative sample sizes, you can set it,like"7:3"

        Returns:
            show report and res
        """
        neg_data = 0
        neg_correct_dt = 0
        neg_wrong_dt = 0
        neg_redundant_entities = 0

        pos_data = 0
        pos_correct_dt = 0
        pos_wrong_dt = 0
        pos_correct_entities = 0
        pos_wrong_entities = 0
        pos_omitted_entities = 0
        pos_redundant_entities = 0

        for i, j in zip(y_true, y_pred):
            if i == set():
                neg_data += 1
                if j == set():
                    neg_correct_dt += 1
                else:
                    neg_wrong_dt += 1
                    neg_redundant_entities += len(j)
            else:
                pos_data += 1
                true_pred = len(i & j)
                pos_correct_entities += true_pred

                if i == j:
                    pos_correct_dt += 1
                elif len(i) >= len(j):
                    pos_wrong_dt += 1
                    pos_wrong_entities += (len(j) - true_pred)
                    pos_omitted_entities += (len(i) - len(j))
                else:
                    pos_wrong_dt += 1
                    pos_redundant_entities += (len(j) - len(i))
                    pos_wrong_entities += (len(i) - true_pred)

        all_pos_entities = pos_correct_entities + pos_wrong_entities + pos_omitted_entities + pos_redundant_entities
        if neg_data == 0:
            neg_metric = 0
        else:
            neg_metric = neg_correct_dt / (neg_correct_dt + neg_redundant_entities)
        if pos_data == 0:
            pos_metric = 0
        else:
            pos_metric = pos_correct_entities / all_pos_entities

        sum_metric_micro = (pos_correct_entities + neg_correct_dt) / (
                neg_correct_dt + neg_redundant_entities + all_pos_entities)
        # sum_metric_macro = neg_metric * 0.5 + pos_metric * 0.5

        if pos_neg_ratio:
            pos_all = float(pos_neg_ratio.split(':')[0])
            neg_all = float(pos_neg_ratio.split(':')[1])
            pos_ratio = pos_all / (pos_all + neg_all)
            neg_ratio = neg_all / (pos_all + neg_all)
        else:
            pos_ratio = pos_data / (pos_data + neg_data)
            neg_ratio = neg_data / (pos_data + neg_data)

        sum_metric_weighted = pos_ratio * pos_metric + neg_ratio * neg_metric
        # precision = pos_correct_dt / (neg_correct_dt + pos_correct_dt)
        # recall = pos_correct_dt / pos_data
        r = {
            'positive data': [str(pos_data), pos_correct_dt, pos_wrong_dt, pos_correct_entities,
                              pos_wrong_entities, pos_omitted_entities, pos_redundant_entities, pos_metric],
            'negative data': [neg_data, neg_correct_dt, neg_wrong_dt, '-', '-', '-', neg_redundant_entities,
                              neg_metric],

            'all data ': [str(pos_data + neg_data), neg_correct_dt + pos_correct_dt, neg_wrong_dt + pos_wrong_dt,
                          pos_correct_entities, pos_wrong_entities, pos_omitted_entities,
                          pos_redundant_entities + neg_redundant_entities,
                          sum_metric_micro],
            # 'precision': ['', '', '', '', '', '', '', precision],
            # 'recall': ['', '', '', '', '', '', '', recall],
            # 'f1 score': ['', '', '', '', '', '', '', (2 * precision * recall) / (precision + recall)],
            # 'accuracy score': ['', '', '', '', '', '', '', (neg_correct_dt + pos_correct_dt) / (pos_data + neg_data)],
            # 'micro score': ['', '', '', '', '', '', '', sum_metric_micro],
            # 'macro score': ['', '', '', '', '', '', '', sum_metric_macro],
            'weighted score': ['', '', '', '', '', '', '', sum_metric_weighted],
        }
        index = ['| data_num', '| correct_data', '| wrong_data', '| correct_entities', '| wrong_entities',
                 '| omitted_entities', '| redundant_entities', '| score']

        res_df = pd.DataFrame(r, index=index).T
        pd.set_option('precision', 4)
        pd.set_option('display.width', None)
        pd.set_option('display.max_columns', None)
        pd.set_option("colheader_justify", "center")
        print(res_df)

        print(
            f"正样本集得分为：{pos_correct_entities} / ({pos_correct_entities}+{pos_wrong_entities}+{pos_omitted_entities}+"
            f"{pos_redundant_entities}) = {round(pos_metric, 4)}，负样本集得分为：{neg_correct_dt} / ({neg_correct_dt} + "
            f"{neg_redundant_entities})={round(neg_metric, 4)}，",
            f"总体得分为： ({pos_correct_entities} + {neg_correct_dt}) / ({all_pos_entities}+{neg_correct_dt + neg_redundant_entities})={round(sum_metric_micro, 4)}。")

        return res_df

    @staticmethod
    def eval_by_auto_batch_size(job, eval_df, initial_eval_batch_size=600):
        """

        Args:
            job: you function. if run error, return None.
            eval_df: eval dataframe
            initial_eval_batch_size:

        Returns:

        """
        eval_batch_size = initial_eval_batch_size
        q = mp.Queue()
        pl = {'eval_batch_size': eval_batch_size}
        res = None
        while not res:
            eval_batch_size = int(eval_batch_size * 0.8)
            print(f'try eval_batch_size: {eval_batch_size}')
            pl['eval_batch_size'] = eval_batch_size
            eval_process = mp.Process(target=job, args=(pl, q, eval_df,))
            eval_process.start()
            eval_process.join()
            res = q.get()
            print(res)

    @staticmethod
    def eval_by_different_parameters(job, parameter_cfg: dict, eval_df):
        q = mp.Queue()
        parameters_list = MT5Utils.get_parameters_list(parameter_cfg)
        for pl in parameters_list:
            eval_process = mp.Process(target=job, args=(pl, q, eval_df,))
            eval_process.start()
            eval_process.join()
            print(q.get())

    @staticmethod
    def get_parameters_list(parameter_cfg: dict):
        """

        Args:
            parameter_cfg: like:{'truncating_size': [100,10], 'overlapping_size': [10],'max_seq_length':[100,30]}

        Returns:[{'truncating_size': 100, 'overlapping_size': 10, 'max_seq_length': 100}, {'truncating_size': 100,
                  'overlapping_size': 10, 'max_seq_length': 30}, {'truncating_size': 10, 'overlapping_size': 10,
                  'max_seq_length': 100}, {'truncating_size': 10, 'overlapping_size': 10, 'max_seq_length': 30}]

        """
        parameters_list = []
        keys = []
        values = []
        for i, j in parameter_cfg.items():
            keys.append(i)
            values.append(j)
        for para in product(*values):  # 求多个可迭代对象的笛卡尔积
            cfg = dict(zip(keys, para))
            parameters_list.append(cfg)
        return parameters_list  # type:list

    @staticmethod
    def cut_entities(input_entities: list, prefixes: list):
        assert len(input_entities) == len(prefixes)  # a input_text corresponds a prefix
        input_texts_ids = range(len(input_entities))

        cut_ids = []
        cut_input_entities = []
        cut_prefixes = []

        for id, i_e, p in zip(input_texts_ids, input_entities, prefixes):
            if not isinstance(i_e, set):
                cut_i_e = MT5Utils.revise_target_text(target_text=i_e, return_format='set', delimiter='|')
            else:
                cut_i_e = i_e
            if cut_i_e != set():
                for c_i_t in cut_i_e:
                    cut_ids.append(id)
                    cut_input_entities.append(c_i_t)
                    cut_prefixes.append(p)
        return cut_ids, cut_input_entities, cut_prefixes  # type:list

    @staticmethod
    def combine_cut_entities(cut_entities: list, cut_ids: list):
        dic = dict()
        for i, j in zip(cut_ids, cut_entities):
            if i not in dic.keys():
                dic[i] = j
            else:
                if isinstance(j, str):
                    dic[i] = dic[i] + '|' + j
                else:
                    dic[i].update(j)
        return dic


if __name__ == '__main__':
    pass
