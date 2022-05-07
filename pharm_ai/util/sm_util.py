# utility functions for simpletransformers
import os.path

import datetime
import json
import os
import pandas as pd
import random
from loguru import logger
from seqeval.metrics import classification_report as cr_phrase
from simpletransformers.ner.ner_model import NERModel
from simpletransformers.t5 import T5Model
from sklearn.metrics import classification_report as cr_word
from typing import List

# from pharm_ai.util.ner_util.rule_ner import IndexedRuleNER
from pharm_ai.util.utils import Utilfuncs as ui


class SMUtil:
    # use this method after running "model.train_model(train, eval_df=test)"
    @staticmethod
    def auto_rm_outputs_dir(outputs_dir='outputs/'):
        if not os.path.isdir(outputs_dir):
            logger.error(outputs_dir + ' does not exist!')
        else:
            cmd = 'rm -rf ' + outputs_dir
            os.system(cmd)
            logger.info(outputs_dir + ' was successfully removed :)')

    # columns which must be included test_df: sentence_id, words, labels; predicted_labels is optional
    @staticmethod
    def eval_ner(test_df, title, model=None):
        groups = test_df.groupby('sentence_id')
        trues = []
        preds_ = []
        if_predicted_labels = False
        if 'predicted_labels' in test_df:
            if_predicted_labels = True
        for _, g in groups:
            trues.append(g['labels'].tolist())
            if if_predicted_labels:
                preds_.append(g['predicted_labels'].tolist())
        if not preds_:
            r, m, preds_ = model.eval_model(test_df)
        preds = []
        c = 0
        for t, p in zip(trues, preds_):
            if len(p) < len(t):
                c += 1
                p = ['O'] * len(t)
            preds.append(p)
        print('num of truncated sentences: ' + str(c))
        print('total num of tested sentences: ' + str(len(trues)))
        cp = cr_phrase(trues, preds, digits=4)
        print(' ' * 25 + title + ':')
        print(cp)
        print('*' * 100)
        trues = [i for j in trues for i in j]
        preds = [i for j in preds for i in j]
        cw = cr_word(trues, preds, digits=4)
        print(cw)
        print('#' * 100)
        print(' ')

    @staticmethod
    def ner_predict(text, model: NERModel, prepro, rule_ner=None, len_threhold=900):
        sents = prepro.tokenize_hybrid_text_generic(text)
        labeled_sens = []
        for sen in sents:
            sen = ' '.join(sen)
            if len(sen) > len_threhold:
                if rule_ner is not None:
                    labeled_sen = rule_ner.label_sentence(sen)
                else:
                    labeled_sen = SMUtil.set_to_non_entity(sen)
            else:
                pred, _ = model.predict([sen])
                labeled_sen = []
                for pr in pred[0]:
                    for k, v in pr.items():
                        one = [k, v]
                        labeled_sen.append(one)
            labeled_sens.append(labeled_sen)
        return labeled_sens

    @staticmethod
    def ner2xlsx(raw_ner_result, article_ids,
                 xlsx_path, sheet_name, label_li):
        today = datetime.datetime.today()
        t = '-' + today.strftime("%Y%m%d")
        c = 0
        res = []
        for text_res, aid in zip(raw_ner_result, article_ids):
            for sen_res in text_res:
                labels = [x + '_labels' for x in label_li]
                df = pd.DataFrame(sen_res, columns=['words'] + labels)
                sentence_id = str(c) + t
                df['sentence_id'] = sentence_id
                df['article_id'] = aid
                res.append(df)
                c += 1
        res = pd.concat(res).reset_index(drop=True)
        u.to_excel(res, xlsx_path, str(sheet_name))
        logger.info(xlsx_path + ' was saved.')

    @staticmethod
    def save_ner(raw_ner_result, article_ids,
                 file_path, label_li, sheet_name='default'):
        today = datetime.datetime.today()
        t = '-' + today.strftime("%Y%m%d")
        c = 0
        res = []
        for text_res, aid in zip(raw_ner_result, article_ids):
            for sen_res in text_res:
                labels = [x + '_labels' for x in label_li]
                df = pd.DataFrame(sen_res, columns=['words'] + labels)
                sentence_id = str(c) + t
                df['sentence_id'] = sentence_id
                df['article_id'] = aid
                res.append(df)
                c += 1
        res = pd.concat(res).reset_index(drop=True)

        file_suffix = file_path.split('.')[-1]
        effective_path = True
        if file_suffix == 'xlsx':
            u.to_excel(res, file_path, str(sheet_name))
        elif file_suffix == 'json':
            res.to_json(file_path)
        elif file_suffix == 'dict':
            res.to_dict(file_path)
        elif file_suffix == 'csv':
            res.to_csv(file_path)
        else:
            effective_path = False
        if effective_path:
            logger.info(file_path + ' was saved.')
        else:
            logger.info('No file was saved.')

    @staticmethod
    def hide_labels_arg(model, option="train"):
        def wrapper_train(*args, **kwargs):
            train_df = args[0]
            labels = train_df.labels.unique()
            model.args.labels_list = labels.tolist()
            logger.info("labels = {}", labels)
            result = model.train_model(*args, **kwargs)
            return result

        if isinstance(model, NERModel) and option == "train":
            return wrapper_train
        elif option == "predict":
            json_path = os.path.join(model.args.model_name, 'model_args.json')
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
            if "labels_list" in json_dict.keys():
                model.args.labels_list = json_dict['labels_list']
            return model

    @staticmethod
    def set_to_non_entity(sentence):
        return [[s, 'O'] for s in sentence.split(' ')]

    @staticmethod
    def recover_ner_entity(raw_result: List[List[List]], handle_incomplete_entity=False):
        """
        Transform NER labels of words to phrase entities.
        example:
            - input: raw_result=[[[['PD-1', 'O', 'b-target'], ['是', 'O', 'O'], ['啥', 'O', 'O'], ['？', 'O', 'O']],
                       [['心', 'b-disease', 'O'], ['脏', 'i-disease', 'O'], ['病', 'i-disease', 'O'], ['咋', 'O', 'O'],
                        ['治', 'O', 'O']]], [[['What', 'O', 'O'], ['is', 'O', 'O'], ['diabetes?', 'O', 'O']]]]
            - output: [{'target': ['PD-1'], 'disease': ['心脏病']}, {}]
        :param raw_result: List[paragraph: List[sentences: List[words result: List[words, *NER_labels]]]]
        :param bool handle_incomplete_entity: Whether to unify the entity labels of nearby previous normal entity
            and the entity labels of an imcoplete entity, into the label with major chars.
            eg. Process `['高', 'b-c3'], ['瓴', 'i-c2'], ['资', 'i-c2'], ['本', 'i-c2']` to `{'c2':'高瓴资本'}`
            instead of `{'c3':'高'}`.
        :return: List[paragraph: dict[NER_label->sentence_entities: List[entity]]]
        """
        label_class = [[*set(label.split('-')[1] for label in labels if label.find('-') == 1)]
                       for labels in zip(*(word[1:] for para in raw_result for sent in para for word in sent))]
        raw = [[[[word[0]] + [(l.split('-')[0] if l.split('-')[1] == lc else 'O') if l.find('-') > -1 else 'O'
                              for l, c in zip(word[1:], label_class) for lc in c]
                 for word in sent]
                for sent in para]
               for para in raw_result]
        res = ((ind_para, ind_sent, *word)
               for ind_para, para in enumerate(raw)
               for ind_sent, sent in enumerate(para)
               for word in sent)
        label_cls_all = [l for lc in label_class for l in lc]
        res_tb = pd.DataFrame(res, columns=['paragraph_id', 'sentence_id', 'word'] + label_cls_all)

        def fix_incomplete(df: pd.DataFrame):
            res = df.applymap(lambda x: {'O': 0, 'b': 1, 'i': 3}.get(x))
            dfd = res.apply(lambda s: s.diff().fillna(0).astype(int))
            incomplete_start_pos = dfd[(dfd == 3).any(axis=1).values]
            df_ = df.copy()
            for incomplete_start_pos_, incomplete_start_pos_each in incomplete_start_pos.iterrows():
                incomplete_grp = incomplete_start_pos_each[incomplete_start_pos_each == 3].index[0]
                incomplete_previous = incomplete_start_pos_each[incomplete_start_pos_each.isin([-1, -2, -3])]
                if incomplete_previous.empty:
                    continue
                previous_end_at_incomplete_start_grp = incomplete_previous.index[0]
                previous_dfd = dfd[:incomplete_start_pos_][previous_end_at_incomplete_start_grp]
                if not previous_dfd.isin([1, -2]).any():
                    continue
                previous_start_pos = previous_dfd[previous_dfd.isin([1, -2])].index[-1]
                incomplete_each_dfd = dfd[incomplete_start_pos_:][incomplete_grp]
                incomplete_end_pos = incomplete_each_dfd[incomplete_each_dfd.isin([-2, -3])].index[0]
                if incomplete_end_pos - incomplete_start_pos_ > incomplete_start_pos_ - previous_start_pos:
                    df_.loc[previous_start_pos:incomplete_start_pos_ - 1, incomplete_grp] = \
                        df.loc[previous_start_pos:incomplete_start_pos_ - 1,
                        previous_end_at_incomplete_start_grp].copy()
                    df_.loc[previous_start_pos:incomplete_start_pos_ - 1, previous_end_at_incomplete_start_grp] = 'O'
                else:
                    df_.loc[incomplete_start_pos_:incomplete_end_pos - 1, previous_end_at_incomplete_start_grp] = \
                        df.loc[incomplete_start_pos_:incomplete_end_pos - 1, incomplete_grp]
                    df_.loc[incomplete_start_pos_:incomplete_end_pos - 1, incomplete_grp] = 'O'
            return df_

        if handle_incomplete_entity:
            res_tb[label_cls_all] = fix_incomplete(res_tb[label_cls_all])

        def get_entity_list(ser, words):
            s = ser.map({'O': 0, 'b': 1, 'i': 3})
            s = s.append(pd.Series([0], index=[s.index[-1] + 1]))
            ds = s - pd.Series([0] + s.to_list()[:-1], index=s.index)
            sl = list(zip(s[(ds == 1) | (ds == 3) | (ds == -2)].index, s[(ds == -1) | (ds == -3) | (ds == -2)].index))
            sl = [sl_ for sl_ in sl if ds[sl_[0]] != 3]
            return [words.loc[sli[0]:sli[1] - 1].to_list() for sli in sl]

        # entity_list: List[paragraph: List[sentence: [entity_class: [entity: [token: str]]]]]
        entity_lists = [[[class_ser for class_ser in
                          sent_tb[label_cls_all].apply(lambda s: get_entity_list(s, sent_tb['word']),
                                                       result_type='reduce')]
                         for _, sent_tb in para_tb.groupby('sentence_id')]
                        for _, para_tb in res_tb.groupby('paragraph_id')]
        # entity_res: List[paragraph: List[sentence: List[entity: List[entity_phrase, entity_class]]]]
        entity_res = [[[[''.join(word), c_name] if all(
            '\u4e00' <= ch_ <= '\u9fa5' or ch_.isdigit() for ch_ in word) else [' '.join(word), c_name]
                        for c, c_name in zip(sent, label_cls_all) for word in c if word]
                       for sent in para] for para in entity_lists]
        entity_res = [
            {k: v[0].to_list() for k, v in pd.DataFrame(phrase for sent in para if sent for phrase in sent).groupby(1)}
            if any(sent for sent in para) else dict()
            for para in entity_res]
        return entity_res

    @staticmethod
    def ner_upsampling(df: pd.DataFrame, random_state=123, shuffle=True):
        """
        :param pandas.DataFrame df: input dataframe, including columns: 'sentence_id', 'words', 'labels'
        :param int random_state: set random state.
        :param bool shuffle: Whether to shuffle data at sentence level.
        :return: result dataframe.
        :rtype: pandas.DataFrame.

        Upsampling NER dataset to balance sentences that have all 'O' lables and ones not.
        example:
            >>> test_df
               sentence_id      word labels
            0          111      What      O
            1          111        is      O
            2          111  diabetes    b-d
            3          111         ?      O
            4          222        OK      O
            5          222         .      O
            6           33       How      O
            7           33       are      O
            8           33       you      O
            9           44      What      O
            10          44        is      O
            11          44      heat    b-d
            12          44   disease    i-d
            13          44         ?      O
            14          55     Hello      O
            15          55     world      O
            16          55         !      O
            >>> SMUtil.ner_upsampling(test_df)
               sentence_id      word labels
            0   1-20201202      What      O
            1   1-20201202        is      O
            2   1-20201202  diabetes    b-d
            3   1-20201202         ?      O
            0          111      What      O
            1          111        is      O
            2          111  diabetes    b-d
            3          111         ?      O
            4          222        OK      O
            5          222         .      O
            6           33       How      O
            7           33       are      O
            8           33       you      O
            9           44      What      O
            10          44        is      O
            11          44      heat    b-d
            12          44   disease    i-d
            13          44         ?      O
            14          55     Hello      O
            15          55     world      O
            16          55         !      O
        """
        is_all_O = df.groupby('sentence_id')['labels'].apply(lambda s: (s == 'O').all())
        all_Os = is_all_O[is_all_O].index.to_series()
        not_all_Os = is_all_O[~is_all_O].index.to_series()
        today_str = datetime.datetime.today().strftime('%Y%m%d')

        def generate_sentence_id():
            i = 1
            while True:
                res = '%d-%s' % (i, today_str)
                if res not in df['sentence_id']:
                    yield res
                i += 1

        g = generate_sentence_id()
        random.seed(random_state)
        if all_Os.size > not_all_Os.size:
            sel_df_grs = df[df['sentence_id'].isin(not_all_Os)].groupby('sentence_id')
            sel_df = random.choices([d for _, d in sel_df_grs], k=all_Os.size - not_all_Os.size)
        else:
            sel_df_grs = df[df['sentence_id'].isin(all_Os)].groupby('sentence_id')
            sel_df = random.choices([d for _, d in sel_df_grs], k=not_all_Os.size - all_Os.size)
        for d in sel_df:
            d['sentence_id'] = next(g)
        res = pd.concat([df, *sel_df])
        if shuffle:
            res = pd.concat([d for _, d in res.groupby('sentence_id')])
        return res

    @staticmethod
    def deal_with_target_text(to_predict_text: str, target_text: str, check_in_text: bool, delimiter='|'):
        """
        deal with a target text
        Args:
            to_predict_text: a text to predict, like: 'fruit: a man w ith no hair likes apple ,pear and orange'
            target_text: a t5 model's target text, like: '|apple|pear|sd|'
            check_in_text: True means removing the entities which not in predict_text
            delimiter : the symbol to split the different tokens
        #>>>SMUtil.deal_with_target_text(to_predict_text, target_text, check_in_text=True)
        Returns: format: {'entity_type':{'entity1','entity2'}}
        #>>>'fruit',{'apple', 'pear'}
        """
        entities = [str(i).strip() for i in target_text.split(delimiter)]
        while '' in entities:
            entities.remove('')
        prefix_input_text = to_predict_text.split(': ')
        if len(prefix_input_text) <= 1:
            prefix = 'None'
        else:
            prefix = prefix_input_text[0]

        entities_ = set(entities)
        if check_in_text:
            for entity in list(entities_):
                if entity not in to_predict_text:
                    entities_.remove(entity)
        return prefix, entities_

    @staticmethod
    def deal_with_target_texts(to_predict_texts: list, target_texts: list, check_in_text: bool,
                               return_format='dataframe', delimiter='|'):
        """
        deal with target texts and return a dataframe
        Args:
            to_predict_texts: a list of some to_predicted texts,like ['language: i like Chinese ,English,Japanese',
                                                             'fruit: a man w ith no hair likes apple ,pear and orange']
            target_texts: a list of some target texts,like ['|Chines|Eng|','|apple|pear|sd|']
            check_in_text: True means removing the entities which not in input_text
            delimiter : the symbol to split the different tokens
        #>>> SMUtil.deal_with_target_texts(to_predict_texts,target_texts,check_in_text=True)
        Returns: a entities list or a dataframe ,[{'Eng', 'Chines'}, {'apple', 'sd', 'pear'}] or
        #>>>    prefix  ...       entities
            0  language  ...  {Chines, Eng}
            1     fruit  ...  {pear, apple}
        """

        prefixes = []
        entities = []
        new_target_texts = []
        input_texts = []
        for i in range(len(target_texts)):
            prefix, entity = SMUtil.deal_with_target_text(to_predict_texts[i], target_texts[i], check_in_text,
                                                          delimiter)
            input_texts.append(to_predict_texts[i][len(prefix) + 2:])
            prefixes.append(prefix)
            entities.append(entity)
            new_target_text = '|'
            for j in list(entity):
                new_target_text += (str(j) + '|')
            new_target_texts.append(new_target_text)

        if return_format == 'dataframe':
            return pd.DataFrame(
                {'prefix': prefixes,
                 'input_text': input_texts,
                 'target_text': new_target_texts,
                 'entity': entities,
                 }
            )
        else:
            return entities

    @staticmethod
    def ner_predict_v2(texts: list, model: T5Model, check_in_text, return_format='dataframe', delimiter='|'):
        """
        use t5 model to predict a list of texts
        Args:
            texts: a list of texts to predict,like ['target: Use of sglt homolog',
                                                    'target: a man with no hair likes apple ,pear and orange']
            model: a trained t5 model
            return_format: 'dataframe'[clumns:['prefix','input_text','target_text','entities']],
                            or else(the entity's start location and end location)
            check_in_text: True means removing the entities which not in input_text
            delimiter : the symbol to split the different tokens
        #>>>SMUtil.ner_predict_v2(texts, self.model, check_in_text=True,return_format='df')
        Returns:
            df :    prefix                                 input_text        target_text  entity
                0  target                              Use of sglt homolog      |sglt|  {sglt}
               1  target  a man with no hair likes apple ,pear and orange           |      {}
        """
        predict_results = model.predict(texts)
        df_ = SMUtil.deal_with_target_texts(texts, predict_results, check_in_text, 'dataframe', delimiter)

        if return_format == 'dataframe':
            return df_
        elif return_format == 'to_save':
            save_columns = ['input_text']
            df_save = pd.DataFrame(columns=save_columns)
            for prefix_, prefix_df in df_.groupby('prefix'):
                prefix_df.rename(columns={'target_text': prefix_}, inplace=True)
                prefix_df = prefix_df[['input_text', prefix_]]
                df_save = pd.merge(df_save, prefix_df, how='outer', on=['input_text'])
            return df_save
        elif return_format == 'sdsd':
            res = []
            for index in range(len(texts)):
                prefix_entities = {df_.loc[index]['prefix']: df_.loc[index]['entity']}
                res_ = IndexedRuleNER.ner(df_.loc[index]['input_text'], prefix_entities, ignore_nested=True)
                res.append(res_)
            return res

    @staticmethod
    def eval_ner_v2(eval_data, model: T5Model, metrics_style='loose', check_in_text=True, delimiter='|'):
        """
        eval a dataframe by using a t5 model and a metric
        Args:
            eval_data: eval data:pd.Dataframe,columns:['prefix','input_text','target_text']
            model: a t5 model
            metrics_style: a evaluation standard, strict means: if the label tokens of a text is different from the
                           predicted tokens, mark 0; loose means: the result depends on the hit ratio
            check_in_text: True means removing the entities which not in input_text
            delimiter : the symbol to split the different tokens
        Returns: a dict
            like: {'disease':0.6492,'target':0.5901,'sum':0.6314}
        """
        prefixes = set(eval_data['prefix'].values)

        to_predict_text = list(eval_data['prefix'].map(str) + ': ' + eval_data['input_text'].map(str))
        pred_df = SMUtil.ner_predict_v2(to_predict_text, model, check_in_text, 'dataframe', delimiter)

        label_texts = list(eval_data['target_text'].map(str))
        label_df = SMUtil.deal_with_target_texts(to_predict_text, label_texts, False, 'dataframe', delimiter)
        res = {}
        top_sum = 0
        bottom_sum = 0
        for prefix in prefixes:
            entity_label = list(label_df[label_df['prefix'] == prefix]['entity'])
            entity_pred = list(pred_df[pred_df['prefix'] == prefix]['entity'])
            top = 0
            bottom = 0

            if metrics_style == 'strict':
                for i, j in zip(entity_label, entity_pred):
                    is_same = 1 if i == j else 0
                    top += is_same
                    bottom += 1
            elif metrics_style == 'loose':
                for i, j in zip(entity_label, entity_pred):
                    if i == set() and j == set():
                        top += 1
                        bottom += 1
                    else:
                        top += len(i & j)
                        bottom += max(len(i), len(j))
            else:
                for i, j in zip(entity_label, entity_pred):
                    top += len(i & j)
                    bottom += max(len(i), len(j))
            res[prefix] = top / bottom
            top_sum += top
            bottom_sum += bottom
        res['sum'] = top_sum / bottom_sum
        return res

    @staticmethod
    def parallel_predict(model_class, to_predict, model_init_args=dict(), cuda_devices=[0, 1], **kwargs):
        if isinstance(to_predict, dict):
            ks = list(to_predict.keys())
            batch_size = len(ks) // 2
            ks1, ks2 = ks[:batch_size], ks[batch_size:]
            batch1 = {k: v for k, v in to_predict.items() if k in ks1}
            batch2 = {k: v for k, v in to_predict.items() if k in ks2}
        elif isinstance(to_predict, list):
            batch_size = len(to_predict) // 2
            batch1, batch2 = to_predict[:batch_size], to_predict[batch_size:]
        cuda_iter = cycle(cuda_devices)

        def do_predict(to_predict):
            select_cuda = next(cuda_iter)
            model = model_class(cuda_device=select_cuda, **model_init_args)
            model.model.args.update_from_dict(kwargs)
            res_batch = model.predict(to_predict)
            return res_batch

        with ThreadPoolExecutor(max_workers=len(cuda_devices)) as executor:
            res_iter = executor.map(do_predict, [batch1, batch2])
        res = list(res_iter)
        result = res[0]
        if isinstance(result, dict):
            result.update(res[1])
        elif isinstance(result, list):
            result.extend(res[1])
        return result


if __name__ == '__main__':
    # predict_text = 'fruit: a man w ith no hair likes apple ,pear and orange'
    # target_text = '|apple|pear|sd'
    # a = SMUtil.deal_with_target_text(predict_text, target_text, check_in_text=True)
    # print(a)
    ##########
    # input_texts = ['language: i like Chinese ,English,Japanese',
    #                'fruit: a man w ith no hair likes apple ,pear and orange']
    # target_texts = ['|Chines|Eng|', '|apple|pear|sd|']
    # a = SMUtil.deal_with_target_texts(input_texts, target_texts, check_in_text=True, return_format='dataframe')
    # print(a)
    print('end')
