import ast
import json
import pickle
import re
from os import path

from loguru import logger

# from pharm_ai.patent_ner.dt import get_entity_dics
from pharm_ai.util.prepro import Prepro


class RuleNER:
    def __init__(self, dic=None, dic_path=None):
        self.prepro = Prepro()
        self.dic = self.prep_dic(dic, dic_path)

    def prep_dic(self, dic, dic_path):
        if not path.exists(dic_path) and dic:
            dic = self.label_dic(dic)
            dic_f = open(dic_path, 'wb')
            pickle.dump(dic, dic_f)
            dic_f.close()
            f = open(dic_path, 'rb')
            dic = pickle.load(f)
            logger.info(dic_path + ' was saved and loaded.')
        else:
            f = open(dic_path, 'rb')
            dic = pickle.load(f)
            logger.info(dic_path + ' was loaded.')
        return dic

    def remove_sub_li(self, li):
        curr_res = []
        result = []
        for ele in sorted(map(set, li), key=len, reverse=True):
            if not any(ele <= req for req in curr_res):
                curr_res.append(ele)
                result.append(list(ele))
        return result

    # def stratify_entity_list(self, entity_li):
    #     new_entity_li = []
    #     for entity in entity_li:
    #         union = [ent for ent in entity_li if (entity in ent or ent in entity)]
    #         new_entity_li.append(union)
    #     new_entity_li = self.remove_sub_li(new_entity_li)
    #     new_entity_li = [sorted(li, key=len, reverse=True) for li in new_entity_li]
    #     return new_entity_li

    def stratify_entity_list(self, entity_dic):
        entity_pairs = []
        for entity_type, entity_li in entity_dic.items():
            entity_pair_li = [[entity_type, entity] for entity in entity_li]
            entity_pairs.extend(entity_pair_li)
        new_entity_li = []
        for entity_pair in entity_pairs:
            union = [ent for ent in entity_pairs if (entity_pair[1] in ent[1] or ent[1] in entity_pair[1])]
            new_entity_li.append([str(x) for x in union])
        nnew_entity_li = []
        for incl_entities in self.remove_sub_li(new_entity_li):
            nnew_entity_li.append([ast.literal_eval(pair) for pair in incl_entities])
        # sort inclusive entities by entity string length in descending order
        new_entity_li = [sorted(li, key=lambda x: len(x[1]), reverse=True) for li in nnew_entity_li]
        return new_entity_li

    def label_entity(self, entity_type, entity_sub_li):
        labeled_entity_sub_li = dict()
        labels = []
        begin_label = 'b-' + str(entity_type)
        inside_label = 'i-' + str(entity_type)
        labels.append(begin_label)
        if len(entity_sub_li) > 1:
            inside_labels = [inside_label] * len(entity_sub_li[1:])
            labels.extend(inside_labels)
        labeled_entity_sub_li.update({'entity_tokens': entity_sub_li,
                                      'labels': labels,
                                      'entity_tokens#': len(entity_sub_li)})
        return labeled_entity_sub_li

    # def label_entity_li(self, entity_type, entity_li):
    #     res = [[self.label_entity(entity_type, tuple(self.prepro.tokenize_hybrid_text(entity)))
    #             for entity in sub_li] for sub_li in entity_li]
    #     return res
    #
    # def label_entity_li_(self, entity_type_li, entity_li):
    #     res = [[self.label_entity(entity_type, tuple(self.prepro.tokenize_hybrid_text(entity)))
    #             for entity in sub_li] for entity_type, sub_li in zip(entity_type_li, entity_li)]
    #     return res

    def remove_stopwords(self, dic, stopwords_li):
        new_dic = dict()
        for entity_type, entities in dic.items():
            entities = [ent for ent in entities if ent not in stopwords_li]
            new_dic[entity_type] = entities
        return new_dic

    def remove_dic_stopwords(self, dic):
        p_en = 'en_stopwords.json'
        p_cn = 'cn_stopwords.json'
        d_en = json.load(open(p_en, 'r'))['words']
        d_cn = json.load(open(p_cn, 'r'))['words']
        dic = self.remove_stopwords(dic, d_en)
        dic = self.remove_stopwords(dic, d_cn)
        return dic

    def label_dic(self, dic):
        dic = self.remove_dic_stopwords(dic)
        stratified_entities = self.stratify_entity_list(dic)
        res = []
        for se in stratified_entities:
            labels, entities = map(list, zip(*se))
            entity_tokenized_li = [tuple(self.prepro.tokenize_hybrid_text(entity))
                                   for entity in entities]
            sub_res = []
            for entity_type, entity_tokens in zip(labels, entity_tokenized_li):
                sub_res.append(self.label_entity(entity_type, entity_tokens))
            res.append(sub_res)
        return res

    def find_sub_list(self, small_li, large_li):
        results = []
        sll = len(small_li)
        for ind in (i for i, e in enumerate(large_li) if e == small_li[0]):
            if large_li[ind:ind + sll] == small_li:
                results.append((ind, ind + sll - 1))
        return results

    def label_sentence_by_1_entity(self, sent_tokens, entity_sub_li, sent_labels):
        for i, entity in enumerate(entity_sub_li):
            entity_tokens = list(entity['entity_tokens'])
            entity_labels = entity['labels']

            entity_tokens = [etok.lower() for etok in entity_tokens]
            entity_tokens_n = entity['entity_tokens#']
            sent_labels_indices = self.find_sub_list(entity_tokens, sent_tokens)
            if sent_labels_indices:
                for labels_indices in sent_labels_indices:
                    sent_labels[labels_indices[0]: labels_indices[1] + 1] = entity_labels
                    sent_tokens[labels_indices[0]: labels_indices[1] + 1] = ['#labeled#'] * entity_tokens_n

        return sent_tokens, sent_labels

    def label_sentence_by_entity_type(self, sent_tokens_temp, sent_labels, entities):
        for entity_sub_li in entities:
            # sent_tokens_temp = [str(token).lower() for token in sent_tokens_temp]
            # entity_sub_li = self.lower_entity_sub_li(entity_sub_li)
            sent_tokens_temp, sent_labels = \
                self.label_sentence_by_1_entity(sent_tokens_temp, entity_sub_li, sent_labels)
        return sent_tokens_temp, sent_labels

    # def lower_entity_sub_li(self, entity_sub_li):
    #     res = []
    #     for ent in entity_sub_li:
    #         entity_tokens = tuple([str(token).lower() for token in ent['entity_tokens']])
    #         ent['entity_tokens'] = entity_tokens
    #         res.append(ent)
    #     return res

    def label_sentence(self, sentence):
        sent_tokens_temp = self.prepro.tokenize_hybrid_text(sentence, sep_end_punc=True)
        sent_tokens_temp = [tok.lower() for tok in sent_tokens_temp]
        sent_labels = ['O'] * len(sent_tokens_temp)
        sent_tokens_temp, sent_labels = \
            self.label_sentence_by_entity_type(sent_tokens_temp, sent_labels, self.dic)
        sent_tokens = self.prepro.tokenize_hybrid_text(sentence, sep_end_punc=True)
        return [list(x) for x in zip(sent_tokens, sent_labels)]


class RuleNER2:
    def __init__(self):
        self.irn = IndexedRuleNER

    def label_sentence(self, text: str, dic: dict, return_format: str = 'target_text',
                       ignore_nested: bool = True) -> object:
        """
        use dict to label sentence
        Args:
            text: a text, usually a sentence, like: '我不吃苹果，梨子， 红富士，紫苹果，蓝苹果，兔子，红兔子，绿兔子，蓝兔子。'
            dic: a dict like: {'fruits':{'苹果'，'梨子'， '红富士'},'animals':{'兔子'，'耗子'，'虎子'，'狮子'}}
            return_format: 'set'(default) / 'target_text' / 'list' /'locate_entities'

        Returns: depending on return_format
            locate_entities:{'fruits': [[3, 5, '苹果'], [19, 21, '苹果'], [6, 8, '梨子'], [14, 17, '紫苹果'], [10, 13, '红富士']],
                             'animals': [[22, 24, '兔子'], [26, 28, '兔子'], [30, 32, '兔子'], [34, 36, '兔子']]}
            target_text: {'fruits': '|红富士|紫苹果|苹果|梨子|', 'animals': '|兔子|'}
            set: {'fruits': {'梨子', '紫苹果', '苹果', '红富士'}, 'animals': {'兔子'}}
            list: {'fruits': ['红富士', '苹果', '梨子', '紫苹果'], 'animals': ['兔子']}
        """
        res = dict()
        for entity_type, entities in dic.items():
            res_list = set()
            for entity in set(entities):
                if entity in text:
                    res_list.add(entity)
            res[entity_type] = res_list
        if return_format == 'target_text':
            res_ = dict()
            for entity_type, entities in res.items():
                res_target_text = '|'
                for entity in entities:
                    res_target_text += (str(entity) + '|')
                res_[entity_type] = res_target_text
            return res_
        elif return_format == 'locate_entities':
            res_ = self.irn.ner(text, res, ignore_nested=ignore_nested)
            return res_
        elif return_format == 'list':
            return {i: list(j) for i, j in res.items()}
        else:
            return res


class IndexedRuleNER:
    def __init__(self):
        pass

    @staticmethod
    def ner(text: str, entities: dict, ignore_nested=True):
        """
        find the loaction of entities in a text
        Args:
            text: a text, like '我爱吃苹果、大苹果，小苹果，苹果【II】，梨子，中等梨子，雪梨，梨树。'
            entities: {'entity_type1':{entity_str1,entity_str2...},
                       'entity_type2':{entity_str1,entity_str2...},
                       ...}
                       like : {'apple': ['苹果', '苹果【II】'], 'pear': ['梨', '梨子'],}
            ignore_nested: if nested
        #>>>IndexedRuleNER().ner(text, entities, False)
        Returns:
            indexed_entities:{'entity_type1':[[start_index,end_index,entity_str],
                                              [start_index,end_index,entity_str]...]
                              'entity_type2':[[start_index,end_index,entity_str],
                                              [start_index,end_index,entity_str]...]
                                              ...}
        #>>>{'apple': [[3, 5, '苹果'], [7, 9, '苹果'], [11, 13, '苹果'], [14, 16, '苹果'], [14, 20, '苹果【II】']],
        'pear': [[21, 22, '梨'], [26, 27, '梨'], [30, 31, '梨'], [32, 33, '梨'], [21, 23, '梨子'], [26, 28, '梨子']]}
        """
        indexed_entities = dict()
        for every_type, every_value in entities.items():
            every_type_value = []
            for every_entity in list(every_value):
                special_character = set(re.findall('\W', str(every_entity)))
                for i in special_character:
                    every_entity = every_entity.replace(i, '\\' + i)
                re_result = re.finditer(every_entity, text)
                for i in re_result:
                    res = [i.span()[0], i.span()[1], i.group()]
                    if res != []:
                        every_type_value.append([i.span()[0], i.span()[1], i.group()])
            indexed_entities[every_type] = every_type_value
        if ignore_nested:
            for key, value in indexed_entities.items():
                all_indexs = [set(range(i[0], i[1])) for i in value]
                for i in range(len(all_indexs)):
                    for j in range(i, len(all_indexs)):
                        if i != j and all_indexs[j].issubset(all_indexs[i]):
                            value.remove(value[j])
                            indexed_entities[key] = value
                        elif i != j and all_indexs[i].issubset(all_indexs[j]):
                            value.remove(value[i])
                            indexed_entities[key] = value
        return indexed_entities


if __name__ == '__main__':
    # text2 = r'我爱吃苹果、大苹果，小苹果，苹果【II】，梨子，中等梨子，雪梨，梨树。'
    # entities = {
    #     'apple': ['苹果', '苹果【II】'],
    #     'pear': ['梨', '梨子'],
    # }
    # re1 = IndexedRuleNER().ner(text2, entities, False)
    # print(re1)
    import pandas as pd

    # s = '4. The method of claim 1, wherein the condition is selected from the group consisting of skin cancers, ' \
    #     'cancers of the central nervous system, cancers of the gastrointestinal tract.'
    # file = "/home/zyl/disk_a/PharmAI/pharm_ai/util/ner_util/online_entities-20201104.xlsx"
    # dic1 = pd.read_excel(file, sheet_name='disease')
    # dic2 = pd.read_excel(file, sheet_name='target')
    # dic1 = dic1.to_dict(orient='list')
    # dic2 = dic2.to_dict(orient='list')
    # dic1.update(dic2)
    s = '我不吃苹果，梨子， 红富士，紫苹果，蓝苹果，兔子，红兔子，绿兔子，蓝兔子。'
    d = {'fruits': {'苹果', '梨子', '紫苹果', '红富士'}, 'animals': {'兔子', '耗子', '虎子', '狮子'}}
    r = RuleNER2().label_sentence(s, d, return_format='set')
    print(r)
    # R = RuleNER(dic_path=file)
    #
    # print(R.dic)
