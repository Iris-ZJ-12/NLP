# -*- coding: utf-8 -*-
import re
# from harvesttext import HarvestText
import langid
from nltk.tokenize import PunktSentenceTokenizer
import datetime
import pandas as pd
import random
from pharm_ai.util.utils import Utilfuncs as u



class Prepro:
    def __init__(self):
        self.ht = HarvestText()  # cn
        self.pst = PunktSentenceTokenizer()  # en

        # jp

    # # used for tokenizing one sentence
    # def tokenize_hybrid_text(self, text, sep_end_punc=True):
    #     # separate Chinese characters
    #     pattern = re.compile(r'([\u4e00-\u9fa5]|[、，/])')
    #     chars = pattern.split(text)
    #     chars_ = [w for w in chars if len(w) > 0]
    #     chars = []
    #     # separate English words
    #     for char in chars_:
    #         if ' ' in char:
    #             # keep spaces if necessary
    #             # char_li = [c for c in re.split('(\s+)', char) if c]
    #             chars.extend(char.split())
    #         else:
    #             chars.append(char)
    #
    #     # separate ending punctuation
    #     res_ = []
    #     if sep_end_punc:
    #         for char in chars:
    #             if len(char) > 1:
    #                 if not char[-1].isalnum():
    #                     res_.extend([char[:-1], char[-1]])
    #                 else:
    #                     res_.append(char)
    #             else:
    #                 res_.append(char)
    #     res = []
    #     for char in res_:
    #         if '(' in char and ')' in char:
    #
    #             res.extend(re.split('(\\()|(\\))', char))
    #         elif '（' in char and '）' in char:
    #
    #             res.extend(re.split('(（)|(）)', char))
    #         else:
    #             res.append(char)
    #
    #     chars = [c for c in res if c]
    #
    #     anti_prefixes = ['anti-', 'anti–', 'Anti-', 'ANTI-', 'Anti–', 'ANTI–']
    #     for anti in anti_prefixes:
    #         res_tmp = []
    #         for c in chars:
    #             if c.startswith(anti) and len(c) > 5:
    #                 res_tmp.extend([anti, c[5:]])
    #             else:
    #                 res_tmp.append(c)
    #         chars = res_tmp
    #
    #     return chars
    #

    # used for tokenizing one sentence
    def tokenize_hybrid_text(self, text, sep_end_punc=True):
        # separate Chinese characters
        pattern = re.compile(r'([\u4e00-\u9fa5]|[、，/])')
        chars = pattern.split(text)
        chars_ = [w for w in chars if len(w) > 0]
        chars = []
        # separate English words
        for char in chars_:
            if ' ' in char:
                # keep spaces if necessary
                # char_li = [c for c in re.split('(\s+)', char) if c]
                chars.extend(char.split())
            else:
                chars.append(char)

        res = []
        for char in chars:
            if '(' in char and ')' in char:
                res.extend(re.split('(\\()|(\\))', char))
            elif '（' in char and '）' in char:
                res.extend(re.split('(（)|(）)', char))
            else:
                res.append(char)
        res = [r for r in res if r]

        # separate ending punctuation
        res_ = []
        if sep_end_punc:
            for char in res:
                if len(char) > 1:
                    if not char[-1].isalnum():
                        res_.extend([char[:-1], char[-1]])
                    else:
                        res_.append(char)
                else:
                    res_.append(char)


        chars = [c for c in res_ if c]

        chars__ = []
        for char in chars:
            if len(char) > 1:
                if not char[0].isalnum():
                    chars__.extend([char[0], char[1:]])
                elif not char[-1].isalnum():
                    chars__.extend([char[:-1], char[-1]])
                else:
                    chars__.append(char)
            else:
                chars__.append(char)
        chars = chars__

        anti_prefixes = ['anti-', 'anti–', 'Anti-', 'ANTI-', 'Anti–', 'ANTI–']
        for anti in anti_prefixes:
            res_tmp = []
            for c in chars:
                if c.startswith(anti) and len(c) > 5:
                    res_tmp.extend([anti, c[5:]])
                else:
                    res_tmp.append(c)
            chars = res_tmp

        return chars

    # used for tokenizing text containing multiple sentences
    def tokenize_hybrid_text_generic(self, text):
        lang = langid.classify(text)[0]
        if lang == 'zh':
            sents = self.ht.cut_sentences(text)
            sents = [self.tokenize_hybrid_text
                     (sen, sep_end_punc=True) for sen in sents]
        else:
            sents_idx = self.pst.span_tokenize(text)
            sents = []
            for sid in sents_idx:
                sen = text[sid[0]:sid[1]]
                sen = self.tokenize_hybrid_text(str(sen))
                sents.append(sen)
        return sents

    # prep ner data which to be predicted later
    # texts: list of texts, each text can either be one sentence or
    # contain multiple sentences.
    def bert_ner_prepro(self, texts):
        res = []
        for text in texts:
            sents = self.tokenize_hybrid_text_generic(text)
            sents = [' '.join(sent) for sent in sents]
            res.append(sents)
        return res

    # prepare training data for NER with default labels 'O', save to excel
    # texts: a list of paragraphs, each paragraph may contain one or more sentences.
    # shuffle between sentences, orders of words within one sentence remain unchanged.
    def prep_ner_train_dt(self, texts, excel_path, shuffle=True):
        today = datetime.datetime.today()
        t = '-' + today.strftime("%Y%m%d")
        c = 0
        sen_ids_all = []
        words_all = []
        labels_all = []
        for text in texts:
            sents = self.tokenize_hybrid_text_generic(str(text))
            for sen in sents:
                words_all.extend(sen)
                labels = ['O'] * len(sen)
                labels_all.extend(labels)
                sen_ids = [str(c) + t] * len(sen)
                sen_ids_all.extend(sen_ids)
                c += 1
        df_ = pd.DataFrame({'words': words_all, 'labels': labels_all,
                            'sentence_id': sen_ids_all})

        if shuffle:
            df = [df for _, df in df_.groupby('sentence_id')]
            random.shuffle(df)
            df_ = pd.concat(df).reset_index(drop=True)
        u.to_excel(df_, excel_path, 'unlabeled' + t)

    # used for tokenizing texts containing multiple sentences
    @staticmethod
    def cut_sentences(texts):
        import nltk
        sent_tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
        lang = langid.classify(texts)[0]
        if lang == 'zh':
            sentences = HarvestText().cut_sentences(texts)
        else:
            sentences = sent_tokenizer.tokenize(texts)
        return sentences


if __name__ == '__main__':
    p = Prepro()
    d = p.cut_sentences('A promoter sequence of the human p55 TNF-R gene is provided.')
    f = 'A promoter sequence of the human p55 TNF-R gene is provided.'
    import re
    re.search(f,'promoter')
    print(d)
