import re
import math
import copy
import numpy as np
from typing import List, Optional


class keywords():

    def __init__(self, inds_cdd: Optional[List[str]], ids_cdd: Optional[List[str]], indications):
        self.inds_cdd = inds_cdd
        self.ids_cdd = ids_cdd
        self.indications = indications

    # 得到中位数
    def get_median(self, data):
        data = sorted(data)
        size = len(data)
        if size % 2 == 0:  # 判断列表长度为偶数
            median = (data[size // 2] + data[size // 2 - 1]) / 2
            data[0] = median
        if size % 2 == 1:  # 判断列表长度为奇数
            median = data[(size - 1) // 2]
            data[0] = median
        return data[0]

    # 在文章中查找适应症id对应的所有名字，得到位置频率和句子占比
    def find_inds(self, ids, article):
        article = article.lower()
        article = article.replace('-', '')
        article = article.replace(' ', '')
        article = article.replace('!', '。')
        article = article.replace('?', '。')

        char_s = re.split("。", article)
        n_sen = (len(char_s)) - 1
        if n_sen == 0:
            n_sen = 1

        sen = []
        count = []
        ppp = []
        for each in ids:
            s = 0
            c = 0
            for item in self.indications:
                if each == item.id:
                    each_name = item.all_name
            # each_name是当前id对应的所有名字的列表
            mm = []
            for ea in each_name:
                ea = ea.lower()
                ea = ea.replace('-', '')
                ea = ea.replace(' ', '')
                for con in char_s:
                    if ea in con:
                        s += 1
                pp = []
                for match in re.finditer(ea, article):
                    p = match.start()
                    # e = match.end()
                    c += 1
                    # print('Found {!r} at {:d}:{:d}'.format(article[p:e], p,e))
                    pp.append(p)
                if pp:
                    m1 = self.get_median(pp)
                    mm.append(m1)
            if mm:
                m = self.get_median(mm)
            else:
                m = len(article)
            # 取位置的中位数，如果没有出现，取为文章的长度
            sen.append(s / n_sen)
            count.append(c)
            ppp.append(m)

        # print('适应症关键词出现次数：',count)
        # print('出现的位置：',ppp)
        # print('句子占比',sen)
        return count, ppp, sen

    def ranking(self, abstract, content):

        k = 0.8
        w11 = 1.8
        w12 = 2.2
        w21 = 0.8
        w22 = 1.2
        th = 2.5
        # 参数
        content = re.sub("\n", " ", content)
        content = str(re.findall(r'(>.*?\<)', content))
        content = content.replace('<', '')
        content = content.replace('>', '')
        content = content.replace(',', '')
        content = content.replace("'", '')

        # print("title:",ar.get('title'))
        # print("inds in title and abstract:",inds_t)

        # print("在标题摘要中")
        c1, seq1, TSS = self.find_inds(self.ids_cdd, abstract)
        # print("在正文中")
        c2, seq2, TS = self.find_inds(self.ids_cdd, content)

        cm1 = np.mean(c1)
        cm2 = np.mean(c2)
        # print("TS",TS)

        # 把代表各项分值的数组相加，得到每个inds_tags的分值
        score = []
        TP1 = []
        TP2 = []
        TF1 = []
        TF2 = []
        c = len(c1)  # 关键词数量

        for i in range(c):
            TP1.append(math.log(math.log(3 + seq1[i])))
            TP2.append(math.log(math.log(3 + seq2[i])))
            TF1.append(c1[i] / (cm1 + k))
            TF2.append(c2[i] / (cm2 + k))

        for i in range(c):
            score.append((w11 * TP1[i] + w21 * TP2[i]) / (w12 * TF1[i] + w22 * TF2[i] + TS[i]))

        a = copy.copy(score)
        b = []
        for i in range(len(a)):
            b.append(a.index(min(a)))
            a[a.index(min(a))] = math.inf
        # b是关键词分数从小到大排序的索引

        # 阈值以上的词忽略
        kw = []
        scores = []
        k_ids = []
        for i in range(len(b)):
            if self.inds_cdd[b[i]] in kw or self.ids_cdd[b[i]] in k_ids:
                pass
            else:
                if score[b[i]] < th:
                    kw.append(self.inds_cdd[b[i]])
                    scores.append(score[b[i]])
                    k_ids.append(self.ids_cdd[b[i]])
                else:
                    pass
        return kw, k_ids, scores
