import os
import json
import itertools
import re
import math
import copy
import datetime
from wsgiref.headers import tspecials
import numpy as np
import pandas as pd
from pandas import Series
import sys
sys.path.append('PharmAI-master')
from config import ConfigFilePaths
from typing import List, Union, Optional
import rule_match
import unicodedata

#需要查找关键词的文章路径
path1 = '/home/zj/PharmAI/pharm_ai/kwranking/ran.json'
with open(path1, 'r+',encoding='UTF-8-sig') as f:
    print("Load str file from {}".format(path1))
    str1 = f.read()
    orig_ar = json.loads(str1)
#词典路径
path2 = '/home/zj/PharmAI/pharm_ai/kwranking/indications.json'
with open(path2, 'r+',encoding='utf-8') as f:
    print("Load str file from {}".format(path2))
    str2 = f.read()
    indications = json.loads(str2)

#得到中位数
def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0: # 判断列表长度为偶数
        median = (data[size//2]+data[size//2-1])/2
        data[0] = median
    if size % 2 == 1: # 判断列表长度为奇数
        median = data[(size-1)//2]
        data[0] = median
    return data[0]

#在文章中查找适应症id对应的所有名字，得到位置频率和句子占比
def find_inds(ids,article):
    article = article.lower()
    article = article.replace('-','')
    char_s = re.split("。",article)
    n_sen = (len(char_s))-1
    if n_sen==0:
        n_sen=1
    sen=[]
    count = []
    ppp = []
    for each in ids:
        s=0
        c=0
        m=0
        for item in indications:
            if each == item.get('_id'):
                if 'name_synonyms' not in item or item.get('name_synonyms')=="":
                    each_name=[item.get('name'),item.get('name_en')]
                else:
                    each_name=[item.get('name'),item.get('name_en')]+item.get('name_synonyms')
        #each_name是当前id对应的所有名字的列表
        mm=[]
        for ea in each_name:
            ea=ea.lower()
            ea=ea.replace('-','')
            ea=ea.replace(' ','')
            for con in char_s:
                if ea in con:
                    s+=1
            pp=[]
            for match in re.finditer(ea,article):
                p = match.start()
                e = match.end()
                c += 1
                #print('Found {!r} at {:d}:{:d}'.format(article[p:e], p,e))  
                pp.append(p)
            if pp:
                m1=get_median(pp)
                mm.append(m1)
        if mm:
            m=get_median(mm)
        else:
            m=len(article)
        #取位置的中位数，如果没有出现，取为文章的长度
        sen.append(s/n_sen)
        count.append(c)
        ppp.append(m)

    #print('适应症关键词出现次数：',count)
    #print('出现的位置：',ppp)
    #print('句子占比',sen)
    return count,ppp,sen
    

def find_id(name):
    id=[]
    for n in name:
        for ind in indications:
            if 'name_synonyms' not in ind or not ind.get('name_synonyms'):
                if n==ind.get('name') and len(n)==len(ind.get('name')):
                    id.append(ind.get('_id'))
                else:
                    if n==ind.get('name_en') and len(n)==len(ind.get('name_en')):
                        id.append(ind.get('_id'))
            else:
                if n==ind.get('name') and len(n)==len(ind.get('name')):
                    id.append(ind.get('_id'))
                else:
                    if n==ind.get('name_en') and len(n)==len(ind.get('name_en')):
                        id.append(ind.get('_id'))
                    else:
                        for item in ind.get('name_synonyms'):
                            if n==item and len(n)==len(item):
                                id.append(ind.get('_id'))
    return id

k=0.8
w11=1.8
w12=2.2
w21=0.8
w22=1.2
th=2.5
#参数
results=[]

for ar in orig_ar:
    if 'tags' not in ar:
       pass
    else:
        words = ar.get('content')
        words = re.sub("\n"," ",words)
        content = str(re.findall(r'(>.*?\<)',words))
        content = "".join(content.split())
        content = unicodedata.normalize("NFKD", content)
        content = content.replace('<','')
        content = content.replace('>','')
        content = content.replace(',','')
        content = content.replace("'",'')
        content = content.replace('!','。')
        content = content.replace('?','。')
        content = content.replace(' ','')
        if ar.get('abstract') == None:
            text="\n" + ar.get('title') + "。"
        else:
            text="\n" + ar.get('title') + "。" + ar.get('abstract')+ "。"
        text = text.replace(' ','')
        tags = ar.get('tags')    


        matcher = rule_match.IndicationMatcher()
        inds_t=matcher.match(text,allow_string_include=True)
        ids_t=find_id(inds_t)
        #标题摘要中的适应症


        if inds_t:
            #print("title:",ar.get('title'))
            #print("inds in title and abstract:",inds_t)

            #print("在标题摘要中")
            c1,seq1,TS1 = find_inds(ids_t,text)
            #print("在正文中")
            c2,seq2,TS = find_inds(ids_t,content)

            cm1 = np.mean(c1)
            cm2 = np.mean(c2)
            #print("TS",TS)

            #把代表各项分值的数组相加，得到每个inds_tags的分值
            score=[]
            TP1=[]
            TP2=[]
            TF1=[]
            TF2=[]
            c = len(c1)#关键词数量

            for i in range(c):
                TP1.append(math.log(math.log(3+seq1[i])))
                TP2.append(math.log(math.log(3+seq2[i])))
                TF1.append(c1[i]/(cm1+k))
                TF2.append(c2[i]/(cm2+k))
            #print("TP1",TP1)
            #print("TF1",TF1)
            #print("TP2",TP2)
            #print("TF2",TF2)
            
            for i in range(c):    
                score.append((w11*TP1[i]+w21*TP2[i])/(w12*TF1[i]+w22*TF2[i]+TS[i]))
            #print("score",score)


            a = copy.copy(score)
            b = []
            for i in range(len(a)):
                b.append(a.index(min(a)))
                a[a.index(min(a))] = math.inf
            #b是关键词分数从小到大排序的索引

            #阈值以上的词忽略
            keywords = []
            scores = []
            k_ids = []
            print('keywords ranking')
            for i in range(len(b)):
                if inds_t[b[i]] in keywords or ids_t[b[i]] in k_ids:
                    pass
                else:
                    if score[b[i]] < th:
                        keywords.append(inds_t[b[i]])
                        scores.append(score[b[i]])
                        k_ids.append(ids_t[b[i]])
                    else:
                        pass
            #print(keywords)
            #print(scores)
            #print(k_ids)
            
            result={'title':ar.get('title'),'abstract':ar.get('abstract'),'content':content,'candidates':inds_t,'score':score,'keywords':keywords}
            results.append(result)

#关键词筛选排序的结果路径
json_file_path = '/home/zj/PharmAI/pharm_ai/kwranking/ranres1.json'
json_file = open(json_file_path, mode='w',encoding='utf8')
json.dump(results, json_file, indent=4, ensure_ascii=False)

#if __name__ == "__main__":
