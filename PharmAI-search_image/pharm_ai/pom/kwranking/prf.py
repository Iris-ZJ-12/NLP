import json
import numpy as np

#算法筛选的
path1 = '/home/zj/PharmAI/pharm_ai/kwranking/ranres1.json'
with open(path1, 'r+',encoding='UTF-8-sig') as f:
    print("Load str file from {}".format(path1))
    strd = f.read()
    kwars = json.loads(strd)

#人工筛选的
path2 = '/home/zj/PharmAI/pharm_ai/kwranking/randel.json'
with open(path2, 'r+',encoding='UTF-8-sig') as f:
    print("Load str file from {}".format(path2))
    strd = f.read()
    moars = json.loads(strd)

#keywords和scores是按分数从低到高对应排列的
#存每篇文章的值
TP=[]
FP=[]
FN=[]
P=[]
R=[]
F=[]
j=0
#人工只看了前一百篇
while j<100:
    tp=0
    fp=0
    fn=0
    p=0
    r=0
    f=0
    #放当前文章的值
    
    if 'modified tags' not in moars[j]:
        pass
    else:
        keywords = kwars[j].get('keywords')
        modified = moars[j].get('modified tags')
        for item in keywords:
            if item in modified:
                tp+=1
            else:
                fp+=1
        for ind in modified:
            if ind in keywords:
                pass
            else:
                fn+=1
        print(kwars[j].get('title'))
        print('kw',keywords)
        print('mo',modified)

        print('TP:{:d},FP:{:d},FN:{:d}'.format(tp,fp,fn))
        if tp==0 and fp==0:
            p=1
        else:
            p=tp/(tp+fp)
        if tp==0 and fn==0:
            r=1
        else:
            r=tp/(tp+fn)
        if p==0 and r==0:
            f=1
        else:
            f=2*p*r/(p+r)
        print('P:{:.2f},R:{:.2f},F:{:.2f}'.format(p,r,f))
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
        P.append(p)
        R.append(r)
        F.append(f)
    j+=1

print('\n') 
Pa=np.mean(P)
Ra=np.mean(R)
Fa=np.mean(F)
print("Pa=",Pa)
print("Ra=",Ra)
print("Fa=",Fa)
print('\n') 

Pi = sum(TP)/(sum(TP)+sum(FP))   
Ri = sum(TP)/(sum(TP)+sum(FN))  
Fi = 2 * Pi * Ri / (Pi + Ri)
print("Pi=",Pi)
print("Ri=",Ri)
print("Fi=",Fi)
