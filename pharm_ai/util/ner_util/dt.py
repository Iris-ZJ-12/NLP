import pandas as pd
from pharm_ai.util.utils import Utilfuncs as u
from pharm_ai.config import ConfigFilePaths as cfp
from pharm_ai.util.ner_util.rule_ner import RuleNER
import json


def dt0630():
    f = 'fin_org_dic.xlsx'
    df = pd.read_excel(f, 'dic').dropna(how='all')
    abbrev = df['abbrev']
    abbrev_en = df['abbrev_en']
    alias = df['alias']
    df = pd.concat([abbrev, abbrev_en, alias]).reset_index(drop=True).dropna().drop_duplicates().tolist()
    words = []
    for w in df:
        if '|' in w:
            words.extend(w.split('|'))
        else:
            words.append(w)
    df = pd.DataFrame({'words': words})
    u.to_excel(df, f, 'dic_clean')


def dt0701():
    h5 = 'fin_org_dic20200701.h5'
    df = pd.read_hdf(h5)
    words = df['words'].tolist()
    del words[34333]
    df = pd.DataFrame({'fin_org': words})
    h5 = 'fin_org_dic20200702.h5'
    df.to_hdf(h5, 'dic_clean')
    df = pd.read_hdf(h5)
    words = df['fin_org'].tolist()
    return {'fin_org': words}


def dt0702():
    h5 = cfp.project_dir + '/ner_util/fin_org_dic20200702.h5'
    df = pd.read_hdf(h5)
    words = df['fin_org'].tolist()
    return {'fin_org': words}


def dt0922():
    p = 'cn_stopwords.json'
    dt = json.load(open(p, 'r'))
    print(dt)
    # print(data)
    # p = 'cn_stopwords.json'
    # f = open(p, 'w', encoding='utf-8')
    # json.dump(data, f, indent=4, ensure_ascii=False)


def dt0923():
    c = pd.read_excel('fin_org_dic.xlsx', 'dic_clean').dropna().drop_duplicates()['words'].values
    c = {'c': c}
    rn = RuleNER(c, 'fin_org-20200923.pkl')
    s = """泰霖基金管理(深圳)有限公司是啥"""
    print(rn.label_sentence(s))


if __name__ == '__main__':
    dt0923()
