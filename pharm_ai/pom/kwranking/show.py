import streamlit as st
import time
from PIL import Image
import json
import numpy
import copy
import re

st.set_page_config(
    page_title="Keyword Ranking",
    page_icon="🎈",
)

def tags():
    with open('/home/zj/PharmAI/pharm_ai/kwranking/ranres.json', 'r+',encoding='utf-8') as f:
        str1 = f.read()
        ranres = json.loads(str1)   
        f.close()

    num=st.sidebar.number_input('Index of article',step=1,min_value=1, max_value=len(ranres)+1,)

    st.write('Title:')
    st.write(ranres[num-1].get('title'))
    st.write('Abstract:')
    st.write(ranres[num-1].get('abstract'))
    with st.expander("Content: "):
        st.write(ranres[num-1].get('content'))
    st.write('Candidates: ',ranres[num-1].get('candidates'))
    st.write('Keywords: ',ranres[num-1].get('keywords'))
    st.write('Scores: ',ranres[num-1].get('scores'))
    
    container = st.sidebar.container()
    a=st.sidebar.radio('About the keyword results:', ('Accept','Modify'),index=0)
    
    if a == 'Accept':
        fd={'modified tags':ranres[num-1].get('keywords')}
    if a == 'Modify':
        op=copy.copy(ranres[num-1].get('candidates'))
        op.append("other")
        op1=ranres[num-1].get('keywords')
        
        options = st.sidebar.multiselect('Indications to stay',op,op1)
        if "other" in options:
            oi = st.sidebar.text_input('other indication')
            oi = re.split("，",oi)
            oi = re.split(",",oi)
            options.append(oi)
            options.remove("other")

        fd={'modified tags':options}
    #st.sidebar.write(fd)
    ranres[num-1].update(fd)
    #st.write(ranres[num-1])

    if st.sidebar.button('Submit'):
        with open('/home/zj/PharmAI/pharm_ai/kwranking/randel.json', 'r',encoding='utf-8') as f:
            model=json.load(f)
            model[num-1]=ranres[num-1]
            f.close()
        with open('/home/zj/PharmAI/pharm_ai/kwranking/randel.json', 'w',encoding='utf-8') as f:
            json.dump(model, f, indent=4, ensure_ascii=False)
            st.sidebar.write('Changes saved')
            #st.balloons()

def test():
    op=[
            "血友病",
            "发热"
        ]
    op1=[
        
        ]
    options = st.sidebar.multiselect('Tags',op,op1)
    de={'modified tags':options}
    st.sidebar.write(de)

    json_file_path = '/home/zj/PharmAI/pharm_ai/kwranking/randel.json'
    json_file = open(json_file_path, mode='w',encoding='utf8')
    json.dump(de, json_file, indent=4, ensure_ascii=False)
    st.sidebar.write('Changes saved')
    st.balloons()
    st.sidebar.write('Finish all modification?')


tags()