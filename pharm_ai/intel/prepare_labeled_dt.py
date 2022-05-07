# encoding: utf-8
'''
@author: zyl
@file: prepare_labeled_dt.py
@time: 2021/9/24 10:51
@desc:
'''

# # Download spaCy models:
# models = {
#     'en_core_web_sm': spacy.load("en_core_web_sm"),
#     'en_core_web_lg': spacy.load("en_core_web_lg")
# }
#
#
# # This function converts spaCy docs to the list of named entity spans in Label Studio compatible JSON format:
# def doc_to_spans(doc):
#     tokens = [(tok.text, tok.idx, tok.ent_type_) for tok in doc]
#     results = []
#     entities = set()
#     for entity, group in groupby(tokens, key=lambda t: t[-1]):
#         if not entity:
#             continue
#         group = list(group)
#         _, start, _ = group[0]
#         word, last, _ = group[-1]
#         text = ' '.join(item[0] for item in group)
#         end = last + len(word)
#         results.append({
#             'from_name': 'label',
#             'to_name': 'text',
#             'type': 'labels',
#             'value': {
#                 'start': start,
#                 'end': end,
#                 'text': text,
#                 'labels': [entity]
#             }
#         })
#         entities.add(entity)
#
#     return results, entities
#
#
# # Now load the dataset and include only lines containing "Easter ":
# df = pd.read_csv('lines_clean.csv')
# df = df[df['line_text'].str.contains("Easter ", na=False)]
# print(df.head())
# texts = df['line_text']
#
# # Prepare Label Studio tasks in import JSON format with the model predictions:
# entities = set()
# tasks = []
# for text in texts:
#     predictions = []
#     for model_name, nlp in models.items():
#         doc = nlp(text)
#         spans, ents = doc_to_spans(doc)
#         entities |= ents
#         predictions.append({'model_version': model_name, 'result': spans})
#     tasks.append({
#         'data': {'text': text},
#         'predictions': predictions
#     })
#
# # Save Label Studio tasks.json
# print(f'Save {len(tasks)} tasks to "tasks.json"')
# with open('tasks.json', mode='w') as f:
#     json.dump(tasks, f, indent=2)
#
# # Save class labels as a txt file
# print('Named entities are saved to "named_entities.txt"')
# with open('named_entities.txt', mode='w') as f:
#     f.write('\n'.join(sorted(entities)))


# import json
# import jsonlines
# df = df[0:20]
# df = df[['id','text_type','input_text','predicted']]  # type:pd.DataFrame
# df.rename(columns={'input_text':'ner'},inplace=True)

# a = df.to_dict(orient='records')
# a = [{'data':i} for i in a]
# print(a)
# # for i in a:
# #     j = json.dumps(i)
#
# with open("mydict.json",'w') as t:
#     json.dump(a, t)

import re


class Labeler:

    def __init__(self):
        pass

    def run(self):
        pass

    def test(self):
        pass

    @staticmethod
    def find_location(text, all_text):
        # print(text)
        all_find = re.finditer(text, all_text)
        res = []
        while True:
            try:
                elem = next(all_find)
                res.append({'text': text, 'start': elem.start(), 'end': elem.end()})
            except StopIteration:
                break
        return res

    # def


if __name__ == '__main__':
    txt = './data/test.xlsx'
    # print(Labeler.find_location('z', txt))

    import json
    import pandas as pd

    df = pd.read_excel(txt)[0:20]

    df = df[['raw_text', 'pdf_name', 'company_name_rule', 'project_name_rule', ]]
    df['company_name_rule'] = df['company_name_rule'].apply(lambda x: '|' if not eval(x) else '|'.join(eval(x)))
    df['project_name_rule'] = df['project_name_rule'].apply(lambda x: '|' if not eval(x) else '|'.join(eval(x)))
    df.rename(columns={'raw_text': 'ner'}, inplace=True)
    a = df.to_dict(orient='records')
    res2 = []
    predictions = []
    for i in a:

        predictions_res = []
        text = i.get('ner')
        if not text:
            text = ''

        s1 = i.get('company_name_rule')
        if s1=='|' or s1=='':
            continue
        else:
            s1 = s1.split('|')
            while '' in s1:
                s1.remove(s1)
            for s in s1:
                res = Labeler.find_location(s,text)

                for r in res:
                    predictions_res.append(
                        {'from_name': 'label',
                        'to_name': 'text',
                        'type': 'labels',
                        'value': {
                        'start': r.get('start'),
                        'end': r.get('end'),
                        'text': text,
                        'labels': ['company_name_rule']}}
                    )

        predictions.append({'model_version': 'v0', 'result': predictions_res})
    #     predictions = a['company_name_rule']
    res2.append({'data': a, 'predictions': predictions})
    print(res2)

        # print(a)
        # predictions = [{'model_version': model_name, 'result': spans}]
        # a = [{'data': i,'predictions':predictions} for i in a]

    with open("./data/mydict.json",'w') as t:
        json.dump(res2, t)

        # from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer
        # import torch
        # # Use in Transformers
        # tokenizer = AutoTokenizer.from_pretrained("/large_files/5T/zyl_tmp_model/chinese_pretrain_mrc_roberta_wwm_ext_large/")
        # model = AutoModelForQuestionAnswering.from_pretrained("/large_files/5T/zyl_tmp_model/chinese_pretrain_mrc_roberta_wwm_ext_large/")
        #
        # text = """
        # Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
        # architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
        # Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
        # TensorFlow 2.0 and PyTorch.
        # """
        #
        # questions = [
        #     "How many pretrained models are available in Transformers?",
        #     "What does Transformers provide?",
        #     "Transformers provides interoperability between which frameworks?",
        # ]

        # for question in questions:
        #     inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        #     input_ids = inputs["input_ids"].tolist()[0]
        #
        #     text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        #     answer_start_scores, answer_end_scores = model(**inputs)
        #
        #     answer_start = torch.argmax(
        #         answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        #     answer_end = torch.argmax(
        #         answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        #
        #     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        #
        #     print(f"Question: {question}")
        #     print(f"Answer: {answer}\n")

        #     from transformers import pipeline
        #
        #     nlp = pipeline("question-answering",model=model,tokenizer=tokenizer)
        #     t ="""
        #     包头市中心医院专项法律服务采购项目中标结果公示\n（招标编号：FZZB-BT2021-015\n、中标人信息：\n标段（包）[001]包头市中心医院专项法律服务采购项目：\n中标人：内蒙古瑞安律师事务所\n中标价格：29万元\n二、其他：\n包头市中心医院专项法律服务采购项目\n中标结果公示\n招标编号：FZZB-BT2021-015\n项目名称：包头市中心医院专项法律服务采购项目\n一、本次采购范围具体内容如下：\n包头市中心医院专项法律服务采购项目\n二、结果公示内容：\n中标单位：内蒙古瑞安律师事务所\n中标价：贰拾玖万元整（￥290000.00）\n服务期限：按采购人要求\n服务地点：采购人指定地点\n质量标准：应符合国家及行业标准\n此项目公示时间为：1个工作日\n采购人：包头市中心医院\n采购代理机构：法正项目管理集团有限公司\n地址：包头市昆都仑区昆河东路瑞星佳园（包头蒙电有限公司院内3楼）\n联系人：杜工\n电话：0472-6985309\n邮箱：fazhengzbgs@163.com\n扫描全能主创建\n三、监督部门\n本招标项目的监督部门为/\n四、联系方式\n招标人：包头市中心医院\n地\n址：包头市东河区\n联系\n系人：1\n电\n话：/\n电子邮件：1\n正项目\n招标代理机构：法正项目管理集团有限公司\n地址：包头市昆都仑区昆河东路瑞星佳园（包头蒙电有限公司院内3楼）\n联系人：杜工\n电话：0472-6985309\n电子邮件：fazhengzbgs@163.com\n文（签名）\n招标人或其招标代理机构主要负责人（项目负责人\n理集\n1\n微盖章）\n招标人或其招标代理机构：\n成\n扫描全能王创建
        # """
        #     from pharm_ai.intel.try_extractor import IntelExtractor
        #
        #     pdf = "/home/zyl/disk/PharmAI/pharm_ai/intel/data/2021-09-22招标pdf/7c84fb43e996be63f797e096e4289410.pdf"
        #     print(IntelExtractor().extract_fields_text_from_pdf(pdf))
        #     print(nlp(question="中标人是谁?", context=t)['answer'])
        #     print(nlp(question="项目名称是什么?", context=t).get('answer'))
        #     print(nlp(question="招标人或采购人是谁?", context=t))
        #     print(nlp(question="中标价格是多少?", context=t))
        #     print(nlp(question="产品名称是什么?", context=t))
