# encoding: utf-8
'''
@author: zyl
@file: mrc_api.py
@time: 2021/9/24 16:30
@desc:
'''
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

app = FastAPI(
    title="🏃‍♂️机器阅读理解MRC(Machine Reading Comprehension)",
    version="v0.test",
    description="""
    问答

## 接口说明
    φ(*￣0￣)

## 更新日志：
- v0(09/16):
    - 问答系统-公共模型测试版

    """,
    openapi_tags=[
        {
            'name': 'mrc_1',
            'description': '中文常规问答，https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large',
        },
        {
            'name': 'mrc_2',
            'description': '英文生物语料库问答，SciFive: a text-to-text transformer model for biomedical literature,',
        }
    ]
)


class MRC_1:
    def __init__(self):
        self.nlp = self.get_pipeline()

    def get_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "/large_files/5T/zyl_tmp_model/chinese_pretrain_mrc_roberta_wwm_ext_large/")
        model = AutoModelForQuestionAnswering.from_pretrained(
            "/large_files/5T/zyl_tmp_model/chinese_pretrain_mrc_roberta_wwm_ext_large/")
        nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
        return nlp

    def get_answer(self, question, text_):
        return self.nlp(question=question, context=text_).get('answer')


mrc_1 = MRC_1().get_answer


# MRC1###################################
class MRCInput_1(BaseModel):
    text: str = Field(default='我叫小明，我住在苏州,我喜欢吃枣子。', example="我叫小明，我住在苏州,我喜欢吃枣子。")
    questions: list = Field(default=['我住在哪?', '我喜欢吃什么水果?'], example=['我住在哪?', '我喜欢吃什么水果?'])


class MRCOutput_1(BaseModel):
    questions_answers: list


@app.post("/mrc_1/", tags=['mrc_1'], response_model=MRCOutput_1)
async def predict1(mrc_input: MRCInput_1):
    text = mrc_input.text
    questions = mrc_input.questions
    questions_answers = []
    for q in questions:
        questions_answers.append({'questions': q,
                                  'answers': mrc_1(q, text)})
    return {'questions_answers': questions_answers}


class MRC_2:
    def __init__(self, cuda_device=2):
        self.cuda_device = cuda_device
        self.model, self.tokenizer = self.get_model_and_tokenizer()

    def get_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC")
        model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC")
        model = model.to(f"cuda:{self.cuda_device}")
        return model, tokenizer

    def get_answer(self, question, text_, encoding_max_length=512, output_max_length=32):
        input_text = f'question: {question} context: {text_}'

        encoding = self.tokenizer.encode_plus(input_text, padding='max_length', return_tensors="pt",
                                              max_length=encoding_max_length)
        input_ids = encoding["input_ids"].to(f"cuda:{self.cuda_device}")
        attention_masks = encoding["attention_mask"].to(f"cuda:{self.cuda_device}")

        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=output_max_length,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


mrc_2 = MRC_2().get_answer

default_mrc2_input_text = "Drug treatments in Alzheimer's disease. Despite the significant public health issue " \
                          "that it poses, only five medical treatments have been approved for Alzheimer's " \
                          "disease (AD) and these act to control symptoms rather than alter the course of the disease." \
                          " Studies of potential disease-modifying therapy have generally been undertaken in patients " \
                          "with clinically detectable disease, yet evidence suggests that the pathological changes " \
                          "associated with AD begin several years before this. It is possible that pharmacological " \
                          "therapy may be beneficial in this pre-clinical stage before the neurodegenerative process " \
                          "is established. Techniques providing earlier diagnosis, such as cerebrospinal fluid " \
                          "biomarkers and amyloid positron emission tomography neuroimaging, are key to testing " \
                          "this theory in clinical trials. Recent results from trials of agents such as aducanumab " \
                          "are encouraging but must also be interpreted with caution. Such medicines could potentially" \
                          " delay the onset of dementia and would therefore markedly reduce its prevalence. However," \
                          " we currently remain a good distance away from clinically available disease-modifying therapy."


class MRCInput_2(BaseModel):
    text: str = Field(default=default_mrc2_input_text, example=default_mrc2_input_text)
    questions: list = Field(default=['What disease is the drug aducanumab targeting?'],
                            example=['What disease is the drug aducanumab targeting?'])


class MRCOutput_2(BaseModel):
    questions_answers: list


@app.post("/mrc_2/", tags=['mrc_2'], response_model=MRCOutput_2)
async def predict2(mrc_input: MRCInput_2):
    text = mrc_input.text
    questions = mrc_input.questions
    questions_answers = []
    for q in questions:
        questions_answers.append({'questions': q,
                                  'answers': mrc_2(q, text)})
    return {'questions_answers': questions_answers}


#
# # nlp2 = get_mrc_2()
# @app.post("/predict2/", tags=['predict2'], response_model=OverAllOutput2)
# async def predict2(overall_input: OverAllInput2):
#     questions = overall_input.text
#     questions_answers = []
#     for q in questions:
#         # sentence = "Identification of APC2 , a homologue of the adenomatous polyposis coli tumour suppressor ."
#         # text = "ncbi_ner: " + sentence + " </s>"
#
#         encoding = t2.encode_plus(q, pad_to_max_length=True, return_tensors="pt")
#         input_ids, attention_masks = encoding["input_ids"].to("cuda:1"), encoding["attention_mask"].to("cuda:1")
#
#         outputs = m2.generate(
#             input_ids=input_ids, attention_mask=attention_masks,
#             max_length=256,
#             early_stopping=True
#         )
#         # print('o:',outputs)
#         # for output in outputs:
#         #     line = t2.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         #     print(line)
#
#         questions_answers.append({'questions': q,
#                                   'answers': outputs})
#     print(questions_answers)
#     return {'questions_answers': questions_answers}
#
#
# tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC")
# model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC")
# model = model.to("cuda:2")
#
#
# def get_answer(text, m1=512, m2=32):
#     # sentence = "Identification of APC2 , a homologue of the adenomatous polyposis coli tumour suppressor ."
#     # text = "ncbi_ner: " + sentence + " </s>"
#
#     encoding = tokenizer.encode_plus(text, padding='max_length', return_tensors="pt", max_length=m1)
#     input_ids, attention_masks = encoding["input_ids"].to("cuda:2"), encoding["attention_mask"].to("cuda:2")
#
#     outputs = model.generate(
#         input_ids=input_ids, attention_mask=attention_masks,
#         max_length=m2,
#         early_stopping=True
#     )
#
#     for output in outputs:
#         line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         print('#' * 10)
#         print(line)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5215)
    # text = "question*: What disease is the drug aducanumab targeting? context*: Drug treatments in Alzheimer's disease. Despite the significant public health issue that it poses, only five medical treatments have been approved for Alzheimer's disease (AD) and these act to control symptoms rather than alter the course of the disease. Studies of potential disease-modifying therapy have generally been undertaken in patients with clinically detectable disease, yet evidence suggests that the pathological changes associated with AD begin several years before this. It is possible that pharmacological therapy may be beneficial in this pre-clinical stage before the neurodegenerative process is established. Techniques providing earlier diagnosis, such as cerebrospinal fluid biomarkers and amyloid positron emission tomography neuroimaging, are key to testing this theory in clinical trials. Recent results from trials of agents such as aducanumab are encouraging but must also be interpreted with caution. Such medicines could potentially delay the onset of dementia and would therefore markedly reduce its prevalence. However, we currently remain a good distance away from clinically available disease-modifying therapy."
    #
    # get_answer(text)
    # print('1')
    # from simpletransformers.t5 import T5Model
    # m = T5Model(model_type='t5', model_name="razent/SciFive-base-Pubmed_PMC",tokenizer="razent/SciFive-base-Pubmed_PMC", use_cuda=True,
    #                        cuda_device=1)
    # m.args.use_multiprocessed_decoding = False
    # questions  =["what is the mode of inheritance of Romano Ward long QT syndrome?" ,
    # "Who are the 4 members of The Beatles?" , "How many teeth do humans have?",]
    #
    # text = "question*: Where are Paneth cells located? context*: Paneth's disease. In about 70% of patients Crohn's disease (CD) affects the small intestine. This disease location is stable over time and associated with a genetic background different from isolated colonic disease. A characteristic feature of small intestinal host defense is the presence of Paneth cells at the bottom of the crypts of Lieberk眉hn. These cells produce different broad spectrum antimicrobial peptides (AMPs) most abundantly the 伪-defensins HD-5 and -6 (DEFA5 und DEFA6). In small intestinal Crohn's disease both these PC products are specifically reduced. As a functional consequence, ileal extracts from Crohn's disease patients are compromised in clearing bacteria and enteroadherent E. coli colonize the mucosa. Mechanisms for defective antimicrobial Paneth cell function are complex and include an association with a NOD2 loss of function mutation, a disturbance of the Wnt pathway transcription factor TCF7L2 (also known as TCF4), the autophagy factor ATG16L1, the endosomal stress protein XBP1, the toll-like receptor TLR9, the calcium mediated potassium channel KCNN4 as well as mutations or inactivation of HD5. Thus we conclude that small intestinal Crohn's disease is most likely a complex disease of the Paneth cell: Paneth's disease."
    # question = [""]

    # bioasq5b: context: Phenotypic spectrum of patients with PLA2G6 mutation and PARK14-linked parkinsonism. BACKGROUND: PLA2G6 is the causative gene for infantile neuroaxonal dystrophy, neurodegeneration associated with brain iron accumulation, and Karak syndrome. Based on previous reports, patients with PLA2G6 mutations could show axonal dystrophy, dystonia, dementia, and cerebellar signs. Recently, PLA2G6 was also reported as the causative gene for early-onset PARK14-linked dystonia-parkinsonism. METHODS: To clarify the role of PLA2G6 mutation in parkinsonism, we conducted mutation analysis in 29 selected patients with very early-onset ( < 30, mean 21.2 卤 8.4 years, 卤 SD) parkinsonism. These patients had other clinical features (e.g., mental retardation/dementia [14/29], psychosis [15/29], dystonia [11/29], and hyperreflexia [11/29]). RESULTS: Two novel compound heterozygous PLA2G6 mutations were detected (patient A: p.F72L/p.R635Q; patients B1 and B2: p.Q452X/p.R635Q). All 3 patients had early-onset l-dopa-responsive parkinsonism with dementia and frontotemporal lobar atrophy. Disease progression was relatively rapid. SPECT in patient B1 showed frontotemporal lobar hypoperfusion. MRI in patient A showed iron accumulation in the substantia nigra and striatum. CONCLUSIONS: Although the clinical presentation of PLA2G6-associated neurodegeneration was reported to be homogeneous, our findings suggest patients with PLA2G6 mutation could show heterogeneous phenotype such as dystonia-parkinsonism, dementia, frontotemporal atrophy/hypoperfusion, with or without brain iron accumulation. Based on the clinical heterogeneity, the functional roles of PLA2G6 and the roles of PLA2G6 variants including single heterozygous mutations should be further elucidated in patients with atypical parkinsonism, dementia, or Parkinson disease. PLA2G6 mutations should be considered in patients with early-onset l-dopa-responsive parkinsonism and dementia with frontotemporal lobar atrophy. question: Which gene is mutated in the Karak syndrome?
    # PLA2G6

    # print('1')
    # tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-base-Pubmed_PMC")
    # model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-base-Pubmed_PMC")
    # model = model.to("cuda:1")
    # sentence = "Identification of APC2 , a homologue of the adenomatous polyposis coli tumour suppressor ."
    # text = "ncbi_ner: " + sentence + " </s>"
    #
    # encoding = tokenizer.encode_plus(text, padding=True, return_tensors="pt")
    # input_ids, attention_masks = encoding["input_ids"].to("cuda:1"), encoding["attention_mask"].to("cuda:1")
    #
    # outputs = model.generate(
    #     input_ids=input_ids, attention_mask=attention_masks,
    #     max_length=256,
    #     early_stopping=True
    # )
    #
    # for output in outputs:
    #     line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     print(line)
