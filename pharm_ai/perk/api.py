from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from pharm_ai.util.api_util import Logger
from pharm_ai.perk.dt import NerPreprocessor
from pharm_ai.perk.predictor import Classifier, Ner
from typing import List, Tuple, Dict, Optional
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument('-c','--cuda_device', type=int, default=-1)
arg_parser.add_argument('-g','--generative', action='store_true')
arg_parser.add_argument('-p','--port',type=int, default=5021)
arg_parser.add_argument('-u','--use_cpu', action='store_ture')

args = arg_parser.parse_args()

description_text = """
本文档提供药物经济学分类与实体识别功能接口的说明。

## 更新日志
- v2
    - 提高分类与实体识别的准确性
    - 更新实体识别输出格式
    - 新增国家名标准化
"""

app = FastAPI(title="药物经济学文献中指定字段的分类与实体识别", description=description_text, version="v2.0",
              openapi_tags=[
                  {'name':'Classification','description':'使用分类模型进行预测的接口'},
                  {'name': 'Named Entity Recognition', 'description': '实体识别接口'}
              ])

class Paper(BaseModel):
    """
    输入数据格式：
    ```
    {
        "text":[paragraph1, paragraph2, ...]
    }
    ```
    其中`paragraph1`，`paragraph2`等为每个输入文本。
    """
    texts:List[str] = Field(
        ..., example=[
            "The Lescol Intervention Prevention Study (LIPS) showed substantial gains in health outcomes from statins "
            "following PCI. That study was a randomized double-blind placebo-controlled trial undertaken in 77 "
            "centres, predominantly in Europe, of patients with moderate hypercholesterolemia who had undergone their "
            "first PCI. The evidence on cost-effectiveness has been established for the UK, USA and the Netherlands, "
            "but due to different health system cost structures, the results may not be applicable to other European "
            "countries. The aim of this study was to estimate the cost-effectiveness of fluvastatin used following "
            "first PCI in Hungary. Materials and methods: A deterministic Markov model was used to estimate the "
            "incremental costs per quality-adjusted life year gained, with cost data drawn from the Hungarian "
            "National Health Insurance Fund. Effectiveness data on fluvastatin was derived directly from LIPS and "
            "utility weights from previous studies on heart disease. Sensitivity analyses were conducted around key "
            "parameters and analyses were conducted for subgroups identified in LIPS. Results: Treatment with "
            "fluvastatin cost an additional 1,704 euro and resulted in an additional 0.107 QALYs per patient "
            "discounted over 10-years compared with controls. The incremental cost per quality-adjusted life year "
            "gained was 15,910 euro. The key determinants of cost-effectiveness were the effectiveness of "
            "fluvastatin, utility weights, cost of fluvastatin, and the time horizon evaluated. Fluvastatin was "
            "substantially more cost-effective in patients with diabetes, renal disease, multi-vessel disease or "
            "LDL-cholesterol >3.4 mmol/l. Conclusions: Fluvastatin is an economically efficient pharmaceutical for "
            "reducing heart disease in Hungary and other European countries in patients following PCI.",
            "A significant number of preventable cardiac deaths in infancy and childhood are due to long QT syndrome "
            "(LQTS) and to unrecognized neonatal congenital heart diseases (CHDs). Both carry a serious risk for "
            "avoidable mortality and morbidity but effective treatments exist to prevent lethal arrhythmias or to "
            "allow early surgical correction before death or irreversible cardiac damage. As an electrocardiogram ("
            "ECG) allows recognition of LQTS and of some of the CHDs that have escaped medical diagnosis, and as LQTS "
            "also contributes to sudden infant death syndrome, we have analysed the cost-effectiveness of a "
            "nationwide programme of neonatal ECG screening. Our primary analysis focused on LQTS alone; a secondary "
            "analysis focused on the possibility of identifying some CHDs also. Methods and results: A decision "
            "analysis approach was used, building a decision tree for the strategies 'screening'-'no screening'. "
            "Markov processes were used to simulate the natural or clinical histories of the patients. To assess the "
            "impact of potential errors in the estimates of the model parameters, a Monte Carlo sensitivity analysis "
            "was performed by varying all baseline values by +/-30%. Incremental cost-effectiveness analysis for the "
            "primary analysis shows that with the screening programme, the cost per year of life saved is very low: "
            "11,740 euro. The cost for saving one entire life of 70 years would be 820,000 euro. Even by varying "
            "model parameters by +/-30%, the cost per year of life saved remains between 7400 euro and 20,"
            "400 euro. These figures define 'highly cost-effective' screening programmes. The secondary analysis "
            "provides even more cost-effective results. Conclusion: A programme of neonatal ECG screening performed "
            "in a large European country is cost-effective. An ECG performed in the first month of life will allow "
            "the early identification of still asymptomatic infants with LQTS and also of infants with some "
            "correctable CHDs not recognized by routine neonatal examinations. Appropriate therapy will prevent "
            "unnecessary deaths in infants, children, and young adults."
        ]
    )

class ClassifyResult(BaseModel):
    """
    返回格式：
    ```
    {
        "labels":
            [
                [type_result, result_result],
                ...
            ]
    }
    ```
    包含每个文本文本分类的结果，每个结果包含`type_result`和`result_result`，
    分别为**评价类型**和**结果**两个方面的分类标签。

    **评价类型**可能值包括`CEA/CUA`，`CBA`，`CMA`，`CCA`，`BIA`和`Other`，**结果**可能值包括`cost-effective`和`not stated`。
    """
    labels: List[Tuple[str, str]] = Field(..., example=[["CEA/CUA", "cost-effective"], ["CEA/CUA", "not stated"]])

classifier=[]
no_multiprocessing_args={
    "use_multiprocessing": False,
    "use_multiprocessing_for_evaluation": False
}
no_multiprocessing_generative_args ={
    "use_multiprocessing": False,
    "use_multiprocessing_for_evaluation": False,
    "use_multiprocessed_decoding": False
}
for task in [0,1]:
    cur_classifier = Classifier('v2.0', task, args.cuda_device, args.n_gpu, args.generative)
    if args.generative:
        cur_classifier.model.args.update_from_dict(no_multiprocessing_generative_args)
    else:
        cur_classifier.model.args.update_from_dict(no_multiprocessing_args)
    classifier.append(cur_classifier)
# rule_prep = NerPreprocessor()
ner_model = Ner('v2.1', sub_task=3, cuda_device=args.cuda_device)
ner_model.model.args.update_from_dict(no_multiprocessing_generative_args)

@app.post("/classify/", tags=['Classification'], response_model=ClassifyResult)
@Logger.log_input_output()
async def classify_text(paper: Paper):
    """
    本接口提供“评价类型”与“结果”的文本分类。

    - 返回类型：
        - **评价类型**：`CEA/CUA`，`CBA`，`CMA`，`CCA`，`BIA`或`Other`
        - **结果**：`cost-effective`或`not stated`
    """
    to_preds = paper.texts
    # res_list = classifier.predict(to_preds) # For generative
    res_list1 = classifier[0].predict(to_preds)
    res_list2 = classifier[1].predict(to_preds)
    res_list = list(zip(res_list1, res_list2))
    res = {'labels': res_list}
    return res

class TypeResult(BaseModel):
    """
    返回格式：
    ```
    {
        "labels":
            [
                type_result,
                ...
            ]
    }
    ```
    其中`type_result`为评价类型标签。

    包含每个文本分类的**评价类型**标签，可能值包括`CEA/CUA`，`CBA`，`CMA`，`CCA`，`BIA`和`Other`。
    """
    labels: List[str] = Field(..., example=["CEA/CUA", "Other"])

@app.post("/classify/type/", tags=['Classification'], response_model=TypeResult)
@Logger.log_input_output()
async def classify_type(paper: Paper):
    """
    本接口提供“评价类型”的文本分类功能。

    - 返回类型： `CEA/CUA`，`CBA`，`CMA`，`CCA`，`BIA`和`Other`
    """
    res_type = classifier[0].predict(paper.texts)
    res = {'labels': res_type}
    return res

class ResultResult(BaseModel):
    """
    返回格式：
    ```
    {
        "labels":
            [
                result_result,
                ...
            ]
    }
    ```
    其中`result_result`为结果标签。

    包含每个文本分类的**结果**标签，可能值包括`cost-effective`和`not stated`。
    """
    labels: List[str] = Field(..., example=["cost-effective", "not stated"])

@app.post("/classify/result/", tags=['Classification'], response_model=ResultResult)
@Logger.log_input_output()
async def classify_result(paper: Paper):
    """
    本接口提供“结果”的文本分类功能。

    - 返回类型： `cost-effective`或`not stated`
    """
    res_result = classifier[1].predict(paper.texts)
    res = {'labels': res_result}
    return res

class CountryAlias(BaseModel):
    name: str
    name_pinyin: str
    code: str

class CountryResult(BaseModel):
    standard_contry: str
    country_alias: CountryAlias

class NerResult(BaseModel):
    """
**药物、疾病、模型、国家**实体识别的结果以及**国家名标准化**的结果。
    """
    result: Dict[str, List[List[str]]] = Field(
        ...,
        example={
            "drug": [[""], [""]],
            "disease": [["PCI"], ["cardiac deaths", "LQTS", "cardiac death"]],
            "model": [[""], [""]],
            "nation": [["Germany"], [""]]
        })
    country: List[List[Optional[CountryResult]]] = Field(
        ...,
        example=[
            [
                {
                    "standard_contry": "Germany",
                    "country_alias": {
                        "name": "德国",
                        "name_pinyin": "deguo",
                        "code": "DEU"
                    }
                }
            ],
            [
                None
            ]
        ],
    )

@app.post("/ner/", tags=['Named Entity Recognition'], response_model=NerResult)
@Logger.log_input_output()
async def ner_predict(paper: Paper):
    """
    本接口自动识别药物经济学文献中的**疾病**、**药物**、**国家**、**模型**。
    """
    to_preds = paper.texts
    # res = {"result": rule_prep.ner_predict(to_preds)}
    size = len(to_preds)
    pre=['drug', 'disease', 'model', 'nation']
    prefixes = [pre[0]]*size+[pre[1]]*size+[pre[2]]*size+[pre[3]]*size
    to_predicts = to_preds*4
    res_raw = ner_model.predict(prefixes, to_predicts)
    res_ner = {prefix: res_raw[size * i:size * (i + 1)] for i, prefix in enumerate(pre)}
    res_country = [[ner_model.location_dic.get(ent_nation_res) for ent_nation_res in para_nation_res]
                   for para_nation_res in res_ner['nation']]
    return {
        "result": res_ner, "country": res_country
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=args.port)