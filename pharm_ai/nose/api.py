from argparse import ArgumentParser
from tempfile import NamedTemporaryFile
from typing import List

import os
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from pharm_ai.nose.predictor import T5Predictor

parser = ArgumentParser()
parser.add_argument('-c', '--cuda_device', type=int, default=0, help='Cuda device. DEFAULT: 0.')
parser.add_argument('-p', '--port', type=int, default=21444, help='Port number. DEFAULT: 21444.')

args = parser.parse_args()

description_doc = """
本文档提供**适应症标准化**接口的说明。

## 接口说明

**请求数据**的格式见相应部分的**Parameters->Request body->Schema**部分，
参考样例详见相应部分的**Parameters->Request body->Example Value**部分。

测试接口请点击相应部分的**Try it out**。

**响应数据**的格式见相应部分的**Responses->Successful Response (code 200)->Schema**部分，
参考样例详见相应部分的**Responses->Successful Response (code 200)->Example Value**部分。

## 更新日志
- **v1.1**: 优化适应症识别准确率。

"""
version = 'v1.1'

app = FastAPI(title="适应症标准化", version=version, description=description_doc,
              openapi_tags=[
                  {'name': 'main',
                   'description': "主要接口"},
                  {'name': 'test',
                   'description': "测试使用"}
              ])

predictor = T5Predictor(version, args.cuda_device)
# predictor = ClsPredictor(version, args.cuda_device)
predictor.close_multiprocessing()

example_disease_description = [
    '帕博利珠单抗和仑伐替尼联合培美曲塞和铂类化疗适用于表皮生长因子受体（EGFR）基因突变阴性和'
    '间变性淋巴瘤激酶（ALK）阴性的转移性非鳞状非小细胞肺癌（NSCLC）的一线治疗。',
    'COVID-19 Respiratory Infection',
    '晚期结直肠癌',
    '胃食管反流性疾病（GERD），与适当的抗菌疗法联合用药根除幽门螺杆菌，并且需要持续NSAID治疗的患者。',
    'Bilateral Breast Carcinoma'
]


class Input(BaseModel):
    description: List[str] = Field(..., example=example_disease_description)


class Result(BaseModel):
    esid: List[List[str]] = Field(
        ..., example=[['181', '5719'], ['10', '6060'], ['938'], ['485']]
    )
    indication: List[List[str]] = Field(
        ..., example=[["淋巴瘤", "非鳞状非小细胞肺癌"], ["呼吸道感染", "新型冠状病毒感染"],
                      ["结直肠癌"], ["胃食管反流病"], ["乳腺癌"]])

@app.post("/normalize/", tags=['main'], response_model=Result)
def normalize_disease(inp: Input):
    result = predictor.predict(inp.description)
    return result


@app.post("/normalize_test/", tags=["test"])
async def normalize_test(file: UploadFile = File(...)):
    """
    ## 测试用接口
    ### 输入参数
    上传包含适应症描述的excel文件(**.xlsx**)，
    #### 格式说明
    需包含**description**列。
    ### 返回参数
    返回标准化结果的excel文件。
    """
    content = await file.read()
    with NamedTemporaryFile('wb+', suffix='.xlsx', dir='data') as temp_f:
        temp_f.write(content)
        df = pd.read_excel(temp_f.name)
    desc = df['description'].tolist()
    indication = predictor.predict(desc, return_dict=False, workers=min(10, len(desc) // 10 + 1))
    file_basename = file.filename.rsplit('.', 1)[0]
    res_f = f'{file_basename}_result.xlsx'
    indication.rename(columns={'to_predicts': 'description'})[
        ['description', 'indication', 'esid']].to_excel(res_f, index=False)
    return FileResponse(res_f, filename=res_f, background=BackgroundTask(os.remove, res_f))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=args.port)
