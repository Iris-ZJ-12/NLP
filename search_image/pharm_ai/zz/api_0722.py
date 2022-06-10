# encoding: utf-8
'''
@author: zyl
@file: api_0722.py
@time: 2021/7/21 下午12:13
@desc:
'''
import numpy.ma as ma
import numpy as np
import pandas as pd
from fastapi import FastAPI

from pydantic import BaseModel, Field
import uvicorn
from typing import List, Tuple
# from pharm_ai.zz.predictor import TitleClassifier
# from pharm_ai.zz.predictor_bind import News_Title_Binder
from pharm_ai.util.api_util import Logger
import time
from pharm_ai.zz.m1.predict import ZZPredictorM1
from pharm_ai.zz.m2.predict import ZZPredictorM2

dsc_txt = """

- API功能：输入资讯的相关信息（省份、标题、日期），首先判断资讯是否为需要的资讯，结果label为0或1；若结果为1，进一步绑定项目。

- 输入格式--json format:
```
{
    "texts":[
        ["资讯1省份","资讯1标题","资讯1时间"],
        ["资讯2省份","资讯2标题","资讯2时间"],
        ...
    ]
}
```
举例:
```
{"texts": [
    ["贵州", "关于取消四川康福来药业集团有限公司等企业部分中标药品中标（挂网）资格的通知", "2020-06-11"], 
    ["云南", "关于药品生产企业名称变更的公示", "2020-09-21"], 
    ["广西", "关于公示2020年度非免疫规划疫苗集中采购目录的通知", "2020-04-27"], 
    ["浙江", "关于拟恢复部分药品配送企业网上药品配送资格的公示", "2020-11-26"], 
    ["甘肃", "关于公示2020年度甘肃省非免疫规划疫苗阳光挂网复议结果的通知", "2020-09-18"]
]
}

```
- 输出格式：
```
{
  "result": [
    {
      "if_needed": 1,
      "project_id": "4791d93bd6934759a71fc7ea965ab37a"
    },
    {
      "if_needed": 1,
      "project_id": "b4e13cf11e33435b8b196b7e999a02af"
    },
    {
      "if_needed": 1,
      "project_id": "0d83100ef33c4eda8722b63a8da53dec"
    },
    {
      "if_needed": 0,
      "project_id": ""
    },
    {
      "if_needed": 1,
      "project_id": "0d83100ef33c4eda8722b63a8da53dec"
    }
  ]
}
```
- 输出解释:

    每个输出包含`if_needed`、`project_id`字段。
    `if_needed`表示首先判断资讯是否筛选为需要的，为`0`或`1`。`project_id`为匹配到的项目ESID。
    若为`0`，则`project_id`字段为空。
    
## 更新日志：
- v1.7:
    - 更新词典

"""
app = FastAPI(title='政府库黑白名单公告标题识别', description=dsc_txt, version='v7')

# model
classifier = ZZPredictorM1()
classifier.model_version = 'v7_0708'
classifier.args = classifier.set_model_parameter(model_version=classifier.model_version)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
classifier.use_cuda = False
classifier.args.quantized_model = True
classifier.args.onnx = True
classifier.cuda_device = 4
classifier.args.n_gpu = 1
classifier.args.eval_batch_size = 1
m1 = classifier.get_predict_model()

matching = ZZPredictorM2()
matching.model_version = 'v1.7.0.2'
matching.args = matching.set_model_parameter(model_version=matching.model_version)
matching.cuda_device = 4
matching.args.quantized_model = True
matching.args.onnx = True
matching.args.n_gpu = 1
matching.args.eval_batch_size = 1
matching.use_cuda = False
m2 = matching.get_predict_model()


class Input(BaseModel):
    """
    输入：一些标题信息组成的列表,[[省份，标题，日期]...]
    """
    texts: List[Tuple[str, str, str]] = Field(..., description='`List[Tuple[str, str, str]]`,列表', example=[
        ["贵州", "关于取消四川康福来药业集团有限公司等企业部分中标药品中标（挂网）资格的通知", "2020-06-11"],
        ["云南", "关于药品生产企业名称变更的公示", "2020-09-21"]
    ])


class Output(BaseModel):
    """
    输出：每个标题是否是中标资讯以及对应的项目
    """
    result: List[dict] = Field(...,
                           description='`List[dict...]`,输出结果的列表，`''`表示没有',
                           example=[{'if_needed': 1, 'project_id': '4791d93bd6934759a71fc7ea965ab37a'},
                                    {'if_needed': 1, 'project_id': 'b4e13cf11e33435b8b196b7e999a02af'}]
                           )


class RawTitle(BaseModel):
    texts: List[Tuple[str, str, str]]


@app.post("/predict/", tags=['Predict'], description='', response_model=Output)
@Logger.log_input_output()
async def predict_label(request_content: Input):
    titles = list(zip(*request_content.texts))[1]
    joined_texts = list(map(lambda strs: ' '.join(strs), request_content.texts))
    res1 = m1.predict(list(titles))[0]
    if 1 in res1:
        to_predict = ma.array(joined_texts, mask=1 - np.array(res1))
        res2_tmp = matching.predict_proj(m2, to_predict.compressed().tolist(), return_format='esid', query_max_size=100)
        res2 = np.where(res1, None, "")
        res2[np.array(res1, dtype=np.bool)] = res2_tmp
    else:
        res2 = ["" for i in res1]
    res = pd.DataFrame({'if_needed': res1, 'project_id': res2})
    return {'result': res.to_dict('records')}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9103)
