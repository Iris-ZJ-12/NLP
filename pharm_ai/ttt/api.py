# encoding: utf-8
'''
@author: zyl
@file: api.py
@time: 2021/11/15 15:03
@desc:
'''

from enum import Enum
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import List, Set, Optional, Dict, Union

from fastapi import Body, FastAPI, Query
from pydantic import BaseModel, Field, EmailStr

description = """
测试接口--治疗类型分类

输入：标题和入排标准

输出：这个文本（标题+入排标准）里面是否有治疗类型
"""

title = "治疗类型分类---测试版"

version = "0.0.1"

contact = {
    "name": "张玉良",
    "url": "https://docs.qq.com/sheet/DYVpVdmFCWkhPSGVQ?tab=BB08J2",
    "email": "1137379695@qq.com",
}
license_info = {
    "name": "Apache 2.0",
    "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
}

openapi_tags = [
    {
        "name": "TherapyCls",
        "description": "治疗类型分类",
    }
]
app = FastAPI(title=title, version=version, contact=contact, description=description,
              license_info=license_info, openapi_tags=openapi_tags)


class TherapyClsRequest(BaseModel):
    title: str = Field(..., example="Weekly Gemcitabine and Trastuzumab in the Treatment of Patients With Human "
                                    "Epidermal Growth Factor Receptor 2 (HER2) Positive Metastatic Breast Cancer;"
                                    "Phase II Trial of Weekly Gemcitabine and Herceptin in the Treatment of Patients "
                                    "With HER-2 Overexpressing Metastatic Breast Cancer", title="标题",
                       description="str")
    criteria: Optional[str] = Field(..., example="Inclusion Criteria: To be included in this study, you must meet the "
                                                 "following criteria:\n - Her-2 positive metastatic breast cancer "
                                                 "confirmed by biopsy\n - Measurable disease\n - Able to perform activities "
                                                 "of daily living without considerable\n - No previous chemotherapy with "
                                                 "gemcitabine\n - No more than one prior chemotherapy regimen for "
                                                 "metastatic breast cancer\n - Adequate bone marrow, liver and renal "
                                                 "function\n - Normal heart function\n - Give written informed consent"
                                                 " prior to entering this study.", title="入排标准",
                                    description="标题对应的入排标准")


class TherapyClsResponse(BaseModel):
    therapies: Optional[list] = Field(None, example="therapies 1", title="治疗类型", description="有治疗类型就返回，没有拉倒" )


def get_model():
    from pharm_ai.ttt.predict import PredictorV1
    import os
    import pandas as pd
    p = PredictorV1()

    p.use_model = 'mt5'
    p.model_type = 'mt5'
    p.model_version = 'v1.3.0.0'

    p.args = p.set_model_parameter(model_version=p.model_version, args=p._set_args(),
                                            save_dir="/home/zyl/disk/PharmAI/pharm_ai/ttt/")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    p.use_cuda = True
    p.args.n_gpu = 1
    p.cuda_device = 0

    p.args.eval_batch_size = 64  # 256
    p.args.max_seq_length = 512

    return p.get_predict_model()

model = get_model()

from pharm_ai.ttt.dt import TTTDT, therapy_num_en_dict
def data_process(title,criteria):
    criteria = TTTDT.get_inclusion_criteria(criteria)
    return 'classification: '+'Title: ' + str(title) + '\n | Criteria: ' + criteria

def refine_res(res_str:str):
    if res_str==',':
        return None
    else:
        r = set(res_str.split(','))
        return [therapy_num_en_dict.get(int(i)) for i in r]

@app.post("/therapy_classification/", tags=['TherapyCls'], description='治疗分类', response_model=TherapyClsResponse,
          summary="治疗类型分类url", response_description="返回字典，包含key-therapies治疗类型")
async def therapy_classification(MyRequest: TherapyClsRequest):
    to_predict = [data_process(MyRequest.title, MyRequest.criteria)]
    res = model.predict(to_predict)[0]
    res = refine_res(res)
    return {"therapies": res}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3245)
