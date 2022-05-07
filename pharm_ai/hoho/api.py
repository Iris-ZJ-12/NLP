import uvicorn

from typing import List, Dict
from pydantic import BaseModel, Field
from fastapi import FastAPI
from pharm_ai.util.api_util import Logger
from pharm_ai.hoho.predict import HohoPredictor

description = """
本文档提供**医院名称匹配v1.0**的接口文档。

#### 功能描述

输入医院名字和数字N, 返回top N个匹配到的医院名。

match接口针对单条数据, match_batch接口针对多条数据。

#### 使用说明

**请求数据**的格式见相应部分的**Parameters->Request body->Schema**部分，

测试接口请点击相应部分的**Try it out**。

**响应数据**的格式见相应部分的**Responses->Successful Response (code 200)->Description/Schema**部分，
"""
app = FastAPI(title="医院名称匹配", description=description)
predictor = HohoPredictor()


class Query(BaseModel):
    name: str = Field(..., example='广东省揭阳市中医院')
    province: str = Field('', example='广东')
    city: str = Field('', example='')
    district: str = Field('', example='')
    mapping_id: str = Field(..., example='12345')
    topk: int = Field(..., example=10)


class QueryBatch(BaseModel):
    names: List[Dict[str, str]] = Field(
        ...,
        example=[
            {
                'name': '广东省揭阳市中医院',
                'province': '广东',
                'city': '',
                'district': '',
                'mapping_id': '12345'
            },
            {
                'name': '丰县中医医院',
                'province': '江苏',
                'city': '',
                'district': '',
                'mapping_id': '12345'
            }
        ]
    )
    topk: int = Field(..., example=10)


def transform_output(output):
    return {
        'matched_name': output[0],
        'code': output[1],
        'standard_name': output[2],
        'model_output': int(output[3]),
        'model_prob': float(output[4])
    }


@app.post("/match/")
@Logger.log_input_output()
async def predict(query: Query):
    area = [query.province, query.city, query.district]
    outputs = predictor.predict(query.name, area, topk=query.topk)
    outputs = sorted(outputs, key=lambda x: x[-1])[::-1]
    result = {
        'input_name': query.name,
        'mapping_id': query.mapping_id,
        'result': [transform_output(out) for out in outputs]
    }
    return result


@app.post("/match_batch/")
@Logger.log_input_output()
async def predict_batch(querys: QueryBatch):
    result = []
    for query in querys.names:
        area = [query.get('province', ''), query.get('city', ''), query.get('district', '')]
        outputs = predictor.predict(query['name'], area, topk=querys.topk)
        outputs = sorted(outputs, key=lambda x: x[-1])[::-1]
        result.append({
            'input_name': query['name'],
            'mapping_id': query['mapping_id'],
            'result': [transform_output(out) for out in outputs]
        })
    return result


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=16228)
