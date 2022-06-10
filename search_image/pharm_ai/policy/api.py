import re
from argparse import ArgumentParser
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from mart.api_util.logger import Logger
from pydantic import BaseModel, Field
from scipy.special import softmax

from predict import PolicyIdentifier

arg_parser = ArgumentParser()
arg_parser.add_argument('-p', '--port', type=int, default=5061, help='expose port.')
arg_parser.add_argument('-c', '--cuda_device', type=int, default=-1, help='Cuda device. DEFAULT: -1')
arg_parser.add_argument('-n', '--n_gpu', type=int, default=1, help='GPU number to use. DEFAULT: 1.')
arg_parser.add_argument('-v', '--version', type=int, default=5, help='Model version. DEFAULT: 5')
args = arg_parser.parse_args()

descrip_base = """本文档提供**政策标题处理**的接口文档。
### p1: 政策标题多分类接口
### P2：识别政策类标题接口
+ 使用说明：输入XXX文本
### p3：政策文号提取接口
+ 使用说明：html文本"""
app = FastAPI(
    title='政策分类识别',
    description=descrip_base,
    version="v1",
    openapi_tags=[
        {"name": "p1", "description": "政策分类接口"},
        {"name": "p2", "description": "政策识别接口"},
        {"name": "p3", "description": "政策文号识别接口"}]
)

policy_identifier = PolicyIdentifier()


class Texts(BaseModel):
    """输入格式：一系列标题（字符串数组）。"""
    texts: List[Optional[str]] = Field(
        ...,
        example=[
            "人民日报评论员:严守规矩 增强本领——论学习贯彻习近平总书记在中青年干部培训班开班式上重要讲话",
            "关于切实做好有关异地就医工作的通知 2020-12-07",
            "中美福源获得SFR1882临床试验批件"]
    )


class Result(BaseModel):
    """(预测标签, 预测概率)"""
    result: List[List] = Field(
        ...,
        example=[[0, 0.9928], [1, 0.9351], [0, 0.9971]]
    )


@app.post("/identifyPolicy", tags=['p2'], response_model=Result)
@Logger.log_input_output()
def identify_policy(texts: Texts):
    """批量预测多条资讯的**实际类型**。"""
    res = []
    if not texts.texts:
        return {'result': []}
    predictions, raw_outputs = policy_identifier.predict(texts.texts)
    prob = softmax(raw_outputs)
    for i, label in enumerate(predictions):
        prob = round(softmax(raw_outputs[i,]).max(), 4)
        res.append([label, prob])
    return {'result': res}


class DocNum(BaseModel):
    result: str = Field(..., example="京人社发〔2021〕33号")

example_str = '<p style="text-indent: 0em; text-align: center;">京人社事业发〔2021〕33号</p>'


@app.post("/extractPolicyDocNum", tags=['p3'], response_model=DocNum)
@Logger.log_input_output()
def extract_policy_docnum(text: str = example_str):
    # patt = re.compile("\(?([%s]\w{1,}?[\[〔][1-2][0-9]{3}[〕\]][0-9]{1,}?号)\)?"%abbv)
    # num = re.search(patt, text["_source"]["content"])
    abbv = "冀豫云辽黑湘皖鲁苏浙赣鄂甘晋陕吉闽贵粤川青藏琼新桂宁蒙内京津沪渝国中医药卫"
    patt = re.compile(">\(?([%s]\w{1,10}?[\[〔][1-2][0-9]{3}[〕\]][0-9]{1,}?号)\)?<" % abbv)
    num = re.search(patt, text)
    if num:
        res = num.group(1)
    else:
        res = ""
    return {'result': res}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", reload=True, port=args.port)
