import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import List, Dict, Optional

import uvicorn
from fastapi import FastAPI, Body, Request
from loguru import logger
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from uvicorn.config import LOGGING_CONFIG

from mart.api_util.ecslogger import ECSLogger
from pharm_ai.prophet.prophet import Prophet, TimestampLogger

log_file = 'access.log'

if len(logger._core.handlers) < 2:
    printer = logger.add(sys.stderr, filter=lambda record: record["extra"].get("task") == "api", level='DEBUG')
    access_logger = logger.add(log_file, filter=lambda record: record["extra"].get("task") == "api", level='INFO')
api_logger = logger.bind(task="api")


def parse():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cuda_devices", nargs=4, type=int, default=[0, 0, 0, 0],
                        help="cuda devices of news_filter, ira, org_filter, org_ner model. DEFAULT: 0 0 0 0.",
                        metavar='cuda_i')
    parser.add_argument("-e", "--es-host", choices=['online', 'test'], default="test",
                        help="choose which ES server to use. DEFAULT: test.")
    parser.add_argument("-p", "--port", type=int, default=15058,
                        help="listening port. DEFAULT: 15058.")
    parser.add_argument("-g", "--ira-no-generative", action="store_false",
                        help="ira model not using generative.")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="print debug messages or not. DEFAULT: False.")
    parser.add_argument("--no-log-time", action="store_true",
                        help="Log timestamp or not. DEFAULT: False.")
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="Uvicorn workers. DEFAULT: 1.")
    parser.add_argument("--log-validation", action="store_true", help="Capture 422 ValidationError to log file.")
    args = parser.parse_args()
    return args


args = parse()

app = FastAPI(
    title='Prophet-智能医药投融资多语言资讯特定字段提炼系统',
    description="""
本文档提供医药投融资多语言资讯特定字段提炼系统各个接口的说明。

## 接口说明

本接口分为两个模块：`overall`和`partial`。

- `overall`模块内的接口为总接口

	- `Predict`：输入资讯的多个信息，抽取多个字段。
	
- `partial`模块内的单一接口功能，仅抽取资讯的特定字段。

	- `News Filter`：判断资讯的类别。
	
	- `Ira`：提取资讯I（investee）、R（round，轮次）、A（amout，金额）信息。
	
	- `Org Filter`：判断输入段落是否包含投资机构。
	
	- `Org Ner`： 抽取资讯中的机构实体。
	
各接口的使用说明请点击相应位置展开以查看。

**请求数据**的格式见相应部分的**Parameters->Request body->Schema**部分，
参考样例详见相应部分的**Parameters->Request body->Example Value**部分。

测试接口请点击相应部分的**Try it out**。

**响应数据**的格式见相应部分的**Responses->Successful Response (code 200)->Schema**部分，
参考样例详见相应部分的**Responses->Successful Response (code 200)->Example Value**部分。

## 更新日志

- v2.0: 
    - 修改investee、round、amount字段的提取算法，具有改写效果（可与原文内容不同）。
    - **partial**模块取消**Ira_filter**接口。
    - **partial**模块新增**Ira**接口。
- v2.1:
    - 修复资讯折叠的若干逻辑问题。
    - **partial**模块新增单独的**News_Folding**接口。
    - 总接口模块和资讯折叠子接口增加记录**请求事件时间戳**功能。
- v2.2:
    - 取消总接口的资讯折叠功能，降低超时发生的可能性。
- v3.0:
    - 优化各个任务的模型，提高准确性。
- v3.1:
    - 资讯折叠输入字段增加`label`（可选，默认`医药`），以便根据资讯类别（医药/非医药/非相关）判断是否需要折叠。
        - v3.1.1: 修复折叠接口请求数据的investee字段为company_id时无法匹配company_name导致无法折叠的问题。
- v3.2:
    - 投资机构识别增加**投资顾问**类。
- v3.3:
    - 修复**长资讯识别投资机构不全**以及**生成新投资机构**的问题。
    - 修复资讯折叠bug。
        - v3.3.1: 修改判断头条资讯的优先条件：资讯来源 > 发布时间
        - v3.3.2: 优化资讯折叠的查询，优化投资机构识别。
        - v3.3.3: 修复已发布资讯被折叠问题。
        - v3.3.4: 修复非相关参与折叠问题。
""",
    version='v3.3.4',
    openapi_tags=[
        {
            "name": "overall",
            "description": "本模块内的接口提供**整体**输入输出功能"
        },
        {
            "name": "partial",
            "description": "本模块内的接口提供**各个环节**的输入输出功能"
        }
    ]
)

ecslogger = ECSLogger('prophet', fastapi_app=app)


pp = Prophet(cuda_devices=args.cuda_devices, es_host=args.es_host)

# close multiprocessing
pp.close_multiprocessing()

example_esid = "8e1c5a252a418643e02f121fb7d7ab55"
example_title = "36氪首发 | 瞄准创新肿瘤药物，「科赛睿生物」获得近3000万美元B轮融资"
example_paragraphs = [
    "36氪获悉，创新肿瘤靶向药物研发商「科赛睿生物」，宣布获得近3000万美元B轮融资。"
    "本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，"
    "老股东君联资本，联想之星持续加持，易凯资本在本次交易中担任科赛睿生物的独家财务顾问。"
    "据悉，本轮融资所得将主要用于加速推进其多个肿瘤靶向创新药在全球范围的临床开发。",
    "纵观新药研发领域，肿瘤靶向药物是目前最活跃的方向之一，其中较为著名的包括PD-1/PD-L1、VEGF、HER2、TNF-α等。"
    "根据CFDA和CDE网站上的数据显示，目前中国正在申报临床、正在进行临床和正在申报生产的小分子靶向药物约有百种，"
    "大分子靶向药数十种，其中包括国内原研、仿制药和国外已上市的品种。",
    "在这个赛道上，科赛睿生物的竞争优势在于其自主研发的i-CR®技术平台，结合条件性重编程原代肿瘤细胞培养技术和高内涵药物筛选体系。",
    "其优势是能够在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，"
    "同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。",
    "目前，科赛睿已经通过与国内头部的肿瘤医学中心开展合作，进行多项前瞻性比对临床试验，"
    "已初步证明 i-CR®体系可以有效预测药物的的实际临床反应。",
    "研发管线的进度，是考察创新药物企业的重要环节。"
    "这一方面，科赛睿已经在合成致死和免疫治疗领域开发出针对癌症的一系列新药产品线，前期研发成果已经申请了多项国际和国内专利。",
    "其中，核心产品PC-002是一个针对MYC基因变异肿瘤的first-in-class小分子药物。"
    "MYC蛋白在超过50%的肿瘤中高表达，为最重要的“不可成药”肿瘤靶点之一，PC-002通过独特的MOA靶向MYC蛋白降解，"
    "选择性诱导MYC依赖的肿瘤细胞凋亡，即将开展美国临床2期。",
    "另外一个管线产品CTB-02针对pan-KRAS变异的肠癌和非小细胞肺癌，预期在2021年首先在澳大利亚进入1期临床。",
    "图片来源于前瞻研究院",
    "在快速增长的创新药市场中，全球化竞争加剧，因此产品管线研发的效率以及创新性显得尤其重要。",
    "靶向药物的特点是针对特定靶点产生作用，每个病人的病情不尽相同，适用的药物也各有不同，因此可以对肿瘤进行精准治疗。",
    "根据IMS的数据，预计目前全球肿瘤药物市场规模可达1500亿美元，肿瘤处方药销售额约为1100亿美元。"
    "援引前瞻研究院的分析，预计单克隆抗体类药物和小分子靶向药物在未来将占据最大的市场份额。"
]
example_publish_date = "2021-01-19"
example_source = "36氪"


class SingleArgBody(BaseModel):
    title: str = Field(..., example=example_title)
    paragraphs: List[str] = Field(..., example=example_paragraphs)
    publish_date: str = Field(..., example=example_publish_date)
    news_source: str = Field(..., example=example_source)


class Args(BaseModel):
    """
    输入字段`input_dic`和`request_start_time`：

    - `input_dic`: 包括资讯的**ID**（`article_id`）、**标题**（`title`）、**段落**（`paragraphs`）、
    **资讯发布日期**（`publish_date`）、**资讯出处**（`news_source`）信息。其中**资讯发布日期**需为`年-月-日`格式。
    - `request_start_time`: *（可选）*本次请求发起的时间戳，为精确到毫秒的长整数型。

    示例参考*Example Value*。
    """
    input_dic: Dict[str, SingleArgBody] = Field(
        ...,
        example={
            example_esid:
                {
                    "title": example_title,
                    "paragraphs": example_paragraphs,
                    "publish_date": example_publish_date,
                    "news_source": example_source}
        }
    )
    request_start_time: Optional[int] = Field(None, example=1619595258104)

    def dict(self, **kwargs):
        input_dict = super(Args, self).dict(**kwargs)
        input_dict['input_dic'] = [{'esid': esid, **fields} for esid, fields in input_dict['input_dic'].items()]
        return input_dict


example_org_output = [
    {
        "c1": [
            "清松资本"
        ],
        "c2": [
            "夏尔巴投资",
            "新世界投资",
            "Harbinger Venture",
            "招银国际",
            "君联资本",
            "朗盛资本",
            "联想之星"
        ],
        "f": [
            "CEC Capital"
        ]
    },
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
]
example_prophet_output = {
    "final_output": {
        "1153-20201216": {
            "news_filter_label": "医药",
            "amount": "3000万美元",
            "investee": "科赛睿生物",
            "round": "B轮",
            "investor/consultant": [example_org_output[0], [{}] * 11]
        }
    },
    "intermediary_output": {
        "8e1c5a252a418643e02f121fb7d7ab55": {
            "paragraphs": example_paragraphs,
            "news_filter_label": ["医药"],
            "ira_filter_labels": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "org_filter_labels": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "investor/consultant": [example_org_output[0], [{}] * 11]
        }
    }
}


def convert_prophet_output_to_log(output: dict):
    output_dict = output.copy()
    res = [{'final_output': output_field, 'intermediary_output': intermediary_field}
           for (esid, output_field), (_, intermediary_field) in zip(output_dict['final_output'].items(),
                                                                    output_dict['intermediary_output'].items())]
    return res


@app.post("/prophet/", tags=['overall'],
          responses={
              200: {
                  "description": """
- 输出格式：


其中`final_output`为最终需要展示所需的数据，`intermediary_output`无需展示，但需保存以便后续迭代调取。
输出字典包括`final_output`和`intermediary_output`。
                  """,
                  "content": {
                      "application/json": {
                          "example": example_prophet_output
                      }
                  }
              }
          })
@ecslogger.log_input_output(output_func=convert_prophet_output_to_log)
async def predict(input_args: Args = Body(...)):
    """
    给定一条中英文资讯的**标题**、**段落**、**资讯出处**、**资讯发布日期**信息,
    输出该资讯的**融资方**、**融资金额**、**融资轮次**、**投资方/投资中介**。
    """
    if not args.no_log_time:
        time_loggers = []
        arrival_time = TimestampLogger.get_time()
        for k in input_args.input_dic.keys():
            time_logger = TimestampLogger(k)
            time_logger.log_event_arrival_time(arrival_time)
            time_logger.log_event_create_time(input_args.request_start_time)
            time_loggers.append(time_logger)
    to_pred = {k: v.dict() for k, v in input_args.input_dic.items()}
    if not args.no_log_time:
        out = pp.prophet(to_pred, iter(time_loggers))
        for time_logger in time_loggers:
            time_logger.upload_to_es()
    else:
        out = pp.prophet(to_pred)
    return out


class News(BaseModel):
    title: str = Field(..., example=example_title)
    paragraphs: List[str] = Field(..., example=example_paragraphs)


@app.post("/news_filter/", tags=['partial'],
          responses={200: {"content": {"application/json": {"example": "医药"}}}})
@ecslogger.log_input_output()
async def news_filter(news: News):
    """
    资讯折叠接口，输入一条资讯的**标题**和**段落**，输出该资讯的类别(`医药`，`非医药`，`非相关`)
    """
    title = news.title
    paras = news.paragraphs
    res = pp.news_filter.predict(title, paras)
    return res[0]


class SinglePara(BaseModel):
    input: str = Field(..., example=example_paragraphs[0])


@app.post("/ira/", tags=['partial'],
          responses={200: {"content": {"application/json": {
              "example": ["科赛睿生物", "B轮", "3000万美元"]}}}})
@ecslogger.log_input_output()
async def ira(text: SinglePara = Body(...)):
    """
    输入：
        需要提取**investee**、**round**、**amount**的一条资讯文本。
    输出：
        提取的**investee**、**round**、**amount**结果。
    """
    res = pp.ner.predict_ira([1], [text.input])
    return res


class FoldingInput(BaseModel):
    """资讯折叠的输入格式。

    输入字段：

    - **资讯ID**（`article_id`）

    - **发表日期**（``publish_date）

    - **来源**（`source`）

    - **融资方**（`investee`）

    - **融资轮次**（`round`）

    - **融资金额**（`amount`）

    - **资讯类别**（`label`）：（可选字段）。值：医药（默认）、非医药、非相关。

    - `request_start_time`：（可选字段）调用本接口的网络时间戳，用于保存以便后续进行性能分析。
    """

    article_id: str = Field(..., example="23b2714b68100b3354ad2df3cd32fad3")
    publish_date: str = Field(..., example="2021-02-22")
    source: str = Field(..., example="光速创投")
    investee: Optional[str] = Field('', example="黑湖智造")
    round: Optional[str] = Field('', example="近5亿元人民币")
    amount: Optional[str] = Field('', example="C轮")
    label: Optional[str] = Field(None, example="医药")
    request_start_time: Optional[int] = Field(None, example=1619595258104)


class FoldingResult(BaseModel):
    """资讯折叠的输出格式。

    返回：
        - 当前资讯是否为头条head。
        - 当前资讯从属的头条资讯ID。若资讯本身为头条资讯，则返回自身ID。
        - 经过本次预测后，ES数据库中已有资讯的从属头条资讯更新情况。`{发生变化的资讯ID: 新的从属头条资讯ID}`
    """
    is_head: bool = Field(..., example=False)
    head_article: str = Field(..., example='13b530a01eae2b2b813bbf604f1789af')
    history_article_changes: Dict = Field(..., example={'e3f2d3ed77cfac98d9278badd4dd808':
                                                            'e7f9e17648cc1b7c2ff4b6f0e6668a85'})


def convert_folding_result_to_log(result: FoldingResult):
    result_dict = result.copy()
    result_dict['history_article_changes'] = [
        {'from': esid_from, 'to': esid_to}
        for esid_from, esid_to in result_dict['history_article_changes'].items()
    ]
    return result_dict


@app.post("/news_folding/", tags=['partial'], response_model=FoldingResult)
@ecslogger.log_input_output(output_func=convert_folding_result_to_log)
async def news_folding(inputs: FoldingInput = Body(...)):
    """
        # 资讯折叠接口

        ## 使用场景
        用于对资讯进行**折叠**判断，尤其用于**校正资讯折叠结果**。

        ## 校正折叠的机制
        在初次调用接口折叠时，结果并不完全准确，因为资讯具有实时性，存在先后顺序。
        初次折叠时，先折叠的资讯无法获取后折叠的资讯信息。

        ## 使用方法
        多次调用本接口，**延迟适当时间进行二次或多次调用**。

        延迟时间的设定需等待足够多的新资讯清洗完毕，以便校正折叠时有充分的判断依据。

        延迟时间越大，折叠准确率越高。建议设置大于1分钟。

        ## 注意事项

        *注意：自v2.1版本后，调用资讯折叠接口，除了返回请求数据的折叠结果，同时也会在预测时修改es中已有资讯的折叠结果(`similar_esid`字段)！*
    """
    if not args.no_log_time:
        time_logger = TimestampLogger(inputs.article_id)
        time_logger.log_event_arrival_time()
        time_logger.log_event_create_time(inputs.request_start_time)
        res = pp.news_folding.predict(inputs.article_id, inputs.publish_date, inputs.investee, inputs.round,
                                      inputs.amount, news_filter_label=inputs.label or '医药',
                                      return_changes=True, time_logger=time_logger)
        time_logger.upload_to_es()
    else:
        res = pp.news_folding.predict(inputs.article_id, inputs.publish_date, inputs.investee, inputs.round,
                                      inputs.amount, news_filter_label=inputs.label or '医药', return_changes=True)
    return {'is_head': res[0], 'head_article': res[1], 'history_article_changes': res[2]}


class Texts(BaseModel):
    """
    输入字段：

    - paragraphs： 各个段落文本。
    - dates：对应的发布日期, 日期格式：字符串`YYYY-MM-DD[T]HH:MM[:SS[.ffffff]][Z or [±]HH[:]MM]]]` 或Unix时间浮点数。
    如`1619596258213`, `"2020-1-3T12:50:31"`。
    """
    paragraphs: List[str] = Field(..., example=example_paragraphs)
    dates: Optional[datetime] = Field(None, example=1619596258213)


@app.post("/org_filter/", tags=['partial'],
          responses={200: {"content": {"application/json": {
              "example": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          }}}})
@ecslogger.log_input_output()
async def org_filter(texts: Texts = Body(...)):
    """
    投资段落分类识别接口

    **功能**

    输入一条资讯的各个段落(`paragraphs`)和发布日期（`dates`）， 返回每个段落的分类结果（`0` or `1`）。
    """
    paras = texts.paragraphs
    dates = texts.dates
    res = pp.org_filter.predict(date=dates, para=paras)
    return res


@app.post("/org_ner/", tags=['partial'],
          responses={200: {"content": {"application/json": {"example": example_org_output}}}})
@ecslogger.log_input_output()
async def org_ner(texts: Texts):
    """
    投资机构识别接口。

    输入需要识别投资机构的各个段落，返回每个段落识别的投资机构实体，包括`c1`、`c2`、`c3`或`f`类。
    """
    to_predict = texts.paragraphs
    res = pp.ner.predict_org("医药", [1] * len(to_predict), to_predict)
    return res


if args.log_validation:
    api_logger.debug('Log 422 error is ON.')


    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        api_logger.info(exc.body)
        error_info = [{k: v for k, v in rr.items() if k in ['loc', 'msg', 'type']} for rr in exc.errors()]
        api_logger.error(error_info)
        return await request_validation_exception_handler(request, exc)

def add_uvicorn_file_log():
    """Modify default uvicorn LOGGING_CONFIG, so that logs are also recorded to file."""
    LOGGING_CONFIG['formatters']['default']['fmt'] = "%(asctime)s %(levelprefix)s %(message)s"
    LOGGING_CONFIG['formatters']['access']['fmt'] = '%(asctime)s %(levelprefix)s %(client_addr)s - ' \
                                                    '"%(request_line)s" %(status_code)s'
    LOGGING_CONFIG['handlers']['file_default'] = {'formatter': 'default', 'class': 'logging.FileHandler',
                                                  'filename': log_file}
    LOGGING_CONFIG['handlers']['file_access'] = {'formatter': 'access', 'class': 'logging.FileHandler',
                                                 'filename': log_file}
    LOGGING_CONFIG['loggers']['uvicorn']['handlers'].append('file_default')
    LOGGING_CONFIG['loggers']['uvicorn.access']['handlers'].append('file_access')

if __name__ == "__main__":
    add_uvicorn_file_log()
    uvicorn.run("api:app", host="0.0.0.0", port=args.port, workers=args.workers, log_config=LOGGING_CONFIG)
