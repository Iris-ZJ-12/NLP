# -*- coding: UTF-8 -*-
"""
Description : 
"""
import uvicorn

from asyncio import create_task
from fastapi import FastAPI
from pydantic import BaseModel, Field
from mart.api_util.ecslogger import ECSLogger
from pharm_ai.pom.model import PomPredictor

app = FastAPI(
    title="🤪媒体文章nlp处理系统\nPOM\n(Public Opinion Monitoring)",
    version="v1.3",
    description="""
本文档提供媒体新闻库相关接口的说明。
给定一篇文章，通过标题判断它是否是属于医药行业的文章（`is_medical_news`），对于一篇属于医药行业的文章，给其打上分类标签（`news_classification_labels`）
，并给它打上内容标签（`news_content_labels`），另外，对于一篇医药行业问题，根据前面部分句子撰写摘要（`abstract_generation`）。

## 接口说明

各接口的使用说明请点击相应位置展开以查看。

**请求数据**的格式见相应部分的**Parameters->Request body->Schema**部分，
参考样例详见相应部分的**Parameters->Request body->Example Value**部分。

测试接口请点击相应部分的**Try it out**。

**响应数据**的格式见相应部分的**Responses->Successful Response (code 200)->Schema**部分，
参考样例详见相应部分的**Responses->Successful Response (code 200)->Example Value**部分。

## 更新日志：
- v1(21/04/21):
    - 添加新闻文章过滤功能
    - 添加新闻文章分类功能
    - 添加新闻文章打标签功能
    - 添加新闻文章摘要撰写功能
- v2(21/05/27):
    - 根据标题文字，过滤明显不是医药新闻的文章，如含”招聘“的文章不是医药新闻;
    - 新增文章来源（source）字段，参与过滤文章是否是医药新闻，如“医药魔方”公众号下的文章是医药新闻;
    - 修正二级分类，输出中新增“其他”类别;
    - 优化内容标签的输出，输出中加入指定重点标签，去除指定不重要标签，去除重复标签，比如加入“国采”，去除“B村”;
    - 优化摘要输出，将一些常见非摘要句去除，比如去除：“▎药明康德内容团队编辑”;
- v3(21/07/29):
    - 移除单个功能接口
    - 更新二分类模型
    - 更新摘要模型
- v4(21/10/26):
    - 拆分摘要和翻译接口
    - 异步调用优化速度
- v5(21/12/16):
    - 标签提取增加适应症规则
    - 增加标签提取单独接口
    """,
    openapi_tags=[
        {
            'name': 'overall',
            'description': '总体功能:输入一篇文章的标题（title）和内容(content)，输出该文章是否是医药信息，若是医药文章，对其进行分类、打标签，摘要生成。',
        },
        {
            'name': 'label',
            'description': '单独打标签功能，输入标题和摘要，提取标签。'
        }
    ]
)

logger = ECSLogger('pom', app)
predictor = PomPredictor()


# #####OverAll###################################
class OverAllInput(BaseModel):
    """
    输入字段包括： 标题（title）和 文本（texts）
    """
    source: str = Field(default='医药魔方', example="医药魔方")
    title: str = Field(..., example="聚焦ASH 2020：BTKi暗潮涌动")
    paragraphs: str = Field(..., example=
    'BTK，即布鲁顿酪氨酸蛋白激酶（Bruton’s tyrosine kinase），是B细胞受体通路重要信号分子，在B细胞的各个发育阶段表达，参与调控B细胞的'
    '增殖、分化与凋亡，在恶性B细胞的生存及扩散中起着重要作用，是针对B细胞类肿瘤及B细胞类免疫疾病的研究热点。'
    '当前，BTK抑制剂已经上市的产品共有4款，分别是：伊布替尼、阿卡替尼、tirabrutinib和泽布替尼。'
    '然而，市场表现仍然是伊布替尼一家独大，2020年将突破100亿美元大关，持续担当BTK领域的“仙界领袖”。'
    '正所谓，人红是非多，后起之星都在努力着将伊布替尼拉下神坛。'
    '在下月即将召开的62届美国血液学会（ASH）年会上，包括伊布替尼、阿卡替尼、泽布替尼、奥布替尼以及二代BTK抑制剂LOXO-305等都带来了各自修炼成果。'
    '作为首登仙位的伊布替尼，虽然存在选择性等一些问题，但却牢牢把控着市场脉搏，也未停止自身的进阶之路。'
    '至今，伊布替尼已经获批了包括慢性淋巴细胞白血病、小淋巴细胞性淋巴瘤、套细胞淋巴瘤、华氏巨球蛋白血症、移植物抗宿主病和边缘区淋巴瘤等6个适应症，同时也在探索联合用药解决自身的耐药问题等。'
    '本届会议伊布替尼也带来多项研究成功，但更多的是长期随访数据。'
    '阿卡替尼、泽布替尼和奥布替尼在本届ASH会议也是众多研究发布，限于篇幅我们不能全部收列于此，仅对这几款上市或即将上的药物进行一次适应症的选择性比较。'
    '慢性淋巴细胞白血病（CLL）/小淋巴细胞性淋巴瘤（SLL）是BTK产品的重要领域，上市的4款产品中，3个覆盖该适应症，处于国内上市审批阶段的奥布替尼也将该适应症与MCL一起作为首发领域。'
    '伊布替尼在CLL/SLL一线治疗的多个随机Ⅲ期研究中显示出显著的PFS和OS获益。'
    '基于单药或联合治疗的报告进一步证明，伊布替尼在一线和复发/难治性携带TP53异常的患者有良好的PFS益处，但长期预后数据有限。'
                            )


class OverAllOutput(BaseModel):
    """
    输出字段:是否是医药新闻，它的分类标签是什么，它的内容标签是什么,摘要生成
    """
    is_medical_news: str
    news_classification_labels: str
    news_content_labels: str
    abstract_generation: str


class LabelInput(BaseModel):
    title: str = Field(..., example="聚焦ASH 2020：BTKi暗潮涌动")
    abstract: str = Field(..., example="BTK，即布鲁顿酪氨酸蛋白激酶（Bruton’s tyrosine kinase），是B细胞受体通路重要信号分子，在B"
                                       "细胞的各个发育阶段表达，参与调控B细胞的增殖、分化与凋亡，在恶性B细胞的生存及扩散中起着重要作用"
                                       "，是针对B细胞类肿瘤及B细胞类免疫疾病的研究热点。 当前，BTK抑制剂已经上市的产品共有4款，分别是"
                                       "：伊布替尼、阿卡替尼、tirabrutinib和泽布替尼。 然而，市场表现仍然是伊布替尼一家独大，2020年"
                                       "将突破100亿美元大关，持续担当BTK领域的“仙界领袖”。")


class LabelOutput(BaseModel):
    news_content_labels: str


@app.post("/overall/", tags=['overall'], response_model=OverAllOutput)
@logger.log_input_output()
async def predict(overall_input: OverAllInput):
    text_title = overall_input.title
    source = overall_input.source
    is_medical_news = predictor.predict_medical_news(text_title, source)

    if is_medical_news == '1':
        news_classification_labels_task = create_task(predictor.predict_news_classification_label(text_title))

        doc = overall_input.paragraphs
        cleaned = predictor.clean_text(doc)
        abstract_task = create_task(predictor.predict_summary(cleaned[:10]))

        news_classification_labels = await news_classification_labels_task
        abstract = await abstract_task
        content = " ".join(cleaned)
        news_content_labels = await predictor.predict_news_content_label(text_title, content, abstract)
        res = {'is_medical_news': '1', 'news_classification_labels': news_classification_labels,
               'news_content_labels': news_content_labels,
               'abstract_generation': abstract}
    else:
        res = {'is_medical_news': '0', 'news_classification_labels': '', 'news_content_labels': '',
               'abstract_generation': ''}
    return res


@app.post("/label/", tags=['label'], response_model=LabelOutput)
async def label(label_input: LabelInput):
    title = label_input.title
    abstract = label_input.abstract
    news_content_labels = await predictor.predict_news_content_label(title, abstract)
    return {'news_content_labels': news_content_labels}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6034)
