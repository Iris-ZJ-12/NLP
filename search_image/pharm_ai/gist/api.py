import uvicorn

from typing import List
from fastapi import FastAPI
from harvesttext import HarvestText
from pydantic import BaseModel, Field
from mart.api_util.ecslogger import ECSLogger
from pharm_ai.gist.models.pipeline import Pipeline


app = FastAPI(
    title="通用摘要接口\ngist\nExtractive Summarization Generation",
    version="v1.0",
    description="""
本文档提供通用摘要生成相关接口的说明。
本接口目前提供抽取式摘要生成功能，摘要由原文中的句子组成，默认句子数为3。

## 接口说明

各接口的使用说明请点击相应位置展开以查看。

**请求数据**的格式见相应部分的**Parameters->Request body->Schema**部分，
参考样例详见相应部分的**Parameters->Request body->Example Value**部分。

测试接口请点击相应部分的**Try it out**。

**响应数据**的格式见相应部分的**Responses->Successful Response (code 200)->Schema**部分，
参考样例详见相应部分的**Responses->Successful Response (code 200)->Example Value**部分。
    """,
    openapi_tags=[
        {
            'name': 'doc_summarize',
            'description': '摘要生成',
        },
        {
            'name': 'sents_summarize',
            'description': '摘要生成(已分句)',
        }
    ]
)

logger = ECSLogger('gist', app)
pipe = Pipeline('cuda')
ht = HarvestText()


class DocInput(BaseModel):
    doc: str = Field(
        ...,
        example='BTK，即布鲁顿酪氨酸蛋白激酶（Bruton’s tyrosine kinase），是B细胞受体通路重要信号分子，在B细胞的各个发育阶段表达，参与调控B细胞的'
                '增殖、分化与凋亡，在恶性B细胞的生存及扩散中起着重要作用，是针对B细胞类肿瘤及B细胞类免疫疾病的研究热点。当前，BTK抑制剂已经上市的产品'
                '共有4款，分别是：伊布替尼、阿卡替尼、tirabrutinib和泽布替尼。然而，市场表现仍然是伊布替尼一家独大，2020年将突破100亿美元大关，持'
                '续担当BTK领域的“仙界领袖”。正所谓，人红是非多，后起之星都在努力着将伊布替尼拉下神坛。在下月即将召开的62届美国血液学会（ASH）年会上，'
                '包括伊布替尼、阿卡替尼、泽布替尼、奥布替尼以及二代BTK抑制剂LOXO-305等都带来了各自修炼成果。作为首登仙位的伊布替尼，虽然存在选择性等'
                '一些问题，但却牢牢把控着市场脉搏，也未停止自身的进阶之路。至今，伊布替尼已经获批了包括慢性淋巴细胞白血病、小淋巴细胞性淋巴瘤、套细胞淋'
                '巴瘤、华氏巨球蛋白血症、移植物抗宿主病和边缘区淋巴瘤等6个适应症，同时也在探索联合用药解决自身的耐药问题等。本届会议伊布替尼也带来多项'
                '研究成功，但更多的是长期随访数据。阿卡替尼、泽布替尼和奥布替尼在本届ASH会议也是众多研究发布，限于篇幅我们不能全部收列于此，仅对这几款上'
                '市或即将上的药物进行一次适应症的选择性比较。慢性淋巴细胞白血病（CLL）/小淋巴细胞性淋巴瘤（SLL）是BTK产品的重要领域，上市的4款产品中，'
                '3个覆盖该适应症，处于国内上市审批阶段的奥布替尼也将该适应症与MCL一起作为首发领域。伊布替尼在CLL/SLL一线治疗的多个随机Ⅲ期研究中显示出'
                '显著的PFS和OS获益。基于单药或联合治疗的报告进一步证明，伊布替尼在一线和复发/难治性携带TP53异常的患者有良好的PFS益处，但长期预后数据有'
                '限。'
    )
    top_k: int = Field(3, example=3)


class DocOutput(BaseModel):
    doc: str


class SentsInput(BaseModel):
    sents: List[str] = Field(..., example=[
        'BTK，即布鲁顿酪氨酸蛋白激酶（Bruton’s tyrosine kinase），是B细胞受体通路重要信号分子，在B细胞的各个发育阶段表达，参与调控B细胞的增殖、分化与凋亡，在恶性B细胞的生存及扩散中'
        '起着重要作用，是针对B细胞类肿瘤及B细胞类免疫疾病的研究热点。',
        '当前，BTK抑制剂已经上市的产品共有4款，分别是：伊布替尼、阿卡替尼、tirabrutinib和泽布替尼。',
        '然而，市场表现仍然是伊布替尼一家独大，2020年将突破100亿美元大关，持续担当BTK领域的“仙界领袖”。',
        '正所谓，人红是非多，后起之星都在努力着将伊布替尼拉下神坛。',
        '在下月即将召开的62届美国血液学会（ASH）年会上，包括伊布替尼、阿卡替尼、泽布替尼、奥布替尼以及二代BTK抑制剂LOXO-305等都带来了各自修炼成果。',
        '作为首登仙位的伊布替尼，虽然存在选择性等一些问题，但却牢牢把控着市场脉搏，也未停止自身的进阶之路。',
        '至今，伊布替尼已经获批了包括慢性淋巴细胞白血病、小淋巴细胞性淋巴瘤、套细胞淋巴瘤、华氏巨球蛋白血症、移植物抗宿主病和边缘区淋巴瘤等6个适应症，同时也在探索联合用药解决自身的耐药问题等。',
        '本届会议伊布替尼也带来多项研究成功，但更多的是长期随访数据。',
        '阿卡替尼、泽布替尼和奥布替尼在本届ASH会议也是众多研究发布，限于篇幅我们不能全部收列于此，仅对这几款上市或即将上的药物进行一次适应症的选择性比较。',
        '慢性淋巴细胞白血病（CLL）/小淋巴细胞性淋巴瘤（SLL）是BTK产品的重要领域，上市的4款产品中，3个覆盖该适应症，处于国内上市审批阶段的奥布替尼也将该适应症与MCL一起作为首发领域。',
        '伊布替尼在CLL/SLL一线治疗的多个随机Ⅲ期研究中显示出显著的PFS和OS获益。',
        '基于单药或联合治疗的报告进一步证明，伊布替尼在一线和复发/难治性携带TP53异常的患者有良好的PFS益处，但长期预后数据有限。']
    )
    top_k: int = Field(3, example=3)


class SentsOutput(BaseModel):
    sents: List[str]


@app.post("/doc_summarize/", tags=['doc_summarize'], response_model=DocOutput)
@logger.log_input_output()
async def doc_summarize(doc_input: DocInput):
    sents = ht.cut_sentences(doc_input.doc)
    summary_sents = pipe(sents, doc_input.top_k)
    res = {'doc': ' '.join(summary_sents)}
    return res


@app.post("/sents_summarize/", tags=['sents_summarize'], response_model=SentsOutput)
@logger.log_input_output()
async def sents_summarize(sents_input: SentsInput):
    summary_sents = pipe(sents_input.sents, sents_input.top_k)
    res = {'sents': summary_sents}
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4396)
