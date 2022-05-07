from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from mart.api_util.ecslogger import logger, log_input_output
from argparse import ArgumentParser

from pharm_ai.ak.predictor import T5Predictor

parser = ArgumentParser()
parser.add_argument("-c", "--cuda_device", type=int, default=0,
                    help="Cuda device. DEFAULT: 0.")
parser.add_argument("-p", "--port", type=int, default=11515,
                    help="listening port. DEFAULT: 11515.")
args = parser.parse_args()


doc = """
本文档提供HTML中标文档提取指定产品名称的接口说明。

## 接口说明

各接口的使用说明请点击相应位置展开以查看。

**请求数据**的格式见相应部分的**Parameters->Request body->Schema**部分，
参考样例详见相应部分的**Parameters->Request body->Example Value**部分。

测试接口请点击相应部分的**Try it out**。

**响应数据**的格式见相应部分的**Responses->Successful Response (code 200)->Schema**部分，
参考样例详见相应部分的**Responses->Successful Response (code 200)->Example Value**部分。

## 更新日志
"""
version = 'v1.0'

app = FastAPI(
    title="HTML文本中标文档指定产品名称的提取",
    description=doc,
    version=version,
    openapi_tags=[
        {"name": "main", "description": "主要接口"}
    ]
)

predictor = T5Predictor(version=version, cuda_device=args.cuda_device)
predictor.close_multiprocessing()

example_esid = "48a85ae1f8744a17db679002361a3eb5"
example_html = """
<p style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">一、项目编号（或招标编号、政府采购计划编号、采购计划备案文号等，如有）：</span>
</p><p style="line-height:150%"><span
        style="font-size: 16px;line-height:150%;font-family:宋体">442000-202009-hp059-0057</span></p><p
        style="line-height:150%"><span
        style="font-size:16px;line-height:150%;font-family:宋体">二、项目名称：中山市黄圃人民医院医学检验外送服务项目</span></p><p
        style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">三、中标（成交）信息</span></p><p
        style="line-height:150%"><span style="font-size: 16px;line-height:150%;font-family:宋体">1</span><span
        style="font-size:16px;line-height:150%;font-family:宋体">：供应商名称 广州华银医学检验中心有限公司 ；供应商地址 广州高新技术产业开发区科学城揽月路80号广州科技创新基地A区第3层304-306、307-319单元 ；中标（成交）金额 1267500；备注 服务期：1年 。</span>
</p><p style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">四、主要标的信息</span></p>
<table>
    <tbody>
    <tr style=";height:34px">
        <td rowspan="2" style="border: 1px solid black; padding: 1px 1px 0px;"><p
                style="text-align:center;vertical-align:middle"><span style=";font-family:   宋体;color:black">服务类</span>
        </p></td>
        <td style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style=";font-family:   宋体;color:black">序号</span>
            </p></td>
        <td style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style=";font-family:   宋体;color:black">标的名称</span>
            </p></td>
        <td style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style=";font-family:   宋体;color:black">服务范围</span>
            </p></td>
        <td style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style=";font-family:   宋体;color:black">服务要求</span>
            </p></td>
        <td style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style=";font-family:   宋体;color:black">服务时间</span>
            </p></td>
        <td style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style=";font-family:   宋体;color:black">服务标准</span>
            </p></td>
    </tr>
    <tr style=";height:51px">
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style=";font-family:宋体;color:black">&nbsp;1</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style=";font-family:宋体;color:black">&nbsp;</span><span style=";font-family:宋体;color:black">医学检验外送服务项目</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style=";font-family:宋体;color:black">&nbsp;</span><span
                    style=";font-family:宋体;color:black">详见用户需求书</span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style=";font-family:   宋体;color:black">详见用户需求书</span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style=";font-family:宋体;color:black">&nbsp;1</span><span style=";font-family:宋体;color:black">年</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style=";font-family:   宋体;color:black">按招标文件要求</span></p></td>
    </tr>
    </tbody>
</table><p style="line-height:150%"><span
        style="font-size:16px;line-height:150%;font-family:宋体">五、评审专家（单一来源采购人员）名单：</span></p><p
        style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">评审委员会总人数：5</span></p><p
        style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">随机抽取专家名单：陈念（组长）、黄柏开、苏建明、黄燕尔</span>
</p><p style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">采购人代表名单：冯广满</span></p><p
        style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">自行选定专家名单：无</span></p><p
        style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">六、代理服务收费标准及金额：</span></p>
<p style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">代理收费标准：按照招标文件约定&nbsp; 收费金额（元）：22600</span>
</p><p style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">七、公告期限</span></p><p
        style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">自本公告发布之日起1个工作日。</span></p>
<p style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">八、其他补充事宜&nbsp; </span></p>
<table>
    <tbody>
    <tr style=";height:32px">
        <td rowspan="2" style="border: 1px solid black; padding: 1px 1px 0px;"><p
                style="text-align:center;vertical-align:middle"><strong><span
                style="font-size:16px;font-family:宋体;color:black">分包号</span></strong></p></td>
        <td rowspan="2"
            style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">分包名称</span></strong></p></td>
        <td rowspan="2"
            style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">投标供应商名称</span></strong></p></td>
        <td rowspan="2"
            style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">资格性/符合性审查</span></strong></p></td>
        <td style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">价格得分</span></strong></p></td>
        <td style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">技术得分</span></strong></p></td>
        <td style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">商务得分</span></strong></p></td>
        <td rowspan="2"
            style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">最终得分</span></strong></p></td>
        <td rowspan="2"
            style="border-top: 1px solid black; border-right: 1px solid black; border-bottom: 1px solid black; border-image: initial; border-left: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">排名</span></strong></p></td>
    </tr>
    <tr style=";height:19px">
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">权重20%</span></strong></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">权重50%</span></strong></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><strong><span
                    style="font-size:16px;font-family:宋体;color:black">权重30%</span></strong></p></td>
    </tr>
    <tr style=";height:45px">
        <td rowspan="3"
            style="border-right: 1px solid black; border-bottom: 1px solid black; border-left: 1px solid black; border-image: initial; border-top: none; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">01</span></p></td>
        <td rowspan="3"
            style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style="font-size:16px;font-family:宋体;color:black">医学检验外送服务项目</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style="font-size:16px;font-family:宋体;color:black">广州华银医学检验中心有限公司</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style="font-size:16px;font-family:宋体;color:black">符合</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:15px;font-family:   宋体;color:black">20.00 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">50.00 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">30.00 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">100.00 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">1</span></p></td>
    </tr>
    <tr style=";height:45px">
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style="font-size:16px;font-family:宋体;color:black">达瑞医学检验（广州）有限公司</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style="font-size:16px;font-family:宋体;color:black">符合</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">18.84 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">9.2</span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">4.00 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">32.04 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">2</span></p></td>
    </tr>
    <tr style=";height:45px">
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style="font-size:16px;font-family:宋体;color:black">厦门麦克奥迪医学检验所有限公司</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span style="font-size:16px;font-family:宋体;color:black">符合</span>
            </p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">19.70 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">1.4</span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">0.00 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">21.10 </span></p></td>
        <td style="border-top: none; border-left: none; border-bottom: 1px solid black; border-right: 1px solid black; padding: 1px 1px 0px;">
            <p style="text-align:center;vertical-align:middle"><span
                    style="font-size:16px;font-family:   宋体;color:black">3</span></p></td>
    </tr>
    </tbody>
</table><p style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">九、凡对本次公告内容提出询问，请按以下方式联系。</span>
</p><p style="line-height:150%"><span style="font-size: 16px;line-height:150%;font-family:宋体">1.</span><span
        style="font-size:16px;line-height:150%;font-family:宋体">采购人信息</span></p><p style="line-height:150%"><span
        style="font-size:16px;line-height:150%;font-family:宋体">名称：中山市黄圃人民医院</span></p><p style="line-height:150%"><span
        style="font-size:16px;line-height:150%;font-family:宋体">地址：中山市黄圃镇龙安街32号</span></p><p style="line-height:150%">
    <span style="font-size:16px;line-height:150%;font-family:宋体">联系方式：0760-23210186</span></p><p
        style="line-height:150%"><span style="font-size: 16px;line-height:150%;font-family:宋体">2.</span><span
        style="font-size:16px;line-height:150%;font-family:宋体">采购代理机构信息</span></p><p style="line-height:150%"><span
        style="font-size:16px;line-height:150%;font-family:宋体">名称：广东海虹药通电子商务有限公司</span></p><p style="line-height:150%">
    <span style="font-size:16px;line-height:150%;font-family:宋体">地址：广东省广州市越秀区沿江中路298号中区2501、2512室</span></p><p
        style="line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">联系方式：0760-88311474</span>
</p><p style="line-height:150%"><span style="font-size: 16px;line-height:150%;font-family:宋体">3.</span><span
        style="font-size:16px;line-height:150%;font-family:宋体">项目联系方式</span></p><p style="line-height:150%"><span
        style="font-size:16px;line-height:150%;font-family:宋体">项目联系人：冯先生(采购人)</span></p><p style="line-height:150%">
    <span style="font-size:16px;line-height:150%;font-family:宋体">电话：0760-23232054</span></p><p style="line-height:150%">
    <span style="font-size:16px;line-height:150%;font-family:宋体">十、附件(无)</span></p><p
        style="text-align:right;line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">发布人：广东海虹药通电子商务有限公司</span>
</p><p style="text-align:right;line-height:150%"><span style="font-size:16px;line-height:150%;font-family:宋体">发布时间：2020年11月16日</span>
</p><p><br></p>
"""

class News(BaseModel):
    """
    ## 输入字段
    - esid： （可选）资讯的ESID。
    - html： 资讯的HTML源码。

    ## 示例
    示例请参考*Example Value*。
    """
    esid: Optional[str] = Field(None, example=example_esid)
    html: str = Field(..., example=example_html)


class Result(BaseModel):
    """
    ## 输出字段
    - result: 识别的产品名称。

    ## 示例
    参考*Example Value*。
    """
    result: List[Optional[str]] = Field(..., example=['医学检验服务外送项目'])

@app.post("/product/", tags=['main'], response_model=Result)
@log_input_output('ak', parse_json=True)
def predict(news: News):
    """输入一篇资讯的HTML，输出该资讯所包含的*产品名称*。"""
    return {'result': predictor.predict(news.html)}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=args.port)
