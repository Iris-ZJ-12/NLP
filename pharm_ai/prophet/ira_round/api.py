# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from loguru import logger
from pharm_ai.util.api_util import Logger
from pharm_ai.prophet.ira_round.ira_round import IraRound
app = FastAPI(
    title='Prophet/ira_round 😁',
    description="""
从段落里识别融资轮次

接收list of texts, 每个text是一篇资讯中的一段

- 输入格式:
```
{
  "texts": [
    "In a new investment round, Visiopharm added BankInvest to the group of institutional investors that already includes ATP, NorthCap, Vækstfonden, and C.L. Davids Fond. BankInvest now holds almost 6% of the equity of the company. All existing investors co-invested in this round.", 
    "元禾原点是苏州元禾控股股份有限公司成员企业，是市场化运作的专业化早期股权投资平台，重点关注TMT和Healthcare两大领域内初创期和成长期创业企业的投资机会，旗下六支基金总规模约34亿元人民币，既有VC基金，也有针对北上广深苏杭六大区域的天使基金。元禾原点在TMT和Healthcare领域有专业的投资团队、丰富的投资经验和资源积累。"
  ]
}
```
- 输出格式：
```
{'result': ['A轮', '']}
```
    """,
    version="1"
)

p = IraRound()


class Args(BaseModel):
    paragraphs: list
    titles: list


@app.post("/")
@Logger.log_input_output()
async def predict(args: Args):
    try:
        res = p.predict_api(args.paragraphs, args.titles)
        print(res)
    except Exception as e:
        logger.error(e)
        logger.error(args.paragraphs)
        logger.error(args.titles)
        res = ['error']
    return {'result': res}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6950)