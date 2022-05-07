import uvicorn

from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pharm_ai.pom.folder import Folder

app = FastAPI(
    title="👶🏽媒体文章nlp处理系统\n折叠接口",
    version="v1.0",
    description="""
本接口折叠结果直接更新在对应的ES索引中，返回结果为涉及到更新的折叠ID。

若请求文章已经被处理过，返回格式为
{
    'status': 'ignored'
}

其他情况下返回
{
    'status': 'success',
    'clusterid': [...]
}
    """,
    openapi_tags=[
        {
            'name': 'fold',
            'description': '折叠接口，返回更新的折叠ID',
        }
    ]
)

folder = Folder()


class FoldInput(BaseModel):
    title: str
    id: str
    publish_time: int


class FoldOutput(BaseModel):
    status: str
    clusterid: Optional[List[str]]


@app.post("/fold/", tags=['fold'], response_model=FoldOutput, response_model_exclude_unset=True)
async def fold(news: FoldInput):
    news = news.dict()
    ret = folder(news, threshold=0.6)
    return ret

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6039)
