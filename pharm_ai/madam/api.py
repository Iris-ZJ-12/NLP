from pharm_ai.madam.predictor import MadamPredictorV1
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from typing import List
from mart.api_util.logger import Logger
from tempfile import NamedTemporaryFile
import uvicorn
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('-c', '--cuda', type=int, default=0, help='CUDA device. Default: 0.')
argparser.add_argument('-p', '--port', type=int, default=9797, help='Port. Default: 9797.')
args = argparser.parse_args()

predictor = MadamPredictorV1(args.cuda)
app_description = """
本文档提供临床医学PDF文档疾病、药品实体识别接口的说明。

各接口的**功能描述**请点击各接口展开查看，点击**Try it out**测试，调用接口需要传入的数据格式请参见**Request body**部分的**Schema**；

接口响应的数据格式参见**Reponses->Code=200**的**Description:Schema**，示例数据参考**Description: Example Value**。
"""

app = FastAPI(
    title='临床医学指南PDF文档疾病、药品实体识别',
    description=app_description,
    version='v1.0',
    openapi_tags=[{'name': 'main', 'description': '主要接口'}]
)

class MadamResult(BaseModel):
    """
    输出字段：
    - `drug`: 识别的各个**药物**词
    - `disease`: 识别的各个**疾病**词
    """
    drug: List[str] = Field(..., example=['colorectal', 'anaerobe', 'methane', 'CFTR', 'Delorme'])
    disease: List[str] = Field(..., example=['rectocele', '阻型便秘', 'anaerobe', 'colonic mucosa', 'CFTR', 'lubiprostone'])


@app.post('/',tags=['main'], response_model=MadamResult)
@Logger.log_input_output()
async def madam(file: UploadFile = File(...)):
    """
    Madam接口：
    输入参数：PDF文件（可复制粘贴，已解密）。
    """
    Logger.logger_.info('Upload file: {}', file.filename)
    content = await file.read()
    with NamedTemporaryFile('wb+', suffix='.pdf', dir='api_data') as temp_f:
        temp_f.write(content)
        result = predictor.predict(temp_f.name)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=args.port)