import tempfile
import uvicorn
import traceback

from fastapi import FastAPI, File, UploadFile
from mart.api_util.ecslogger import ECSLogger
from pharm_ai.papa.extractors import PapaUs, PapaCn, PapaEp, PapaJp
from pharm_ai.papa.result_processor import (
    CNProcessor,
    USProcessor,
    EPProcessor,
    JPProcessor,
    TemplateGenerator
)

description = """
接口调试：点击接口下拉箭头 **↓** 后点击 **Try it out**

#### 正常返回格式:

    {
      "country": "US",
      "filename": "xxx.pdf",
      "middle_result": {
        ...
      },
      "final_result": {
        "api_status": "normal",
        "relationship_info": [
          {
            "sub_id": "43186f0f-2c81-4084-ae5a-8944ed97696f",
            "piece": "",
            "inid_code": "",
            "keyword": "",
            "type": "",
            "application_num": "",
            "application_docdb_comb": "",
            "publication_num": "",
            "publication_docdb_comb": "",
            "date_1_str": "",
            "date_1": null,
            "date_2_str": "",
            "date_2": null
          },
          ...
        ]
      }
    }

#### 错误返回格式

    {
      "country": "US",
      "filename": "xxx.pdf",
      "middle_result": {
        ...
      },
      "final_result": {
        "api_status": "internal_error",
        "error_message": error msg,
        "relationship_info": [
            ...
        ]
      }
    }
"""

app = FastAPI(
    title='😬papa-专利PDF文档扉页信息提取',
    version='v1.0',
    description=description,
    openapi_tags=[
            {
                'name': 'us',
                'description': '美国专利扉页信息提取',
            },
            {
                'name': 'cn',
                'description': '中国专利扉页信息提取',
            },
            {
                'name': 'ep',
                'description': '欧洲专利扉页信息提取',
            },
            {
                'name': 'jp',
                'description': '日本专利扉页信息提取',
            }
        ]
)
logger = ECSLogger('papa', app)

ocr_api = 'http://localhost:1990/2txt'


extractor_us = PapaUs(ocr_api=ocr_api)
us_pro = USProcessor()

extractor_cn = PapaCn(ocr_api=ocr_api)
cn_pro = CNProcessor()

extractor_ep = PapaEp(ocr_api=ocr_api)
ep_pro = EPProcessor()

extractor_jp = PapaJp(ocr_api=ocr_api)
jp_pro = JPProcessor()


@app.post("/us", tags=['us'])
@logger.log_input_output()
async def us(file: UploadFile = File(...)):
    fn = file.filename
    contents = await file.read()
    ret = {'country': 'US', 'filename': fn}
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(contents)
        try:
            middle_result = extractor_us.pipeline(tmp.name, fn)
            ret['middle_result'] = middle_result
            final_result = us_pro.standardize_result(middle_result)
            ret['final_result'] = final_result
        except:
            if 'middle_result' not in ret:
                ret['middle_result'] = {}
                ret['final_result'] = TemplateGenerator.internal_error('unknown error')
            else:
                ret['final_result'] = TemplateGenerator.internal_error('standardization error')
            print(traceback.print_exc())
    return ret


@app.post("/cn", tags=['cn'])
@logger.log_input_output()
async def cn(file: UploadFile = File(...)):
    fn = file.filename
    contents = await file.read()
    ret = {'country': 'CN', 'filename': fn}
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(contents)
        try:
            middle_result = extractor_cn.pipeline(tmp.name, fn)
            ret['middle_result'] = middle_result
            final_result = cn_pro.standardize_result(middle_result)
            ret['final_result'] = final_result
        except:
            if 'middle_result' not in ret:
                ret['middle_result'] = {}
                ret['final_result'] = TemplateGenerator.internal_error('unknown error')
            else:
                ret['final_result'] = TemplateGenerator.internal_error('standardization error')
            print(traceback.print_exc())
    return ret


@app.post("/ep", tags=['ep'])
@logger.log_input_output()
async def ep(file: UploadFile = File(...)):
    fn = file.filename
    contents = await file.read()
    ret = {'country': 'EP', 'filename': fn}
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(contents)
        try:
            middle_result = extractor_ep.pipeline(tmp.name, fn)
            ret['middle_result'] = middle_result
            final_result = ep_pro.standardize_result(middle_result)
            ret['final_result'] = final_result
        except:
            if 'middle_result' not in ret:
                ret['middle_result'] = {}
                ret['final_result'] = TemplateGenerator.internal_error('unknown error')
            else:
                ret['final_result'] = TemplateGenerator.internal_error('standardization error')
            print(traceback.print_exc())
    return ret


@app.post("/jp", tags=['jp'])
@logger.log_input_output()
async def jp(file: UploadFile = File(...)):
    fn = file.filename
    contents = await file.read()
    ret = {'country': 'JP', 'filename': fn}
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(contents)
        try:
            middle_result = extractor_jp.pipeline(tmp.name, fn)
            ret['middle_result'] = middle_result
            final_result = jp_pro.standardize_result(middle_result)
            ret['final_result'] = final_result
        except:
            if 'middle_result' not in ret:
                ret['middle_result'] = {}
                ret['final_result'] = TemplateGenerator.internal_error('unknown error')
            else:
                ret['final_result'] = TemplateGenerator.internal_error('standardization error')
            print(traceback.print_exc())
    return ret


if __name__ == '__main__':

    uvicorn.run(app, host="0.0.0.0", port=7353)
