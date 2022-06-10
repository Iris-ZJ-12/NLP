from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
import uvicorn
from pharm_ai.label.old.JPN.if_fields_extractor.pdf_extractor import PdfExtractor
from fastapi.responses import FileResponse
from pharm_ai.util.api_util import Logger
from loguru import logger
from tempfile import NamedTemporaryFile
import os.path
from typing import Optional, List

app_description="""
本文档提供日本IF文件指定字段抽取接口的说明。

各接口的**功能描述**请点击各接口展开查看，点击**Try it out**测试，调用接口需要传入的数据格式请参见**Request body**部分的**Schema**；

接口响应的数据格式参见**Reponses->Code=200**的**Description:Schema**，示例数据参考**Description: Example Value**。
"""

app = FastAPI(title="日本IF文件指定字段的抽取", description=app_description, version='v1.0',
              openapi_tags=[
                  {
                      'name':'extract',
                      'description':'本模块内的接口提供抽取批量或单个日本IF文件字段的接口'
                  }
              ])

p = PdfExtractor()

@app.post('/extract/batch/', tags=['extract'])
async def batch_extract(file: UploadFile = File(...)):
    """本接口提供批量抽取日本IF文件指定字段的功能。

    - **使用方法**：点击**Choose File**按钮上传zip压缩包。

    - **传入文件**：传入的文件须为.zip格式的压缩包，并且压缩包内均为pdf格式的日本IF文件，PDF文件需可选择文字，且不能处于加密状态，
    才能进行识别与抽取。

    - **数据返回方式**：返回的结果以excel文件的形式呈现，点击**Download链接**进行下载。
    """
    logger.info("Upload file: {}", file.filename)
    assert file.filename.endswith('zip'), "请上传.zip压缩包"
    content = await file.read()
    file_basename = file.filename.rsplit(".", 1)[0]
    # Use TemporaryFile to remove the uploaded file automatically after extracting
    with NamedTemporaryFile('wb+', prefix=file_basename, dir='raw_data') as zip_f:
        zip_f.write(content)
        saved_xlsx_name = f'{os.path.basename(zip_f.name)}.xlsx'
        p.extract_from_zip(zip_f, saved_xlsx_name)
        logger.success('Extract result was saved to {}', saved_xlsx_name)
    download_name = f'{file_basename}_result.xlsx'
    return FileResponse(saved_xlsx_name, filename=download_name)

class ExtractResult(BaseModel):
    """**数据返回格式**：
    包含如下字段：
    - formulation：剂型
    - version：版本和日期
    - product_no：商品分类号
    - drug_category：药效分类
    - otc：处方药标志
    - inn_jp：通用名（日文）
    - inn_en：通用名（英文）
    - application：持证商
    - indication：适应症
    - approval_no：批准文号
    - geriatric_use：老年人用药
    - pregnant_use：孕产妇用药
    """
    formulation: Optional[str] = Field(..., example="錠剤（裸錠）")
    version: Optional[str] = Field(..., example="2020年5月改訂（第11版）")
    product_no: Optional[str] = Field(..., example="872149")
    drug_category: Optional[str] = Field(..., example="血圧降下剤")
    otc: Optional[str] = Field(..., example="処方箋医薬品：注意－医師等の処方箋により使用すること")
    inn_jp: Optional[List[str]] = Field(..., example=[
        "①ベンチルヒドロクロロチアジド", "②レセルピン", "③カルバゾクロム", "②Reserpine","③Carbazochrome"])
    inn_en: Optional[List[str]] = Field(..., example=["②レセルピン", "③カルバゾクロム", "①Benzylhydrochlorothiazide",
        "②Reserpine", "③Carbazochrome"])
    application: Optional[str] = Field(..., example="製造販売元：杏林製薬株式会社")
    indication: Optional[str] = Field(..., example="高血圧症（本態性、腎性等）、悪性高血圧症")
    approval_no: Optional[List[str]] = Field(..., example=["22100AMX01200000"])
    geriatric_use: Optional[str] = Field(..., example="""（「高齢者への投与」の項参照）\n(11)乳児（「小児等への投与」の項参照）\n(3)～(
    11)項は「重大な副作用」の項及び「その他の副作用」の項の代謝異常参照\n(12)交感神経切除後の患者\n［本剤の降圧作用が増強される。］\n(
    13)消化性潰瘍、潰瘍性大腸炎の既往歴のある患者\n［症状を再発させるおそれがある。］\n(14)てんかん等の痙攣性疾患及びその既往歴のある患者\n［痙攣閾値を低下させるおそれがある。］\n(
    15)気管支喘息又はアレルギー性疾患の既往歴のある患者\n［過敏症を増強させることがある。］""")
    pregnant_use: Optional[str] = Field(..., example="""（1）妊娠初期又は妊娠している可能性のある婦人には投与しないこと。\n
    ［レセルピンの動物実験（ラット）で催奇形作用が報告されている。］\n（2）妊娠末期、授乳中の婦人への投与を避けること。\n［レセルピン、ベンチルヒドロクロロチアジドは経胎盤的に胎児に移行し、また、授乳婦にあっては母乳を\n
    介して新生児に移行し、新生児に気道内分泌物の増加、鼻充血、チアノーゼ、食欲不振、高ビリルビン血症、\n血小板減少等を起こすおそれがある。また、ベンチルヒドロクロロチアジドの利尿効果に基づく血漿量減少、\n
    血液濃縮、子宮・胎盤血流量減少があらわれることがある。］""")


@app.post('/extract/single/', tags=['extract'], response_model=ExtractResult)
async def extract_pdf(file: UploadFile = File(...)):
    """本接口提供抽取单个日本IF文件指定字段的抽取功能。

    - **使用方法**：点击**Choose File**按钮上传日本IF文件。

    - **传入文件**：传入文件需为OCR过的pdf格式文件，且文件需解密方能识别与抽取。
    """
    logger.info("Upload file: {}", file.filename)
    assert file.filename.endswith('pdf'), "请上传.pdf文档"
    content = await file.read()
    file_basename = file.filename.rsplit('.',1)[0]
    with NamedTemporaryFile('wb+', prefix=file_basename, dir='raw_data') as pdf_f:
        pdf_f.write(content)
        result=p.preprocess_pdf(pdf_f.name)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=7380)
