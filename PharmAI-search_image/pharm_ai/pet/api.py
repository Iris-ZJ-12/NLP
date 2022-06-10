from fastapi import FastAPI, File, UploadFile
import uvicorn
from loguru import logger
from pharm_ai.pet.extractor import Extractor
from io import BytesIO
from fastapi.responses import FileResponse
from datetime import datetime
from pharm_ai.util.api_util import Logger


app = FastAPI(title='pet-PDF表格内容抽取存成*.xlsx且提供下载链接🌿', version='1')

e = Extractor()


@app.post("/")
@Logger.log_input_output()
async def extract_pdf_tables(file: UploadFile = File(...)):
    fn = file.filename
    print(fn)
    excel = fn[:-4] + '.xlsx'
    dt_string = datetime.now().strftime("%d-%m-%Y-H-%M-%S")
    excel_path = f'{e.d}{dt_string}-{excel}'
    logger.info(f'Tables were saved into {excel}.')
    e.extract_tables(file.file, fn, excel_path)
    response = FileResponse(excel_path, filename=excel)
    return response


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=7963)
