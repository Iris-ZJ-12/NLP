from fastapi import FastAPI, File, UploadFile
import uvicorn
from loguru import logger
from pharm_ai.util.pdf_util.extractor_v2 import Extractor
from pharm_ai.Jazz.dt import PdfProcessor
import pandas as pd
from fastapi.responses import FileResponse
from random import randint
import os
from pharm_ai.util.api_util import Logger

with open("api_description.md", 'r') as md_file:
    api_describ = md_file.read()

app = FastAPI(title="日本专利再审查段落抽取与延期年限抽取",
              description=api_describ)

e = Extractor()
p = PdfProcessor()

@app.post("/extract/")
@Logger.log_input_output()
async def extract_pdf_app(file: UploadFile = File(...)):
    logger.info("Upload file: {}", file.filename)
    content = await file.read()
    int_ = randint(0, 1000)
    if file.filename.endswith('zip'):
        if "raw_data" not in os.listdir():
            os.mkdir("raw_data")
        for cur_file in os.listdir():
            if cur_file.startswith('extract_result'):
                os.remove(cur_file)
        pdf_zip_name = "raw_data/upload_pdf_{}.zip".format(int_)
        with open(pdf_zip_name, 'wb+') as zip_f:
            zip_f.write(content)
            zip_f.seek(0)
            extract_text, extract_table = e.extract(pdf_zip_name, zip_f)
        os.remove(pdf_zip_name)
        df = pd.DataFrame({
            'pdf_name': list(extract_text.keys()),
            'pdf_content': list(extract_text.values()),
            'pdf_tables': list(extract_table.values())
        })
        result_xls = 'extract_result_{}.xlsx'.format(int_)
        p.extract(df, result_xls)
        logger.success("Result: {}", df.to_records())
        return FileResponse(result_xls, filename=result_xls)
    elif e.check_if_pdf(file.filename):
        pdf_name = "raw_data/upload_pdf_{}.pdf".format(int_)
        with open(pdf_name, 'wb+') as pdf_f:
            pdf_f.write(content)
        result = p.extract_single_pdf(pdf_name)
        logger.success("Result: {}", result)
        os.remove(pdf_name)
        return result

if __name__=='__main__':
    uvicorn.run(app, host="0.0.0.0", port=9104)