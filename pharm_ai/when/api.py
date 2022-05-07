from fastapi import FastAPI, File, UploadFile
import asyncio
import uvicorn
import functools
from loguru import logger
from pydantic import BaseModel
import os
import sys
import io
from INN import get_left_names, ocr_get_left_names

app = FastAPI(title='WHO-INN🚎', description="""
## v1: WHO 药品INN PDF中通用名称的提取
---
- inn: 可复制粘体的PDF通用名称的提取
- inn_ocr：不可复制粘体的PDF通用名称的提取（使用OCR技术）
### 
###         ©PharmAI
""", version="version-1.0")


class Logger:
    if not os.path.exists("Logs"):
        os.makedirs("Logs")
    logger.remove()
    logger_ = logger.opt(depth=1)
    default_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> || <level>{level: <8}</level> || <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> || <level>{message}</level>'
    logger_.add("./Logs/result_{time}.log", rotation="100 MB", compression="tar",
                enqueue=True, format=default_format)
    logger_.add(sys.stdout)

    @classmethod
    def log_input_output(cls):
        def wrapper(func):

            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                cls.logger_.opt(depth=1)
                try:
                    cls.logger_.info("Excecuting function '{}'", func.__name__)
                    for k, v in kwargs.items():
                        if isinstance(v, BaseModel):
                            cls.logger_.info("Input data: {}", dict(v))
                    if asyncio.iscoroutinefunction(func):
                        result = asyncio.run(func(*args, **kwargs))
                    else:
                        result = func(*args, **kwargs)
                    cls.logger_.success("Output :{}", result)
                    return result
                except:
                    cls.logger_.exception("Error occured while excecuting funciton '{}'", func.__name__)
            return wrapped
        return wrapper


@app.post("/inn")
@Logger.log_input_output()
async def upload(file: UploadFile = File(...)):
    res = get_left_names(io.BytesIO(await file.read()))
    return res


@app.post("/inn_ocr")
@Logger.log_input_output()
async def upload(file: UploadFile = File(...)):
    res = ocr_get_left_names(io.BytesIO(await file.read()))
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1028)
