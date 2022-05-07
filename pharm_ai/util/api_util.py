import functools
from loguru import logger
from pydantic import BaseModel
import asyncio
import sys
from time import time
import requests

class Logger:
    logger.remove()
    logger_ = logger.opt(depth=1)
    default_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> || <level>{level: <8}</level> || <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> || <level>{message}</level>'
    logger_.add("result_{time}.log", rotation="100 MB", compression="tar",
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
                    for k,v in kwargs.items():
                        if isinstance(v, BaseModel):
                            cls.logger_.info("Input data: {}", dict(v))
                        else:
                            cls.logger_.info("Input data: {}", v)
                    result = func(*args, **kwargs)
                    cls.logger_.success("Output :{}", result)
                    return result
                except:
                    cls.logger_.exception("Error occured while excecuting funciton '{}'", func.__name__)

            @functools.wraps(func)
            async def wrapped_async(*args, **kwargs):
                cls.logger_.opt(depth=1)
                try:
                    cls.logger_.info("Excecuting function '{}'", func.__name__)
                    for k, v in kwargs.items():
                        if isinstance(v, BaseModel):
                            cls.logger_.info("Input data: {}", dict(v))
                        else:
                            cls.logger_.info("Input data: {}", v)
                    result = await func(*args, **kwargs)
                    cls.logger_.success("Output :{}", result)
                    return result
                except:
                    cls.logger_.exception("Error occured while excecuting funciton '{}'", func.__name__)
            if asyncio.iscoroutinefunction(func):
                return wrapped_async
            else:
                return wrapped
        return wrapper

def get_time_spent(url, data):
    """
    :param url:[str] API url.
    :param data: [dict] JSON data
    :return: Time spent totally (unit: s).
    """
    t1 = time()
    response = requests.post(url, json=data)
    print(response.text)
    t2 = time()
    print('total request time(network latency & function call): ' + str((t2 - t1) / 60) + 'minutes.')
    return t2-t1
