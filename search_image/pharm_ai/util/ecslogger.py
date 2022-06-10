import time
import functools
import logging
import ecs_logging
import asyncio
import json

from pydantic import BaseModel


logger = logging.getLogger("ecs")
logger.setLevel(logging.INFO)


class FormatterWrapper(ecs_logging.StdlibFormatter):
    def format(self, record):
        return json.dumps(json.loads(super().format(record)), ensure_ascii=False)


# Add an ECS formatter to the Handler
handler = logging.handlers.RotatingFileHandler(
    '/home/public/disk/logs/test.log',
    maxBytes=1_000_000_000,
    backupCount=10,
    encoding='utf-8'
)
handler.setFormatter(FormatterWrapper(exclude_fields=["log.original", "log.logger", "log.origin"]))
logger.addHandler(logging.StreamHandler())
logger.addHandler(handler)


def log_input_output(project):
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            now = time.time()
            # always add input information, only BaseModel is supported currently
            info = {}
            if len(kwargs) == 1:
                inputs = list(kwargs.values())[0]
                if isinstance(inputs, BaseModel):
                    info["Input data"] = dict(inputs)
                    for k_, v_ in info["Input data"].items():
                        if type(v_) == str and len(v_) > 10000:
                            info["Input data"][k_] = f'String too long({len(v_)}), emitted.'

            try:
                if asyncio.iscoroutinefunction(func):
                    result = asyncio.run(func(*args, **kwargs))
                else:
                    result = func(*args, **kwargs)
                info["Output"] = dict(result)
                return result
            finally:
                elapsed_time = time.time() - now
                extras = {
                    "project": project,
                    "elapsed_time": elapsed_time
                }
                logger.exception(json.dumps(info, ensure_ascii=False), extra=extras)
        return wrapped
    return wrapper
