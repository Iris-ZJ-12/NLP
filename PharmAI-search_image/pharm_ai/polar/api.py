# encoding: utf-8
'''
@author: zyl
@file: api.py
@time: 2021/7/22 上午1:37
@desc:
'''
import time
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from pharm_ai.polar.dt import PolarDT
from pharm_ai.polar.predict import PolarPredictor
from pharm_ai.util.api_util import Logger
import psutil
import pynvml

doc_text = """

- API功能：输入若干政策文件的标题,判断这些标题对应的文件是否是政策文件

- 输入举例--json format:
```
{ "titles":
    [
        "一图读懂《浙江省应急物资保障体系“十四五”规划》",
        "拟批准设置医疗机构公示",
        "2017年中国文化事业费增11%"
        ...
    ]
}

```
- 输出举例：
```
{
  "result": [
    1,1,0
  ]
}
```
- 输出解释: 有序的输出列表，输出'1'表示对应的文件是政策文件，反之，'0'为不是

## 更新日志：
- v0.0:
    - 初次迭代

"""

pynvml.nvmlInit()
init_used_m = float(psutil.virtual_memory().used) / 1024 / 1024
init_used_nm = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(4)).used/1024/1024
app = FastAPI(title="Polar政策文件标题分类", description=doc_text, version='v0')

polar = PolarPredictor()
polar.model_version = 'v0.0.0.1'
polar.args = PolarPredictor.set_model_parameter(model_version=polar.model_version)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
polar.use_cuda = False
polar.args.quantized_model = True
polar.args.onnx = True

polar.cuda_device = 4
polar.args.n_gpu = 1
polar.args.eval_batch_size = 1
model = polar.get_predict_model()


class Input(BaseModel):
    """
    输入：一些文件标题组成的列表,[标题,标题...]
    """
    titles: List[str] = Field(..., description='`List[str...]`,列表', example=[
        "一图读懂《浙江省应急物资保障体系“十四五”规划》", "拟批准设置医疗机构公示", "2017年中国文化事业费增11%"
    ])


class Output(BaseModel):
    """
    输出：每个标题对应的文件是否是政策文件,1/0
    """
    result: List = Field(...,
                         description='`List[str...]`,输出结果的列表，`1`表示是,`0`表示不是',
                         example=[1, 1, 0]
                         )


@app.post("/predict/", tags=['Predict'], response_model=Output)
@Logger.log_input_output()
async def predict(request_content: Input):
    titles = request_content.titles
    titles = [PolarDT.clean_text(i) for i in titles]
    predicted_labels = model.predict(titles)[0].tolist()

    used_m = float(psutil.virtual_memory().used) / 1024 / 1024
    print(f'used memory {used_m-init_used_m}')
    used_nm = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(4)).used/1024/1024
    print(f'used nvidia memory {used_nm - init_used_nm}')

    return {'result': predicted_labels}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3724)
