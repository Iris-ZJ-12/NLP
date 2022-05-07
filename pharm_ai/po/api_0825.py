# encoding: utf-8
'''
@author: zyl
@file: api_0825.py
@time: 2021/8/25 16:14
@desc:
'''

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from pharm_ai.po.predict import V4Predictor
from pharm_ai.util.api_util import Logger
from simpletransformers.classification import ClassificationArgs

app = FastAPI(title="Pubmed临床结果论文识别V4")

p = V4Predictor()
p.model_version = 'v4.2.0.4'
p.args = p.set_model_parameter(model_version=p.model_version, args=ClassificationArgs(),
                               save_dir='po')
p.use_cuda = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
p.cuda_device = 2
p.args.n_gpu = 1
p.args.eval_batch_size = 5
p.args.max_seq_length = 512
model = p.get_predict_model()


class Item(BaseModel):
    texts: list


@app.post("/predict/")
@Logger.log_input_output()
async def predict(request_content: Item):
    if request_content.texts == [] or request_content.texts == [[]]:
        return {'result': []}
    try:
        to_predict_texts = p.api_clean_dt(request_content.texts)
        predicted_labels = model.predict(to_predict_texts)[0]
        predicted_labels = predicted_labels.tolist()
    except Exception as e:
        print(str(e))
        print('there is an error,call me')
        predicted_labels = [0] * len(request_content.texts)
    return {'result': predicted_labels}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6007)
