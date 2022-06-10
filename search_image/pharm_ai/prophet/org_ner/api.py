from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
from pharm_ai.util.api_util import Logger
from pharm_ai.prophet.org_ner.org_ner import OrgNER

with open("org_ner_describe.md", "r") as f:
    describ_text = f.read()
app = FastAPI(title="医药投融资机构实体识别",
              description=describ_text)
predictor = OrgNER(cuda_device=1)

class News(BaseModel):
    texts: List[str]

@app.post('/predict/')
@Logger.log_input_output()
async def org_ner_predict(news: News):
    result = predictor.predict_ner(news.texts, return_raw=False)
    return {'result':result}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9106)
