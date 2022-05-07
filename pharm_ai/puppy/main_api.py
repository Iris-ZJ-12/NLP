import asyncio
import time
from typing import Optional

import aiohttp
import nest_asyncio
# from mart.api_util.logger import Logger
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from puppy import Blast

nest_asyncio.apply()

app = FastAPI(
    title='生物序列比对搜索',
    description="- This API is used for aligning biological sequence with proprietary"
                "patent database and public nr/nt database. The current version have"
                "3 access APIs available.",
    version="1.0"
)

blast = Blast()


async def post(url, data):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=data.dict()) as resp:
                result = await resp.json()
        except Exception as e:
            raise e
    return result


async def post_new(session, url, data):
    async with session.post(url, json=data.dict()) as resp:
        result = await resp.json()
    return result


def merge_blast_out(result):
    for i in result[1:]:
        result[0]["result"].extend(i["result"])
    # one hit may have more than one alignment and retain all for now
    new = sorted(result[0]["result"], key=lambda x: x["hsps"][0]["evalue"])
    return new


#######################################################
################ sequence search  ###################

class SeqParas(BaseModel):
    sequence: str = Field(...,
                          example="LDPERLEVFSTVKEITGYLNIEGTHPQFRNLSYFRNLETIHGRQLMESMFAALAIVKSSL"
                                  "YSLEMXNLKQISSGSVVIQHNRDLCYVSNIRWPAIQK")
    seqtype: str = Field(..., example="protein")
    dbtype: str = Field(..., example="protein")
    dbname: str = Field(..., example="pat")
    task: str = Field(..., example="blastp")
    evalue: float = Field(..., example=0.001)
    wordsize: int = Field(..., example=6)
    gapopen: int = Field(..., example=6)
    gapextend: int = Field(..., example=2)
    matrix: Optional[str] = Field(None, example="BLOSUM62")
    penalty: Optional[int] = Field(None, example=1)
    reward: Optional[int] = Field(None, example=-3)
    ungapped: bool = False
    max_target_num: int = Field(10, example=10)


@app.post("/sequence_search/")
# @Logger.log_input_output()
async def seq_blast(paras: SeqParas):
    start = time.time()
    url1 = "http://192.168.100.210:16221/sequence_search_new/"
    url2 = "http://192.168.100.210:16222/sequence_search/"
    urls = [url1, url2]

    # zz = requests.post(url1,data=paras.json())

    # tasks = [asyncio.create_task( post(url, paras)) for url in urls]
    # tasks = [post(url, paras) for url in urls]
    # loop = asyncio.get_event_loop()
    # result = loop.run_until_complete(asyncio.gather(*tasks))

    loop = asyncio.get_event_loop()
    async with aiohttp.ClientSession(loop=loop) as session:
        result = await asyncio.gather(*[post_new(session, url, paras) for url in urls])
    new = merge_blast_out(result)

    end = time.time()
    cost = round(end - start, 4)
    return {"cost time(s)": cost, "result": new}


#######################################################
################ antibody aligment  ###################
class AntibodyParas(BaseModel):
    dbtype: str = Field(..., example="protein")
    dbname: str = Field(..., example="pat")
    cdr_eval: float = Field(..., example=1000.0)
    hl_eval: float = Field(..., example=0.001)
    cdr_wsize: int = Field(..., example=2)
    hl_wsize: int = Field(..., example=3)
    cdr_matrix: str = Field(..., example="PAM30")
    hl_matrix: str = Field(..., example="BLOSUM62")
    cdr_gp_open: int = Field(..., example=9)
    cdr_gp_extend: int = Field(..., example=1)
    hl_gp_open: int = Field(..., example=11)
    hl_gp_extend: int = Field(..., example=1)
    hl_ungap: bool = False

    hcdr1: Optional[str] = Field(None)
    hcdr2: Optional[str] = Field(None)
    hcdr3: Optional[str] = Field(None)
    lcdr1: Optional[str] = Field(None)
    lcdr2: Optional[str] = Field(None)
    lcdr3: Optional[str] = Field(None)
    light: Optional[str] = Field(None)
    heavy: Optional[str] = Field(None)
    num_descrip: int = Field(..., example=10)


@app.post("/ab_search/")
# @Logger.log_input_output()
async def antibody_blast(abparas: AntibodyParas):
    start = time.time()
    url1 = "http://192.168.100.210:16221/ab_search_new/"
    result = requests.post(url1, data=abparas.json())
    end = time.time()
    cost = round(end - start, 4)
    return {"cost time(s)": cost, "result": result.json()}


#######################################################
################ motif aligment  ###################
class MotifParas(BaseModel):
    pattern: str = Field(..., example="DAE{3}[KJWTR]X{2,3}M")
    seqtype: str = Field(..., example="protein")
    dbtype: str = Field(..., example="protein")
    dbname: str = Field(..., example="pat")
    evalue: float = Field(..., example=1000)


# for motif
@app.post("/motif_search/")
# @Logger.log_input_output()
async def motif_blast(mparas: MotifParas):
    start = time.time()
    url1 = "http://192.168.100.210:16221/motif_search_new/"
    result = requests.post(url1, data=mparas.json())
    end = time.time()
    cost = round(end - start, 4)
    return {"cost time(s)": cost, "result": result.json()}


if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=16223, reload=True)
