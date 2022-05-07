import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from puppy import Blast

# from mart.api_util.logger import Logger
# from mart.api_util.ecslogger import ECSLogger

app = FastAPI(
    title='生物序列比对搜索',
    description="## This API is used for aligning biological sequence with proprietary"
                "patent database and public nr/nt database. The current version have"
                "3 access APIs available.",
    version="1.0"
)

blast = Blast()


####################################################################
######################## sequence search  ########################

class SeqParas(BaseModel):
    sequence: str = Field(...,
                          example="LDPERLEVFSTVKEITGYLNIEGTHPQFRNLSYFRNLETIHGRQLMESMFAALAIVKSSLYSLEMXNLKQISSGSVVIQHNRDLCYVSNIRWPAIQK")
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
    max_target_num: int = Field(100, example=100)


@app.post("/sequence_search_new/")
# @Logger.log_input_output()
async def seq_blast(paras: SeqParas):
    print("----------------")
    start = time.time()
    print("###", paras)
    result = blast.sequence_search(paras.sequence, paras.seqtype, paras.dbtype, paras.dbname, paras.task, paras.evalue,
                                   paras.wordsize, paras.gapopen, paras.gapextend, paras.matrix, paras.penalty,
                                   paras.reward, paras.ungapped, paras.max_target_num)
    end = time.time()
    cost = end - start
    return {"cost time": cost, "result": result}


@app.post("/sequence_search/")
# @Logger.log_input_output()
async def seq_blast(
        sequence: str = "LDPERLEVFSTVKEITGYLNIEGTHPQFRNLSYFRNLETIHGRQLMESMFAALAIVKSSLYSLEMXNLKQISSGSVVIQHNRDLCYVSNIRWPAIQK", \
        seqtype: str = Query("protein", enum=["protein", "nucleotide"]), \
        dbtype: str = Query("protein", enum=["protein", "nucleotide"]), \
        dbname: str = Query("pat", enum=["pat", "nr", "pat,nr", "nt", "pat,nt"]), \
        task: str = Query("blastp",
                          enum=["blastp", "blastp-short", "blastp-fast", "blastn", "blastn-short", "megablast",
                                "blastx", "blastx-fast", "tblastn", "tblastn-fast"]), \
        evalue: float = 0.001, \
        wordsize: int = 6, \
        gapopen: int = 6, \
        gapextend: int = 2, \
        matrix: Optional[str] = Query("BLOSUM62", enum=["BLOSUM62", "PAM30"]), \
        penalty: Optional[int] = 1, \
        reward: Optional[int] = -3, \
        ungapped: bool = False, \
        max_target_num: Optional[int] = 10):
    start = time.time()
    result = blast.sequence_search(sequence, seqtype, dbtype, dbname, task, evalue, wordsize, gapopen, gapextend,
                                   matrix, penalty, reward, ungapped, max_target_num)
    end = time.time()
    cost = round(end - start, 4)
    return {"cost time(s)": cost, "result": result}


####################################################################
######################## antibody aligment  ########################
@app.post("/ab_search/")
# @Logger.log_input_output()
async def antibody_blast(dbtype: str = "protein", \
                         dbname: str = Query("pat", enum=["pat", "nr", "pat,nr"]), \
                         cdr_eval: str = "1000", \
                         hl_eval: str = "0.001", \
                         cdr_wsize: int = 2, \
                         hl_wsize: int = 3, \
                         cdr_matrix: str = "PAM30", \
                         hl_matrix: str = "BLOSUM62", \
                         cdr_gp_open: int = 9, \
                         cdr_gp_extend: int = 1, \
                         hl_gp_open: int = 11, \
                         hl_gp_extend: int = 1, \
                         hl_ungap: bool = False, \
                         hcdr1: Optional[str] = None, hcdr2: Optional[str] = None, hcdr3: Optional[str] = None, \
                         lcdr1: Optional[str] = None, lcdr2: Optional[str] = None, lcdr3: Optional[str] = None, \
                         light: Optional[str] = None, heavy: Optional[str] = None, \
                         num_descrip: Optional[int] = 10):  ## input clearly
    start = time.time()
    result = blast.cdr_blast(dbtype, dbname, cdr_eval, hl_eval, cdr_wsize, hl_wsize, cdr_matrix, hl_matrix, \
                             cdr_gp_open, cdr_gp_extend, hl_gp_open, hl_gp_extend, hl_ungap, \
                             hcdr1, hcdr2, hcdr3, lcdr1, lcdr2, lcdr3, light, heavy, num_descrip)
    end = time.time()
    cost = round(end - start, 4)
    return {"cost time(s)": cost, "result": result}


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


@app.post("/ab_search_new/")
# @Logger.log_input_output()
async def antibody_blast(ab: AntibodyParas):  ## input clearly
    start = time.time()
    result = blast.cdr_blast(ab.dbtype, ab.dbname, ab.cdr_eval, ab.hl_eval, ab.cdr_wsize, ab.hl_wsize,
                             ab.cdr_matrix, ab.hl_matrix, ab.cdr_gp_open, ab.cdr_gp_extend, ab.hl_gp_open,
                             ab.hl_gp_extend, ab.hl_ungap, ab.hcdr1, ab.hcdr2, ab.hcdr3, ab.lcdr1,
                             ab.lcdr2, ab.lcdr3, ab.light, ab.heavy, ab.num_descrip)
    end = time.time()
    cost = round(end - start, 4)
    return {"cost time(s)": cost, "result": result}


####################################################################
######################## antibody aligment  ########################
@app.post("/motif_search/")
# @Logger.log_input_output()
async def motif_blast(pattern: str = "DAE{3}[KJWTR]X{2,3}M", \
                      seqtype: str = Query("protein", enum=["protein", "nucleotide"]), \
                      dbtype: str = Query("protein", enum=["protein", "nucleotide"]), \
                      dbname: str = Query("pat", enum=["pat", "nr", "pat,nr", "nt", "pat,nt"]), \
                      evalue: str = "1000"):
    start = time.time()
    result = blast.motif_blast(pattern, seqtype, dbtype, dbname, evalue)
    end = time.time()
    cost = round(end - start, 4)
    return {"cost time(s)": cost, "result": result}


class MotifParas(BaseModel):
    pattern: str = Field(..., example="DAE{3}[KJWTR]X{2,3}M")
    seqtype: str = Field(..., example="protein")
    dbtype: str = Field(..., example="protein")
    dbname: str = Field(..., example="pat")
    evalue: float = Field(..., example=1000)


@app.post("/motif_search_new/")
# @Logger.log_input_output()
async def motif_blast(mpa: MotifParas):
    start = time.time()
    result = blast.motif_blast(mpa.pattern, mpa.seqtype, mpa.dbtype, mpa.dbname, mpa.evalue)
    end = time.time()
    cost = round(end - start, 4)
    return {"cost time(s)": cost, "result": result}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=16221, reload=True)
    # uvicorn.run("api:app", host="0.0.0.0",  reload=False)
