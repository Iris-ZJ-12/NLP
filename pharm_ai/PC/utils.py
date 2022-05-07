from pharm_ai.PC.dt import GenerativeDataProcessor
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from loguru import logger
from time import time
import requests
import aiohttp
import asyncio

random_state=610
np.random.seed(random_state)

def test_api_speed(host='gpu176', port=21310, sample_patent=300, sample_person=1000, async_request=False,
                   batch_number=None, num_per_request=1, timeout=None, limit=None):
    to_pred_claim, to_perd_person = load_api_test_data(sample_patent, sample_person)
    if batch_number:
        to_pred_claim_batch = [{'patent_input':t} for t in chunk(to_pred_claim, batch_number)]
        to_pred_person_batch = [{'classification_input':t} for t in chunk(to_perd_person, batch_number)]
    else:
        to_pred_claim_batch, to_pred_person_batch = [{'patent_input': to_pred_claim}], [{'classification_input':to_perd_person}]
    address_patent, address_person = f"http://{host}:{port}/", f"http://{host}:{port}/classification/"
    for item, to_pred, address, num in zip(['patent', 'person'],
                                           [to_pred_claim_batch, to_pred_person_batch],
                                           [address_patent, address_person],
                                           [sample_patent, sample_person]):
        logger.info('Start test {}', item)
        t1 = time()
        if async_request:
            logger.info('batch_size={}, num_per_request={}.', len(list(to_pred[0].values())[0]), num_per_request)
        else:
            logger.info('batch_size={}', len(list(to_pred[0].values())[0]))
        result = []
        tlogger=logger.add(lambda msg: tqdm.write(msg))
        pbar = tqdm(to_pred)
        for i_, batch in enumerate(pbar):
            pbar.set_description(f'Start {i_ + 1} request batch')
            if async_request:
                cur_res = asyncio.run(get_from_api_main(batch, num_per_request, address, timeout, limit))
                result.append(cur_res)
            else:
                response = requests.post(address, json=batch)
                result.append(json.loads(response.text))
        t2 = time()
        logger.remove(tlogger)
        if item=='patent':
            patent_sents = sum(len(p[1]) for p in to_pred_claim.items())
            logger.info('Request {} {} data ({} sentences) using {}s={}min', num, item, patent_sents, t2-t1, (t2-t1)/60)
        else:
            logger.info('Request {} {} data using {}s={}min', num, item, t2-t1, (t2-t1)/60)
    return result

def load_api_test_data(sample_patent=300, sample_person=1000):
    json_file = Path('results')/'test_api_data.json'
    if json_file.exists():
        with open(json_file, 'r') as f:
            dt, res_person = json.load(f)
    else:
        load_json_file = Path('results')/'to_pred_large_batch1.json'
        with open(load_json_file, 'r') as f:
            dt = json.load(f)

        p=GenerativeDataProcessor('v4.0')
        df = p.get_from_h5()
        res_person = df[df['prefix'] == 'person']['input_text'].tolist()

        res = (dt, res_person)
        with open(json_file, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        logger.info('{} claim and {} person test data saved to {}', len(dt), len(res_person), json_file.as_posix())
    sel_ids=np.random.choice(np.arange(len(dt)), sample_patent, replace=False)
    res_claim = dict(itm for i_, itm in enumerate(dt.items()) if i_ in sel_ids)
    res_person = np.random.choice(res_person, sample_person, replace=False).tolist()

    return (res_claim, res_person)

async def get_from_api_main(to_predicts, num_per_request, url, timeout=None, limit=None):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout),
                                     connector=aiohttp.TCPConnector(limit=limit)) as session:
        task_list = []
        key_ = list(to_predicts.keys())[0]
        batches = [{key_:v} for v in chunk(to_predicts[key_], num_per_request)]
        for to_predict in batches:
            req = get_from_api_async(session, to_predict, url)
            task = asyncio.create_task(req)
            task_list.append(task)
        results = await asyncio.gather(*task_list)
    return results

async def get_from_api_async(client, to_predict, url):
    async with client.post(url, json=to_predict) as response:
        text_ = await response.text()
        result = json.loads(text_)
    return result

def chunk(d, size):
    res_ls = []
    if isinstance(d,dict):
        for i,(k,v) in enumerate(d.items()):
            res_ls.append((k, v))
            if (i+1) % size==0 or i==len(d)-1:
                yield dict(res_ls)
                res_ls = []
    elif isinstance(d, list):
        for i,l in enumerate(d):
            res_ls.append(l)
            if (i+1) % size==0 or i==len(d)-1:
                yield res_ls
                res_ls=[]

if __name__ == '__main__':
    test_api_speed(host='gpu176', sample_patent=300, sample_person=500, async_request=True, batch_number=6)