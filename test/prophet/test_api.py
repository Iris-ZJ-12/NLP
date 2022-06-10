import asyncio
import json
import time
import sys
from pathlib import Path

import aiohttp
import pytest
from loguru import logger

from mart.api_util.util import get_from_api
from pharm_ai.prophet.utils import delete_bulk, get_es, index_bulk, update_bulk, int2date

logger.add(sys.stderr, level='DEBUG')

class TestAPI:
    data_dir = Path(__file__).parent/'data'
    prophet_url = "http://localhost:15058/prophet/"
    folding_url = "http://localhost:15058/news_folding/"
    news_filter_url = "http://localhost:15058/news_filter/"
    ira_url = "http://localhost:15058/ira/"
    org_filter_url = "http://localhost:15058/org_filter/"
    org_ner_url = "http://localhost:15058/org_ner/"
    test_es = get_es('test')
    filebeat_es = get_es('176')
    filebeat_hostname = 'c46d161c7273'
    filebeat_index = 'pharm-prophet-7.13.2'

    def load_prophet_data(self, data_file):
        load_json_path = self.data_dir / (data_file + '.json')
        with load_json_path.open('r') as f:
            to_predict = json.load(f)
        return {'input_dic': to_predict}

    @pytest.mark.parametrize(
        'data_file',
        [
            'data1',
            'data5'
        ]
    )
    def test_prophet_api(self, data_file):
        data_file = 'data1'
        to_predict = self.load_prophet_data(data_file)
        result, status_code = get_from_api(to_predict, self.prophet_url)
        assert status_code == 200
        assert bool(result)

    @pytest.mark.parametrize(
        'data_file',
        [
            'data5',
            'data6'
        ]
    )
    def test_once_request_prophet(self, data_file):
        loaded_data = self.load_prophet_data(data_file)
        to_predicts = [{'input_dic': {esid: loaded_data['input_dic'][esid]}}
                       for esid in loaded_data['input_dic']]
        results, status_code = [], []
        for to_predict in to_predicts:
            res, status = get_from_api(to_predict, self.prophet_url)
            results.append(res)
            status_code.append(status)
        n_success = sum(status == 200 for status in status_code)
        success_ratio = n_success / len(results) * 100
        logger.info('Success requests: {}/{}={}%', n_success,
                    len(results), success_ratio)
        assert success_ratio == 100
        assert all(k in ['final_output', 'intermediary_output'] for res in results for k in res)

    def test_batch_request_prophet(self):
        """Simulate concurrent request."""
        data_file = 'data5'
        loaded_data = self.load_prophet_data(data_file)
        to_predicts = [{'input_dic': {esid: loaded_data['input_dic'][esid]}}
                       for esid in loaded_data['input_dic']]
        raw_results = asyncio.run(self.prophet_api_main(to_predicts))
        results, status_code = [], []
        for res, status in raw_results:
            results.append(res)
            status_code.append(status)
        assert len(results) == len(to_predicts)
        n_success = sum(status == 200 for status in status_code)
        success_ratio = n_success / len(results) * 100
        logger.info('Success requests: {}/{}={}%', n_success,
                    len(results), success_ratio)
        assert success_ratio == 100
        assert all(k in ['final_output', 'intermediary_output'] for res in results for k in res)

    def test_batch_request_prophet_case_rep(self):
        data_file = 'data7'
        loaded_data = self.load_prophet_data(data_file)
        to_predicts = [loaded_data] * 500
        raw_results = asyncio.run(self.prophet_api_main(to_predicts))
        results, status_code = [], []
        for res, status in raw_results:
            results.append(res)
            status_code.append(status)
        assert len(results) == len(to_predicts)
        n_success = sum(status == 200 for status in status_code)
        success_ratio = n_success / len(results) * 100
        logger.info('Success requests: {}/{}={}%', n_success,
                    len(results), success_ratio)
        assert success_ratio == 100
        assert all(k in ['final_output', 'intermediary_output'] for res in results for k in res)

    async def prophet_api_main(self, to_predicts, timeout=None, limit=30):
        session_timeout = aiohttp.ClientTimeout(total=timeout)
        conn = aiohttp.TCPConnector(limit=limit)
        async with aiohttp.ClientSession(timeout=session_timeout, connector=conn) as session:
            task_list = [asyncio.create_task(self.request_prophet_api_async(session, to_predict))
                         for to_predict in to_predicts]
            results = await asyncio.gather(*task_list)
        return results

    async def request_prophet_api_async(self, client: aiohttp.ClientSession, to_predict):
        async with client.post(self.prophet_url, json=to_predict) as response:
            text = await response.text()
        return json.loads(text), response.status

    def test_folding_api(self):
        example = {
            'article_id': '23b2714b68100b3354ad2df3cd32fad3',
            'publish_date': '2021-02-22',
            'source': '光速创投',
            'investee': '黑湖智造',
            'round': '近5亿元人民币',
            'amount': 'C轮',
            'label': '医药',
            'request_start_time': 1619595258104
        }
        result, status_code = get_from_api(example, self.folding_url)
        assert status_code == 200

    def test_news_filter(self):
        example = {
            'title': "36氪首发 | 瞄准创新肿瘤药物，「科赛睿生物」获得近3000万美元B轮融资",
            'paragraphs': [
                "36氪获悉，创新肿瘤靶向药物研发商「科赛睿生物」，宣布获得近3000万美元B轮融资。"
                "本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，"
                "老股东君联资本，联想之星持续加持，易凯资本在本次交易中担任科赛睿生物的独家财务顾问。"
                "据悉，本轮融资所得将主要用于加速推进其多个肿瘤靶向创新药在全球范围的临床开发。",
                "纵观新药研发领域，肿瘤靶向药物是目前最活跃的方向之一，其中较为著名的包括PD-1/PD-L1、VEGF、HER2、TNF-α等。"
                "根据CFDA和CDE网站上的数据显示，目前中国正在申报临床、正在进行临床和正在申报生产的小分子靶向药物约有百种，"
                "大分子靶向药数十种，其中包括国内原研、仿制药和国外已上市的品种。",
                "在这个赛道上，科赛睿生物的竞争优势在于其自主研发的i-CR®技术平台，结合条件性重编程原代肿瘤细胞培养技术和高内涵药物筛选体系。",
                "其优势是能够在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，"
                "同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。",
                "目前，科赛睿已经通过与国内头部的肿瘤医学中心开展合作，进行多项前瞻性比对临床试验，"
                "已初步证明 i-CR®体系可以有效预测药物的的实际临床反应。",
                "研发管线的进度，是考察创新药物企业的重要环节。"
                "这一方面，科赛睿已经在合成致死和免疫治疗领域开发出针对癌症的一系列新药产品线，前期研发成果已经申请了多项国际和国内专利。",
                "其中，核心产品PC-002是一个针对MYC基因变异肿瘤的first-in-class小分子药物。"
                "MYC蛋白在超过50%的肿瘤中高表达，为最重要的“不可成药”肿瘤靶点之一，PC-002通过独特的MOA靶向MYC蛋白降解，"
                "选择性诱导MYC依赖的肿瘤细胞凋亡，即将开展美国临床2期。",
                "另外一个管线产品CTB-02针对pan-KRAS变异的肠癌和非小细胞肺癌，预期在2021年首先在澳大利亚进入1期临床。",
                "图片来源于前瞻研究院",
                "在快速增长的创新药市场中，全球化竞争加剧，因此产品管线研发的效率以及创新性显得尤其重要。",
                "靶向药物的特点是针对特定靶点产生作用，每个病人的病情不尽相同，适用的药物也各有不同，因此可以对肿瘤进行精准治疗。",
                "根据IMS的数据，预计目前全球肿瘤药物市场规模可达1500亿美元，肿瘤处方药销售额约为1100亿美元。"
                "援引前瞻研究院的分析，预计单克隆抗体类药物和小分子靶向药物在未来将占据最大的市场份额。"
            ]
        }
        result, status_code = get_from_api(example, self.news_filter_url)
        assert status_code == 200

    def test_ira(self):
        example = {
            'input': "36氪获悉，创新肿瘤靶向药物研发商「科赛睿生物」，宣布获得近3000万美元B轮融资。"
                     "本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，"
                     "老股东君联资本，联想之星持续加持，易凯资本在本次交易中担任科赛睿生物的独家财务顾问。"
                     "据悉，本轮融资所得将主要用于加速推进其多个肿瘤靶向创新药在全球范围的临床开发。"
        }
        result, status_code = get_from_api(example, self.ira_url)
        assert status_code == 200

    def test_org_filter(self):
        example = {
            'paragraphs': [
                "36氪获悉，创新肿瘤靶向药物研发商「科赛睿生物」，宣布获得近3000万美元B轮融资。"
                "本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，"
                "老股东君联资本，联想之星持续加持，易凯资本在本次交易中担任科赛睿生物的独家财务顾问。"
                "据悉，本轮融资所得将主要用于加速推进其多个肿瘤靶向创新药在全球范围的临床开发。",
                "纵观新药研发领域，肿瘤靶向药物是目前最活跃的方向之一，其中较为著名的包括PD-1/PD-L1、VEGF、HER2、TNF-α等。"
                "根据CFDA和CDE网站上的数据显示，目前中国正在申报临床、正在进行临床和正在申报生产的小分子靶向药物约有百种，"
                "大分子靶向药数十种，其中包括国内原研、仿制药和国外已上市的品种。",
                "在这个赛道上，科赛睿生物的竞争优势在于其自主研发的i-CR®技术平台，结合条件性重编程原代肿瘤细胞培养技术和高内涵药物筛选体系。",
                "其优势是能够在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，"
                "同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。",
                "目前，科赛睿已经通过与国内头部的肿瘤医学中心开展合作，进行多项前瞻性比对临床试验，"
                "已初步证明 i-CR®体系可以有效预测药物的的实际临床反应。",
                "研发管线的进度，是考察创新药物企业的重要环节。"
                "这一方面，科赛睿已经在合成致死和免疫治疗领域开发出针对癌症的一系列新药产品线，前期研发成果已经申请了多项国际和国内专利。",
                "其中，核心产品PC-002是一个针对MYC基因变异肿瘤的first-in-class小分子药物。"
                "MYC蛋白在超过50%的肿瘤中高表达，为最重要的“不可成药”肿瘤靶点之一，PC-002通过独特的MOA靶向MYC蛋白降解，"
                "选择性诱导MYC依赖的肿瘤细胞凋亡，即将开展美国临床2期。",
                "另外一个管线产品CTB-02针对pan-KRAS变异的肠癌和非小细胞肺癌，预期在2021年首先在澳大利亚进入1期临床。",
                "图片来源于前瞻研究院",
                "在快速增长的创新药市场中，全球化竞争加剧，因此产品管线研发的效率以及创新性显得尤其重要。",
                "靶向药物的特点是针对特定靶点产生作用，每个病人的病情不尽相同，适用的药物也各有不同，因此可以对肿瘤进行精准治疗。",
                "根据IMS的数据，预计目前全球肿瘤药物市场规模可达1500亿美元，肿瘤处方药销售额约为1100亿美元。"
                "援引前瞻研究院的分析，预计单克隆抗体类药物和小分子靶向药物在未来将占据最大的市场份额。"
            ]
        }
        result, status_code = get_from_api(example, self.org_filter_url)
        assert status_code == 200

    def test_org_ner(self):
        example = {
            'paragraphs': [
                "36氪获悉，创新肿瘤靶向药物研发商「科赛睿生物」，宣布获得近3000万美元B轮融资。"
                "本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，"
                "老股东君联资本，联想之星持续加持，易凯资本在本次交易中担任科赛睿生物的独家财务顾问。"
                "据悉，本轮融资所得将主要用于加速推进其多个肿瘤靶向创新药在全球范围的临床开发。",
                "纵观新药研发领域，肿瘤靶向药物是目前最活跃的方向之一，其中较为著名的包括PD-1/PD-L1、VEGF、HER2、TNF-α等。"
                "根据CFDA和CDE网站上的数据显示，目前中国正在申报临床、正在进行临床和正在申报生产的小分子靶向药物约有百种，"
                "大分子靶向药数十种，其中包括国内原研、仿制药和国外已上市的品种。",
                "在这个赛道上，科赛睿生物的竞争优势在于其自主研发的i-CR®技术平台，结合条件性重编程原代肿瘤细胞培养技术和高内涵药物筛选体系。",
                "其优势是能够在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，"
                "同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。",
                "目前，科赛睿已经通过与国内头部的肿瘤医学中心开展合作，进行多项前瞻性比对临床试验，"
                "已初步证明 i-CR®体系可以有效预测药物的的实际临床反应。",
                "研发管线的进度，是考察创新药物企业的重要环节。"
                "这一方面，科赛睿已经在合成致死和免疫治疗领域开发出针对癌症的一系列新药产品线，前期研发成果已经申请了多项国际和国内专利。",
                "其中，核心产品PC-002是一个针对MYC基因变异肿瘤的first-in-class小分子药物。"
                "MYC蛋白在超过50%的肿瘤中高表达，为最重要的“不可成药”肿瘤靶点之一，PC-002通过独特的MOA靶向MYC蛋白降解，"
                "选择性诱导MYC依赖的肿瘤细胞凋亡，即将开展美国临床2期。",
                "另外一个管线产品CTB-02针对pan-KRAS变异的肠癌和非小细胞肺癌，预期在2021年首先在澳大利亚进入1期临床。",
                "图片来源于前瞻研究院",
                "在快速增长的创新药市场中，全球化竞争加剧，因此产品管线研发的效率以及创新性显得尤其重要。",
                "靶向药物的特点是针对特定靶点产生作用，每个病人的病情不尽相同，适用的药物也各有不同，因此可以对肿瘤进行精准治疗。",
                "根据IMS的数据，预计目前全球肿瘤药物市场规模可达1500亿美元，肿瘤处方药销售额约为1100亿美元。"
                "援引前瞻研究院的分析，预计单克隆抗体类药物和小分子靶向药物在未来将占据最大的市场份额。"
            ],
            'dates': 1619596258213
        }
        result, status_code = get_from_api(example, self.org_ner_url)
        assert status_code == 200


    @pytest.mark.parametrize(
        "data_name",
        [
            "data22",
            "data23",
            "data24",
            "data25"
        ]
    )
    def test_ecslogger(self, data_name):
        """After request API, news should be logged to 176es."""
        json_file = self.data_dir / f'{data_name}.json'
        with json_file.open('r') as f:
            raw_data = json.load(f)
        request_inputs = [
            {
                'article_id': r['esid'],
                'publish_date': int2date(r['publish_date']),
                'source': r['resource'],
                'investee': r['company'],
                'round': r['round'],
                'amount': r['amount'],
                'label': r['label'],
                'request_start_time': int(time.time() * 1000)
            }
            for r in raw_data
        ]
        esids = [r['esid'] for r in raw_data]
        for r in raw_data:
            r['is_new'] = 1
        results = []
        for d, new_data in zip(request_inputs, raw_data):
            start_time = int(time.time() * 1000)
            index_bulk(self.test_es, [new_data])
            api_result, status_code = get_from_api(d, self.folding_url)
            assert status_code == 200
            to_upload = [
                {
                    'esid': d['article_id'],
                    'similar_esid': api_result['head_article']
                }
            ]
            update_bulk(self.test_es, to_upload)
            logger.debug('Waiting 15s for the logs to be uploaded.')
            time.sleep(15)
            body = {
                "query": {
                    "bool": {
                        "must": {
                            "term": {"host.name": self.filebeat_hostname}
                        },
                        "filter": {
                            "range": {"@timestamp": {"gte": start_time}}
                        }
                    }
                }
            }
            log_results = self.filebeat_es.search(index=self.filebeat_index, body=body)
            log_results = log_results['hits']['hits']
            results.extend(log_results)
        assert len(results) == len(raw_data)
        delete_bulk(self.test_es, esids)

