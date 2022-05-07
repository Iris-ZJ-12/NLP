from pharm_ai.prophet.predictor import NewsFilterPredictor, IraFilterPredictor, OrgFilterPredictor, NerPredictor
from pharm_ai.prophet.news_folding.news_folding import NewsFolding
from pharm_ai.prophet.utils import set_cuda_environ
from datetime import datetime
from getmac import get_mac_address
from loguru import logger
import uuid
import ntplib
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class Prophet:
    datetime_format = "%Y-%m-%d"

    def __init__(self, cuda_devices=(0, 0, 0, 0), es_host=None):
        set_cuda_environ(cuda_devices)
        self.news_filter = NewsFilterPredictor(cuda_devices[0])
        self.ira_filter = IraFilterPredictor(cuda_devices[1])
        self.news_folding = NewsFolding(es_host=es_host)
        self.org_filter = OrgFilterPredictor(cuda_devices[2])
        self.ner = NerPredictor(cuda_devices[3])

    def close_multiprocessing(self):
        self.news_filter.sm_model.args.use_multiprocessing_for_evaluation = False
        self.news_filter.sm_model.args.use_multiprocessing = False
        self.org_filter.sm_model.args.use_multiprocessing = False
        self.org_filter.sm_model.args.use_multiprocessing_for_evaluation = False
        self.ira_filter.sm_model.args.use_multiprocessing = False
        self.ira_filter.sm_model.args.use_multiprocessing_for_evaluation = False
        self.ner.sm_model.args.use_multiprocessing = False
        self.ner.sm_model.args.use_multiprocessing_for_evaluation = False
        self.ner.sm_model.args.use_multiprocessed_decoding = False

    def prophet(self, input_dic: dict, timestamp_loggers_iter=None):
        result = dict()
        total_final_output = dict()
        total_intermediary_output = dict()
        for article_id, article_input_fields in input_dic.items():
            if timestamp_loggers_iter:
                time_logger = next(timestamp_loggers_iter)
                time_logger.log_event_start_processing_times()
            final_output = dict()
            intermediary_output = dict()
            title = article_input_fields["title"]
            paras = article_input_fields["paragraphs"]
            date_str = article_input_fields["publish_date"]
            date = datetime.strptime(date_str, self.datetime_format) if date_str else datetime.today()
            src = article_input_fields["news_source"]

            intermediary_output["paragraphs"] = paras

            news_filter_label = self.news_filter.predict(title, paras)[0]
            final_output['news_filter_label'] = news_filter_label
            intermediary_output["news_filter_label"] = news_filter_label

            ira_filter_labels = self.ira_filter.predict(date, paras)
            selected_investee, selected_round, selected_amount = self.ner.predict_ira(ira_filter_labels, paras)

            intermediary_output["ira_filter_labels"] = ira_filter_labels

            final_output["investee"] = selected_investee
            final_output["round"] = selected_round
            final_output["amount"] = selected_amount

            org_filter_labels = self.org_filter.predict(date, paras)
            intermediary_output["org_filter_labels"] = org_filter_labels
            selected_org_ner = self.ner.predict_org(news_filter_label, org_filter_labels, paras)
            final_output["investor/consultant"] = selected_org_ner
            intermediary_output["investor/consultant"] = selected_org_ner

            total_final_output.update({article_id: final_output})
            total_intermediary_output.update({article_id: intermediary_output})
            if timestamp_loggers_iter:
                time_logger.log_event_processing_finished_timestamp()
        result["final_output"] = total_final_output
        result["intermediary_output"] = total_intermediary_output
        return result


class TimestampLogger:
    ntp_client = ntplib.NTPClient()
    es_server = Elasticsearch(hosts=[{"host": "gpu176", "port": "9200"}],
                              http_auth=('fzq', 'fzqfzqfzq'), timeout=60)
    local_mac_address = get_mac_address()

    def __init__(self, news_id, online=False):
        self.request_id = uuid.uuid1()
        self.news_id = news_id
        self.online = online

    @classmethod
    def get_time(cls, online=False):
        """
        :param online: Get network timestamp.
        :return: current timestamp (ms).
        """
        if online:
            ntp_servers = ['0.cn.pool.ntp.org', '1.cn.pool.ntp.org', '2.cn.pool.ntp.org', '3.cn.pool.ntp.org']
            for server in ntp_servers:
                try:
                    response = cls.ntp_client.request(server)
                    t_ms = int(response.tx_time * 1000)
                    break
                except:
                    if server == '3.cn.pool.ntp.org':
                        logger.error('Get network time error.')
                        t_ms = None
                    else:
                        continue
        else:
            t_ms = int(datetime.now().timestamp() * 1000)
        return t_ms

    def log_event_create_time(self, input_time=None):
        self.event_create_timestamp = input_time

    def log_event_arrival_time(self, time=None):
        self.event_arrival_timestamp = TimestampLogger.get_time(self.online) if not time else time

    def log_event_start_processing_times(self):
        self.event_start_processing_timestamp = TimestampLogger.get_time(self.online)

    def log_event_processing_finished_timestamp(self):
        self.event_processing_finished_timestamp = TimestampLogger.get_time(self.online)

    def upload_to_es(self):
        actions = [
            {
                '_op_type': 'index',
                '_index': 'logging_timestamps',
                '_id': self.request_id,
                'doc': {
                    'news_id': self.news_id,
                    'event_created_timestamp': self.event_create_timestamp,
                    'event_arrival_timestamp': self.event_arrival_timestamp,
                    'event_start_processing_timestamp': self.event_start_processing_timestamp,
                    'event_processing_finished_timestamp': self.event_processing_finished_timestamp
                }
            }
        ]
        if not self.online:
            actions[0]['doc'].update({'local_mac_address': TimestampLogger.local_mac_address})
        bulk(self.es_server, actions, request_timeout=120)


if __name__ == "__main__":
    test_input_1 = {
        "10e43070d9b8edd39737a77815a9ce9e": {
            "title": "科赛睿生物完成3000万美元B轮融资",
            "paragraphs": [
                "投资界（ID：pedaily2012）1月20日消息，近日，智康博药母公司科赛睿生物（Cothera Bioscience）宣布完成近3000万美元B轮融资，由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛投资、昆仑资本、Harbinger Venture等国内外基金共同参与，老股东联想之星、君联资本持续加持。据了解，筹集的资金将主要用于加速推进公司多个肿瘤靶向创新药在全球范围的临床开发。科赛睿生物由原知名CRO CrownBio的创始团队核心成员创立，是一家创新型肿瘤新药研发公司，专注于创新肿瘤靶向药物研发。本轮融资将主要用于加速推进公司多个肿瘤靶向创新药在全球范围的临床开发。科赛睿通过与国内顶尖肿瘤医学中心的合作，开展多项前瞻性比对临床试验，已初步证明i-CR®体系可以有效预测药物的的实际临床反应，从而有望大幅提高抗肿瘤新药研发的效率和临床成功率。科赛睿生物共同创始人，首席执行官吴越博士表示：“非常感谢新老投资人对公司的支持，共同探索肿瘤新药开发的新领域、发掘新价值。展望未来，我们还要继续以开发全球首创新药（first-in-class）为立足之本，充分发挥公司多年积累的转化医学的壁垒和特长，以扎实的药物机理研究和肿瘤生物学认知为根本，实现知其然并知其所以然，大大提高新药开发成功率，以创新产品开拓全球市场，为解决肿瘤病人未满足的医疗需求做出我们的贡献。”",
                "【本文为投资界原创，网页转载须在文首注明来源投资界（微信公众号ID：PEdaily2012）及作者名字。微信转载，须在微信原文评论区联系授权。如不遵守，投资界将向其追究法律责任。】"
            ],
            "publish_date": "2021-01-20",
            "news_source": "投资界"
        },
        "1a9e701a6be4fc3ff324af53010c16c8": {
            "title": "药融资丨科赛睿生物完成近3000万美元B轮融资，加速肿瘤靶向创新药全球临床开发",
            "paragraphs": [
                "生物药创新技术大会（BPIT）即将于2021年3月25日-26日在上海拉开序幕，四大分会场，精彩纷呈（查看往年盛况）。点击“阅读原文”，立即报名！",
                "2021年1月19日/药融资新闻 DrugFunds News/--科赛睿生物宣布完成近3000万美元B轮融资。本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，老股东君联资本，联想之星持续加持。易凯资本在本次交易中担任科赛睿生物的独家财务顾问。",
                "据悉，本轮筹集的资金将主要用于加速推进公司多个肿瘤靶向创新药在全球范围的临床开发。",
                "科赛睿生物成立于2012年，由原CRO中美冠科（CrownBio）的创始团队核心成员创立，专注于创新肿瘤靶向药物研发。目前，其自主研发的i-CR®技术平台结合条件性重编程（conditional reprogramming）原代肿瘤细胞培养技术和高内涵药物筛选体系，在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。",
                "当下，通过开展多项前瞻性比对临床试验，科赛睿生物已初步证明 i-CR®体系可以有效预测药物的实际临床反应。而利用i-CR®技术平台与临床药物反应的高度相关性，科赛睿已经在合成致死和免疫治疗领域开发出针对癌症的一系列新药产品线。",
                "其中，核心产品PC-002是一个针对MYC基因变异肿瘤的first-in-class小分子药物。MYC蛋白在超过50%的肿瘤中高表达，为最重要的“不可成药”肿瘤靶点之一，PC-002通过独特的MOA靶向MYC蛋白降解，选择性诱导MYC依赖的肿瘤细胞凋亡。目前，PC-002即将在美国开展临床2期，有望通过2期试验结果快速获批，成为泛癌种重磅产品。",
                "另外一个管线产品是CTB-02，其主要针对pan-KRAS变异的肠癌和非小细胞肺癌，预期在2021年首先在澳大利亚进入1期临床。",
                "科赛睿生物共同创始人、首席执行官吴越表示：“展望未来，我们还要继续以开发全球首创新药（first-in-class）为立足之本，充分发挥公司多年积累的转化医学的壁垒和特长，以扎实的药物机理研究和肿瘤生物学认知为根本，实现知其然并知其所以然，大大提高新药开发成功率，以创新产品开拓全球市场，为解决肿瘤病人未满足的医疗需求做出我们的贡献。”",
                "文章来源：亿欧《",
                "首发丨科",
                "赛睿生物完成B轮融资，加速肿瘤靶向创新药全球临床开发",
                "》",
                "申明：药融资所刊载内容版权归原创作者及权利人专属所有。文中出现的任何观点和信息仅供参考，不作为投资决策。如对上述项目感兴趣或有投融资消息披露、寻求采访报道，可以联系工作人员：13621634647（微信同号）。"
            ],
            "publish_date": "2021-01-20",
            "news_source": "药融资"
        },
        "e068c3bffccf39ee4010018d5ee155b0": {
            "title": "专注创新肿瘤靶向药物研发，科赛睿生物完成近3000万美元B轮融资",
            "paragraphs": [
                "创新肿瘤靶向药物研发商科赛睿生物宣布获得近3000万美元B轮融资。本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛投资、HarbingerVenture等国内外基金共同参与，老股东君联资本，联想之星持续加持。据介绍，本轮融资完成之后，所获资金将主要用于加速推进其多个肿瘤靶向创新药在全球范围的临床开发。\r",
                "                        (投中网)"
            ],
            "publish_date": "2021-01-20",
            "news_source": "投中网"
        },
        "1ef9df8fea32be7cdfa501c7d0c923e1": {
            "title": "科赛睿生物完成B轮融资",
            "paragraphs": [
                "                                            创业邦获悉，科赛睿生物（Cothera Bioscience）宣布完成近3000万美元B轮融资。本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛投资、Harbinger Venture等国内外基金共同参与，老股东君联资本，联想之星持续加持。筹集的资金将主要用于加速推进公司多个肿瘤靶向创新药在全球范围的临床开发。科赛睿生物由原知名CRO中美冠科（CrownBio）的创始团队核心成员创立，专注于创新肿瘤靶向药物研发，其自主研发的i-CR®技术平台结合条件性重编程（conditional reprogramming）原代肿瘤细胞培养技术和高内涵药物筛选体系，在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。                                    "
            ],
            "publish_date": "2021-01-19",
            "news_source": "创业邦"
        },
        "49f29b6026f264aa7274ba6e5b67dacc": {
            "title": "科赛睿生物获近3000万美元B轮融资",
            "paragraphs": [
                "投资界（ID：pedaily2012）1月19日消息，据36氪报道，创新肿瘤靶向药物研发商——科赛睿生物，宣布获得近3000万美元B轮融资。本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，老股东君联资本，联想之星持续加持，易凯资本在本次交易中担任科赛睿生物的独家财务顾问。本轮融资所得将主要用于加速推进其多个肿瘤靶向创新药在全球范围的临床开发。据了解，在肿瘤靶向药物方向上，科赛睿生物的竞争优势在于其自主研发的i-CR®技术平台，结合条件性重编程原代肿瘤细胞培养技术和高内涵药物筛选体系。其优势是能够在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。目前，科赛睿已经通过与国内头部的肿瘤医学中心开展合作，进行多项前瞻性比对临床试验，已初步证明 i-CR®体系可以有效预测药物的的实际临床反应。核心产品PC-002是一个针对MYC基因变异肿瘤的first-in-class小分子药物。MYC蛋白在超过50%的肿瘤中高表达，为最重要的“不可成药”肿瘤靶点之一，PC-002通过独特的MOA靶向MYC蛋白降解，选择性诱导MYC依赖的肿瘤细胞凋亡，即将开展美国临床2期。",
                "【本文为投资界原创，网页转载须在文首注明来源投资界（微信公众号ID：PEdaily2012）及作者名字。微信转载，须在微信原文评论区联系授权。如不遵守，投资界将向其追究法律责任。】"
            ],
            "publish_date": "2021-01-19",
            "news_source": "投资界"
        },
        "8e1c5a252a418643e02f121fb7d7ab55": {
            "title": "36氪首发 | 瞄准创新肿瘤药物，「科赛睿生物」获得近3000万美元B轮融资",
            "paragraphs": [
                "36氪获悉，创新肿瘤靶向药物研发商「科赛睿生物」，宣布获得近3000万美元B轮融资。本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，老股东君联资本，联想之星持续加持，易凯资本在本次交易中担任科赛睿生物的独家财务顾问。据悉，本轮融资所得将主要用于加速推进其多个肿瘤靶向创新药在全球范围的临床开发。",
                "纵观新药研发领域，肿瘤靶向药物是目前最活跃的方向之一，其中较为著名的包括PD-1/PD-L1、VEGF、HER2、TNF-α等。根据CFDA和CDE网站上的数据显示，目前中国正在申报临床、正在进行临床和正在申报生产的小分子靶向药物约有百种，大分子靶向药数十种，其中包括国内原研、仿制药和国外已上市的品种。",
                "在这个赛道上，科赛睿生物的竞争优势在于其自主研发的i-CR®技术平台，结合条件性重编程原代肿瘤细胞培养技术和高内涵药物筛选体系。",
                "其优势是能够在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。",
                "目前，科赛睿已经通过与国内头部的肿瘤医学中心开展合作，进行多项前瞻性比对临床试验，已初步证明 i-CR®体系可以有效预测药物的的实际临床反应。",
                "研发管线的进度，是考察创新药物企业的重要环节。这一方面，科赛睿已经在合成致死和免疫治疗领域开发出针对癌症的一系列新药产品线，前期研发成果已经申请了多项国际和国内专利。",
                "其中，核心产品PC-002是一个针对MYC基因变异肿瘤的first-in-class小分子药物。MYC蛋白在超过50%的肿瘤中高表达，为最重要的“不可成药”肿瘤靶点之一，PC-002通过独特的MOA靶向MYC蛋白降解，选择性诱导MYC依赖的肿瘤细胞凋亡，即将开展美国临床2期。",
                "另外一个管线产品CTB-02针对pan-KRAS变异的肠癌和非小细胞肺癌，预期在2021年首先在澳大利亚进入1期临床。",
                "图片来源于前瞻研究院",
                "在快速增长的创新药市场中，全球化竞争加剧，因此产品管线研发的效率以及创新性显得尤其重要。",
                "靶向药物的特点是针对特定靶点产生作用，每个病人的病情不尽相同，适用的药物也各有不同，因此可以对肿瘤进行精准治疗。",
                "根据IMS的数据，预计目前全球肿瘤药物市场规模可达1500亿美元，肿瘤处方药销售额约为1100亿美元。援引前瞻研究院的分析，预计单克隆抗体类药物和小分子靶向药物在未来将占据最大的市场份额。"
            ],
            "publish_date": "2021-01-19",
            "news_source": "36氪"
        },
        "eb492084a0a4b8db799e8a76211eba80": {
            "title": "首发丨科赛睿生物完成B轮融资，加速肿瘤靶向创新药全球临床开发",
            "paragraphs": [
                "科赛睿生物宣布完成近3000万美元B轮融资。本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，老股东君联资本，联想之星，昆仑资本持续加持。易凯资本在本次交易中担任科赛睿生物的独家财务顾问。据悉，本轮筹集的资金将主要用于加速推进公司多个肿瘤靶向创新药在全球范围的临床开发。科赛睿生物成立于2012年，由原CRO中美冠科（CrownBio）的创始团队核心成员创立，专注于创新肿瘤靶向药物研发。目前，其自主研发的i-CR®技术平台结合条件性重编程（conditional reprogramming）原代肿瘤细胞培养技术和高内涵药物筛选体系，在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。当下，通过开展多项前瞻性比对临床试验，科赛睿生物已初步证明 i-CR®体系可以有效预测药物的实际临床反应。而利用i-CR®技术平台与临床药物反应的高度相关性，科赛睿已经在合成致死和免疫治疗领域开发出针对癌症的一系列新药产品线。其中，核心产品PC-002是一个针对MYC基因变异肿瘤的first-in-class小分子药物。MYC蛋白在超过50%的肿瘤中高表达，为最重要的“不可成药”肿瘤靶点之一，PC-002通过独特的MOA靶向MYC蛋白降解，选择性诱导MYC依赖的肿瘤细胞凋亡。目前，PC-002即将在美国开展临床2期，有望通过2期试验结果快速获批，成为泛癌种重磅产品。另外一个管线产品是CTB-02，其主要针对pan-KRAS变异的肠癌和非小细胞肺癌，预期在2021年首先在澳大利亚进入1期临床。科赛睿生物共同创始人，首席执行官吴越表示：“展望未来，我们还要继续以开发全球首创新药（first-in-class）为立足之本，充分发挥公司多年积累的转化医学的壁垒和特长，以扎实的药物机理研究和肿瘤生物学认知为根本，实现知其然并知其所以然，大大提高新药开发成功率，以创新产品开拓全球市场，为解决肿瘤病人未满足的医疗需求做出我们的贡献。”"
            ],
            "publish_date": "2021-01-19",
            "news_source": "亿欧"
        },
        "c1b4f26598eee03298299fc8f5286423": {
            "title": "【首发】科赛睿生物完成B轮融资，持续推进肿瘤靶向创新药研发",
            "paragraphs": [
                "科赛睿生物（Cothera Bioscience）宣布完成近3000万美元B轮融资。本轮融资由清松资本领投，招银国际、夏尔巴投资、新世界投资、朗盛资本、Harbinger Venture等国内外基金共同参与，老股东君联资本，联想之星持续加持。筹集的资金将主要用于加速推进公司多个肿瘤靶向创新药在全球范围的临床开发。 易凯资本在本次交易中担任科赛睿生物的独家财务顾问。 科赛睿生物由原知名CRO中美冠科（CrownBio）的创始团队核心成员创立，专注于创新肿瘤靶向药物研发，其自主研发的i-CR®技术平台结合条件性重编程（conditional reprogramming）原代肿瘤细胞培养技术和高内涵药物筛选体系，在体外高效、无差别扩增患者原代肿瘤细胞，保留了患者肿瘤的异质性，同时结合高内涵药物筛选体系，利用患者原代肿瘤细胞在体外进行高效药物筛选，更适用于临床患者的个体化药物筛选和新药研发。科赛睿通过与国内顶尖肿瘤医学中心的合作，开展多项前瞻性比对临床试验，已初步证明 i-CR®体系可以有效预测药物的的实际临床反应，从而有望大幅提高抗肿瘤新药研发的效率和临床成功率。 利用i-CR®技术平台与临床药物反应的高度相关性，科赛睿已经在合成致死 (synthetic lethality) 和免疫治疗领域开发出针对癌症的一系列新药产品线, 前期研发成果已经申请了多项国际和国内专利。 公司核心产品PC-002是一个针对MYC基因变异肿瘤的first-in-class小分子药物。MYC蛋白在超过50%的肿瘤中高表达，为最重要的“不可成药”肿瘤靶点之一，PC-002通过独特的MOA靶向MYC蛋白降解，选择性诱导MYC依赖的肿瘤细胞凋亡。PC-002即将开展美国临床2期，有望通过2期试验结果快速获批，成为泛癌种重磅产品。公司另外一个管线产品CTB-02针对pan-KRAS变异的肠癌和非小细胞肺癌，预期在2021年首先在澳大利亚进入1期临床。 科赛睿生物共同创始人，首席执行官吴越博士表示：非常感谢新老投资人对公司的支持，共同探索肿瘤新药开发的新领域、发掘新价值。科赛睿能发展到今天，靠的是坚持科学，坚持创新，为他人所不能为的勇气和实力。展望未来，我们还要继续以开发全球首创新药（first-in-class）为立足之本，充分发挥公司多年积累的转化医学的壁垒和特长，以扎实的药物机理研究和肿瘤生物学认知为根本，实现知其然并知其所以然，大大提高新药开发成功率，以创新产品开拓全球市场，为解决肿瘤病人未满足的医疗需求做出我们的贡献。清松资本合伙人张松博士表示：我们看好以吴越博士为首的原中美冠科创始团队丰富的药物开发经验和对转化肿瘤医学领域的深刻理解，坚定支持Cothera团队基于独特的药物筛选体系对于多个“不可成药”靶点的探索和药物开发。清松资本希望通过此次携手，能够与团队长期相伴，在创新药市场竞争加剧的背景下，助力公司first-in-class的肿瘤靶向药物在全球范围内的开发，并期待与公司一起为肿瘤治疗领域带来新的突破。 易凯资本董事总经理张骁表示：科赛睿生物拥有世界一流的转化医学团队，基于其独有的i-CR药物筛选平台，多年来专注于不可成药靶点的药品开发，并在Myc，kras等靶点新药研发上取得重要进展。我们很荣幸能够帮助科赛睿一起完成本轮融资，我们期待着公司未来更多的临床进展，早日令更多的患者获益。"
            ],
            "publish_date": "2021-01-19",
            "news_source": "动脉网"
        }
    }
    p = Prophet(cuda_devices=(0, 1, 2, 3))
    p.close_multiprocessing()
    res = p.prophet(test_input_1)
    print(res)
