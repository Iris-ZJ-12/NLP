from pharm_ai.prophet.prophet import Prophet
from pharm_ai.prophet.news_folding.news_folding import NewsFolding
from pharm_ai.prophet.utils import (date2int, migrate_data_to_test_es, get_es, delete_bulk,
                                    get_from_esids, update_bulk, index_bulk)
from datetime import datetime
from pathlib import Path
import json
import pytest
import sys
from pprint import pprint
from loguru import logger

logger.add(sys.stdout, filter=lambda record: record["extra"].get("task") == "test")
test_logger = logger.bind(task="test")


class TestProphet:
    @pytest.fixture(scope='class')
    def prophet_obj(self):
        prophet_obj = Prophet(cuda_devices=(8, 8, 8, 8))
        prophet_obj.close_multiprocessing()
        return prophet_obj

    @pytest.mark.parametrize(
        "loading_file_name",
        [
            "data2", "data3",
            "data4",  # empty paragraph
            "data7", "data14",
            "data15",
            "data18",
            "data19"
        ]
    )
    def test_prophet(self, loading_file_name, prophet_obj):
        data_dir = Path(__file__).parent/'data'
        loading_json_path = data_dir / (loading_file_name + '.json')
        with loading_json_path.open('r') as f:
            to_predict = json.load(f)
        result = prophet_obj.prophet(to_predict)
        assert all(k in ['final_output', 'intermediary_output'] for k in result)
        for field in ['investee', 'round', 'amount']:
            assert all(not t['paragraphs'] or (r[field] and ';' not in r[field])
                       for t, r in zip(to_predict.values(), result['final_output'].values())), \
                f'{field} output format wrong!'

    @pytest.mark.xfail(reason='org_ner could be empty.')
    def test_case1(self, prophet_obj):
        """org_ner should not be empty."""
        data_file = Path(__file__).parent/'data'/'data21.json'
        with data_file.open() as f:
            to_predict = json.load(f)
        test_esid = list(to_predict.keys())[0]
        result = prophet_obj.prophet(to_predict)
        assert any(result['final_output'][test_esid]['investor/consultant'])


class TestNewsFolding:
    data_dir = Path(__file__).parent/'data'
    index = 'invest_news'
    news_folder = NewsFolding()
    es = get_es('test')

    fake_investee1, fake_round1, fake_amount1 = '黑湖智造', 'C轮', '近5亿元人民币'


    @pytest.fixture
    def data_case1(self):

        raw = self.get_json_data('data6')
        to_upload, folding_inputs, esids = [], [], []
        for esid, body in raw.items():
            to_upload.append({
                'esid': esid,
                'title': body['title'],
                'content': '\n'.join(body['paragraphs']),
                'publish_date': date2int(body['publish_date']),
                'resource': body['news_source'],
                'company': self.fake_investee1,
                'round': self.fake_round1,
                'amount': self.fake_amount1,
                'is_new': '1'
            })
            folding_inputs.append({
                'article_id': esid, 'date': body['publish_date'],
                'selected_investee': self.fake_investee1,
                'selected_round': self.fake_round1,
                'selected_amount': self.fake_amount1
            })
            esids.append(esid)
        yield from self.pre_post_test(esids, folding_inputs, to_upload)

    def get_json_data(self, data_name):
        data_file_name = self.data_dir / (data_name + '.json')
        with data_file_name.open('r') as f:
            raw: dict = json.load(f)
        return raw

    def upload_data(self, to_upload, type_='update'):
        """
        Args:
            to_upload: List[Dict]
            type_: str. "index" or "update"
        """
        if type_ == 'update':
            update_bulk(self.es, to_upload)
        else:
            index_bulk(self.es, to_upload)

    def test_case1(self, data_case1):
        esids, folding_inputs, to_upload = data_case1
        expected_heads = ["c81ff1072eda75993cf3ad116752ee16"]
        folding_res = self.imitate_news_folding(folding_inputs, to_upload, esids)
        assert (all(folding_res[esid][0] for esid in expected_heads) and
                all(not folding_res[esid][0] for esid in esids if esid not in expected_heads))

    def imitate_news_folding(self, folding_inputs, to_uploads, esids, epoch=1, upload_each=True):
        """
        Imitate request folding results by one-by-one request twice.
        Args:
            - folding_inputs: List[Dict]. Parameters input to news_folding.predict.
            - to_uploads: List[Dict]. Prepared data to upload to ES before executing news folding each.
            - esids: The ESIDs of folding results (could be more than folding inputs).
            - upload_each: bool. Upload each data to es before folding each, else upload all data before folding.
        Returns:
            Folding results. Dict[ESID -> bool].
        """
        # set default results
        folding_res = {esid: (True, esid) for esid in esids}
        if not upload_each:
            self.upload_data(to_uploads, 'index')
        for run_time in range(epoch):
            test_logger.info('Starting folding epoch {}', run_time + 1)
            for d, new_data in zip(folding_inputs, to_uploads):
                if run_time == 0 and upload_each:
                    self.upload_data([new_data], 'index')
                if folding_res[d['article_id']][0]:
                    *cur_result, head_changes = self.news_folder.predict(**d, return_changes=True)
                    if cur_result[1] != d['article_id']:
                        to_upload = [
                            {
                                'esid': d['article_id'],
                                'similar_esid': cur_result[1]
                            }
                        ]
                        self.upload_data(to_upload)
            # get folding results
            folding_res = self.get_folding_results(esids)
            pprint(folding_res)
        return folding_res

    def delete_es_testing_data(self, esids, verbose=True):
        """Delete testing data in ES."""
        delete_bulk(self.es, esids)
        if verbose:
            test_logger.info('Auto removed from test ES: {}', esids)

    def get_folding_results(self, esids):
        raw_folding_results = get_from_esids(self.es, esids, fields=['similar_esid'])
        folding_res = {r['esid']: (r.get('similar_esid') == r['esid'] or not r.get('similar_esid'),
                                   r.get('similar_esid') or r['esid']) for r in raw_folding_results}
        for esid in esids:
            if esid not in folding_res:
                folding_res[esid] = (True, esid)
        return folding_res

    @pytest.fixture
    def data_case_manual_add(self):
        """Manually added news should not be folded.

        Manaul adding -> 已同步;

        已同步：is_new=0 && label=医药.
        """
        raw = self.get_json_data('data6')

        # make news '3c95' as manual news
        manual_article = '3c953eb420adee39bdf64b7c8e544633'
        expected_heads = [manual_article, "c81ff1072eda75993cf3ad116752ee16"]
        to_upload, folding_inputs, esids = [], [], []
        for esid, body in raw.items():
            current = {
                'esid': esid,
                'title': body['title'],
                'content': '\n'.join(body['paragraphs']),
                'publish_date': date2int(body['publish_date']),
                'resource': body['news_source'],
                'company': self.fake_investee1,
                'round': self.fake_round1,
                'amount': self.fake_amount1,
                'is_new': 1,
                'label': '医药'
            }
            if esid == manual_article:
                current['is_new'] = 0
                body.pop('similar_esid', None)
            else:
                # not request manual news
                folding_inputs.append({
                    'article_id': esid, 'date': body['publish_date'],
                    'selected_investee': self.fake_investee1,
                    'selected_round': self.fake_round1,
                    'selected_amount': self.fake_amount1
                })
            to_upload.append(current)
            esids.append(esid)
        pre_post_gen = self.pre_post_test(esids, folding_inputs, to_upload)
        esids, folding_inputs, to_upload = next(pre_post_gen)
        yield esids, folding_inputs, to_upload, expected_heads
        try:
            next(pre_post_gen)
        except StopIteration:
            pass

    def test_case_manual_news(self, data_case_manual_add):
        esids, folding_inputs, to_uploads, expected_heads = data_case_manual_add
        folding_res = self.imitate_news_folding(folding_inputs, to_uploads, esids)
        assert (all(folding_res[esid][0] for esid in expected_heads) and
                all(not folding_res[esid][0] for esid in esids if esid not in expected_heads))

    @pytest.fixture
    def fake_irrelevant_news(self):
        """Irrelevant articles should not be head news."""
        raw = self.get_json_data("data6")

        irrelevant_article = 'c81ff1072eda75993cf3ad116752ee16'
        expected_head_article = 'c2220e35bf78e1d6669f92853d0447ed'
        to_upload, folding_inputs, esids = [], [], []
        for esid, body in raw.items():
            to_upload.append({
                'esid': esid,
                'title': body['title'],
                'content': '\n'.join(body['paragraphs']),
                'publish_date': date2int(body['publish_date']),
                'resource': body['news_source'],
                'company': self.fake_investee1,
                'round': self.fake_round1,
                'amount': self.fake_amount1,
                'is_new': 1,
                'label_prediction': '医药' if esid != irrelevant_article else '非相关'
            })
            folding_inputs.append({
                'article_id': esid, 'date': body['publish_date'],
                'selected_investee': self.fake_investee1,
                'selected_round': self.fake_round1,
                'selected_amount': self.fake_amount1,
                'news_filter_label': '医药' if esid != irrelevant_article else '非相关'
            })
            esids.append(esid)
        # sort articles to make head article the last
        esids.sort(key=lambda x: x == expected_head_article)
        folding_inputs.sort(key=lambda x: x['article_id'] == expected_head_article)
        to_upload.sort(key=lambda x: x['esid'] == expected_head_article)

        gen = self.pre_post_test(esids, folding_inputs, to_upload)
        yield *next(gen), irrelevant_article
        try:
            next(gen)
        except StopIteration:
            pass

    @pytest.fixture
    def data_company_id(self):
        to_fold = ['62e7ccb2f9b8a48153d6ef514a7f211d', 'b0ed65eadbe807c8c44292d8026dc347',
                   '0f704a6f3469110526da25afccb30ced', '54a6e648f10706839d1af86b71295fd8']
        json_file = self.data_dir / 'data9.json'
        with json_file.open('r') as f:
            to_uploads = json.load(f)
        folding_inputs = [
            {
                'article_id': r['esid'],
                'date': datetime.fromtimestamp(r['publish_date'] / 1000),
                'selected_investee': r['company'],
                'selected_round': r['round'],
                'selected_amount': r['amount']
            }
            for r in to_uploads if r['esid'] in to_fold
        ]
        esids = [r['esid'] for r in to_uploads]
        pre_uploads = [t for t in to_uploads if t['esid'] not in to_fold]
        self.upload_data(pre_uploads, 'index')
        post_uploads = [t for t in to_uploads if t['esid'] in to_fold]
        self.news_folder.convert_company_id = True
        test_logger.info('Enable converting company id.')
        yield esids, folding_inputs, to_fold, post_uploads
        self.news_folder.convert_company_id = False
        test_logger.info('Disable converting company id.')
        self.delete_es_testing_data(esids)

    def test_case_company_id(self, data_company_id):
        """Query `company_id` should equal to query `company_name`."""
        esids, folding_inputs, to_fold, to_uploads = data_company_id
        # set the request investee field of an article as company name rather than company id
        for input in folding_inputs:
            if input['article_id'] == 'b0ed65eadbe807c8c44292d8026dc347':
                input['selected_investee'] = '臻络科技'
        folding_res = self.imitate_news_folding(folding_inputs, to_uploads, esids)
        assert all(not folding_res[r][0] if r in to_fold else folding_res[r][0] for r in folding_res)

    def test_case_fake_irrelevant(self, fake_irrelevant_news):
        esids, folding_inputs, to_uploads, irrelevant_article = fake_irrelevant_news
        folding_res = self.imitate_news_folding(folding_inputs, to_uploads, esids)
        assert all(folding_res[esid][1] != irrelevant_article if esid != irrelevant_article else folding_res[esid][0]
                   for esid in esids)

    @pytest.fixture
    def data_case_irrelevant(self, request):
        data_file = request.param
        esids, folding_inputs, raw = self.get_from_json(data_file, process_label=True)
        irrelevant_esids = [d['article_id'] for d in folding_inputs if d['news_filter_label'] == '非相关']
        gen = self.pre_post_test(esids, folding_inputs, raw)
        yield *next(gen), irrelevant_esids
        try:
            next(gen)
        except StopIteration:
            pass

    @pytest.mark.parametrize(
        "data_case_irrelevant",
        ["data32", "data33", "data34", "data35"],
        indirect=True
    )
    def test_irrelevant(self, data_case_irrelevant):
        esids, folding_inputs, to_uploads, irrelevant_articles = data_case_irrelevant
        folding_res = self.imitate_news_folding(folding_inputs, to_uploads, esids)
        assert all(folding_res[article][0] for article in irrelevant_articles)

    @pytest.fixture
    def data_publish(self):
        raw_data = self.get_json_data('data10')
        esids, folding_inputs = [], []
        selected_articles = ['7a861d62f307eaee03ab6f56b7868e73',
                             '9b50a2d8e05574bf68898e11e2b72aa7',
                             'fc121c6e11c6e25e4cf72bb35716df61']
        published_article = selected_articles[0]
        for r in raw_data:
            esid = r['esid']
            if esid in selected_articles:
                esids.append(esid)
                r['similar_esid'] = None
                r['label'] = '医药'
                r['label_prediction'] = '医药'
                if r['esid'] == published_article:
                    r['is_publish'] = 1
                    r['is_new'] = 0
                    r['publish_date'] = r['publish_date'] - 8 * 24 * 3600 * 1000  # make 8d earlier
                else:
                    r['is_publish'] = 0
                    r['is_new'] = 1
                    folding_inputs.append({
                        'article_id': r['esid'],
                        'date': datetime.fromtimestamp(r['publish_date'] / 1000),
                        'selected_investee': r['company'],
                        'selected_round': r['round'],
                        'selected_amount': r['amount']
                    })
        gen = self.pre_post_test(esids, folding_inputs, raw_data)
        yield *next(gen), published_article
        try:
            next(gen)
        except StopIteration:
            pass

    def test_case_publish(self, data_publish):
        esids, folding_inputs, raw_data, published_article = data_publish
        folding_res = self.imitate_news_folding(folding_inputs, raw_data, esids, upload_each=False)
        assert all(folding_res[r][0] if r == published_article else not folding_res[r][0] for r in folding_res)

    @pytest.fixture
    def data_group(self, request):
        esids, folding_inputs, raw_data = self.get_from_json(request.param, is_new=1)
        yield folding_inputs, raw_data, esids
        self.delete_es_testing_data(esids)

    @pytest.mark.parametrize(
        "data_group",
        [
            "data11",
            "data12",
            "data13",
            "data17",
            "data20",
            "data22",
            "data23",
            "data24",
            "data25",
            "data26",
            "data27",
            "data28"
        ],
        indirect=True
    )
    def test_case_group(self, data_group):
        """Data in the same group should fold together"""
        folding_inputs, to_uploads, esids = data_group
        folding_res = self.imitate_news_folding(folding_inputs, to_uploads, esids)
        assert sum(r for esid, (r, rid) in folding_res.items()) <= 1

    @pytest.mark.parametrize(
        "data_weight, expected_head",
        [
            ('data16', '1a730a55ee490633cd35a3908ea6afaa'),
            ('data29', '35e0a692d36972c82ca405fc89c3b895')
        ],
        indirect=['data_weight']
    )
    def test_case_weight(self, data_weight, expected_head):
        """Check weight of head articles to be folded. For example: news from Pharmcube should be the head."""
        esids, folding_inputs, raw_data = data_weight
        folding_res = self.imitate_news_folding(folding_inputs, raw_data, esids)
        for id_, (r, rid) in folding_res.items():
            if id_ == expected_head:
                assert r, f'{id_} should be the head news!'
            else:
                assert not r, f'Wrong: {id_} should be folded to {expected_head}!'

    @pytest.fixture
    def data_weight(self, request):
        esids, folding_inputs, raw_data = self.get_from_json(request.param, is_new=1, is_publish=0)
        yield from self.pre_post_test(esids, folding_inputs, raw_data)

    def pre_post_test(self, esids, folding_inputs, raw_data):
        folding_esids = [folding_input['article_id'] for folding_input in folding_inputs]
        all_esids, folding_raw_data, extra_raw_data, extra_esids = [], [], [], []
        for r in raw_data:
            cur_esid = r['esid']
            all_esids.append(cur_esid)
            if cur_esid in folding_esids:
                folding_raw_data.append(r)
            else:
                extra_esids.append(cur_esid)
                extra_raw_data.append(r)
        folding_raw_data.sort(key=lambda x: folding_esids.index(x['esid']))
        backup_data = self.backup_es_data(all_esids)
        if backup_data:
            backup_esids = [d['esid'] for d in backup_data]
            new_esids = [esid for esid in all_esids if esid not in backup_esids]
            self.delete_es_testing_data(backup_esids, False)
            test_logger.info('{} data backup and deleted: {} ', len(backup_esids), esids)
            if extra_raw_data:
                self.upload_data(extra_raw_data, 'index')
                test_logger.info('{} data pre-uploaded: {}', len(extra_raw_data), extra_esids)
            yield esids, folding_inputs, folding_raw_data
            self.upload_data(backup_data, 'index')
            test_logger.info('{} backup data restored: {}', len(backup_esids), backup_esids)
            if new_esids:
                self.delete_es_testing_data(new_esids, False)
                test_logger.info('{} new-created test data deleted: {} ', len(new_esids), new_esids)
        else:
            if extra_raw_data:
                self.upload_data(extra_raw_data, 'index')
                test_logger.info('{} data pre-uploaded: {}', len(extra_raw_data), extra_esids)
            yield esids, folding_inputs, folding_raw_data
            self.delete_es_testing_data(all_esids)

    def get_from_json(self, data_file, is_new: int = None, is_publish: int = None, process_label=False):
        raw_data = self.get_json_data(data_file)
        folding_inputs, esids = [], []
        for d in raw_data:
            if is_new is not None:
                d['is_new'] = is_new
            if is_publish is not None:
                d['is_publish'] = is_publish
            d.pop('similar_esid', None)
            esid = d['esid']
            cur_folding_input = {
                'article_id': esid,
                'date': datetime.fromtimestamp(d['publish_date'] / 1000),
                'selected_investee': d['company'],
                'selected_round': d['round'],
                'selected_amount': d['amount']
            }
            if process_label:
                is_irrelevant = not d.get('label_prediction') or d['label_prediction'] == '非相关'
                cur_folding_input['news_filter_label'] = '非相关' if is_irrelevant else d['label']
            folding_inputs.append(cur_folding_input)
            esids.append(esid)
        return esids, folding_inputs, raw_data

    @pytest.fixture
    def data_synced(self):
        esids, folding_inputs, raw_data = self.get_from_json('data30', is_publish=0)
        # make the last article highest weight, state=new; the others state=synced
        for i, r in enumerate(raw_data):
            if i == len(raw_data) - 1:
                r['is_new'] = 1
                r['resource'] = '医药魔方'
            else:
                r['is_new'] = 0
        yield from self.pre_post_test(esids, folding_inputs, raw_data)

    def backup_es_data(self, esids):
        body = {"query": {"terms": {"_id": esids}}}
        raw = self.es.search(body=body, index=self.index)
        hits = raw['hits']['hits']
        if hits:
            test_logger.info('Get {} data to backup for {}', len(hits), esids)
            return [{'esid': hit['_id'], **hit['_source']} for hit in hits]
        else:
            test_logger.info('No data to backup for {}', esids)
            return []

    def test_case_synced(self, data_synced):
        """Articles with state=synced should not be folded to article with state=new"""
        esids, folding_inputs, raw_data = data_synced
        folding_res = self.imitate_news_folding(folding_inputs, raw_data, esids)
        assert not folding_res[esids[-1]][0] and folding_res[esids[-1]][1] in esids[:-1]

    @pytest.fixture
    def data_pre_stored(self, request):
        """Pre-stored multiple data and request folding."""
        data_file, expected_head = request.param

        esids, folding_inputs, raw_data = self.get_from_json(data_file, is_new=1, is_publish=0)
        # sort articles to make head the first
        esids.sort(key=lambda x: x == expected_head, reverse=True)
        folding_inputs.sort(key=lambda x: x['article_id'] == expected_head, reverse=True)
        raw_data.sort(key=lambda x: x['esid'] == expected_head, reverse=True)

        gen = self.pre_post_test(esids, folding_inputs, raw_data)
        yield *next(gen), expected_head
        try:
            next(gen)
        except StopIteration:
            pass

    @pytest.mark.parametrize(
        'data_pre_stored',
        [
            ('data29', '35e0a692d36972c82ca405fc89c3b895'),
            ('data31', '2b2496e1deee8b47d9f3f02fdb599e12')
        ],
        indirect=True
    )
    def test_case_pre_stored(self, data_pre_stored):
        esids, folding_inputs, raw_data, expected_head = data_pre_stored

        folding_res = self.imitate_news_folding(folding_inputs, raw_data, esids, upload_each=False)
        assert folding_res[expected_head][0]
        assert all(not r[0] for esid, r in folding_res.items() if esid != expected_head)

