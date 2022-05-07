from abc import ABC, abstractmethod

from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ESObject(ABC):

    def __init__(
            self,
            index_name,
            hosts=None,
            user_name='esjava',
            user_password='esjava123abc',
            port=9325
    ):
        if not hosts:
            hosts = ['esnode3.cubees.com', 'esnode5.cubees.com', 'esnode4.cubees.com',
                     'esnode6.cubees.com', 'esnode7.cubees.com', 'esnode8.cubees.com', 'esnode9.cubees.com']
        self.index_name = index_name

        self.es = Elasticsearch(
            hosts=hosts,
            http_auth=(user_name, user_password),
            port=port,
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False
        )

        exist = self.es.indices.exists(index=self.index_name)
        if not exist:
            raise AssertionError(f'{index_name} does not exist!')

    @abstractmethod
    def get_everything_body(self):
        ...

    def upload_single_data(self, data, id=None):
        res = self.es.index(index=self.index_name, body=data, id=id, refresh=True)
        return res

    def update_single_data(self, data, id):
        res = self.es.update(index=self.index_name, body=data, id=id, refresh=True)
        return res

    def delete_by_id(self, id):
        self.es.delete(index=self.index_name, id=id, refresh=True)

    def delete_by_query(self, query_field_name, query_field_value):
        query_doc = {
            "query": {
                "term": {
                    query_field_name: query_field_value
                },
            },
        }
        self.es.delete_by_query(index=self.index_name, body=query_doc, refresh=True)

    def delete_index(self):
        self.es.indices.delete(index=self.index_name)
        print(f'Index {self.index_name} deleted!')

    def count_index(self, body=None):
        if not body:
            body = {}
        count = self.es.count(index=self.index_name, body=body)['count']
        return count

    def fuzzy_match(self, query_field_name, query_field_value, get_fields=None, query_max_size=100):
        query_doc = {
            "query": {
                "match": {
                    query_field_name: query_field_value
                },
            },
            'size': query_max_size
        }
        re = self.es.search(index=self.index_name, body=query_doc, request_timeout=60)
        re = re['hits']['hits']
        if not get_fields:
            res = [i['_source'] for i in re]
        else:
            res = [{field: i['_source'][field] for field in get_fields} for i in re]
        return res

    def exact_match(self, query_field_name, query_field_value):
        """
        Return List[Tuple[dict, int]]
        """
        query_doc = {
            "query": {
                "term": {
                    query_field_name: query_field_value
                },
            },
        }
        re = self.es.search(index=self.index_name, body=query_doc, request_timeout=60)
        re = re['hits']['hits']
        return [(re[i]['_source'], re[i]['_id']) for i in range(len(re))]

    def exact_match_one(self, query_field_name, query_field_value):
        res = self.exact_match(query_field_name, query_field_value)
        if len(res) != 1:
            raise ValueError(f'Found {len(res)} result with {query_field_name} {query_field_value}')
        return res[0]

    def clear_index(self):
        self.es.delete_by_query(self.index_name, body={"query": {"match_all": {}}}, refresh=True)
        print(f'{self.index_name} index data wiped.')

    def get_index_setting(self):
        r = self.es.indices.get_settings(index=self.index_name)
        print(r)

    def upload_bulk(self, data, ids=None):
        if ids:
            assert len(data) == len(ids)
            actions = [
                {
                    '_op_type': 'index',
                    '_index': self.index_name,
                    '_id': ids[i],
                    '_source': data[i]
                } for i in range(len(data))
            ]
        else:
            actions = [
                {
                    '_op_type': 'index',
                    '_index': self.index_name,
                    '_source': d
                } for d in data
            ]
        print('Start bulk uploading:...')
        bulk(self.es, actions, request_timeout=360)
        self.es.indices.refresh(index=self.index_name)
        print(f'{len(data)} records uploaded!')

    def get_everything(self, size=3000):
        body = self.get_everything_body()

        data = []
        res = self.es.search(index=self.index_name, body=body)
        for r in res['hits']['hits']:
            doc = r['_source']
            data.append(doc)
        bookmark = [res['hits']['hits'][-1]['sort'][0], res['hits']['hits'][-1]['sort'][1]]

        total = self.es.count(index=self.index_name, body={})['count']
        for _ in tqdm(range(total // size)):
            body['search_after'] = bookmark
            res = self.es.search(index=self.index_name, body=body)
            if not res['hits']['hits']:
                break
            for r in res['hits']['hits']:
                doc = r['_source']
                data.append(doc)
            bookmark = [res['hits']['hits'][-1]['sort'][0], res['hits']['hits'][-1]['sort'][1]]
        return data


class ESNews(ESObject):

    def __init__(
            self,
            index_name='news_library_mid_fold'
    ):
        super().__init__(index_name)

    def get_everything_body(self, size=2000):
        body = {
            "size": size,
            "query": {
                "match_all": {}
            },
            "sort": [
                {"publish_time": "asc"},
                {"milvusid": "asc"}
            ]
        }
        return body


class ESCluster(ESObject):

    def __init__(
            self,
            index_name='news_library_fold'
    ):
        super().__init__(index_name)

    def get_everything_body(self, size=2000):
        body = {
            "size": size,
            "query": {
                "match_all": {}
            },
            "sort": [
                {"clusterid": "asc"},
                {"cluster": "asc"}
            ]
        }
        return body
