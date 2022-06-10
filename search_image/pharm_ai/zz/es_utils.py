from elasticsearch import Elasticsearch
import pandas as pd


class ESObject:
    def __init__(self, index_name='zz_projects_0125', index_type='_doc', hosts='101.201.249.176', user_name='elastic',
                 user_password='Zyl123123', port=9200):
        self.index_name = index_name
        self.index_type = index_type

        # 176 es port:9200 ; 176 zyl port:9325;
        self.es = Elasticsearch(hosts=hosts, http_auth=(user_name, user_password), port=port)  # on zyl

        if self.es.indices.exists(index=self.index_name) is not True:
            print('no index:', index_name)
            self.create_index()

    def create_index(self):
        _index_mappings = {
            "mappings": {
                "properties": {
                    "project_name": {
                        "type": "text",
                        "fields": {
                            "value": {
                                "type": "keyword",
                            },
                        },
                    },
                    "text_b": {
                        "type": "text",
                        "fields": {
                            "value": {
                                "type": "keyword",
                            },
                        },
                    },
                    "es_id": {
                        "type": "keyword",
                    },
                    "province": {
                        "type": "keyword",
                    },
                },
            },
        }
        self.es.indices.create(index=self.index_name, body=_index_mappings, ignore=400)
        print('final creating index!')

    def insert_single_data(self, data, id=False):
        if id:
            res = self.es.create(index=self.index_name, doc_type=self.index_type, id=id, body=data)
        else:
            res = self.es.index(index=self.index_name, doc_type=self.index_type, body=data)
        return res

    def updata_single_data(self, data, id=False):
        if id:
            res = self.es.update(index=self.index_name, doc_type=self.index_type, body=data, id=id)
        else:
            res = self.es.index(index=self.index_name, doc_type=self.index_type, body=data)
        return res

    def delete_index(self):
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])

    def fuzzy_match(self, query_filed_name, query_filed_value, get_filed, query_max_size=100):
        query_doc = {
            "query": {
                "match": {
                    query_filed_name: query_filed_value
                },
            },
            'size': query_max_size
        }
        re = self.es.search(index=self.index_name, doc_type=self.index_type, body=query_doc, request_timeout=60)
        re = re['hits']['hits']
        res = []
        for i in re:
            res.append(i['_source'][get_filed])
        return res

    def accurate_match(self, query_filed_name, query_filed_value):
        query_doc = {
            "query": {
                "term": {
                    query_filed_name + '.value': query_filed_value
                },
            },
        }
        re = self.es.search(index=self.index_name, doc_type=self.index_type, body=query_doc, request_timeout=60)
        re = re['hits']['hits']
        if len(re) != 1:
            return {}
        else:
            return re[0]['_source']

    def get_all_values_about_filed(self, filed, max_size=10000):
        query_doc = {
            "query": {
                "match_all": {},
            },
            'size': max_size
        }
        query_re = self.es.search(index=self.index_name, doc_type=self.index_type, body=query_doc, request_timeout=60)
        query_re = query_re['hits']['hits']
        res = []
        for i in query_re:
            res.append(i['_source'][filed])
        return res

    def index_data_from_xlsx(self, proj_df):
        '''
        从xlsx文件中读取数据，并存储到es中
        :param xlsx_file: xlsx文件，包括完整路径
        :return:
        '''
        df = proj_df[['project_name', 'ESID', 'province', 'text_b']]
        df.rename(columns={'ESID': 'es_id'}, inplace=True)
        datas = df.to_dict(orient='records')

        for data in datas:
            self.insert_single_data(data)


if __name__ == '__main__':
    # # obj = ESObject(index_name='zz', index_type='projects', port=9200)
    # obj = ESObject(index_name='zz_projects_0125', index_type='_doc', hosts='101.201.249.176', user_name='elastic',
    #                user_password='Zyl123123', port=9325)
    # # df = pd.read_excel('./data/v6/processed_data.xlsx', sheet_name='projects')
    # # df.fillna('', inplace=True)
    # # obj.index_data_from_xlsx(df)
    # re = obj.fuzzy_match(query_filed_name='project_name', query_filed_value='2021年上海药品集中采购项目（SH-DL2020-1）'
    #                                                                         ' 进行中 2021省级 上海', get_filed='text_b'
    #                      , query_max_size=100)
    # re2 = obj.get_all_values_about_filed(filed='text_b')
    # print(re)

    obj = ESObject(index_name='zz_projects_0708', index_type='_doc', hosts='101.201.249.176', user_name='elastic',
                   user_password='Zyl123123', port=9325)
    df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/zz/data/v1.7/processed_dt0708.xlsx", sheet_name='projects')
    df.fillna('', inplace=True)
    obj.index_data_from_xlsx(df)