# encoding: utf-8
'''
@author: zyl
@file: milvus_util.py
@time: 2021/8/11 14:09
@desc:
'''
import time

from milvus import Milvus, IndexType, MetricType


class MilvusHelper:

    def __init__(
            self,
            collection='t1',
            dimension=768,
            index_file_size=1024,
            metric_type=MetricType.IP,
            clear_collection=False,
            host='101.201.249.176',
            port='19530',
    ):
        _HOST = host
        _PORT = port
        self.dim = dimension

        self.milvus = Milvus(_HOST, _PORT)
        self.collection = collection
        status, ok = self.milvus.has_collection(collection)
        print(status)
        if clear_collection and ok:
            self.drop_collection()
        if not ok or clear_collection:
            self.create_collection(index_file_size, metric_type)

    def create_collection(self, index_file_size, metric_type):
        param = {
            'collection_name': self.collection,
            'dimension': self.dim,
            'index_file_size': index_file_size,  # optional
            'metric_type': metric_type  # optional
        }
        status = self.milvus.create_collection(param)
        print(f'create collection {self.collection} with size {self.dim} {status}')

    def list_collection(self):
        status, collections = self.milvus.list_collections()
        print(f'list collection {status}')
        return collections

    def drop_collection(self):
        status = self.milvus.drop_collection(self.collection)
        print(f'drop collection {status}')
        time.sleep(3)

    def count(self):
        status, num = self.milvus.count_entities(collection_name=self.collection)
        print(f'count collection {status}')
        return num

    def insert(self, records, ids=None, partition_tag=None):
        status, ids = self.milvus.insert(self.collection, records, ids, partition_tag)
        print(f'insert {len(records)} records {status}')
        return ids

    def get_entity_by_ids(self, ids, partition_tag=None):
        status, vec = self.milvus.get_entity_by_id(self.collection, ids, partition_tag=partition_tag)
        print(f'get entity {status}')
        return vec

    def delete_entity_by_ids(self, ids, partition_tag=None):
        status = self.milvus.delete_entity_by_id(self.collection, ids, partition_tag=partition_tag)
        print(f'delete entity {status}')

    def create_index(self, index_type=IndexType.IVFLAT, params=None):
        if params is None:
            params = {'nlist': 1024}
        status = self.milvus.create_index(self.collection, index_type=index_type, params=params)
        print(f'create index {status}')

    def search(self, top_k, query, partition_tag=None, params=None, verbose=False):
        if params is None:
            params = {'nprobe': 8}
        status, result = self.milvus.search(self.collection, top_k, query, partition_tag, params)
        if verbose:
            print(f'search {status}')
        return result

# if __name__ == '__main__':
#     import numpy as np
#     from sklearn.preprocessing import normalize
#
#     dim = 768
#     num = 10000
#     data = normalize(np.random.rand(num, dim))
#     m = MilvusHelper(clear_collection=True)
#     print(m.list_collection())
#     ids = m.insert(data, ids=list(range(num)))
#     m.create_index()
#     print(m.count())
