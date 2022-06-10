# encoding: utf-8
'''
@author: zyl
@file: t.py
@time: 2021/10/18 9:15
@desc:
'''
if __name__ == '__main__':

    from haystack import Pipeline

    from pathlib import Path
    PIPELINE = Pipeline.load_from_yaml(Path("/home/zyl/disk/PharmAI/pharm_ai/haystack_test/haystack-master/rest_api/pipeline/pipelines.yaml"),
                                       pipeline_name='query')
    RETRIEVER = PIPELINE.get_node(name="Retriever")
    print(RETRIEVER.document_store.get_document_count())