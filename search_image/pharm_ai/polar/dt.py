# encoding: utf-8
'''
@author: zyl
@file: dt.py
@time: 2021/7/22 上午12:13
@desc:
'''
import pandas as pd
from sklearn.utils import resample
from pharm_ai.util.utils import Utilfuncs
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
import html


class PolarDT:
    def __init__(self):
        pass

    def run(self):
        # self.dt_0722()
        self.test()

    @staticmethod
    def cut_train_eval(df):
        raw_df = resample(df, replace=False)
        cut_point = min(8000, int(0.2 * len(raw_df)))
        eval_df = raw_df[0:cut_point]
        train_df = raw_df[cut_point:]
        return train_df, eval_df

    @staticmethod
    def up_sampling(train_df):
        negative_df = train_df[train_df['labels'] == 0]
        neg_len = negative_df.shape[0]
        positive_df = train_df[train_df['labels'] == 1]
        pos_len = positive_df.shape[0]
        if neg_len > pos_len:
            up_sampling_df = resample(positive_df, replace=True, n_samples=(neg_len - pos_len), random_state=3242)
            return pd.concat([train_df, up_sampling_df], ignore_index=True)
        elif neg_len < pos_len:
            up_sampling_df = resample(negative_df, replace=True, n_samples=(pos_len - neg_len), random_state=3242)
            return pd.concat([train_df, up_sampling_df], ignore_index=True)
        else:
            return train_df

    @staticmethod
    def analyze_df(dt_df: pd.DataFrame):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        print('length:', len(dt_df))
        print(dt_df['labels'].value_counts())

        print(dt_df.describe())
        print(dt_df.info())
        print(dt_df.head())
        print(dt_df['labels'].value_counts())



    def dt_0722(self):
        self.version = 'v0'
        df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/polar/data/20210722-政策库-数据标注.xlsx")
        df.rename(columns={'标题': 'text', 'filter': 'labels'}, inplace=True)
        df['text'] = df['text'].apply(PolarDT.clean_text)
        # PolarDT.analyze_df(df)
        train_df, eval_df = PolarDT.cut_train_eval(df)
        # PolarDT.analyze_df(train_df)
        Utilfuncs.to_excel(df, './data/processed_dt_0722.xlsx', 'all')
        Utilfuncs.to_excel(train_df, './data/processed_dt_0722.xlsx', 'train')
        Utilfuncs.to_excel(eval_df, './data/processed_dt_0722.xlsx', 'eval')
        train_df = PolarDT.up_sampling(train_df)
        Utilfuncs.to_excel(train_df, './data/processed_dt_0722.xlsx', 'train_up')

    @staticmethod
    def clean_text(text: str):
        text = ILLEGAL_CHARACTERS_RE.sub(r'', str(text))
        text = html.unescape(text)
        replaced_chars = ['\u200b', '\ufeff', '\ue601', '\ue317', '\n', '\t', '\ue000', '\ue005']
        for i in replaced_chars:
            if i in text:
                text = text.replace(i, '')
        if '&middot;' in text:
            text = text.replace('&middot;', '·')
        text = ' '.join(text.split())
        text = text.strip()
        return text

    def test(self):
        df = pd.read_excel('./data/processed_dt_0722.xlsx', 'all')
        PolarDT.analyze_df(df)
        df = pd.read_excel('./data/processed_dt_0722.xlsx', 'train')
        PolarDT.analyze_df(df)
        df = pd.read_excel('./data/processed_dt_0722.xlsx', 'eval')
        PolarDT.analyze_df(df)

# if __name__ == '__main__':
#     PolarDT().run()

import pandas as pd


class Haystack:
    def __init__(self):
        pass

    def run(self):
        self.test()

    def get_document(self, init_db=False):
        from haystack.document_store import MilvusDocumentStore

        documents = MilvusDocumentStore(sql_url="sqlite:////home/zyl/disk/PharmAI/pharm_ai/panel/mydb.db",
                                        milvus_url="tcp://101.201.249.176:19530",
                                        index='test_haystack',
                                        return_embedding=False,
                                        embedding_field='embedding')

        if init_db:
            df = pd.read_excel("/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/data/v2/dt_0930.xlsx", "di_dict")
            library = list(set(df['entry'].tolist()))
            docs = []
            id = 0
            for l in library:
                id += 1
                docs.append({
                    'text': str(l),
                    'meta': {'text_id': id, 'embedding': str(l)}
                })

            documents.write_documents(docs)

        return documents

    def get_retriver(self, documents):
        from haystack.retriever import EmbeddingRetriever
        retriver_model = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/v2/v2.1.0/"
        retriever = EmbeddingRetriever(document_store=documents,
                                       embedding_model=retriver_model,
                                       model_format='sentence_transformers',
                                       )
        return retriever

    def get_reader(self):
        from haystack.reader import FARMReader
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True, num_processes=0)

        return reader

    def get_pipeline(self, init_db=False):
        documents = self.get_document(init_db)
        print(f'document_num: {documents.get_document_count()}')
        retriever = self.get_retriver(documents)
        reader = self.get_reader()
        if init_db:
            documents.update_embeddings(retriever, batch_size=64)

        from haystack.pipeline import ExtractiveQAPipeline
        pipeline = ExtractiveQAPipeline(reader, retriever)

        return pipeline

    def test(self):
        p = self.get_pipeline(init_db=False)
        prediction = p.run(query="癌症")
        print(prediction)


    def test2(self):
        from haystack import Pipeline

        from pathlib import Path

        PIPELINE = Pipeline.load_from_yaml(
            Path("/home/zyl/disk/PharmAI/pharm_ai/haystack_test/haystack-master/rest_api/pipeline/pipelines.yaml"),
            pipeline_name='query')
        RETRIEVER = PIPELINE.get_node(name="Retriever")
        print(RETRIEVER.document_store.get_document_count())


if __name__ == '__main__':
    Haystack().run()