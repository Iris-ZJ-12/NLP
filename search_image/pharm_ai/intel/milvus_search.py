# encoding: utf-8
'''
@author: zyl
@file: milvus_search.py
@time: 2021/9/28 14:24
@desc:
'''
from pharm_ai.pom.milvus_client import MilvusHelper
import json

class DT:
    def __init__(self):
        pass

    def run(self):
        self.dt_0928()
        pass

    def test(self):
        pass

    def dt_0928(self):
        file = "/large_files/5T/mrc_data/ChineseSquad-master/squad_2.0/train-v2.0-zh.json"
        contexts = []
        with open(file, 'r') as load_f:
            load_dict = json.load(load_f)
            all_dt = load_dict['data']

            for a_d in all_dt:
                all_paragraphs = a_d.get('paragraphs')
                for a_p in all_paragraphs:
                    contexts.append(a_p.get('context'))
        return contexts
        # with open('/home/zyl/disk/PharmAI/pharm_ai/intel/data/mrc/squad_2_0_train_context.txt', 'w') as f:
        #     for c in contexts:
        #         f.write(str(c))
        #         f.write('\n')

class MilvusSearch:
    def __init__(self,contexts,cuda_device=1,collection_name='test',creat_new_collection=True,model_dim=768):
        self.cuda_device = cuda_device
        self.model = self.get_model()
        self.model_dim = model_dim


        # milvus
        if creat_new_collection:
            self.milvus = MilvusHelper(dimension=model_dim, clear_collection=True, collection=collection_name)

            vecs = self.model.encode(contexts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
            milvusids = self.milvus.insert(vecs,ids=list(range(len(contexts))))
            self.milvus.create_index()
            self.id_dict = {i: j for i, j in zip(milvusids, contexts)}
        else:
            self.milvus = MilvusHelper(dimension=model_dim, clear_collection=False, collection=collection_name)
            self.id_dict = {i: j for i, j in zip(list(range(len(contexts))), contexts)}
        pass

    def run(self):
        pass

    def test(self):
        pass

    def get_model(self):
        from sentence_transformers import SentenceTransformer

        # sbert = SentenceTransformer('msmarco-distilbert-base-tas-b', device='cuda:1')
        # sbert = SentenceTransformer("distiluse-base-multilingual-cased",device='cuda:1')
        m = "/home/zyl/disk/PharmAI/pharm_ai/panel/entry_match/best_model/em9/"
        sbert = SentenceTransformer(m, device=f'cuda:{self.cuda_device}')
        return sbert

    def get_search_ids(self,to_predict:list):
        vecs2 = self.model.encode(to_predict, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

        res_ids = self.milvus.search(top_k=10, query=vecs2.tolist())

        results = []
        for r in res_ids:
            results.append([self.id_dict.get(each_entry.id) for each_entry in r])
        return results



if __name__ == '__main__':
    # model_dim = 768
    # from sentence_transformers import SentenceTransformer
    contexts = DT().dt_0928()
    to_predict = ["命运的孩子什么时候发行了他们的第二张专辑？","命运之子的第一首主打歌是什么？"]
    print(MilvusSearch(contexts=contexts,creat_new_collection=False).get_search_ids(to_predict))


    print('end')