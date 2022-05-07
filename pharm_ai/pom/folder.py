import os

from uuid import uuid4
from typing import List, Dict, Tuple, Set, Optional
from milvus.client.abstract import TopKQueryResult
from sentence_transformers import SentenceTransformer
from pharm_ai.pom.milvus_client import MilvusHelper
from pharm_ai.pom.es_utils import ESNews, ESCluster
from pharm_ai.config import ConfigFilePaths


WEEK = 604800 * 1000
MONTH = 2629743 * 1000


class Folder:

    def __init__(self):
        model_path = os.path.join(ConfigFilePaths.project_dir, 'pom', 'mpnet_simcse_supervised')
        self.sbert = SentenceTransformer(model_path, device='cpu')
        self.milvus = MilvusHelper(dimension=768, clear_collection=False)
        self.es_news = ESNews()
        self.es_cluster = ESCluster()

    def filter(
            self,
            results: TopKQueryResult,
            timestamp: int,
            score_threshold: float
    ) -> List[Tuple[Dict, float]]:
        """
        Inputs: secondary milvus result and others
        Returns: matched docs and scores
        """
        matched = []
        for res in results:
            score = res.distance
            if score < score_threshold:
                break
            r = self.es_news.exact_match('milvusid', res.id)
            if len(r) != 1:
                raise ValueError(f'Found {len(r)} news match milvusid {res.id}')
            candidate, _ = r[0]
            cts = candidate['publish_time']
            if (not timestamp) or (cts and timestamp - WEEK < cts < timestamp + WEEK):
                matched.append((candidate, res.distance))
        return matched

    def update_cluster(self, news: Dict, clusterid: int, add: bool = True) -> Optional[str]:
        """
        Update news to corresponding cluster
        Either add to or remove from cluster
        """
        record, rid = self.es_cluster.exact_match_one('clusterid', clusterid)
        cluster_content = record['cluster']
        if add:
            cluster_content = set(cluster_content)
            # if the news to be added is already in cluster, ignore this operation
            if news['id'] in cluster_content:
                return
            cluster_content.add(news['id'])
            cluster_content = list(cluster_content)
        else:
            cluster_content.remove(news['id'])
        body = {
            "doc": {
                "cluster": cluster_content
            }
        }
        self.es_cluster.update_single_data(body, id=rid)
        return rid

    def check_and_add(self, news: Dict, score: float, clusterid: int) -> Set[str]:
        """
        Checks if cluster needs to be changed
        """
        modified = set()
        if 'cluster' in news:
            if score > news['score']:
                # remove news from original cluster
                org_cluster = news['clusterid']
                modified.add(self.update_cluster(news, org_cluster, add=False))
                news['score'] = score
            else:
                return modified
        else:
            news['score'] = score
        news['clusterid'] = clusterid
        self.es_news.upload_single_data(news, news['id'])
        # add news to cluster
        c = self.update_cluster(news, clusterid, add=True)
        if c:
            modified.add(c)
        return modified

    def add_to_cluster(
            self,
            news: Dict,
            matched_news: List[Dict],
            scores: List[float],
            clusterid: int
    ) -> Set[str]:
        score = 0 if len(scores) == 0 else max(scores)
        modified = set()
        modified = modified.union(self.check_and_add(news, score, clusterid))
        for match, score in zip(matched_news, scores):
            modified = modified.union(self.check_and_add(match, score, clusterid))
        return modified

    def __call__(self, news: Dict, threshold: float) -> Dict:
        r = self.es_news.exact_match('id', news['id'])
        if len(r) == 1:
            return {'status': 'ignored'}
        elif len(r) > 1:
            return {'status': 'error'}

        # encode title into vec
        vec = self.sbert.encode([news['title'].replace(' ', '')], normalize_embeddings=True)

        # find similar titles
        results = self.milvus.search(top_k=100, query=vec)[0]
        matched = self.filter(results, news['publish_time'], threshold)
        milvusid = self.milvus.insert(vec)[0]
        news['milvusid'] = milvusid
        if not matched:
            matched_news = []
            scores = []
        else:
            matched_news, scores = list(zip(*matched))

        # find clusters and insert
        clusterid = matched_news[0]['clusterid'] if matched_news else None
        modified = set()
        if not clusterid:
            clusterid = str(uuid4())
            d = {
                'clusterid': clusterid,
                'cluster': [news['id']]
            }
            self.es_cluster.upload_single_data(d, clusterid)
            modified.add(clusterid)
        modified = modified.union(self.add_to_cluster(news, matched_news, scores, clusterid))
        return {'status': 'success', 'clusterid': list(modified)}
