"""
Pipeline:
(maybe pull from es)
clean crawled doc in zh, cutted into sentences
=====>
translate and map zh to en
=====>
run en summarizer, get top sentences ids and map back to zh
"""  # give some color
import requests

from typing import List
from pharm_ai.gist.models.summarizer_wrapper import SummarizerWrapper


class Pipeline:

    def __init__(self, device: str):
        self.trans_url = 'http://localhost:4399/translate_sentences/'
        self.summarizer = SummarizerWrapper(device)

    def translate(self, source_text: List[str]) -> List[str]:
        re = requests.post(self.trans_url, json={'sents': source_text})
        eng_text = re.json()['sents']
        return eng_text

    def run_pipeline(self, source_text: List[str], top_k: int = 3) -> List[str]:
        """
        Pipeline runner
        Args:
            source_text: document cutted sentences
            top_k: no. sentences to include in summary
        Returns:
            summary as list of strings
        """
        if not source_text:
            return []
        en_text = self.translate(source_text)
        idxs, _ = self.summarizer.predict([en_text], top_k=top_k)
        idxs = sorted(idxs[0])
        summary = [source_text[i] for i in idxs]
        return summary

    def __call__(self, source_text: List[str], top_k: int = 3) -> List[str]:
        return self.run_pipeline(source_text, top_k)
