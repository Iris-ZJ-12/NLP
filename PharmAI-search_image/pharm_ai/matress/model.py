import os
import warnings

from typing import List
from harvesttext import HarvestText
from pharm_ai.matress.easynmt import EasyNMT
from pharm_ai.config import ConfigFilePaths

warnings.simplefilter('ignore', UserWarning)


class Translator:

    def __init__(self):
        trans_model_path = os.path.join(ConfigFilePaths.project_dir, 'matress', 'opus-mt-zh-en')
        self.translator = EasyNMT(trans_model_path, device='cuda', max_length=256)
        self.ht = HarvestText()

    def translate_sentences(self, source_text: List[str]) -> List[str]:
        en_text = self.translator.translate(
            source_text,
            source_lang='zh',
            target_lang='en',
            perform_sentence_splitting=False,
            beam_size=3,
            batch_size=12,
            max_length=256
        )
        return en_text

    def translate(self, doc: str) -> str:
        sents = self.ht.cut_sentences(doc)
        en_sents = self.translate_sentences(sents)
        en_doc = " ".join(en_sents)
        return en_doc
