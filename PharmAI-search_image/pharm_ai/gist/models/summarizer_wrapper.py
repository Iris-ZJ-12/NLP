import os
import torch
import argparse
import numpy as np

from pharm_ai.config import ConfigFilePaths
from pharm_ai.gist.models.model_builder import ExtSummarizer
from pharm_ai.gist.models.prepro import DataPre


class SummarizerWrapper:

    def __init__(self, device):
        args = self.get_args()
        checkpoint = args.test_from
        self.model = ExtSummarizer(args, device, checkpoint)
        self.pre = DataPre(args)
        self.model.eval()
        self.device = device

    @staticmethod
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(self, c, p):
        tri_c = self._get_ngrams(3, c.split())
        for s in p:
            tri_s = self._get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    @torch.no_grad()
    def predict(self, to_predict, top_k=3, block_trigram=False):
        test_iter = self.pre.prepare_dataloader(to_predict, self.device)
        pred = []
        pred_idx = []
        for batch in test_iter:
            src = batch.src
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src
            mask_cls = batch.mask_cls

            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()
            selected_ids = np.argsort(-sent_scores, 1)

            for i, idx in enumerate(selected_ids):
                _pred = []
                _pred_idx = []
                if len(batch.src_str[i]) == 0:
                    continue
                for j in selected_ids[i][:len(batch.src_str[i])]:
                    if j >= len(batch.src_str[i]):
                        continue
                    candidate = batch.src_str[i][j].strip()
                    if block_trigram:
                        if not self._block_tri(candidate, _pred):
                            _pred.append(candidate)
                            _pred_idx.append(j)
                    else:
                        _pred.append(candidate)
                        _pred_idx.append(j)

                    if len(_pred) == top_k:
                        break
                pred.append(_pred)
                pred_idx.append(_pred_idx)
        return pred_idx, pred

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-max_pos", default=512, type=int)

        # params for EXT
        parser.add_argument("-ext_dropout", default=0.2, type=float)
        parser.add_argument("-ext_layers", default=2, type=int)
        parser.add_argument("-ext_hidden_size", default=768, type=int)
        parser.add_argument("-ext_heads", default=8, type=int)
        parser.add_argument("-ext_ff_size", default=2048, type=int)

        model_path = os.path.join(ConfigFilePaths.project_dir,
                                  'gist',
                                  'bertext_cnndm_transformer',
                                  'pytorch.pt')
        parser.add_argument("-test_from", default=model_path)
        args = parser.parse_args()
        return args
