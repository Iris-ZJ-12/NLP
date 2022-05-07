import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig

from pharm_ai.gist.models.encoder import ExtTransformerEncoder


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.model = BertModel(BertConfig())

    def forward(self, x, segs, mask):
        top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):

    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert()

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)

        if args.max_pos > 512:
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = \
                self.bert.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(args.max_pos-512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint), strict=True)
        else:
            raise ValueError('No checkpoint provided!')

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
