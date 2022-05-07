import torch

from pytorch_transformers import BertTokenizer


class DataPre:

    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.args = args

    def _process_src(self, device, text):
        raw = text.strip().lower()
        raw = raw.replace('[cls]', '[CLS]').replace('[sep]', '[SEP]')
        src_subtokens = self.tokenizer.tokenize(raw)
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:self.args.max_pos]
        src_subtoken_idxs[-1] = self.sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        segs = segs[:self.args.max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0

        return src, mask_src, segments_ids, clss, mask_cls

    def prepare_dataloader(self, inputs, device):
        """
        Prepare model input data
        Args:
            inputs: List[List[str]], list of documents that split into sentences
            device: cpu or gpu to run model on

        Returns:
            dataloader
        """
        tokens_to_add = ' [CLS] [SEP] '
        # truncate inputs here to avoid maximum length longer than 512
        for i in range(len(inputs)):
            while len(tokens_to_add.join(inputs[i]).split(" ")) > 400:
                inputs[i].pop()
        processed = [tokens_to_add.join(doc) for doc in inputs]
        for x in processed:
            src, mask_src, segments_ids, clss, mask_cls = self._process_src(device, x)
            segs = torch.tensor(segments_ids)[None, :].to(device)
            batch = Batch()
            batch.src = src
            batch.tgt = None
            batch.mask_src = mask_src
            batch.mask_tgt = None
            batch.segs = segs
            batch.src_str = [[sent.replace('[SEP]', '').strip() for sent in x.split('[CLS]')]]
            batch.tgt_str = ['']
            batch.clss = clss
            batch.mask_cls = mask_cls

            batch.batch_size = 1
            yield batch


class Batch(object):

    def _pad(self, data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_src_sent_labels = [x[4] for x in data]

            src = torch.tensor(self._pad(pre_src, 0))
            tgt = torch.tensor(self._pad(pre_tgt, 0))

            segs = torch.tensor(self._pad(pre_segs, 0))
            mask_src = 1 - (src == 0)
            mask_tgt = 1 - (tgt == 0)

            clss = torch.tensor(self._pad(pre_clss, -1))
            src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
            mask_cls = 1 - (clss == -1)
            clss[clss == -1] = 0
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src_sent_labels', src_sent_labels.to(device))

            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))

            if is_test:
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size
