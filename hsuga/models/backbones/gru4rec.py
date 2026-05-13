import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from hsuga.models.backbones.base_model import BaseSeqModel


class GRU4RecBackbone(nn.Module):

    def __init__(self, device, args) -> None:
        super().__init__()
        self.dev = device
        self.gru = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bias=False,
            batch_first=True
        )

    def forward(self, seqs, log_seqs):
        log_feats, _ = self.gru(seqs)
        return log_feats


class GRU4Rec(BaseSeqModel):

    def __init__(self, user_num, item_num, device, args) -> None:
        super(GRU4Rec, self).__init__(user_num, item_num, device, args)
        self.dev = device

        if getattr(args, 'item_from_llm', False):
            id_item_emb = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
            id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
            id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
            self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
            self.item_emb.weight.requires_grad = True
        else:
            self.item_emb = nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)

        self.backbone = GRU4RecBackbone(device, args)
        self.loss_func = nn.BCEWithLogitsLoss()
        self._init_weights()

    def _get_embedding(self, log_seqs):
        return self.item_emb(log_seqs)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats

    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq)
        log_feats = log_feats[:, -1, :].unsqueeze(1)

        pos_embs = self._get_embedding(pos.unsqueeze(1))
        neg_embs = self._get_embedding(neg)

        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1)
        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)

        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])

        return pos_loss + neg_loss

    def predict(self, seq, item_indices, positions, **kwargs):
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def get_user_emb(self, seq, **kwargs):
        log_feats = self.log2feats(seq)
        return log_feats[:, -1, :]


class GRU4Rec_seq(GRU4Rec):

    def __init__(self, user_num, item_num, device, args) -> None:
        super().__init__(user_num, item_num, device, args)

    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq)
        pos_embs = self._get_embedding(pos)
        neg_embs = self._get_embedding(neg)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])

        return pos_loss + neg_loss
