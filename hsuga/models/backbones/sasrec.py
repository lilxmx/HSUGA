import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from hsuga.models.backbones.base_model import BaseSeqModel
from hsuga.models.modules import PointWiseFeedForward


class SASRecBackbone(nn.Module):

    def __init__(self, device, args) -> None:
        super().__init__()
        self.dev = device
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.hidden_size, eps=1e-8)

        for _ in range(args.trm_num):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_size, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(args.hidden_size, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_size, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_size, args.dropout_rate))

    def forward(self, seqs, log_seqs):
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats


class SASRec(BaseSeqModel):

    def __init__(self, user_num, item_num, device, args):
        super(SASRec, self).__init__(user_num, item_num, device, args)

        if getattr(args, 'item_from_llm', False):
            id_item_emb = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
            id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
            id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
            self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
            self.item_emb.weight.requires_grad = True
        else:
            self.item_emb = nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)

        self.pos_emb = nn.Embedding(args.max_len+100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)
        self.loss_func = nn.BCEWithLogitsLoss()
        self._init_weights()

    def _get_embedding(self, log_seqs):
        return self.item_emb(log_seqs)

    def log2feats(self, log_seqs, positions):
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats

    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
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
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def get_user_emb(self, seq, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        return log_feats[:, -1, :]


class SASRec_seq(SASRec):

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)

    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
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
