import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from hsuga.models.backbones.base_model import BaseSeqModel
from hsuga.models.modules import PointWiseFeedForward


class BertBackbone(nn.Module):

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


class Bert4Rec(BaseSeqModel):

    def __init__(self, user_num, item_num, device, args):
        super(Bert4Rec, self).__init__(user_num, item_num, device, args)
        self.mask_token = item_num + 1

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
        self.backbone = BertBackbone(self.dev, args)
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
        mask_index = torch.where(pos > 0)
        log_feats = log_feats[mask_index]

        pos_embs = self._get_embedding(pos)[mask_index]
        neg_embs = self._get_embedding(neg)[mask_index]

        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1)

        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        pos_loss = self.loss_func(pos_logits, pos_labels)
        neg_loss = self.loss_func(neg_logits, neg_labels)

        return pos_loss + neg_loss

    def predict(self, seq, item_indices, positions, **kwargs):
        log_seqs = torch.cat([seq, self.mask_token * torch.ones(seq.shape[0], 1, device=self.dev)], dim=1)
        pred_position = positions[:, -1] + 1
        positions = torch.cat([positions, pred_position.unsqueeze(1)], dim=1)
        log_feats = self.log2feats(log_seqs[:, 1:].long(), positions[:, 1:].long())
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def get_user_emb(self, seq, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        return log_feats[:, -1, :]
