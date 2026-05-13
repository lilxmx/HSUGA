"""
HSUGA Core Model: Dual-view LLM-enhanced Sequential Recommendation
with Hierarchical Semantic Understanding and Group-Aware Alignment.

Combines DualLLMSRS (dual-view backbone) and LLMESR (alignment loss).
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from hsuga.models.backbones.sasrec import SASRec_seq
from hsuga.models.backbones.gru4rec import GRU4Rec
from hsuga.models.backbones.bert4rec import Bert4Rec
from hsuga.models.modules import Multi_CrossAttention, Contrastive_Loss2
from hsuga.utils.misc import masked_mean


class _DualLLMSASRec(SASRec_seq):
    """Dual-view (collaborative + semantic) SASRec backbone."""

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = getattr(args, 'use_cross_att', True)
        self.use_llm2rec = getattr(args, 'use_llm2rec', False)

        if self.use_llm2rec:
            llm2rec_emb_path = getattr(args, 'llm2rec_emb_path', None) or \
                "LLM2rec/item_info/Qwen2-0.5B-LLM2Rec-IEM_step105_title_item_embs.npy"
            llm_item_emb = np.load(llm2rec_emb_path)
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        else:
            llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            self.adapter = nn.Sequential(
                nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
                nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
            )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
        self.id_item_emb.weight.requires_grad = True

        self.pos_emb = nn.Embedding(args.max_len+100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if getattr(args, 'freeze', False):
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    def _get_embedding(self, log_seqs):
        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.adapter(self.llm_item_emb(log_seqs))
        return torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

    def log2feats(self, log_seqs, positions):
        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.adapter(self.llm_item_emb(log_seqs))
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)
        return torch.cat([id_log_feats, llm_log_feats], dim=-1)


class _DualLLMGRU4Rec(GRU4Rec):
    """Dual-view (collaborative + semantic) GRU4Rec backbone."""

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.mask_token = item_num + 1
        self.num_heads = getattr(args, 'num_heads', 1)
        self.use_cross_att = getattr(args, 'use_cross_att', True)
        self.use_llm2rec = getattr(args, 'use_llm2rec', False)

        if self.use_llm2rec:
            llm2rec_emb_path = getattr(args, 'llm2rec_emb_path', None) or \
                "LLM2rec/item_info/Qwen2-0.5B-LLM2Rec-IEM_step105_title_item_embs.npy"
            llm_item_emb = np.load(llm2rec_emb_path)
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        else:
            llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            self.adapter = nn.Sequential(
                nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
                nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
            )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
        self.id_item_emb.weight.requires_grad = True

        self.pos_emb = nn.Embedding(args.max_len+100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if getattr(args, 'freeze', False):
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    def _get_embedding(self, log_seqs):
        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.adapter(self.llm_item_emb(log_seqs))
        return torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

    def log2feats(self, log_seqs):
        id_seqs = self.id_item_emb(log_seqs)
        llm_seqs = self.adapter(self.llm_item_emb(log_seqs))

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)
        return torch.cat([id_log_feats, llm_log_feats], dim=-1)


class _DualLLMBert4Rec(Bert4Rec):
    """Dual-view (collaborative + semantic) BERT4Rec backbone."""

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = getattr(args, 'use_cross_att', True)
        self.use_llm2rec = getattr(args, 'use_llm2rec', False)

        if self.use_llm2rec:
            llm2rec_emb_path = getattr(args, 'llm2rec_emb_path', None) or \
                "LLM2rec/item_info/Qwen2-0.5B-LLM2Rec-IEM_step105_title_item_embs.npy"
            llm_item_emb = np.load(llm2rec_emb_path)
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        else:
            llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            self.adapter = nn.Sequential(
                nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
                nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
            )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
        self.id_item_emb.weight.requires_grad = True

        self.pos_emb = nn.Embedding(args.max_len+100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if getattr(args, 'freeze', False):
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    def _get_embedding(self, log_seqs):
        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.adapter(self.llm_item_emb(log_seqs))
        return torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

    def log2feats(self, log_seqs, positions):
        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.adapter(self.llm_item_emb(log_seqs))
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)
        return torch.cat([id_log_feats, llm_log_feats], dim=-1)


# =============================================================================
# HSUGA models with Group-Aware Alignment (GAA) loss
# =============================================================================

class HSUGA_SASRec(_DualLLMSASRec):
    """HSUGA with SASRec backbone: dual-view modeling + self-distillation alignment."""

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.alpha = getattr(args, 'alpha', 0.1)
        self.user_sim_func = getattr(args, 'user_sim_func', 'kd')
        self.item_reg = getattr(args, 'item_reg', False)

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError(f"Unknown user_sim_func: {self.user_sim_func}")

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = getattr(args, 'beta', 0.1)
            self.reg = Contrastive_Loss2()

        self._init_weights()

    def forward(self, seq, pos, neg, positions, **kwargs):
        loss = super().forward(seq, pos, neg, positions, **kwargs)
        bs, sim_num, seq_len = kwargs["sim_seq"].shape

        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq = kwargs["sim_seq"].view(-1, seq_len)
        sim_positions = kwargs["sim_positions"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)

        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)

        if len(valid_idx) > 0:
            valid_feats = self.log2feats(sim_seq[valid_idx], sim_positions[valid_idx])[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_feats

        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)

        teacher_vec, valid_sample_idx = masked_mean(sim_log_feats_all, valid_mask_batch)

        if len(valid_sample_idx) > 0:
            align_loss = self.align(log_feats[valid_sample_idx], teacher_vec[valid_sample_idx])
            loss += self.alpha * align_loss

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq > 0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            loss += self.beta * self.reg(llm_item_emb, id_item_emb)

        return loss


class HSUGA_GRU4Rec(_DualLLMGRU4Rec):
    """HSUGA with GRU4Rec backbone: dual-view modeling + self-distillation alignment."""

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.alpha = getattr(args, 'alpha', 0.1)
        self.user_sim_func = getattr(args, 'user_sim_func', 'kd')
        self.item_reg = getattr(args, 'item_reg', False)

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError(f"Unknown user_sim_func: {self.user_sim_func}")

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = getattr(args, 'beta', 0.1)
            self.reg = Contrastive_Loss2()

        self._init_weights()

    def forward(self, seq, pos, neg, positions, **kwargs):
        loss = super().forward(seq, pos, neg, positions, **kwargs)
        bs, sim_num, seq_len = kwargs["sim_seq"].shape

        log_feats = self.log2feats(seq)[:, -1, :]
        sim_seq = kwargs["sim_seq"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)

        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)

        if len(valid_idx) > 0:
            valid_feats = self.log2feats(sim_seq[valid_idx])[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_feats

        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)

        teacher_vec, valid_sample_idx = masked_mean(sim_log_feats_all, valid_mask_batch)

        if len(valid_sample_idx) > 0:
            align_loss = self.align(log_feats[valid_sample_idx], teacher_vec[valid_sample_idx])
            loss += self.alpha * align_loss

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq > 0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            loss += self.beta * self.reg(llm_item_emb, id_item_emb)

        return loss


class HSUGA_Bert4Rec(_DualLLMBert4Rec):
    """HSUGA with BERT4Rec backbone: dual-view modeling + self-distillation alignment."""

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.alpha = getattr(args, 'alpha', 0.1)
        self.user_sim_func = getattr(args, 'user_sim_func', 'kd')
        self.item_reg = getattr(args, 'item_reg', False)

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError(f"Unknown user_sim_func: {self.user_sim_func}")

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = getattr(args, 'beta', 0.1)
            self.reg = Contrastive_Loss2()

        self._init_weights()

    def forward(self, seq, pos, neg, positions, **kwargs):
        loss = super().forward(seq, pos, neg, positions, **kwargs)
        bs, sim_num, seq_len = kwargs["sim_seq"].shape

        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq = kwargs["sim_seq"].view(-1, seq_len)
        sim_positions = kwargs["sim_positions"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)

        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)

        if len(valid_idx) > 0:
            valid_feats = self.log2feats(sim_seq[valid_idx], sim_positions[valid_idx])[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_feats

        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)

        teacher_vec, valid_sample_idx = masked_mean(sim_log_feats_all, valid_mask_batch)

        if len(valid_sample_idx) > 0:
            align_loss = self.align(log_feats[valid_sample_idx], teacher_vec[valid_sample_idx])
            loss += self.alpha * align_loss

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq > 0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            loss += self.beta * self.reg(llm_item_emb, id_item_emb)

        return loss
