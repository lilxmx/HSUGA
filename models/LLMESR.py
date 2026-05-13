# here put the import lib
import torch
import torch.nn as nn
from models.DualLLMSRS import DualLLMSASRec, DualLLMGRU4Rec, DualLLMBert4Rec
from models.utils import Contrastive_Loss2
from models.HSUGatedFusion import HSUFusionMixin
from utils.utils import masked_mean



class LLMESR_SASRec(DualLLMSASRec, HSUFusionMixin):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        
        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        if getattr(args, 'use_hsu_fusion', False):
            orig_hidden = args.hidden_size
            args.hidden_size = 2 * orig_hidden  # LLMESR log_feats dim = 2*hidden_size
            self._init_hsu_fusion(args, device)
            args.hidden_size = orig_hidden

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)
        bs, sim_num, seq_len = kwargs["sim_seq"].shape

        log_feats = self.log2feats(seq, positions)[:, -1, :]

        # Apply HSU fusion if enabled
        if getattr(self, 'use_hsu_fusion', False) and 'user_id' in kwargs:
            log_feats = self._apply_hsu_fusion(log_feats, kwargs['user_id'])

        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq_len), kwargs["sim_positions"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)

        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        sim_seq_valid = sim_seq[valid_idx]
        sim_positions_valid = sim_positions[valid_idx]

        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)
        if len(valid_idx) > 0:
            valid_sim_log_feats = self.log2feats(sim_seq_valid, sim_positions_valid)[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_sim_log_feats

        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)

        valid_sample_idx = (valid_mask_batch.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        teacher_vec, valid_sample_idx = masked_mean(sim_log_feats_all, valid_mask_batch)

        align_loss = 0.0
        if len(valid_sample_idx) > 0:
            if self.user_sim_func in ["cl", "kd"]:
                align_loss = self.align(log_feats[valid_sample_idx], teacher_vec[valid_sample_idx])
            loss += self.alpha * align_loss

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        return loss
    


class LLMESR_GRU4Rec(DualLLMGRU4Rec, HSUFusionMixin):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        if getattr(args, 'use_hsu_fusion', False):
            orig_hidden = args.hidden_size
            args.hidden_size = 2 * orig_hidden
            self._init_hsu_fusion(args, device)
            args.hidden_size = orig_hidden

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)
        bs, sim_num, seq_len = kwargs["sim_seq"].shape

        log_feats = self.log2feats(seq)[:, -1, :]

        # Apply HSU fusion if enabled
        if getattr(self, 'use_hsu_fusion', False) and 'user_id' in kwargs:
            log_feats = self._apply_hsu_fusion(log_feats, kwargs['user_id'])

        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq_len), kwargs["sim_positions"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)
        
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        sim_seq_valid = sim_seq[valid_idx]

        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)
        
        if len(valid_idx) > 0:
            valid_sim_log_feats = self.log2feats(sim_seq_valid)[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_sim_log_feats

        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)

        valid_sample_idx = (valid_mask_batch.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        teacher_vec, valid_sample_idx = masked_mean(sim_log_feats_all, valid_mask_batch)

        align_loss = 0.0
        if len(valid_sample_idx) > 0:
            if self.user_sim_func in ["cl", "kd"]:
                align_loss = self.align(log_feats[valid_sample_idx], teacher_vec[valid_sample_idx])
            loss += self.alpha * align_loss

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        return loss



class LLMESR_Bert4Rec(DualLLMBert4Rec, HSUFusionMixin):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.reg = Contrastive_Loss2()

        if getattr(args, 'use_hsu_fusion', False):
            orig_hidden = args.hidden_size
            args.hidden_size = 2 * orig_hidden
            self._init_hsu_fusion(args, device)
            args.hidden_size = orig_hidden

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)
        bs, sim_num, seq_len = kwargs["sim_seq"].shape

        log_feats = self.log2feats(seq, positions)[:, -1, :]

        # Apply HSU fusion if enabled
        if getattr(self, 'use_hsu_fusion', False) and 'user_id' in kwargs:
            log_feats = self._apply_hsu_fusion(log_feats, kwargs['user_id'])

        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq_len), kwargs["sim_positions"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)

        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        sim_seq_valid = sim_seq[valid_idx]
        sim_positions_valid = sim_positions[valid_idx]

        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)
        if len(valid_idx) > 0:
            valid_sim_log_feats = self.log2feats(sim_seq_valid, sim_positions_valid)[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_sim_log_feats
        
        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)

        valid_sample_idx = (valid_mask_batch.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        teacher_vec, valid_sample_idx = masked_mean(sim_log_feats_all, valid_mask_batch)

        align_loss = 0.0
        if len(valid_sample_idx) > 0:
            if self.user_sim_func in ["cl", "kd"]:
                align_loss = self.align(log_feats[valid_sample_idx], teacher_vec[valid_sample_idx])
            loss += self.alpha * align_loss

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        return loss



