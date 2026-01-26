# here put the import lib
import torch
import torch.nn as nn
from models.DualLLMSRS import DualLLMSASRec, DualLLMGRU4Rec, DualLLMBert4Rec
from models.utils import Contrastive_Loss2
from utils.utils import masked_mean



class LLMESR_SASRec(DualLLMSASRec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha  # Default: 0.1
        self.user_sim_func = args.user_sim_func  # Default: kd
        self.item_reg = args.item_reg  # Default: False

        if self.user_sim_func == "cl":  # Contrastive Learning
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":  # Knowledge Distillation
            self.align = nn.MSELoss()
        else:
            raise ValueError
        # args.hidden_size = 64
        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        
        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # Get the original loss: sum of positive and negative sample losses
        bs, sim_num, seq_len = kwargs["sim_seq"].shape
        # log_feats is V_{n_u+1} in the figure, seq.shape = torch.Size([128, 200])
        log_feats = self.log2feats(seq, positions)[:, -1, :]  # Dual-view Modeling on the left side of method figure, log_feats.shape=torch.Size([128, 128])
        # TODO: log_feats can add LLM user embedding

        # kwargs["sim_seq"].shape = (128, 10, 200), both below are torch.Size([1280, 200]). Here all similar user sequences are concatenated together, merge first, then compute, finally split
        # TODO: Can we add a (128, 10) index here to indicate which similar users are valid and not filtered?
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq_len), kwargs["sim_positions"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)  # (bs*sim_num,)

        # Only keep valid similar users
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        sim_seq_valid = sim_seq[valid_idx]
        sim_positions_valid = sim_positions[valid_idx]

        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)
        # Pass retrieved similar user sequences and positions to left side Dual-view modeling. If both seq and position are 0, output is also 0
        if len(valid_idx) > 0:
            valid_sim_log_feats = self.log2feats(sim_seq_valid, sim_positions_valid)[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_sim_log_feats


        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)  # (bs, sim_num, hidden_size)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)             # (bs, sim_num)

        # 5. Find valid samples in batch (at least one valid similar user)
        valid_sample_idx = (valid_mask_batch.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

        # 6. Calculate teacher vector mean & valid sample indices
        teacher_vec, valid_sample_idx = masked_mean(sim_log_feats_all, valid_mask_batch)

        
        align_loss = 0.0
        if len(valid_sample_idx) > 0:
            if self.user_sim_func in ["cl", "kd"]:
                align_loss = self.align(log_feats[valid_sample_idx], teacher_vec[valid_sample_idx])
            loss += self.alpha * align_loss

        if self.item_reg:  # Default is False
            unfold_item_id = torch.masked_select(seq, seq>0)  # Filter all values greater than 0 from sequence seq, these are considered valid item IDs
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))  # Two-layer linear transformation for dimensionality reduction
            id_item_emb = self.id_item_emb(unfold_item_id)  # self.id_item_emb is collaborative part embedding on the right, initialized from pca64 llm emb
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss


        return loss
    


class LLMESR_GRU4Rec(DualLLMGRU4Rec):

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

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        bs, sim_num, seq_len = kwargs["sim_seq"].shape

        log_feats = self.log2feats(seq)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq_len), kwargs["sim_positions"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)
        
        # Only keep valid similar users
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        sim_seq_valid = sim_seq[valid_idx]

        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)
        
        if len(valid_idx) > 0:
            valid_sim_log_feats = self.log2feats(sim_seq_valid)[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_sim_log_feats

        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)  # (bs, sim_num, hidden_size)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)     

        # 5. Find valid samples in batch (at least one valid similar user)
        valid_sample_idx = (valid_mask_batch.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

        # 6. Calculate teacher vector mean & valid sample indices
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



class LLMESR_Bert4Rec(DualLLMBert4Rec):

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

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        bs, sim_num, seq_len = kwargs["sim_seq"].shape

        log_feats = self.log2feats(seq, positions)[:, -1, :]
        
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq_len), kwargs["sim_positions"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)  # (bs*sim_num,)
        
        # Only keep valid similar users
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        sim_seq_valid = sim_seq[valid_idx]
        sim_positions_valid = sim_positions[valid_idx]

        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)
        # Pass retrieved similar user sequences and positions to left side Dual-view modeling. If both seq and position are 0, output is also 0
        if len(valid_idx) > 0:
            valid_sim_log_feats = self.log2feats(sim_seq_valid, sim_positions_valid)[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_sim_log_feats
        
        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)  # (bs, sim_num, hidden_size)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)  

        # 5. Find valid samples in batch (at least one valid similar user)
        valid_sample_idx = (valid_mask_batch.sum(dim=1) > 0).nonzero(as_tuple=True)[0]

        # 6. Calculate teacher vector mean & valid sample indices
        teacher_vec, valid_sample_idx = masked_mean(sim_log_feats_all, valid_mask_batch)


        align_loss = 0.0
        if len(valid_sample_idx) > 0:
            if self.user_sim_func in ["cl", "kd"]:
                align_loss = self.align(log_feats[valid_sample_idx], teacher_vec[valid_sample_idx])
            loss += self.alpha * align_loss

        if self.item_reg:  # Default is False
            unfold_item_id = torch.masked_select(seq, seq>0)  # Filter all values greater than 0 from sequence seq, these are considered valid item IDs
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))  # Two-layer linear transformation for dimensionality reduction
            id_item_emb = self.id_item_emb(unfold_item_id)  # self.id_item_emb is collaborative part embedding on the right, initialized from pca64 llm emb
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss


        return loss



