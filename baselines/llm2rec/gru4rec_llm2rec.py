"""
Lightweight GRU4Rec + LLM2Rec integration
- GRU4Rec_LLM2Rec: Pure GRU4Rec backbone + LLM2Rec semantic insertion + adapter
- GRU4Rec_WithAlignment: Add similar user contrastive enhancement on top of GRU4Rec_LLM2Rec (reuse GAA_Mixin)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from models.GRU4Rec import GRU4Rec
from models.SASRec_LLM2Rec import get_llm2rec_emb_path, validate_item_embedding
from models.LLMEmb_GAA import GAA_Mixin


class GRU4Rec_LLM2Rec(GRU4Rec):
    """
    Pure GRU4Rec + LLM2Rec semantic insertion + adapter
    Does not include dual-view, cross attention and other additional components
    """
    
    def __init__(self, user_num, item_num, device, args):
        # Call parent class initialization first (will initialize item_emb)
        super().__init__(user_num, item_num, device, args)
        
        self.use_llm2rec = args.use_llm2rec if hasattr(args, 'use_llm2rec') else False
        
        if self.use_llm2rec:
            # Freeze item_emb inherited from parent class (not used in LLM2Rec mode)
            self.item_emb.weight.requires_grad = False
            
            # Dynamically get LLM2Rec embedding path
            llm2rec_emb_path = get_llm2rec_emb_path(args)
            print(f"[LLM2Rec] Loading item embeddings from: {llm2rec_emb_path}")
            llm_item_emb = np.load(llm2rec_emb_path)
            
            # Validate item count
            validate_item_embedding(llm_item_emb, item_num, args.dataset)
            original_shape = llm_item_emb.shape
            
            # Add padding (index 0) and mask token (index item_num+1)
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            
            print(f"[LLM2Rec] Embedding shape: {original_shape} -> {llm_item_emb.shape} (added padding & mask)")
            print(f"[LLM2Rec] Index mapping: 0=padding, 1~{item_num}=items, {item_num+1}=mask")
            
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            
            # LLM2Rec adapter: lightweight linear projection
            self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
            
            # Freeze LLM embedding (if needed)
            if args.freeze:
                self.llm_item_emb.weight.requires_grad = False
            
            # 初始化 adapter 权重
            nn.init.xavier_normal_(self.adapter.weight)
            if self.adapter.bias is not None:
                nn.init.constant_(self.adapter.bias, 0)
    
    
    def _get_embedding(self, log_seqs):
        """
        Get item embedding
        
        Strictly follows LLM2Rec paper setting:
        - When use_llm2rec=True, only use adapter(llm_item_emb) as item representation
        - item_emb is not involved (although inherited from parent class, not used)
        """
        if self.use_llm2rec:
            # Only use LLM2Rec semantic embedding (consistent with paper setting)
            llm_emb = self.llm_item_emb(log_seqs)
            llm_emb = self.adapter(llm_emb)
            return llm_emb
        else:
            # When not using LLM2Rec, use original ID embedding
            return self.item_emb(log_seqs)


class GRU4Rec_WithAlignment(GRU4Rec_LLM2Rec, GAA_Mixin):
    """
    Add similar user contrastive enhancement on top of GRU4Rec_LLM2Rec (reuse GAA_Mixin)
    """
    
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        # Use alpha parameter name (LLM2rec style)
        self._init_gaa(args, use_alpha_param=True)
        self._user_num = user_num  # Save user count for bank initialization
    
    def _log2feats_for_gaa(self, log_seqs, positions):
        """log2feats interface provided for GAA module (GRU4Rec doesn't use positions)"""
        return self.log2feats(log_seqs)
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        """Forward pass: base loss + similar user contrastive enhancement loss"""
        # 1. Calculate base loss (from GRU4Rec_LLM2Rec)
        base_loss = super(GRU4Rec_WithAlignment, self).forward(seq, pos, neg, positions, **kwargs)
        
        # 2. 如果 gaa_alpha=0，直接返回基础损失
        if self.gaa_alpha == 0:
            return base_loss
        
        # 3. 计算用户表示
        user_feats = self.log2feats(seq)[:, -1, :]  # [bs, hidden_size] - GRU4Rec 不需要 positions
        
        # 4. 懒加载初始化 + 更新 user_bank
        if self.gaa_use_user_bank:
            self._init_user_bank(self._user_num, user_feats.size(-1), user_feats.device)
            if 'user_ids' in kwargs:
                self._update_user_bank(kwargs['user_ids'], user_feats)
        
        # 5. 检查是否有相似用户数据
        has_sim_data = ('sim_seq' in kwargs and 'valid_mask' in kwargs) or \
                       ('sim_user_ids' in kwargs and 'valid_mask' in kwargs)
        if not has_sim_data:
            return base_loss
        
        # 6. 计算 GAA 损失 (复用 GAA_Mixin)
        gaa_loss = self._compute_gaa_loss(user_feats, kwargs)
        
        return base_loss + self.gaa_alpha * gaa_loss

