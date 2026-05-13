"""
Lightweight SASRec + LLM2Rec integration
- SASRec_LLM2Rec: Pure SASRec backbone + LLM2Rec semantic insertion + adapter
- SASRec_WithAlignment: Add similar user contrastive enhancement on top of SASRec_LLM2Rec (reuse GAA_Mixin)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from models.SASRec import SASRec_seq
from models.LLMEmb_GAA import GAA_Mixin


# Dataset-specific LLM2Rec embedding filename mapping (fixed filenames per dataset)
LLM2REC_EMB_FILES = {
    "steam": "LLM2rec/item_info/Steam/Qwen2-0.5B-LLM2Rec-IEM_step105_title_item_embs.npy",
    "fashion": "LLM2rec/item_info/Fashion/Qwen2-0.5B-LLM2Rec-IEM_step190_title_item_embs.npy",
    "beauty": "LLM2rec/item_info/Beauty/Qwen2-0.5B-LLM2Rec-IEM_step1000_title_item_embs.npy",
}

# Item counts for each dataset (for validation, excluding padding/mask)
DATASET_ITEM_COUNTS = {
    "steam": 5237,
    "fashion": 4722,
    "beauty": 57289,
}


def get_llm2rec_emb_path(args):
    """Dynamically get LLM2Rec embedding path based on dataset"""
    # Prefer user-specified path
    if hasattr(args, 'llm2rec_emb_path') and args.llm2rec_emb_path:
        return args.llm2rec_emb_path
    
    # Auto-select based on dataset
    dataset = args.dataset.lower()
    if dataset in LLM2REC_EMB_FILES:
        return LLM2REC_EMB_FILES[dataset]
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: {list(LLM2REC_EMB_FILES.keys())}")


def validate_item_embedding(llm_item_emb, item_num, dataset):
    """
    Validate whether embedding dimensions match dataset item count
    
    Note:
    - llm_item_emb: Original loaded embedding, shape = (real_item_count, emb_dim)
    - item_num: Maximum item ID read from generator (= actual item count)
    - Backbone's item_emb size is item_num + 2 (including padding and mask token)
    - So llm_item_emb should have item_num rows, becomes item_num+2 after adding padding/mask
    """
    emb_item_count = llm_item_emb.shape[0]
    expected_count = DATASET_ITEM_COUNTS.get(dataset.lower(), None)
    
    print(f"[LLM2Rec] Validation:")
    print(f"  - Embedding file items: {emb_item_count}")
    print(f"  - Expected items (from config): {expected_count}")
    print(f"  - Model item_num (from data): {item_num}")
    print(f"  - Backbone item_emb size will be: {item_num + 2} (with padding & mask)")
    
    # Validate whether embedding row count matches expected config
    if expected_count is not None and emb_item_count != expected_count:
        raise ValueError(
            f"[LLM2Rec] Item embedding mismatch for dataset '{dataset}'!\n"
            f"  Embedding file has {emb_item_count} items, but config expects {expected_count}.\n"
            f"  Please check the embedding file."
        )
    
    # Validate whether embedding row count matches model item_num
    if emb_item_count != item_num:
        print(f"[LLM2Rec WARNING] Embedding items ({emb_item_count}) != model item_num ({item_num})!")
        print(f"  This may cause index out of bounds errors.")
        print(f"  Possible reasons:")
        print(f"    1. Data has item IDs > {emb_item_count}")
        print(f"    2. Embedding file is for a different dataset version")
        # Don't raise exception, only warn, let user decide whether to continue


class SASRec_LLM2Rec(SASRec_seq):
    """
    Pure SASRec + LLM2Rec semantic insertion + adapter
    Does not include dual-view, cross attention and other additional components
    """
    
    def __init__(self, user_num, item_num, device, args):
        # Call parent class initialization first (will initialize item_emb)
        super().__init__(user_num, item_num, device, args)
        
        self.use_llm2rec = args.use_llm2rec if hasattr(args, 'use_llm2rec') else False
        
        if self.use_llm2rec:
            # Freeze item_emb inherited from parent class (not used in LLM2Rec mode)
            # This can: 1) Make printing cleaner 2) Save optimizer state memory
            self.item_emb.weight.requires_grad = False
            
            # Dynamically get LLM2Rec embedding path
            llm2rec_emb_path = get_llm2rec_emb_path(args)
            print(f"[LLM2Rec] Loading item embeddings from: {llm2rec_emb_path}")
            llm_item_emb = np.load(llm2rec_emb_path)
            
            # 校验物品数量
            validate_item_embedding(llm_item_emb, item_num, args.dataset)
            original_shape = llm_item_emb.shape
            
            # 添加 padding (索引0) 和 mask token (索引 item_num+1)
            # 处理后: 索引0=padding, 索引1~item_num=真实物品, 索引item_num+1=mask
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            
            print(f"[LLM2Rec] Embedding shape: {original_shape} -> {llm_item_emb.shape} (added padding & mask)")
            print(f"[LLM2Rec] Index mapping: 0=padding, 1~{item_num}=items, {item_num+1}=mask")
            
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            
            # LLM2Rec adapter: 轻量级线性投影
            self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
            
            # 冻结 LLM embedding（如果需要）
            if args.freeze:
                self.llm_item_emb.weight.requires_grad = False
            
            # 初始化 adapter 权重
            nn.init.xavier_normal_(self.adapter.weight)
            if self.adapter.bias is not None:
                nn.init.constant_(self.adapter.bias, 0)
    
    
    def _get_embedding(self, log_seqs):
        """
        获取 item embedding
        
        严格遵循 LLM2Rec 论文设定：
        - 当 use_llm2rec=True 时，只使用 adapter(llm_item_emb) 作为 item representation
        - item_emb 不参与（虽然从父类继承，但不使用）
        """
        if self.use_llm2rec:
            # 只使用 LLM2Rec 语义 embedding（符合论文设定）
            llm_emb = self.llm_item_emb(log_seqs)
            llm_emb = self.adapter(llm_emb)
            return llm_emb
        else:
            # 不使用 LLM2Rec 时，使用原始 ID embedding
            return self.item_emb(log_seqs)


class SASRec_WithAlignment(SASRec_LLM2Rec, GAA_Mixin):
    """
    在 SASRec_LLM2Rec 基础上添加相似用户对比增强 (复用 GAA_Mixin)
    """
    
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        # 使用 alpha 参数名 (LLM2rec 风格)
        self._init_gaa(args, use_alpha_param=True)
        self._user_num = user_num  # 保存用户数量用于 bank 初始化
    
    def _log2feats_for_gaa(self, log_seqs, positions):
        """为 GAA 模块提供的 log2feats 接口"""
        return self.log2feats(log_seqs, positions)
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        """前向传播：基础损失 + 相似用户对比增强损失"""
        # 1. 计算基础损失（来自 SASRec_LLM2Rec）
        base_loss = super(SASRec_WithAlignment, self).forward(seq, pos, neg, positions, **kwargs)
        
        # 2. 如果 gaa_alpha=0，直接返回基础损失
        if self.gaa_alpha == 0:
            return base_loss
        
        # 3. 计算用户表示
        user_feats = self.log2feats(seq, positions)[:, -1, :]  # [bs, hidden_size]
        
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

