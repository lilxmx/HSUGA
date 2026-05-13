"""
LLMEmb: Large Language Model Empowered Embedding Generator for Sequential Recommendation

Core idea:
1. Use LLM-generated item embeddings (obtained through SCFT fine-tuning)
2. Project LLM embeddings to required dimensions through Adapter (two-layer MLP)
3. Add alignment loss: align adapter(LLM_emb) with pre-trained SRS_emb

Key components:
- llmemb_pca.pkl: LLM-generated item embeddings (PCA reduced)
- srs_emb: Pre-trained SRS item embeddings (collaborative embeddings for alignment)
- Adapter: Two-layer MLP that projects LLM embeddings to hidden_size
- alpha: Alignment loss weight
- tau: Contrastive learning temperature
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.GRU4Rec import GRU4Rec, GRU4Rec_seq
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.utils import Contrastive_Loss2


# ==============================================================================
# LLMEmb 数据集配置
# ==============================================================================

# LLMEmb embedding 文件名映射
LLMEMB_EMB_FILES = {
    "steam": "data/steam/handled/llmemb_pca.pkl",
    "fashion": "data/fashion/handled/llmemb_pca.pkl",
    "beauty": "data/beauty/handled/llmemb_pca.pkl",
}

# SRS embedding filename template (for alignment)
# This is item embedding extracted from pre-trained SRS model
# Need to run first: python main.py --model_name sasrec --do_item_emb ... to generate
SRS_EMB_TEMPLATE = "data/{dataset}/handled/itm_emb_sasrec.pkl"


def get_llmemb_path(args):
    """Get LLMEmb embedding path"""
    if hasattr(args, 'llmemb_path') and args.llmemb_path:
        return args.llmemb_path
    dataset = args.dataset.lower()
    if dataset in LLMEMB_EMB_FILES:
        return LLMEMB_EMB_FILES[dataset]
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: {list(LLMEMB_EMB_FILES.keys())}")


def get_srs_emb_path(args):
    """
    Get SRS embedding path (for alignment)
    
    This is collaborative item embedding extracted from pre-trained SRS model.
    Need to run the following commands to generate:
        1. Train SASRec: python main.py --model_name sasrec --dataset {dataset} ...
        2. Extract embedding: python main.py --model_name sasrec --dataset {dataset} --do_item_emb ...
    
    Generated file: data/{dataset}/handled/itm_emb_sasrec.pkl
    """
    # Prefer user-specified path
    if hasattr(args, 'srs_emb_path') and args.srs_emb_path:
        return args.srs_emb_path
    
    # Use default path
    dataset = args.dataset.lower()
    srs_path = SRS_EMB_TEMPLATE.format(dataset=dataset)
    
    # Check if file exists
    if os.path.exists(srs_path):
        return srs_path
    else:
        raise ValueError(
            f"SRS embedding file not found: {srs_path}\n"
            f"Please generate it first:\n"
            f"  1. Train SASRec: python main.py --model_name sasrec --dataset {dataset} ...\n"
            f"  2. Extract embedding: python main.py --model_name sasrec --dataset {dataset} --do_item_emb ...\n"
            f"Or specify --srs_emb_path manually."
        )


def load_embedding_file(path):
    """Load embedding file (supports pkl and npy)"""
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            emb = pickle.load(f)
    elif path.endswith('.npy'):
        emb = np.load(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return emb


# ==============================================================================
# LLMEmb + GRU4Rec
# ==============================================================================

class GRU4Rec_LLMEmb(GRU4Rec):
    """
    GRU4Rec + LLMEmb
    
    核心思路:
    1. 用 LLM embedding + Adapter 替换原始 item embedding
    2. 添加对齐损失: 让 adapter(LLM_emb) 与预训练的 SRS_emb 对齐
    """
    
    def __init__(self, user_num, item_num, device, args):
        # 先调用父类初始化
        super().__init__(user_num, item_num, device, args)
        
        self.hidden_size = args.hidden_size
        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.0
        self.tau = args.tau if hasattr(args, 'tau') else 1.0
        
        # 加载 LLM embedding
        llmemb_path = get_llmemb_path(args)
        print(f"[LLMEmb] Loading LLM embeddings from: {llmemb_path}")
        llm_item_emb = load_embedding_file(llmemb_path)
        
        # 添加 padding (索引0) 和 mask token (索引 item_num+1)
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        
        print(f"[LLMEmb] LLM embedding shape: {llm_item_emb.shape}")
        
        # LLM item embedding (替换原始 item_emb)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        
        # 是否冻结 LLM embedding
        freeze_emb = args.freeze_emb if hasattr(args, 'freeze_emb') else True
        self.llm_item_emb.weight.requires_grad = not freeze_emb
        print(f"[LLMEmb] LLM embedding frozen: {freeze_emb}")
        
        # Adapter: 两层 MLP，将 LLM embedding 投影到 hidden_size
        llm_dim = llm_item_emb.shape[1]
        self.adapter = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 2),
            nn.Linear(llm_dim // 2, args.hidden_size)
        )
        
        # 冻结父类的 item_emb (LLMEmb 模式下不使用)
        self.item_emb.weight.requires_grad = False
        
        # 加载 SRS embedding (用于对齐)
        if self.alpha > 0:
            srs_emb_path = get_srs_emb_path(args)
            print(f"[LLMEmb] Loading SRS embeddings for alignment from: {srs_emb_path}")
            srs_item_emb = load_embedding_file(srs_emb_path)
            
            # 添加 padding
            srs_item_emb = np.insert(srs_item_emb, 0, values=np.zeros((1, srs_item_emb.shape[1])), axis=0)
            srs_item_emb = np.concatenate([srs_item_emb, np.zeros((1, srs_item_emb.shape[1]))], axis=0)
            
            print(f"[LLMEmb] SRS embedding shape: {srs_item_emb.shape}")
            
            self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
            self.srs_emb.weight.requires_grad = False  # SRS embedding 始终冻结
            
            # 对齐损失函数
            self.align_loss_func = Contrastive_Loss2(self.tau)
        
        # 初始化 Adapter 权重
        self._init_adapter_weights()
    
    def _init_adapter_weights(self):
        """初始化 Adapter 权重"""
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _get_embedding(self, log_seqs):
        """获取 item embedding (LLM embedding + Adapter)"""
        llm_emb = self.llm_item_emb(log_seqs)
        item_emb = self.adapter(llm_emb)
        return item_emb
    
    def log2feats(self, log_seqs):
        """重写 log2feats，使用 LLM embedding"""
        seqs = self._get_embedding(log_seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        """前向传播: 基础损失 + 对齐损失"""
        # 计算基础损失 (使用父类的 forward 逻辑，但 _get_embedding 已被重写)
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
        base_loss = pos_loss + neg_loss
        
        # 计算对齐损失
        if self.alpha > 0:
            # 获取 LLM embedding (经过 Adapter)
            llm_embs = self._get_embedding(pos[indices])
            # 获取 SRS embedding
            srs_embs = self.srs_emb(pos[indices])
            # 计算对齐损失
            align_loss = self.align_loss_func(srs_embs, llm_embs)
            total_loss = base_loss + self.alpha * align_loss
        else:
            total_loss = base_loss
        
        return total_loss


# ==============================================================================
# LLMEmb + SASRec
# ==============================================================================

class SASRec_LLMEmb(SASRec_seq):
    """
    SASRec + LLMEmb
    
    核心思路:
    1. 用 LLM embedding + Adapter 替换原始 item embedding
    2. 添加对齐损失: 让 adapter(LLM_emb) 与预训练的 SRS_emb 对齐
    """
    
    def __init__(self, user_num, item_num, device, args):
        # 先调用父类初始化
        super().__init__(user_num, item_num, device, args)
        
        self.hidden_size = args.hidden_size
        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.0
        self.tau = args.tau if hasattr(args, 'tau') else 1.0
        
        # 加载 LLM embedding
        llmemb_path = get_llmemb_path(args)
        print(f"[LLMEmb] Loading LLM embeddings from: {llmemb_path}")
        llm_item_emb = load_embedding_file(llmemb_path)
        
        # 添加 padding (索引0) 和 mask token (索引 item_num+1)
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        
        print(f"[LLMEmb] LLM embedding shape: {llm_item_emb.shape}")
        
        # LLM item embedding
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        
        # 是否冻结 LLM embedding
        freeze_emb = args.freeze_emb if hasattr(args, 'freeze_emb') else True
        self.llm_item_emb.weight.requires_grad = not freeze_emb
        print(f"[LLMEmb] LLM embedding frozen: {freeze_emb}")
        
        # Adapter: 两层 MLP
        llm_dim = llm_item_emb.shape[1]
        self.adapter = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 2),
            nn.Linear(llm_dim // 2, args.hidden_size)
        )
        
        # 冻结父类的 item_emb
        self.item_emb.weight.requires_grad = False
        
        # 加载 SRS embedding (用于对齐)
        if self.alpha > 0:
            srs_emb_path = get_srs_emb_path(args)
            print(f"[LLMEmb] Loading SRS embeddings for alignment from: {srs_emb_path}")
            srs_item_emb = load_embedding_file(srs_emb_path)
            
            srs_item_emb = np.insert(srs_item_emb, 0, values=np.zeros((1, srs_item_emb.shape[1])), axis=0)
            srs_item_emb = np.concatenate([srs_item_emb, np.zeros((1, srs_item_emb.shape[1]))], axis=0)
            
            print(f"[LLMEmb] SRS embedding shape: {srs_item_emb.shape}")
            
            self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
            self.srs_emb.weight.requires_grad = False
            
            self.align_loss_func = Contrastive_Loss2(self.tau)
        
        self._init_adapter_weights()
    
    def _init_adapter_weights(self):
        """初始化 Adapter 权重"""
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _get_embedding(self, log_seqs):
        """获取 item embedding (LLM embedding + Adapter)"""
        llm_emb = self.llm_item_emb(log_seqs)
        item_emb = self.adapter(llm_emb)
        return item_emb
    
    def log2feats(self, log_seqs, positions):
        """重写 log2feats，使用 LLM embedding"""
        seqs = self._get_embedding(log_seqs)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        """前向传播: 基础损失 + 对齐损失"""
        # 计算基础损失 (seq-to-seq loss)
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
        base_loss = pos_loss + neg_loss
        
        # 计算对齐损失
        if self.alpha > 0:
            llm_embs = self._get_embedding(pos[indices])
            srs_embs = self.srs_emb(pos[indices])
            align_loss = self.align_loss_func(srs_embs, llm_embs)
            total_loss = base_loss + self.alpha * align_loss
        else:
            total_loss = base_loss
        
        return total_loss


# ==============================================================================
# LLMEmb + Bert4Rec
# ==============================================================================

class Bert4Rec_LLMEmb(Bert4Rec):
    """
    Bert4Rec + LLMEmb
    
    核心思路:
    1. 用 LLM embedding + Adapter 替换原始 item embedding
    2. 添加对齐损失: 让 adapter(LLM_emb) 与预训练的 SRS_emb 对齐
    """
    
    def __init__(self, user_num, item_num, device, args):
        # 先调用父类初始化
        super().__init__(user_num, item_num, device, args)
        
        self.hidden_size = args.hidden_size
        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.0
        self.tau = args.tau if hasattr(args, 'tau') else 1.0
        
        # 加载 LLM embedding
        llmemb_path = get_llmemb_path(args)
        print(f"[LLMEmb] Loading LLM embeddings from: {llmemb_path}")
        llm_item_emb = load_embedding_file(llmemb_path)
        
        # 添加 padding (索引0) 和 mask token (索引 item_num+1)
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        
        print(f"[LLMEmb] LLM embedding shape: {llm_item_emb.shape}")
        
        # LLM item embedding
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        
        # 是否冻结 LLM embedding
        freeze_emb = args.freeze_emb if hasattr(args, 'freeze_emb') else True
        self.llm_item_emb.weight.requires_grad = not freeze_emb
        print(f"[LLMEmb] LLM embedding frozen: {freeze_emb}")
        
        # Adapter: 两层 MLP
        llm_dim = llm_item_emb.shape[1]
        self.adapter = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 2),
            nn.Linear(llm_dim // 2, args.hidden_size)
        )
        
        # Mask embedding (Bert4Rec 特有)
        self.mask_embedding = nn.Parameter(torch.zeros(self.hidden_size).normal_(0, 0.01))
        
        # 冻结父类的 item_emb
        self.item_emb.weight.requires_grad = False
        
        # 加载 SRS embedding (用于对齐)
        if self.alpha > 0:
            srs_emb_path = get_srs_emb_path(args)
            print(f"[LLMEmb] Loading SRS embeddings for alignment from: {srs_emb_path}")
            srs_item_emb = load_embedding_file(srs_emb_path)
            
            srs_item_emb = np.insert(srs_item_emb, 0, values=np.zeros((1, srs_item_emb.shape[1])), axis=0)
            srs_item_emb = np.concatenate([srs_item_emb, np.zeros((1, srs_item_emb.shape[1]))], axis=0)
            
            print(f"[LLMEmb] SRS embedding shape: {srs_item_emb.shape}")
            
            self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
            self.srs_emb.weight.requires_grad = False
            
            self.align_loss_func = Contrastive_Loss2(self.tau)
        
        self._init_adapter_weights()
    
    def _init_adapter_weights(self):
        """初始化 Adapter 权重"""
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _get_embedding(self, log_seqs):
        """获取 item embedding (LLM embedding + Adapter)"""
        llm_emb = self.llm_item_emb(log_seqs)
        item_emb = self.adapter(llm_emb)
        
        # 处理 mask token (Bert4Rec 特有)
        item_emb[log_seqs == self.mask_token] = self.mask_embedding
        
        return item_emb
    
    def log2feats(self, log_seqs, positions):
        """重写 log2feats，使用 LLM embedding"""
        seqs = self._get_embedding(log_seqs)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        """前向传播: 基础损失 + 对齐损失"""
        # 计算基础损失
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
        base_loss = pos_loss + neg_loss
        
        # 计算对齐损失
        if self.alpha > 0:
            llm_embs = self._get_embedding(pos[indices])
            srs_embs = self.srs_emb(pos[indices])
            align_loss = self.align_loss_func(srs_embs, llm_embs)
            total_loss = base_loss + self.alpha * align_loss
        else:
            total_loss = base_loss
        
        return total_loss


