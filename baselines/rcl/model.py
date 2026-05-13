# RCL.py
# RCL (Robust Contrastive Learning) 模型，使用 HSUGA 的 backbone
# 参考论文: RCL: Reliable Contrastive Learning for Sequential Recommendation
# 
# 新增功能: HSU 门控融合 (B2: Gated Fusion)
# 用于将离线计算的 HSU 用户语义向量注入到 RCL 的用户表示中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.GRU4Rec import GRU4RecBackbone
from models.Bert4Rec import BertBackbone
# 从 HSUGatedFusion.py 导入通用的 HSUFusionMixin（与 ICSRec 共用）
from models.HSUGatedFusion import HSUFusionMixin


class RCL_SASRec(BaseSeqModel, HSUFusionMixin):
    """
    RCL + SASRec Backbone
    该模型的 forward 返回 (pos_logits, neg_logits, log_feats)，用于 RCL 的 SSL 训练
    与 SASRec_seq 保持一致的架构，只是返回值不同
    
    新增: HSU 门控融合功能，可通过 args.use_hsu_fusion 启用
    """
    
    def __init__(self, user_num, item_num, device, args):
        super(RCL_SASRec, self).__init__(user_num, item_num, device, args)
        
        self.max_len = args.max_len
        self.hidden_size = args.hidden_size
        
        # Item embedding (与原模型保持一致，使用 item_num + 2)
        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        
        # Position embedding
        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # 使用 HSUGA 的 SASRecBackbone
        self.backbone = SASRecBackbone(device, args)
        
        # 初始化 HSU 融合模块
        self._init_hsu_fusion(args, device)
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        """获取 item embedding"""
        return self.item_emb(log_seqs)
    
    def _generate_positions(self, log_seqs):
        """内部生成 positions (用于 get_embed 等不传入 positions 的方法)"""
        batch_size = log_seqs.shape[0]
        seq_len = log_seqs.shape[1]
        positions = np.tile(np.array(range(seq_len)), [batch_size, 1])
        return torch.LongTensor(positions).to(self.dev)
    
    def log2feats(self, log_seqs, positions):
        """获取序列特征表示 (使用外部传入的 positions，与原模型保持一致)"""
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        
        log_feats = self.backbone(seqs, log_seqs)
        
        return log_feats
    
    def forward(self, seq, pos, neg, positions, user_id=None, **kwargs):
        """
        RCL 训练接口
        返回: (pos_logits, neg_logits, log_feats) 用于 RCL SSL 损失计算
        
        新增参数:
            user_id: 用户索引，用于 HSU 融合。如果不提供，则不进行融合。
        """
        log_feats = self.log2feats(seq, positions)
        
        # === HSU 门控融合 ===
        # 对最后一个位置的用户表示进行融合
        if self.use_hsu_fusion and user_id is not None:
            # 获取最后位置的表示
            h_base = log_feats[:, -1, :]  # [B, D]
            h_fused = self._apply_hsu_fusion(h_base, user_id)
            
            # 将融合后的表示放回 log_feats 的最后位置
            # 注意：我们只融合最后一个位置，因为这是用于 ranking 和 CL 的主要表示
            log_feats = log_feats.clone()
            log_feats[:, -1, :] = h_fused
        
        pos_embs = self._get_embedding(pos)
        neg_embs = self._get_embedding(neg)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits, log_feats
    
    def predict(self, seq, item_indices, positions, user_id=None, **kwargs):
        """推理接口"""
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            final_feat = self._apply_hsu_fusion(final_feat, user_id)
        
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
    
    def get_embed(self, log_seqs, user_id=None):
        """获取序列最后位置的 embedding，用于 RCL 的 SSL (内部生成 positions)"""
        positions = self._generate_positions(log_seqs)
        log_feats = self.log2feats(log_seqs, positions)
        h_base = log_feats[:, -1, :]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = self._apply_hsu_fusion(h_base, user_id)
        
        return h_base
    
    def get_user_emb(self, seq, positions, user_id=None, **kwargs):
        """获取用户表示"""
        log_feats = self.log2feats(seq, positions)
        h_base = log_feats[:, -1, :]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = self._apply_hsu_fusion(h_base, user_id)
        
        return h_base


class RCL_GRU4Rec(BaseSeqModel, HSUFusionMixin):
    """
    RCL + GRU4Rec Backbone
    与 GRU4Rec 保持一致的架构
    
    新增: HSU 门控融合功能
    """
    
    def __init__(self, user_num, item_num, device, args):
        super(RCL_GRU4Rec, self).__init__(user_num, item_num, device, args)
        
        self.max_len = args.max_len
        self.hidden_size = args.hidden_size
        
        # Item embedding (与原模型保持一致，使用 item_num + 2)
        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # 使用 HSUGA 的 GRU4RecBackbone
        self.backbone = GRU4RecBackbone(device, args)
        
        # 初始化 HSU 融合模块
        self._init_hsu_fusion(args, device)
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        """获取 item embedding"""
        return self.item_emb(log_seqs)
    
    def log2feats(self, log_seqs):
        """获取序列特征表示 (GRU4Rec 不需要 positions)"""
        seqs = self._get_embedding(log_seqs)
        seqs = self.emb_dropout(seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats
    
    def forward(self, seq, pos, neg, positions, user_id=None, **kwargs):
        """
        RCL 训练接口 (适配 GeneratorAllUser 的数据格式)
        
        注意: GRU4Rec 使用 GeneratorAllUser，pos/neg 是单个值 [B]，不是序列 [B, T]
        因此 pos_logits/neg_logits 返回的是 [B]，不是 [B, T]
        但 log_feats 仍然返回完整的 [B, T, D] 用于 SSL
        """
        log_feats = self.log2feats(seq)  # [B, T, D]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = log_feats[:, -1, :]
            h_fused = self._apply_hsu_fusion(h_base, user_id)
            log_feats = log_feats.clone()
            log_feats[:, -1, :] = h_fused
        
        # GRU4Rec: pos 是单个值 [B]，取最后位置的 log_feats 计算 logits
        final_feat = log_feats[:, -1, :]  # [B, D]
        
        pos_embs = self._get_embedding(pos)  # [B, D]
        neg_embs = self._get_embedding(neg)  # [B, D] 或 [B, neg_num, D]
        
        # pos_logits: [B]
        pos_logits = (final_feat * pos_embs).sum(dim=-1)
        
        # neg_logits: [B] 或 [B, neg_num]
        if neg_embs.dim() == 2:
            neg_logits = (final_feat * neg_embs).sum(dim=-1)
        else:
            neg_logits = (final_feat.unsqueeze(1) * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits, log_feats
    
    def predict(self, seq, item_indices, positions, user_id=None, **kwargs):
        """推理接口"""
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            final_feat = self._apply_hsu_fusion(final_feat, user_id)
        
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
    
    def get_embed(self, log_seqs, user_id=None):
        """获取序列最后位置的 embedding"""
        log_feats = self.log2feats(log_seqs)
        h_base = log_feats[:, -1, :]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = self._apply_hsu_fusion(h_base, user_id)
        
        return h_base
    
    def get_user_emb(self, seq, positions, user_id=None, **kwargs):
        """获取用户表示"""
        log_feats = self.log2feats(seq)
        h_base = log_feats[:, -1, :]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = self._apply_hsu_fusion(h_base, user_id)
        
        return h_base


class RCL_Bert4Rec(BaseSeqModel, HSUFusionMixin):
    """
    RCL + Bert4Rec Backbone
    与 Bert4Rec 保持一致的架构
    
    新增: HSU 门控融合功能
    """
    
    def __init__(self, user_num, item_num, device, args):
        super(RCL_Bert4Rec, self).__init__(user_num, item_num, device, args)
        
        self.max_len = args.max_len
        self.hidden_size = args.hidden_size
        
        # Item embedding (与原模型保持一致，使用 item_num + 2)
        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        
        # Position embedding
        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # 使用 HSUGA 的 BertBackbone
        self.backbone = BertBackbone(device, args)
        
        # 初始化 HSU 融合模块
        self._init_hsu_fusion(args, device)
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        """获取 item embedding"""
        return self.item_emb(log_seqs)
    
    def _generate_positions(self, log_seqs):
        """内部生成 positions"""
        batch_size = log_seqs.shape[0]
        seq_len = log_seqs.shape[1]
        positions = np.tile(np.array(range(seq_len)), [batch_size, 1])
        return torch.LongTensor(positions).to(self.dev)
    
    def log2feats(self, log_seqs, positions):
        """获取序列特征表示 (使用外部传入的 positions)"""
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        
        log_feats = self.backbone(seqs, log_seqs)
        
        return log_feats
    
    def forward(self, seq, pos, neg, positions, user_id=None, **kwargs):
        """
        RCL 训练接口 (适配 BertGeneratorAllUser 的数据格式)
        
        注意: Bert4Rec 使用 BertGeneratorAllUser，pos/neg 是序列 [B, T]（与 SASRec 相同）
        因为 BertRecTrainDatasetAllUser 返回的 labels/neg_labels 是序列格式
        """
        log_feats = self.log2feats(seq, positions)  # [B, T, D]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = log_feats[:, -1, :]
            h_fused = self._apply_hsu_fusion(h_base, user_id)
            log_feats = log_feats.clone()
            log_feats[:, -1, :] = h_fused
        
        # Bert4Rec: pos/neg 是序列 [B, T]，与 SASRec 相同
        pos_embs = self._get_embedding(pos)  # [B, T, D]
        neg_embs = self._get_embedding(neg)  # [B, T, D]
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # [B, T]
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # [B, T]
        
        return pos_logits, neg_logits, log_feats
    
    def predict(self, seq, item_indices, positions, user_id=None, **kwargs):
        """推理接口"""
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            final_feat = self._apply_hsu_fusion(final_feat, user_id)
        
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
    
    def get_embed(self, log_seqs, user_id=None):
        """获取序列最后位置的 embedding"""
        positions = self._generate_positions(log_seqs)
        log_feats = self.log2feats(log_seqs, positions)
        h_base = log_feats[:, -1, :]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = self._apply_hsu_fusion(h_base, user_id)
        
        return h_base
    
    def get_user_emb(self, seq, positions, user_id=None, **kwargs):
        """获取用户表示"""
        log_feats = self.log2feats(seq, positions)
        h_base = log_feats[:, -1, :]
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = self._apply_hsu_fusion(h_base, user_id)
        
        return h_base
