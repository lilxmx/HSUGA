# -*- coding: utf-8 -*-
"""
ICSRec 模型实现
使用 HSUGA 的序列编码器 backbone (SASRec, BERT4Rec, GRU4Rec)
实现意图对比学习 (Intent Contrastive Learning)

重要说明：
1. 原始ICSRec使用CrossEntropyLoss（全物品分类），只对最后一个位置计算损失
2. 为了与HSUGA的数据流兼容，SASRec和BERT4Rec可以同时支持seq2seq模式

关键设计：
- forward(): 返回推荐损失，使用CrossEntropyLoss（与原始ICSRec一致）
- log2feats(): 返回完整序列表示 [B, L, H]，供对比学习使用
- predict_full(): 返回全物品预测分数

新增功能 (HSU 融合增强):
- ICSRec_HSU_SASRec / ICSRec_HSU_BERT4Rec / ICSRec_HSU_GRU4Rec
- 通过 HSUFusionMixin 将离线计算的 LLM 用户语义向量注入到 ICSRec 中
- 可插拔设计：use_hsu_fusion=False 时与原 ICSRec 完全一致
"""

import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.Bert4Rec import BertBackbone
from models.GRU4Rec import GRU4RecBackbone
# 从 HSUGatedFusion.py 导入通用的 HSUFusionMixin（与 RCL 共用）
from models.HSUGatedFusion import HSUFusionMixin


class ICSRec_SASRec(BaseSeqModel):
    """
    ICSRec with SASRec backbone
    
    使用 SASRec 作为序列编码器，结合意图对比学习
    
    与原始ICSRec保持一致：
    - 推荐损失: CrossEntropyLoss（全物品分类）
    - 只对最后一个位置计算推荐损失
    """
    
    def __init__(self, user_num, item_num, device, args):
        super(ICSRec_SASRec, self).__init__(user_num, item_num, device, args)
        
        # Item embedding
        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        # Position embedding
        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # SASRec backbone
        self.backbone = SASRecBackbone(device, args)
        
        # Loss function: 使用CrossEntropyLoss与原始ICSRec保持一致
        self.ce_loss = nn.CrossEntropyLoss()
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        """获取物品嵌入"""
        return self.item_emb(log_seqs)
    
    def log2feats(self, log_seqs, positions):
        """
        序列编码：将物品序列转换为特征表示
        
        Args:
            log_seqs: 物品序列 [B, L]
            positions: 位置序列 [B, L]
            
        Returns:
            log_feats: 序列特征 [B, L, H]
        """
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats
    
    def forward(self, seq, positions, target_item, **kwargs):
        """
        训练前向传播（ICSRec原始推荐损失）
        
        与原始ICSRec保持一致：
        - 使用CrossEntropyLoss全物品分类
        - 只对最后一个位置计算损失
        
        Args:
            seq: 输入序列 [B, L]
            positions: 位置序列 [B, L]
            target_item: 目标物品ID [B]（最后一个位置的下一个物品）
            
        Returns:
            loss: 推荐损失
        """
        log_feats = self.log2feats(seq, positions)  # [B, L, H]
        seq_output = log_feats[:, -1, :]  # [B, H] 取最后一个位置
        
        # 全物品预测
        logits = self.predict_full(seq_output)  # [B, num_items]
        
        # CrossEntropyLoss
        rec_loss = self.ce_loss(logits, target_item.long())
        
        return rec_loss
    
    def predict(self, seq, item_indices, positions, **kwargs):
        """
        预测阶段（采样评估）
        
        Args:
            seq: 物品序列 [B, L]
            item_indices: 候选物品 [B, num_candidates]
            positions: 位置序列 [B, L]
            
        Returns:
            logits: 预测分数 [B, num_candidates]
        """
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]  # [B, H]
        item_embs = self._get_embedding(item_indices)  # [B, num_candidates, H]
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
    
    def predict_full(self, seq_output):
        """
        全物品预测
        
        Args:
            seq_output: 序列输出 [B, H]
            
        Returns:
            logits: 对所有物品的预测分数 [B, num_items]
        """
        test_item_emb = self.item_emb.weight
        rating_pred = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_user_emb(self, seq, positions, **kwargs):
        """获取用户嵌入表示"""
        log_feats = self.log2feats(seq, positions)
        return log_feats[:, -1, :]


class ICSRec_BERT4Rec(BaseSeqModel):
    """
    ICSRec with BERT4Rec backbone
    
    使用 BERT4Rec 作为序列编码器，结合意图对比学习
    
    注意：BERT4Rec在训练时使用mask机制，在ICSRec中：
    - 训练时使用CrossEntropyLoss（与原始ICSRec一致）
    - 对比学习时使用最后一个位置的表示
    """
    
    def __init__(self, user_num, item_num, device, args):
        super(ICSRec_BERT4Rec, self).__init__(user_num, item_num, device, args)
        
        self.mask_token = item_num + 1
        
        # Item embedding
        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        # Position embedding
        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # BERT backbone
        self.backbone = BertBackbone(device, args)
        
        # Loss function: 使用CrossEntropyLoss与原始ICSRec保持一致
        self.ce_loss = nn.CrossEntropyLoss()
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        """获取物品嵌入"""
        return self.item_emb(log_seqs)
    
    def log2feats(self, log_seqs, positions):
        """序列编码（不添加mask token，用于获取当前序列表示）"""
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats
    
    def log2feats_with_mask(self, log_seqs, positions):
        """序列编码（添加mask token用于预测下一个物品）"""
        # 在序列末尾添加mask token
        mask_tokens = self.mask_token * torch.ones(log_seqs.shape[0], 1, device=self.dev, dtype=log_seqs.dtype)
        log_seqs_masked = torch.cat([log_seqs, mask_tokens], dim=1)[:, 1:]  # 移除第一个，保持长度
        
        # 更新位置
        pred_position = positions[:, -1:] + 1
        positions_masked = torch.cat([positions, pred_position], dim=1)[:, 1:]
        
        seqs = self._get_embedding(log_seqs_masked)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions_masked.long())
        seqs = self.emb_dropout(seqs)
        
        log_feats = self.backbone(seqs, log_seqs_masked)
        return log_feats
    
    def forward(self, seq, positions, target_item, **kwargs):
        """
        训练前向传播（ICSRec原始推荐损失）
        
        Args:
            seq: 输入序列 [B, L]
            positions: 位置序列 [B, L]
            target_item: 目标物品ID [B]
            
        Returns:
            loss: 推荐损失
        """
        # 使用带mask的编码获取预测表示
        log_feats = self.log2feats_with_mask(seq, positions)  # [B, L, H]
        seq_output = log_feats[:, -1, :]  # [B, H] 取最后一个位置（mask位置）
        
        # 全物品预测
        logits = self.predict_full(seq_output)  # [B, num_items]
        
        # CrossEntropyLoss
        rec_loss = self.ce_loss(logits, target_item.long())
        
        return rec_loss
    
    def predict(self, seq, item_indices, positions, **kwargs):
        """预测阶段"""
        log_feats = self.log2feats_with_mask(seq, positions)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
    
    def predict_full(self, seq_output):
        """全物品预测"""
        test_item_emb = self.item_emb.weight
        rating_pred = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_user_emb(self, seq, positions, **kwargs):
        """获取用户嵌入表示（使用不带mask的表示）"""
        log_feats = self.log2feats(seq, positions)
        return log_feats[:, -1, :]


class ICSRec_GRU4Rec(BaseSeqModel):
    """
    ICSRec with GRU4Rec backbone
    
    使用 GRU4Rec 作为序列编码器，结合意图对比学习
    
    注意：GRU4Rec不使用位置编码，推荐损失与原始ICSRec保持一致
    """
    
    def __init__(self, user_num, item_num, device, args):
        super(ICSRec_GRU4Rec, self).__init__(user_num, item_num, device, args)
        
        # Item embedding
        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        
        # GRU backbone
        self.backbone = GRU4RecBackbone(device, args)
        
        # Loss function: 使用CrossEntropyLoss与原始ICSRec保持一致
        self.ce_loss = nn.CrossEntropyLoss()
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        """获取物品嵌入"""
        return self.item_emb(log_seqs)
    
    def log2feats(self, log_seqs, positions=None):
        """
        序列编码
        
        注意：GRU4Rec 不需要位置编码
        
        Args:
            log_seqs: 物品序列 [B, L]
            positions: 位置序列（GRU4Rec不使用，保留参数兼容性）
            
        Returns:
            log_feats: 序列特征 [B, L, H]
        """
        seqs = self.item_emb(log_seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats
    
    def forward(self, seq, positions, target_item, **kwargs):
        """
        训练前向传播（ICSRec原始推荐损失）
        
        Args:
            seq: 输入序列 [B, L]
            positions: 位置序列（GRU4Rec不使用）
            target_item: 目标物品ID [B]
            
        Returns:
            loss: 推荐损失
        """
        log_feats = self.log2feats(seq)  # [B, L, H]
        seq_output = log_feats[:, -1, :]  # [B, H] 取最后一个位置
        
        # 全物品预测
        logits = self.predict_full(seq_output)  # [B, num_items]
        
        # CrossEntropyLoss
        rec_loss = self.ce_loss(logits, target_item.long())
        
        return rec_loss
    
    def predict(self, seq, item_indices, positions=None, **kwargs):
        """预测阶段"""
        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
    
    def predict_full(self, seq_output):
        """全物品预测"""
        test_item_emb = self.item_emb.weight
        rating_pred = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_user_emb(self, seq, positions=None, **kwargs):
        """获取用户嵌入表示"""
        log_feats = self.log2feats(seq)
        return log_feats[:, -1, :]


# ============================================================================
# HSU 融合增强版 ICSRec 模型
# 通过 HSUFusionMixin 将离线 LLM 用户语义向量注入到 ICSRec 中
# 注意: HSUFusionMixin 已从 HSUGatedFusion.py 导入（与 RCL 共用）
# ============================================================================

class ICSRec_HSU_SASRec(BaseSeqModel, HSUFusionMixin):
    """
    ICSRec + HSU 融合增强 (SASRec backbone)
    
    在 ICSRec_SASRec 基础上，将离线计算的 LLM 用户语义向量
    通过门控机制注入到用户表示中。
    """
    
    def __init__(self, user_num, item_num, device, args):
        super(ICSRec_HSU_SASRec, self).__init__(user_num, item_num, device, args)
        
        # Item embedding
        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        # Position embedding
        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # SASRec backbone
        self.backbone = SASRecBackbone(device, args)
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()
        
        # === HSU 融合模块 ===
        self._init_hsu_fusion(args, device)
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        return self.item_emb(log_seqs)
    
    def log2feats(self, log_seqs, positions, user_id=None):
        """
        序列编码（支持 HSU 融合）
        
        Args:
            log_seqs: 物品序列 [B, L]
            positions: 位置序列 [B, L]
            user_id: 用户ID [B]，用于 HSU 融合
            
        Returns:
            log_feats: 序列特征 [B, L, H]（最后一个位置可能被 HSU 融合增强）
        """
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        
        log_feats = self.backbone(seqs, log_seqs)
        
        # === HSU 门控融合（仅对最后一个位置） ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = log_feats[:, -1, :]  # [B, D]
            h_fused = self._apply_hsu_fusion(h_base, user_id)
            log_feats = log_feats.clone()
            log_feats[:, -1, :] = h_fused
        
        return log_feats
    
    def forward(self, seq, positions, target_item, user_id=None, **kwargs):
        """训练前向传播"""
        log_feats = self.log2feats(seq, positions, user_id)
        seq_output = log_feats[:, -1, :]
        
        logits = self.predict_full(seq_output)
        rec_loss = self.ce_loss(logits, target_item.long())
        
        return rec_loss
    
    def predict(self, seq, item_indices, positions, user_id=None, **kwargs):
        """预测阶段"""
        log_feats = self.log2feats(seq, positions, user_id)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
    
    def predict_full(self, seq_output):
        """全物品预测"""
        test_item_emb = self.item_emb.weight
        rating_pred = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_user_emb(self, seq, positions, user_id=None, **kwargs):
        """获取用户嵌入表示"""
        log_feats = self.log2feats(seq, positions, user_id)
        return log_feats[:, -1, :]


class ICSRec_HSU_BERT4Rec(BaseSeqModel, HSUFusionMixin):
    """
    ICSRec + HSU 融合增强 (BERT4Rec backbone)
    """
    
    def __init__(self, user_num, item_num, device, args):
        super(ICSRec_HSU_BERT4Rec, self).__init__(user_num, item_num, device, args)
        
        self.mask_token = item_num + 1
        
        # Item embedding
        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        # Position embedding
        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # BERT backbone
        self.backbone = BertBackbone(device, args)
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()
        
        # === HSU 融合模块 ===
        self._init_hsu_fusion(args, device)
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        return self.item_emb(log_seqs)
    
    def log2feats(self, log_seqs, positions, user_id=None):
        """序列编码（不添加mask token，用于获取当前序列表示）"""
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        
        log_feats = self.backbone(seqs, log_seqs)
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = log_feats[:, -1, :]
            h_fused = self._apply_hsu_fusion(h_base, user_id)
            log_feats = log_feats.clone()
            log_feats[:, -1, :] = h_fused
        
        return log_feats
    
    def log2feats_with_mask(self, log_seqs, positions, user_id=None):
        """序列编码（添加mask token用于预测下一个物品）"""
        # 在序列末尾添加mask token
        mask_tokens = self.mask_token * torch.ones(log_seqs.shape[0], 1, device=self.dev, dtype=log_seqs.dtype)
        log_seqs_masked = torch.cat([log_seqs, mask_tokens], dim=1)[:, 1:]
        
        # 更新位置
        pred_position = positions[:, -1:] + 1
        positions_masked = torch.cat([positions, pred_position], dim=1)[:, 1:]
        
        seqs = self._get_embedding(log_seqs_masked)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs += self.pos_emb(positions_masked.long())
        seqs = self.emb_dropout(seqs)
        
        log_feats = self.backbone(seqs, log_seqs_masked)
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = log_feats[:, -1, :]
            h_fused = self._apply_hsu_fusion(h_base, user_id)
            log_feats = log_feats.clone()
            log_feats[:, -1, :] = h_fused
        
        return log_feats
    
    def forward(self, seq, positions, target_item, user_id=None, **kwargs):
        """训练前向传播"""
        log_feats = self.log2feats_with_mask(seq, positions, user_id)
        seq_output = log_feats[:, -1, :]
        
        logits = self.predict_full(seq_output)
        rec_loss = self.ce_loss(logits, target_item.long())
        
        return rec_loss
    
    def predict(self, seq, item_indices, positions, user_id=None, **kwargs):
        """预测阶段"""
        log_feats = self.log2feats_with_mask(seq, positions, user_id)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
    
    def predict_full(self, seq_output):
        """全物品预测"""
        test_item_emb = self.item_emb.weight
        rating_pred = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_user_emb(self, seq, positions, user_id=None, **kwargs):
        """获取用户嵌入表示"""
        log_feats = self.log2feats(seq, positions, user_id)
        return log_feats[:, -1, :]


class ICSRec_HSU_GRU4Rec(BaseSeqModel, HSUFusionMixin):
    """
    ICSRec + HSU 融合增强 (GRU4Rec backbone)
    """
    
    def __init__(self, user_num, item_num, device, args):
        super(ICSRec_HSU_GRU4Rec, self).__init__(user_num, item_num, device, args)
        
        # Item embedding
        self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        
        # GRU backbone
        self.backbone = GRU4RecBackbone(device, args)
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()
        
        # === HSU 融合模块 ===
        self._init_hsu_fusion(args, device)
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        return self.item_emb(log_seqs)
    
    def log2feats(self, log_seqs, positions=None, user_id=None):
        """序列编码（GRU4Rec 不需要 positions）"""
        seqs = self.item_emb(log_seqs)
        log_feats = self.backbone(seqs, log_seqs)
        
        # === HSU 门控融合 ===
        if self.use_hsu_fusion and user_id is not None:
            h_base = log_feats[:, -1, :]
            h_fused = self._apply_hsu_fusion(h_base, user_id)
            log_feats = log_feats.clone()
            log_feats[:, -1, :] = h_fused
        
        return log_feats
    
    def forward(self, seq, positions, target_item, user_id=None, **kwargs):
        """训练前向传播（positions 参数保留用于接口兼容性）"""
        log_feats = self.log2feats(seq, user_id=user_id)
        seq_output = log_feats[:, -1, :]
        
        logits = self.predict_full(seq_output)
        rec_loss = self.ce_loss(logits, target_item.long())
        
        return rec_loss
    
    def predict(self, seq, item_indices, positions=None, user_id=None, **kwargs):
        """预测阶段"""
        log_feats = self.log2feats(seq, user_id=user_id)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
    
    def predict_full(self, seq_output):
        """全物品预测"""
        test_item_emb = self.item_emb.weight
        rating_pred = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_user_emb(self, seq, positions=None, user_id=None, **kwargs):
        """获取用户嵌入表示"""
        log_feats = self.log2feats(seq, user_id=user_id)
        return log_feats[:, -1, :]

