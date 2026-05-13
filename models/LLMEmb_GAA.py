"""
LLMEmb + GAA (Group-Aware Alignment) 增强版

在 LLMEmb 基础上添加相似用户对比增强 (GAA 模块)：
1. LLMEmb 原有功能: LLM embedding + Adapter + 对齐损失 (alpha)
2. GAA 模块: 相似用户表示对齐 (gaa_alpha)

损失函数:
    total_loss = base_loss + alpha * align_loss + gaa_alpha * gaa_loss

其中:
- base_loss: 推荐任务基础损失 (BPR loss)
- align_loss: LLMEmb 原有的 SRS-LLM 对齐损失 (Contrastive)
- gaa_loss: 相似用户表示对齐损失 (KD 或 Contrastive)

支持两种 GAA 模式:
- encoder 模式 (默认): 每个 step 对相似用户序列过 encoder
- user_bank 模式: 使用 EMA 维护的用户表示 bank，避免重复计算
"""

import torch
import torch.nn as nn
import warnings

from models.LLMEmb import (
    GRU4Rec_LLMEmb,
    SASRec_LLMEmb, 
    Bert4Rec_LLMEmb
)
from models.utils import Contrastive_Loss2
from utils.utils import masked_mean


# ==============================================================================
# LLMEmb + GAA Mixin (复用 GAA 逻辑)
# ==============================================================================

class GAA_Mixin:
    """
    GAA (Group-Aware Alignment) 模块的混入类
    
    提供相似用户对比增强功能，子类需要实现 _log2feats_for_gaa 方法。
    
    支持两种参数命名方式（向后兼容）：
    - LLMEmb 风格: gaa_alpha, gaa_tau
    - LLM2rec 风格: alpha (通过 use_alpha_param=True 启用)
    
    支持两种 GAA 计算模式:
    - encoder 模式 (gaa_use_user_bank=False): 每次对相似用户序列过 encoder
    - user_bank 模式 (gaa_use_user_bank=True): 从 EMA bank 获取相似用户表示
    
    参数:
        gaa_alpha/alpha: GAA 损失权重，默认 0.0（不启用）
        user_sim_func: 用户相似度函数类型
            - 'kd': Knowledge Distillation (MSE Loss)
            - 'cl': Contrastive Learning (Contrastive Loss)
    """
    
    # 类变量：警告只打印一次
    _bank_fallback_warned = False
    
    def _init_gaa(self, args, use_alpha_param=False):
        """
        初始化 GAA 模块
        
        Args:
            args: 参数对象
            use_alpha_param: 是否使用 'alpha' 作为权重参数名
                - False (默认): 使用 gaa_alpha (LLMEmb 风格)
                - True: 使用 alpha (LLM2rec 风格)
        """
        # 根据参数名风格获取权重
        if use_alpha_param:
            self.gaa_alpha = args.alpha if hasattr(args, 'alpha') else 0.0
        else:
            self.gaa_alpha = args.gaa_alpha if hasattr(args, 'gaa_alpha') else 0.0
        
        self.user_sim_func = args.user_sim_func if hasattr(args, 'user_sim_func') else 'kd'
        
        # ========== User Bank 参数 ==========
        self.gaa_use_user_bank = getattr(args, 'gaa_use_user_bank', False)
        self.gaa_bank_momentum = getattr(args, 'gaa_bank_momentum', 0.9)
        self.gaa_bank_dtype = getattr(args, 'gaa_bank_dtype', 'fp32')
        self.gaa_bank_on_cpu = getattr(args, 'gaa_bank_on_cpu', True)
        self.gaa_warmup_epochs = getattr(args, 'gaa_warmup_epochs', 1)
        self.gaa_bank_min_fill_ratio = getattr(args, 'gaa_bank_min_fill_ratio', 0.1)
        
        # Bank 状态 (懒加载，在第一次 forward 时初始化)
        self._user_bank = None
        self._bank_filled = None
        self._bank_initialized = False
        self._current_epoch = 0  # 由外部 trainer 设置
        
        if self.gaa_alpha > 0:
            # 根据用户相似度函数类型选择损失函数
            if self.user_sim_func == "kd":  # Knowledge Distillation
                self.gaa_loss_func = nn.MSELoss()
            elif self.user_sim_func == "cl":  # Contrastive Learning
                gaa_tau = args.gaa_tau if hasattr(args, 'gaa_tau') else 1.0
                self.gaa_loss_func = Contrastive_Loss2(gaa_tau)
            else:
                self.gaa_loss_func = nn.MSELoss()
            
            param_name = 'alpha' if use_alpha_param else 'gaa_alpha'
            mode_str = "user_bank" if self.gaa_use_user_bank else "encoder"
            print(f"[GAA] enabled: {param_name}={self.gaa_alpha}, func={self.user_sim_func}, mode={mode_str}")
            
            if self.gaa_use_user_bank:
                print(f"[GAA-Bank] momentum={self.gaa_bank_momentum}, dtype={self.gaa_bank_dtype}, "
                      f"on_cpu={self.gaa_bank_on_cpu}, warmup_epochs={self.gaa_warmup_epochs}, "
                      f"min_fill_ratio={self.gaa_bank_min_fill_ratio}")
    
    def _init_user_bank(self, num_users, hidden_size, device):
        """
        懒加载初始化 user_bank
        
        Args:
            num_users: 用户数量
            hidden_size: 表示维度
            device: 当前设备
        """
        if self._bank_initialized:
            return
        
        # 确定 bank 存储位置和精度
        bank_device = torch.device('cpu') if self.gaa_bank_on_cpu else device
        bank_dtype = torch.float16 if self.gaa_bank_dtype == 'fp16' else torch.float32
        
        # 初始化 bank: [num_users, hidden_size]
        self._user_bank = torch.zeros(num_users, hidden_size, dtype=bank_dtype, device=bank_device)
        self._bank_filled = torch.zeros(num_users, dtype=torch.bool, device=bank_device)
        self._bank_initialized = True
        
        print(f"[GAA-Bank] Initialized: shape={self._user_bank.shape}, "
              f"dtype={bank_dtype}, device={bank_device}")
    
    def _update_user_bank(self, user_ids, user_feats):
        """
        使用 EMA 更新 user_bank
        
        Args:
            user_ids: 当前 batch 的用户 ID [bs]
            user_feats: 当前 batch 的用户表示 [bs, hidden_size] (已 detach)
        """
        if not self.gaa_use_user_bank or not self._bank_initialized:
            return
        
        # 确保数据在正确设备上
        user_ids_cpu = user_ids.cpu()
        user_feats_detach = user_feats.detach()
        
        # 转换精度
        if self._user_bank.dtype == torch.float16:
            user_feats_detach = user_feats_detach.half()
        
        # 转移到 bank 设备
        if self.gaa_bank_on_cpu:
            user_feats_detach = user_feats_detach.cpu()
        
        # EMA 更新
        m = self.gaa_bank_momentum
        for i, uid in enumerate(user_ids_cpu):
            uid = uid.item()
            if self._bank_filled[uid]:
                # 已有值，EMA 更新
                self._user_bank[uid] = m * self._user_bank[uid] + (1 - m) * user_feats_detach[i]
            else:
                # 首次填充
                self._user_bank[uid] = user_feats_detach[i]
                self._bank_filled[uid] = True
    
    def _get_bank_fill_ratio(self):
        """获取 bank 填充率"""
        if self._bank_filled is None:
            return 0.0
        return self._bank_filled.sum().item() / self._bank_filled.numel()
    
    def set_epoch(self, epoch):
        """
        设置当前 epoch (由 trainer 调用)
        
        Args:
            epoch: 当前 epoch 编号 (从 0 开始)
        """
        self._current_epoch = epoch
    
    def _compute_gaa_loss_from_bank(self, log_feats, kwargs):
        """
        从 user_bank 计算 GAA 损失 (不过 encoder)
        
        Args:
            log_feats: 当前用户的表示 [bs, hidden_size]
            kwargs: 包含相似用户数据的字典
                - sim_user_ids: 相似用户 ID [bs, sim_num]
                - valid_mask: 有效掩码 [bs, sim_num]
                - sim_weights (可选): 相似度权重 [bs, sim_num]
                - epoch (可选): 当前 epoch 编号
        
        Returns:
            gaa_loss: GAA 损失值
        """
        # 检查 bank 填充率
        fill_ratio = self._get_bank_fill_ratio()
        if fill_ratio < self.gaa_bank_min_fill_ratio:
            return 0.0
        
        # 检查热身期 (epoch 级别)
        # 优先使用 kwargs 中的 epoch，否则使用 set_epoch 设置的值
        current_epoch = kwargs.get('epoch', self._current_epoch)
        if current_epoch < self.gaa_warmup_epochs:
            return 0.0
        
        # 检查必要字段
        if 'sim_user_ids' not in kwargs or 'valid_mask' not in kwargs:
            return 0.0
        
        sim_user_ids = kwargs['sim_user_ids']  # [bs, sim_num]
        valid_mask = kwargs['valid_mask']       # [bs, sim_num]
        sim_weights = kwargs.get('sim_weights', None)  # [bs, sim_num] 可选
        
        bs, sim_num = sim_user_ids.shape
        hidden_size = log_feats.size(-1)
        
        # 从 bank 获取相似用户表示
        sim_user_ids_flat = sim_user_ids.view(-1).cpu()  # [bs*sim_num]
        
        # 检查 bank 中是否已填充
        bank_filled_mask = self._bank_filled[sim_user_ids_flat].view(bs, sim_num)  # [bs, sim_num]
        
        # 有效 mask = 原始 valid_mask AND bank 已填充
        effective_mask = valid_mask.cpu() & bank_filled_mask  # [bs, sim_num]
        
        # 从 bank gather 表示
        sim_feats = self._user_bank[sim_user_ids_flat].view(bs, sim_num, hidden_size)  # [bs, sim_num, H]
        
        # 转换到正确设备和精度
        if self.gaa_bank_on_cpu:
            sim_feats = sim_feats.to(log_feats.device)
        if sim_feats.dtype != log_feats.dtype:
            sim_feats = sim_feats.to(log_feats.dtype)
        effective_mask = effective_mask.to(log_feats.device)
        
        # 计算教师向量
        if sim_weights is not None:
            # 加权平均
            weights = sim_weights * effective_mask.float()  # [bs, sim_num]
            weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [bs, 1]
            weights_norm = weights / weights_sum  # [bs, sim_num]
            teacher_vec = (sim_feats * weights_norm.unsqueeze(-1)).sum(dim=1)  # [bs, H]
            valid_sample_mask = weights.sum(dim=1) > 0  # [bs]
        else:
            # 普通 masked mean
            teacher_vec, valid_sample_idx = masked_mean(sim_feats.detach(), effective_mask)
            valid_sample_mask = torch.zeros(bs, dtype=torch.bool, device=log_feats.device)
            valid_sample_mask[valid_sample_idx] = True
        
        # 计算对齐损失
        gaa_loss = 0.0
        if valid_sample_mask.any():
            gaa_loss = self.gaa_loss_func(
                log_feats[valid_sample_mask],
                teacher_vec[valid_sample_mask].detach()
            )
        
        return gaa_loss
    
    def _compute_gaa_loss_from_encoder(self, log_feats, kwargs):
        """
        从 encoder 计算 GAA 损失 (原有实现)
        
        Args:
            log_feats: 当前用户的表示 [bs, hidden_size]
            kwargs: 包含相似用户数据的字典
                - sim_seq: 相似用户序列 [bs, sim_num, seq_len]
                - sim_positions: 相似用户位置 [bs, sim_num, seq_len]
                - valid_mask: 有效掩码 [bs, sim_num]
        
        Returns:
            gaa_loss: GAA 损失值，如果没有有效相似用户则返回 0
        """
        # 检查是否有相似用户数据
        if 'sim_seq' not in kwargs or 'valid_mask' not in kwargs:
            return 0.0
        
        # 处理相似用户数据
        bs, sim_num, seq_len = kwargs["sim_seq"].shape
        sim_seq = kwargs["sim_seq"].view(-1, seq_len)
        sim_positions = kwargs["sim_positions"].view(-1, seq_len)
        valid_mask = kwargs["valid_mask"].view(-1)  # (bs*sim_num,)
        
        # 只编码有效的相似用户
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        
        hidden_size = log_feats.size(-1)
        sim_log_feats_all = torch.zeros(bs * sim_num, hidden_size, device=log_feats.device)
        
        if len(valid_idx) > 0:
            sim_seq_valid = sim_seq[valid_idx]
            sim_positions_valid = sim_positions[valid_idx]
            # 调用子类的 log2feats 方法
            valid_sim_log_feats = self._log2feats_for_gaa(sim_seq_valid, sim_positions_valid)[:, -1, :]
            sim_log_feats_all[valid_idx] = valid_sim_log_feats
        
        # 重塑并计算教师向量
        sim_log_feats_all = sim_log_feats_all.detach().view(bs, sim_num, -1)  # (bs, sim_num, hidden_size)
        valid_mask_batch = kwargs["valid_mask"].view(bs, sim_num)  # (bs, sim_num)
        
        # 计算有效相似用户的平均表示（教师向量）
        teacher_vec, valid_sample_idx = masked_mean(sim_log_feats_all, valid_mask_batch)
        
        # 计算对齐损失
        gaa_loss = 0.0
        if len(valid_sample_idx) > 0:
            gaa_loss = self.gaa_loss_func(
                log_feats[valid_sample_idx], 
                teacher_vec[valid_sample_idx]
            )
        
        return gaa_loss
    
    def _compute_gaa_loss(self, log_feats, kwargs):
        """
        计算 GAA 损失 (自动选择 encoder 或 bank 模式)
        
        Args:
            log_feats: 当前用户的表示 [bs, hidden_size]
            kwargs: 包含相似用户数据的字典
        
        Returns:
            gaa_loss: GAA 损失值
        """
        if self.gaa_use_user_bank:
            # 尝试使用 bank 模式
            if 'sim_user_ids' in kwargs:
                return self._compute_gaa_loss_from_bank(log_feats, kwargs)
            else:
                # 缺少 sim_user_ids，回退到 encoder 模式
                if not GAA_Mixin._bank_fallback_warned:
                    warnings.warn(
                        "[GAA-Bank] gaa_use_user_bank=True but 'sim_user_ids' not found in kwargs. "
                        "Falling back to encoder mode. This warning will only be shown once."
                    )
                    GAA_Mixin._bank_fallback_warned = True
                return self._compute_gaa_loss_from_encoder(log_feats, kwargs)
        else:
            # 使用 encoder 模式
            return self._compute_gaa_loss_from_encoder(log_feats, kwargs)


# ==============================================================================
# GRU4Rec + LLMEmb + GAA
# ==============================================================================

class GRU4Rec_LLMEmb_GAA(GRU4Rec_LLMEmb, GAA_Mixin):
    """
    GRU4Rec + LLMEmb + GAA
    
    在 GRU4Rec_LLMEmb 基础上添加 GAA 模块。
    """
    
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self._init_gaa(args)
        self._user_num = user_num  # 保存用户数量用于 bank 初始化
    
    def _log2feats_for_gaa(self, log_seqs, positions):
        """为 GAA 模块提供的 log2feats 接口 (GRU4Rec 不使用 positions)"""
        return self.log2feats(log_seqs)
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        """前向传播: 基础损失 + 对齐损失 + GAA 损失"""
        # 计算基础损失 + LLMEmb 对齐损失
        log_feats = self.log2feats(seq)
        log_feats_last = log_feats[:, -1, :].unsqueeze(1)
        user_feats = log_feats[:, -1, :]  # [bs, hidden_size]
        
        # 懒加载初始化 + 更新 user_bank
        if self.gaa_use_user_bank and self.gaa_alpha > 0:
            self._init_user_bank(self._user_num, user_feats.size(-1), user_feats.device)
            if 'user_ids' in kwargs:
                self._update_user_bank(kwargs['user_ids'], user_feats)
        
        pos_embs = self._get_embedding(pos.unsqueeze(1))
        neg_embs = self._get_embedding(neg)
        
        pos_logits = torch.mul(log_feats_last, pos_embs).sum(dim=-1)
        neg_logits = torch.mul(log_feats_last, neg_embs).sum(dim=-1)
        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        
        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
        base_loss = pos_loss + neg_loss
        
        # LLMEmb 对齐损失
        total_loss = base_loss
        if self.alpha > 0:
            llm_embs = self._get_embedding(pos[indices])
            srs_embs = self.srs_emb(pos[indices])
            align_loss = self.align_loss_func(srs_embs, llm_embs)
            total_loss = total_loss + self.alpha * align_loss
        
        # GAA 损失
        if self.gaa_alpha > 0:
            gaa_loss = self._compute_gaa_loss(user_feats, kwargs)
            total_loss = total_loss + self.gaa_alpha * gaa_loss
        
        return total_loss


# ==============================================================================
# SASRec + LLMEmb + GAA
# ==============================================================================

class SASRec_LLMEmb_GAA(SASRec_LLMEmb, GAA_Mixin):
    """
    SASRec + LLMEmb + GAA
    
    在 SASRec_LLMEmb 基础上添加 GAA 模块。
    """
    
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self._init_gaa(args)
        self._user_num = user_num  # 保存用户数量用于 bank 初始化
    
    def _log2feats_for_gaa(self, log_seqs, positions):
        """为 GAA 模块提供的 log2feats 接口"""
        return self.log2feats(log_seqs, positions)
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        """前向传播: 基础损失 + 对齐损失 + GAA 损失"""
        # 计算基础损失 (seq-to-seq loss)
        log_feats = self.log2feats(seq, positions)
        user_feats = log_feats[:, -1, :]  # [bs, hidden_size]
        
        # 懒加载初始化 + 更新 user_bank
        if self.gaa_use_user_bank and self.gaa_alpha > 0:
            self._init_user_bank(self._user_num, user_feats.size(-1), user_feats.device)
            if 'user_ids' in kwargs:
                self._update_user_bank(kwargs['user_ids'], user_feats)
        
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
        
        # LLMEmb 对齐损失
        total_loss = base_loss
        if self.alpha > 0:
            llm_embs = self._get_embedding(pos[indices])
            srs_embs = self.srs_emb(pos[indices])
            align_loss = self.align_loss_func(srs_embs, llm_embs)
            total_loss = total_loss + self.alpha * align_loss
        
        # GAA 损失
        if self.gaa_alpha > 0:
            gaa_loss = self._compute_gaa_loss(user_feats, kwargs)
            total_loss = total_loss + self.gaa_alpha * gaa_loss
        
        return total_loss


# ==============================================================================
# Bert4Rec + LLMEmb + GAA
# ==============================================================================

class Bert4Rec_LLMEmb_GAA(Bert4Rec_LLMEmb, GAA_Mixin):
    """
    Bert4Rec + LLMEmb + GAA
    
    在 Bert4Rec_LLMEmb 基础上添加 GAA 模块。
    """
    
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self._init_gaa(args)
        self._user_num = user_num  # 保存用户数量用于 bank 初始化
    
    def _log2feats_for_gaa(self, log_seqs, positions):
        """为 GAA 模块提供的 log2feats 接口"""
        return self.log2feats(log_seqs, positions)
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        """前向传播: 基础损失 + 对齐损失 + GAA 损失"""
        # 计算基础损失
        log_feats = self.log2feats(seq, positions)
        user_feats = log_feats[:, -1, :]  # [bs, hidden_size]
        
        # 懒加载初始化 + 更新 user_bank
        if self.gaa_use_user_bank and self.gaa_alpha > 0:
            self._init_user_bank(self._user_num, user_feats.size(-1), user_feats.device)
            if 'user_ids' in kwargs:
                self._update_user_bank(kwargs['user_ids'], user_feats)
        
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
        
        # LLMEmb 对齐损失
        total_loss = base_loss
        if self.alpha > 0:
            llm_embs = self._get_embedding(pos[indices])
            srs_embs = self.srs_emb(pos[indices])
            align_loss = self.align_loss_func(srs_embs, llm_embs)
            total_loss = total_loss + self.alpha * align_loss
        
        # GAA 损失
        if self.gaa_alpha > 0:
            gaa_loss = self._compute_gaa_loss(user_feats, kwargs)
            total_loss = total_loss + self.gaa_alpha * gaa_loss
        
        return total_loss
