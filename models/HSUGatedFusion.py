# HSUGatedFusion.py
# HSU (Hierarchical Semantic User) 门控融合模块
# 用于将离线计算的 HSU 用户语义向量注入到 RCL 的用户表示中
"""
B2 门控融合公式实现：

(1) 维度对齐投影（可训练）：
    h_hsu = W_proj(h_hsu_raw)  # 若 D_h != D

(2) 门控 g 的两种形式：
    A) 标量门控（最稳，推荐默认）：
       g 是一个可学习标量参数 λ（或 sigmoid(λ)），初始化为 0
       h_fused = LN( h_base + g * h_hsu )
       
    B) 向量门控（效果更强，但更复杂）：
       g = sigmoid( W_gate([h_base ; h_hsu]) )  # concat 后线性层，输出 [B, D]
       h_fused = LN( h_base + g ⊙ h_hsu )

核心设计原则：
- 可插拔：开关关闭时，模型行为与原 RCL 完全一致
- 平滑退化：初始时 gate≈0，避免训练不稳定
- HSU 不训练：HSU 向量来自离线文件，不对其反向传播
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HSUGatedFusion(nn.Module):
    """
    HSU 门控融合模块
    
    输入：
        h_base: 序列编码器输出的用户表示，shape [B, D]
        user_indices: 用户索引，用于从 HSU bank 查表，shape [B]
    
    输出：
        h_fused: 融合后的用户表示，shape [B, D]
    
    公式对应关系：
        A) 标量门控：h_fused = LN( h_base + sigmoid(gate_scalar) * h_hsu )
        B) 向量门控：h_fused = LN( h_base + sigmoid(W_gate([h_base; h_hsu])) ⊙ h_hsu )
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 hsu_dim: int,
                 gate_type: str = "scalar",
                 gate_init: float = 0.0,
                 fusion_dropout: float = 0.1,
                 use_hsu_layernorm: bool = False):
        """
        Args:
            hidden_dim: 模型隐藏层维度 D (h_base 的维度)
            hsu_dim: HSU 向量原始维度 D_h
            gate_type: 门控类型，"scalar" 或 "vector"
            gate_init: 门控初始化值，默认 0.0（确保退化到原模型）
            fusion_dropout: 融合时的 dropout 概率
            use_hsu_layernorm: 是否对 h_hsu 额外做 LayerNorm
        """
        super(HSUGatedFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.hsu_dim = hsu_dim
        self.gate_type = gate_type
        self.gate_init = gate_init
        
        # === 维度对齐投影 ===
        # 即使 D_h == D，也保留投影层便于空间/尺度适配
        self.proj = nn.Linear(hsu_dim, hidden_dim)
        
        # === 门控机制 ===
        if gate_type == "scalar":
            # A) 标量门控：可学习的标量参数
            # 初始化为 gate_init (默认 0)，sigmoid(0) = 0.5
            # 为了真正退化，我们用一个很负的值初始化，比如 -10，这样 sigmoid(-10) ≈ 0
            # 或者直接用 gate_init 作为 raw 值，然后 scale
            self.gate_scalar = nn.Parameter(torch.tensor(gate_init))
        elif gate_type == "vector":
            # B) 向量门控：[h_base; h_hsu] -> W -> sigmoid -> [B, D]
            self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
            # 初始化 bias 使得初始输出接近 0
            nn.init.zeros_(self.gate_linear.weight)
            nn.init.constant_(self.gate_linear.bias, gate_init)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}, expected 'scalar' or 'vector'")
        
        # === 后处理层 ===
        self.fusion_dropout = nn.Dropout(p=fusion_dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 可选：对 h_hsu 也做 LayerNorm
        self.use_hsu_layernorm = use_hsu_layernorm
        if use_hsu_layernorm:
            self.hsu_layer_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 投影层使用 xavier 初始化
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, h_base: torch.Tensor, h_hsu_raw: torch.Tensor) -> torch.Tensor:
        """
        门控融合前向传播
        
        Args:
            h_base: 序列编码器输出的用户表示，[B, D]
            h_hsu_raw: 从 HSU bank 查表得到的原始 HSU 向量，[B, D_h]
                       注意：h_hsu_raw 应该已经 detach，不参与反向传播
        
        Returns:
            h_fused: 融合后的用户表示，[B, D]
        """
        # (1) 维度对齐投影
        h_hsu = self.proj(h_hsu_raw)  # [B, D]
        
        # 可选：对 h_hsu 做 LayerNorm
        if self.use_hsu_layernorm:
            h_hsu = self.hsu_layer_norm(h_hsu)
        
        # 对 h_hsu 应用 dropout
        h_hsu = self.fusion_dropout(h_hsu)
        
        # (2) 计算门控 g
        if self.gate_type == "scalar":
            # A) 标量门控
            g = torch.sigmoid(self.gate_scalar)  # scalar in (0, 1)
            fusion_term = g * h_hsu  # [B, D]
        else:
            # B) 向量门控
            concat_feat = torch.cat([h_base, h_hsu], dim=-1)  # [B, 2D]
            g = torch.sigmoid(self.gate_linear(concat_feat))  # [B, D]
            fusion_term = g * h_hsu  # [B, D]，逐元素乘
        
        # (3) 融合 + LayerNorm
        h_fused = self.layer_norm(h_base + fusion_term)
        
        return h_fused
    
    def get_gate_value(self) -> torch.Tensor:
        """获取当前门控值，用于监控/debug"""
        if self.gate_type == "scalar":
            return torch.sigmoid(self.gate_scalar).item()
        else:
            # 向量门控返回平均值
            return None  # 向量门控没有单一值


class HSUBank:
    """
    HSU Bank 管理类
    
    负责加载和管理离线计算的 HSU 用户语义向量
    支持 user-level 和 sample-level 两种索引方式
    """
    
    def __init__(self, 
                 hsu_bank_path: str,
                 device: torch.device,
                 index_type: str = "user"):
        """
        Args:
            hsu_bank_path: HSU bank 文件路径 (pkl 或 npy)
            device: 目标设备 (用于转移 batch 数据)
            index_type: 索引类型，"user" (按 user_id) 或 "sample" (按 sample_idx)
        """
        self.device = device
        self.index_type = index_type
        
        # 加载 HSU bank
        self._load_bank(hsu_bank_path)
    
    def _load_bank(self, hsu_bank_path: str):
        """加载 HSU bank 到内存"""
        if not os.path.exists(hsu_bank_path):
            raise FileNotFoundError(f"HSU bank file not found: {hsu_bank_path}")
        
        # 根据文件扩展名选择加载方式
        if hsu_bank_path.endswith('.pkl'):
            with open(hsu_bank_path, 'rb') as f:
                data = pickle.load(f)
        elif hsu_bank_path.endswith('.npy'):
            data = np.load(hsu_bank_path)
        elif hsu_bank_path.endswith('.pt'):
            data = torch.load(hsu_bank_path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported file format: {hsu_bank_path}")
        
        # 转换为 numpy array（统一处理）
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        # 转换为 torch tensor 并保持在 CPU（节省 GPU 内存）
        self.bank = torch.from_numpy(data).float()
        
        self.num_entries = self.bank.shape[0]
        self.hsu_dim = self.bank.shape[1]
    
    def get_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """
        根据索引获取 HSU embeddings
        
        Args:
            indices: 索引张量，[B]
        
        Returns:
            HSU embeddings，[B, D_h]，已 detach（不参与反向传播）
        """
        # 确保 indices 是 CPU tensor
        if indices.is_cuda:
            indices_cpu = indices.cpu()
        else:
            indices_cpu = indices
        
        # 从 bank 查表
        h_hsu_raw = self.bank[indices_cpu]  # [B, D_h]
        
        # 移动到目标设备，并 detach（HSU 不参与训练）
        h_hsu_raw = h_hsu_raw.to(self.device).detach()
        
        return h_hsu_raw
    
    @property
    def embedding_dim(self) -> int:
        """返回 HSU 向量维度"""
        return self.hsu_dim


# ============================================================================
# 升级版融合方法 (HSUAdvancedFusion)
# ============================================================================

class HSUAdvancedFusion(nn.Module):
    """
    HSU 高级融合模块 - 提供多种可选的融合策略
    
    相比基础版 HSUGatedFusion，新增以下融合方式：
    1. attention: 注意力加权融合
    2. mlp: MLP 非线性融合
    3. adaptive: 自适应门控（基于 h_base 的 norm 动态调整）
    4. cross_attention: 交叉注意力融合
    5. bilinear: 双线性交互融合
    6. residual_mlp: 残差 MLP 融合
    
    设计原则：
    - 向后兼容：fusion_type="scalar" 或 "vector" 时行为与原 HSUGatedFusion 一致
    - 可插拔：通过 args.hsu_fusion_type 选择融合方式
    """
    
    def __init__(self,
                 hidden_dim: int,
                 hsu_dim: int,
                 fusion_type: str = "scalar",
                 gate_init: float = 0.0,
                 fusion_dropout: float = 0.1,
                 num_heads: int = 4,
                 mlp_ratio: float = 2.0,
                 use_hsu_layernorm: bool = False,
                 proj_type: str = "linear",
                 proj_bottleneck_dim: int = 512):
        """
        Args:
            hidden_dim: 模型隐藏层维度 D
            hsu_dim: HSU 向量原始维度 D_h
            fusion_type: 融合类型
                - "scalar": 标量门控（基础版）
                - "vector": 向量门控（基础版）
                - "attention": 注意力融合
                - "mlp": MLP 融合
                - "adaptive": 自适应门控
                - "cross_attention": 交叉注意力
                - "bilinear": 双线性融合
                - "residual_mlp": 残差 MLP
            gate_init: 门控初始化值
            fusion_dropout: 融合 dropout
            num_heads: 注意力头数（用于 attention/cross_attention）
            mlp_ratio: MLP 扩展比例
            use_hsu_layernorm: 是否对 h_hsu 做 LayerNorm
            proj_type: 投影类型
                - "linear": 单层线性投影（默认，参数多）
                - "bottleneck": 瓶颈结构 hsu_dim -> bottleneck -> hidden_dim（渐进压缩）
                - "lowrank": 低秩分解（参数效率高）
            proj_bottleneck_dim: 瓶颈层维度（用于 bottleneck/lowrank），默认 512
        """
        super(HSUAdvancedFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.hsu_dim = hsu_dim
        self.fusion_type = fusion_type
        self.gate_type = fusion_type  # 兼容性别名，trainer 中使用 gate_type
        self.gate_init = gate_init
        self.proj_type = proj_type
        
        # === 维度对齐投影（根据 proj_type 选择）===
        if proj_type == "linear":
            # 单层线性投影（原始方式）
            self.proj = nn.Linear(hsu_dim, hidden_dim)
        elif proj_type == "bottleneck":
            # 瓶颈结构：渐进式压缩，更平滑
            # hsu_dim -> bottleneck_dim -> hidden_dim
            self.proj = nn.Sequential(
                nn.Linear(hsu_dim, proj_bottleneck_dim),
                nn.LayerNorm(proj_bottleneck_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(proj_bottleneck_dim, hidden_dim)
            )
        elif proj_type == "lowrank":
            # 低秩分解：参数效率高
            # 将 W: hsu_dim -> hidden_dim 分解为 W1 @ W2
            # 参数量从 hsu_dim * hidden_dim 降到 hsu_dim * k + k * hidden_dim
            self.proj_down = nn.Linear(hsu_dim, proj_bottleneck_dim, bias=False)
            self.proj_up = nn.Linear(proj_bottleneck_dim, hidden_dim, bias=True)
            self.proj = lambda x: self.proj_up(self.proj_down(x))
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}, expected 'linear', 'bottleneck', or 'lowrank'")
        
        # === 根据 fusion_type 初始化不同模块 ===
        if fusion_type == "scalar":
            # 基础版：标量门控
            self.gate_scalar = nn.Parameter(torch.tensor(gate_init))
            
        elif fusion_type == "vector":
            # 基础版：向量门控
            self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
            nn.init.zeros_(self.gate_linear.weight)
            nn.init.constant_(self.gate_linear.bias, gate_init)
            
        elif fusion_type == "attention":
            # 注意力融合：[h_base, h_hsu] -> softmax attention -> weighted sum
            self.attn_query = nn.Linear(hidden_dim, hidden_dim)
            self.attn_key = nn.Linear(hidden_dim, hidden_dim)
            self.attn_value = nn.Linear(hidden_dim, hidden_dim)
            self.attn_scale = hidden_dim ** 0.5
            
        elif fusion_type == "mlp":
            # MLP 融合：concat -> MLP -> output
            mlp_hidden = int(hidden_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, mlp_hidden),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(mlp_hidden, hidden_dim),
                nn.Dropout(fusion_dropout)
            )
            
        elif fusion_type == "adaptive":
            # 自适应门控：根据 h_base 的特征自动调整融合强度
            # g = sigmoid(W_adapt(h_base) + bias)，bias 初始化为 gate_init
            self.adaptive_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
            # 初始化最后一层 bias
            nn.init.constant_(self.adaptive_gate[-1].bias, gate_init)
            
        elif fusion_type == "cross_attention":
            # 交叉注意力：h_base 作为 query，h_hsu 作为 key/value
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            assert hidden_dim % num_heads == 0
            
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
            
            # 门控：控制 cross-attention 输出的影响
            self.ca_gate = nn.Parameter(torch.tensor(gate_init))
            
        elif fusion_type == "bilinear":
            # 双线性融合：h_base^T W h_hsu 捕捉交互
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
            self.bilinear_gate = nn.Parameter(torch.tensor(gate_init))
            
        elif fusion_type == "residual_mlp":
            # 残差 MLP：两路分别处理，然后门控融合
            self.base_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout)
            )
            self.hsu_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout)
            )
            self.fusion_gate = nn.Parameter(torch.tensor(gate_init))
            
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # === 公共后处理层 ===
        self.fusion_dropout_layer = nn.Dropout(p=fusion_dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.use_hsu_layernorm = use_hsu_layernorm
        if use_hsu_layernorm:
            self.hsu_layer_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 根据 proj_type 初始化投影层
        if self.proj_type == "linear":
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif self.proj_type == "bottleneck":
            # Sequential 中的 Linear 层
            for module in self.proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif self.proj_type == "lowrank":
            nn.init.xavier_uniform_(self.proj_down.weight)
            nn.init.xavier_uniform_(self.proj_up.weight)
            nn.init.zeros_(self.proj_up.bias)
    
    def forward(self, h_base: torch.Tensor, h_hsu_raw: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h_base: 序列编码器输出 [B, D]
            h_hsu_raw: HSU 原始向量 [B, D_h]，已 detach
        
        Returns:
            h_fused: 融合后的表示 [B, D]
        """
        # (1) 维度对齐
        h_hsu = self.proj(h_hsu_raw)  # [B, D]
        
        if self.use_hsu_layernorm:
            h_hsu = self.hsu_layer_norm(h_hsu)
        
        h_hsu = self.fusion_dropout_layer(h_hsu)
        
        # (2) 根据 fusion_type 执行不同融合策略
        if self.fusion_type == "scalar":
            g = torch.sigmoid(self.gate_scalar)
            h_fused = h_base + g * h_hsu
            
        elif self.fusion_type == "vector":
            concat_feat = torch.cat([h_base, h_hsu], dim=-1)
            g = torch.sigmoid(self.gate_linear(concat_feat))
            h_fused = h_base + g * h_hsu
            
        elif self.fusion_type == "attention":
            # 注意力融合
            q = self.attn_query(h_base)  # [B, D]
            k = self.attn_key(h_hsu)  # [B, D]
            v = self.attn_value(h_hsu)  # [B, D]
            
            # 计算注意力分数 (简化版：只有两个向量)
            attn_score = (q * k).sum(dim=-1, keepdim=True) / self.attn_scale  # [B, 1]
            attn_weight = torch.sigmoid(attn_score)  # 用 sigmoid 代替 softmax（只有一个值）
            
            h_fused = h_base + attn_weight * v
            
        elif self.fusion_type == "mlp":
            # MLP 融合
            concat_feat = torch.cat([h_base, h_hsu], dim=-1)  # [B, 2D]
            h_fused = h_base + self.mlp(concat_feat)  # 残差连接
            
        elif self.fusion_type == "adaptive":
            # 自适应门控：根据 h_base 的"确定性"调整融合
            g = torch.sigmoid(self.adaptive_gate(h_base))  # [B, 1]
            h_fused = h_base + g * h_hsu
            
        elif self.fusion_type == "cross_attention":
            # 交叉注意力
            B = h_base.shape[0]
            
            q = self.q_proj(h_base).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(h_hsu).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(h_hsu).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            
            out = (attn @ v).transpose(1, 2).contiguous().view(B, self.hidden_dim)
            out = self.out_proj(out)
            
            g = torch.sigmoid(self.ca_gate)
            h_fused = h_base + g * out
            
        elif self.fusion_type == "bilinear":
            # 双线性融合
            interaction = self.bilinear(h_base, h_hsu)  # [B, D]
            g = torch.sigmoid(self.bilinear_gate)
            h_fused = h_base + g * interaction
            
        elif self.fusion_type == "residual_mlp":
            # 残差 MLP
            h_base_processed = self.base_mlp(h_base)
            h_hsu_processed = self.hsu_mlp(h_hsu)
            g = torch.sigmoid(self.fusion_gate)
            h_fused = h_base + h_base_processed + g * h_hsu_processed
        
        # (3) LayerNorm
        h_fused = self.layer_norm(h_fused)
        
        return h_fused
    
    def get_gate_value(self):
        """获取门控值（用于监控）"""
        if self.fusion_type == "scalar":
            return torch.sigmoid(self.gate_scalar).item()
        elif self.fusion_type in ["cross_attention", "bilinear", "residual_mlp"]:
            gate_param = getattr(self, f"{self.fusion_type.split('_')[0]}_gate", None)
            if gate_param is None:
                gate_param = getattr(self, "ca_gate", None) or getattr(self, "bilinear_gate", None) or getattr(self, "fusion_gate", None)
            if gate_param is not None:
                return torch.sigmoid(gate_param).item()
        return None


# ============================================================================
# HSU 融合 Mixin 类（通用模块，供 RCL 和 ICSRec 等模型使用）
# ============================================================================

# 基础融合类型（使用 HSUGatedFusion）
BASIC_FUSION_TYPES = {"scalar", "vector"}
# 高级融合类型（使用 HSUAdvancedFusion）
ADVANCED_FUSION_TYPES = {"attention", "mlp", "adaptive", "cross_attention", "bilinear", "residual_mlp"}


class HSUFusionMixin:
    """
    HSU 融合功能的 Mixin 类（通用模块）
    
    提供 HSU 门控融合的初始化和调用接口，供 RCL、ICSRec 等模型类继承使用。
    实现可插拔设计：当 use_hsu_fusion=False 时，模型行为与原模型完全一致。
    
    支持的融合类型 (hsu_fusion_type):
    - 基础版 (HSUGatedFusion):
        - "scalar": 标量门控
        - "vector": 向量门控
    - 高级版 (HSUAdvancedFusion):
        - "attention": 注意力融合
        - "mlp": MLP 非线性融合
        - "adaptive": 自适应门控
        - "cross_attention": 交叉注意力
        - "bilinear": 双线性交互
        - "residual_mlp": 残差 MLP
    """
    
    def _init_hsu_fusion(self, args, device):
        """
        初始化 HSU 融合模块
        
        Args:
            args: 命令行参数，包含 HSU 相关配置
            device: 目标设备
        
        相关参数:
            args.use_hsu_fusion: 是否启用 HSU 融合
            args.hsu_bank_path: HSU bank 文件路径
            args.hsu_dim: HSU 向量维度
            args.hsu_fusion_type: 融合类型 (scalar/vector/attention/mlp/adaptive/cross_attention/bilinear/residual_mlp)
            args.hsu_gate_type: [兼容旧参数] 等价于 hsu_fusion_type
            args.hsu_gate_init: 门控初始化值
            args.hsu_fusion_dropout: 融合 dropout
            args.hsu_num_heads: 注意力头数 (用于 attention/cross_attention)
            args.hsu_mlp_ratio: MLP 扩展比例 (用于 mlp/residual_mlp)
            args.hsu_proj_type: 投影类型 (linear/bottleneck/lowrank)
            args.hsu_proj_bottleneck_dim: 瓶颈层维度
            args.use_hsu_pca: 是否启用 PCA 降维分支
            args.hsu_pca_dim: PCA 降维后的维度
        """
        # 从 args 获取配置
        self.use_hsu_fusion = getattr(args, 'use_hsu_fusion', False)
        
        if not self.use_hsu_fusion:
            self.hsu_bank = None
            self.hsu_fusion = None
            return
        
        # HSU 配置
        hsu_bank_path = getattr(args, 'hsu_bank_path', None)
        hsu_dim = getattr(args, 'hsu_dim', 3584)
        
        # === PCA 降维分支 ===
        # 动机：将高维静态语义向量(3584维)压缩到更紧凑的语义子空间，
        #      降低融合参数量、提升训练稳定性，减少对主 ranking 学习的干扰
        # 公平性约束：PCA 拟合只使用训练集数据（通过 data/pca_reduce_hsu.py 预处理）
        use_hsu_pca = getattr(args, 'use_hsu_pca', False)
        hsu_pca_dim = getattr(args, 'hsu_pca_dim', 512)
        self.use_hsu_pca = use_hsu_pca  # 保存到实例，便于后续监控
        
        if use_hsu_pca and hsu_bank_path is not None:
            # 自动切换到 PCA 降维后的文件
            # 例如: usr_emb_np_qwen_mean.pkl -> usr_emb_np_qwen_mean_pca512.pkl
            import re
            pca_path = re.sub(r'\.pkl$', f'_pca{hsu_pca_dim}.pkl', hsu_bank_path)
            
            if os.path.exists(pca_path):
                hsu_bank_path = pca_path
                hsu_dim = hsu_pca_dim  # 更新维度为 PCA 后的维度
                print(f"[HSU] PCA enabled: loading {pca_path} (dim={hsu_pca_dim})")
            else:
                print(f"[HSU] Warning: PCA file not found: {pca_path}")
                print(f"[HSU] Please run: python data/pca_reduce_hsu.py --target_dim {hsu_pca_dim}")
                print(f"[HSU] Falling back to original embeddings: {hsu_bank_path}")
                use_hsu_pca = False  # 回退到原始向量
        
        # 兼容旧参数：优先使用 hsu_fusion_type，否则使用 hsu_gate_type
        fusion_type = getattr(args, 'hsu_fusion_type', None)
        if fusion_type is None:
            fusion_type = getattr(args, 'hsu_gate_type', 'scalar')
        
        gate_init = getattr(args, 'hsu_gate_init', -10.0)  # sigmoid(-10) ≈ 0
        fusion_dropout = getattr(args, 'hsu_fusion_dropout', 0.1)
        num_heads = getattr(args, 'hsu_num_heads', 4)
        mlp_ratio = getattr(args, 'hsu_mlp_ratio', 2.0)
        use_hsu_layernorm = getattr(args, 'hsu_use_layernorm', False)
        proj_type = getattr(args, 'hsu_proj_type', 'linear')
        proj_bottleneck_dim = getattr(args, 'hsu_proj_bottleneck_dim', 512)
        
        if hsu_bank_path is None:
            raise ValueError("hsu_bank_path must be specified when use_hsu_fusion=True")
        
        # 加载 HSU bank
        self.hsu_bank = HSUBank(
            hsu_bank_path=hsu_bank_path,
            device=device,
            index_type="user"
        )
        
        # 验证 HSU 维度
        if self.hsu_bank.hsu_dim != hsu_dim:
            print(f"[Warning] hsu_dim mismatch: args={hsu_dim}, bank={self.hsu_bank.hsu_dim}. Using bank dim.")
            hsu_dim = self.hsu_bank.hsu_dim
        
        # 根据 fusion_type 选择融合模块
        if fusion_type in BASIC_FUSION_TYPES:
            # 使用基础版 HSUGatedFusion（保持向后兼容）
            self.hsu_fusion = HSUGatedFusion(
                hidden_dim=args.hidden_size,
                hsu_dim=hsu_dim,
                gate_type=fusion_type,
                gate_init=gate_init,
                fusion_dropout=fusion_dropout,
                use_hsu_layernorm=use_hsu_layernorm
            )
            print(f"[HSU] Using basic fusion: {fusion_type}")
        elif fusion_type in ADVANCED_FUSION_TYPES:
            # 使用高级版 HSUAdvancedFusion
            self.hsu_fusion = HSUAdvancedFusion(
                hidden_dim=args.hidden_size,
                hsu_dim=hsu_dim,
                fusion_type=fusion_type,
                gate_init=gate_init,
                fusion_dropout=fusion_dropout,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_hsu_layernorm=use_hsu_layernorm,
                proj_type=proj_type,
                proj_bottleneck_dim=proj_bottleneck_dim
            )
            print(f"[HSU] Using advanced fusion: {fusion_type}, proj_type: {proj_type}, bottleneck_dim: {proj_bottleneck_dim}")
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. "
                           f"Expected one of {BASIC_FUSION_TYPES | ADVANCED_FUSION_TYPES}")
    
    def _apply_hsu_fusion(self, h_base, user_indices=None):
        """
        应用 HSU 门控融合
        
        Args:
            h_base: 序列编码器输出的用户表示，[B, D]
            user_indices: 用户索引，[B]。如果为 None 且需要融合，则跳过
        
        Returns:
            h_fused: 融合后的用户表示（如果启用），或原始 h_base（如果未启用）
        """
        if not self.use_hsu_fusion or self.hsu_fusion is None:
            return h_base
        
        if user_indices is None:
            # 没有 user_indices 时，无法查表，退化为原模型
            return h_base
        
        # 从 HSU bank 获取向量（已 detach，不参与反向传播）
        h_hsu_raw = self.hsu_bank.get_embeddings(user_indices)
        
        # 门控融合
        h_fused = self.hsu_fusion(h_base, h_hsu_raw)
        
        return h_fused
    
    def get_hsu_gate_value(self):
        """获取当前 HSU 门控值（用于监控）"""
        if not self.use_hsu_fusion or self.hsu_fusion is None:
            return None
        return self.hsu_fusion.get_gate_value()


# ============================================================================
# 自测函数
# ============================================================================

def _test_gated_fusion():
    """
    自测：验证 GatedFusion 模块的正确性
    
    检查项：
    1. 输出 shape 正确
    2. gate=0 时退化到 LN(h_base)
    3. 反向传播正确：encoder 参数有梯度，HSU bank 无梯度
    """
    print("=" * 60)
    print("Testing HSUGatedFusion Module")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    hidden_dim = 64
    hsu_dim = 3584
    
    # ===== Test 1: 标量门控 =====
    print("\n[Test 1] Scalar Gate Fusion")
    
    fusion_scalar = HSUGatedFusion(
        hidden_dim=hidden_dim,
        hsu_dim=hsu_dim,
        gate_type="scalar",
        gate_init=-10.0,  # sigmoid(-10) ≈ 0
        fusion_dropout=0.0  # 测试时关闭 dropout
    ).to(device)
    
    # 随机输入
    h_base = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)
    h_hsu_raw = torch.randn(batch_size, hsu_dim, device=device)  # 模拟从 bank 获取（detach）
    
    # 前向
    h_fused = fusion_scalar(h_base, h_hsu_raw.detach())
    
    # 检查 shape
    assert h_fused.shape == (batch_size, hidden_dim), \
        f"Shape mismatch: expected {(batch_size, hidden_dim)}, got {h_fused.shape}"
    print(f"  ✓ Output shape correct: {h_fused.shape}")
    
    # 检查 gate 值
    gate_val = fusion_scalar.get_gate_value()
    print(f"  Gate value (should be ~0): {gate_val:.6f}")
    assert gate_val < 0.001, f"Gate should be ~0 for gate_init=-10, got {gate_val}"
    print(f"  ✓ Gate value correct (退化验证通过)")
    
    # 检查退化：当 gate ≈ 0 时，h_fused ≈ LN(h_base)
    with torch.no_grad():
        expected = fusion_scalar.layer_norm(h_base)
        diff = (h_fused - expected).abs().max().item()
        print(f"  h_fused vs LN(h_base) max diff: {diff:.6f}")
        assert diff < 0.01, f"Should degrade to LN(h_base) when gate=0, diff={diff}"
    print(f"  ✓ Degradation check passed")
    
    # ===== Test 2: 向量门控 =====
    print("\n[Test 2] Vector Gate Fusion")
    
    fusion_vector = HSUGatedFusion(
        hidden_dim=hidden_dim,
        hsu_dim=hsu_dim,
        gate_type="vector",
        gate_init=-10.0,
        fusion_dropout=0.0
    ).to(device)
    
    h_fused_v = fusion_vector(h_base, h_hsu_raw.detach())
    assert h_fused_v.shape == (batch_size, hidden_dim)
    print(f"  ✓ Output shape correct: {h_fused_v.shape}")
    
    # ===== Test 3: 反向传播检查 =====
    print("\n[Test 3] Backward Pass Check")
    
    # 模拟 encoder 参数
    encoder_param = nn.Linear(hidden_dim, hidden_dim).to(device)
    h_base_from_encoder = encoder_param(torch.randn(batch_size, hidden_dim, device=device))
    
    # HSU 向量（从 bank 获取，应该 detach）
    h_hsu_raw_detached = torch.randn(batch_size, hsu_dim, device=device).detach()
    
    # 融合
    h_fused_test = fusion_scalar(h_base_from_encoder, h_hsu_raw_detached)
    
    # 反向传播
    loss = h_fused_test.sum()
    loss.backward()
    
    # 检查梯度
    # 1) encoder 参数应该有梯度
    assert encoder_param.weight.grad is not None, "Encoder params should have gradients"
    print(f"  ✓ Encoder params have gradients")
    
    # 2) 投影层应该有梯度
    assert fusion_scalar.proj.weight.grad is not None, "Proj layer should have gradients"
    print(f"  ✓ Proj layer has gradients")
    
    # 3) 门控参数应该有梯度
    assert fusion_scalar.gate_scalar.grad is not None, "Gate scalar should have gradients"
    print(f"  ✓ Gate scalar has gradients")
    
    # 4) HSU raw 不应该有梯度（已 detach）
    assert not h_hsu_raw_detached.requires_grad, "HSU raw should be detached"
    print(f"  ✓ HSU bank vectors are detached (no gradients)")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def _test_hsu_bank():
    """测试 HSU Bank 加载"""
    print("\n[Test HSU Bank]")
    
    # 创建临时测试文件
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        test_data = np.random.randn(100, 3584).astype(np.float32)
        np.save(f.name, test_data)
        temp_path = f.name
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bank = HSUBank(temp_path, device)
        
        print(f"  ✓ Bank loaded: {bank.num_entries} entries, dim={bank.hsu_dim}")
        
        # 测试查表
        indices = torch.tensor([0, 5, 10, 50])
        embeddings = bank.get_embeddings(indices)
        
        assert embeddings.shape == (4, 3584)
        assert embeddings.device == device
        print(f"  ✓ Embedding retrieval works: shape={embeddings.shape}")
        
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    _test_gated_fusion()
    _test_hsu_bank()

