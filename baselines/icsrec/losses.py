# -*- coding: utf-8 -*-
"""
ICSRec 对比学习损失函数
包含 CICL (粗粒度意图对比学习) 和 FICL (细粒度意图对比学习)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICLLoss(nn.Module):
    """
    Intent Contrastive Learning Loss
    整合 CICL 和 FICL 两种对比学习损失
    
    Args:
        temperature: 温度参数
        sim: 相似度计算方式 ('dot' 或 'cos')
        use_fnm: 是否使用 False Negative Mining
    """
    
    def __init__(self, temperature=1.0, sim='dot', use_fnm=True):
        super(ICLLoss, self).__init__()
        self.temperature = temperature
        self.sim = sim
        self.use_fnm = use_fnm
        self.ce_loss = nn.CrossEntropyLoss()
    
    def _mask_correlated_samples(self, batch_size):
        """
        创建对比学习的掩码，排除自身和正样本对
        
        Args:
            batch_size: 批次大小
            
        Returns:
            mask: [2*batch_size, 2*batch_size] 的掩码
        """
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def _mask_false_negatives(self, intent_id):
        """
        False Negative Mining: 排除相同意图的样本作为负样本
        
        Args:
            intent_id: 每个样本的意图 ID [B]
            
        Returns:
            mask: [2*B, 2*B] 的掩码，相同意图的位置为 False
        """
        label = intent_id.view(1, -1)
        label = label.expand((2, label.shape[-1])).reshape(1, -1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask == 0  # 相同意图返回 False，不同意图返回 True
    
    def info_nce(self, z_i, z_j, batch_size, intent_id=None):
        """
        InfoNCE 对比学习损失
        
        Args:
            z_i: 第一个视图的表示 [B, H]
            z_j: 第二个视图的表示 [B, H]
            batch_size: 批次大小
            intent_id: 意图 ID，用于 FNM [B]
            
        Returns:
            logits: 对比学习的 logits [2*B, 2*B-1] 或 [2*B, 2*B]
            labels: 标签 [2*B]
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)  # [2*B, H]
        
        # 计算相似度
        if self.sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        else:  # dot
            sim = torch.mm(z, z.t()) / self.temperature
        
        # 提取正样本对的相似度
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        # 处理负样本
        if self.use_fnm and intent_id is not None:
            # False Negative Mining
            mask = self._mask_false_negatives(intent_id)
            negative_samples = sim.clone()
            negative_samples[mask == 0] = float("-inf")
        else:
            # 标准对比学习
            mask = self._mask_correlated_samples(batch_size).to(z.device)
            negative_samples = sim[mask].reshape(N, -1)
        
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        return logits, labels
    
    def cicl_loss(self, seq_output_1, seq_output_2, target_item=None):
        """
        Coarse-grained Intent Contrastive Learning Loss
        粗粒度意图对比学习：拉近同一用户不同增强视图的表示
        
        Args:
            seq_output_1: 第一个视图的序列输出 [B, H]
            seq_output_2: 第二个视图的序列输出 [B, H]
            target_item: 目标物品 ID，用于 FNM [B]（可选）
            
        Returns:
            loss: CICL 损失
        """
        batch_size = seq_output_1.shape[0]
        logits, labels = self.info_nce(
            seq_output_1, 
            seq_output_2, 
            batch_size,
            intent_id=target_item
        )
        return self.ce_loss(logits, labels)
    
    def ficl_loss(self, seq_output, cluster):
        """
        Fine-grained Intent Contrastive Learning Loss
        细粒度意图对比学习：将序列表示与聚类中心对齐
        
        Args:
            seq_output: 序列输出 [B, H]
            cluster: KMeans 聚类器实例
            
        Returns:
            loss: FICL 损失
        """
        batch_size = seq_output.shape[0]
        
        # 获取序列表示对应的聚类中心
        seq_np = seq_output.detach().cpu().numpy()
        intent_id, cluster_centers = cluster.query(seq_np)
        
        # 计算序列表示与聚类中心的对比损失
        logits, labels = self.info_nce(
            seq_output.view(batch_size, -1),
            cluster_centers.view(batch_size, -1),
            batch_size,
            intent_id=intent_id
        )
        
        return self.ce_loss(logits, labels)
    
    def forward(self, seq_output_1, seq_output_2, cluster=None, 
                target_item=None, lambda_0=0.1, beta_0=0.1, cl_mode='cf'):
        """
        计算总的意图对比学习损失
        
        Args:
            seq_output_1: 第一个视图的序列输出 [B, L, H] 或 [B, H]
            seq_output_2: 第二个视图的序列输出 [B, L, H] 或 [B, H]
            cluster: KMeans 聚类器实例（用于 FICL）
            target_item: 目标物品 ID [B]（用于 FNM）
            lambda_0: CICL 损失权重
            beta_0: FICL 损失权重
            cl_mode: 对比学习模式 ('c': 仅 CICL, 'f': 仅 FICL, 'cf': 两者都有)
            
        Returns:
            loss: 总的对比学习损失
        """
        # 提取最后一个时间步的表示（如果是序列）
        if seq_output_1.dim() == 3:
            seq_output_1 = seq_output_1[:, -1, :]
        if seq_output_2.dim() == 3:
            seq_output_2 = seq_output_2[:, -1, :]
        
        loss = 0.0
        
        # CICL 损失
        if cl_mode in ['c', 'cf']:
            cicl = self.cicl_loss(seq_output_1, seq_output_2, target_item)
            loss += lambda_0 * cicl
        
        # FICL 损失
        if cl_mode in ['f', 'cf'] and cluster is not None:
            ficl_1 = self.ficl_loss(seq_output_1, cluster)
            ficl_2 = self.ficl_loss(seq_output_2, cluster)
            loss += beta_0 * (ficl_1 + ficl_2)
        
        return loss


class RecommendationLoss(nn.Module):
    """
    推荐任务损失（交叉熵损失）
    """
    
    def __init__(self):
        super(RecommendationLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, target):
        """
        计算推荐损失
        
        Args:
            logits: 预测分数 [B, num_items]
            target: 目标物品 ID [B]
            
        Returns:
            loss: 交叉熵损失
        """
        return self.ce_loss(logits, target)

