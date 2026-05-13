# -*- coding: utf-8 -*-
"""
ICSRec 聚类模块
用于学习细粒度意图表示 (Fine-grained Intent Contrastive Learning)
"""

import numpy as np
import torch
import torch.nn as nn
import faiss


class KMeans:
    """
    基于 Faiss 的 GPU 加速 KMeans 聚类
    用于 ICSRec 中的细粒度意图对比学习 (FICL)
    
    Args:
        num_cluster: 聚类中心数量
        seed: 随机种子
        hidden_size: 隐藏层维度
        gpu_id: GPU 设备 ID
        device: torch 设备
    """
    
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = gpu_id
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self._init_cluster(self.hidden_size)
        self.centroids = None
    
    def _init_cluster(self, hidden_size, verbose=False, niter=20, nredo=5,
                      max_points_per_centroid=4096, min_points_per_centroid=0):
        """初始化 Faiss 聚类器"""
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid
        
        # 使用 GPU 加速
        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        
        return clus, index
    
    def train(self, x):
        """
        训练聚类模型
        
        Args:
            x: 输入特征 [N, hidden_size]，numpy array
        """
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        
        # 获取聚类中心
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(
            self.num_cluster, self.hidden_size
        )
        # 转换为 torch tensor 并归一化
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)
    
    def query(self, x):
        """
        查询最近的聚类中心
        
        Args:
            x: 输入特征 [B, hidden_size]，numpy array
            
        Returns:
            seq2cluster: 每个样本对应的聚类 ID [B]
            cluster_centers: 对应的聚类中心 [B, hidden_size]
        """
        D, I = self.index.search(x, 1)  # 找到每个样本最近的聚类中心
        seq2cluster = [int(n[0]) for n in I]
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        
        return seq2cluster, self.centroids[seq2cluster]


class CPUKMeans:
    """
    CPU 版本的 KMeans（用于不支持 GPU Faiss 的环境）
    """
    
    def __init__(self, num_cluster, seed, hidden_size, device="cpu"):
        self.num_cluster = num_cluster
        self.seed = seed
        self.hidden_size = hidden_size
        self.device = device
        self.centroids = None
        self.index = None
    
    def _init_index(self):
        """初始化 CPU Faiss 索引"""
        self.index = faiss.IndexFlatL2(self.hidden_size)
    
    def train(self, x):
        """
        训练聚类模型 (CPU 版本)
        
        Args:
            x: 输入特征 [N, hidden_size]，numpy array
        """
        if x.shape[0] < self.num_cluster:
            return
        
        # 使用 CPU KMeans
        clus = faiss.Kmeans(
            self.hidden_size, 
            self.num_cluster, 
            niter=20, 
            verbose=False, 
            seed=self.seed
        )
        clus.train(x)
        
        # 获取聚类中心
        centroids = torch.Tensor(clus.centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)
        
        # 更新索引
        self._init_index()
        self.index.add(clus.centroids)
    
    def query(self, x):
        """
        查询最近的聚类中心 (CPU 版本)
        
        Args:
            x: 输入特征 [B, hidden_size]，numpy array
            
        Returns:
            seq2cluster: 每个样本对应的聚类 ID [B]
            cluster_centers: 对应的聚类中心 [B, hidden_size]
        """
        if self.index is None or self.centroids is None:
            raise RuntimeError("Must call train() before query()")
        
        D, I = self.index.search(x, 1)
        seq2cluster = [int(n[0]) for n in I]
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        
        return seq2cluster, self.centroids[seq2cluster]


def create_kmeans(num_cluster, seed, hidden_size, gpu_id=0, device="cpu", use_gpu=True):
    """
    工厂函数：根据环境创建合适的 KMeans 实例
    
    Args:
        num_cluster: 聚类中心数量
        seed: 随机种子
        hidden_size: 隐藏层维度
        gpu_id: GPU 设备 ID
        device: torch 设备
        use_gpu: 是否使用 GPU 加速
        
    Returns:
        KMeans 实例
    """
    if use_gpu:
        try:
            return KMeans(num_cluster, seed, hidden_size, gpu_id, device)
        except Exception as e:
            print(f"Failed to create GPU KMeans: {e}")
            print("Falling back to CPU KMeans")
            return CPUKMeans(num_cluster, seed, hidden_size, device)
    else:
        return CPUKMeans(num_cluster, seed, hidden_size, device)

