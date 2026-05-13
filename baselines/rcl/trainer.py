# rcl_trainer.py
# RCL 专用的 Trainer，包含 SSL 损失计算逻辑
# 修复版本：修复了 user_indices 对齐、normalize、hard_neg、weight 等 bug
import os
import time
import random
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy import sparse
from collections import defaultdict
from trainers.trainer import Trainer
from utils.utils import metric_report, metric_len_report, record_csv, metric_pop_report


class RCLTrainer(Trainer):
    """
    RCL Trainer: 继承自 Trainer，添加 RCL 的 SSL 损失计算逻辑
    
    修复的 bug：
    1. user_indices 对齐 - 现在必须从 Dataset 获取全局 user_id
    2. normalize 用法 - 只对 query/key normalize，不对 logits normalize
    3. hard negatives - 真正参与 InfoNCE loss
    4. weight 生效 - 用 pos_sim_weights 加权 loss
    5. same_y 改用倒排索引 - 不再保存 U×U 矩阵
    6. cache 轻量化 - 只保存 O(U) 的数据结构
    """
    
    def __init__(self, args, logger, writer, device, generator):
        super().__init__(args, logger, writer, device, generator)
        
        # RCL 特有参数
        self.ssl_type = args.rcl_ssl
        self.rcl_scale = args.rcl_scale
        self.rcl_neg_size = args.rcl_neg_size
        self.rcl_perc = args.rcl_perc
        self.rcl_neg_perc1 = args.rcl_neg_perc1
        self.rcl_neg_perc2 = args.rcl_neg_perc2
        self.rcl_smooth_loss = args.rcl_smooth_loss
        self.rcl_max = args.rcl_max
        self.rcl_temperature = getattr(args, 'rcl_temperature', 0.1)  # 温度参数
        
        # 用户数量
        self.usernum = self.user_num
        
        # 预计算用户相似性矩阵
        self._precompute_user_similarity()
        
        # Debug: 训练前验证
        self._debug_print_cache_info()
    
    def _precompute_user_similarity(self):
        """预计算用户序列相似性矩阵，用于 RCL 的 SSL"""
        self.logger.info("Precomputing user similarity matrices for RCL...")
        
        # 获取所有用户的训练序列
        user_seqs = self.generator.get_all_user_seqs()
        
        usernum = len(user_seqs)
        self.usernum = usernum
        max_len = self.args.max_len
        
        # 构建用户序列矩阵
        self.mat_user_train = np.zeros([usernum, max_len], dtype=np.int32)
        for u_idx, seq in enumerate(user_seqs):
            seq_list = seq.cpu().numpy() if isinstance(seq, torch.Tensor) else seq
            seq_len = min(len(seq_list), max_len)
            self.mat_user_train[u_idx, max_len - seq_len:] = seq_list[-seq_len:]
        
        # 计算相似性矩阵（仅在 ssl >= 6 时需要）
        if self.ssl_type >= 6:
            # 使用 v2 轻量缓存格式
            cache_path = f'rcl_cache/{self.args.dataset}_perc{self.rcl_perc}_ssl{self.ssl_type}_v2.npz'
            os.makedirs('rcl_cache', exist_ok=True)
            
            if os.path.exists(cache_path):
                self.logger.info(f"Loading cached similarity matrices from {cache_path}")
                self._load_lightweight_cache(cache_path)
            else:
                self.logger.info("Computing similarity matrices (this may take a while)...")
                self._compute_similarity_matrices_lightweight(usernum)
                self._save_lightweight_cache(cache_path)
                self.logger.info(f"Saved lightweight cache to {cache_path}")
            
            # 构建 last_item 到 users 的倒排索引（用于 same-y 采样）
            self._build_last_item_index()
        
        self.logger.info("User similarity matrices ready.")
    
    def _load_lightweight_cache(self, cache_path):
        """加载轻量缓存（只包含 O(U) 数据）"""
        cache = np.load(cache_path, allow_pickle=True)
        
        # 必要字段
        self.global_pos = cache['global_pos']
        self.global_hard_neg = cache['global_hard_neg']
        self.pos_sim_weights = cache['pos_sim_weights']
        
        # 可选字段
        if 'last_item' in cache.files:
            self.last_item = cache['last_item']
        else:
            # 从 mat_user_train 重新计算
            self.last_item = self.mat_user_train[:, -1].astype(np.int32)
        
        # 验证数据一致性
        assert len(self.global_pos) == self.usernum, \
            f"global_pos length {len(self.global_pos)} != usernum {self.usernum}"
        assert len(self.global_hard_neg) == self.usernum, \
            f"global_hard_neg length {len(self.global_hard_neg)} != usernum {self.usernum}"
        assert len(self.pos_sim_weights) == self.usernum, \
            f"pos_sim_weights length {len(self.pos_sim_weights)} != usernum {self.usernum}"
        
        self.logger.info(f"Loaded lightweight cache: global_pos shape={self.global_pos.shape}, "
                        f"global_hard_neg shape={self.global_hard_neg.shape}, "
                        f"pos_sim_weights shape={self.pos_sim_weights.shape}")
    
    def _save_lightweight_cache(self, cache_path):
        """保存轻量缓存"""
        np.savez_compressed(
            cache_path,
            global_pos=self.global_pos,
            global_hard_neg=self.global_hard_neg,
            pos_sim_weights=self.pos_sim_weights,
            last_item=self.last_item,
            # 元信息
            usernum=np.array([self.usernum]),
            rcl_perc=np.array([self.rcl_perc]),
            rcl_neg_perc1=np.array([self.rcl_neg_perc1]),
            rcl_neg_perc2=np.array([self.rcl_neg_perc2]),
            rcl_neg_size=np.array([self.rcl_neg_size])
        )
        
        # 验证文件大小
        file_size = os.path.getsize(cache_path)
        self.logger.info(f"Saved cache file size: {file_size / 1024 / 1024:.2f} MB")
    
    def _compute_similarity_matrices_lightweight(self, usernum):
        """
        计算用户相似性矩阵（轻量版本）
        只保存 O(U) 数据，不保存 U×U 矩阵
        """
        
        # 第一步：使用稀疏矩阵计算用户两两相似性
        self.logger.info(f"Step 1/3: Computing pairwise similarity for {usernum:,} users (sparse matrix method)")
        
        # 1.1 构建用户-物品稀疏矩阵
        self.logger.info("  Building user-item sparse matrix...")
        itemnum = int(self.mat_user_train.max()) + 1
        
        # 构建 COO 格式的稀疏矩阵数据
        rows = []
        cols = []
        for u in tqdm(range(usernum), desc="Building sparse matrix", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            items = self.mat_user_train[u]
            items = items[items > 0]  # 去除 padding (0)
            unique_items = np.unique(items)
            rows.extend([u] * len(unique_items))
            cols.extend(unique_items.tolist())
        
        # 创建稀疏矩阵 (usernum x itemnum)，值为 1
        data = np.ones(len(rows), dtype=np.float32)
        user_item_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(usernum, itemnum))
        
        # 1.2 计算 A @ A.T = 用户两两共同物品数（保持稀疏格式）
        self.logger.info("  Computing user-user similarity (A @ A.T) as sparse matrix...")
        similarity_sparse = user_item_matrix @ user_item_matrix.T
        similarity_sparse.setdiag(0)  # 对角线置0
        
        # 1.3 计算 last_item
        self.logger.info("  Computing last_item array...")
        self.last_item = self.mat_user_train[:, -1].astype(np.int32)
        
        # 第二步：使用稀疏矩阵计算 percentile 阈值，然后采样 global_pos
        self.logger.info("Step 2/3: Sampling positive users using sparse matrix...")
        
        # 从稀疏矩阵的数据中计算 percentile
        all_sim_values = similarity_sparse.data
        global_sim_perc = np.percentile(all_sim_values, self.rcl_perc)
        self.logger.info(f"  Positive threshold (percentile {self.rcl_perc}): {global_sim_perc}")
        
        self.global_pos = np.zeros(usernum, dtype=np.int32)
        self.pos_sim_weights = np.zeros(usernum, dtype=np.float32)
        
        for i in tqdm(range(usernum), desc="Positive sampling", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            row = similarity_sparse.getrow(i).toarray().flatten()
            candidates = np.where(row > global_sim_perc)[0]
            
            if len(candidates) > 0:
                probs = row[candidates].astype(float)
                probs = probs / probs.sum()
                chosen = np.random.choice(candidates, 1, p=probs)[0]
                self.global_pos[i] = chosen
                self.pos_sim_weights[i] = row[chosen]
            else:
                # fallback: 选择自己
                self.global_pos[i] = i
                self.pos_sim_weights[i] = 0.0
        
        # 归一化 pos_sim_weights
        max_weight = self.pos_sim_weights.max()
        if max_weight > 0:
            self.pos_sim_weights = self.pos_sim_weights / max_weight
        
        # 第三步：计算 global_hard_neg (难负样本)
        self.logger.info("Step 3/3: Sampling hard negative users...")
        hard_neg_perc1 = np.percentile(all_sim_values, self.rcl_neg_perc1)
        hard_neg_perc2 = np.percentile(all_sim_values, self.rcl_neg_perc2)
        self.logger.info(f"  Hard neg range: [{hard_neg_perc2}, {hard_neg_perc1}]")
        
        self.global_hard_neg = np.zeros((usernum, self.rcl_neg_size), dtype=np.int32)
        
        for i in tqdm(range(usernum), desc="Hard neg sampling", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            row = similarity_sparse.getrow(i).toarray().flatten()
            candidates = np.where((row > hard_neg_perc2) & (row < hard_neg_perc1))[0]
            
            if len(candidates) > 0:
                probs = row[candidates].astype(float)
                probs = probs / probs.sum()
                # 采样 rcl_neg_size 个
                if len(candidates) >= self.rcl_neg_size:
                    chosen = np.random.choice(candidates, self.rcl_neg_size, p=probs, replace=False)
                else:
                    chosen = np.random.choice(candidates, self.rcl_neg_size, p=probs, replace=True)
                self.global_hard_neg[i] = chosen
            else:
                # fallback: 随机用户
                self.global_hard_neg[i] = np.random.randint(0, usernum, self.rcl_neg_size)
        
        self.logger.info("Lightweight similarity computation done!")
    
    def _build_last_item_index(self):
        """构建 last_item 到 users 的倒排索引"""
        self.logger.info("Building last_item to users inverted index...")
        
        # 如果 last_item 不存在，从 mat_user_train 计算
        if not hasattr(self, 'last_item'):
            self.last_item = self.mat_user_train[:, -1].astype(np.int32)
        
        # 构建倒排索引: item -> list of users
        self.last_item_to_users = defaultdict(list)
        for u, item in enumerate(self.last_item):
            if item > 0:  # 排除 padding
                self.last_item_to_users[item].append(u)
        
        self.logger.info(f"Built inverted index: {len(self.last_item_to_users)} unique last items")
    
    def _debug_print_cache_info(self):
        """Debug: 打印 cache 信息，验证没有 U×U 数组"""
        self.logger.info("=" * 60)
        self.logger.info("RCL Cache Debug Info")
        self.logger.info("=" * 60)
        
        if self.ssl_type >= 6:
            self.logger.info(f"usernum: {self.usernum}")
            self.logger.info(f"global_pos: shape={self.global_pos.shape}, dtype={self.global_pos.dtype}")
            self.logger.info(f"global_hard_neg: shape={self.global_hard_neg.shape}, dtype={self.global_hard_neg.dtype}")
            self.logger.info(f"pos_sim_weights: shape={self.pos_sim_weights.shape}, dtype={self.pos_sim_weights.dtype}")
            
            # 检查是否有 U×U 数组（不应该有）
            has_uxu = False
            for name in ['seq_intersec', 'same_y', 'global_mask']:
                if hasattr(self, name):
                    arr = getattr(self, name)
                    if hasattr(arr, 'shape') and len(arr.shape) == 2:
                        if arr.shape[0] == self.usernum and arr.shape[1] == self.usernum:
                            self.logger.warning(f"WARNING: Found U×U array: {name} with shape {arr.shape}")
                            has_uxu = True
            
            if not has_uxu:
                self.logger.info("✓ No U×U arrays found (good!)")
            
            # 打印样本
            self.logger.info(f"global_pos[:10]: {self.global_pos[:10]}")
            self.logger.info(f"global_hard_neg[:2]: {self.global_hard_neg[:2]}")
            self.logger.info(f"pos_sim_weights[:10]: {self.pos_sim_weights[:10]}")
            
            # 断言检查
            assert len(self.global_pos) == self.usernum, "global_pos length mismatch"
            assert len(self.global_hard_neg) == self.usernum, "global_hard_neg length mismatch"
            assert len(self.pos_sim_weights) == self.usernum, "pos_sim_weights length mismatch"
        
        self.logger.info("=" * 60)
    
    def _compute_ssl_loss(self, model, seq, pos, neg, positions, user_indices, log_feats):
        """
        计算 RCL 的 SSL 损失
        
        修复:
        1. query/key 先 normalize，logits 不 normalize
        2. hard negatives 真正参与 InfoNCE loss
        3. weight 真正生效（加权损失）
        """
        ssl_loss = torch.tensor(0.0, device=self.device)
        
        if self.ssl_type == 0:
            return ssl_loss
        
        T = self.rcl_temperature  # 温度参数
        
        # SSL type 1-4: 基础对比学习（自监督增强）
        if self.ssl_type in [1, 2, 3, 4]:
            _, _, aug_embed1 = model(seq, pos, neg, positions)
            _, _, aug_embed2 = model(seq, pos, neg, positions)
            
            # 修复: 先 normalize query/key，再计算 logits
            query = F.normalize(aug_embed1[:, -1, :], dim=-1)
            positive_key = F.normalize(aug_embed2[:, -1, :], dim=-1)
            
            # logits 不再 normalize
            logits = query @ positive_key.T
            labels = torch.arange(len(query), device=query.device)
            
            ssl_loss = F.cross_entropy(logits / T, labels, reduction='mean')
        
        # SSL type 5-11: RCL 核心逻辑
        elif self.ssl_type >= 5:
            _, _, aug_embed1 = model(seq, pos, neg, positions)
            # 修复: 先 normalize query
            query = F.normalize(aug_embed1[:, -1, :], dim=-1)
            
            # 获取用户索引
            u_indices = user_indices.cpu().numpy() if isinstance(user_indices, torch.Tensor) else user_indices
            batch_size = len(u_indices)
            
            # ========== 相似标签用户的对比学习 (same-y) ==========
            if self.rcl_max == 1:
                # 使用倒排索引采样有相同 last_item 的用户
                y_users = self._sample_same_y_users(u_indices)
                y_seq = torch.LongTensor(self.mat_user_train[y_users]).to(self.device)
                y_positions = self._generate_positions(y_seq)
                _, _, y_embed = model(y_seq, pos, neg, y_positions)
                
                # 修复: normalize key
                y_positive_key = F.normalize(y_embed[:, -1, :], dim=-1)
                y_logits = query @ y_positive_key.T
                labels = torch.arange(len(query), device=query.device)
                
                y_loss = F.cross_entropy(y_logits / T, labels, reduction='mean')
                ssl_loss += y_loss * self.rcl_scale
            
            # ========== 相似序列用户的对比学习 (InfoNCE with hard negatives) ==========
            # 获取正样本
            aug_users = self.global_pos[u_indices]
            sim_aug_seq = torch.LongTensor(self.mat_user_train[aug_users]).to(self.device)
            sim_positions = self._generate_positions(sim_aug_seq)
            _, _, sim_aug_embed = model(sim_aug_seq, pos, neg, sim_positions)
            
            # 修复: normalize positive key
            pos_key = F.normalize(sim_aug_embed[:, -1, :], dim=-1)  # (B, D)
            
            # 修复: 获取 hard negatives 并通过完整 forward 获得表征
            neg_users = self.global_hard_neg[u_indices]  # (B, K)
            neg_users_flat = neg_users.reshape(-1)  # (B*K,)
            neg_seq = torch.LongTensor(self.mat_user_train[neg_users_flat]).to(self.device)
            neg_positions = self._generate_positions(neg_seq)
            
            # 准备 pos/neg 参数，确保与 neg_seq batch size 一致
            # 注意：GRU4Rec/Bert4Rec 的 pos/neg 是单值 [B]，SASRec 的是序列 [B, T]
            neg_seq_len = len(neg_seq)
            if pos.dim() > 1:
                # SASRec: seq2seq 格式 [B, T]
                pos_for_neg = pos[:1].expand(neg_seq_len, -1)
                neg_for_neg = neg[:1].expand(neg_seq_len, -1)
            else:
                # GRU4Rec/Bert4Rec: 单值格式 [B]
                pos_for_neg = pos[:1].expand(neg_seq_len)
                neg_for_neg = neg[:1].expand(neg_seq_len) if neg.dim() == 1 else neg[:1].expand(neg_seq_len, -1)
            
            # 通过完整 forward 获得 neg_key（与 query 同空间）
            _, _, neg_embed = model(neg_seq, pos_for_neg, neg_for_neg, neg_positions)
            neg_key = F.normalize(neg_embed[:, -1, :], dim=-1)  # (B*K, D)
            neg_key = neg_key.reshape(batch_size, self.rcl_neg_size, -1)  # (B, K, D)
            
            # 修复: 构造 InfoNCE logits
            # pos_logit: (B, 1)
            pos_logit = (query * pos_key).sum(dim=-1, keepdim=True)
            
            # neg_logit: (B, K) 使用 einsum
            neg_logit = torch.einsum('bd,bkd->bk', query, neg_key)
            
            # 合并 logits: (B, 1+K)，正样本在第 0 列
            logits = torch.cat([pos_logit, neg_logit], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            
            # 修复: 加权损失
            # 计算逐样本 loss
            per_sample_loss = F.cross_entropy(logits / T, labels, reduction='none')  # (B,)
            
            # 获取 weight
            weight = torch.tensor(self.pos_sim_weights[u_indices], 
                                  device=self.device, dtype=torch.float)  # (B,)
            
            # 幂次平滑
            weight = weight ** self.rcl_smooth_loss
            
            # 归一化 weight（避免 loss scale 变化太大）
            weight = weight / (weight.mean() + 1e-8)
            
            # 加权平均
            sim_loss = (per_sample_loss * weight).mean()
            ssl_loss += sim_loss * self.rcl_scale
        
        return ssl_loss
    
    def _sample_same_y_users(self, user_indices):
        """
        为当前 batch 采样有相同最后 item 的用户
        使用倒排索引，不再使用 U×U 的 same_y 矩阵
        """
        global_y = []
        for u in user_indices:
            last_item = self.last_item[u]
            candidates = self.last_item_to_users.get(last_item, [])
            
            # 过滤掉自己
            candidates = [v for v in candidates if v != u]
            
            if len(candidates) > 0:
                # 随机选一个
                global_y.append(random.choice(candidates))
            else:
                # fallback: 选择自己
                global_y.append(u)
        
        return np.array(global_y, dtype=np.int32)
    
    def _generate_positions(self, seqs):
        """生成位置向量"""
        batch_size = seqs.shape[0]
        seq_len = seqs.shape[1]
        positions = np.tile(np.array(range(seq_len)), [batch_size, 1])
        return torch.LongTensor(positions).to(self.device)
    
    def _train_one_epoch(self, epoch):
        """RCL 训练一个 epoch"""
        tr_loss = 0
        ssl_loss_total = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')
        
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        # Debug: 首个 epoch 的首个 batch 打印 user_indices 信息
        first_batch_printed = False
        
        for batch_idx, batch in enumerate(prog_iter):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_train_inputs(batch)
            
            seq = inputs['seq']
            pos = inputs['pos']
            neg = inputs['neg']
            positions = inputs['positions']
            
            # 修复: 获取用户索引 - 必须是全局 user_id
            if 'user_id' in inputs:
                user_indices = inputs['user_id']
                if isinstance(user_indices, torch.Tensor):
                    user_indices = user_indices.cpu().numpy()
            else:
                # 严格模式：RCL 必须有 user_id
                raise RuntimeError(
                    "RCL requires global user_id in batch inputs! "
                    "Please ensure your Dataset returns user_id and var_name includes 'user_id'."
                )
            
            # 修复: 验证 user_indices 范围
            assert user_indices.min() >= 0, f"user_indices.min() = {user_indices.min()} < 0"
            assert user_indices.max() < self.usernum, \
                f"user_indices.max() = {user_indices.max()} >= usernum = {self.usernum}"
            
            # Debug: 首个 batch 打印信息
            if epoch == 0 and not first_batch_printed:
                self.logger.info(f"[Debug] First batch user_indices: min={user_indices.min()}, "
                               f"max={user_indices.max()}, shape={user_indices.shape}")
                self.logger.info(f"[Debug] usernum={self.usernum}, global_pos shape={self.global_pos.shape}")
                first_batch_printed = True
            
            # RCL 模型 forward 返回 (pos_logits, neg_logits, log_feats)
            pos_logits, neg_logits, log_feats = self.model(seq, pos, neg, positions)
            
            # 基础 BCE 损失
            # 注意: SASRec 返回 pos_logits [B, T]，只用最后一个位置
            #       GRU4Rec/Bert4Rec 返回 pos_logits [B]，直接使用
            if pos_logits.dim() == 2:
                # SASRec: seq2seq 格式，取最后一个位置
                pos_logits_last = pos_logits[:, -1]
                neg_logits_last = neg_logits[:, -1]
            else:
                # GRU4Rec/Bert4Rec: 单值格式，直接使用
                pos_logits_last = pos_logits
                neg_logits_last = neg_logits
            
            pos_labels = torch.ones(pos_logits_last.shape, device=self.device)
            neg_labels = torch.zeros(neg_logits_last.shape, device=self.device)
            
            loss = bce_criterion(pos_logits_last, pos_labels)
            loss += bce_criterion(neg_logits_last, neg_labels)
            
            # RCL SSL 损失
            ssl_loss = self._compute_ssl_loss(self.model, seq, pos, neg, positions, user_indices, log_feats)
            loss += ssl_loss
            
            loss.backward()
            
            tr_loss += loss.item()
            ssl_loss_total += ssl_loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1
            
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps),
                                  ssl='%.4f' % (ssl_loss_total / nb_tr_steps))
            
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)
        self.writer.add_scalar('train/ssl_loss', ssl_loss_total / nb_tr_steps, epoch)
        
        # 如果启用了 HSU 融合，记录 gate 值
        if hasattr(self.model, 'use_hsu_fusion') and self.model.use_hsu_fusion:
            gate_val = self.model.get_hsu_gate_value()
            if gate_val is not None:
                self.writer.add_scalar('train/hsu_gate', gate_val, epoch)
                if epoch % 10 == 0:  # 每 10 个 epoch 打印一次
                    self.logger.info(f"[HSU Fusion] Gate value at epoch {epoch}: {gate_val:.6f}")
    
    def eval(self, epoch=0, test=False):
        """评估 (与 SeqTrainer 相同)"""
        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        
        for batch in tqdm(test_loader, desc=desc):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"] > 0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():
                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)
                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])
        
        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), 
                                         seq_len.detach().cpu().numpy(), 
                                         aug_len=self.args.aug_seq_len, 
                                         args=self.args)
        res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), 
                                         self.item_pop, 
                                         target_items.detach().cpu().numpy(), 
                                         args=self.args)
        
        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        
        if test:
            self.logger.info("User Group Performance:")
            for k, v in res_len_dict.items():
                self.logger.info('\t %s: %.5f' % (k, v))
            self.logger.info("Item Group Performance:")
            for k, v in res_pop_dict.items():
                self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict, **res_pop_dict}
        
        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
