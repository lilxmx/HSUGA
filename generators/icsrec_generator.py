# -*- coding: utf-8 -*-
"""
ICSRec Data Generator
Inherits from HSUGA's Generator, provides dual-view data augmentation required by ICSRec
"""

import os
import copy
import random
import numpy as np
import pickle
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from generators.generator import Generator, Seq2SeqGeneratorAllUser, GeneratorAllUser
from generators.data import SeqDataset
from utils.utils import unzip_data, random_neq


class ICSRecSeq2SeqDataset(Dataset):
    """
    ICSRec-specific Seq2Seq Dataset (for SASRec backbone)
    
    Core features:
    1. Returns two augmented views (seq1, seq2) for CICL contrastive learning
    2. Returns target item ID for False Negative Mining
    3. Supports Dynamic Segmentation (DS) data augmentation
    
    Returns: (seq, pos, neg, positions, user_id, seq_aug, positions_aug, target_item)
    """
    
    def __init__(self, args, data, item_num, max_len, neg_num=1, 
                 target_dict=None, user_id_list=None):
        """
        Args:
            args: Configuration parameters
            data: User sequence data [list of lists]
            item_num: Number of items
            max_len: Maximum sequence length
            neg_num: Number of negative samples
            target_dict: Mapping from target item to sequences {target_item: [sequences]}
            user_id_list: User ID list (for matching original users)
        """
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.args = args
        self.target_dict = target_dict if target_dict else {}
        self.user_id_list = user_id_list if user_id_list else list(range(len(data)))
        
        # Build mapping from target item to sequence indices
        self._build_target_mapping()
        
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", 
                        "seq_aug", "positions_aug", "target_item"]
    
    def _build_target_mapping(self):
        """Build mapping from target item to sequence indices"""
        self.target_to_indices = defaultdict(list)
        for idx, seq in enumerate(self.data):
            if len(seq) >= 2:
                target = seq[-1]  # Target item is the last element of the sequence
                self.target_to_indices[target].append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def _get_augmented_sequence(self, index, target_item):
        """
        Get augmented sequence (different subsequence with same target item)
        
        Implements Dynamic Segmentation (DS) idea: find other subsequences with same target item as positive pairs
        """
        # Get other sequence indices with same target item
        candidate_indices = self.target_to_indices.get(target_item, [])
        
        # Filter out self
        candidate_indices = [i for i in candidate_indices if i != index]
        
        if len(candidate_indices) > 0:
            # Randomly select a sequence with same target
            aug_index = random.choice(candidate_indices)
            aug_seq = self.data[aug_index]
        else:
            # If no other sequences with same target, use self
            aug_seq = self.data[index]
        
        # Construct augmented sequence
        seq_aug = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(aug_seq[:-1]):  # Exclude target item
            seq_aug[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        # Construct position encoding
        if len(aug_seq) > self.max_len:
            mask_len = 0
            positions_aug = list(range(1, self.max_len + 1))
        else:
            mask_len = self.max_len - (len(aug_seq) - 1)
            positions_aug = list(range(1, len(aug_seq) - 1 + 1))
        
        positions_aug = positions_aug[-self.max_len:]
        positions_aug = [0] * mask_len + positions_aug
        positions_aug = np.array(positions_aug)
        
        return seq_aug, positions_aug
    
    def __getitem__(self, index):
        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        target_item = inter[-1]  # Target item
        
        # Main sequence (view 1)
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num + 1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        # Position encoding
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len + 1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter) - 1 + 1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)
        
        # Augmented sequence (view 2) - for contrastive learning
        seq_aug, positions_aug = self._get_augmented_sequence(index, target_item)
        
        user_id = index
        
        return (seq, pos, neg, positions, user_id, 
                seq_aug, positions_aug, target_item)


class ICSRecSeqDataset(Dataset):
    """
    ICSRec-specific Seq Dataset (for GRU4Rec backbone)
    
    Returns: (seq, pos, neg, positions, user_id, seq_aug, positions_aug, target_item)
    """
    
    def __init__(self, args, data, item_num, max_len, neg_num=1,
                 target_dict=None, user_id_list=None):
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.args = args
        
        self._build_target_mapping()
        self.var_name = ["seq", "pos", "neg", "positions", "user_id",
                        "seq_aug", "positions_aug", "target_item"]
    
    def _build_target_mapping(self):
        """Build mapping from target item to sequence indices"""
        self.target_to_indices = defaultdict(list)
        for idx, seq in enumerate(self.data):
            if len(seq) >= 2:
                target = seq[-1]
                self.target_to_indices[target].append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def _get_augmented_sequence(self, index, target_item):
        """Get augmented sequence"""
        candidate_indices = self.target_to_indices.get(target_item, [])
        candidate_indices = [i for i in candidate_indices if i != index]
        
        if len(candidate_indices) > 0:
            aug_index = random.choice(candidate_indices)
            aug_seq = self.data[aug_index]
        else:
            aug_seq = self.data[index]
        
        seq_aug = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(aug_seq[:-1]):
            seq_aug[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(aug_seq) > self.max_len:
            mask_len = 0
            positions_aug = list(range(1, self.max_len + 1))
        else:
            mask_len = self.max_len - (len(aug_seq) - 1)
            positions_aug = list(range(1, len(aug_seq) - 1 + 1))
        
        positions_aug = positions_aug[-self.max_len:]
        positions_aug = [0] * mask_len + positions_aug
        positions_aug = np.array(positions_aug)
        
        return seq_aug, positions_aug
    
    def __getitem__(self, index):
        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        target_item = inter[-1]
        pos = inter[-1]
        
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num + 1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len + 1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter) - 1 + 1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)
        
        seq_aug, positions_aug = self._get_augmented_sequence(index, target_item)
        
        user_id = index
        
        return (seq, pos, neg, positions, user_id,
                seq_aug, positions_aug, target_item)


class ICSRecBertDataset(Dataset):
    """
    ICSRec-specific BERT4Rec Dataset
    
    Returns: (seq, pos, neg, positions, user_id, seq_aug, positions_aug, target_item)
    """
    
    def __init__(self, args, data, item_num, max_len, neg_num=1):
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_prob = args.mask_prob
        self.mask_token = item_num + 1
        
        self._build_target_mapping()
        self.var_name = ["seq", "pos", "neg", "positions", "user_id",
                        "seq_aug", "positions_aug", "target_item"]
    
    def _build_target_mapping(self):
        """Build mapping from target item to sequence indices"""
        self.target_to_indices = defaultdict(list)
        for idx, seq in enumerate(self.data):
            if len(seq) >= 2:
                target = seq[-1]
                self.target_to_indices[target].append(idx)
    
    def __len__(self):
        return 2 * len(self.data)
    
    def _get_augmented_sequence(self, index, target_item):
        """Get augmented sequence"""
        candidate_indices = self.target_to_indices.get(target_item, [])
        candidate_indices = [i for i in candidate_indices if i != index]
        
        if len(candidate_indices) > 0:
            aug_index = random.choice(candidate_indices)
            aug_seq = self.data[aug_index]
        else:
            aug_seq = self.data[index]
        
        seq_aug = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(aug_seq[:-1]):
            seq_aug[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(aug_seq) > self.max_len:
            mask_len = 0
            positions_aug = list(range(1, self.max_len + 1))
        else:
            mask_len = self.max_len - (len(aug_seq) - 1)
            positions_aug = list(range(1, len(aug_seq) - 1 + 1))
        
        positions_aug = positions_aug[-self.max_len:]
        positions_aug = [0] * mask_len + positions_aug
        positions_aug = np.array(positions_aug)
        
        return seq_aug, positions_aug
    
    def __getitem__(self, index):
        tokens = []
        labels, neg_labels = [], []
        
        if index >= len(self.data):
            real_index = index - len(self.data)
            seq = self.data[real_index]
            target_item = seq[-1]
            
            for s in seq:
                tokens.append(s)
                labels.append(0)
                neg_labels.append(0)
            labels[-1] = tokens[-1]
            neg_labels[-1] = random_neq(1, self.item_num + 1, seq)
            tokens[-1] = self.mask_token
            user_id = real_index
        else:
            seq = self.data[index]
            target_item = seq[-1]
            
            for s in seq:
                prob = random.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob
                    
                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(random.randint(1, self.item_num))
                    else:
                        tokens.append(s)
                    
                    labels.append(s)
                    neg = random_neq(1, self.item_num + 1, seq)
                    neg_labels.append(neg)
                else:
                    tokens.append(s)
                    labels.append(0)
                    neg_labels.append(0)
            user_id = index
        
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_labels = neg_labels[-self.max_len:]
        pos = list(range(1, len(tokens) + 1))
        pos = pos[-self.max_len:]
        
        mask_len = self.max_len - len(tokens)
        
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        neg_labels = [0] * mask_len + neg_labels
        pos = [0] * mask_len + pos
        
        # Get augmented sequence
        if index >= len(self.data):
            real_idx = index - len(self.data)
        else:
            real_idx = index
        seq_aug, positions_aug = self._get_augmented_sequence(real_idx, target_item)
        
        return (np.array(tokens), np.array(labels), np.array(neg_labels), 
                np.array(pos), user_id, seq_aug, positions_aug, target_item)


# =============================================================================
# ICSRec Generator Classes
# =============================================================================

class ICSRecSeq2SeqGenerator(Generator):
    """
    ICSRec Seq2Seq Generator (for SASRec backbone)
    Inherits from HSUGA's Generator
    """
    
    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)
    
    def make_trainloader(self):
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = ICSRecSeq2SeqDataset(
            self.args, train_dataset, self.item_num, 
            self.args.max_len, self.args.train_neg
        )
        
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.bs,
            num_workers=self.num_workers
        )
        
        return train_dataloader
    
    def make_clusterloader(self):
        """Create DataLoader for clustering training (sequential sampling)"""
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        cluster_dataset = ICSRecSeq2SeqDataset(
            self.args, train_dataset, self.item_num,
            self.args.max_len, self.args.train_neg
        )
        
        cluster_dataloader = DataLoader(
            cluster_dataset,
            sampler=SequentialSampler(cluster_dataset),
            batch_size=self.bs,
            num_workers=self.num_workers
        )
        
        return cluster_dataloader


class ICSRecSeqGenerator(Generator):
    """
    ICSRec Seq Generator (for GRU4Rec backbone)
    """
    
    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)
    
    def make_trainloader(self):
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = ICSRecSeqDataset(
            self.args, train_dataset, self.item_num,
            self.args.max_len, self.args.train_neg
        )
        
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.bs,
            num_workers=self.num_workers
        )
        
        return train_dataloader
    
    def make_clusterloader(self):
        """创建用于聚类训练的 DataLoader"""
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        cluster_dataset = ICSRecSeqDataset(
            self.args, train_dataset, self.item_num,
            self.args.max_len, self.args.train_neg
        )
        
        cluster_dataloader = DataLoader(
            cluster_dataset,
            sampler=SequentialSampler(cluster_dataset),
            batch_size=self.bs,
            num_workers=self.num_workers
        )
        
        return cluster_dataloader


class ICSRecBertGenerator(Generator):
    """
    ICSRec BERT Generator (for BERT4Rec backbone)
    """
    
    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)
    
    def make_trainloader(self):
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = ICSRecBertDataset(
            self.args, train_dataset, self.item_num,
            self.args.max_len, self.args.train_neg
        )
        
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.bs,
            num_workers=self.num_workers
        )
        
        return train_dataloader
    
    def make_clusterloader(self):
        """创建用于聚类训练的 DataLoader"""
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        cluster_dataset = ICSRecBertDataset(
            self.args, train_dataset, self.item_num,
            self.args.max_len, self.args.train_neg
        )
        
        cluster_dataloader = DataLoader(
            cluster_dataset,
            sampler=SequentialSampler(cluster_dataset),
            batch_size=self.bs,
            num_workers=self.num_workers
        )
        
        return cluster_dataloader

