# here put the import lib
import os
import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import random_neq
import pickle
import pandas as pd


def get_sim_user_file_path(args):
    """
    获取相似用户文件路径
    
    Args:
        args: 包含 sim_user_file_path, dataset, hidden_mode 等参数
    
    Returns:
        相似用户文件路径
    """
    if hasattr(args, 'sim_user_file_path') and args.sim_user_file_path is not None:
        return args.sim_user_file_path
    else:
        return os.path.join("./data", args.dataset, "handled", f"sim_user_100{args.hidden_mode}.pkl")


def get_sim_user_score_file_path(args):
    """
    获取相似用户分数文件路径
    
    Args:
        args: 包含 sim_user_score_file_path, dataset, filter_similar_metric, hidden_mode 等参数
    
    Returns:
        相似用户分数文件路径，如果不需要则返回 None
    """
    if hasattr(args, 'sim_user_score_file_path') and args.sim_user_score_file_path is not None:
        return args.sim_user_score_file_path
    elif hasattr(args, 'filter_similar_metric'):
        return os.path.join("./data", args.dataset, "handled", f"sim_user_{args.filter_similar_metric}_score_100{args.hidden_mode}.pkl")
    else:
        return None


def load_dynamic_k_from_analysis(args):
    """
    Load user signals from analysis file, compute dynamic K online based on w1, w2
    
    Args:
        args: Contains the following parameters
            - dynamic_k_analysis_path: Analysis file path (CSV)
            - dynamic_k_w1: Length signal weight
            - dynamic_k_w2: Divergence signal weight  
            - dynamic_k_min: Minimum K value
            - dynamic_k_max: Maximum K value
            - dataset: Dataset name
    
    Returns:
        dict: user_id -> K mapping
    """
    # Get parameters
    w1 = getattr(args, 'dynamic_k_w1', 0.5)
    w2 = getattr(args, 'dynamic_k_w2', 0.5)
    k_min = getattr(args, 'dynamic_k_min', 2)
    k_max = getattr(args, 'dynamic_k_max', 18)
    
    # Determine analysis file path (placed under each dataset)
    analysis_path = getattr(args, 'dynamic_k_analysis_path', None)
    if analysis_path is None:
        dataset = getattr(args, 'dataset', 'steam')
        analysis_path = f"./data/{dataset}/handled/user_dynamic_k_analysis.csv"
    
    if not os.path.exists(analysis_path):
        print(f"[DynamicK] Warning: {analysis_path} not found")
        return None
    
    # Load analysis file
    df = pd.read_csv(analysis_path)
    
    # Check required columns
    required_cols = ['user_id', 's_len', 's_div']
    if not all(col in df.columns for col in required_cols):
        print(f"[DynamicK] Warning: Missing required columns in {analysis_path}")
        return None
    
    # Use quantile normalization (consistent with offline analysis)
    df['s_len_norm'] = df['s_len'].rank(pct=True)
    df['s_div_norm'] = df['s_div'].rank(pct=True, na_option='keep')
    
    # Calculate need value: w1 * s_len_norm + w2 * s_div_norm
    # For missing s_div_norm, fill with s_len_norm
    df['need'] = np.clip(
        w1 * df['s_len_norm'] + w2 * df['s_div_norm'].fillna(df['s_len_norm']),
        0, 1
    )
    
    # Calculate K value
    df['K'] = k_min + np.round((k_max - k_min) * df['need']).astype(int)
    df['K'] = df['K'].clip(k_min, k_max)
    
    # Create user_id -> K dictionary
    user_dynamic_k = dict(zip(df['user_id'].astype(int), df['K'].astype(int)))
    
    # Print statistics
    k_values = list(user_dynamic_k.values())
    print(f"[DynamicK] Computed K with w1={w1}, w2={w2}, K_range=[{k_min}, {k_max}]")
    print(f"[DynamicK] Users: {len(user_dynamic_k)}, K: mean={np.mean(k_values):.2f}, "
          f"std={np.std(k_values):.2f}, range=[{min(k_values)}, {max(k_values)}]")
    
    return user_dynamic_k

class SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, data, item_num, max_len, neg_num=1):
        """data is a list. Each element is a user interaction item list
        item_num is the number of items
        neg_num=1 for training set
        neg_num=100 for validation and test"""
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):
        # Here is each user's interaction item list. Note: if it's training set, the list no longer contains the last 2 items from original dataset. Unless the user's interaction records are originally less than 3
        inter = self.data[index]  # User interaction list corresponding to index
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]  # Last interacted item as positive sample
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)  # Randomly sample items user hasn't interacted with from entire item set as negative samples
            neg.append(per_neg)
            non_neg.append(per_neg)  # Continuously update list to prevent sampling duplicate items
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        # Insert items into seq list from last item forward
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))  # [1,2,..,200]
        else:
            # If user's interaction records are less than 200, e.g., 50. mask_len = 200 - 50 + 1 = 151
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))  # [1,2,...,49] Last element is for test, positive sample
        
        positions= positions[-self.max_len:]  # Get all contents of positions
        positions = [0] * mask_len + positions  # Fill front positions with 0
        positions = np.array(positions)

        # Test set: neg negative samples length 100  Training set: neg length 1
        # pos positive sample length 1
        # positions length max_len
        # seq length max_len
        return seq, pos, neg, positions
    


class SeqDatasetAllUser(SeqDataset):
    '''The train dataset for Sequential recommendation
        GRU4Rec series
    '''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__(data, item_num, max_len, neg_num)
        self.sim_user_num = args.sim_user_num # 10
        self.sim_long_user_num = args.sim_long_user_num
        self.max_sim_user_num = max(self.sim_user_num, self.sim_long_user_num)
        sim_user_path = get_sim_user_file_path(args)
        self.sim_users = pickle.load(open(sim_user_path, "rb"))
        self.filter_similar_users = args.filter_similar_user
        # if args.weight_sum:
        self.similar_gate = args.similar_gate
        self.ts_user = args.ts_user
        # New: Relative percentile filtering parameter (per-user adaptive filtering)
        self.sim_filter_percentile = getattr(args, 'sim_filter_percentile', 0.0)
        # If percentile filtering or global threshold filtering is enabled, need to load similarity scores
        if args.filter_similar_user or self.sim_filter_percentile > 0:
            sim_score_path = get_sim_user_score_file_path(args)
            if sim_score_path and os.path.exists(sim_score_path):
                self.sim_users_scores = pickle.load(open(sim_score_path, "rb"))
            else:
                self.sim_users_scores = None
        else:
            self.sim_users_scores = None
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions", "valid_mask", "sim_user_ids"]  # "sim_user_scores"
        if args.filter_item:
            self.filted_item_dict = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", f"item_filter.pickle"), "rb"))
        else:
            self.filted_item_dict = {}
        
        # Dynamic K support: Online computation of recall count based on user interaction length + consistency
        self.use_dynamic_k = getattr(args, 'use_dynamic_k', False)
        self.user_dynamic_k = None
        if self.use_dynamic_k:
            # Use online computation method, dynamically compute K based on w1, w2 parameters
            self.user_dynamic_k = load_dynamic_k_from_analysis(args)
            if self.user_dynamic_k is not None:
                # Update max_sim_user_num to maximum value of dynamic K
                max_dynamic_k = max(self.user_dynamic_k.values())
                self.max_sim_user_num = max(self.max_sim_user_num, max_dynamic_k)
            else:
                print(f"[DynamicK] Failed to load, using fixed sim_user_num")
                self.use_dynamic_k = False


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        # TODO: Whether user's own interaction records need filtering
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        ### Get the sequence of similar users
        # Get recall count of similar users for current user
        if self.use_dynamic_k and self.user_dynamic_k is not None and index in self.user_dynamic_k:
            # Use dynamic K: recall count computed based on user interaction length + consistency
            current_sim_num = self.user_dynamic_k[index]
        elif len(self.data[index]) >= self.ts_user:
            # Long users use sim_long_user_num
            current_sim_num = self.sim_long_user_num
        else:
            # Short users use sim_user_num
            current_sim_num = self.sim_user_num
        sim_users = self.sim_users[index][:current_sim_num]
        sim_seq, sim_positions, filtered_sim_user_ids = self._filter_similar_users(index, sim_users)
        valid_mask = torch.zeros(self.max_sim_user_num, dtype=torch.int32) 
        valid_mask[:sim_seq.shape[0]] = 1

        remain_sim_seq = np.zeros((self.max_sim_user_num-sim_seq.shape[0], seq.shape[0]), dtype=np.int32)
        remain_sim_positions = np.zeros((self.max_sim_user_num-sim_positions.shape[0], positions.shape[0]), dtype=np.int64)
        remain_sim_user_ids = np.full(self.max_sim_user_num - len(filtered_sim_user_ids), -1, dtype=np.int64)
        if sim_seq.shape[0] == 0:
            sim_seq = remain_sim_seq
            sim_positions = remain_sim_positions
            sim_user_ids = remain_sim_user_ids
        else:
            sim_seq = np.concatenate([sim_seq, remain_sim_seq], axis=0)
            sim_positions = np.concatenate([sim_positions, remain_sim_positions], axis=0)
            sim_user_ids = np.concatenate([filtered_sim_user_ids, remain_sim_user_ids], axis=0)
        # if self.sim_users_scores is not None:
        #     # Add similarity between a user and top 100 most similar users, multiply by 100 because similarities are too close, making softmax too close
        #     return seq, pos, neg, positions, index, sim_seq, sim_positions, self.sim_users_scores[index][:self.sim_user_num] *100
        # else:
        return seq, pos, neg, positions, index, sim_seq, sim_positions, valid_mask, sim_user_ids
    

    def _get_user_seq(self, user):

        ### Get the sequence of required user
        inter = self.data[user]
        if user+1 in self.filted_item_dict:
            inter = [elem for elem in inter if elem not in self.filted_item_dict.get(user+1)]
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions
    
    
    def _filter_similar_users(self, index, sim_users):
        sim_seq, sim_positions, filtered_sim_user_ids = [], [], []
        
        # New: Relative percentile filtering (per-user adaptive filtering)
        if self.sim_filter_percentile > 0 and self.sim_users_scores is not None:
            sim_user_scores = self.sim_users_scores[index][:len(sim_users)]
            if len(sim_user_scores) > 0:
                # Calculate similarity threshold for current user (based on own distribution)
                # percentile=0.5 means keep top 50% similar users
                threshold = np.percentile(sim_user_scores, (1 - self.sim_filter_percentile) * 100)
                for i, sim_user in enumerate(sim_users):
                    if sim_user_scores[i] >= threshold:
                        meta_seq, meta_positions = self._get_user_seq(sim_user)
                        sim_seq.append(meta_seq)
                        sim_positions.append(meta_positions)
                        filtered_sim_user_ids.append(sim_user)
            # If no similar users after filtering, at least keep the most similar one
            if len(sim_seq) == 0 and len(sim_users) > 0:
                meta_seq, meta_positions = self._get_user_seq(sim_users[0])
                sim_seq.append(meta_seq)
                sim_positions.append(meta_positions)
                filtered_sim_user_ids.append(sim_users[0])
        elif self.filter_similar_users: 
            sim_user_scores = self.sim_users_scores[index][:len(sim_users)]  # Get specific similarity
            # Only filter for long users here
            if len(self.data[index]) > self.ts_user:
                for i,sim_user in enumerate(sim_users):
                    if sim_user_scores[i] > self.similar_gate:
                        # Same idea as above, get current user's sequence and positions
                        # TODO: Similar users' interaction sequences may not all be needed here
                        meta_seq, meta_positions = self._get_user_seq(sim_user)
                        sim_seq.append(meta_seq)
                        sim_positions.append(meta_positions)
                        filtered_sim_user_ids.append(sim_user)
            else:
                for i,sim_user in enumerate(sim_users):
                    meta_seq, meta_positions = self._get_user_seq(sim_user)
                    sim_seq.append(meta_seq)
                    sim_positions.append(meta_positions)
                    filtered_sim_user_ids.append(sim_user)
        else:
            for i,sim_user in enumerate(sim_users):
                meta_seq, meta_positions = self._get_user_seq(sim_user)
                sim_seq.append(meta_seq)
                sim_positions.append(meta_positions)
                filtered_sim_user_ids.append(sim_user)
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)
        filtered_sim_user_ids = np.array(filtered_sim_user_ids, dtype=np.int64)
        return sim_seq, sim_positions, filtered_sim_user_ids



class Seq2SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation with seq-to-seq loss'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data  # data is a list, each element is also a list containing item_ids user has interacted with
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq  # Default is False
        self.aug_seq_len = args.aug_seq_len  # Default is 0
        self.var_name = ["seq", "pos", "neg", "positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions



class Seq2SeqDatasetAllUser(Seq2SeqDataset):
    """
        SASRec series
    """
    def __init__(self, args, data, item_num, max_len, neg_num=1):
        """ 
        max_len = 200 
        data is a list, each element is also a list containing item_ids user has interacted with
        """

        super().__init__(args, data, item_num, max_len, neg_num)
        self.sim_user_num = args.sim_user_num  # Default: 10
        self.sim_long_user_num = args.sim_long_user_num
        self.max_sim_user_num = max(self.sim_user_num, self.sim_long_user_num)
        # Use relative path, relative to project root directory
        sim_user_path = get_sim_user_file_path(args)
        self.sim_users = pickle.load(open(sim_user_path, "rb"))
        self.similar_gate = args.similar_gate
        # New: Relative percentile filtering parameter (per-user adaptive filtering)
        self.sim_filter_percentile = getattr(args, 'sim_filter_percentile', 0.0)
        # If percentile filtering or global threshold filtering is enabled, need to load similarity scores
        if args.filter_similar_user or self.sim_filter_percentile > 0:
            sim_score_path = get_sim_user_score_file_path(args)
            if sim_score_path and os.path.exists(sim_score_path):
                self.sim_users_scores = pickle.load(open(sim_score_path, "rb"))
            else:
                self.sim_users_scores = None
        else:
            self.sim_users_scores = None
        self.filter_similar_users = args.filter_similar_user    
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions", "valid_mask", "sim_user_ids"]
        self.ts_user=args.ts_user
        self.args = args
        if args.filter_item:
            self.filted_item_dict = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", f"item_filter.pickle"), "rb"))
        else:
            self.filted_item_dict = {}
        
        # Dynamic K support: Online computation of recall count based on user interaction length + consistency
        self.use_dynamic_k = getattr(args, 'use_dynamic_k', False)
        self.user_dynamic_k = None
        if self.use_dynamic_k:
            # Use online computation method, dynamically compute K based on w1, w2 parameters
            self.user_dynamic_k = load_dynamic_k_from_analysis(args)
            if self.user_dynamic_k is not None:
                # Update max_sim_user_num to maximum value of dynamic K
                max_dynamic_k = max(self.user_dynamic_k.values())
                self.max_sim_user_num = max(self.max_sim_user_num, max_dynamic_k)
            else:
                print(f"[DynamicK] Failed to load, using fixed sim_user_num")
                self.use_dynamic_k = False

    def __getitem__(self, index):
        # 
        inter = self.data[index]  # A list, a user's interaction sequence, item_id
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)

        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            # Randomly select a number from [1, self.item_num+1], this number cannot be in non_neg
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        # aug_seq defaults to False, aug_seq_len defaults to 0
        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
        # If user's sequence is greater than max_len, then position length is max_len; otherwise it's the sequence's own length
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]  # [-200: ] If user's sequence length > max_len, this line gets all contents of positions. Else positions length is actual sequence length, also < max_len. So this line always gets all contents of positions
        positions = [0] * mask_len + positions  # If user's sequence length > max_len, positions length is max_len, no need to add new. Else user's sequence length < max_len, then positions length is actual sequence length, because length must be max_len, so missing parts are filled with 0

        positions = np.array(positions)

        ### Get the sequence of similar users
        # Get recall count of similar users for current user
        if self.use_dynamic_k and self.user_dynamic_k is not None and index in self.user_dynamic_k:
            # Use dynamic K: recall count computed based on user interaction length + consistency
            current_sim_num = self.user_dynamic_k[index]
        elif len(inter) >= self.ts_user:
            # Long users use sim_long_user_num
            current_sim_num = self.sim_long_user_num
        else:
            # Short users use sim_user_num
            current_sim_num = self.sim_user_num
        sim_users = self.sim_users[index][:current_sim_num]

        
        sim_seq, sim_positions, filtered_sim_user_ids = self._filter_similar_users(index, sim_users)
        valid_mask = torch.zeros(self.max_sim_user_num, dtype=torch.int32) 
        valid_mask[:sim_seq.shape[0]] = 1
        
        remain_sim_seq = np.zeros((self.max_sim_user_num-sim_seq.shape[0], seq.shape[0]), dtype=np.int32)
        remain_sim_positions = np.zeros((self.max_sim_user_num-sim_positions.shape[0], positions.shape[0]), dtype=np.int64)
        remain_sim_user_ids = np.full(self.max_sim_user_num - len(filtered_sim_user_ids), -1, dtype=np.int64)
        if sim_seq.shape[0] == 0:  # Represents all filtered out
            sim_seq = remain_sim_seq
            sim_positions = remain_sim_positions
            sim_user_ids = remain_sim_user_ids
        else:
            sim_seq = np.concatenate([sim_seq, remain_sim_seq], axis=0)
            sim_positions = np.concatenate([sim_positions, remain_sim_positions], axis=0)
            sim_user_ids = np.concatenate([filtered_sim_user_ids, remain_sim_user_ids], axis=0)
        
        return seq, pos, neg, positions, index, sim_seq, sim_positions, valid_mask, sim_user_ids
    

    def _get_user_seq(self, user):

        ### Get the sequence of required user
        # self.data is a list, each element is also a list containing item_ids user has interacted with
        inter = self.data[user]
        # TODO: Filter here
        if user+1 in self.filted_item_dict:
            inter = [elem for elem in inter if elem not in self.filted_item_dict.get(user+1)]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions
    

    def _filter_similar_users(self, index, sim_users):
        sim_seq, sim_positions, filtered_sim_user_ids = [], [], []
        
        # New: Relative percentile filtering (per-user adaptive filtering)
        if self.sim_filter_percentile > 0 and self.sim_users_scores is not None:
            sim_user_scores = self.sim_users_scores[index][:len(sim_users)]
            if len(sim_user_scores) > 0:
                # Calculate similarity threshold for current user (based on own distribution)
                threshold = np.percentile(sim_user_scores, (1 - self.sim_filter_percentile) * 100)
                for i, sim_user in enumerate(sim_users):
                    if sim_user_scores[i] >= threshold:
                        meta_seq, meta_positions = self._get_user_seq(sim_user)
                        sim_seq.append(meta_seq)
                        sim_positions.append(meta_positions)
                        filtered_sim_user_ids.append(sim_user)
            # If no similar users after filtering, at least keep the most similar one
            if len(sim_seq) == 0 and len(sim_users) > 0:
                meta_seq, meta_positions = self._get_user_seq(sim_users[0])
                sim_seq.append(meta_seq)
                sim_positions.append(meta_positions)
                filtered_sim_user_ids.append(sim_users[0])
        elif self.filter_similar_users: 
            sim_user_scores = self.sim_users_scores[index][:len(sim_users)]  # Get specific similarity
            # Only filter for long users here
            # TODO: Missing ablation hyperparameters for this component
            if len(self.data[index]) > self.ts_user:
                for i,sim_user in enumerate(sim_users):
                    if sim_user_scores[i] > self.similar_gate:
                        # Same idea as above, get current user's sequence and positions
                        # TODO: Similar users' interaction sequences may not all be needed here
                        meta_seq, meta_positions = self._get_user_seq(sim_user)
                        sim_seq.append(meta_seq)
                        sim_positions.append(meta_positions)
                        filtered_sim_user_ids.append(sim_user)
            else:
                for i,sim_user in enumerate(sim_users):
                    meta_seq, meta_positions = self._get_user_seq(sim_user)
                    sim_seq.append(meta_seq)
                    sim_positions.append(meta_positions)
                    filtered_sim_user_ids.append(sim_user)
        else:
            for i,sim_user in enumerate(sim_users):
                meta_seq, meta_positions = self._get_user_seq(sim_user)
                sim_seq.append(meta_seq)
                sim_positions.append(meta_positions)
                filtered_sim_user_ids.append(sim_user)
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)
        filtered_sim_user_ids = np.array(filtered_sim_user_ids, dtype=np.int64)
        return sim_seq, sim_positions, filtered_sim_user_ids


class BertRecTrainDatasetAllUser(Dataset):
    '''The train dataset for Bert4Rec'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_prob = args.mask_prob
        self.sim_user_num = args.sim_user_num  # Default: 10
        self.sim_long_user_num = args.sim_long_user_num
        self.max_sim_user_num = max(self.sim_user_num, self.sim_long_user_num)
        self.mask_token = item_num + 1
        # Use relative path, relative to project root directory
        sim_user_path = get_sim_user_file_path(args)
        self.sim_users = pickle.load(open(sim_user_path, "rb"))
        self.similar_gate = args.similar_gate
        self.ts_user=args.ts_user
        # New: Relative percentile filtering parameter (per-user adaptive filtering)
        self.sim_filter_percentile = getattr(args, 'sim_filter_percentile', 0.0)
        # If percentile filtering or global threshold filtering is enabled, need to load similarity scores
        if args.filter_similar_user or self.sim_filter_percentile > 0:
            sim_score_path = get_sim_user_score_file_path(args)
            if sim_score_path and os.path.exists(sim_score_path):
                self.sim_users_scores = pickle.load(open(sim_score_path, "rb"))
            else:
                self.sim_users_scores = None
        else:
            self.sim_users_scores = None
        self.filter_similar_users = args.filter_similar_user
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions", "valid_mask", "sim_user_ids"]
        if args.filter_item:
            self.filted_item_dict = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", f"item_filter.pickle"), "rb"))
        else:
            self.filted_item_dict = {}
        
        # Dynamic K support: Online computation of recall count based on user interaction length + consistency
        self.use_dynamic_k = getattr(args, 'use_dynamic_k', False)
        self.user_dynamic_k = None
        if self.use_dynamic_k:
            # Use online computation method, dynamically compute K based on w1, w2 parameters
            self.user_dynamic_k = load_dynamic_k_from_analysis(args)
            if self.user_dynamic_k is not None:
                # Update max_sim_user_num to maximum value of dynamic K
                max_dynamic_k = max(self.user_dynamic_k.values())
                self.max_sim_user_num = max(self.max_sim_user_num, max_dynamic_k)
            else:
                print(f"[DynamicK] Failed to load, using fixed sim_user_num")
                self.use_dynamic_k = False


    def __len__(self):

        return 2 * len(self.data)

    def __getitem__(self, index):

        tokens = []
        labels, neg_labels = [], []

        if index >= len(self.data):
            seq = self.data[index - len(self.data)]
            for s in seq:
                tokens.append(s)
                labels.append(0)
                neg_labels.append(0)
            labels[-1] = tokens[-1]
            neg_labels[-1] = random_neq(1, self.item_num+1, seq)
            tokens[-1] = self.mask_token

        else:
            seq = self.data[index]
   
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
                    neg = random_neq(1, self.item_num+1, seq)
                    neg_labels.append(neg)

                else:
                    tokens.append(s)
                    labels.append(0)
                    neg_labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_labels = neg_labels[-self.max_len:]
        pos = list(range(1, len(tokens)+1))
        pos= pos[-self.max_len:]

        mask_len = self.max_len - len(tokens)
        
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        neg_labels = [0] * mask_len + neg_labels
        pos = [0] * mask_len + pos

        if index >= len(self.data):
            user_id = index - len(self.data)
        else:
            user_id = index

        ### Get the sequence of similar users
        # Get recall count of similar users for current user
        if self.use_dynamic_k and self.user_dynamic_k is not None and user_id in self.user_dynamic_k:
            # Use dynamic K: recall count computed based on user interaction length + consistency
            current_sim_num = self.user_dynamic_k[user_id]
        elif len(self.data[user_id]) >= self.ts_user:
            # Long users use sim_long_user_num
            current_sim_num = self.sim_long_user_num
        else:
            # Short users use sim_user_num
            current_sim_num = self.sim_user_num
        sim_users = self.sim_users[user_id][:current_sim_num]
        sim_seq, sim_positions, filtered_sim_user_ids = self._filter_similar_users(user_id, sim_users)

        valid_mask = torch.zeros(self.max_sim_user_num, dtype=torch.int32) 
        valid_mask[:sim_seq.shape[0]] = 1

        remain_sim_seq = np.zeros((self.max_sim_user_num-sim_seq.shape[0], self.max_len), dtype=np.int32)
        remain_sim_positions = np.zeros((self.max_sim_user_num-sim_positions.shape[0], self.max_len), dtype=np.int64)
        remain_sim_user_ids = np.full(self.max_sim_user_num - len(filtered_sim_user_ids), -1, dtype=np.int64)
        if sim_seq.shape[0] == 0:
            sim_seq = remain_sim_seq
            sim_positions = remain_sim_positions
            sim_user_ids = remain_sim_user_ids
        else:
            sim_seq = np.concatenate([sim_seq, remain_sim_seq], axis=0)
            sim_positions = np.concatenate([sim_positions, remain_sim_positions], axis=0)
            sim_user_ids = np.concatenate([filtered_sim_user_ids, remain_sim_user_ids], axis=0)

        return np.array(tokens), np.array(labels), np.array(neg_labels), np.array(pos), user_id, sim_seq, sim_positions, valid_mask, sim_user_ids


    def _get_user_seq(self, user):

        ### get the sequence of required user
        inter = self.data[user]
        if user+1 in self.filted_item_dict:
            inter = [elem for elem in inter if elem not in self.filted_item_dict.get(user+1)]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions


    def _filter_similar_users(self, index, sim_users):
        sim_seq, sim_positions, filtered_sim_user_ids = [], [], []
        
        # New: Relative percentile filtering (per-user adaptive filtering)
        if self.sim_filter_percentile > 0 and self.sim_users_scores is not None:
            sim_user_scores = self.sim_users_scores[index][:len(sim_users)]
            if len(sim_user_scores) > 0:
                # Calculate similarity threshold for current user (based on own distribution)
                threshold = np.percentile(sim_user_scores, (1 - self.sim_filter_percentile) * 100)
                for i, sim_user in enumerate(sim_users):
                    if sim_user_scores[i] >= threshold:
                        meta_seq, meta_positions = self._get_user_seq(sim_user)
                        sim_seq.append(meta_seq)
                        sim_positions.append(meta_positions)
                        filtered_sim_user_ids.append(sim_user)
            # If no similar users after filtering, at least keep the most similar one
            if len(sim_seq) == 0 and len(sim_users) > 0:
                meta_seq, meta_positions = self._get_user_seq(sim_users[0])
                sim_seq.append(meta_seq)
                sim_positions.append(meta_positions)
                filtered_sim_user_ids.append(sim_users[0])
        elif self.filter_similar_users: 
            sim_user_scores = self.sim_users_scores[index][:len(sim_users)]  # Get specific similarity
            # Only filter for long users here
            if len(self.data[index]) > self.ts_user:
                for i,sim_user in enumerate(sim_users):
                    if sim_user_scores[i] > self.similar_gate:
                        # Same idea as above, get current user's sequence and positions
                        # TODO: Similar users' interaction sequences may not all be needed here
                        meta_seq, meta_positions = self._get_user_seq(sim_user)
                        sim_seq.append(meta_seq)
                        sim_positions.append(meta_positions)
                        filtered_sim_user_ids.append(sim_user)
            else:
                for i,sim_user in enumerate(sim_users):
                    meta_seq, meta_positions = self._get_user_seq(sim_user)
                    sim_seq.append(meta_seq)
                    sim_positions.append(meta_positions)
                    filtered_sim_user_ids.append(sim_user)
        else:
            for i,sim_user in enumerate(sim_users):
                meta_seq, meta_positions = self._get_user_seq(sim_user)
                sim_seq.append(meta_seq)
                sim_positions.append(meta_positions)
                filtered_sim_user_ids.append(sim_user)
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)
        filtered_sim_user_ids = np.array(filtered_sim_user_ids, dtype=np.int64)
        return sim_seq, sim_positions, filtered_sim_user_ids


# ============================================================================
# RCL-specific Dataset: Returns global user_id for RCL SSL loss calculation
# ============================================================================

class RCLSeq2SeqDataset(Seq2SeqDataset):
    """
    RCL-specific Seq2Seq Dataset
    Key difference: Returns global user_id (index) for RCL's global_pos/global_hard_neg indexing
    
    Returns: (seq, pos, neg, positions, user_id)
    """
    
    def __init__(self, args, data, item_num, max_len, neg_num=1):
        super().__init__(args, data, item_num, max_len, neg_num)
        # Update var_name to include user_id
        self.var_name = ["seq", "pos", "neg", "positions", "user_id"]
    
    def __getitem__(self, index):
        # Call parent class to get base data
        seq, pos, neg, positions = super().__getitem__(index)
        
        # Return global user_id (i.e., index, since data is ordered by user)
        user_id = index
        
        return seq, pos, neg, positions, user_id


class RCLSeqDataset(SeqDataset):
    """
    RCL-specific Seq Dataset (for models like GRU4Rec that only need the last position)
    Key difference: Returns global user_id
    
    Returns: (seq, pos, neg, positions, user_id)
    """
    
    def __init__(self, data, item_num, max_len, neg_num=1):
        super().__init__(data, item_num, max_len, neg_num)
        # Update var_name to include user_id
        self.var_name = ["seq", "pos", "neg", "positions", "user_id"]
    
    def __getitem__(self, index):
        # Call parent class to get base data
        seq, pos, neg, positions = super().__getitem__(index)
        
        # Return global user_id
        user_id = index
        
        return seq, pos, neg, positions, user_id


class RCLBertRecTrainDataset(Dataset):
    """
    RCL-specific BERT4Rec Dataset
    Key difference: Returns global user_id, does not use sim_users (RCL has its own similar user mechanism)
    
    Returns: (seq, pos, neg, positions, user_id)
    """
    
    def __init__(self, args, data, item_num, max_len, neg_num=1):
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_prob = args.mask_prob
        self.mask_token = item_num + 1
        # RCL-specific: var_name only contains base fields and user_id
        self.var_name = ["seq", "pos", "neg", "positions", "user_id"]
    
    def __len__(self):
        # BERT4Rec typically doubles the dataset (once for MLM, once for predict last)
        return 2 * len(self.data)
    
    def __getitem__(self, index):
        tokens = []
        labels, neg_labels = [], []
        
        if index >= len(self.data):
            # Second half: predict last token
            real_index = index - len(self.data)
            seq = self.data[real_index]
            for s in seq:
                tokens.append(s)
                labels.append(0)
                neg_labels.append(0)
            labels[-1] = tokens[-1]
            neg_labels[-1] = random_neq(1, self.item_num+1, seq)
            tokens[-1] = self.mask_token
            user_id = real_index
        else:
            # First half: random mask
            seq = self.data[index]
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
                    neg = random_neq(1, self.item_num+1, seq)
                    neg_labels.append(neg)
                else:
                    tokens.append(s)
                    labels.append(0)
                    neg_labels.append(0)
            user_id = index
        
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_labels = neg_labels[-self.max_len:]
        pos = list(range(1, len(tokens)+1))
        pos = pos[-self.max_len:]
        
        mask_len = self.max_len - len(tokens)
        
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        neg_labels = [0] * mask_len + neg_labels
        pos = [0] * mask_len + pos
        
        return np.array(tokens), np.array(labels), np.array(neg_labels), np.array(pos), user_id
