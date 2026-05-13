import os
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
np.random.seed(42)
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.special import softmax
import argparse


parser = argparse.ArgumentParser(description="相似用户检索脚本")

# 添加参数
parser.add_argument('--dataset', default="steam", type=str,
                    help='数据集名称')
parser.add_argument('--sim_metric', default="cos", type=str,
                    help='相似度计算方法: cos (余弦相似度) 或 sin (点积)')
parser.add_argument('--topk', default=18, type=int,
                    help='返回的top-k相似用户数量')
parser.add_argument('--mode', default="mean_rnn", type=str,
                    help='模式名称（用于从pt文件合并时使用）')
parser.add_argument('--emb_mode', default="llm", type=str,
                    help='嵌入模式')
parser.add_argument('--model', default="llmesr_mean_sasrec", type=str,
                    help='模型名称')
parser.add_argument('--input_emb_file', type=str, default=None,
                    help='直接指定输入嵌入文件路径，如: data/steam/handled/usr_emb_router_wo_select_qwen2.5-7b-instruct_doubao_pca128.pkl')
parser.add_argument('--skip_concat', action='store_true',
                    help='跳过concat步骤（当使用--input_emb_file时自动跳过）')


# 解析参数
args = parser.parse_args()


class ItemVectorSimilarity:
    def __init__(self, user_item_dict):
        """Initialize item vector similarity calculator"""
        # Build item ID to index mapping
        self.item_to_index = {}
        self.index_to_item = []
        self._build_item_index(user_item_dict)
        
        # Number of items, used for vector construction
        self.item_num = len(self.index_to_item)
        
        # Pre-compute user vectors
        self.user_vectors = self._build_user_vectors(user_item_dict)
        
        # Set of valid user IDs
        self.valid_user_ids = set(self.user_vectors.keys())
        
    def _build_item_index(self, user_item_dict):
        """Build item ID to index mapping"""
        all_items = set()
        for items in user_item_dict.values():
            all_items.update(items)
        
        self.index_to_item = list(all_items)
        self.item_to_index = {item: idx for idx, item in enumerate(self.index_to_item)}
        print("Total number of items:", len(self.index_to_item))

    def _build_user_vectors(self, user_item_dict):
        """Build user vectors (one-hot encoding)"""
        user_vectors = {}
        for user_id, items in user_item_dict.items():
            vector = self.items_to_vector(items)
            user_vectors[user_id] = vector
        return user_vectors
        
    def items_to_vector(self, items):
        """Convert item list to one-hot encoded vector"""
        vector = [0] * self.item_num
        for item in items:
            if item in self.item_to_index:
                vector[self.item_to_index[item]] = 1
        return vector
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a**2 for a in vec1))
        norm_b = math.sqrt(sum(b**2 for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot_product / (norm_a * norm_b)

    def cosine_similarity_group(self, vec, vec_list):
        """Calculate cosine similarity between a single vector and all vectors in a list"""
        # Convert to NumPy arrays
        vec = np.array(vec)
        vec_list = np.array(vec_list)
        
        # Pre-compute norm of input vector
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return [0] * len(vec_list)
        
        # Vectorized computation of all dot products
        dot_products = np.dot(vec_list, vec)
        
        # Pre-compute norms of all vectors in the list
        vec_list_norms = np.linalg.norm(vec_list, axis=1)
        
        # Handle zero norm cases
        mask = vec_list_norms == 0
        vec_list_norms[mask] = 1  # Avoid division by zero, these will be set to 0
        
        # Calculate cosine similarity
        similarities = dot_products / (vec_norm * vec_list_norms)
        
        # Set similarity to 0 for zero-norm vectors
        similarities[mask] = 0
        
        return similarities.tolist()

    def jaccard_similarity(self, vec1, vec2):
        """Calculate Jaccard similarity"""
        intersection = sum(1 for a, b in zip(vec1, vec2) if a == b == 1)
        union = sum(1 for a, b in zip(vec1, vec2) if a == 1 or b == 1)
        if union == 0:
            return 0
        return intersection / union
    
    def calculate_similarity(self, user1_id, user2_id, method="cosine"):
        """Calculate similarity between two users"""
        if user1_id not in self.user_vectors or user2_id not in self.user_vectors:
            raise ValueError(f"User ID {user1_id} or {user2_id} does not exist")
        
        vec1 = self.user_vectors[user1_id]
        vec2 = self.user_vectors[user2_id]
        
        if method == "cosine":
            return self.cosine_similarity(vec1, vec2)
        elif method == "jaccard":
            return self.jaccard_similarity(vec1, vec2)
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")
        
    def calculate_similarity_group(self, user1_id, user_id_list, method="cosine"):
        """Calculate similarity between target user and multiple users"""
        if user1_id not in self.user_vectors:
            raise ValueError(f"User ID {user1_id} does not exist")
        
        # Filter out non-existent user IDs
        valid_user_ids = [user_id for user_id in user_id_list if user_id in self.user_vectors]
        
        if not valid_user_ids:
            return [0] * len(user_id_list)
        
        vec1 = self.user_vectors[user1_id]
        vec_list = [self.user_vectors[user_id] for user_id in valid_user_ids]
        
        # Build result mapping, non-existent user IDs have similarity 0
        similarity_map = {}
        if method == "cosine":
            similarities = self.cosine_similarity_group(vec1, vec_list)
            similarity_map = {user_id: sim for user_id, sim in zip(valid_user_ids, similarities)}
        else:
            raise ValueError(f"Unsupported similarity calculation method: {method}")
        
        # Return similarity scores in original order
        return [similarity_map.get(user_id, 0) for user_id in user_id_list]

    def pearson_correlation(self, target_user_id, similar_user_ids):
        """
        Calculate Pearson correlation coefficient between target user vector and multiple similar user vectors.
        Handles cases where similar user IDs do not exist, ensuring vector dimensions are consistent.
        """
        if target_user_id not in self.user_vectors:
            print(target_user_id)
            raise ValueError(f"Target user ID {target_user_id} does not exist")
        
        # Filter out non-existent similar user IDs
        valid_similar_user_ids = [user_id for user_id in similar_user_ids 
                                  if user_id in self.user_vectors]
        
        if not valid_similar_user_ids:
            # If no valid similar users, return zero array
            return np.zeros(len(similar_user_ids))
        
        # Get target user vector and valid similar user vectors
        target_vector = np.array(self.user_vectors[target_user_id])
        similar_vectors = np.array([self.user_vectors[user_id] 
                                    for user_id in valid_similar_user_ids])
        
        # Calculate Pearson correlation coefficient
        if len(valid_similar_user_ids) == 1:
            # Handle case with only one similar user
            similar_vector = similar_vectors[0]
            if np.var(target_vector) == 0 or np.var(similar_vector) == 0:
                return np.array([0.0])
            correlation = np.corrcoef(target_vector, similar_vector)[0, 1]
            correlations = np.array([correlation])
        else:
            # Case with multiple similar users
            all_vectors = np.vstack([target_vector, similar_vectors])
            correlation_matrix = np.corrcoef(all_vectors)
            correlations = correlation_matrix[0, 1:]
        
        # Build result mapping, non-existent user IDs have correlation 0
        correlation_map = {user_id: corr for user_id, corr 
                          in zip(valid_similar_user_ids, correlations)}
        
        # Return correlation scores in original order
        return np.array([correlation_map.get(user_id, 0.0) 
                        for user_id in similar_user_ids])


def concat_user_emb(is_save=True):
    all_dict = {}
    # Use relative path, relative to project root directory
    folder_path = f"./prompt/data/{args.dataset}/{args.mode}"
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            temp_dict = torch.load(file_path)
            all_dict.update(temp_dict)
    sorted_keys = sorted([int(k) for k in all_dict.keys()])

    # Add corresponding values to list in sorted key order
    sorted_values = [all_dict[str(key)] for key in sorted_keys]

    # Convert list to ndarray
    result_array = np.array(sorted_values)
    if is_save:
        pickle.dump(result_array, open(f"./data/{args.dataset}/handled/usr_emb_np_qwen_{args.mode}.pkl", "wb"))
        print("concat user emb over")

def load_user_emb(input_file=None):
    """
    加载用户嵌入向量
    
    Args:
        input_file: 输入文件路径，如果为None则使用默认路径
    
    Returns:
        用户嵌入向量 (n_users, emb_dim)
    """
    if input_file is None:
        # 使用默认路径
        emb_path = os.path.join(args.dataset, "handled", f"usr_emb_np_qwen_{args.mode}.pkl")
    else:
        # 使用指定的文件路径
        emb_path = input_file
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"嵌入文件不存在: {emb_path}")
    
    print(f"加载用户嵌入: {emb_path}")
    user_llm_emb = pickle.load(open(emb_path, "rb"))
    print(f"嵌入向量形状: {user_llm_emb.shape}")
    print("load user emb over")
    return user_llm_emb


def get_output_basename(input_file=None):
    """
    根据输入文件名生成输出文件的基础名称
    
    Args:
        input_file: 输入文件路径
    
    Returns:
        输出文件的基础名称（不含扩展名）
    """
    if input_file:
        # 从输入文件名提取基础名称
        input_basename = os.path.basename(input_file)
        # 移除扩展名
        base_name = os.path.splitext(input_basename)[0]
        return base_name
    else:
        # 使用默认命名方式
        return f"usr_emb_np_qwen_{args.mode}"


def get_output_dir(input_file=None):
    """
    获取输出文件目录
    
    Args:
        input_file: 输入文件路径
    
    Returns:
        输出文件目录路径
    """
    if input_file:
        # 从输入文件路径提取目录
        output_dir = os.path.dirname(input_file)
        return output_dir
    else:
        # 使用默认目录
        return os.path.join("data", args.dataset, "handled")


def rank_by_similar(user_llm_emb, is_save=True, output_basename=None, output_dir=None):
    """
    基于用户嵌入计算相似度并排序
    
    Args:
        user_llm_emb: 用户嵌入向量
        is_save: 是否保存结果
        output_basename: 输出文件的基础名称（不含扩展名）
        output_dir: 输出文件目录
    
    Returns:
        相似用户排名矩阵 (n_users, topk)
    """
    # Calculate the similarity score between users based on LLM user embedding
    if args.emb_mode == "llm":
        print("Using LLM embeddings for similarity retrieval")
        user_emb = user_llm_emb
    # else:
    #     print("Using ESR encoder embeddings for similarity retrieval")
    #     user_emb = user_collab_emb
    
    print(f"计算相似度矩阵 (相似度方法: {args.sim_metric})...")
    if args.sim_metric == "sin":
        score_matrix = np.dot(user_emb, user_emb.T)
    elif args.sim_metric == "cos":
        score_matrix = cosine_similarity(user_emb, user_emb)
    else:
        raise ValueError(f"不支持的相似度方法: {args.sim_metric}")

    rank_matrix = np.argsort(-score_matrix, axis=-1)    # User id starts from 0
    final_rank_matrix = rank_matrix[:, 1:]  # The first value (rank_matrix[:, 0]) in each row is the user's own index, as the highest similarity is always with itself
    final_rank_matrix = final_rank_matrix[:, :args.topk]  # Remove self-similarity from original rank_matrix and keep topk users with highest similarity to current user
    
    if is_save:
        if output_basename is None:
            output_basename = f"sim_user_100_qwen_{args.mode}_{args.emb_mode}"
        if output_dir is None:
            output_dir = os.path.join("data", args.dataset, "handled")
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_basename}_sim_user_top{args.topk}.pkl")
        pickle.dump(final_rank_matrix, open(output_path, "wb"))
        print(f"已保存相似用户矩阵: {output_path}")
    print("rank_by_similar over")
    return final_rank_matrix


def load_sequence():
    """
    加载用户-物品交互序列
    
    Returns:
        用户-物品交互字典，格式: {user_id: [item_id1, item_id2, ...]}
    """
    User = defaultdict(list)
    seq_len = []
    usernum, itemnum = 0, 0
    
    # 使用正确的路径格式
    inter_file = os.path.join("data", args.dataset, "handled", "inter.txt")
    if not os.path.exists(inter_file):
        raise FileNotFoundError(f"交互文件不存在: {inter_file}")
    
    print(f"加载用户序列: {inter_file}")
    f = open(inter_file, 'r')
    for line in f:  # use a dict to save all seqeuces of each user
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user, seq in User.items():
        seq_len.append(len(seq))
    print(f"load_sequence over: {len(User)} 个用户, {itemnum} 个物品")
    return User


def cal_person_similar(User, final_rank_matrix, is_save=True, output_basename=None, output_dir=None):
    """
    计算Pearson相关系数
    
    Args:
        User: 用户-物品交互字典
        final_rank_matrix: 相似用户排名矩阵
        is_save: 是否保存结果
        output_basename: 输出文件的基础名称（不含扩展名）
        output_dir: 输出文件目录
    """
    user_item_dict = User
    
    # Initialize similarity calculator
    similarity_calculator = ItemVectorSimilarity(user_item_dict)
    
    # Calculate Pearson correlation coefficient for all users and their similar users
    sim_scores = []
    # Note: Assume row indices of final_rank_matrix are consistent with user IDs in user_item_dict
    # If not consistent, corresponding mapping conversion is needed
    print("计算Pearson相关系数...")
    for i in tqdm(range(final_rank_matrix.shape[0])):
        # Get current user ID (assumed to be i+1, adjust according to actual situation)
        user_id = i + 1
        # user_id = i
        # Get similar user list for this user
        similar_users = final_rank_matrix[i]
        # Calculate Pearson correlation coefficient
        sim = similarity_calculator.pearson_correlation(user_id, similar_users)
        sim_scores.append(sim)
    
    # Convert to NumPy array for better shape viewing
    sim_scores = np.array(sim_scores)
    flat_sims = sim_scores[:,:18].flatten() 
    octiles = pd.Series(flat_sims).quantile(q=[i/8 for i in range(1,8)])
    print("Pearson相关系数分位数:")
    for i,octil in enumerate(octiles,1):
        print(f"  {i}/8: {octil:.4f}")
    
    ## Save LLM embedding based similar users
    if is_save:
        if output_basename is None:
            output_basename = f"sim_user_pearson_score_100_qwen_{args.mode}_{args.emb_mode}"
        if output_dir is None:
            output_dir = os.path.join("data", args.dataset, "handled")
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_basename}_pearson_score.pkl")
        pickle.dump(np.array(sim_scores), open(output_path, "wb"))
        print(f"已保存Pearson相关系数: {output_path}")
    print("cal_person_similar over")


if __name__ == "__main__":
    print("=" * 80)
    print("相似用户检索")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"相似度方法: {args.sim_metric}")
    print(f"Top-K: {args.topk}")
    print("=" * 80)
    
    # 确定是否跳过concat步骤
    skip_concat = args.skip_concat or args.input_emb_file is not None
    
    # 获取输出文件基础名称和目录
    output_basename = None
    output_dir = None
    if args.input_emb_file:
        output_basename = get_output_basename(args.input_emb_file)
        output_dir = get_output_dir(args.input_emb_file)
        print(f"使用指定的嵌入文件: {args.input_emb_file}")
        print(f"输出文件基础名称: {output_basename}")
        print(f"输出目录: {output_dir}")
    else:
        print(f"模式: {args.mode}")
        if not skip_concat:
            print("从pt文件合并嵌入...")
            concat_user_emb(is_save=True)
    
    # 加载用户嵌入
    user_llm_emb = load_user_emb(args.input_emb_file)
    
    # 计算相似用户
    final_rank_matrix = rank_by_similar(user_llm_emb, is_save=True, output_basename=output_basename, output_dir=output_dir)
    
    # 加载用户序列
    User = load_sequence()
    
    # 计算Pearson相关系数
    cal_person_similar(User, final_rank_matrix, is_save=True, output_basename=output_basename, output_dir=output_dir)
    
    print("=" * 80)
    print("处理完成!")
    print("=" * 80)