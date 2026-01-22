# here put the import lib
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # only add when conv in your model


def get_n_params(model, logger):
    '''Get the number of parameters of model'''
    pp = 0
    for p in list(model.named_parameters()):
        nn = 1
        for s in list(p[1].size()):
            nn = nn*s
        # logger.info(f"param name={p[0]}")
        # logger.info(f"param size={nn}")
        pp += nn
    return pp


def get_n_params_(parameter_list):
    '''Get the number of parameters of model'''
    pp = 0
    for p in list(parameter_list):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def unzip_data(data, aug=True, aug_num=0):
    """data is a dict, key is user_id, value is list containing item_ids"""
    res = []
    # Default is False
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)

            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i])
    else:
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)

    return res


def unzip_data_with_user(data, aug=True, aug_num=0):

    res = []
    users = []
    user_id = 1
    
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)

            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i])
                users.append(user_id)

            user_id += 1

    else:
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)
            users.append(user_id)
            user_id += 1

    return res, users


def concat_data(data_list):
    """Input a list, may contain three or two elements. Corresponding to training, validation and test parts of all users respectively.
    Function is to concatenate training, validation, test items of each user to form complete sequences
    Example: [[1,2,3,4,5],[1,2,3,4,5,6],[3,4,5]]
    """

    res = []

    if len(data_list) == 2:  # Corresponding to eval

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            res.append(train[user]+valid[user])
    
    elif len(data_list) == 3:  # Corresponding to test

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:
            # Merge two item_id lists
            res.append(train[user]+valid[user]+test[user])

    else:

        raise ValueError

    return res


def concat_aug_data(data_list):

    res = []

    train = data_list[0]
    valid = data_list[1]

    for user in train:

        if len(valid[user]) == 0:
            res.append([train[user][0]])
        
        else:
            res.append(train[user]+valid[user])

    return res


def concat_data_with_user(data_list):

    res = []
    users = []
    user_id = 1

    if len(data_list) == 2:

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            res.append(train[user]+valid[user])
            users.append(user_id)
            user_id += 1
    
    elif len(data_list) == 3:

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:

            res.append(train[user]+valid[user]+test[user])
            users.append(user_id)
            user_id += 1

    else:

        raise ValueError

    return res, users


def filter_data(data, thershold=5):
    '''Filter out the sequence shorter than threshold'''
    res = []

    for user in data:

        if len(user) > thershold:
            res.append(user)
        else:
            continue
    
    return res



def random_neq(l, r, s=[]):    
    """Randomly sample a number between l and r, this number cannot be in list s"""
    
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t



def metric_report(data_rank, topk=10):

    NDCG, HT = 0, 0
    
    for rank in data_rank:

        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return {'NDCG@10': NDCG / len(data_rank),
            'HR@10': HT / len(data_rank)}



def metric_analysis_len_report(data_rank, data_len, pop_dict, target_items, topk=10, aug_len=0, aug_pop=0, args=None):
    """data_len: effective sequence length for each user
    User Group
    """
    if args is not None:
        ts_user = args.ts_user
        ts_tail = args.ts_item
    else:
        ts_user = 10
        ts_item = 10

    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0
    # Short sequence users: users with interaction sequence length less than ts_user + aug_len
    count_s = len(data_len[data_len<ts_user+aug_len])
    count_l = len(data_len[data_len>=ts_user+aug_len])
    item_pop = pop_dict[target_items.astype("int64")]
    plot_short_user_x = []
    plot_short_user_y = []
    plot_short_user_label = []
    plot_short_user_index = []
    plot_long_user_x = []
    plot_long_user_y = []
    plot_long_user_label = []
    plot_long_user_index = []

    plot_tail_item_x = []
    plot_tail_item_y = []
    plot_tail_item_label = []
    plot_tail_item_index = []
    plot_pop_item_x = []
    plot_pop_item_y = []
    plot_pop_item_label = []
    plot_pop_item_index = []

    for i, rank in enumerate(data_rank):

        if data_len[i] < ts_user+aug_len:
            # Short sequence users
            plot_short_user_x.append(data_len[i])
            plot_short_user_y.append(rank)
            plot_short_user_index.append(i)
            if rank < topk:
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            if item_pop[i] < ts_tail+aug_pop:
                plot_short_user_label.append("tail")
            elif item_pop[i] >= ts_tail+aug_pop:
                plot_short_user_label.append("pop")
        elif data_len[i] >= ts_user+aug_len:
            plot_long_user_x.append(data_len[i])
            plot_long_user_y.append(rank)
            plot_long_user_index.append(i)
            # Long sequence users
            # If it's a long sequence user, and the prediction for target_item is not in top 10, then is that target_item pop or non-pop
            if rank < topk:
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1
            if item_pop[i] < ts_tail+aug_pop:
                plot_long_user_label.append("tail")
            elif item_pop[i] >= ts_tail+aug_pop:
                plot_long_user_label.append("pop")
        
        if item_pop[i] < ts_tail+aug_pop:
            plot_tail_item_x.append(item_pop[i])
            plot_tail_item_y.append(rank)
            plot_tail_item_index.append(i)
            if data_len[i] < ts_user+aug_len:
                plot_tail_item_label.append("tail")
            elif data_len[i] >= ts_user+aug_len:
                plot_tail_item_label.append("pop")

        elif item_pop[i] >= ts_tail+aug_pop:
            plot_pop_item_x.append(item_pop[i])
            plot_pop_item_y.append(rank)
            plot_pop_item_index.append(i)
            if data_len[i] < ts_user+aug_len:
                plot_pop_item_label.append("tail")
            elif data_len[i] >= ts_user+aug_len:
                plot_pop_item_label.append("pop")
    
    short_user_df = pd.DataFrame({"x": plot_short_user_x, "y": plot_short_user_y, "label": plot_short_user_label, "index": plot_short_user_index})
    # Use relative path, relative to project root directory
    short_user_df.to_csv("./data/steam/analysis/llmesr_mean_bert4rec/short_user.csv", index=False)
    long_user_df = pd.DataFrame({"x": plot_long_user_x, "y": plot_long_user_y, "label": plot_long_user_label, "index": plot_long_user_index})
    long_user_df.to_csv("./data/steam/analysis/llmesr_mean_bert4rec/long_user.csv", index=False)

    tail_item_df = pd.DataFrame({"x": plot_tail_item_x, "y": plot_tail_item_y, "label": plot_tail_item_label, "index": plot_tail_item_index})
    tail_item_df.to_csv("./data/steam/analysis/llmesr_mean_bert4rec/tail_item.csv", index=False)
    pop_item_df = pd.DataFrame({"x": plot_pop_item_x, "y": plot_pop_item_y, "label": plot_pop_item_label, "index": plot_pop_item_index})
    pop_item_df.to_csv("./data/steam/analysis/llmesr_mean_bert4rec/pop_item.csv", index=False)


    return {'Short NDCG@10': NDCG_s / count_s if count_s!=0 else 0, # avoid division of 0
            'Short HR@10': HT_s / count_s if count_s!=0 else 0,
            'Long NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
            'Long HR@10': HT_l / count_l if count_l!=0 else 0,}


def metric_len_report(data_rank, data_len, topk=10, aug_len=0, args=None):
    """data_len: effective sequence length for each user
    User Group
    """
    if args is not None:
        ts_user = args.ts_user
    else:
        ts_user = 10

    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0
    # Short sequence users: users with interaction sequence length less than ts_user + aug_len
    count_s = len(data_len[data_len<ts_user+aug_len])
    count_l = len(data_len[data_len>=ts_user+aug_len])

    for i, rank in enumerate(data_rank):

        if rank < topk:

            if data_len[i] < ts_user+aug_len:
                # Short sequence users
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            else:
                # Long sequence users
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1

    return {'Short NDCG@10': NDCG_s / count_s if count_s!=0 else 0, # avoid division of 0
            'Short HR@10': HT_s / count_s if count_s!=0 else 0,
            'Long NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
            'Long HR@10': HT_l / count_l if count_l!=0 else 0,}


def metric_pop_report(data_rank, pop_dict, target_items, topk=10, aug_pop=0, args=None):
    """
    data_rank: user prediction ranking results
    target_items: pos list
    Report the metrics according to target item's popularity
    pop_dict: the array of item's popularity
    """
    if args is not None:  # the threshold to split the long-tail and popular items
        ts_tail = args.ts_item
    else:
        ts_tail = 20

    NDCG_s, HT_s = 0, 0
    NDCG_l, HT_l = 0, 0
    item_pop = pop_dict[target_items.astype("int64")]  # TODO: check the distribution here
    count_s = len(item_pop[item_pop<ts_tail+aug_pop])
    count_l = len(item_pop[item_pop>=ts_tail+aug_pop])
    # len(data_rank) = 23310
    for i, rank in enumerate(data_rank):

        if i == 0:  # skip the padding index TODO: WHY
            continue

        if rank < topk:

            if item_pop[i] < ts_tail+aug_pop:
                NDCG_s += 1 / np.log2(rank + 2)
                HT_s += 1
            else:
                NDCG_l += 1 / np.log2(rank + 2)
                HT_l += 1

    return {'Tail NDCG@10': NDCG_s / count_s if count_s!=0 else 0,
            'Tail HR@10': HT_s / count_s if count_s!=0 else 0,
            'Popular NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
            'Popular HR@10': HT_l / count_l if count_l!=0 else 0,}



def metric_len_5group(pred_rank, 
                      seq_len, 
                      thresholds=[5, 10, 15, 20], 
                      topk=10):

    NDCG = np.zeros(5)
    HR = np.zeros(5)    
    for i, rank in enumerate(pred_rank):

        target_len = seq_len[i]
        if rank < topk:

            if target_len < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_len < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            elif target_len < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

            elif target_len < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1

            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    count[0] = len(seq_len[seq_len>=0]) - len(seq_len[seq_len>=thresholds[0]])
    count[1] = len(seq_len[seq_len>=thresholds[0]]) - len(seq_len[seq_len>=thresholds[1]])
    count[2] = len(seq_len[seq_len>=thresholds[1]]) - len(seq_len[seq_len>=thresholds[2]])
    count[3] = len(seq_len[seq_len>=thresholds[2]]) - len(seq_len[seq_len>=thresholds[3]])
    count[4] = len(seq_len[seq_len>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count



def metric_pop_5group(pred_rank, 
                      pop_dict, 
                      target_items, 
                      thresholds=[10, 30, 60, 100], 
                      topk=10):

    NDCG = np.zeros(5)
    HR = np.zeros(5)    
    for i, rank in enumerate(pred_rank):

        target_pop = pop_dict[int(target_items[i])]
        if rank < topk:

            if target_pop < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_pop < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            elif target_pop < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

            elif target_pop < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1

            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    pop = pop_dict[target_items.astype("int64")]
    count[0] = len(pop[pop>=0]) - len(pop[pop>=thresholds[0]])
    count[1] = len(pop[pop>=thresholds[0]]) - len(pop[pop>=thresholds[1]])
    count[2] = len(pop[pop>=thresholds[1]]) - len(pop[pop>=thresholds[2]])
    count[3] = len(pop[pop>=thresholds[2]]) - len(pop[pop>=thresholds[3]])
    count[4] = len(pop[pop>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count



def seq_acc(true, pred):

    true_num = np.sum((true==pred))
    total_num = true.shape[0] * true.shape[1]

    return {'acc': true_num / total_num}


def load_pretrained_model(pretrain_dir, model, logger, device):

    logger.info("Loading pretrained model ... ")
    checkpoint_path = os.path.join(pretrain_dir, 'pytorch_model.bin')

    model_dict = model.state_dict()

    # To be compatible with the new and old version of model saver
    try:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    except:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)

    # filter out required parameters
    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    # Print how many parameters are updated
    logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
    model.load_state_dict(model_dict)

    return model


def record_csv(args, res_dict, path='log'):
    
    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    columns = list(res_dict.keys())
    columns.insert(0, "model_name")
    columns.insert(1, "hyper_params")
    columns.insert(2, "hidden_mode")
    res_dict["model_name"] = model_name
    
    # Build hyper_params based on model type
    base_params = f"lr={args.lr} dropout_rate={args.dropout_rate} seed={args.seed} hidden_size={args.hidden_size}"
    
    if args.model_name.startswith('icsrec_hsu_'):
        # ICSRec + HSU enhanced version parameters
        intent_num = getattr(args, 'intent_num', 512)
        ics_temperature = getattr(args, 'ics_temperature', 1.0)
        ics_lambda = getattr(args, 'ics_lambda', 0.1)
        ics_beta = getattr(args, 'ics_beta', 0.1)
        ics_rec_weight = getattr(args, 'ics_rec_weight', 1.0)
        # Prioritize reading hsu_fusion_type, fallback to hsu_gate_type if not available
        hsu_fusion_type = getattr(args, 'hsu_fusion_type', None) or getattr(args, 'hsu_gate_type', 'scalar')
        hsu_gate_init = getattr(args, 'hsu_gate_init', 0.0)
        hsu_fusion_dropout = getattr(args, 'hsu_fusion_dropout', 0.1)
        res_dict["hyper_params"] = (
            f"{base_params} intent_num={intent_num} temp={ics_temperature} "
            f"lambda={ics_lambda} beta={ics_beta} rec_w={ics_rec_weight} "
            f"fusion_type={hsu_fusion_type} gate_init={hsu_gate_init} fusion_dropout={hsu_fusion_dropout}"
        )
    elif args.model_name.startswith('icsrec_'):
        # ICSRec intent contrastive learning model parameters
        intent_num = getattr(args, 'intent_num', 512)
        ics_temperature = getattr(args, 'ics_temperature', 1.0)
        ics_lambda = getattr(args, 'ics_lambda', 0.1)
        ics_beta = getattr(args, 'ics_beta', 0.1)
        ics_rec_weight = getattr(args, 'ics_rec_weight', 1.0)
        ics_cl_mode = getattr(args, 'ics_cl_mode', 'cf')
        ics_use_fnm = getattr(args, 'ics_use_fnm', True)
        ics_sim = getattr(args, 'ics_sim', 'dot')
        res_dict["hyper_params"] = (
            f"{base_params} intent_num={intent_num} temp={ics_temperature} "
            f"lambda={ics_lambda} beta={ics_beta} rec_w={ics_rec_weight} "
            f"cl_mode={ics_cl_mode} fnm={ics_use_fnm} sim={ics_sim}"
        )
    elif args.model_name.startswith('llmemb_gaa_'):
        # LLMEmb + GAA (Bank mode) parameters
        alpha = getattr(args, 'alpha', 0.01)  # LLMEmb alignment loss weight
        gaa_alpha = getattr(args, 'gaa_alpha', 0.1)  # GAA loss weight
        sim_user_num = getattr(args, 'sim_user_num', 10)
        sim_long_user_num = getattr(args, 'sim_long_user_num', 10)
        sim_filter_percentile = getattr(args, 'sim_filter_percentile', 0.5)
        gaa_use_user_bank = getattr(args, 'gaa_use_user_bank', False)
        gaa_bank_momentum = getattr(args, 'gaa_bank_momentum', 0.9)
        gaa_warmup_epochs = getattr(args, 'gaa_warmup_epochs', 1)
        filter_similar_metric = getattr(args, 'filter_similar_metric', 'pearson')
        similar_gate = getattr(args, 'similar_gate', -1)
        # Dynamic K parameters
        use_dynamic_k = getattr(args, 'use_dynamic_k', False)
        dynamic_k_w1 = getattr(args, 'dynamic_k_w1', 0.5)
        dynamic_k_w2 = getattr(args, 'dynamic_k_w2', 0.5)
        dynamic_k_min = getattr(args, 'dynamic_k_min', 2)
        dynamic_k_max = getattr(args, 'dynamic_k_max', 18)
        res_dict["hyper_params"] = (
            f"{base_params} alpha={alpha} gaa_alpha={gaa_alpha} sim_users={sim_user_num} "
            f"long_sim_users={sim_long_user_num} sim_filter_pct={sim_filter_percentile} "
            f"use_bank={gaa_use_user_bank} bank_mom={gaa_bank_momentum} warmup={gaa_warmup_epochs} "
            f"filter_metric={filter_similar_metric} similar_gate={similar_gate} "
            f"dynamic_k={use_dynamic_k} dk_w1={dynamic_k_w1} dk_w2={dynamic_k_w2} "
            f"dk_min={dynamic_k_min} dk_max={dynamic_k_max}"
        )
    elif args.model_name.startswith('rcl_hsu_'):
        # RCL + HSU enhanced version parameters
        hsu_gate_type = getattr(args, 'hsu_gate_type', 'scalar')
        hsu_gate_init = getattr(args, 'hsu_gate_init', -10.0)
        hsu_fusion_dropout = getattr(args, 'hsu_fusion_dropout', 0.1)
        rcl_scale = getattr(args, 'rcl_scale', 0.1)
        rcl_ssl = getattr(args, 'rcl_ssl', 8)
        res_dict["hyper_params"] = f"{base_params} rcl_ssl={rcl_ssl} rcl_scale={rcl_scale} gate_type={hsu_gate_type} gate_init={hsu_gate_init} fusion_dropout={hsu_fusion_dropout}"
    elif args.model_name.startswith('rcl_'):
        # RCL base version parameters
        rcl_scale = getattr(args, 'rcl_scale', 0.1)
        rcl_ssl = getattr(args, 'rcl_ssl', 8)
        rcl_perc = getattr(args, 'rcl_perc', 95)
        res_dict["hyper_params"] = f"{base_params} rcl_ssl={rcl_ssl} rcl_scale={rcl_scale} rcl_perc={rcl_perc}"
    else:
        # Original model parameters (LLMEmb, etc.)
        res_dict["hyper_params"] = "lr=" + str(args.lr) + " dropout_rate=" + str(args.dropout_rate) + " gru_layer=" + str(args.num_layers) + " seed=" + str(args.seed) +" alpha=" + str(args.alpha) + " trm_num=" + str(args.trm_num) + " num_heads=" + str(args.num_heads) + " sim_users="+str(args.sim_user_num) + " long_sim_users="+str(args.sim_long_user_num) + " mask_prob=" +str(args.mask_prob) + " filter_similar_metric=" + str(args.filter_similar_metric) + " similar_gate=" + str(args.similar_gate)
    
    res_dict["hidden_mode"] = args.hidden_mode
    # columns = ["model_name", "HR@10", "NDCG@10", "Short HR@10", "Short NDCG@10", "Medium HR@10", "Medium NDCG@10", "Long HR@10", "Long NDCG@10",]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):

        df = pd.DataFrame(new_res_dict)
        df = df[columns]    # reindex the columns
        df.to_csv(csv_path, index=False)

    else:

        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)



def record_group(args, res_dict, path='log'):
    
    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    columns = list(res_dict.keys())
    columns.insert(0, "model_name")
    res_dict["model_name"] = model_name
    # columns = ["model_name", "HR@10", "NDCG@10", "Short HR@10", "Short NDCG@10", "Medium HR@10", "Medium NDCG@10", "Long HR@10", "Long NDCG@10",]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):

        df = pd.DataFrame(new_res_dict)
        df = df[columns]    # reindex the columns
        df.to_csv(csv_path, index=False)

    else:

        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)


# def masked_mean(sim_log_feats):
#     # Check if each row is all zeros (shape: [batch, num_users])
#     is_nonzero = torch.any(sim_log_feats != 0, dim=2)
    
#     # Calculate number of valid rows for each sample (shape: [batch])
#     valid_counts = is_nonzero.sum(dim=1, keepdim=True)
    
#     # Avoid division by zero error
#     valid_counts = torch.clamp(valid_counts, min=1)
    
#     # Create mask (shape: [batch, num_users, 1])
#     mask = is_nonzero.unsqueeze(-1).float()
    
#     # Apply mask and calculate weighted sum
#     masked_sum = (sim_log_feats * mask).sum(dim=1)
    
#     # Calculate dynamic mean
#     dynamic_mean = masked_sum / valid_counts
    
#     return dynamic_mean

def masked_mean(sim_log_feats, valid_mask, eps=1e-8):
    """
    Calculate mean of valid similar users for each batch sample

    sim_log_feats: [batch, sim_num, hidden_size]
    valid_mask:    [batch, sim_num]  (0/1)

    Returns:
        dynamic_mean: [batch, hidden_size], samples without valid similar users are all zeros
        valid_sample_idx: batch indices of valid samples
    """
    mask = valid_mask.unsqueeze(-1).float()        # [batch, sim_num, 1]
    masked_sum = (sim_log_feats * mask).sum(dim=1) # [batch, hidden_size]
    valid_counts = mask.sum(dim=1)                 # [batch, 1]

    # Find valid samples
    valid_sample_idx = (valid_counts.squeeze(-1) > 0).nonzero(as_tuple=True)[0]

    # Avoid division by zero
    valid_counts = valid_counts + eps

    # Calculate mean
    dynamic_mean = masked_sum / valid_counts

    return dynamic_mean, valid_sample_idx


