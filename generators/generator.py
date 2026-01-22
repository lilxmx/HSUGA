# here put the import lib
import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from generators.data import SeqDataset, SeqDatasetAllUser, Seq2SeqDatasetAllUser, RCLSeq2SeqDataset, RCLSeqDataset
from utils.utils import unzip_data, concat_data


class Generator(object):

    def __init__(self, args, logger, device):

        self.args = args
        self.aug_file = args.aug_file
        self.inter_file = args.inter_file
        self.dataset = args.dataset
        self.num_workers = args.num_workers
        self.bs = args.train_batch_size
        self.logger = logger
        self.device = device
        self.aug_seq = args.aug_seq

        self.logger.info("Loading dataset ... ")
        start = time.time()
        self._load_dataset()
        end = time.time()
        self.logger.info("Dataset is loaded: consume %.3f s" % (end - start))

    
    def _load_dataset(self):
        '''Load train, validation, test dataset'''

        usernum = 0
        itemnum = 0
        User = defaultdict(list)  # Default value is a blank list
        user_train = {}
        user_valid = {}
        user_test = {}
        # Assume user/item index starting from 1
        # Use relative path, relative to project root directory
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', self.dataset, 'handled', f'{self.inter_file}.txt')
        f = open(data_path, 'r')
        for line in f:  # use a dict to save all seqeuces of each user
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
        
        self.user_num = usernum
        self.item_num = itemnum

        for user in tqdm(User):
            # default args.aug_seq_len = 0
            nfeedback = len(User[user]) - self.args.aug_seq_len
            #nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        
        self.train = user_train
        self.valid = user_valid
        self.test = user_test


    
    def make_trainloader(self):
        # self.train is a dictionary, key is user_id, value is user interaction item_id_list. For users with history sequence length < 3, all sequences are included. Otherwise, last 2 items are not included
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        # train_dataset is a list. Each element is a user interaction item list
        # train_neg defaults to 1
        self.train_dataset = SeqDataset(train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
    

        return train_dataloader


    def make_evalloader(self, test=False):
        """Generate dataloader for validation and test, test=True for test loader
        TODO: Why do validation and test datasets include training part?
        """
        if test:
            eval_dataset = concat_data([self.train, self.valid, self.test])

        else:
            eval_dataset = concat_data([self.train, self.valid])
        # max_len = 200 test_neg = 100
        self.eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg)
        eval_dataloader = DataLoader(self.eval_dataset,
                                    sampler=SequentialSampler(self.eval_dataset),  # Select samples sequentially according to dataset order
                                    batch_size=100,
                                    num_workers=self.num_workers)
        
        return eval_dataloader

    
    def get_user_item_num(self):

        return self.user_num, self.item_num
    

    def get_item_pop(self):
        """get item popularity according to item index. return a np-array"""
        all_data = concat_data([self.train, self.valid, self.test])
        pop = np.zeros(self.item_num+1) # item index starts from 0
        
        for items in all_data:
            # items here is a list [1,2,3,4] corresponding to a user's interaction records
            pop[items] += 1 

        return pop
    

    def get_user_len(self):
        """get sequence length according to user index. return a np-array"""
        all_data = concat_data([self.train, self.valid])
        lens = []

        for user in all_data:
            lens.append(len(user))

        return np.array(lens)
    
    def get_all_user_seqs(self):
        """Get all users' training sequences for RCL pre-computation of user similarity"""
        return unzip_data(self.train, aug=False, aug_num=0)



class GeneratorAllUser(Generator):
    # gru

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):

        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        self.train_dataset = SeqDatasetAllUser(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader

    

class Seq2SeqGeneratorAllUser(Generator):
    # sas
    
    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):
        # self.train already excludes the last two items
        # self.train is a dict, key is user_id, value is item_id list. self.args.aug_seq_len = 0, self.args.aug defaults to False
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        # train_dataset is a list, each element is also a list containing item_ids the user has interacted with
        # max_len = 200
        # train_neg = 1, the number of negative samples for training
        self.train_dataset = Seq2SeqDatasetAllUser(self.args, train_dataset, self.item_num, self.args.max_len, self.args.train_neg)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader


# ============================================================================
# RCL-specific Generator: Returns global user_id for RCL SSL loss calculation
# ============================================================================

class RCLSeq2SeqGenerator(Generator):
    """
    RCL-specific Seq2Seq Generator (for SASRec, BERT4Rec, etc.)
    Key difference: Training set returns global user_id
    """
    
    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)
    
    def make_trainloader(self):
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        # Use RCL-specific Dataset, will return user_id
        self.train_dataset = RCLSeq2SeqDataset(self.args, train_dataset, self.item_num, 
                                                self.args.max_len, self.args.train_neg)
        
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader


class RCLSeqGenerator(Generator):
    """
    RCL-specific Seq Generator (for GRU4Rec, etc.)
    Key difference: Training set returns global user_id
    """
    
    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)
    
    def make_trainloader(self):
        train_dataset = unzip_data(self.train, aug=self.args.aug, aug_num=self.args.aug_seq_len)
        # Use RCL-specific Dataset, will return user_id
        self.train_dataset = RCLSeqDataset(train_dataset, self.item_num, 
                                           self.args.max_len, self.args.train_neg)
        
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader

    