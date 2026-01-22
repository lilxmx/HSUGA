# here put the import lib
from generators.generator import Generator
from generators.data import BertRecTrainDatasetAllUser, RCLBertRecTrainDataset
from torch.utils.data import DataLoader, RandomSampler
from utils.utils import unzip_data



class BertGeneratorAllUser(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):
        
        train_dataset = unzip_data(self.train, aug=self.args.aug)
        self.train_dataset = BertRecTrainDatasetAllUser(self.args, train_dataset, self.item_num, self.args.max_len)

        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
    
        return train_dataloader


class RCLBertGenerator(Generator):
    """
    RCL-specific BERT Generator
    Key difference: Training set returns global user_id
    """
    
    def __init__(self, args, logger, device):
        super().__init__(args, logger, device)
    
    def make_trainloader(self):
        train_dataset = unzip_data(self.train, aug=self.args.aug)
        # Use RCL-specific Dataset
        self.train_dataset = RCLBertRecTrainDataset(self.args, train_dataset, 
                                                     self.item_num, self.args.max_len)
        
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
        
        return train_dataloader
