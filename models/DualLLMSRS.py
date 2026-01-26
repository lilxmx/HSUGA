# here put the import lib
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.GRU4Rec import GRU4Rec
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.utils import Multi_CrossAttention



class DualLLMGRU4Rec(GRU4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        self.use_llm2rec = args.use_llm2rec if hasattr(args, 'use_llm2rec') else False

        # Load LLM embedding as item embedding
        if self.use_llm2rec:
            # Use LLM2Rec pre-trained item embeddings
            llm2rec_emb_path = args.llm2rec_emb_path if hasattr(args, 'llm2rec_emb_path') else "LLM2rec/item_info/Qwen2-0.5B-LLM2Rec-IEM_step105_title_item_embs.npy"
            llm_item_emb = np.load(llm2rec_emb_path)
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            # LLM2Rec adapter: lightweight linear projection
            self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        else:
            # Original LLM embedding loading logic
            llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
            self.llm_item_emb.weight.requires_grad = True  # Grad is False by default, unified freezing done later via self._freeze()
            self.adapter = nn.Sequential(
                nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
                nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
            )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs):

        id_seqs = self.id_item_emb(log_seqs)
        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
    


class DualLLMSASRec(SASRec_seq):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        self.use_llm2rec = args.use_llm2rec if hasattr(args, 'use_llm2rec') else False
        

        # self.llm_item_emb is Ese (semantic embedding)
        # Load LLM embedding as item embedding
        if self.use_llm2rec:
            # Use LLM2Rec pre-trained item embeddings
            llm2rec_emb_path = args.llm2rec_emb_path if hasattr(args, 'llm2rec_emb_path') else "LLM2rec/item_info/Qwen2-0.5B-LLM2Rec-IEM_step105_title_item_embs.npy"
            llm_item_emb = np.load(llm2rec_emb_path)
            # Insert a zero vector at the first row (index 0) of item embedding matrix to add a default vector for item ID 0, usually used for padding items
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            # Append a zero vector at the end of embedding matrix to add zero vector for a special item ID (e.g., n+1), which may be a reserved special identifier
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            # LLM2Rec adapter: lightweight linear projection
            self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        else:
            # Original LLM embedding loading logic
            # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
            llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))  # This is LLM emb before PCA
            # Insert a zero vector at the first row (index 0) of item embedding matrix to add a default vector for item ID 0, usually used for padding items
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)

            # Append a zero vector at the end of embedding matrix to add zero vector for a special item ID (e.g., n+1), which may be a reserved special identifier
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)

            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
            self.llm_item_emb.weight.requires_grad = True  # Grad is False by default
            # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
            self.adapter = nn.Sequential(
                nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
                nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
            )

        # self.id_item_emb is Eco (collaborative embedding)

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)  # Usually used to represent special "padding" or "unknown" item ID, position at index 0 corresponds to this all-zero embedding
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)  # May be used to represent another special ID, such as "mask" or "end" token, at index num_items+1
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):
        # id_seqs is the dimensionally reduced vector on the right side, dimension is 64
        id_seqs = self.id_item_emb(log_seqs)  # id_seqs.shape = torch.Size([bs, seq_len, hidden_size])
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)
        # llm is the vector without dimensionality reduction
        llm_seqs = self.llm_item_emb(log_seqs)  # dim = 1536
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            # llm_seqs->q  id_seqs->(k, v)
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)  # cross_id_seqs.shape = [100,200,64]
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
            cross_id_seqs = 1 * cross_id_seqs + 0 * id_seqs
            cross_llm_seqs = 1 * cross_llm_seqs + 0 * llm_seqs
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        # id_log_feats represents Uco, llm_log_feats represents Use
        id_log_feats = self.backbone(cross_id_seqs, log_seqs)  # [100,200,64]
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)
        # TODO: This direct concat seems a bit simple, can we add a linear layer for weighted sum?
        # log_feats.shape = torch.Size([100, 200, 128])
        return log_feats

    def log2only_collab_and_cross_feats(self, log_seqs, positions):
        # id_seqs is the dimensionally reduced vector on the right side, dimension is 64
        id_seqs = self.id_item_emb(log_seqs)  # id_seqs.shape = torch.Size([100, 200, 64])
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)
        # llm is the vector without dimensionality reduction
        llm_seqs = self.llm_item_emb(log_seqs)  # dim = 1536
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            # llm_seqs->q  id_seqs->(k, v)
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)  # cross_id_seqs.shape = [100,200,64]
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
            cross_id_seqs = 1 * cross_id_seqs + 0 * id_seqs
            cross_llm_seqs = 1 * cross_llm_seqs + 0 * llm_seqs
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        # id_log_feats represents Uco, llm_log_feats represents Use
        id_log_feats = self.backbone(cross_id_seqs, log_seqs)  # [100,200,64]

        log_feats = id_log_feats
        # log_feats.shape = torch.Size([100, 200, 128])
        return log_feats
    


class DualLLMBert4Rec(Bert4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att
        self.use_llm2rec = args.use_llm2rec if hasattr(args, 'use_llm2rec') else False

        # load llm embedding as item embedding
        if self.use_llm2rec:
            # Use LLM2Rec pretrained item embeddings
            llm2rec_emb_path = args.llm2rec_emb_path if hasattr(args, 'llm2rec_emb_path') else "LLM2rec/item_info/Qwen2-0.5B-LLM2Rec-IEM_step105_title_item_embs.npy"
            llm_item_emb = np.load(llm2rec_emb_path)
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            self.llm_item_emb.weight.requires_grad = True
            # LLM2Rec adapter: lightweight linear projection
            self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        else:
            # Original LLM embedding loading logic
            # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
            llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
            self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
            self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
            # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
            self.adapter = nn.Sequential(
                nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
                nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
            )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):

        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
