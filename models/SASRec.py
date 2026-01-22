# here put the import lib
import numpy as np
import os
import torch
import torch.nn as nn
import pickle
from models.BaseModel import BaseSeqModel
from models.utils import PointWiseFeedForward



class SASRecBackbone(nn.Module):

    def __init__(self, device, args) -> None:
        
        super().__init__()

        self.dev = device
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)

        for _ in range(args.trm_num):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            # 1. Self-attention layer
            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_size,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            # 2. Feed-forward network layer
            new_fwd_layer = PointWiseFeedForward(args.hidden_size, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    
    def forward(self, seqs, log_seqs):
        """
        log_seqs: Original user interaction item sequence IDs, padded with 0 by default  torch.Size([batch_size, max_len])
        seqs: Sequence after embedding or other models, seq+positions  torch.Size([batch_size, max_len, hidden_size])
        """

        #timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)  # Broadcast in last dim: mask padding positions. Set elements in seqs corresponding to padding (id=0) to 0

        tl = seqs.shape[1]  # Time dim len for enforce causality: 200
        # torch.tril generates a lower triangular matrix with shape [seq_len, seq_len] to enforce causality
        # Causality means: item at position i can only attend to items before position i, cannot see future items
        # torch.tril keeps lower triangular elements and sets others to 0
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))  # torch.Size([200, 200]) upper triangle is True, lower triangle is False
        # attention_mask is now upper triangle True, lower triangle False
        for i in range(len(self.attention_layers)):
            # torch.nn.MultiheadAttention requires input shape [seq_len, batch_size, hidden_size], so transpose here
            seqs = torch.transpose(seqs, 0, 1)  # After transpose, dimension becomes [200, 100, 64]
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs)  # Self-attention: q, k, v are the same
                                                    #   TODO attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        # torch.Size([128, 200, 64])
        return log_feats  # [100, 200, 64]



class SASRec(BaseSeqModel):
    
    def __init__(self, user_num, item_num, device, args):
        
        super(SASRec, self).__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device

        # self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        # TODO: Add a parameter to control whether item_emb is randomly initialized or initialized by LLM
        if args.item_from_llm:
            print("Using LLM embeddings to initialize item embeddings")
            # Use relative path, relative to project root directory
            id_item_emb = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
            id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)  # Usually used to represent special "padding" or "unknown" item ID, position at index 0 corresponds to this all-zero embedding
            id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)  # May be used to represent another special ID, such as "mask" or "end" token, at index num_items+1
            self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
            self.item_emb.weight.requires_grad = True
        else:
            self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size)  # TO IMPROVE TODO: Why +100 here?
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.backbone = SASRecBackbone(device, args)
        # Directly pass model's raw output (without sigmoid) to loss function, sigmoid will be applied internally
        self.loss_func = torch.nn.BCEWithLogitsLoss()

        # self.filter_init_modules = []
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)

        return item_seq_emb


    def log2feats(self, log_seqs, positions):
        '''Get the representation of given sequence (equivalent to encoder)'''
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5  # Scaling factor, similar to 1/sqrt(dk) in attention
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)  # log_feats.shape = torch.Size([128, 200, 64])

        return log_feats


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions): # for training        
        '''Used to calculate pos and neg logits for loss'''
        log_feats = self.log2feats(seq, positions)  # (bs, max_len, hidden_size)   
        # TODO
        log_feats = log_feats[:, -1, :].unsqueeze(1)  # (bs, hidden_size)

        pos_embs = self._get_embedding(pos.unsqueeze(1)) # (bs, 1, hidden_size)
        neg_embs = self._get_embedding(neg) # (bs, neg_num, hidden_size)

        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1) # (bs, 1)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1) # (bs, neg_num)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)    # do not calculate the padding units
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss # loss


    def predict(self,
                seq, 
                item_indices, 
                positions,
                **kwargs): # for inference
        '''Used to predict the score of item_indices given log_seqs  
            seq
            item_indices.shape = [100, 101] candidate item indices
        LLM-SASRec eval goes here'''
        log_feats = self.log2feats(seq, positions)  # user_ids hasn't been used yet, goes to DualLLMSRS/DualLLMGRU4Rec's log2feats()
        # log_feats.shape = torch.Size([100, 200, 128]) log_feats is user representation after LLM-ESR. 100=batch_size, 200=max_len, 128=dimension
        final_feat = log_feats[:, -1, :]  # Only use last QKV classifier, a waste: take last step's hidden layer
        item_embs = self._get_embedding(item_indices)  # (U, I, C) item_indices are candidate items shape = [100, 101]
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits  # preds # (U, I) dimension here is [100, 101]
    

    def get_user_emb(self,
                     seq,
                     positions,
                     **kwargs):
        log_feats = self.log2feats(seq, positions) # user_ids hasn't been used yet
        # log_feats = self.log2only_collab_and_cross_feats(seq, positions)
        # log_feats.shape = torch.Size([100, 200, 128])
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        
        return final_feat



class SASRec_seq(SASRec):
    # sasrec backbone

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        '''Apply the seq-to-seq loss
        seq is user interaction sequence torch.Size([128, 200])
        pos is positive samples torch.Size([128, 200])
        neg is negative samples torch.Size([128, 200])
        positions is position vectors torch.Size([128, 200])
        '''
        log_feats = self.log2feats(seq, positions)  # torch.Size([128, 200, 128])
        # TODO: log_feats can add LLM user embedding
        pos_embs = self._get_embedding(pos)  # (bs, max_seq_len, hidden_size)
        neg_embs = self._get_embedding(neg)  # (bs, max_seq_len, hidden_size)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (batch_size, max_seq_len)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # (batch_size, max_seq_len)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)  # Do not calculate the padding units
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        # Contrastive learning
        loss = pos_loss + neg_loss

        return loss




