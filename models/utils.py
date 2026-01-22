# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()
        """
        1D convolution is equivalent to fully connected layer per time step
        MLP:
        Each layer of MLP applies a linear transformation (fully connected layer) on feature dimension (hidden_size).
        It processes features independently for each time step in the sequence, so it doesn't directly utilize order information between time steps.
        Conv1D:
        Conv1D also applies linear transformation on feature dimension (hidden_size), but introduces more flexibility.
        When kernel_size=1, Conv1D degenerates to independent linear transformation per time step (similar to MLP behavior).
        Conv1D naturally supports batch operations, and its implementation is usually more efficient than MLP (Conv1D has better hardware optimization support).
        """
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """
        Feed-forward network introduces non-linear transformation, processes features independently for each time step (point-wise operation)
        Input dimension is (batch_size, seq_len, hidden_units), but Conv1d requires input shape (batch_size, hidden_units, seq_len)
        """
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(
                            inputs.transpose(-1, -2)
                        )
                    )
                )
            )
        )
        outputs = outputs.transpose(-1, -2)  # As Conv1D requires (N, C, Length)
        # Residual connection
        outputs += inputs
        return outputs
    


class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau  # Used to adjust similarity scale. Smaller temperature makes model learn sharper similarities, while larger temperature makes similarity distribution smoother


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature  # Calculate dot product, result is scaled by temperature self.temperature. This is equivalent to smoothing when calculating similarity
        # logits is a [batch_size, batch_size] matrix representing pairwise similarities
        # Below calculates internal similarities of each vector set
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        # Target similarity is average of two similarity matrices, converted to probability representation
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    


class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention



class Multi_CrossAttention(nn.Module):
    """
    forward时, first parameter is used for query, second parameter is used for key and value
    """
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self, x, y, log_seqs):
        """
        cross-attention: x,y are hidden layers of two models, x is used as input for q, y is used as input for k and v
        """

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output



