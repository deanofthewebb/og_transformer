import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F

#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2, dropout=0.3):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model//num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # Create list of layers for K, V, and Q
        self.linear_Qs = nn.ModuleList([nn.Linear(self.d_model, self.d) for _ in range(self.num_heads)])
        # Better approach
        # self.linear_Qs = nn.Linear(self.d_model, self.d_model)
        # [B x seq_len x D] reshapes to [B x seq_len x self.num_heads, self.d]

        self.linear_Ks = nn.ModuleList([nn.Linear(self.d_model, self.d) for _ in range(self.num_heads)])

        self.linear_Vs = nn.ModuleList([nn.Linear(self.d_model, self.d) for _ in range(self.num_heads)])

        self.mha_linear = nn.Linear(self.d_model,self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q.shape, K.shape, V.shape = [batch x seq_len x D//num_heads] = [B x T x d_k]
 
        # Instead of Transpose, we use permute to flip the last 2 dimensions
        Q_K_matmul = Q @ K.permute(0, 2, 1)
        # Q_K_matmul.shape = [B x seq_len x seq_len] 
        scores = Q_K_matmul/m.sqrt(self.d) 
        # scores.shape = [batch x seq_len x seq_len] = [B x T x T]

        if mask is not None:
            # Large negative number: 1e-9 such that any computation is negligible since the model doesn't know how to handle
            scores = scores.masked_fill(mask == False, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights.shape = [B x T x T]

        output = attention_weights @ V
        # output.shape = [B x seq_len x D//num_heads] = [B x T x d_k]

        return output, attention_weights
    
    def forward(self, pre_q, pre_k, pre_v, mask=None):
        # pre_q.shape, pre_k.shape, pre_v.shape = [B x seq_len x D]
        # Making a distinction due to the decoder
        # mask is for the decoder
        
        Q = [linear_Q(pre_q) for linear_Q in self.linear_Qs]
        K = [linear_K(pre_k) for linear_K in self.linear_Ks]
        V = [linear_V(pre_v) for linear_V in self.linear_Vs]
        # Q.shape, K.shape, V.shape = [B x seq_len, D//num_heads] * num_heads

        output_per_head = [] # len == num_heads
        # Store attn_weights in case we want to visualize them
        attn_weights_per_head = []
        # output_per_head.shape = [B x seq_len x D//num_heads] * num_heads
        # attn_weights_per_head.shape = [B x seq_len x seq_len] * num_heads
        for Q_, K_, V_ in zip(Q, K, V):
            # Run scaled_dot_product_attention
            output, attn_weight = self.scaled_dot_product_attention(Q_, K_, V_, mask)
            # output.shape = [B x seq_len x D//num_heads]
            # attn_weight.shape = [B x seq_len x seq_len] we want every word to # pay attention to every other word 
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)
        # Example output_per_head = [
        #   [
        #       [0.00, 0.01],
        #       [0.10, 0.11],
        #       [0.20, 021]  
        #   ], (tensor)
        #   [
        #       [1.00, 1.01],
        #       [1.10, 1.11],
        #       [1.20, 1.21]  
        #   ], (tensor)   
        #   [
        #       [2.00, 2.01],
        #       [2.10, 2.11],
        #       [2.20, 2.21]  
        #   ] (tensor)
        # ]
          
        output = torch.cat(output_per_head, -1)
        # Example output = [
        #   [0.00, 0.01, 1.00, 1.01, 2.00, 2.01],
        #   [0.10, 0.11. 1.10, 1.11, 2.10, 2.11],
        #   [0.20, 0.21, 1.20, 1.21, 2.20, 2.21]
        # ].shape = [3x6]

        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)
        # output.shape = [B x seq_len x D]
        # attn_weights.shape = [B x num_heads x seq_len x seq_len]

        projection = self.dropout(self.mha_linear(output))
        return projection, attn_weights





# %%
#