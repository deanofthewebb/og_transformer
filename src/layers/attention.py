import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2, dropout=0.3):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model//num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.dense_Q = nn.Linear(self.d_model, self.d_model)
        self.dense_K = nn.Linear(self.d_model, self.d_model)
        self.dense_V = nn.Linear(self.d_model, self.d_model)

        self.mha_linear = nn.Linear(self.d_model,self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q.shape, K.shape, V.shape = [batch x seq_len x D//num_heads] = [B x T x d_k]
 
        # Instead of Transpose, we use permute to flip the last two dimensions for matmul
        # reshaped(K) = [B x num_heads x Q_len x KV_len]
        Q_K_matmul = Q @ K.permute(0, 1, 3, 2)
        scores = Q_K_matmul/m.sqrt(self.d) 
        # scores.shape = [B x num_heads x Q_len x KV_len]

        if mask is not None:
            # Large negative number: 1e-9 such that any computation is negligible since the model doesn't know how to handle
            scores = scores.masked_fill(mask == False, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights.shape = [B x num_heads x Q_len x KV_len]

        output = attention_weights @ V
        # output.shape = [B x num_heads x Q_len x D//num_heads]

        return output, attention_weights
    
    def forward(self, pre_q, pre_k, pre_v, mask=None):
        # pre_q.shape, pre_k.shape, pre_v.shape = [B x seq_len x D]
        # Making a distinction due to the decoder
        # mask is for the decoder

        Q = self.dense_Q(pre_q)
        K = self.dense_K(pre_k)
        V = self.dense_V(pre_v)
        # Q.shape = [B x seq_len x D] (if in encoder, seq_len = SRC_seq_len; if in decoder, seq_len = TRG_seq_len)
        # K.shape,V.shape = [B x seq_len x D] (always SRC_seq_len unless in masked-multihead-attention)

        batch_size = pre_q.shape[0]
        
        # Reshape to rank 4 Tensor
        Q = Q.reshape(batch_size, self.num_heads, -1, self.d)
        K = K.reshape(batch_size, self.num_heads, -1, self.d)
        V = V.reshape(batch_size, self.num_heads, -1, self.d)
        # Q.shape, K.shape, V.shape = [B x num_heads x seq_len, D//num_heads]

        # Run scaled_dot_product_attention
        output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        # output.shape = [B x num_heads x Q_len x D//num_heads]
        # attn_weights.shape = [B x num_heads x Q_len x KV_len]
        
        # Reshape back into rank 3 Tensor, collapse num_heads and seq_len together
        output = output.reshape(batch_size, -1, self.d_model)
        # output.shape = [B x seq_len x D]

        projection = self.dropout(self.mha_linear(output))
        return projection, attn_weights