import torch.nn as nn
from .residual_layer_norm import ResidualLayerNorm
from .loop_attention import MultiHeadAttention as LoopMultiHeadAttention
from .attention import MultiHeadAttention 
from .pw_ffn import PositionWiseFFN

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3, efficient_mha=True):
        super().__init__()

        self.norm_1 = ResidualLayerNorm(d_model, dropout)
        self.norm_2 = ResidualLayerNorm(d_model, dropout)
        # Each norm will learn different lambdas and betas during # training
        
        if efficient_mha:
            self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        else:
            self.mha = LoopMultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, mask):
        # x.shape = [B x seq_len x D]

        mha, encoder_attention_weights = self.mha(x, x, x, mask=mask)
        # mha.shape = [B x seq_len x D]
        # encoder_attention_weights.shape = [B x seq_len, seq_len]

        norm_1 = self.norm_1(mha, x)
        # norm_1.shape = [B x seq_len x D]
        
        ff = self.ff(norm_1)
        norm_2 = self.norm_2(ff, norm_1)
        # ff.shape = [B x seq_len x D]
        # norm2.shape = [B x seq_len x D]

        return norm_2, encoder_attention_weights

