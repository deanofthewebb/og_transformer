import torch.nn as nn
from .loop_attention import MultiHeadAttention as LoopMultiHeadAttention
from .attention import MultiHeadAttention 
from .pw_ffn import PositionWiseFFN
from .residual_layer_norm import ResidualLayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3, efficient_mha=True):
        super().__init__()
        self.norm_1 = ResidualLayerNorm(d_model, dropout)
        self.norm_2 = ResidualLayerNorm(d_model, dropout)
        self.norm_3 = ResidualLayerNorm(d_model,dropout)

        if efficient_mha:
            self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout)
            self.crs_mha = MultiHeadAttention(d_model, num_heads, dropout)
        else:
            self.masked_mha = LoopMultiHeadAttention(d_model, num_heads, dropout)
            self.crs_mha = LoopMultiHeadAttention(d_model, num_heads, dropout)

        self.ff = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, encoder_outputs, trg_mask, src_mask):
        # x.shape = [B x TRG_seq_len x D]
        # encoder_outputs.shape = [B x TRG_seq_len x D]

        masked_mha, masked_mha_attn_weights = self.masked_mha(x, x, x, mask=trg_mask)
        # masked_mha.shape = [B x TRG_seq_len x D]
        # masked_mha_attn_weights.shape = [B x num_heads x TRG_seq_len x TRG_seq_len]

        norm_1 = self.norm_1(masked_mha, x)
        # norm_1.shape = [B x TRG_seq_len x D]

        crs_mha, crs_mha_attn_weights = self.crs_mha(norm_1, encoder_outputs, encoder_outputs, mask=src_mask)
        # crs_mha.shape = [B x TRG_seq_len x D]

        norm_2 = self.norm_2(crs_mha, norm_1)
        # norm_2.shape = [Bx TRG_seq_len x D]

        ff = self.ff(norm_2)
        # ff.shape = [B x TRG_seq_len x D]
        # norm_3.shape = [B x TRG_seq_len x D]

        norm_3 = self.norm_3(ff, norm_2)
        # norm_3.shape = [Bx TRG_seq_len x D]

        return norm_3, masked_mha_attn_weights