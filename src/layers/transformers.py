import torch
import torch.nn as nn
from .embed import Embeddings
from .encoder import EncoderStack
from .decoder import DecoderStack


class Transformer(nn.Module):
    def __init__(self, src_vocab_len, trg_vocab_len, d_model, d_ff,
                num_layers, num_heads, src_pad_idx, trg_pad_idx, dropout=0.3, device='cpu', efficient_mha=True):
        super().__init__()
        print("Transformer Device:\t", device)

        self.num_heads = num_heads
        self.device = device
        self.efficient_mha = efficient_mha
        
        encoder_Embedding = Embeddings(
            src_vocab_len, src_pad_idx, d_model)
        decoder_Embedding = Embeddings(
            trg_vocab_len, trg_pad_idx, d_model)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder_stack = EncoderStack(encoder_Embedding, d_model,
                               num_heads, num_layers, d_ff, self.device, dropout)
        self.decoder_stack = DecoderStack(decoder_Embedding, d_model, 
                               num_heads, num_layers, d_ff, self.device, dropout)
        
        self.linear_layer = nn.Linear(d_model, trg_vocab_len)

        # Initialize parameters with xavier
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def create_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        
        # Consider whether mha returns rank 3 or rank 4 Tensor
        if self.efficient_mha:
            src_mask = src_mask.unsqueeze(2)
        return src_mask
    
    def create_trg_mask(self, trg):
        if self.efficient_mha:
            trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
            # 0_th idx: (1) will be broadcasst over all batch samples
            mask = torch.ones((1, self.num_heads, trg.shape[1], trg.shape[1])).triu(1).to(self.device)
        else:
            trg_mask = (trg != self.trg_pad_idx).unsqueeze(1)
            mask = torch.ones((1, trg.shape[1], trg.shape[1])).triu(1).to(self.device)
        mask = mask == 0
        # Use this instead of mask_fill
        trg_mask = trg_mask & mask
        return trg_mask
    
    def forward(self, src, trg):
        # src.shape = [B x SRC_seq_len]
        # trg.shape = [B x TRG_seq_len]
        
        src_mask = self.create_src_mask(src)
        trg_mask = self.create_trg_mask(trg)
        # src_mask.shape = [B x 1 x SRC_seq_len]
        # trg_mask.shape = [B x 1 x TRG_seq_len]

        encoder_outputs, encoder_mha_attn_weights = self.encoder_stack(src, src_mask)
        # encoder_outputs.shape  = [B x SRC_seq_len x D]
        # encoder_mha_attn_weights.shape  = [B x num_heads x SRC_seq_len x SRC_seq_len]

        decoder_outputs, _ = self.decoder_stack(trg, encoder_outputs, trg_mask, src_mask)
        # decoder_outputs.shape  = [B x SRC_seq_len x D]
        # crs_mha_attn_weights.shape  = [B x num_heads x TRG_seq_len x SRC_seq_len]

        logits = self.linear_layer(decoder_outputs)
        # logits.shape  = [B x TRG_seq_len x TRG_vocab_size]

        return logits

