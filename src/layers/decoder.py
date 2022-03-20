import torch
import torch.nn as nn
from .decoder_layer import DecoderLayer
from .positional_encoding import PositionalEncoding
from .embed import Embeddings

# Full Decoder stack
class DecoderStack(nn.Module):
    def __init__(self, Embedding: Embeddings, d_model, 
                num_heads, num_layers, 
                d_ff, device="cpu", dropout=0.3):
        super().__init__()

        self.Embedding = Embedding

        self.PE = PositionalEncoding(d_model, device=device)

        self.decoders = nn.ModuleList([DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        ) for layer in range(num_layers)])

    def forward(self, x, encoder_output, trg_mask, src_mask):
        # x.shape = [B x SRC_seq_len x D]

        embeddings = self.Embedding(x)
        decoding = self.PE(embeddings)
        # embeddings.shape = [B x SRC_seq_len x D]
        # decodings.shape = [B x SRC_seq_len x D]

        for decoder in self.decoders:
            decoding, masked_decoder_attention_weights = decoder(decoding, encoder_output, trg_mask, src_mask)
            # decoding.shape = [B x SRC_seq_len x D]
            # decoder_attention_weights.shape = [B x num_heads x TRG_seq_len x TRG_seq_len]

        return decoding, masked_decoder_attention_weights

    