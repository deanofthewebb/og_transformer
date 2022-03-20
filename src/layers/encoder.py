import torch
import torch.nn as nn
from .encoder_layer import EncoderLayer
from .positional_encoding import PositionalEncoding
from .embed import Embeddings

# Full Encoder stack
class EncoderStack(nn.Module):
    def __init__(self, Embedding: Embeddings, d_model, 
                num_heads, num_layers, 
                d_ff, device="cpu", dropout=0.3):
        super().__init__()

        self.Embedding = Embedding

        self.PE = PositionalEncoding(d_model, dropout, device=device)

        self.encoders = nn.ModuleList([EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        ) for layer_module in range(num_layers)])

    def forward(self, x, mask=None):
        # x.shape = [B x SRC_seq_len x D]

        embeddings = self.Embedding(x)
        encoding = self.PE(embeddings)
        # embeddings.shape = [B x SRC_seq_len x D]
        # encodings.shape = [B x SRC_seq_len x D]

        for encoder in self.encoders:
            encoding, encoder_attention_weights = encoder(encoding, mask)
            # encoding.shape = [B x SRC_seq_len x D]
            # encoder_attention_weights.shape = [B x num_heads x SRC_seq_len x SRC_seq_len]

        return encoding, encoder_attention_weights

    