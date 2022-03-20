import torch.nn as nn
import math as m

class Embeddings(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_idx)


    def forward(self, x):
        # x.shape = [B x seq_len]

        # create embedding on x, so it is the size of D
        embedding = self.embed(x)
        # embedding.shape = [B x seq_len x D]

        return embedding * m.sqrt(self.d_model)