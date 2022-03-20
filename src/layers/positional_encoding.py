import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_seq_len=200, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, self.d_model).float()
        pos = torch.arange(max_seq_len).unsqueeze(1).float()

        two_i = torch.arange(0, d_model, step=2).float()
        div_term = torch.pow(10_000, (two_i/torch.Tensor([d_model]))).float()
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)

        # Account for batch size
        # Allows us to broadcast tensor across entire batch
        pe = pe.unsqueeze(0)

        # Assigns the first argument to a class variable
        # i.e. self.pe
        # Parameter that is not learnable
        pe = pe.to(device)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.shape = [B x seq_len x D]

        # Recall that the embedding is of size 'max_seq_len', we want to # only to the seq_len of input x, which is presumably less than # the max_seq_len. 

        # We detach because we don't want to learn the parameters
        pe = self.pe[:, x.shape[1]]
        x = x.add(pe)
        # x.shape = [B x seq_len x D]
        return self.dropout(x)