import torch.nn as nn

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        # x.shape = [B x seq_len aka T x D]

        ff = self.ff(x)
        # ff.shape = [B x seq_len x D]

        return ff