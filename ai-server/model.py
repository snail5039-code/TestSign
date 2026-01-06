import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnBiGRU(nn.Module):
    def __init__(self, input_dim=822, hidden=256, layers=2, classes=30, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, x):
        h, _ = self.gru(x)                 # (B,T,2H)
        w = self.attn(h).squeeze(-1)       # (B,T)
        a = F.softmax(w, dim=1).unsqueeze(-1)
        pooled = (h * a).sum(dim=1)        # (B,2H)
        return self.head(pooled)
