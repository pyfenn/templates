import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=2, num_classes=4,
                 dropout=0.2, bidirectional=False, pooling="last"):

        super().__init__()
        self.pooling = pooling
        self.bi = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,      # (B, T, D)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, D)
        out, (h_n, c_n) = self.lstm(x)  # out: (B, T, H*dir)

        if self.pooling == "last":
            # last layer hidden, (dir, B, H) -> (B, H*dir)
            if self.bi:
                # h_n shape: (num_layers*2, B, H)
                h_last_fwd = h_n[-2]
                h_last_bwd = h_n[-1]
                feat = torch.cat([h_last_fwd, h_last_bwd], dim=-1)
            else:
                feat = h_n[-1]
        elif self.pooling == "mean":
            feat = out.mean(dim=1)
        elif self.pooling == "max":
            feat = out.max(dim=1).values
        else:
            raise ValueError("pooling must be one of: last, mean, max")

        logits = self.head(feat)
        return logits