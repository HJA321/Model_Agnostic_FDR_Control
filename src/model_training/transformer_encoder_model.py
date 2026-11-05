import math

import torch
import torch.nn as nn


class TransformerEncoderRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        output_dim,
        dropout,
        nhead=4,
        dim_feedforward=None,
    ):
        super(TransformerEncoderRegressor, self).__init__()

        if nhead <= 0:
            raise ValueError("nhead must be >= 1")
        if hidden_dim % nhead != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})."
            )

        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_p = float(dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def _sinusoidal_pos_encoding(self, seq_len, device, dtype):
        # Returns (1, seq_len, hidden_dim)
        pe = torch.zeros(seq_len, self.hidden_dim, device=device, dtype=dtype)
        position = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x: (batch, time, input_dim)
        bsz, seq_len, _ = x.shape

        h = self.input_proj(x)
        pe = self._sinusoidal_pos_encoding(seq_len, x.device, h.dtype)
        h = self.dropout(h + pe)

        h = self.encoder(h)
        out = self.fc(h[:, -1, :])
        return out
