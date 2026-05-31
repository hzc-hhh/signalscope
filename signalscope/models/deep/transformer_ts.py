"""
Time-Series Transformer for biomedical sensor signals.

Uses sinusoidal positional encoding + multi-head self-attention
with a [CLS] token for sequence-level prediction.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from signalscope.models.base import BaseModel
from signalscope.core.registry import register_model


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


@register_model("transformer_ts")
class TransformerTS(BaseModel):
    """
    Time-Series Transformer.

    Parameters
    ----------
    d_model : int
        Model dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Feedforward network dimension.
    dropout : float
        Dropout rate.
    max_len : int
        Maximum sequence length.
    num_classes : int
        Output dimension (1 for regression).
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
        num_classes: int = 1,
        **config,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes,
            **config,
        )
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, L) or (B, 1, L)
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.unsqueeze(-1)  # (B, L, 1)

        x = self.input_proj(x)  # (B, L, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+L, d_model)

        x = self.pos_encoder(x)
        x = self.transformer(x)
        cls_out = x[:, 0, :]  # CLS token output
        out = self.fc(cls_out)
        return out.squeeze(-1) if self.num_classes == 1 else out
