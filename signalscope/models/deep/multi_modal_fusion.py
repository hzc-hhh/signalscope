"""
Multi-modal fusion for sensor signals.

Cross-attention based fusion of signals from different modalities
(e.g., radar + PPG + ECG).
"""


import torch
import torch.nn as nn

from signalscope.core.registry import register_model
from signalscope.models.base import BaseModel


@register_model("multi_modal_fusion")
class MultiModalFusion(BaseModel):
    """
    Cross-attention fusion for multi-modal sensor data.

    Each modality is encoded independently, then cross-attention layers
    fuse representations before a final prediction head.

    Parameters
    ----------
    modality_dims : list[int]
        Input dimension for each modality.
    hidden_dim : int
        Shared hidden dimension after projection.
    num_heads : int
        Number of cross-attention heads.
    num_layers : int
        Number of cross-attention layers.
    num_classes : int
        Output dimension.
    """

    def __init__(
        self,
        modality_dims: list = (1, 1, 1),
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 1,
        **config,
    ):
        super().__init__(
            modality_dims=modality_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            **config,
        )
        self.modality_dims = modality_dims
        self.num_classes = num_classes

        # Per-modality encoders (simple 1D CNN)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, hidden_dim, 7, stride=2, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            for dim in modality_dims
        ])

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * len(modality_dims), hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        *modalities : torch.Tensor
            One tensor per modality, each (B, L) or (B, C, L).

        Returns
        -------
        torch.Tensor
            Prediction.
        """
        encoded = []
        for encoder, mod in zip(self.encoders, modalities):
            if mod.dim() == 2:
                mod = mod.unsqueeze(1)
            encoded.append(encoder(mod))

        # Stack: (B, num_modalities, hidden_dim)
        stacked = torch.stack(encoded, dim=1)

        # Cross-attention
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)

        # Concatenate and fuse
        fused = attn_out.reshape(attn_out.size(0), -1)
        for layer in self.fusion_layers:
            fused = layer(fused)

        out = self.fc(fused)
        return out.squeeze(-1) if self.num_classes == 1 else out
