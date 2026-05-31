"""
Self-Supervised Pretraining for sensor signals.

SimCLR-style contrastive learning adapted for 1D time-series.
Two augmented views of the same signal should have similar representations.
"""


import torch
import torch.nn as nn

from signalscope.core.registry import register_model
from signalscope.models.base import BaseModel


@register_model("ssl_pretrain")
class SSLPretrain(BaseModel):
    """
    Self-supervised pretraining via contrastive learning.

    Uses a 1D CNN encoder with a projection head.
    After pretraining, the projection head is discarded and the encoder
    is fine-tuned for downstream tasks.

    Parameters
    ----------
    encoder : nn.Module
        Backbone encoder (e.g., ResNet1D).
    proj_dim : int
        Projection head output dimension.
    temperature : float
        NT-Xent loss temperature.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        proj_dim: int = 128,
        temperature: float = 0.07,
        **config,
    ):
        super().__init__(proj_dim=proj_dim, temperature=temperature, **config)
        self.encoder = encoder or self._default_encoder()
        self.projection = nn.Sequential(
            nn.Linear(self._encoder_output_dim(), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        self.temperature = temperature

    def _default_encoder(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def _encoder_output_dim(self) -> int:
        dummy = torch.randn(1, 1, 1000)
        with torch.no_grad():
            return self.encoder(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return projection for contrastive loss."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.encoder(x)
        return self.projection(features)

    def get_encoder(self) -> nn.Module:
        """Return encoder backbone for downstream fine-tuning."""
        return self.encoder
