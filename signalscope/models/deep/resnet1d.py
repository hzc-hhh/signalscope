"""
1D ResNet for time-series sensor signals.

Stacked residual blocks with 1D convolutions. Suitable for:
- Vital sign estimation from radar phase signals
- Heart rate prediction from PPG
- General time-series regression/classification

Configurable depth, kernel size, and dropout.
"""


import torch
import torch.nn as nn

from signalscope.core.registry import register_model
from signalscope.models.base import BaseModel


class ResidualBlock1D(nn.Module):
    """1D residual block with two conv layers and optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


@register_model("resnet1d")
class ResNet1D(BaseModel):
    """
    1D ResNet for time-series signals.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 1 for single-channel sensor).
    num_classes : int
        Number of output classes (1 for regression, >1 for classification).
    base_channels : int
        Number of channels in the first layer (doubled per block).
    num_blocks : list[int]
        Number of residual blocks per stage (controls depth).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        num_blocks: tuple = (2, 2, 2, 2),
        kernel_size: int = 7,
        dropout: float = 0.1,
        **config,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            num_blocks=num_blocks,
            **config,
        )
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.initial = nn.Sequential(
            nn.Conv1d(
                in_channels, base_channels,
                kernel_size=kernel_size, stride=2,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.res_blocks = nn.ModuleList()
        channels = base_channels
        for i, num_block in enumerate(num_blocks):
            out_channels = base_channels * (2**i)
            for j in range(num_block):
                stride = 2 if j == 0 and i > 0 else 1
                self.res_blocks.append(
                    ResidualBlock1D(channels, out_channels, stride=stride, dropout=dropout)
                )
                channels = out_channels

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, C, L) or (B, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)

        out = self.initial(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.avgpool(out).squeeze(-1)
        out = self.fc(out)
        return out.squeeze(-1) if self.num_classes == 1 else out
