"""
SignalScope Model Zoo — from classical DSP to deep learning.

Models:
- CNN1D, ResNet1D: Convolutional feature extractors for time-series
- TransformerTS: Time-Series Transformer with positional encoding
- SSLPretrain: Self-supervised pretraining (SimCLR-style) for sensor signals
- MultiModalFusion: Cross-attention fusion for multi-modal sensor data

All models inherit from BaseModel and are registered in MODEL_REGISTRY.
"""

from signalscope.models.base import BaseModel
from signalscope.models.classical.signal_process import ClassicalBaseline
from signalscope.models.deep.multi_modal_fusion import MultiModalFusion
from signalscope.models.deep.resnet1d import ResNet1D
from signalscope.models.deep.ssl_pretrain import SSLPretrain
from signalscope.models.deep.transformer_ts import TransformerTS
from signalscope.models.zoo import ModelZoo

__all__ = [
    "BaseModel",
    "ResNet1D",
    "TransformerTS",
    "SSLPretrain",
    "MultiModalFusion",
    "ClassicalBaseline",
    "ModelZoo",
]
