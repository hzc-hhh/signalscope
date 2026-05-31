"""
Interpretability tools for biomedical sensor signals.

Methods to understand what a model learns from physiological signals:
- Signal attribution (which time segments matter most)
- Medical semantic mapping (which latent dimensions correspond to known physiology)
"""

from signalscope.interpretability.medical_mapping import (
    latent_correlation,
)
from signalscope.interpretability.signal_attribution import (
    attention_weights,
    gradient_attribution,
)

__all__ = [
    "attention_weights",
    "gradient_attribution",
    "latent_correlation",
]
