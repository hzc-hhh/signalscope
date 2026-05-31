"""
SignalScope — Unified AI Research Framework for Biomedical Sensor Signals
"""

__version__ = "0.1.0"
__author__ = "SignalScope Contributors"

from signalscope.core.pipeline import Pipeline
from signalscope.core.registry import Registry

__all__ = ["Pipeline", "Registry", "__version__"]
