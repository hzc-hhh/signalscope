"""
Multi-modal sensor synchronization utilities.

Align signals from different sensor modalities (radar, PPG, ECG)
sampled at different rates or with different time bases.
"""

from signalscope.preprocessing.sync.aligner import MultiModalAligner

__all__ = ["MultiModalAligner"]
