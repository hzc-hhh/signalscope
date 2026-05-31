"""
Photoplethysmography (PPG) signal preprocessing.

Standard pipeline: filtering → peak detection → physiological feature extraction.
"""

from signalscope.preprocessing.ppg.processor import PPGProcessor

__all__ = ["PPGProcessor"]
