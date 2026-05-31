"""
SignalScope Preprocessing — multi-modal biomedical sensor signal preprocessing.

Supported modalities:
- radar: IQ-to-vital-signs pipeline (clutter removal, DACM phase demodulation,
  phase unwrapping, bandpass filtering, rate estimation)
- ppg: PPG filtering, peak detection, heart rate estimation
- ecg: ECG filtering, QRS detection (TBD)
- sync: Multi-modal signal alignment
"""

from signalscope.preprocessing.radar import IQPipeline, VitalSigns
from signalscope.preprocessing.ppg import PPGProcessor
from signalscope.preprocessing.ecg import ECGProcessor
from signalscope.preprocessing.sync import MultiModalAligner

__all__ = [
    "IQPipeline",
    "VitalSigns",
    "PPGProcessor",
    "ECGProcessor",
    "MultiModalAligner",
]
