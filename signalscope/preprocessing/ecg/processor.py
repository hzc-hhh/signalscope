"""
ECG signal preprocessing: filtering, QRS detection, R-R interval extraction (placeholder).
"""

from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from signalscope.core.pipeline import Pipeline, PipelineResult


class ECGProcessor(Pipeline):
    """ECG preprocessing pipeline (basic)."""

    def __init__(
        self,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        sample_rate: float = 256.0,
        **config,
    ):
        super().__init__(
            lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, **config
        )

    def __call__(self, ecg_signal: np.ndarray, sample_rate: Optional[float] = None) -> PipelineResult:
        sr = sample_rate or self.config["sample_rate"]
        try:
            filtered = self._bandpass_filter(ecg_signal, sr)
            return PipelineResult(data={"filtered": filtered, "peaks": None})
        except Exception as e:
            return PipelineResult(data=None, success=False, error=str(e))

    def _bandpass_filter(self, sig: np.ndarray, sr: float, order: int = 4) -> np.ndarray:
        nyq = 0.5 * sr
        b, a = scipy_signal.butter(
            order,
            [self.config["lowcut"] / nyq, self.config["highcut"] / nyq],
            btype="band",
        )
        return scipy_signal.filtfilt(b, a, sig)
