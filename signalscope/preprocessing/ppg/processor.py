"""
PPG signal preprocessing: bandpass filtering, peak detection, feature extraction.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

from signalscope.core.pipeline import Pipeline, PipelineResult


class PPGProcessor(Pipeline):
    """Standard PPG preprocessing pipeline."""

    def __init__(
        self,
        lowcut: float = 0.5,
        highcut: float = 8.0,
        sample_rate: float = 100.0,
        **config,
    ):
        super().__init__(
            lowcut=lowcut, highcut=highcut, sample_rate=sample_rate, **config
        )

    def __call__(self, ppg_signal: np.ndarray, sample_rate: Optional[float] = None) -> PipelineResult:
        sr = sample_rate or self.config["sample_rate"]
        try:
            filtered = self._bandpass_filter(ppg_signal, sr)
            peaks = self._find_peaks(filtered, sr)
            hr = self._estimate_heart_rate(peaks, sr, len(filtered))

            return PipelineResult(
                data={
                    "filtered": filtered,
                    "peaks": peaks,
                    "heart_rate_bpm": hr,
                }
            )
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

    def _find_peaks(self, sig: np.ndarray, sr: float) -> np.ndarray:
        from scipy.signal import find_peaks

        height = np.mean(sig) + 0.3 * np.std(sig)
        distance = int(0.3 * sr)  # min 300 ms between beats
        peaks, _ = find_peaks(sig, height=height, distance=max(1, distance))
        return peaks

    def _estimate_heart_rate(
        self, peaks: np.ndarray, sr: float, n_samples: int
    ) -> Optional[float]:
        if len(peaks) < 2:
            return None
        intervals = np.diff(peaks) / sr
        mean_interval = np.mean(intervals)
        return float(60.0 / mean_interval) if mean_interval > 0 else None
