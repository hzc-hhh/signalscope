"""
Classical signal processing baselines.

Provides model-free baselines (FFT peak detection, bandpower analysis)
that serve as lower-bound comparisons for deep learning models.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

from signalscope.models.base import BaseModel


class ClassicalBaseline(BaseModel):
    """
    Classical signal processing baseline (not a learned model).

    Uses FFT-based peak detection and bandpower analysis.
    Useful as a sanity-check baseline before training deep models.

    Parameters
    ----------
    method : str
        'fft_peak', 'bandpower', or 'wavelet'
    sample_rate : float
        Signal sample rate (Hz).
    freq_range : tuple
        (low, high) Hz for the target physiological signal.
    """

    def __init__(
        self,
        method: str = "fft_peak",
        sample_rate: float = 100.0,
        freq_range: Tuple[float, float] = (0.8, 3.0),
        **config,
    ):
        super().__init__(method=method, sample_rate=sample_rate, **config)
        self.method = method
        self.sample_rate = sample_rate
        self.freq_range = freq_range

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Not used for classical baselines. Use predict() instead."""
        raise NotImplementedError("ClassicalBaseline uses predict(), not forward().")

    def fit(self, *args, **kwargs):
        """No-op: classical methods don't train."""
        return {"train_loss": [], "val_loss": []}

    def predict(self, x: "torch.Tensor", device: Optional[str] = None) -> np.ndarray:
        """
        Estimate dominant frequency in BPM.

        Parameters
        ----------
        x : torch.Tensor, shape (B, L) or (L,)
            Input signal batch.

        Returns
        -------
        np.ndarray
            Estimated BPM per sample.
        """
        if hasattr(x, "numpy"):
            x = x.numpy()
        x = np.asarray(x)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        results = []
        for i in range(len(x)):
            bpm = self._fft_peak_bpm(x[i], self.sample_rate, self.freq_range)
            results.append(bpm if bpm is not None else 0.0)

        return np.array(results)

    @staticmethod
    def _fft_peak_bpm(
        signal: np.ndarray,
        sr: float,
        freq_range: Tuple[float, float],
    ) -> Optional[float]:
        n = len(signal)
        if n < 2:
            return None
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if not np.any(mask):
            return None
        peak_freq = freqs[mask][np.argmax(fft[mask])]
        return float(peak_freq * 60.0)
