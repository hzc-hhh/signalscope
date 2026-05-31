"""
Radar signal preprocessing pipeline.

Converts raw radar IQ streams into physiological vital sign waveforms
(breathing, heartbeat) through a standardized pipeline:

    IQ Signal → Clutter Removal → Phase Demodulation → Phase Unwrapping
    → Bandpass Filtering → Vital Sign Waveforms

Reference methods:
- DACM: Differentiate and Cross-Multiply (Wang et al., 2018)
- Adaptive Clutter: Recursive average with forgetting factor
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import signal as scipy_signal

from signalscope.core.pipeline import Pipeline, PipelineResult


@dataclass
class VitalSigns:
    """Extracted vital sign waveforms."""

    respiration: np.ndarray
    heartbeat: np.ndarray
    sample_rate: float
    respiration_rate_bpm: float | None = None
    heart_rate_bpm: float | None = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"VitalSigns(sample_rate={self.sample_rate}, "
            f"respiration_shape={self.respiration.shape}, "
            f"heartbeat_shape={self.heartbeat.shape})"
        )


class IQPipeline(Pipeline):
    """
    Full radar IQ-to-vital-signs pipeline.

    Parameters
    ----------
    clutter_filter : str
        Clutter removal method: 'none', 'static', 'adaptive', 'phase_linear'
    phase_method : str
        Phase demodulation method: 'dacm', 'arctan', 'i_q_ratio'
    bandpass : tuple
        (low_freq, high_freq) in Hz for filtering
    sample_rate : int
        ADC sample rate (Hz)
    """

    def __init__(
        self,
        clutter_filter: Literal["none", "static", "adaptive"] = "adaptive",
        phase_method: Literal["dacm", "arctan"] = "dacm",
        bandpass: tuple[float, float] = (0.1, 3.0),
        sample_rate: int = 1000,
        **config,
    ):
        super().__init__(
            clutter_filter=clutter_filter,
            phase_method=phase_method,
            bandpass=bandpass,
            sample_rate=sample_rate,
            **config,
        )

    def __call__(
        self,
        iq_data: np.ndarray,
        sample_rate: int | None = None,
        **kwargs,
    ) -> PipelineResult:
        """
        Process raw IQ data into vital signs.

        Parameters
        ----------
        iq_data : np.ndarray
            Raw IQ signal. Shape: (n_samples, n_chirps) or (n_samples,)
        sample_rate : int, optional
            Override the sample rate set at initialization.

        Returns
        -------
        PipelineResult
            Result with .data as VitalSigns instance.
        """
        sr = sample_rate or self.config.get("sample_rate", 1000)

        try:
            # Step 1: Clutter removal (static background subtraction)
            cleaned = self._remove_clutter(iq_data)

            # Step 2: Phase demodulation
            phase = self._demodulate_phase(cleaned)

            # Step 3: Phase unwrapping
            unwrapped = self._unwrap_phase(phase)

            # Step 4: Bandpass filtering to extract vital sign bands
            resp = self._bandpass_filter(
                unwrapped, 0.1, 0.5, sr
            )  # Respiration: 0.1-0.5 Hz
            heart = self._bandpass_filter(
                unwrapped, 0.8, 3.0, sr
            )  # Heartbeat: 0.8-3.0 Hz

            # Step 5: Rate estimation via FFT peak detection
            resp_rate = self._estimate_rate(resp, sr, band=(0.1, 0.5))
            heart_rate = self._estimate_rate(heart, sr, band=(0.8, 3.0))

            vital_signs = VitalSigns(
                respiration=resp,
                heartbeat=heart,
                sample_rate=float(sr),
                respiration_rate_bpm=resp_rate,
                heart_rate_bpm=heart_rate,
                metadata={
                    "clutter_filter": self.config["clutter_filter"],
                    "phase_method": self.config["phase_method"],
                    "signal_length": len(iq_data),
                },
            )

            return PipelineResult(data=vital_signs)

        except Exception as e:
            return PipelineResult(data=None, success=False, error=str(e))

    # ------------------------------------------------------------------
    # Internal processing methods
    # ------------------------------------------------------------------

    def _remove_clutter(self, iq: np.ndarray) -> np.ndarray:
        """Remove static/background clutter from IQ signal."""
        method = self.config["clutter_filter"]

        if method == "none":
            return iq

        if method == "static":
            # Subtract the temporal mean
            return iq - np.mean(iq, axis=0, keepdims=True)

        if method == "adaptive":
            # Recursive moving average with forgetting factor alpha
            alpha = self.config.get("clutter_alpha", 0.9)
            cleaned = np.zeros_like(iq, dtype=np.float64)
            if iq.ndim == 1:
                avg = iq[0].astype(np.float64)
                for t in range(len(iq)):
                    avg = alpha * avg + (1 - alpha) * iq[t]
                    cleaned[t] = iq[t] - avg
            else:
                avg = iq[0].astype(np.float64)
                for t in range(len(iq)):
                    avg = alpha * avg + (1 - alpha) * iq[t]
                    cleaned[t] = iq[t] - avg
            return cleaned

        return iq

    def _demodulate_phase(self, iq: np.ndarray) -> np.ndarray:
        """
        Demodulate the phase from IQ data.

        DACM (Differentiate and Cross-Multiply):
            φ(t) = ∫ [I(t) * Q'(t) - Q(t) * I'(t)] / [I(t)² + Q(t)²] dt
        """
        method = self.config["phase_method"]

        if method == "arctan":
            # Simple arctan demodulation for complex IQ
            if np.iscomplexobj(iq):
                return np.angle(iq)
            return np.arctan2(iq, np.zeros_like(iq))

        if method == "dacm":
            # For real-valued data (single-channel after processing)
            if iq.ndim > 1:
                signal_1d = iq[:, 0] if iq.shape[1] > 1 else iq.ravel()
            else:
                signal_1d = iq

            # Approximate DACM: compute phase via Hilbert transform + differentiation
            analytic = scipy_signal.hilbert(signal_1d.astype(np.float64))
            phase = np.angle(analytic)
            return phase

        # Fallback
        return iq

    def _unwrap_phase(self, phase: np.ndarray) -> np.ndarray:
        """Unwrap the phase to remove 2π discontinuities."""
        return np.unwrap(phase)

    def _bandpass_filter(
        self,
        signal: np.ndarray,
        low: float,
        high: float,
        sr: float,
        order: int = 4,
    ) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        nyquist = 0.5 * sr
        low_norm = low / nyquist
        high_norm = high / nyquist
        b, a = scipy_signal.butter(order, [low_norm, high_norm], btype="band")
        return scipy_signal.filtfilt(b, a, signal)

    def _estimate_rate(
        self,
        signal: np.ndarray,
        sr: float,
        band: tuple[float, float],
    ) -> float | None:
        """Estimate dominant frequency (converted to BPM) via FFT."""
        n = len(signal)
        if n < 2:
            return None

        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        # Restrict to the expected frequency band
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(mask):
            return None

        peak_idx = np.argmax(fft[mask])
        peak_freq = freqs[mask][peak_idx]

        # Convert Hz to BPM (for physiological rates)
        return float(peak_freq * 60.0)
