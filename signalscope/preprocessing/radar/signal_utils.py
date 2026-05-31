"""
Radar signal utilities: clutter removal, phase demodulation, and filtering.

These are standalone functions that can be used independently of the pipeline.
"""

from typing import Literal

import numpy as np
from scipy import signal as scipy_signal


def remove_clutter(
    iq: np.ndarray,
    method: Literal["static", "adaptive"] = "adaptive",
    alpha: float = 0.9,
) -> np.ndarray:
    """
    Remove static/background clutter from IQ signal.

    Parameters
    ----------
    iq : np.ndarray, shape (n_samples,) or (n_samples, n_channels)
        Raw IQ signal.
    method : str
        'static': subtract temporal mean
        'adaptive': recursive moving average with forgetting factor
    alpha : float
        Forgetting factor for adaptive filter (0 < alpha < 1).

    Returns
    -------
    np.ndarray
        Clutter-removed signal.
    """
    if method == "static":
        return iq - np.mean(iq, axis=0, keepdims=True)

    if method == "adaptive":
        cleaned = np.zeros_like(iq, dtype=np.float64)
        if iq.ndim == 1:
            avg = float(iq[0])
            for t in range(len(iq)):
                avg = alpha * avg + (1 - alpha) * float(iq[t])
                cleaned[t] = iq[t] - avg
        else:
            avg = iq[0].astype(np.float64)
            for t in range(len(iq)):
                avg = alpha * avg + (1 - alpha) * iq[t]
                cleaned[t] = iq[t] - avg
        return cleaned

    return iq


def dacm_demodulate(iq_real: np.ndarray, iq_imag: np.ndarray) -> np.ndarray:
    """
    Differentiate and Cross-Multiply (DACM) phase demodulation.

    φ(t) = ∫ [I·Q' - Q·I'] / [I² + Q²] dt

    Parameters
    ----------
    iq_real : np.ndarray
        In-phase component.
    iq_imag : np.ndarray
        Quadrature component.

    Returns
    -------
    np.ndarray
        Demodulated phase signal.
    """
    i = iq_real.astype(np.float64)
    q = iq_imag.astype(np.float64)

    # Numerical differentiation
    di = np.gradient(i)
    dq = np.gradient(q)

    # DACM formula
    numerator = i * dq - q * di
    denominator = i**2 + q**2 + 1e-10  # avoid division by zero

    phase_derivative = numerator / denominator
    phase = np.cumsum(phase_derivative)
    return phase


def extract_vital_signals(
    phase: np.ndarray,
    sample_rate: float,
    resp_band: tuple[float, float] = (0.1, 0.5),
    heart_band: tuple[float, float] = (0.8, 3.0),
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract respiration and heartbeat waveforms from the demodulated phase.

    Parameters
    ----------
    phase : np.ndarray
        Demodulated phase signal.
    sample_rate : float
        Sampling rate in Hz.
    resp_band : tuple
        (low, high) Hz for respiration bandpass.
    heart_band : tuple
        (low, high) Hz for heartbeat bandpass.
    order : int
        Butterworth filter order.

    Returns
    -------
    (respiration, heartbeat) : tuple of np.ndarray
        Extracted physiological waveforms.
    """
    nyquist = 0.5 * sample_rate

    # Respiration filter
    b_resp, a_resp = scipy_signal.butter(
        order,
        [resp_band[0] / nyquist, resp_band[1] / nyquist],
        btype="band",
    )
    respiration = scipy_signal.filtfilt(b_resp, a_resp, phase)

    # Heartbeat filter
    b_heart, a_heart = scipy_signal.butter(
        order,
        [heart_band[0] / nyquist, heart_band[1] / nyquist],
        btype="band",
    )
    heartbeat = scipy_signal.filtfilt(b_heart, a_heart, phase)

    return respiration, heartbeat


def estimate_bpm(
    signal: np.ndarray,
    sample_rate: float,
    freq_range: tuple[float, float],
) -> float | None:
    """
    Estimate dominant frequency in BPM using FFT.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    sample_rate : float
        Sampling rate in Hz.
    freq_range : tuple
        (low, high) frequency range in Hz to search.

    Returns
    -------
    float or None
        Estimated rate in BPM.
    """
    n = len(signal)
    if n < 2:
        return None

    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(mask):
        return None

    peak_idx = np.argmax(fft[mask])
    peak_freq = freqs[mask][peak_idx]

    return float(peak_freq * 60.0)
