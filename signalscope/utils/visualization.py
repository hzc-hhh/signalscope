"""Signal and result visualization."""

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_signal(
    signal: np.ndarray,
    sample_rate: float = 1.0,
    title: str = "Signal",
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
    figsize: Tuple[int, int] = (12, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a 1D time-series signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    sample_rate : float
        Hz.
    title, xlabel, ylabel : str
    figsize : tuple
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    t = np.arange(len(signal)) / sample_rate
    ax.plot(t, signal, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    return ax


def plot_spectrogram(
    signal: np.ndarray,
    sample_rate: float = 1.0,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (10, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a spectrogram of a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
    sample_rate : float
    title : str
    figsize : tuple
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    spec, freqs, times, im = ax.specgram(
        signal, Fs=sample_rate, cmap="viridis", NFFT=256, noverlap=128
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=ax, label="Power (dB)")

    return ax


def plot_leaderboard(
    leaderboard: "Leaderboard",
    metric: str = "mae_mean",
    title: str = "Benchmark Leaderboard",
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Axes:
    """
    Bar chart of leaderboard results.

    Parameters
    ----------
    leaderboard : signalscope.benchmark.Leaderboard
    metric : str
        Metric to plot (lower is better).
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib Axes
    """
    results = leaderboard.to_dict()
    if not results:
        _, ax = plt.subplots(figsize=figsize)
        ax.set_title("No benchmark results")
        return ax

    models = [r["model"] for r in results]
    values = [r.get(metric, 0) for r in results]

    _, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(models, values, color="steelblue", edgecolor="white")
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.invert_yaxis()  # Best (lowest) at top

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    return ax
