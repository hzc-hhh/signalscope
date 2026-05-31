"""
Biomedical sensor signal metrics.

Standard and domain-specific evaluation metrics.
"""


import numpy as np
from scipy import stats


def heart_rate_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Heart rate estimation error metrics.

    Returns MAE (BPM), RMSE (BPM), and percentage within ±5 BPM.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted heart rates in BPM.

    Returns
    -------
    dict
        {'mae_bpm': ..., 'rmse_bpm': ..., 'within_5bpm_pct': ...}
    """
    error = y_pred - y_true
    return {
        "mae_bpm": float(np.mean(np.abs(error))),
        "rmse_bpm": float(np.sqrt(np.mean(error**2))),
        "within_5bpm_pct": float(np.mean(np.abs(error) <= 5) * 100),
        "std_bpm": float(np.std(error)),
    }


def respiration_rate_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Respiration rate estimation error metrics.

    Returns MAE (breaths/min), RMSE, and percentage within ±2 breaths/min.
    """
    error = y_pred - y_true
    return {
        "mae_brpm": float(np.mean(np.abs(error))),
        "rmse_brpm": float(np.sqrt(np.mean(error**2))),
        "within_2brpm_pct": float(np.mean(np.abs(error) <= 2) * 100),
        "std_brpm": float(np.std(error)),
    }


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Bland-Altman limits of agreement.

    Returns bias, lower LOA, upper LOA (95%), and standard deviation.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted values.

    Returns
    -------
    dict
        {'bias': ..., 'lower_loa': ..., 'upper_loa': ..., 'std': ...}
    """
    diff = y_pred - y_true
    bias = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    return {
        "bias": bias,
        "lower_loa": bias - 1.96 * std_diff,
        "upper_loa": bias + 1.96 * std_diff,
        "std": std_diff,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
) -> dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    task : str
        'regression', 'heart_rate', 'respiration_rate'

    Returns
    -------
    dict
    """
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    r2 = 1 - np.sum(error**2) / max(
        1e-10, np.sum((y_true - np.mean(y_true)) ** 2)
    )

    base_metrics = {"mae": mae, "rmse": rmse, "r2": float(r2)}

    if task == "heart_rate":
        return {**base_metrics, **heart_rate_error(y_true, y_pred)}
    elif task == "respiration_rate":
        return {**base_metrics, **respiration_rate_error(y_true, y_pred)}

    return base_metrics
