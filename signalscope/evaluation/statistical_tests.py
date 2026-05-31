"""
Statistical tests for evaluating model improvements.

Provides standard tests for comparing two models/methods:
- Paired t-test
- Wilcoxon signed-rank test (non-parametric)
- Bootstrap confidence intervals
"""


import numpy as np
from scipy import stats


def paired_ttest(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Paired t-test comparing errors from two methods.

    Null hypothesis: the two methods have the same mean error.

    Parameters
    ----------
    errors_a, errors_b : np.ndarray
        Per-sample errors (lower is better).
    alpha : float
        Significance level.

    Returns
    -------
    dict
        {'statistic', 'p_value', 'significant', 'mean_diff'}
    """
    t_stat, p_val = stats.ttest_rel(errors_a, errors_b)
    return {
        "statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < alpha),
        "mean_diff": float(np.mean(errors_a) - np.mean(errors_b)),
    }


def wilcoxon_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Wilcoxon signed-rank test (non-parametric paired test).

    Parameters
    ----------
    errors_a, errors_b : np.ndarray
        Per-sample errors.
    alpha : float
        Significance level.

    Returns
    -------
    dict
    """
    stat, p_val = stats.wilcoxon(errors_a, errors_b)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": bool(p_val < alpha),
        "mean_diff": float(np.mean(errors_a) - np.mean(errors_b)),
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    statistic_fn : callable
        Function that computes the statistic (e.g., np.mean).
    n_bootstrap : int
        Number of bootstrap samples.
    alpha : float
        Confidence level = 1 - alpha.

    Returns
    -------
    dict
        {'statistic', 'lower', 'upper', 'std'}
    """
    rng = np.random.RandomState(random_seed)
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = data[rng.choice(n, n, replace=True)]
        bootstrap_stats[i] = statistic_fn(sample)

    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return {
        "statistic": float(statistic_fn(data)),
        "lower_ci": lower,
        "upper_ci": upper,
        "std": float(np.std(bootstrap_stats)),
    }
