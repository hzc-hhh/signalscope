"""
Tests for evaluation metrics and statistical tests.
"""

import numpy as np

from signalscope.evaluation.metrics import (
    bland_altman,
    compute_metrics,
    heart_rate_error,
)
from signalscope.evaluation.statistical_tests import (
    bootstrap_confidence_interval,
    paired_ttest,
)


class TestMetrics:
    def test_heart_rate_error_perfect(self):
        y = np.array([60, 70, 80], dtype=float)
        err = heart_rate_error(y, y)
        assert err["mae_bpm"] == 0.0
        assert err["within_5bpm_pct"] == 100.0

    def test_heart_rate_error_off_by_3(self):
        truth = np.array([60, 70, 80], dtype=float)
        pred = np.array([63, 73, 83], dtype=float)
        err = heart_rate_error(truth, pred)
        assert abs(err["mae_bpm"] - 3.0) < 0.1
        assert err["within_5bpm_pct"] == 100.0

    def test_bland_altman(self):
        truth = np.array([60, 70, 80], dtype=float)
        pred = truth + 2.0
        ba = bland_altman(truth, pred)
        assert abs(ba["bias"] - 2.0) < 0.01

    def test_compute_metrics_regression(self):
        y_true = np.array([1, 2, 3], dtype=float)
        y_pred = np.array([1.1, 2.1, 3.1], dtype=float)
        m = compute_metrics(y_true, y_pred)
        assert "mae" in m
        assert "rmse" in m
        assert "r2" in m


class TestStatisticalTests:
    def test_paired_ttest_significant(self):
        a = np.random.randn(100)
        b = a + 1.0  # Clearly different
        result = paired_ttest(a, b)
        assert result["significant"]

    def test_bootstrap_ci(self):
        data = np.random.randn(200)
        ci = bootstrap_confidence_interval(data, np.mean, n_bootstrap=1000)
        assert ci["lower_ci"] < ci["statistic"] < ci["upper_ci"]
