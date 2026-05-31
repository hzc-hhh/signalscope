"""
Evaluation metrics for biomedical sensor signal tasks.

Domain-aware metrics beyond standard regression/classification:
- Heart rate error (MAE in BPM)
- Respiration rate error
- Pearson correlation
- Bland-Altman limits of agreement
- Statistical significance tests
"""

from signalscope.evaluation.cross_validation import (
    SubjectWiseSplit,
    TimeSeriesSplit,
)
from signalscope.evaluation.metrics import (
    bland_altman,
    compute_metrics,
    heart_rate_error,
    pearson_correlation,
    respiration_rate_error,
)
from signalscope.evaluation.statistical_tests import (
    bootstrap_confidence_interval,
    paired_ttest,
    wilcoxon_test,
)

__all__ = [
    "compute_metrics",
    "heart_rate_error",
    "respiration_rate_error",
    "pearson_correlation",
    "bland_altman",
    "paired_ttest",
    "wilcoxon_test",
    "bootstrap_confidence_interval",
    "TimeSeriesSplit",
    "SubjectWiseSplit",
]
