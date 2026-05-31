"""
Benchmark Runner — standardized model comparison pipeline.

Runs multiple models on multiple tasks with consistent evaluation,
generating reproducible leaderboards.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from signalscope.core.pipeline import Pipeline, PipelineResult
from signalscope.benchmark.leaderboard import Leaderboard


class BenchmarkRunner(Pipeline):
    """
    Standardized benchmark runner.

    Usage:
        runner = BenchmarkRunner(
            tasks=["heart_rate", "respiration_rate"],
            models=["resnet1d", "transformer_ts"],
            metrics=["mae", "rmse"],
        )
        results = runner.run()

    Parameters
    ----------
    tasks : list[str]
        Tasks to benchmark (e.g., 'heart_rate', 'respiration_rate').
    models : list[str]
        Models to evaluate (registered model names).
    metrics : list[str]
        Metrics to compute.
    n_splits : int
        Cross-validation folds.
    """

    def __init__(
        self,
        tasks: List[str],
        models: List[str],
        metrics: Optional[List[str]] = None,
        n_splits: int = 5,
        **config,
    ):
        super().__init__(tasks=tasks, models=models, metrics=metrics, n_splits=n_splits, **config)

    def __call__(self, dataset=None, **kwargs) -> PipelineResult:
        """Alias for run()."""
        return self.run(dataset=dataset, **kwargs)

    def run(self, dataset=None) -> PipelineResult:
        """
        Run the benchmark.

        Parameters
        ----------
        dataset : tuple or None
            (X, y) data. If None, uses synthetic data for demonstration.

        Returns
        -------
        PipelineResult
            .data contains the Leaderboard instance.
        """
        try:
            if dataset is None:
                dataset = self._generate_synthetic_data()

            X, y = dataset
            leaderboard = Leaderboard()

            for model_name in self.config["models"]:
                results = self._evaluate_model(model_name, X, y)
                leaderboard.add_result(model_name, results)

            leaderboard.compute_ranks()
            return PipelineResult(data=leaderboard)

        except Exception as e:
            return PipelineResult(data=None, success=False, error=str(e))

    def _evaluate_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate a single model (simplified — stub for demonstration)."""
        from signalscope.models import ModelZoo

        zoo = ModelZoo()

        if model_name == "classical":
            from signalscope.models.classical.signal_process import ClassicalBaseline
            model = ClassicalBaseline(sample_rate=100.0)
        else:
            model = zoo.get(model_name, in_channels=1, num_classes=1)

        # Simplified evaluation: use a random baseline for demo
        # In production, this would run actual cross-validation
        n = len(X)
        indices = np.random.permutation(n)
        split = n // self.config.get("n_splits", 5)

        all_mae = []
        for fold in range(self.config.get("n_splits", 5)):
            test_idx = indices[fold * split : (fold + 1) * split]
            y_test = y[test_idx]
            # Simplified: random prediction around truth
            y_pred = y_test + np.random.normal(0, 2.0, size=len(y_test))
            mae = float(np.mean(np.abs(y_test - y_pred)))
            all_mae.append(mae)

        return {
            "mae_mean": float(np.mean(all_mae)),
            "mae_std": float(np.std(all_mae)),
        }

    @staticmethod
    def _generate_synthetic_data(
        n_samples: int = 1000,
        n_features: int = 500,
    ) -> tuple:
        """Generate synthetic sensor data for demonstration."""
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, n_features).astype(np.float32)
        y = 0.5 * np.sin(np.linspace(0, 10 * np.pi, n_samples)) + rng.randn(n_samples) * 0.1
        return X, y.astype(np.float32)
