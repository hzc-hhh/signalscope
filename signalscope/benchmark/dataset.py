"""
Benchmark dataset interface.

Provides a consistent API for loading public biomedical sensor datasets.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from signalscope.core.pipeline import Pipeline, PipelineResult


class BenchmarkDataset(Pipeline):
    """
    Standardized interface for benchmark datasets.

    Parameters
    ----------
    name : str
        Dataset name (e.g., 'mimic-iii-matched', 'capnobase', 'radar-vital-signs').
    path : str, optional
        Path to cached/downloaded dataset.
    """

    def __init__(self, name: str, path: Optional[str] = None, **config):
        super().__init__(name=name, path=path, **config)
        self.name = name
        self.path = path

    def __call__(self, **kwargs) -> PipelineResult:
        """Load and return the dataset."""
        try:
            data = self._load(**kwargs)
            return PipelineResult(data=data)
        except Exception as e:
            return PipelineResult(data=None, success=False, error=str(e))

    def _load(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Load dataset. Override for specific datasets.

        Returns
        -------
        dict
            {'train': (X_train, y_train), 'test': (X_test, y_test)}
        """
        raise NotImplementedError(
            f"Dataset '{self.name}' loader not implemented. "
            f"Subclass BenchmarkDataset and implement _load()."
        )

    @property
    def info(self) -> Dict:
        """Dataset metadata."""
        return {
            "name": self.name,
            "path": self.path,
        }
