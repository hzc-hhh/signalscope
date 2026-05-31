"""
Benchmark framework for reproducible model comparison.

Standardized interface for:
- Loading public datasets
- Running models with consistent evaluation
- Generating leaderboard comparisons
"""

from signalscope.benchmark.dataset import BenchmarkDataset
from signalscope.benchmark.leaderboard import Leaderboard
from signalscope.benchmark.runner import BenchmarkRunner

__all__ = ["BenchmarkRunner", "BenchmarkDataset", "Leaderboard"]
