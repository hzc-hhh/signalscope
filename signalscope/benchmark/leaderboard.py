"""
Leaderboard — ranked comparison of models on benchmark tasks.
"""

from typing import Any, Dict, List

import numpy as np


class Leaderboard:
    """Ranked leaderboard for benchmark results."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add_result(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Add a model's benchmark result."""
        self.results.append({"model": model_name, **metrics})

    def compute_ranks(self, sort_by: str = "mae_mean") -> None:
        """Sort results by a metric (ascending)."""
        self.results.sort(key=lambda r: r.get(sort_by, float("inf")))

    def top(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return top N results."""
        return self.results[:n]

    def to_table(self) -> str:
        """Pretty-print leaderboard as a table."""
        if not self.results:
            return "No results yet."

        keys = list(self.results[0].keys())
        col_widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in self.results)) for k in keys}

        def fmt_row(vals):
            return " | ".join(str(v).ljust(col_widths[k]) for k, v in zip(keys, vals))

        lines = [fmt_row(keys), "-" * (sum(col_widths.values()) + 3 * (len(keys) - 1))]
        for r in self.results:
            vals = [str(r.get(k, "")) for k in keys]
            lines.append(fmt_row(vals))

        return "\n".join(lines)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return results as list of dicts."""
        return self.results

    def __repr__(self) -> str:
        return self.to_table()
