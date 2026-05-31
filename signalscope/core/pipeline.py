"""
Core pipeline abstraction for SignalScope.

All processing modules (preprocessing, models, evaluation, benchmark)
inherit from this base class to ensure a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Structured result from any SignalScope pipeline step."""

    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            shape = getattr(self.data, "shape", "scalar")
            return f"PipelineResult(success=True, shape={shape})"
        return f"PipelineResult(success=False, error={self.error})"


class Pipeline(ABC):
    """Base pipeline class with shared configuration and logging."""

    def __init__(self, name: Optional[str] = None, **config: Any):
        self.name = name or self.__class__.__name__
        self.config = config
        logger.info(f"Initialized {self.name} with config: {config}")

    @abstractmethod
    def __call__(self, data: Any, **kwargs: Any) -> PipelineResult:
        """Execute the pipeline on the given data."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"
