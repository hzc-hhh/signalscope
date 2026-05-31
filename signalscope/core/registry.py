"""
Model and preprocessing registry for extensible component management.

The Registry allows users and contributors to register custom preprocessing
steps, model architectures, and evaluation metrics dynamically.
"""

from typing import Any, Callable, Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)


class Registry:
    """Generic registry for extensible components (models, processors, metrics)."""

    def __init__(self, name: str = "registry"):
        self.name = name
        self._entries: Dict[str, Any] = {}
        logger.info(f"Created registry: {name}")

    def register(self, key: str, value: Any, overwrite: bool = False) -> None:
        """Register a component under a unique key."""
        if key in self._entries and not overwrite:
            raise KeyError(
                f"'{key}' already registered in '{self.name}'. "
                f"Use overwrite=True to replace."
            )
        self._entries[key] = value
        logger.debug(f"Registered '{key}' in '{self.name}'")

    def get(self, key: str) -> Any:
        """Retrieve a registered component by key."""
        if key not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise KeyError(
                f"'{key}' not found in '{self.name}'. "
                f"Available: {available}"
            )
        return self._entries[key]

    def list(self) -> Dict[str, Any]:
        """Return all registered entries."""
        return dict(self._entries)

    def remove(self, key: str) -> None:
        """Remove a registered component."""
        if key in self._entries:
            del self._entries[key]
            logger.debug(f"Removed '{key}' from '{self.name}'")

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry('{self.name}', entries={len(self._entries)})"


# Global registries
MODEL_REGISTRY = Registry("models")
PREPROCESSING_REGISTRY = Registry("preprocessing")
METRIC_REGISTRY = Registry("metrics")


def register_model(name: str, overwrite: bool = False) -> Callable:
    """Decorator to register a model class."""

    def decorator(cls: Type) -> Type:
        MODEL_REGISTRY.register(name, cls, overwrite=overwrite)
        return cls

    return decorator


def register_preprocessing(name: str, overwrite: bool = False) -> Callable:
    """Decorator to register a preprocessing pipeline."""

    def decorator(cls: Type) -> Type:
        PREPROCESSING_REGISTRY.register(name, cls, overwrite=overwrite)
        return cls

    return decorator
