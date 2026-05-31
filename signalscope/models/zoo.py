"""
Model Zoo — centralized model discovery and instantiation.

Provides a clean API to list, get, and instantiate any registered model.
"""

from typing import Any, Dict, Optional

from signalscope.core.registry import MODEL_REGISTRY


class ModelZoo:
    """
    Central registry for model discovery and instantiation.

    Usage:
        zoo = ModelZoo()
        zoo.list()                    # Show all registered models
        model = zoo.get("resnet1d", in_channels=1, num_classes=1)

    Models are auto-registered via the @register_model decorator.
    """

    def __init__(self):
        self.registry = MODEL_REGISTRY

    def list(self) -> Dict[str, type]:
        """Return all registered models."""
        return self.registry.list()

    def get(self, name: str, **kwargs: Any) -> Any:
        """
        Instantiate a model by name.

        Parameters
        ----------
        name : str
            Registered model name (e.g., 'resnet1d', 'transformer_ts').
        **kwargs
            Passed to the model constructor.

        Returns
        -------
        model instance
        """
        model_cls = self.registry.get(name)
        return model_cls(**kwargs)

    def summary(self) -> str:
        """Pretty-print registered models."""
        lines = [f"Model Zoo ({len(self.registry)} models):"]
        for name, cls in sorted(self.registry.list().items()):
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            lines.append(f"  • {name:20s} — {doc}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
