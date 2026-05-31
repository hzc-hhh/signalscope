"""
Abstract base class for all SignalScope models.

Provides a consistent interface: train(), predict(), save(), load().
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Base model class combining nn.Module with a high-level interface.

    Subclasses must implement:
    - forward(x) -> Tensor
    - fit(train_loader, val_loader, **kwargs) -> Dict
    """

    def __init__(self, name: str | None = None, **config: Any):
        super().__init__()
        self.name = name or self.__class__.__name__
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ...

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Standard training loop.

        Returns a dict with training history.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history: dict[str, Any] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, batch
                x, y = x.to(device).float(), y.to(device).float()

                optimizer.zero_grad()
                pred = self(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_loss)

            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader, criterion, device)
                history["val_loss"].append(val_loss)

        return history

    def _evaluate_loss(
        self,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: str,
    ) -> float:
        self.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, batch
                x, y = x.to(device).float(), y.to(device).float()
                pred = self(x)
                total += criterion(pred, y).item()
        return total / max(1, len(loader))

    def predict(self, x: torch.Tensor, device: str | None = None) -> np.ndarray:
        """Inference. Returns numpy array."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        with torch.no_grad():
            x_t = x.to(device).float()
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            output = self(x_t)
        return output.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights and config."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
                "name": self.name,
            },
            path,
        )

    def load(self, path: str, device: str | None = None) -> None:
        """Load model weights and config."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(checkpoint["state_dict"])
        self.config = checkpoint.get("config", {})

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        return f"{self.name}(params={params:,})"
