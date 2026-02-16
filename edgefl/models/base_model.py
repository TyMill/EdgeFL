"""Base interfaces for EdgeFL models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class BaseModel(ABC):
    """Abstract base class for all models used in EdgeFL."""

    @abstractmethod
    def fit(self, X: Array, y: Array) -> None:
        """Train the model on the provided dataset."""

    @abstractmethod
    def predict(self, X: Array) -> Array:
        """Return predictions for ``X``."""

    @abstractmethod
    def get_weights(self) -> Array:
        """Return a flattened representation of the model weights."""

    @abstractmethod
    def set_weights(self, weights: Array) -> None:
        """Update the model weights from ``weights``."""

    @abstractmethod
    def get_params(self) -> Dict[str, object]:
        """Return model hyper-parameters or configuration values."""
