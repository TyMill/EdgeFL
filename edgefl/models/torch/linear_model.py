"""Reference PyTorch implementation of :class:`BaseModel`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from edgefl.models.base_model import BaseModel

Array = NDArray[np.float64]


def _to_tensor(array: Array) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


@dataclass
class TorchLinearModel(BaseModel):
    """A small fully-connected network suitable for federated experiments."""

    input_dim: int = 3
    hidden_dim: int = 16
    learning_rate: float = 0.01
    _model: nn.Sequential = field(init=False, repr=False)
    _loss_fn: nn.Module = field(init=False, repr=False)
    _optimizer: torch.optim.Optimizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self._loss_fn = nn.MSELoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )

    def fit(self, X: Array, y: Array) -> None:
        self._model.train()
        X_tensor = _to_tensor(X)
        y_tensor = _to_tensor(y).view(-1, 1)
        self._optimizer.zero_grad()
        predictions = self._model(X_tensor)
        loss = self._loss_fn(predictions, y_tensor)
        loss.backward()
        self._optimizer.step()

    def predict(self, X: Array) -> Array:
        self._model.eval()
        with torch.no_grad():
            X_tensor = _to_tensor(X)
            predictions = self._model(X_tensor)
        return predictions.detach().cpu().numpy().reshape(-1)

    def get_weights(self) -> Array:
        weights = [
            param.detach().cpu().numpy().reshape(-1)
            for param in self._model.parameters()
        ]
        if not weights:
            return np.array([], dtype=float)
        return np.concatenate(weights)

    def set_weights(self, weights: Array) -> None:
        pointer = 0
        for param in self._model.parameters():
            numel = param.numel()
            if pointer + numel > weights.size:
                raise ValueError("Provided weights do not match model parameters")
            chunk = weights[pointer : pointer + numel]
            param.data = torch.as_tensor(chunk, dtype=torch.float32).view_as(param.data)
            pointer += numel
        if pointer != weights.size:
            raise ValueError("Unused weights remain after assignment")

    def get_params(self) -> Dict[str, float]:
        return {"learning_rate": float(self.learning_rate)}
