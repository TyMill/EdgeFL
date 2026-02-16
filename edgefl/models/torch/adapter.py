"""PyTorch adapter implementing the :class:`BaseModel` protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from edgefl.models.base_model import BaseModel

Array = NDArray[np.float64]
OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]


@dataclass
class TorchModelAdapter(BaseModel):
    """Adapter enabling any :class:`torch.nn.Module` to satisfy :class:`BaseModel`."""

    model: nn.Module
    optimizer_factory: Optional[OptimizerFactory] = None
    loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        self.device = self.device or torch.device("cpu")
        self.model.to(self.device)
        self.loss_fn = self.loss_fn or nn.MSELoss()
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_factory is not None:
            return self.optimizer_factory(self.model.parameters())
        return torch.optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, X: Array, y: Array) -> None:
        self.model.train()
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device).view(
            -1, 1
        )
        self.optimizer.zero_grad()
        predictions = self.model(X_tensor)
        loss = self.loss_fn(predictions.view_as(y_tensor), y_tensor)
        loss.backward()
        self.optimizer.step()

    def predict(self, X: Array) -> Array:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            predictions = self.model(X_tensor)
        return predictions.detach().cpu().numpy().reshape(-1)

    def get_weights(self) -> Array:
        weights = [
            param.detach().cpu().numpy().reshape(-1)
            for param in self.model.parameters()
        ]
        if not weights:
            return np.array([], dtype=float)
        return np.concatenate(weights)

    def set_weights(self, weights: Array) -> None:
        pointer = 0
        for param in self.model.parameters():
            numel = param.numel()
            if pointer + numel > weights.size:
                raise ValueError("Provided weights do not match model parameters")
            chunk = weights[pointer : pointer + numel]
            param.data = torch.as_tensor(
                chunk, dtype=torch.float32, device=self.device
            ).view_as(param.data)
            pointer += numel
        if pointer != weights.size:
            raise ValueError("Unused weights remain after assignment")

    def get_params(self) -> Dict[str, Any]:
        return {"model": self.model.__class__.__name__, "device": str(self.device)}
