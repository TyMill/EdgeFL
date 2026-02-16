"""Random forest model adapter for federated learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from edgefl.models.base_model import BaseModel

Array = NDArray[np.float64]


@dataclass
class SklearnRandomForestModel(BaseModel):
    """Random forest regressor compatible with :class:`BaseModel`."""

    n_estimators: int = 100
    random_state: Optional[int] = 42
    _model: RandomForestRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state
        )

    def fit(self, X: Array, y: Array) -> None:
        self._model.fit(X, y)

    def predict(self, X: Array) -> Array:
        return self._model.predict(X)

    def get_weights(self) -> Array:
        return np.array([0.0], dtype=float)

    def set_weights(
        self, weights: Array
    ) -> None:  # pragma: no cover - intentional no-op
        _ = weights  # No-op: tree-based models do not expose simple weight vectors

    def get_params(self) -> Dict[str, Union[float, int, None]]:
        return {"n_estimators": self.n_estimators, "random_state": self.random_state}
