"""Scikit-learn based implementations of :class:`BaseModel`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from edgefl.models.base_model import BaseModel

Array = NDArray[np.float64]


@dataclass
class SklearnLinearModel(BaseModel):
    """Linear regression model compatible with the :class:`BaseModel` API."""

    _model: LinearRegression = field(default_factory=LinearRegression, init=False)
    _n_features: Optional[int] = field(default=None, init=False)

    def fit(self, X: Array, y: Array) -> None:
        self._model.fit(X, y)
        self._n_features = int(X.shape[1])

    def predict(self, X: Array) -> Array:
        return self._model.predict(X)

    def get_weights(self) -> Array:
        if hasattr(self._model, "coef_"):
            coef = np.asarray(self._model.coef_, dtype=float)
            intercept = float(getattr(self._model, "intercept_", 0.0))
        else:
            if self._n_features is None:
                raise RuntimeError(
                    "Model weights are not initialised. Fit the model or call set_weights first."
                )
            coef = np.zeros(self._n_features, dtype=float)
            intercept = 0.0
        return np.concatenate([np.atleast_1d(coef), np.array([intercept], dtype=float)])

    def set_weights(self, weights: Array) -> None:
        if weights.ndim != 1:
            raise ValueError("Weights must be provided as a 1-D array")
        if weights.size < 1:
            raise ValueError("Weights array cannot be empty")
        coef, intercept = weights[:-1], weights[-1]
        self._n_features = int(coef.size)
        self._model.coef_ = np.asarray(coef, dtype=float)
        self._model.intercept_ = float(intercept)

    def get_params(self) -> Dict[str, float]:
        return {"fit_intercept": float(self._model.fit_intercept)}
