from typing import Optional

import numpy as np
import pytest

from edgefl.models.base_model import BaseModel
from edgefl.nodes.client import ClientNode


class DummyModel(BaseModel):
    def __init__(self, weights: Optional[np.ndarray] = None):
        self._weights = np.array(
            weights if weights is not None else [0.0, 0.0, 0.0], dtype=float
        )
        self.fit_calls = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: ARG002
        self.fit_calls += 1
        self._weights = self._weights + 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self._weights

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        self._weights = weights.astype(float)

    def get_params(self) -> dict[str, object]:
        return {"fit_calls": self.fit_calls}


def test_client_node_trains_and_updates_weights() -> None:
    X = np.ones((10, 3))
    y = np.ones(10)
    model = DummyModel(weights=np.array([0.0, 0.0, 0.0]))

    client = ClientNode(client_id="client-1", X=X, y=y, model=model)
    before = client.get_weights()
    updated = client.train()

    np.testing.assert_array_equal(before, np.zeros(3))
    np.testing.assert_array_equal(updated, np.ones(3))
    np.testing.assert_array_equal(client.get_weights(), np.ones(3))
    assert model.fit_calls == 1


def test_client_node_rejects_non_base_model() -> None:
    class NotAModel:
        pass

    with pytest.raises(TypeError):
        ClientNode(client_id="bad", X=np.zeros((1, 1)), y=np.zeros(1), model=NotAModel())  # type: ignore[arg-type]


def test_client_node_accepts_factory_and_syncs_weights() -> None:
    X = np.ones((5, 3))
    y = np.ones(5)

    client = ClientNode(
        client_id=42,
        X=X,
        y=y,
        model_factory=lambda: DummyModel(weights=np.array([2.0, 2.0, 2.0])),
    )

    client.set_weights(np.array([3.0, 3.0, 3.0]))
    np.testing.assert_array_equal(client.get_weights(), np.array([3.0, 3.0, 3.0]))
