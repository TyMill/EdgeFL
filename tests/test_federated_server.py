import numpy as np
import pytest

from edgefl.data.generator import generate_clients_data
from edgefl.models.base_model import BaseModel
from edgefl.nodes.client import ClientNode
from edgefl.server.fedavg import FederatedServer


class UpdatingModel(BaseModel):
    """Model that replaces its weights with a predetermined sequence."""

    def __init__(self, initial: np.ndarray, updates: list[np.ndarray]):
        self._weights = initial.astype(float)
        self._updates = [u.astype(float) for u in updates]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: ARG002
        if self._updates:
            self._weights = self._updates.pop(0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self._weights

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        self._weights = weights.astype(float)

    def get_params(self) -> dict[str, object]:
        return {"updates_remaining": len(self._updates)}


class LeastSquaresModel(BaseModel):
    """Simple linear regression solved with least squares."""

    def __init__(self, n_features: int):
        self._weights = np.zeros(n_features, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        solution, *_ = np.linalg.lstsq(X, y, rcond=None)
        self._weights = solution.astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self._weights

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        self._weights = weights.astype(float)

    def get_params(self) -> dict[str, object]:
        return {"kind": "least_squares"}


@pytest.mark.parametrize(
    "client_sizes",
    [
        (5, 10, 15),
        (1, 1, 1),
    ],
)
def test_fedavg_weighted_aggregation(client_sizes: tuple[int, ...]) -> None:
    base_weights = np.array([0.0, 0.0, 0.0])
    client_updates = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]

    clients: list[ClientNode] = []
    for index, (num_examples, update) in enumerate(zip(client_sizes, client_updates)):
        X = np.zeros((num_examples, base_weights.size))
        y = np.zeros(num_examples)
        model = UpdatingModel(initial=base_weights, updates=[update])
        client = ClientNode(client_id=index, X=X, y=y, model=model)
        clients.append(client)

    server_model = UpdatingModel(initial=base_weights, updates=[])
    server = FederatedServer(model=server_model, clients=clients)

    result = server.train_round()

    stacked = np.stack(client_updates)
    weights = np.array(client_sizes, dtype=float)
    expected = np.average(stacked, axis=0, weights=weights)

    np.testing.assert_allclose(result["weights"], expected)
    np.testing.assert_allclose(server.model.get_weights(), expected)
    assert result["num_clients"] == len(client_sizes)


def test_federated_round_integration_matches_manual_average() -> None:
    datasets = generate_clients_data(n_clients=3, n_samples=50, seed=123)
    n_features = datasets[0][0].shape[1]

    clients: list[ClientNode] = []
    client_weights: list[np.ndarray] = []
    client_sizes: list[int] = []

    for client_id, (X, y) in enumerate(datasets):
        client_sizes.append(y.shape[0])
        reference_model = LeastSquaresModel(n_features)
        reference_model.fit(X, y)
        client_weights.append(reference_model.get_weights())

        client = ClientNode(
            client_id=client_id,
            X=X,
            y=y,
            model_factory=lambda nf=n_features: LeastSquaresModel(nf),
        )
        clients.append(client)

    server_model = LeastSquaresModel(n_features)
    server = FederatedServer(model=server_model, clients=clients)

    result = server.train_round()

    expected = np.average(
        np.stack(client_weights), axis=0, weights=np.array(client_sizes, dtype=float)
    )

    np.testing.assert_allclose(result["weights"], expected, rtol=1e-5, atol=1e-5)
    assert server.history, "History should record the completed round"


def test_fedavg_round_accepts_participant_subset() -> None:
    base_weights = np.array([0.0, 0.0, 0.0])
    client_updates = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]

    clients: list[ClientNode] = []
    for index, update in enumerate(client_updates):
        X = np.zeros((2, base_weights.size))
        y = np.zeros(2)
        model = UpdatingModel(initial=base_weights, updates=[update])
        client = ClientNode(client_id=index, X=X, y=y, model=model)
        clients.append(client)

    server_model = UpdatingModel(initial=base_weights, updates=[])
    server = FederatedServer(model=server_model, clients=clients)

    result = server.train_round(participants=[clients[1]])

    np.testing.assert_allclose(result["weights"], client_updates[1])
    np.testing.assert_allclose(server.model.get_weights(), client_updates[1])
    assert result["num_clients"] == 1
