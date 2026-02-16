"""Federated server implementation with configurable aggregation strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from edgefl.models.base_model import BaseModel

try:  # Optional import to avoid hard dependency in environments without sklearn
    from sklearn.metrics import mean_squared_error  # type: ignore
except Exception:  # pragma: no cover - fallback when sklearn is unavailable
    mean_squared_error = None  # type: ignore

from edgefl.nodes.client import ClientNode

Array = NDArray[np.float64]
AggregationStrategy = Literal["weighted", "median", "krum"]


@dataclass
class ClientUpdate:
    """Container for a single client's contribution."""

    client_id: Union[str, int]
    weights: Array
    num_examples: int


@dataclass
class FederatedServer:
    """Coordinate federated training across multiple :class:`ClientNode` objects."""

    model: BaseModel
    clients: Sequence[ClientNode]
    aggregation_strategy: AggregationStrategy = "weighted"
    validation_data: Optional[Tuple[Array, Array]] = None
    validation_metric: Optional[Callable[[Array, Array], float]] = None
    validation_metric_name: str = "mse"
    krum_f: int = 0
    history: List[Dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._validate_strategy()

    def _validate_strategy(self) -> None:
        if self.aggregation_strategy not in {"weighted", "median", "krum"}:
            raise ValueError(
                f"Unsupported aggregation strategy: {self.aggregation_strategy}"
            )
        if self.aggregation_strategy == "krum":
            if self.krum_f < 0:
                raise ValueError("krum_f must be non-negative")
            if len(self.clients) <= 2 + self.krum_f:
                raise ValueError(
                    "Krum requires more clients than 2 + krum_f to operate reliably."
                )

    def train_round(
        self,
        participants: List[ClientNode] | None = None,
    ) -> Dict[str, object]:
        """Run a single round of federated training."""

        active_clients = participants if participants is not None else self.clients

        self._broadcast_global_weights(active_clients)
        updates = self._collect_client_updates(active_clients)
        if updates:
            aggregated_weights = self._aggregate_updates(updates)
        else:
            aggregated_weights = self._safe_get_global_weights()
        if aggregated_weights is not None:
            self.model.set_weights(aggregated_weights)
        validation_result = self.validate()

        round_info = {
            "weights": aggregated_weights,
            "validation": validation_result,
            "num_clients": len(updates),
        }
        self.history.append(round_info)
        return round_info

    def _broadcast_global_weights(self, clients: Iterable[ClientNode]) -> None:
        weights = self._safe_get_global_weights()
        if weights is None:
            return
        for client in clients:
            client.set_weights(weights)

    def _safe_get_global_weights(self) -> Optional[Array]:
        try:
            weights = self.model.get_weights()
        except Exception:
            return None
        return weights

    def _collect_client_updates(
        self,
        clients: Iterable[ClientNode],
    ) -> List[ClientUpdate]:
        updates: List[ClientUpdate] = []
        for client in clients:
            weights = client.train()
            updates.append(
                ClientUpdate(
                    client_id=client.client_id,
                    weights=weights,
                    num_examples=client.num_examples,
                )
            )
        return updates

    def _aggregate_updates(self, updates: Sequence[ClientUpdate]) -> Array:
        weights_stack = self._stack_weights([update.weights for update in updates])
        if self.aggregation_strategy == "weighted":
            client_sizes = np.array(
                [update.num_examples for update in updates], dtype=float
            )
            total = float(np.sum(client_sizes))
            if total == 0:
                return np.mean(weights_stack, axis=0)
            return np.average(weights_stack, axis=0, weights=client_sizes)
        if self.aggregation_strategy == "median":
            return np.median(weights_stack, axis=0)
        if self.aggregation_strategy == "krum":
            return self._krum(weights_stack, self.krum_f)
        raise RuntimeError("Unreachable aggregation strategy")

    @staticmethod
    def _stack_weights(weights: Iterable[Array]) -> Array:
        try:
            return np.stack(list(weights))
        except (
            ValueError
        ) as exc:  # pragma: no cover - helps debugging inconsistent models
            raise ValueError("Inconsistent weight shapes between clients") from exc

    @staticmethod
    def _krum(weights: Array, f: int) -> Array:
        num_clients = weights.shape[0]
        if num_clients <= 2 + f:
            raise ValueError("Not enough clients to apply Krum aggregation")
        scores = np.empty(num_clients, dtype=float)
        for i in range(num_clients):
            distances = np.sum((weights[i] - weights) ** 2, axis=1)
            distances = np.delete(distances, i)
            distances.sort()
            trim = num_clients - f - 2
            trim = max(trim, 1)
            scores[i] = float(np.sum(distances[:trim]))
        winner = int(np.argmin(scores))
        return weights[winner]

    def validate(self) -> Optional[Dict[str, float]]:
        """Evaluate the current global model on the validation dataset."""

        if self.validation_data is None:
            return None
        X_val, y_val = self.validation_data
        predictions = self.model.predict(X_val)
        if self.validation_metric is not None:
            metric_value = float(self.validation_metric(y_val, predictions))
        elif mean_squared_error is not None:
            metric_value = float(mean_squared_error(y_val, predictions))
        else:
            metric_value = float(np.mean(np.square(predictions - y_val)))
        return {self.validation_metric_name: metric_value}

    def set_validation_data(self, X_val: Array, y_val: Array) -> None:
        """Attach or update the validation dataset used during training rounds."""

        self.validation_data = (X_val, y_val)

    @property
    def global_weights(self) -> Optional[Array]:
        """Return the current global weights if available."""

        return self._safe_get_global_weights()
