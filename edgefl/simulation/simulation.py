"""High-level helpers to run simple federated learning simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from edgefl.data import EdgeDataset
from edgefl.data.generator import generate_clients_data, generate_environmental_data
from edgefl.models.sklearn import SklearnLinearModel
from edgefl.agents import Agent, AgentCoordinator
from edgefl.nodes import ClientNode
from edgefl.server import AggregationStrategy, FederatedServer
from edgefl.environment.events import EventScheduler


@dataclass
class SimulationResult:
    """Container for simulation history and final metrics."""

    history: list[dict[str, object]]
    final_metrics: dict[str, float]


def _build_clients(
    clients_data: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[ClientNode]:
    clients: list[ClientNode] = []
    feature_names = ["temperature", "humidity", "pm25"]
    for client_id, (X, y) in enumerate(clients_data):
        dataset = EdgeDataset(X, y, feature_names=feature_names, target_name="aqi")
        clients.append(
            ClientNode(
                client_id=client_id,
                X=dataset.features,
                y=dataset.target,
                model=SklearnLinearModel(),
            )
        )
    return clients


def run_simulation(
    rounds: int = 5, aggregation_strategy: AggregationStrategy = "weighted"
) -> None:
    """Run an example simulation using generated environmental data."""

    clients_data = generate_clients_data(n_clients=5, n_samples=100)
    clients = _build_clients(clients_data)

    df_val = generate_environmental_data(n_samples=100, region_factor=1.2, seed=999)
    val_dataset = EdgeDataset.from_dataframe(df_val, target_column="aqi")
    X_val = val_dataset.features
    y_val = val_dataset.target

    server = FederatedServer(
        model=SklearnLinearModel(),
        clients=clients,
        aggregation_strategy=aggregation_strategy,
        validation_data=(X_val, y_val),
    )

    for round_index in range(rounds):
        print(f"--- Round {round_index + 1} ---")
        result = server.train_round()
        if result["validation"]:
            print(f"Validation {result['validation']}")

    predictions = server.model.predict(X_val)
    mse = float(np.mean(np.square(predictions - y_val)))
    print(f"Final MSE on validation set: {mse:.4f}")


def run_agent_simulation(
    rounds: int = 5,
    aggregation_strategy: AggregationStrategy = "weighted",
    scheduler: EventScheduler | None = None,
) -> SimulationResult:
    """Run an example simulation using the multi-agent layer."""

    clients_data = generate_clients_data(n_clients=5, n_samples=100)
    clients = _build_clients(clients_data)
    agents = [Agent(client=client) for client in clients]

    df_val = generate_environmental_data(n_samples=100, region_factor=1.2, seed=999)
    val_dataset = EdgeDataset.from_dataframe(df_val, target_column="aqi")
    X_val = val_dataset.features
    y_val = val_dataset.target

    coordinator = AgentCoordinator(
        model=SklearnLinearModel(),
        agents=agents,
        aggregation_strategy=aggregation_strategy,
        validation_data=(X_val, y_val),
        scheduler=scheduler,
    )

    for round_index in range(rounds):
        print(f"--- Agent Round {round_index + 1} ---")
        result = coordinator.train_round(round_index=round_index)
        if result["validation"]:
            print(f"Validation {result['validation']}")

    predictions = coordinator.model.predict(X_val)
    mse = float(np.mean(np.square(predictions - y_val)))
    print(f"Final MSE on validation set: {mse:.4f}")
    final_metrics = {"final_mse": mse}
    if coordinator.history and "metrics" in coordinator.history[-1]:
        final_metrics.update(
            {
                key: float(value)
                for key, value in coordinator.history[-1]["metrics"].items()
            }
        )
    return SimulationResult(history=coordinator.history, final_metrics=final_metrics)


if __name__ == "__main__":  # pragma: no cover
    run_simulation()
