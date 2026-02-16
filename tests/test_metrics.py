import numpy as np

from edgefl.agents import Agent, AgentCoordinator
from edgefl.metrics import (
    communication_cost,
    jains_fairness,
    participation_rate,
    system_utility,
)
from edgefl.models.base_model import BaseModel
from edgefl.nodes.client import ClientNode
from edgefl.simulation import run_agent_simulation


class UpdatingModel(BaseModel):
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


def test_metrics_functions_smoke() -> None:
    updates = [1, 2, 3]
    assert communication_cost(updates) == 6.0
    assert participation_rate([True, False, True]) == 2 / 3
    assert jains_fairness([1, 1, 1]) == 1.0
    assert system_utility(participation_rate=1.0, fairness=1.0) == 1.0


def test_agent_coordinator_records_metrics_keys() -> None:
    base_weights = np.array([0.0, 0.0])
    client_updates = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]

    agents = []
    for index, update in enumerate(client_updates):
        X = np.zeros((4, base_weights.size))
        y = np.zeros(4)
        model = UpdatingModel(initial=base_weights, updates=[update])
        client = ClientNode(client_id=index, X=X, y=y, model=model)
        agents.append(Agent(client=client))

    coordinator = AgentCoordinator(
        model=UpdatingModel(initial=base_weights, updates=[]),
        agents=agents,
    )

    result = coordinator.train_round(round_index=0)

    expected_keys = [
        "communication_cost",
        "participation_rate",
        "jains_fairness",
        "system_utility",
    ]
    assert list(result["metrics"].keys()) == expected_keys
    assert list(coordinator.history[0]["metrics"].keys()) == expected_keys


def test_agent_coordinator_metrics_keys_stable_across_rounds() -> None:
    base_weights = np.array([0.0, 0.0])
    client_updates = [
        [np.array([1.0, 1.0]), np.array([1.5, 1.5])],
        [np.array([2.0, 2.0]), np.array([2.5, 2.5])],
    ]

    agents = []
    for index, updates in enumerate(client_updates):
        X = np.zeros((4, base_weights.size))
        y = np.zeros(4)
        model = UpdatingModel(initial=base_weights, updates=updates)
        client = ClientNode(client_id=index, X=X, y=y, model=model)
        agents.append(Agent(client=client))

    coordinator = AgentCoordinator(
        model=UpdatingModel(initial=base_weights, updates=[]),
        agents=agents,
    )

    first_round = coordinator.train_round(round_index=0)
    second_round = coordinator.train_round(round_index=1)

    expected_keys = [
        "communication_cost",
        "participation_rate",
        "jains_fairness",
        "system_utility",
    ]
    assert list(first_round["metrics"].keys()) == expected_keys
    assert list(second_round["metrics"].keys()) == expected_keys
    assert list(coordinator.history[0]["metrics"].keys()) == expected_keys
    assert list(coordinator.history[1]["metrics"].keys()) == expected_keys


def test_simulation_result_exposes_history_and_final_metrics() -> None:
    result = run_agent_simulation(rounds=2)

    assert len(result.history) == 2
    assert "final_mse" in result.final_metrics
    metrics_keys = [
        "communication_cost",
        "participation_rate",
        "jains_fairness",
        "system_utility",
    ]
    for round_entry in result.history:
        assert "selected_agents" in round_entry
        assert "dropped_agents" in round_entry
        assert "metrics" in round_entry
        assert list(round_entry["metrics"].keys()) == metrics_keys
