import numpy as np

from edgefl.agents import Agent, AgentCoordinator
from edgefl.environment.events import (
    EnergyShock,
    EventScheduler,
    NodeDropout,
    NodeReturn,
    TrustPenalty,
)
from edgefl.environment.scenarios import (
    adversarial_scenario,
    churn_scenario,
    energy_constrained_scenario,
)
from edgefl.models.base_model import BaseModel
from edgefl.nodes.client import ClientNode


class StaticModel(BaseModel):
    def __init__(self, weights: np.ndarray):
        self._weights = weights.astype(float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: ARG002
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self._weights

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        self._weights = weights.astype(float)

    def get_params(self) -> dict[str, object]:
        return {"weights": self._weights.tolist()}


def _build_agents(num_agents: int) -> list[Agent]:
    agents: list[Agent] = []
    base_weights = np.zeros(2)
    for idx in range(num_agents):
        X = np.zeros((2, base_weights.size))
        y = np.zeros(2)
        client = ClientNode(
            client_id=idx,
            X=X,
            y=y,
            model=StaticModel(base_weights),
        )
        agents.append(Agent(client=client))
    return agents


def test_churn_scenario_applies_availability_deterministically() -> None:
    num_agents = 3
    rounds = 4
    scheduler = churn_scenario(
        num_agents=num_agents,
        rounds=rounds,
        dropout_rate=0.6,
        return_rate=0.4,
        seed=7,
    )
    agents = _build_agents(num_agents)
    coordinator = AgentCoordinator(
        model=StaticModel(np.zeros(2)),
        agents=agents,
        scheduler=scheduler,
    )

    expected_available = [True] * num_agents
    for round_index in range(rounds):
        for event in scheduler.events_for_round(round_index):
            if isinstance(event, NodeDropout):
                for agent_id in event.agent_ids:
                    expected_available[int(agent_id)] = False
            if isinstance(event, NodeReturn):
                for agent_id in event.agent_ids:
                    expected_available[int(agent_id)] = True

        coordinator.train_round(round_index=round_index)

        availability = [
            agent.state.metadata.get("available", True) for agent in coordinator.agents
        ]
        assert availability == expected_available


def test_scenarios_apply_trust_and_energy_with_fixed_seed() -> None:
    num_agents = 4
    rounds = 3
    energy_scheduler = energy_constrained_scenario(
        num_agents=num_agents,
        rounds=rounds,
        shock_probability=0.8,
        max_shock=0.5,
        seed=42,
    )
    adversarial_scheduler = adversarial_scenario(
        num_agents=num_agents,
        rounds=rounds,
        adversarial_fraction=0.5,
        trust_penalty=0.15,
        seed=99,
    )

    scheduler = EventScheduler()
    for round_index in range(rounds):
        scheduler.add_events(
            round_index, energy_scheduler.events_for_round(round_index)
        )
        scheduler.add_events(
            round_index,
            adversarial_scheduler.events_for_round(round_index),
        )

    agents = _build_agents(num_agents)
    coordinator = AgentCoordinator(
        model=StaticModel(np.zeros(2)),
        agents=agents,
        scheduler=scheduler,
    )

    expected_energy = [1.0] * num_agents
    expected_trust = [1.0] * num_agents
    for round_index in range(rounds):
        for event in scheduler.events_for_round(round_index):
            if isinstance(event, EnergyShock):
                for agent_id in event.agent_ids:
                    expected_energy[int(agent_id)] = max(
                        expected_energy[int(agent_id)] - event.shock,
                        0.0,
                    )
            if isinstance(event, TrustPenalty):
                for agent_id in event.agent_ids:
                    expected_trust[int(agent_id)] = max(
                        expected_trust[int(agent_id)] - event.penalty,
                        0.0,
                    )

        coordinator.train_round(round_index=round_index)

        for agent in coordinator.agents:
            idx = int(agent.client.client_id)
            assert agent.state.energy_budget == expected_energy[idx]
            assert agent.state.trust_score == expected_trust[idx]
