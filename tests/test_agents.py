import numpy as np
import pytest

from edgefl.agents import Agent, AgentCoordinator
from edgefl.models.base_model import BaseModel
from edgefl.nodes.client import ClientNode


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


def test_agent_coordinator_weighted_aggregation() -> None:
    base_weights = np.array([0.0, 0.0, 0.0])
    client_updates = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]
    client_sizes = [5, 10, 15]

    agents: list[Agent] = []
    for index, (num_examples, update) in enumerate(zip(client_sizes, client_updates)):
        X = np.zeros((num_examples, base_weights.size))
        y = np.zeros(num_examples)
        model = UpdatingModel(initial=base_weights, updates=[update])
        client = ClientNode(client_id=index, X=X, y=y, model=model)
        agents.append(Agent(client=client))

    coordinator_model = UpdatingModel(initial=base_weights, updates=[])
    coordinator = AgentCoordinator(model=coordinator_model, agents=agents)

    result = coordinator.train_round()

    stacked = np.stack(client_updates)
    weights = np.array(client_sizes, dtype=float)
    expected = np.average(stacked, axis=0, weights=weights)

    np.testing.assert_allclose(result["weights"], expected)
    np.testing.assert_allclose(coordinator.model.get_weights(), expected)
    assert result["num_agents"] == len(client_sizes)


def test_participation_policy_skips_agent() -> None:
    base_weights = np.array([0.0, 0.0])

    class SkipFirstPolicy:
        def select(
            self, agent: Agent, round_index: int, context=None
        ) -> bool:  # noqa: ARG002
            return agent.client.client_id != 0

    agents: list[Agent] = []
    for index in range(2):
        X = np.zeros((4, base_weights.size))
        y = np.zeros(4)
        model = UpdatingModel(
            initial=base_weights, updates=[np.ones_like(base_weights)]
        )
        client = ClientNode(client_id=index, X=X, y=y, model=model)
        agents.append(Agent(client=client, participation_policy=SkipFirstPolicy()))

    coordinator = AgentCoordinator(
        model=UpdatingModel(initial=base_weights, updates=[]),
        agents=agents,
    )

    result = coordinator.train_round(round_index=0)

    assert result["num_agents"] == 1
    assert not agents[0].state.active
    assert agents[1].state.active


def test_agent_state_transitions_from_observe_act_update() -> None:
    base_weights = np.array([0.0, 0.0])
    local_weights = np.array([1.0, -1.0])

    client_model = UpdatingModel(initial=local_weights, updates=[])
    X = np.zeros((4, base_weights.size))
    y = np.zeros(4)
    client = ClientNode(client_id=0, X=X, y=y, model=client_model)
    agent = Agent(client=client)
    agent.state.energy_budget = 1.0
    agent.state.trust_score = 0.5
    agent.state.cooldown_until_round = 2

    global_model = UpdatingModel(initial=base_weights, updates=[])
    observation = agent.observe(round_index=1, global_model=global_model)

    assert observation.loss_proxy is not None
    np.testing.assert_allclose(observation.loss_proxy, 1.0)

    action = agent.act(observation)
    assert not action.participate

    agent.update(
        {
            "participated": action.participate,
            "energy_spent": 0.0,
            "trust_delta": -0.1,
            "local_loss": 0.25,
            "round_index": 1,
            "delay_rounds": 2,
        }
    )

    assert agent.state.participation_history == [False]
    assert agent.state.trust_score == 0.4
    assert agent.state.last_local_loss == 0.25
    assert agent.state.cooldown_until_round == 3

    observation = agent.observe(round_index=3, global_model=global_model)
    action = agent.act(observation)
    assert action.participate

    agent.update(
        {
            "participated": action.participate,
            "energy_spent": 0.3,
            "trust_delta": 0.2,
        }
    )

    assert agent.state.participation_history == [False, True]
    assert agent.state.energy_budget == 0.7
    assert agent.state.trust_score == pytest.approx(0.6)
