"""Agent abstractions built on top of client nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Union

import numpy as np
from numpy.typing import NDArray

from edgefl.nodes.client import ClientNode
from edgefl.models.base_model import BaseModel

from .policies import (
    AlwaysParticipatePolicy,
    LocalTrainingPolicy,
    ParticipationPolicy,
    TrainingPolicy,
)

Array = NDArray[np.float64]


@dataclass
class AgentState:
    """Track explicit state for a multi-agent participant."""

    agent_id: Union[str, int]
    energy_budget: float = 1.0
    trust_score: float = 1.0
    last_local_loss: float | None = None
    participation_history: list[bool] = field(default_factory=list)
    cooldown_until_round: int = 0
    active: bool = True
    last_round: int | None = None
    last_num_examples: int | None = None
    last_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Observation exposed to an agent at decision time."""

    round_index: int
    loss_proxy: float | None
    energy_budget: float
    trust_score: float
    last_local_loss: float | None
    cooldown_until_round: int
    num_examples: int


@dataclass
class AgentAction:
    """Action selected by an agent for a given round."""

    participate: bool
    partial_update_ratio: float = 1.0
    delay_rounds: int = 0


@dataclass
class Agent:
    """Multi-agent wrapper around a :class:`ClientNode`.

    Parameters
    ----------
    client:
        Underlying federated learning client.
    state:
        Explicit agent state tracked across rounds.
    participation_policy:
        Policy controlling round participation.
    training_policy:
        Policy controlling local training behavior.
    """

    client: ClientNode
    state: AgentState | None = None
    participation_policy: ParticipationPolicy = field(
        default_factory=AlwaysParticipatePolicy
    )
    training_policy: TrainingPolicy = field(default_factory=LocalTrainingPolicy)

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = AgentState(agent_id=self.client.client_id)

    def should_participate(
        self,
        round_index: int,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Return whether the agent participates in this round."""

        return self.participation_policy.select(self, round_index, context)

    def observe(self, round_index: int, global_model: BaseModel) -> Observation:
        """Build an observation of the current environment for MAS decisions."""

        loss_proxy: float | None = None
        local_weights = self.client.get_weights()
        global_weights = global_model.get_weights()
        if local_weights is not None and global_weights is not None:
            loss_proxy = float(np.mean(np.square(local_weights - global_weights)))
        return Observation(
            round_index=round_index,
            loss_proxy=loss_proxy,
            energy_budget=self.state.energy_budget,
            trust_score=self.state.trust_score,
            last_local_loss=self.state.last_local_loss,
            cooldown_until_round=self.state.cooldown_until_round,
            num_examples=self.client.num_examples,
        )

    def act(self, observation: Observation) -> AgentAction:
        """Select an action based on the provided observation."""

        if observation.round_index < observation.cooldown_until_round:
            return AgentAction(
                participate=False, partial_update_ratio=0.0, delay_rounds=0
            )
        if observation.energy_budget <= 0.0:
            return AgentAction(
                participate=False, partial_update_ratio=0.0, delay_rounds=0
            )
        return AgentAction(participate=True, partial_update_ratio=1.0, delay_rounds=0)

    def update(self, round_result: Mapping[str, Any]) -> None:
        """Update the agent state based on the round outcome."""

        participated = bool(round_result.get("participated", True))
        self.state.participation_history.append(participated)
        energy_spent = float(round_result.get("energy_spent", 0.0))
        self.state.energy_budget = max(self.state.energy_budget - energy_spent, 0.0)
        trust_delta = float(round_result.get("trust_delta", 0.0))
        self.state.trust_score = max(self.state.trust_score + trust_delta, 0.0)
        if "local_loss" in round_result:
            self.state.last_local_loss = round_result["local_loss"]
        round_index = round_result.get("round_index")
        delay_rounds = round_result.get("delay_rounds", 0)
        if round_index is not None and delay_rounds:
            self.state.cooldown_until_round = max(
                self.state.cooldown_until_round,
                int(round_index) + int(delay_rounds),
            )

    def train(
        self,
        round_index: int,
        context: Mapping[str, Any] | None = None,
    ) -> Array:
        """Run local training via the configured training policy."""

        weights = self.training_policy.train(self, round_index, context)
        self.state.last_round = round_index
        self.state.last_num_examples = self.client.num_examples
        self.state.active = True
        return weights

    def mark_inactive(self) -> None:
        """Mark the agent as inactive for the current round."""

        self.state.active = False
