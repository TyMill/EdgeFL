"""Coordinator for multi-agent federated simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from edgefl.environment.events import Event, EventScheduler
from edgefl.metrics import (
    communication_cost,
    jains_fairness,
    participation_rate,
    system_utility,
)
from edgefl.models.base_model import BaseModel
from edgefl.server.fedavg import AggregationStrategy, FederatedServer

from .agent import Agent
from .policies import ReportingPolicy, StateReportingPolicy

Array = NDArray[np.float64]


@dataclass
class AgentUpdate:
    """Container for a single agent's contribution."""

    agent_id: Union[str, int]
    weights: Array
    num_examples: int


@dataclass
class AgentCoordinator:
    """Coordinate a multi-agent layer on top of federated clients."""

    model: BaseModel
    agents: Sequence[Agent]
    aggregation_strategy: AggregationStrategy = "weighted"
    validation_data: Optional[Tuple[Array, Array]] = None
    validation_metric: Optional[Callable[[Array, Array], float]] = None
    validation_metric_name: str = "mse"
    krum_f: int = 0
    reporting_policy: ReportingPolicy = field(default_factory=StateReportingPolicy)
    scheduler: EventScheduler | None = None
    history: list[dict[str, object]] = field(default_factory=list, init=False)
    _server: FederatedServer | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_strategy()
        for agent in self.agents:
            agent.state.metadata.setdefault("available", True)
        self._server = FederatedServer(
            model=self.model,
            clients=[agent.client for agent in self.agents],
            aggregation_strategy=self.aggregation_strategy,
            validation_data=self.validation_data,
            validation_metric=self.validation_metric,
            validation_metric_name=self.validation_metric_name,
            krum_f=self.krum_f,
        )

    def _validate_strategy(self) -> None:
        if self.aggregation_strategy not in {"weighted", "median", "krum"}:
            raise ValueError(
                f"Unsupported aggregation strategy: {self.aggregation_strategy}"
            )
        if self.aggregation_strategy == "krum":
            if self.krum_f < 0:
                raise ValueError("krum_f must be non-negative")
            if len(self.agents) <= 2 + self.krum_f:
                raise ValueError(
                    "Krum requires more agents than 2 + krum_f to operate reliably."
                )

    def train_round(
        self,
        round_index: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, object]:
        """Run a single round of multi-agent training."""

        if round_index is None:
            round_index = len(self.history)

        self._apply_events(round_index)

        selected_clients = []
        selected_agents: list[Union[str, int]] = []
        dropped_agents: list[Union[str, int]] = []
        for agent in self.agents:
            if not agent.state.metadata.get("available", True):
                agent.mark_inactive()
                dropped_agents.append(agent.state.agent_id)
                continue
            if agent.should_participate(round_index, context):
                agent.state.active = True
                agent.state.last_round = round_index
                agent.state.last_num_examples = agent.client.num_examples
                selected_clients.append(agent.client)
                selected_agents.append(agent.state.agent_id)
            else:
                agent.mark_inactive()
                dropped_agents.append(agent.state.agent_id)

        if self._server is None:
            raise RuntimeError("Federated server not initialized")

        round_result = self._server.train_round(participants=selected_clients)
        validation_result = round_result["validation"]
        aggregated_weights = round_result["weights"]

        for agent in self.agents:
            self.reporting_policy.update(agent, round_index, validation_result)

        for agent in self.agents:
            agent.state.participation_history.append(
                agent.state.agent_id in selected_agents
            )

        participation_counts = [
            sum(agent.state.participation_history) for agent in self.agents
        ]
        round_participation_rate = participation_rate(
            [agent.state.agent_id in selected_agents for agent in self.agents]
        )
        round_communication_cost = communication_cost(selected_clients)
        round_fairness = jains_fairness(participation_counts)
        global_validation_score: float | None = None
        if validation_result:
            global_validation_score = float(next(iter(validation_result.values())))
        round_metrics = {
            "communication_cost": round_communication_cost,
            "participation_rate": round_participation_rate,
            "jains_fairness": round_fairness,
            "system_utility": system_utility(
                participation_rate=round_participation_rate,
                fairness=round_fairness,
                validation_score=global_validation_score,
                communication_cost=round_communication_cost,
            ),
        }

        round_info = {
            "weights": aggregated_weights,
            "validation": validation_result,
            "num_agents": len(selected_clients),
            "selected_agents": selected_agents,
            "dropped_agents": dropped_agents,
            "metrics": round_metrics,
            "global_validation": global_validation_score,
        }
        self.history.append(round_info)
        return round_info

    def _apply_events(self, round_index: int) -> None:
        if self.scheduler is None:
            return
        for event in self.scheduler.events_for_round(round_index):
            self._apply_event(event)

    def _apply_event(self, event: Event) -> None:
        event.apply(self)

    def _broadcast_global_weights(self) -> None:
        weights = self._safe_get_global_weights()
        if weights is None:
            return
        for agent in self.agents:
            agent.client.set_weights(weights)

    def _safe_get_global_weights(self) -> Optional[Array]:
        try:
            weights = self.model.get_weights()
        except Exception:
            return None
        return weights

    def _collect_agent_updates(
        self,
        round_index: int,
        context: Mapping[str, Any] | None,
    ) -> list[AgentUpdate]:
        updates: list[AgentUpdate] = []
        for agent in self.agents:
            if not agent.should_participate(round_index, context):
                agent.mark_inactive()
                continue
            weights = agent.train(round_index, context)
            updates.append(
                AgentUpdate(
                    agent_id=agent.client.client_id,
                    weights=weights,
                    num_examples=agent.client.num_examples,
                )
            )
        return updates

    def _aggregate_updates(self, updates: Sequence[AgentUpdate]) -> Optional[Array]:
        if not updates:
            return self._safe_get_global_weights()
        weights_stack = self._stack_weights([update.weights for update in updates])
        if self.aggregation_strategy == "weighted":
            agent_sizes = np.array(
                [update.num_examples for update in updates], dtype=float
            )
            total = float(np.sum(agent_sizes))
            if total == 0:
                return np.mean(weights_stack, axis=0)
            return np.average(weights_stack, axis=0, weights=agent_sizes)
        if self.aggregation_strategy == "median":
            return np.median(weights_stack, axis=0)
        if self.aggregation_strategy == "krum":
            return self._krum(weights_stack, self.krum_f)
        raise RuntimeError("Unreachable aggregation strategy")

    @staticmethod
    def _stack_weights(weights: Iterable[Array]) -> Array:
        try:
            return np.stack(list(weights))
        except ValueError as exc:  # pragma: no cover
            raise ValueError("Inconsistent weight shapes between agents") from exc

    @staticmethod
    def _krum(weights: Array, f: int) -> Array:
        num_agents = weights.shape[0]
        if num_agents <= 2 + f:
            raise ValueError("Not enough agents to apply Krum aggregation")
        scores = np.empty(num_agents, dtype=float)
        for i in range(num_agents):
            distances = np.sum((weights[i] - weights) ** 2, axis=1)
            distances = np.delete(distances, i)
            distances.sort()
            trim = num_agents - f - 2
            trim = max(trim, 1)
            scores[i] = float(np.sum(distances[:trim]))
        winner = int(np.argmin(scores))
        return weights[winner]

    def validate(self) -> Optional[dict[str, float]]:
        """Evaluate the current global model on the validation dataset."""

        if self.validation_data is None:
            return None
        X_val, y_val = self.validation_data
        predictions = self.model.predict(X_val)
        if self.validation_metric is not None:
            metric_value = float(self.validation_metric(y_val, predictions))
        else:
            metric_value = float(np.mean(np.square(predictions - y_val)))
        return {self.validation_metric_name: metric_value}
