"""Policy interfaces for multi-agent coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

if TYPE_CHECKING:
    from .agent import Agent


class ParticipationPolicy(Protocol):
    """Policy interface to decide if an agent participates in a round."""

    def select(
        self,
        agent: "Agent",
        round_index: int,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        """Return True if the agent should participate in this round."""


class TrainingPolicy(Protocol):
    """Policy interface to produce a local update from an agent."""

    def train(
        self,
        agent: "Agent",
        round_index: int,
        context: Mapping[str, Any] | None = None,
    ) -> Array:
        """Train the agent and return model weights."""


class ReportingPolicy(Protocol):
    """Policy interface for handling reporting after aggregation."""

    def update(
        self,
        agent: "Agent",
        round_index: int,
        metrics: Mapping[str, float] | None,
    ) -> None:
        """Update agent metadata with metrics from the round."""


class AlwaysParticipatePolicy:
    """Simple participation policy that always opts in."""

    def select(
        self,
        agent: "Agent",
        round_index: int,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        return True


class LocalTrainingPolicy:
    """Default training policy that delegates to the underlying client."""

    def train(
        self,
        agent: "Agent",
        round_index: int,
        context: Mapping[str, Any] | None = None,
    ) -> Array:
        return agent.client.train()


class StateReportingPolicy:
    """Default reporting policy that stores metrics on the agent state."""

    def update(
        self,
        agent: "Agent",
        round_index: int,
        metrics: Mapping[str, float] | None,
    ) -> None:
        if metrics is None:
            return
        agent.state.last_metrics = dict(metrics)


__all__ = [
    "AlwaysParticipatePolicy",
    "LocalTrainingPolicy",
    "ParticipationPolicy",
    "ReportingPolicy",
    "StateReportingPolicy",
    "TrainingPolicy",
]
