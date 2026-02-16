"""Event primitives for multi-agent simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


class Event:
    """Base class for environmental events."""

    def apply(self, simulator: object) -> None:
        """Apply the event to the simulator."""

        raise NotImplementedError


@dataclass(frozen=True)
class NodeDropout(Event):
    """Mark one or more agents as unavailable."""

    agent_ids: Sequence[int | str]
    reason: str | None = None

    def apply(self, simulator: object) -> None:
        for agent in getattr(simulator, "agents", []):
            if agent.client.client_id in self.agent_ids:
                agent.state.metadata["available"] = False
                agent.state.active = False


@dataclass(frozen=True)
class NodeReturn(Event):
    """Restore one or more agents to availability."""

    agent_ids: Sequence[int | str]

    def apply(self, simulator: object) -> None:
        for agent in getattr(simulator, "agents", []):
            if agent.client.client_id in self.agent_ids:
                agent.state.metadata["available"] = True


@dataclass(frozen=True)
class TrustPenalty(Event):
    """Apply a trust penalty to agents."""

    agent_ids: Sequence[int | str]
    penalty: float = 0.1

    def apply(self, simulator: object) -> None:
        for agent in getattr(simulator, "agents", []):
            if agent.client.client_id in self.agent_ids:
                agent.state.trust_score = max(
                    agent.state.trust_score - self.penalty, 0.0
                )


@dataclass(frozen=True)
class EnergyShock(Event):
    """Reduce energy budget for targeted agents."""

    agent_ids: Sequence[int | str]
    shock: float = 0.1

    def apply(self, simulator: object) -> None:
        for agent in getattr(simulator, "agents", []):
            if agent.client.client_id in self.agent_ids:
                agent.state.energy_budget = max(
                    agent.state.energy_budget - self.shock,
                    0.0,
                )


@dataclass
class EventScheduler:
    """Schedule events to be applied each round."""

    events_by_round: Mapping[int, Sequence[Event]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._events: dict[int, list[Event]] = {
            round_index: list(events)
            for round_index, events in self.events_by_round.items()
        }

    def add_event(self, round_index: int, event: Event) -> None:
        self._events.setdefault(round_index, []).append(event)

    def add_events(self, round_index: int, events: Iterable[Event]) -> None:
        self._events.setdefault(round_index, []).extend(events)

    def events_for_round(self, round_index: int) -> list[Event]:
        return list(self._events.get(round_index, []))

    def iter_rounds(self, rounds: int) -> Iterable[tuple[int, list[Event]]]:
        for round_index in range(rounds):
            yield round_index, self.events_for_round(round_index)
