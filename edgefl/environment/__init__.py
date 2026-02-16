"""Scenario and event helpers for EdgeFL simulations."""

from .events import (
    EnergyShock,
    Event,
    EventScheduler,
    NodeDropout,
    NodeReturn,
    TrustPenalty,
)
from .scenarios import (
    adversarial_scenario,
    churn_scenario,
    energy_constrained_scenario,
)

__all__ = [
    "Event",
    "EventScheduler",
    "NodeDropout",
    "NodeReturn",
    "TrustPenalty",
    "EnergyShock",
    "churn_scenario",
    "adversarial_scenario",
    "energy_constrained_scenario",
]
