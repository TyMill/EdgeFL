"""Multi-agent abstractions for EdgeFL."""

from .agent import Agent, AgentAction, AgentState, Observation
from .coordinator import AgentCoordinator, AgentUpdate
from .policies import (
    AlwaysParticipatePolicy,
    LocalTrainingPolicy,
    ParticipationPolicy,
    ReportingPolicy,
    StateReportingPolicy,
    TrainingPolicy,
)

__all__ = [
    "Agent",
    "AgentAction",
    "AgentCoordinator",
    "AgentState",
    "AgentUpdate",
    "Observation",
    "AlwaysParticipatePolicy",
    "LocalTrainingPolicy",
    "ParticipationPolicy",
    "ReportingPolicy",
    "StateReportingPolicy",
    "TrainingPolicy",
]
