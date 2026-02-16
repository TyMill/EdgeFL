"""Scenario builders for multi-agent simulations."""

from __future__ import annotations

import numpy as np

from .events import EnergyShock, EventScheduler, NodeDropout, NodeReturn, TrustPenalty


def churn_scenario(
    *,
    num_agents: int,
    rounds: int,
    dropout_rate: float = 0.1,
    return_rate: float = 0.05,
    seed: int | None = None,
) -> EventScheduler:
    """Simulate stochastic churn with dropouts and returns."""

    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if not 0.0 <= dropout_rate <= 1.0:
        raise ValueError("dropout_rate must be between 0 and 1")
    if not 0.0 <= return_rate <= 1.0:
        raise ValueError("return_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    available = np.ones(num_agents, dtype=bool)
    scheduler = EventScheduler()

    for round_index in range(rounds):
        for agent_id in range(num_agents):
            if available[agent_id]:
                if rng.random() < dropout_rate:
                    scheduler.add_event(round_index, NodeDropout([agent_id]))
                    available[agent_id] = False
            else:
                if rng.random() < return_rate:
                    scheduler.add_event(round_index, NodeReturn([agent_id]))
                    available[agent_id] = True

    return scheduler


def adversarial_scenario(
    *,
    num_agents: int,
    rounds: int,
    adversarial_fraction: float = 0.2,
    trust_penalty: float = 0.2,
    seed: int | None = None,
) -> EventScheduler:
    """Simulate adversarial agents that incur trust penalties."""

    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if not 0.0 <= adversarial_fraction <= 1.0:
        raise ValueError("adversarial_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    if adversarial_fraction <= 0.0:
        adversaries: np.ndarray = np.array([], dtype=int)
    else:
        count = int(np.ceil(num_agents * adversarial_fraction))
        count = min(count, num_agents)
        adversaries = rng.choice(num_agents, size=count, replace=False)

    scheduler = EventScheduler()
    for round_index in range(rounds):
        if adversaries.size:
            scheduler.add_event(
                round_index,
                TrustPenalty(agent_ids=adversaries.tolist(), penalty=trust_penalty),
            )

    return scheduler


def energy_constrained_scenario(
    *,
    num_agents: int,
    rounds: int,
    shock_probability: float = 0.2,
    max_shock: float = 0.4,
    seed: int | None = None,
) -> EventScheduler:
    """Simulate fluctuating energy budgets across agents."""

    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if not 0.0 <= shock_probability <= 1.0:
        raise ValueError("shock_probability must be between 0 and 1")
    if max_shock < 0:
        raise ValueError("max_shock must be non-negative")

    rng = np.random.default_rng(seed)
    scheduler = EventScheduler()

    for round_index in range(rounds):
        for agent_id in range(num_agents):
            if rng.random() < shock_probability:
                shock = float(rng.uniform(0.0, max_shock))
                scheduler.add_event(
                    round_index,
                    EnergyShock(agent_ids=[agent_id], shock=shock),
                )

    return scheduler
