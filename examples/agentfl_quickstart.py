"""Quickstart example for the MAS + FL coordination layer.

Run with:
    PYTHONPATH=. python examples/agentfl_quickstart.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from edgefl import (
    Agent,
    AgentCoordinator,
    ClientNode,
    SklearnLinearModel,
    generate_clients_data,
)
from edgefl.environment import churn_scenario


@dataclass
class EnergyAwareParticipationPolicy:
    """Skip rounds when an agent has low energy or trust."""

    min_energy: float = 0.2
    min_trust: float = 0.5

    def select(
        self,
        agent: Agent,
        round_index: int,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        del round_index, context
        return (
            agent.state.energy_budget >= self.min_energy
            and agent.state.trust_score >= self.min_trust
            and agent.state.metadata.get("available", True)
        )


def main() -> None:
    num_agents = 6
    rounds = 8

    clients_data = generate_clients_data(n_clients=num_agents, n_samples=180, seed=7)
    X_val, y_val = clients_data[0]
    agents: list[Agent] = []

    for client_id, (X, y) in enumerate(clients_data):
        client = ClientNode(client_id=client_id, X=X, y=y, model=SklearnLinearModel())
        agent = Agent(
            client=client,
            participation_policy=EnergyAwareParticipationPolicy(),
        )
        agent.state.energy_budget = 1.0
        agent.state.trust_score = 1.0
        agents.append(agent)

    scheduler = churn_scenario(
        num_agents=num_agents,
        rounds=rounds,
        dropout_rate=0.2,
        return_rate=0.15,
        seed=11,
    )

    coordinator = AgentCoordinator(
        model=SklearnLinearModel(),
        agents=agents,
        scheduler=scheduler,
        validation_data=(X_val, y_val),
    )

    print("Round | Selected | Validation(MSE) | Participation | Fairness | Utility")
    print("-" * 80)
    for round_index in range(rounds):
        result = coordinator.train_round(round_index=round_index)
        metrics = result["metrics"]
        validation = result["validation"] or {}
        mse = validation.get("mse", float("nan"))
        print(
            f"{round_index:>5} | "
            f"{result['num_agents']:>8} | "
            f"{mse:>15.4f} | "
            f"{metrics['participation_rate']:>13.2%} | "
            f"{metrics['jains_fairness']:>8.3f} | "
            f"{metrics['system_utility']:>7.3f}"
        )


if __name__ == "__main__":
    main()
