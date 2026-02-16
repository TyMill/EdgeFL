# EdgeFL

[![Documentation](https://img.shields.io/badge/docs-material-blue.svg)](https://tymill.github.io/EdgeFL/)
[![PyPI version](https://img.shields.io/pypi/v/edgefl.svg)](https://pypi.org/project/edgefl/)
[![Python versions](https://img.shields.io/pypi/pyversions/edgefl.svg)](https://pypi.org/project/edgefl/)
[![Downloads](https://img.shields.io/pypi/dm/edgefl.svg)](https://pypistats.org/packages/edgefl)


EdgeFL is a lightweight experimentation toolkit for federated learning scenarios deployed on edge devices. It provides utilities for generating synthetic datasets, defining lightweight models, and orchestrating end-to-end simulations of collaborative training sessions across heterogeneous clients.

## Features

- **Synthetic data generation** tailored to environmental sensing scenarios.
- **Composable model abstractions** with a shared `BaseModel` interface.
- **Federated simulation tools** including ready-to-use `ClientNode` and `FederatedServer` implementations.
- **Extensible utilities** for experimenting with aggregation strategies and evaluation metrics.

## Installation

EdgeFL is published as a standard Python package and can be installed with `pip`:

```bash
pip install edgefl
```

To include documentation or testing dependencies, install the corresponding extras:

```bash
pip install edgefl[docs]
pip install edgefl[tests]
```

## Quick start

Get up and running with a minimal synchronous simulation:

```bash
git clone https://github.com/edgefl/EdgeFL.git
cd EdgeFL
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -e .[docs]
```

Then launch a basic FL experiment:

```python
from edgefl import ClientNode, FederatedServer, SklearnLinearModel, generate_clients_data

clients_data = generate_clients_data(n_clients=5, n_samples=200)
clients = [
    ClientNode(i, X, y, model=SklearnLinearModel())
    for i, (X, y) in enumerate(clients_data)
]

server = FederatedServer(model=SklearnLinearModel(), clients=clients)
result = server.train_round()
print(result["validation"])
```

You can also layer in the multi-agent system (MAS) abstractions:

```python
from edgefl import (
    Agent,
    AgentCoordinator,
    ClientNode,
    SklearnLinearModel,
    generate_clients_data,
)

clients_data = generate_clients_data(n_clients=5, n_samples=200)
agents = [
    Agent(ClientNode(i, X, y, model=SklearnLinearModel()))
    for i, (X, y) in enumerate(clients_data)
]

coordinator = AgentCoordinator(model=SklearnLinearModel(), agents=agents)
result = coordinator.train_round()
print(result["validation"])
```

For a runnable MAS + FL script, see:

```bash
python examples/agentfl_quickstart.py
```

## MAS layer concepts

EdgeFL's MAS layer adds decision and coordination logic on top of standard FL training.

- **Agent**: wraps a `ClientNode` and tracks state such as energy budget, trust score, activity, and participation history.
- **Policy**: pluggable logic for participation and local training. Policies determine *when* and *how* agents contribute.
- **Coordinator**: `AgentCoordinator` runs rounds, applies policies, schedules events, aggregates updates, and records MAS-level metrics.
- **Scenarios**: environment builders (e.g., churn, adversarial, energy-constrained) that inject events like dropouts, returns, and trust/energy shifts.

## MAS quickstart snippet

```python
from edgefl import Agent, AgentCoordinator, ClientNode, SklearnLinearModel, generate_clients_data
from edgefl.environment import churn_scenario

clients_data = generate_clients_data(n_clients=6, n_samples=180, seed=7)
agents = [
    Agent(ClientNode(i, X, y, model=SklearnLinearModel()))
    for i, (X, y) in enumerate(clients_data)
]

scheduler = churn_scenario(num_agents=6, rounds=8, dropout_rate=0.2, return_rate=0.15, seed=11)
coordinator = AgentCoordinator(model=SklearnLinearModel(), agents=agents, scheduler=scheduler)

for r in range(8):
    result = coordinator.train_round(round_index=r)
    print(r, result["num_agents"], result["metrics"]["participation_rate"])
```

## MAS metrics

Each coordinator round includes the following metrics:

- `communication_cost`: proxy for total communication volume in the round.
- `participation_rate`: fraction of agents that contributed updates.
- `jains_fairness`: fairness index over cumulative participation counts.
- `system_utility`: combined utility signal (participation, fairness, optional validation and communication weighting).

## Research framing: MAS + FL hybrid

EdgeFL is designed as a hybrid of federated learning and multi-agent systems:

- **Policy-driven participation** from MAS allows each agent to adapt to local constraints (for example, trust, energy, or availability).
- **Coordinated aggregation** from FL preserves a global model objective while supporting decentralized, heterogeneous behavior.

This pairing enables controlled studies of strategic participation, robustness, and efficiency under realistic edge-network dynamics.

## Project Structure

- `edgefl/data` – Synthetic dataset generators for typical edge deployments.
- `edgefl/models` – Base classes and reference implementations for models.
- `edgefl/simulation` – Core simulation loop and orchestrator utilities.
- `docs/` – Project documentation sources.
- `examples/` – Usage examples and notebooks.

## Documentation

Build the static site locally with [MkDocs](https://www.mkdocs.org/):

```bash
pip install -e .[docs]
mkdocs serve  # live preview at http://127.0.0.1:8000/
mkdocs build  # outputs site/ with static assets
```

To publish the documentation to GitHub Pages:

1. Enable GitHub Pages in the repository settings, selecting the `gh-pages` branch.
2. Run `mkdocs gh-deploy --force` locally (after authenticating with GitHub). This command builds the site and pushes the result to the `gh-pages` branch.
3. Visit `https://<your-account>.github.io/EdgeFL/` to verify the deployment.

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub to propose enhancements or bug fixes.

## License

EdgeFL is released under the [Apache License 2.0](LICENSE).
