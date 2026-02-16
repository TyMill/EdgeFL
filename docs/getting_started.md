# Getting Started

This guide walks you through installing EdgeFL, running a first end-to-end training round, and launching the MAS (multi-agent system) coordinator.

## Requirements

- Python **3.9+**
- `pip`
- Optional for docs: `mkdocs` dependencies (`pip install -e .[docs]`)

## Installation

### From PyPI

```bash
pip install edgefl
```

### From source (recommended for experimentation)

```bash
git clone https://github.com/example/EdgeFL-demo.git
cd EdgeFL-demo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[tests,docs]
```

## First federated round

The snippet below uses:

- `generate_clients_data` to create synthetic client datasets,
- `ClientNode` for local training,
- `FederatedServer` for aggregation.

```python
from edgefl import ClientNode, FederatedServer, SklearnLinearModel, generate_clients_data

clients_data = generate_clients_data(n_clients=5, n_samples=200, seed=42)
X_val, y_val = clients_data[0]

clients = [
    ClientNode(client_id=i, X=X, y=y, model=SklearnLinearModel())
    for i, (X, y) in enumerate(clients_data)
]

server = FederatedServer(
    model=SklearnLinearModel(),
    clients=clients,
    aggregation_strategy="weighted",
    validation_data=(X_val, y_val),
)

result = server.train_round()
print(result["validation"])  # {'mse': ...}
```

## MAS quickstart

The MAS layer wraps clients in `Agent` objects and lets policies/schedulers control participation.

```python
from edgefl import Agent, AgentCoordinator, ClientNode, SklearnLinearModel, generate_clients_data

clients_data = generate_clients_data(n_clients=6, n_samples=160, seed=7)
X_val, y_val = clients_data[0]

agents = [
    Agent(ClientNode(client_id=i, X=X, y=y, model=SklearnLinearModel()))
    for i, (X, y) in enumerate(clients_data)
]

coordinator = AgentCoordinator(
    model=SklearnLinearModel(),
    agents=agents,
    validation_data=(X_val, y_val),
)

for round_idx in range(3):
    round_result = coordinator.train_round(round_index=round_idx)
    print(round_idx, round_result["metrics"]["participation_rate"])
```

## Run included example

```bash
PYTHONPATH=. python examples/agentfl_quickstart.py
```

## Build documentation locally

```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000`.
