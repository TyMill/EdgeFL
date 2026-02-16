# Urban Sensor Network Simulation

This walkthrough demonstrates a realistic FL loop for distributed air-quality sensors.

## Scenario

- 20 clients represent districts with local environmental variation.
- Each client trains on private data.
- Server aggregates updates and reports validation MSE each round.

## Step 1: generate client datasets

```python
from edgefl import generate_clients_data

clients_data = generate_clients_data(
    n_clients=20,
    n_samples=24 * 10,
    seed=21,
)

X_val, y_val = clients_data[0]
```

## Step 2: build clients and server

```python
from edgefl import ClientNode, FederatedServer, SklearnLinearModel

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
```

## Step 3: train multiple rounds

```python
for round_idx in range(8):
    result = server.train_round()
    print(round_idx, result["num_clients"], result["validation"])
```

## Step 4: inspect history

```python
import pandas as pd

df = pd.DataFrame(server.history)
print(df[["num_clients", "validation"]].tail())
```

## Why this is useful

- Quick benchmark for convergence behavior.
- Baseline for testing robust aggregation (`median`, `krum`).
- Easy to extend with MAS events (`AgentCoordinator` + scheduler).
