# Use Cases

This page maps common edge-learning scenarios to concrete EdgeFL building blocks.

## 1) Smart-city environmental sensing

**Goal:** jointly predict air quality from distributed sensors with heterogeneous local conditions.

**Useful EdgeFL components:**

- `generate_clients_data` with multiple region profiles
- `FederatedServer` with `weighted` or `median` aggregation
- validation dataset for tracking MSE over rounds

```python
from edgefl import ClientNode, FederatedServer, SklearnLinearModel, generate_clients_data

clients_data = generate_clients_data(n_clients=12, n_samples=240, seed=10)
X_val, y_val = clients_data[0]

clients = [
    ClientNode(client_id=i, X=X, y=y, model=SklearnLinearModel())
    for i, (X, y) in enumerate(clients_data)
]

server = FederatedServer(
    model=SklearnLinearModel(),
    clients=clients,
    aggregation_strategy="median",
    validation_data=(X_val, y_val),
)

for _ in range(5):
    print(server.train_round()["validation"])
```

## 2) Churn-heavy IoT fleet (MAS)

**Goal:** model intermittent connectivity where devices drop out and rejoin.

**Useful EdgeFL components:**

- `Agent` + `AgentCoordinator`
- `churn_scenario` from `edgefl.environment`
- coordinator metrics to inspect participation/fairness

```python
from edgefl import Agent, AgentCoordinator, ClientNode, SklearnLinearModel, generate_clients_data
from edgefl.environment import churn_scenario

clients_data = generate_clients_data(n_clients=8, n_samples=160, seed=11)
X_val, y_val = clients_data[0]
agents = [
    Agent(ClientNode(client_id=i, X=X, y=y, model=SklearnLinearModel()))
    for i, (X, y) in enumerate(clients_data)
]

scheduler = churn_scenario(num_agents=8, rounds=6, dropout_rate=0.25, return_rate=0.2, seed=3)
coordinator = AgentCoordinator(
    model=SklearnLinearModel(),
    agents=agents,
    scheduler=scheduler,
    validation_data=(X_val, y_val),
)

result = coordinator.train_round(round_index=0)
print(result["metrics"])
```

## 3) Robust aggregation benchmarking

**Goal:** compare behavior of `weighted`, `median`, and `krum` under noisy clients.

**Useful EdgeFL components:**

- identical client setup across runs
- `aggregation_strategy` switch
- run history (`server.history`) for side-by-side metrics

```python
strategies = ["weighted", "median", "krum"]
for strategy in strategies:
    server = FederatedServer(
        model=SklearnLinearModel(),
        clients=clients,
        aggregation_strategy=strategy,
        krum_f=1 if strategy == "krum" else 0,
        validation_data=(X_val, y_val),
    )
    server.train_round()
    print(strategy, server.history[-1]["validation"])
```
