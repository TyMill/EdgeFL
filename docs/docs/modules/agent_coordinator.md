# AgentCoordinator

`AgentCoordinator` extends federated training with multi-agent decision logic.

## What it adds on top of `FederatedServer`

- participation policies per agent,
- event-driven state updates via scheduler,
- additional round metrics (`participation_rate`, `jains_fairness`, `system_utility`),
- tracking of selected vs dropped agents.

## Typical usage

```python
from edgefl import Agent, AgentCoordinator, ClientNode, SklearnLinearModel, generate_clients_data

clients_data = generate_clients_data(n_clients=6, n_samples=150)
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

round_result = coordinator.train_round(round_index=0)
print(round_result["metrics"])
```
