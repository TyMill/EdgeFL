# MAS Coordination with Churn Events

This example focuses on the MAS layer where agent participation changes over time because of simulated dropouts/returns.

## Setup

```python
from edgefl import Agent, AgentCoordinator, ClientNode, SklearnLinearModel, generate_clients_data
from edgefl.environment import churn_scenario

num_agents = 10
rounds = 6

clients_data = generate_clients_data(n_clients=num_agents, n_samples=180, seed=9)
X_val, y_val = clients_data[0]

agents = [
    Agent(ClientNode(client_id=i, X=X, y=y, model=SklearnLinearModel()))
    for i, (X, y) in enumerate(clients_data)
]

scheduler = churn_scenario(
    num_agents=num_agents,
    rounds=rounds,
    dropout_rate=0.2,
    return_rate=0.15,
    seed=12,
)

coordinator = AgentCoordinator(
    model=SklearnLinearModel(),
    agents=agents,
    scheduler=scheduler,
    validation_data=(X_val, y_val),
)
```

## Run and inspect metrics

```python
for r in range(rounds):
    result = coordinator.train_round(round_index=r)
    m = result["metrics"]
    print(
        r,
        result["num_agents"],
        f"participation={m['participation_rate']:.2f}",
        f"fairness={m['jains_fairness']:.3f}",
        f"utility={m['system_utility']:.3f}",
    )
```

## What to watch

- `selected_agents` and `dropped_agents` per round
- movement of `participation_rate` under churn
- fairness trends over longer runs
