# FederatedServer

`FederatedServer` coordinates training across a collection of `ClientNode` instances.

## Core behavior

For each `train_round()` call, the server:

1. broadcasts current global weights to active clients,
2. collects local model weights from each client,
3. aggregates updates (`weighted`, `median`, or `krum`),
4. optionally validates on `validation_data`,
5. appends round result to `history`.

## Constructor highlights

- `model`: global model implementing `BaseModel`.
- `clients`: sequence of `ClientNode`.
- `aggregation_strategy`: one of `"weighted"`, `"median"`, `"krum"`.
- `validation_data`: optional tuple `(X_val, y_val)`.
- `validation_metric`: optional callable `(y_true, y_pred) -> float`.
- `krum_f`: Byzantine tolerance parameter for Krum.

## Minimal example

```python
from edgefl import ClientNode, FederatedServer, SklearnLinearModel, generate_clients_data

clients_data = generate_clients_data(n_clients=5, n_samples=200)
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

round_result = server.train_round()
print(round_result["validation"])
```
