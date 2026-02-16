# ClientNode

`ClientNode` represents one edge participant in a federated round.

## Responsibilities

- owns local dataset (`X`, `y`),
- stores a local model implementing `BaseModel`,
- trains locally and returns updated weights,
- accepts global weights broadcast by server/coordinator.

## Construction

```python
from edgefl import ClientNode, SklearnLinearModel

client = ClientNode(
    client_id=0,
    X=X_local,
    y=y_local,
    model=SklearnLinearModel(),
)
```

You can also pass `model_factory` when every client needs its own fresh model instance:

```python
client = ClientNode(client_id=1, X=X_local, y=y_local, model_factory=SklearnLinearModel)
```

## Main methods

- `train()` → runs local `fit` and returns weights.
- `get_weights()` → returns current local weights.
- `set_weights(weights)` → updates local model from global weights.
- `evaluate(X_val, y_val)` → returns MSE.
- `update_data(X, y)` → replaces local dataset.
