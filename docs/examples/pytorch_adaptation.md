# Adapting EdgeFL to PyTorch Models

EdgeFL supports PyTorch through `TorchModelAdapter`, which implements the shared `BaseModel` protocol.

## 1) Define a PyTorch model

```python
import torch
from torch import nn

class Regressor(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)
```

## 2) Wrap it with `TorchModelAdapter`

```python
from edgefl.models import TorchModelAdapter


def make_torch_model() -> TorchModelAdapter:
    return TorchModelAdapter(
        model=Regressor(n_features=4),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
    )
```

## 3) Use `model_factory` in clients

Using `model_factory` avoids sharing one stateful model instance across clients.

```python
from edgefl import ClientNode, FederatedServer, generate_clients_data

clients_data = generate_clients_data(n_clients=6, n_samples=150, seed=4)
X_val, y_val = clients_data[0]

clients = [
    ClientNode(client_id=i, X=X, y=y, model_factory=make_torch_model)
    for i, (X, y) in enumerate(clients_data)
]

server = FederatedServer(
    model=make_torch_model(),
    clients=clients,
    aggregation_strategy="weighted",
    validation_data=(X_val, y_val),
)

print(server.train_round()["validation"])
```

## Notes

- `TorchModelAdapter.fit` expects regression-like targets (`float`), matching MSE loss by default.
- Use custom `loss_fn` and `optimizer_factory` for different tasks.
- Ensure model input dimension matches generated feature columns.
