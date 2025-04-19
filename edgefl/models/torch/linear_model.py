import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from edgefl.models.base_model import BaseModel


def numpy_to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def tensor_to_numpy(x):
    return x.detach().cpu().numpy()


class TorchLinearModel(BaseModel):
    """
    Lightweight PyTorch-based MLP model for federated learning.
    Suitable for edge devices.
    """
    def __init__(self, input_dim=3, hidden_dim=16):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.train()
        X_tensor = numpy_to_tensor(X)
        y_tensor = numpy_to_tensor(y).view(-1, 1)
        self.optimizer.zero_grad()
        y_pred = self.model(X_tensor)
        loss = self.loss_fn(y_pred, y_tensor)
        loss.backward()
        self.optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = numpy_to_tensor(X)
            y_pred = self.model(X_tensor)
        return tensor_to_numpy(y_pred).flatten()

    def get_weights(self) -> np.ndarray:
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.parameters()])

    def set_weights(self, weights: np.ndarray):
        pointer = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.data = torch.tensor(weights[pointer:pointer+numel], dtype=torch.float32).view_as(param.data)
            pointer += numel

    def get_params(self) -> dict:
        return {"model_structure": str(self.model)}
