"""Model abstractions and concrete implementations."""

from __future__ import annotations

from .base_model import BaseModel
from .sklearn import SklearnLinearModel, SklearnRandomForestModel

try:  # pragma: no cover - optional dependency
    from .torch import TorchLinearModel, TorchModelAdapter
except ImportError:  # pragma: no cover - torch is optional
    TorchLinearModel = None  # type: ignore[assignment]
    TorchModelAdapter = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = True

__all__ = [
    "BaseModel",
    "SklearnLinearModel",
    "SklearnRandomForestModel",
]

if _TORCH_AVAILABLE:  # pragma: no cover - depends on optional dependency
    __all__ += ["TorchLinearModel", "TorchModelAdapter"]
