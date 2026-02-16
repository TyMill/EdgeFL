"""Utilities for serialising model weights and tracking model versions."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def weights_to_bytes(weights: Array) -> bytes:
    """Serialise a NumPy array of weights into a byte representation."""

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(weights, dtype=float), allow_pickle=False)
    return buffer.getvalue()


def bytes_to_weights(data: bytes) -> Array:
    """Deserialise weights previously produced by :func:`weights_to_bytes`."""

    buffer = io.BytesIO(data)
    buffer.seek(0)
    return np.load(buffer, allow_pickle=False)


@dataclass
class ModelVersion:
    """Simple record describing a stored model version."""

    version: int
    weights: bytes
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelVersionTracker:
    """In-memory tracker that stores multiple versions of model weights."""

    def __init__(self) -> None:
        self._versions: Dict[int, ModelVersion] = {}
        self._latest_version: Optional[int] = None

    def register(
        self, weights: Array, metadata: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """Persist a new version of the provided weights."""

        version = 1 if self._latest_version is None else self._latest_version + 1
        model_version = ModelVersion(
            version=version,
            weights=weights_to_bytes(weights),
            metadata=metadata or {},
        )
        self._versions[version] = model_version
        self._latest_version = version
        return model_version

    def latest(self) -> Optional[ModelVersion]:
        """Return the most recently registered model version."""

        if self._latest_version is None:
            return None
        return self._versions[self._latest_version]

    def get(self, version: int) -> ModelVersion:
        """Return the model version with the specified identifier."""

        if version not in self._versions:
            raise KeyError(f"Unknown model version: {version}")
        return self._versions[version]

    def restore(self, version: int) -> Array:
        """Load the weights stored for ``version`` as a NumPy array."""

        return bytes_to_weights(self.get(version).weights)

    def clear(self) -> None:
        """Remove all stored versions."""

        self._versions.clear()
        self._latest_version = None
