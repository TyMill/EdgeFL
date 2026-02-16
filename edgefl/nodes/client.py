"""Client node abstractions for EdgeFL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from edgefl.models.base_model import BaseModel

Array = NDArray[np.float64]


@dataclass
class ClientNode:
    """Representation of an edge client participating in federated learning.

    Parameters
    ----------
    client_id:
        Unique identifier of the client.
    X:
        Training features local to the client.
    y:
        Training labels local to the client.
    model:
        Instance of :class:`~edgefl.models.base_model.BaseModel` used for
        training. If not provided, ``model_factory`` must be specified.
    model_factory:
        Callable returning a new :class:`BaseModel` instance. This is helpful
        when the same model architecture should be replicated across clients
        without sharing stateful objects.
    metadata:
        Optional dictionary with arbitrary client metadata.
    """

    client_id: Union[str, int]
    X: Array
    y: Array
    model: Optional[BaseModel] = None
    model_factory: Optional[Callable[[], BaseModel]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model is None:
            if self.model_factory is None:
                raise ValueError(
                    "Either 'model' or 'model_factory' must be provided for ClientNode."
                )
            self.model = self.model_factory()
        if not isinstance(self.model, BaseModel):
            raise TypeError("model must implement BaseModel")
        self.num_examples: int = int(self.y.shape[0])

    def train(self) -> Array:
        """Train the local model on the client's dataset.

        Returns
        -------
        numpy.ndarray
            The trained model weights.
        """

        self.model.fit(self.X, self.y)
        return self.model.get_weights()

    def get_weights(self) -> Array:
        """Return the current model weights."""

        return self.model.get_weights()

    def set_weights(self, weights: Array) -> None:
        """Update the local model with the provided weights."""

        self.model.set_weights(weights)

    def evaluate(self, X_val: Array, y_val: Array) -> float:
        """Evaluate the local model using mean squared error.

        Parameters
        ----------
        X_val, y_val:
            Validation dataset.

        Returns
        -------
        float
            Mean squared error on the validation set.
        """

        predictions = self.model.predict(X_val)
        return float(np.mean(np.square(predictions - y_val)))

    def update_data(self, X: Array, y: Array) -> None:
        """Replace the local dataset with new samples."""

        self.X = X
        self.y = y
        self.num_examples = int(y.shape[0])
