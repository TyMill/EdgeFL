from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for all models used in EdgeFL.
    All models must implement these methods.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_weights(self, weights: np.ndarray):
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass
