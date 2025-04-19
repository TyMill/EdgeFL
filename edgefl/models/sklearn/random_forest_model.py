import numpy as np
from sklearn.ensemble import RandomForestRegressor
from edgefl.models.base_model import BaseModel


class SklearnRandomForestModel(BaseModel):
    """
    Random Forest model wrapper using scikit-learn,
    compatible with EdgeFL's BaseModel interface.
    """
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_weights(self) -> np.ndarray:
        # Random Forests are not weight-based, return a dummy
        return np.array([0.0])

    def set_weights(self, weights: np.ndarray):
        # Not applicable, so we raise a warning or ignore
        pass

    def get_params(self) -> dict:
        return self.model.get_params()

