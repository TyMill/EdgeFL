import numpy as np
from sklearn.linear_model import LinearRegression
from edgefl.models.base_model import BaseModel


class SklearnLinearModel(BaseModel):
    """
    Linear Regression model wrapper based on scikit-learn,
    implementing the BaseModel interface for federated learning use.
    """
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_weights(self) -> np.ndarray:
        return np.append(self.model.coef_, self.model.intercept_)

    def set_weights(self, weights: np.ndarray):
        self.model.coef_ = weights[:-1]
        self.model.intercept_ = weights[-1]

    def get_params(self) -> dict:
        return self.model.get_params()
