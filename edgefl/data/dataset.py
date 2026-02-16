"""Dataset utilities for handling generated environmental data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _train_test_split


def _ensure_2d(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one sample")
    return features


def _ensure_1d(target: np.ndarray) -> np.ndarray:
    if target.ndim != 1:
        raise ValueError("target must be a 1D array")
    if target.shape[0] == 0:
        raise ValueError("target must contain at least one sample")
    return target


@dataclass
class EdgeDataset:
    """Container for feature matrices and targets used by EdgeFL.

    Parameters
    ----------
    features:
        Two-dimensional array containing the feature matrix.
    target:
        One-dimensional array with target values aligned with ``features``.
    feature_names:
        Optional names associated with each feature column. When not provided the
        columns are named ``f0``, ``f1`` and so on.
    target_name:
        Optional name for the target column. Defaults to ``"aqi"`` for historical
        compatibility with the synthetic data generator.
    """

    features: np.ndarray
    target: np.ndarray
    feature_names: Optional[Sequence[str]] = None
    target_name: str = "aqi"

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        target = np.asarray(self.target)
        features = _ensure_2d(features)
        target = _ensure_1d(target)
        if features.shape[0] != target.shape[0]:
            raise ValueError("features and target must have the same number of samples")
        if not self.feature_names:
            self.feature_names = [f"f{i}" for i in range(features.shape[1])]
        else:
            self.feature_names = [str(name) for name in self.feature_names]
        if len(self.feature_names) != features.shape[1]:
            raise ValueError("feature_names length must match number of columns")
        if not self.target_name:
            raise ValueError("target_name cannot be empty")
        self.features = features
        self.target = target

    def __len__(self) -> int:
        return self.features.shape[0]

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        target_column: str = "aqi",
        *,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> "EdgeDataset":
        """Create a dataset from a pandas DataFrame.

        Parameters
        ----------
        dataframe:
            Source data containing both features and target column.
        target_column:
            Name of the column to treat as the target.
        feature_columns:
            Optional explicit list of feature column names. When omitted every column
            except ``target_column`` is used.
        """

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' not present in DataFrame"
            )
        if feature_columns is None:
            feature_columns = [col for col in dataframe.columns if col != target_column]
        missing = [col for col in feature_columns if col not in dataframe.columns]
        if missing:
            raise ValueError(f"Feature columns not present in DataFrame: {missing}")
        features = dataframe[feature_columns].to_numpy()
        target = dataframe[target_column].to_numpy()
        return cls(
            features=features,
            target=target,
            feature_names=feature_columns,
            target_name=target_column,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame."""

        data = {
            name: self.features[:, idx] for idx, name in enumerate(self.feature_names)
        }
        data[self.target_name] = self.target
        return pd.DataFrame(data)

    def train_test_split(
        self,
        test_size: float = 0.2,
        *,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> tuple["EdgeDataset", "EdgeDataset"]:
        """Split the dataset into train and test subsets."""

        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be a float between 0 and 1")
        X_train, X_test, y_train, y_test = _train_test_split(
            self.features,
            self.target,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        train = EdgeDataset(
            X_train, y_train, list(self.feature_names), self.target_name
        )
        test = EdgeDataset(X_test, y_test, list(self.feature_names), self.target_name)
        return train, test
