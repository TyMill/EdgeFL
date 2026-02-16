"""Synthetic data generation utilities for EdgeFL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, root_validator, validator

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


class SeasonalComponent(BaseModel):
    """Sinusoidal seasonality added to a variable."""

    amplitude: float = Field(
        0.0, ge=0.0, description="Seasonal amplitude applied as sine wave."
    )
    period: float = Field(
        365.0, gt=0.0, description="Number of samples representing a full season."
    )
    phase: float = Field(
        0.0, description="Phase offset (in samples) for the sine wave."
    )

    def generate(self, n_samples: int) -> np.ndarray:
        t = np.arange(n_samples, dtype=float)
        return self.amplitude * np.sin(2 * np.pi * (t + self.phase) / self.period)


class VariableParams(BaseModel):
    """Parameters describing a single environmental variable."""

    mean: float = Field(..., description="Average value of the variable.")
    std: float = Field(..., gt=0.0, description="Standard deviation of the variable.")
    minimum: Optional[float] = Field(
        None, description="Optional lower bound for the variable."
    )
    maximum: Optional[float] = Field(
        None, description="Optional upper bound for the variable."
    )
    seasonality: Optional[SeasonalComponent] = Field(
        None, description="Optional seasonality added on top of the base distribution."
    )

    @validator("maximum")
    def _validate_bounds(
        cls, maximum: Optional[float], values: Dict[str, Any]
    ) -> Optional[float]:
        minimum = values.get("minimum")
        if maximum is not None and minimum is not None and maximum <= minimum:
            raise ValueError("maximum must be greater than minimum")
        return maximum

    def clip(self, samples: np.ndarray) -> np.ndarray:
        if self.minimum is not None:
            samples = np.maximum(samples, self.minimum)
        if self.maximum is not None:
            samples = np.minimum(samples, self.maximum)
        return samples


class TargetParams(BaseModel):
    """Linear target generation configuration."""

    coefficients: Dict[str, float] = Field(
        default_factory=lambda: {"temperature": 0.5, "humidity": -0.2, "pm25": 0.3},
        description="Per-variable contribution to the target value.",
    )
    intercept: float = Field(
        0.0, description="Constant offset added to the target value."
    )
    noise_std: float = Field(
        5.0, ge=0.0, description="Standard deviation of additive Gaussian noise."
    )


class RegionProfile(BaseModel):
    """Complete description of a regional environmental profile."""

    name: Optional[str] = Field(None, description="Human readable name of the profile.")
    n_samples: Optional[int] = Field(
        None, gt=0, description="Default number of samples to generate."
    )
    variables: Dict[str, VariableParams] = Field(
        default_factory=dict,
        description="Mapping of variable names to distribution parameters.",
    )
    correlations: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Pairwise correlation coefficients between variables (range [-1, 1]).",
    )
    target: TargetParams = Field(default_factory=TargetParams)
    weight: float = Field(
        1.0, gt=0.0, description="Relative weight used for mixture sampling."
    )

    @root_validator(skip_on_failure=True)
    def _validate_correlations(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        variables: Dict[str, VariableParams] = values.get("variables", {})
        correlations: Dict[str, Dict[str, float]] = values.get("correlations", {})
        for left, mapping in correlations.items():
            if left not in variables:
                raise ValueError(f"Correlation defined for unknown variable '{left}'")
            for right, coefficient in mapping.items():
                if right not in variables:
                    raise ValueError(
                        f"Correlation defined for unknown variable '{right}'"
                    )
                if not -1.0 <= coefficient <= 1.0:
                    raise ValueError("Correlation coefficients must be within [-1, 1]")
        return values


def _ensure_region_profile(
    *,
    profile: Optional[Union[RegionProfile, Mapping[str, Any]]],
    n_samples: int,
    region_factor: float,
    variables: Optional[Mapping[str, Union[Mapping[str, Any], VariableParams]]],
    correlations: Optional[Mapping[str, Mapping[str, float]]],
    target: Optional[Union[Mapping[str, Any], TargetParams]],
) -> RegionProfile:
    if profile is not None:
        if not isinstance(profile, RegionProfile):
            profile = RegionProfile.parse_obj(profile)
        update: Dict[str, Any] = {}
        if variables:
            update["variables"] = {
                **profile.variables,
                **_parse_variable_params(variables),
            }
        if correlations:
            update["correlations"] = {
                **profile.correlations,
                **_normalize_correlation_mapping(correlations),
            }
        if target is not None:
            update["target"] = TargetParams.parse_obj(target)
        if update:
            profile = profile.copy(update=update)
        return profile

    default_variables: Dict[str, VariableParams] = {
        "temperature": VariableParams(
            mean=15.0 * region_factor,
            std=4.0 * region_factor,
            seasonality=SeasonalComponent(amplitude=2.5 * region_factor, period=365.0),
        ),
        "humidity": VariableParams(
            mean=50.0, std=8.0 * region_factor, minimum=0.0, maximum=100.0
        ),
        "pm25": VariableParams(
            mean=max(5.0, 15.0 * (2 - region_factor)),
            std=7.0,
            minimum=0.0,
        ),
    }
    if variables:
        default_variables.update(_parse_variable_params(variables))

    default_correlations = {"temperature": {"pm25": 0.45}}
    if correlations:
        default_correlations.update(_normalize_correlation_mapping(correlations))

    default_target = TargetParams()
    if target is not None:
        default_target = TargetParams.parse_obj(target)

    return RegionProfile(
        name="default",
        n_samples=n_samples,
        variables=default_variables,
        correlations=default_correlations,
        target=default_target,
    )


def _parse_variable_params(
    variables: Mapping[str, Union[Mapping[str, Any], VariableParams]],
) -> Dict[str, VariableParams]:
    parsed: Dict[str, VariableParams] = {}
    for name, params in variables.items():
        if isinstance(params, VariableParams):
            parsed[name] = params
        else:
            parsed[name] = VariableParams.parse_obj(params)
    return parsed


def _normalize_correlation_mapping(
    correlations: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for left, mapping in correlations.items():
        normalized.setdefault(left, {})
        for right, coefficient in mapping.items():
            normalized[left][right] = float(coefficient)
    return normalized


def _create_rng(
    seed: Optional[int], rng: Optional[np.random.Generator] = None
) -> np.random.Generator:
    if rng is not None:
        return rng
    bit_generator = np.random.PCG64(seed)
    return np.random.Generator(bit_generator)


def _build_covariance_matrix(
    variables: Dict[str, VariableParams], correlations: Dict[str, Dict[str, float]]
) -> Tuple[np.ndarray, Sequence[str]]:
    names = list(variables.keys())
    stds = np.array([variables[name].std for name in names])
    covariance = np.outer(stds, stds)
    correlation_matrix = np.eye(len(names))
    for i, left in enumerate(names):
        correlation_matrix[i, i] = 1.0
        if left not in correlations:
            continue
        for right, coefficient in correlations[left].items():
            if right not in variables:
                raise ValueError(f"Correlation defined for unknown variable '{right}'")
            j = names.index(right)
            correlation_matrix[i, j] = coefficient
            correlation_matrix[j, i] = coefficient
    covariance *= correlation_matrix
    return covariance, names


def _generate_variable_samples(
    rng: np.random.Generator,
    n_samples: int,
    variables: Dict[str, VariableParams],
    correlations: Dict[str, Dict[str, float]],
) -> Dict[str, np.ndarray]:
    covariance, names = _build_covariance_matrix(variables, correlations)
    means = np.array([variables[name].mean for name in names])
    multivariate_samples = rng.multivariate_normal(
        mean=means, cov=covariance, size=n_samples
    )
    samples: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        column = multivariate_samples[:, idx]
        params = variables[name]
        if params.seasonality is not None:
            column = column + params.seasonality.generate(n_samples)
        samples[name] = params.clip(column)
    return samples


def generate_environmental_data(
    n_samples: int = 100,
    region_factor: float = 1.0,
    seed: Optional[int] = None,
    *,
    profile: Optional[Union[RegionProfile, Mapping[str, Any]]] = None,
    variables: Optional[Mapping[str, Union[Mapping[str, Any], VariableParams]]] = None,
    correlations: Optional[Mapping[str, Mapping[str, float]]] = None,
    target: Optional[Union[Mapping[str, Any], TargetParams]] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Generate correlated environmental observations.

    Parameters
    ----------
    n_samples:
        Number of observations to generate when no profile-specific value is provided.
    region_factor:
        Backwards-compatible scaling factor applied to default parameter values. It is
        ignored when a custom profile is supplied.
    seed:
        Optional seed used to initialise a :class:`numpy.random.Generator` with
        :class:`numpy.random.PCG64`.
    profile:
        Region profile describing full distribution settings. If provided it overrides
        ``n_samples`` and ``region_factor`` defaults.
    variables:
        Optional per-variable overrides applied on top of the chosen profile or the
        defaults. Each item should provide ``mean``, ``std`` and optional bounds and
        seasonality definition.
    correlations:
        Optional overrides for pairwise correlations between variables. Mapping is
        expressed as ``{"left": {"right": coefficient}}`` with coefficients between
        -1 and 1.
    target:
        Optional override for the linear target generation parameters.
    rng:
        Optional reusable random number generator instance.

    Returns
    -------
    pandas.DataFrame
        Generated observations including the computed ``aqi`` target column.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if region_factor <= 0:
        raise ValueError("region_factor must be greater than zero")

    base_profile = _ensure_region_profile(
        profile=profile,
        n_samples=n_samples,
        region_factor=region_factor,
        variables=variables,
        correlations=correlations,
        target=target,
    )
    effective_samples = base_profile.n_samples or n_samples

    rng = _create_rng(seed, rng=rng)
    variable_samples = _generate_variable_samples(
        rng=rng,
        n_samples=effective_samples,
        variables=base_profile.variables,
        correlations=base_profile.correlations,
    )

    noise = np.zeros(effective_samples)
    if base_profile.target.noise_std > 0:
        noise = rng.normal(0.0, base_profile.target.noise_std, size=effective_samples)

    target_values = np.full(effective_samples, base_profile.target.intercept)
    for name, coefficient in base_profile.target.coefficients.items():
        if name not in variable_samples:
            raise ValueError(f"Target coefficient references unknown variable '{name}'")
        target_values += coefficient * variable_samples[name]
    target_values += noise

    data = {**variable_samples, "aqi": target_values}
    return pd.DataFrame(data)


def load_region_profile(path: Union[str, Path]) -> RegionProfile:
    """Load a region profile definition from a JSON or YAML file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:  # pragma: no cover - optional dependency
            raise ImportError("PyYAML is required to load YAML profiles")
        with path.open("r", encoding="utf8") as handle:
            payload = yaml.safe_load(handle)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf8") as handle:
            payload = json.load(handle)
    else:
        raise ValueError(f"Unsupported profile format: {path.suffix}")

    return RegionProfile.parse_obj(payload)


def load_region_profiles(directory: Union[str, Path]) -> Dict[str, RegionProfile]:
    """Load all profiles defined in a directory."""

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(directory)

    profiles: Dict[str, RegionProfile] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        profile = load_region_profile(path)
        profiles[path.stem] = profile
    return profiles


def generate_mixed_environmental_data(
    profiles: Sequence[Union[RegionProfile, Mapping[str, Any]]],
    n_samples: int,
    *,
    weights: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate data sampled from a weighted mixture of region profiles.

    Parameters
    ----------
    profiles:
        Iterable of profiles contributing to the mixture.
    n_samples:
        Total number of samples to draw across all profiles.
    weights:
        Optional sampling weights aligned with ``profiles``. When omitted the
        ``weight`` attribute of each profile is used.
    seed:
        Optional seed controlling the assignment of samples to profiles.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not profiles:
        raise ValueError("At least one profile must be provided")

    parsed_profiles = [
        p if isinstance(p, RegionProfile) else RegionProfile.parse_obj(p)
        for p in profiles
    ]
    total_weight = (
        sum(weights) if weights is not None else sum(p.weight for p in parsed_profiles)
    )
    if total_weight <= 0:
        raise ValueError("Mixture weights must sum to a positive value")

    if weights is None:
        weights = [p.weight for p in parsed_profiles]
    probabilities = np.array(weights, dtype=float) / total_weight

    rng = _create_rng(seed)
    assignments = rng.choice(len(parsed_profiles), size=n_samples, p=probabilities)

    frames: list[pd.DataFrame] = []
    for index, profile in enumerate(parsed_profiles):
        mask = assignments == index
        count = int(np.sum(mask))
        if count == 0:
            continue
        df = generate_environmental_data(
            n_samples=count,
            profile=profile,
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        df.insert(0, "profile", profile.name or f"profile_{index}")
        frames.append(df)

    if not frames:
        raise RuntimeError("No samples were generated; check the mixture configuration")

    return pd.concat(frames, ignore_index=True)


def generate_clients_data(
    n_clients: int = 5,
    n_samples: int = 100,
    *,
    profiles: Optional[Sequence[Union[RegionProfile, Mapping[str, Any]]]] = None,
    profile_directory: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate datasets for multiple clients.

    Parameters
    ----------
    n_clients:
        Number of client datasets to generate.
    n_samples:
        Number of samples per client when profiles do not specify ``n_samples``.
    profiles:
        Explicit list of profiles used for sampling. When fewer profiles than clients
        are provided they will be cycled.
    profile_directory:
        Optional directory containing YAML/JSON profile definitions. Profiles found in
        the directory are used if ``profiles`` is not provided.
    seed:
        Seed used to initialise the generator that selects profile order.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of datasets represented as ``(features, target)`` tuples.
    """

    if n_clients <= 0:
        raise ValueError("n_clients must be a positive integer")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    rng = _create_rng(seed)

    if profiles is None and profile_directory is not None:
        loaded_profiles = load_region_profiles(profile_directory)
        profiles = list(loaded_profiles.values())

    datasets: list[tuple[np.ndarray, np.ndarray]] = []

    if profiles:
        parsed_profiles = [
            p if isinstance(p, RegionProfile) else RegionProfile.parse_obj(p)
            for p in profiles
        ]
        for client_index in range(n_clients):
            profile = parsed_profiles[client_index % len(parsed_profiles)]
            df = generate_environmental_data(
                n_samples=profile.n_samples or n_samples,
                profile=profile,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
            X = df[[col for col in df.columns if col not in {"aqi", "profile"}]].values
            y = df["aqi"].values
            datasets.append((X, y))
        return datasets

    for client_index in range(n_clients):
        region_factor = 1.0 + (client_index * 0.1)
        df = generate_environmental_data(
            n_samples=n_samples,
            region_factor=region_factor,
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        X = df[["temperature", "humidity", "pm25"]].values
        y = df["aqi"].values
        datasets.append((X, y))

    return datasets
