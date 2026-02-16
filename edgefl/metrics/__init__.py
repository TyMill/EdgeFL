"""Metrics utilities for EdgeFL."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import numpy as np


def communication_cost(updates: Iterable[Any]) -> float:
    """Estimate communication cost for a round.

    The proxy sums reported example counts or payload sizes.
    """

    total = 0.0
    for update in updates:
        if update is None:
            continue
        if hasattr(update, "num_examples"):
            total += float(update.num_examples)
            continue
        if isinstance(update, Mapping) and "num_examples" in update:
            total += float(update["num_examples"])
            continue
        if isinstance(update, Real):
            total += float(update)
            continue
        if isinstance(update, np.ndarray):
            total += float(update.size)
            continue
        if isinstance(update, Iterable):
            total += float(len(list(update)))
            continue
    return float(total)


def participation_rate(history: Iterable[Any]) -> float:
    """Compute participation rate from a history of booleans or mappings."""

    participated: list[bool] = []
    for entry in history:
        if isinstance(entry, Mapping) and "participated" in entry:
            participated.append(bool(entry["participated"]))
        else:
            participated.append(bool(entry))
    if not participated:
        return 0.0
    return float(np.mean(participated))


def jains_fairness(participation_counts: Iterable[Real]) -> float:
    """Compute Jain's fairness index for participation counts."""

    counts = np.array([float(count) for count in participation_counts], dtype=float)
    if counts.size == 0:
        return 0.0
    numerator = float(np.square(np.sum(counts)))
    denominator = float(counts.size * np.sum(np.square(counts)))
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def system_utility(
    *,
    participation_rate: float,
    fairness: float,
    validation_score: float | None = None,
    communication_cost: float | None = None,
    communication_cost_weight: float = 0.0,
) -> float:
    """Compute a simple system utility signal."""

    utility = float(participation_rate * fairness)
    if validation_score is not None:
        utility += float(validation_score)
    if communication_cost is not None:
        utility -= float(communication_cost_weight * communication_cost)
    return float(utility)


__all__ = [
    "communication_cost",
    "participation_rate",
    "jains_fairness",
    "system_utility",
]
