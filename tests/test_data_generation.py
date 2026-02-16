import numpy as np
import pandas as pd
import pytest

from edgefl.data.generator import generate_environmental_data


def test_generate_environmental_data_is_deterministic_with_seed() -> None:
    frame_one = generate_environmental_data(n_samples=32, seed=2024)
    frame_two = generate_environmental_data(n_samples=32, seed=2024)

    pd.testing.assert_frame_equal(frame_one, frame_two)


def test_generate_environmental_data_changes_with_different_seed() -> None:
    frame_one = generate_environmental_data(n_samples=32, seed=111)
    frame_two = generate_environmental_data(n_samples=32, seed=222)

    # Allow for the extremely unlikely scenario of equality by checking correlation
    assert not np.array_equal(frame_one.values, frame_two.values)


def test_generate_environmental_data_rejects_invalid_sample_count() -> None:
    with pytest.raises(ValueError):
        generate_environmental_data(n_samples=0)
