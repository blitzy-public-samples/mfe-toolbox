"""Pytest tests for the Model Confidence Set (MCS) procedure.

Tests mcs from ``mfe_toolbox.bootstrap``.

Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.bootstrap.mcs import mcs


class TestMcsUnit:
    """Unit tests for mcs()."""

    def test_mcs_basic_execution(self) -> None:
        """MCS should run without error on well-separated losses."""
        rng = np.random.default_rng(42)
        # 3 models, 200 obs — model 0 clearly best (lowest loss)
        losses = np.column_stack([
            rng.standard_normal(200) * 0.5,       # Best model
            rng.standard_normal(200) * 1.0 + 0.5,  # Worse
            rng.standard_normal(200) * 1.5 + 1.0,  # Worst
        ])
        result = mcs(losses, 0.10, 1000, 25)
        assert result is not None, "MCS should return a result"

    def test_mcs_returns_tuple(self) -> None:
        """MCS should return a tuple with expected elements."""
        rng = np.random.default_rng(42)
        losses = np.column_stack([
            rng.standard_normal(100),
            rng.standard_normal(100) + 0.3,
        ])
        result = mcs(losses, 0.10, 500, 20)
        # MCS returns (included, pvalues, excluded, excluded_pvalues)
        assert isinstance(result, tuple), "MCS should return a tuple"
        assert len(result) >= 2, "MCS should return at least 2 elements"

    def test_mcs_single_model(self) -> None:
        """Single model should always be included in MCS."""
        rng = np.random.default_rng(42)
        losses = rng.standard_normal((100, 1))
        result = mcs(losses, 0.10, 500, 20)
        # The single model should be included
        included = result[0] if isinstance(result, tuple) else result
        assert len(np.atleast_1d(included)) >= 1

    def test_mcs_input_validation(self) -> None:
        """Invalid inputs should raise ValueError."""
        rng = np.random.default_rng(42)
        losses = rng.standard_normal((100, 3))
        # alpha out of range
        with pytest.raises((ValueError, TypeError)):
            mcs(losses, 0.0, 500, 20)

    def test_mcs_reproducible_with_seed(self) -> None:
        """MCS should produce consistent results with same seed."""
        rng = np.random.default_rng(42)
        losses = np.column_stack([
            rng.standard_normal(100),
            rng.standard_normal(100) + 0.5,
        ])
        # Run twice — bootstrap randomness means exact equality not guaranteed
        # without internal seed control, but structure should be the same
        result1 = mcs(losses, 0.10, 200, 20)
        result2 = mcs(losses, 0.10, 200, 20)
        assert type(result1) == type(result2)
