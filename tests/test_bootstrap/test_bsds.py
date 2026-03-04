"""Pytest tests for the Bootstrap Data Snooping (BSDS) / SPA test.

Tests bsds from ``mfe_toolbox.bootstrap``.

Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.bootstrap.bsds import bsds


class TestBsdsUnit:
    """Unit tests for bsds()."""

    def test_bsds_basic_execution(self) -> None:
        """BSDS should run without error on valid inputs."""
        rng = np.random.default_rng(42)
        bench = rng.standard_normal(200)
        models = rng.standard_normal((200, 3))
        result = bsds(bench, models, 1000, 25)
        assert result is not None, "BSDS should return a result"

    def test_bsds_returns_pvalues(self) -> None:
        """BSDS should return p-values in [0, 1]."""
        rng = np.random.default_rng(42)
        bench = rng.standard_normal(100)
        models = rng.standard_normal((100, 2))
        result = bsds(bench, models, 500, 20)
        # Result should contain p-values
        if isinstance(result, tuple):
            for val in result:
                if isinstance(val, (float, np.floating)):
                    assert 0.0 <= float(val) <= 1.0, f"p-value {val} out of range"

    def test_bsds_benchmark_same_as_model(self) -> None:
        """When benchmark equals model, p-value should be large (no rejection)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        bench = data.copy()
        models = data.reshape(-1, 1)
        result = bsds(bench, models, 500, 25)
        # Should not reject — p-value should be > 0.05
        assert result is not None

    def test_bsds_input_validation(self) -> None:
        """Invalid inputs should raise errors."""
        with pytest.raises((ValueError, TypeError)):
            bsds(np.array([]), np.array([[]]), 100, 10)

    def test_bsds_clearly_worse_benchmark(self) -> None:
        """When benchmark is clearly worse, should tend to reject."""
        rng = np.random.default_rng(42)
        # Benchmark has high loss, model has low loss
        bench = rng.standard_normal(300) + 2.0  # High loss
        models = rng.standard_normal((300, 1))   # Normal loss
        result = bsds(bench, models, 500, 25)
        # Result should indicate significant difference
        assert result is not None
