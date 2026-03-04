"""Pytest tests for the Ljung-Box serial correlation test.

Tests ljungbox from ``mfe_toolbox.tests``.

Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.tests.ljungbox import ljungbox
from tests.conftest import ATOL, RTOL


class TestLjungboxUnit:
    """Unit tests for ljungbox()."""

    def test_ljungbox_white_noise(self) -> None:
        """White noise should not reject (p-values > 0.05 in most lags)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        q, pval = ljungbox(x, 10)
        assert q.shape == (10,), f"Expected shape (10,), got {q.shape}"
        assert pval.shape == (10,), f"Expected shape (10,), got {pval.shape}"
        # For white noise, most p-values should be > 0.05
        assert np.sum(pval > 0.05) >= 5, "Most p-values should be > 0.05 for white noise"

    def test_ljungbox_serial_correlation_detects(self) -> None:
        """AR(1) data should show significant serial correlation."""
        rng = np.random.default_rng(42)
        T = 1000
        x = np.zeros(T)
        e = rng.standard_normal(T)
        for t in range(1, T):
            x[t] = 0.8 * x[t - 1] + e[t]
        q, pval = ljungbox(x, 10)
        # AR(1) with phi=0.8 should be detected — p-value at lag 1 should be small
        assert pval[0] < 0.05, f"p-value at lag 1 should be < 0.05, got {pval[0]}"

    def test_ljungbox_q_stats_positive(self) -> None:
        """Q statistics should be non-negative."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        q, pval = ljungbox(x, 5)
        assert np.all(q >= 0), "Q statistics must be non-negative"

    def test_ljungbox_pvalues_in_range(self) -> None:
        """p-values should be in [0, 1]."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        q, pval = ljungbox(x, 10)
        assert np.all(pval >= 0.0) and np.all(pval <= 1.0), "p-values must be in [0, 1]"

    def test_ljungbox_q_stats_nondecreasing(self) -> None:
        """Q statistics should be non-decreasing across lags."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(300)
        q, pval = ljungbox(x, 15)
        assert np.all(np.diff(q) >= -1e-10), "Q stats should be non-decreasing"

    def test_ljungbox_invalid_lags_raises(self) -> None:
        """Non-positive or too-large lags should raise ValueError."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError):
            ljungbox(x, 0)
        with pytest.raises(ValueError):
            ljungbox(x, 5)  # lags >= T should fail

    def test_ljungbox_1d_input(self) -> None:
        """Should accept and process 1-D input."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        q, pval = ljungbox(x, 5)
        assert q.shape == (5,)

    def test_ljungbox_single_lag(self) -> None:
        """Should work with lags=1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        q, pval = ljungbox(x, 1)
        assert q.shape == (1,) and pval.shape == (1,)
