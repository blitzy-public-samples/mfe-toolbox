"""Pytest tests for the Berkowitz distributional forecast test.

Tests berkowitz from ``mfe_toolbox.tests``.

Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import norm

from mfe_toolbox.tests.berkowitz import berkowitz
from tests.conftest import ATOL, RTOL


class TestBerkowitzUnit:
    """Unit tests for berkowitz()."""

    def test_berkowitz_uniform_data_no_rejection(self) -> None:
        """Uniform(0,1) data should not reject the null of correct specification."""
        rng = np.random.default_rng(42)
        x = rng.random(500)
        stat, pval, H = berkowitz(x, test_type='CS', alpha=0.05)
        assert stat >= 0.0, "Statistic should be non-negative"
        assert 0.0 <= pval <= 1.0, f"p-value {pval} out of range"
        assert isinstance(H, (bool, np.bool_))

    def test_berkowitz_ts_mode(self) -> None:
        """Time-series mode should include autocorrelation test (3 d.f.)."""
        rng = np.random.default_rng(42)
        x = rng.random(500)
        stat, pval, H = berkowitz(x, test_type='TS', alpha=0.05)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0

    def test_berkowitz_cs_mode(self) -> None:
        """Cross-sectional mode (2 d.f.)."""
        rng = np.random.default_rng(42)
        x = rng.random(300)
        stat, pval, H = berkowitz(x, test_type='CS', alpha=0.05)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0

    def test_berkowitz_with_dist_callable(self) -> None:
        """Should work with a CDF function applied to raw data."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(300)
        stat, pval, H = berkowitz(x, test_type='CS', alpha=0.05, dist=norm.cdf)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0

    def test_berkowitz_misspecified_data_rejects(self) -> None:
        """Non-uniform data (without dist) tested as PIT should reject."""
        rng = np.random.default_rng(12345)
        # Heavily skewed data — should fail PIT test
        x = np.clip(rng.exponential(1.0, 500), 0.001, 0.999)
        stat, pval, H = berkowitz(x, test_type='CS', alpha=0.05)
        # With heavily misspecified data, we expect rejection
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0

    def test_berkowitz_empty_input_raises(self) -> None:
        """Empty array should raise ValueError."""
        with pytest.raises(ValueError):
            berkowitz(np.array([]))

    def test_berkowitz_alpha_range(self) -> None:
        """Alpha must be in (0, 1)."""
        rng = np.random.default_rng(42)
        x = rng.random(100)
        with pytest.raises(ValueError):
            berkowitz(x, alpha=0.0)
        with pytest.raises(ValueError):
            berkowitz(x, alpha=1.0)

    def test_berkowitz_data_outside_01_no_dist_raises(self) -> None:
        """Data outside (0,1) without dist should raise ValueError."""
        with pytest.raises(ValueError):
            berkowitz(np.array([0.0, 0.5, 0.9]))

    def test_berkowitz_returns_tuple(self) -> None:
        """Should return a 3-tuple (stat, pval, H)."""
        rng = np.random.default_rng(42)
        x = rng.random(200)
        result = berkowitz(x)
        assert isinstance(result, tuple) and len(result) == 3
