"""Pytest tests for Hansen's Skewed Student's t distribution functions.

Tests skewtcdf, skewtinv, skewtpdf, skewtloglik from
``mfe_toolbox.distributions``.

Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.distributions.skewtcdf import skewtcdf
from mfe_toolbox.distributions.skewtinv import skewtinv
from mfe_toolbox.distributions.skewtpdf import skewtpdf
from mfe_toolbox.distributions.skewtloglik import skewtloglik

from tests.conftest import ATOL, RTOL


class TestSkewtCdf:
    """Unit tests for skewtcdf."""

    def test_skewtcdf_symmetric_at_zero(self) -> None:
        """CDF(0, v, lambda=0) should be 0.5 (symmetric case)."""
        p = skewtcdf(np.array([0.0]), np.array([7.0]), np.array([0.0]))
        npt.assert_allclose(p, 0.5, atol=ATOL)

    def test_skewtcdf_monotonic(self) -> None:
        """CDF should be monotonically increasing."""
        x = np.linspace(-4, 4, 50)
        v = np.full_like(x, 7.0)
        lam = np.full_like(x, 0.3)
        p = skewtcdf(x, v, lam)
        assert np.all(np.diff(p) >= -1e-10), "CDF must be non-decreasing"

    def test_skewtcdf_skew_positive(self) -> None:
        """Positive lambda should affect CDF(0) relative to symmetric case."""
        x = np.array([0.0])
        v = np.array([7.0])
        p_zero = skewtcdf(x, v, np.array([0.0]))
        p_pos = skewtcdf(x, v, np.array([0.5]))
        # With positive skew (Hansen's parameterization), CDF(0) differs from 0.5
        # The key property: CDF should still be a valid probability
        assert 0.0 < p_pos.item() < 1.0
        # And it should differ from the symmetric case
        assert abs(p_pos.item() - p_zero.item()) > 0.01

    def test_skewtcdf_v_must_be_gt_2(self) -> None:
        """v <= 2 should produce NaN."""
        p = skewtcdf(np.array([0.0]), np.array([2.0]), np.array([0.0]))
        assert np.isnan(p).all(), "v=2 should give NaN"

    def test_skewtcdf_lambda_range(self) -> None:
        """Lambda values within (-1, 1) should produce valid CDF values."""
        p = skewtcdf(np.array([0.0]), np.array([5.0]), np.array([0.5]))
        assert 0.0 <= p.item() <= 1.0, "CDF should be in [0, 1]"
        p_neg = skewtcdf(np.array([0.0]), np.array([5.0]), np.array([-0.5]))
        assert 0.0 <= p_neg.item() <= 1.0, "CDF should be in [0, 1]"


class TestSkewtInv:
    """Unit tests for skewtinv."""

    def test_skewtinv_median_symmetric(self) -> None:
        """Quantile at p=0.5, lambda=0 should be 0."""
        x = skewtinv(np.array([0.5]), np.array([7.0]), np.array([0.0]))
        npt.assert_allclose(x, 0.0, atol=1e-5)

    def test_skewtinv_cdf_roundtrip(self) -> None:
        """CDF → Inv roundtrip."""
        x_orig = np.array([-1.0, 0.0, 1.0])
        v = np.full_like(x_orig, 7.0)
        lam = np.full_like(x_orig, 0.0)
        p = skewtcdf(x_orig, v, lam)
        x_recovered = skewtinv(p, v, lam)
        npt.assert_allclose(x_recovered, x_orig, atol=1e-5, rtol=1e-4)


class TestSkewtPdf:
    """Unit tests for skewtpdf."""

    def test_skewtpdf_positive(self) -> None:
        """PDF should be non-negative."""
        x = np.linspace(-3, 3, 20)
        v = np.full_like(x, 7.0)
        lam = np.full_like(x, 0.0)
        f = skewtpdf(x, v, lam)
        assert np.all(f >= 0), "PDF must be non-negative"

    def test_skewtpdf_symmetric_case(self) -> None:
        """PDF should be symmetric when lambda=0."""
        x = np.array([1.0, 2.0])
        v = np.full_like(x, 7.0)
        lam = np.zeros_like(x)
        f_pos = skewtpdf(x, v, lam)
        f_neg = skewtpdf(-x, v, lam)
        npt.assert_allclose(f_pos, f_neg, atol=ATOL)


class TestSkewtLoglik:
    """Unit tests for skewtloglik."""

    def test_skewtloglik_returns_scalar(self) -> None:
        """Sum log-likelihood component should be a scalar."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        # skewtloglik returns (sum_ll, per_obs_ll) tuple
        result = skewtloglik(x, 0.0, 1.0, 7.0, 0.0)
        if isinstance(result, tuple):
            ll_sum = result[0]
            assert np.isscalar(ll_sum) or isinstance(ll_sum, np.floating)
            assert np.isfinite(ll_sum)
        else:
            assert np.isfinite(result)

    def test_skewtloglik_finite(self) -> None:
        """Log-likelihood should be finite for valid data."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        # skewtloglik returns (sum_ll, per_obs_ll) tuple
        result = skewtloglik(x, 0.0, 1.0, 7.0, 0.0)
        ll = result[0] if isinstance(result, tuple) else result
        assert np.isfinite(ll), f"LL should be finite, got {ll}"


class TestSkewtFixtureParity:
    """Fixture parity tests for skewed t distribution."""

    def test_skewtcdf_parity(self, distributions_fixture_dir) -> None:
        """Check CDF parity against MATLAB fixture."""
        from tests.conftest import load_fixture_npy
        p_expected = load_fixture_npy(distributions_fixture_dir, "skewtcdf_p")
        x = load_fixture_npy(distributions_fixture_dir, "skewtcdf_x")
        v = load_fixture_npy(distributions_fixture_dir, "skewtcdf_v")
        lam = load_fixture_npy(distributions_fixture_dir, "skewtcdf_lambda")
        p_actual = skewtcdf(x, v, lam)
        npt.assert_allclose(p_actual, p_expected, atol=ATOL, rtol=RTOL)
