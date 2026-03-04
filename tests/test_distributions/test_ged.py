"""Pytest tests for the Generalized Error Distribution (GED) functions.

Tests gedcdf, gedinv, gedpdf, gedloglik, and gedrnd from
``mfe_toolbox.distributions``.

Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.distributions.gedcdf import gedcdf
from mfe_toolbox.distributions.gedinv import gedinv
from mfe_toolbox.distributions.gedpdf import gedpdf
from mfe_toolbox.distributions.gedloglik import gedloglik
from mfe_toolbox.distributions.gedrnd import gedrnd

from tests.conftest import ATOL, RTOL


class TestGedCdf:
    """Unit tests for gedcdf."""

    def test_gedcdf_zero(self) -> None:
        """CDF at x=0 should be 0.5 for symmetric GED."""
        p = gedcdf(np.array([0.0]), np.array([2.0]))
        npt.assert_allclose(p, 0.5, atol=ATOL)

    def test_gedcdf_monotonic(self) -> None:
        """CDF should be monotonically increasing."""
        x = np.linspace(-4, 4, 50)
        v = np.full_like(x, 2.0)
        p = gedcdf(x, v)
        assert np.all(np.diff(p) >= -1e-10), "CDF must be non-decreasing"

    def test_gedcdf_extreme_values(self) -> None:
        """CDF at extreme values should approach 0 and 1."""
        p_low = gedcdf(np.array([-100.0]), np.array([2.0]))
        p_high = gedcdf(np.array([100.0]), np.array([2.0]))
        npt.assert_allclose(p_low, 0.0, atol=1e-10)
        npt.assert_allclose(p_high, 1.0, atol=1e-10)

    def test_gedcdf_v_must_be_gt_1(self) -> None:
        """Shape parameter v < 1 should produce NaN."""
        p = gedcdf(np.array([0.0]), np.array([0.5]))
        assert np.isnan(p).all(), "v<1 should give NaN"

    def test_gedcdf_normal_case(self) -> None:
        """GED with v=2 is the standard normal distribution."""
        from scipy.stats import norm
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        v = np.full_like(x, 2.0)
        p_ged = gedcdf(x, v)
        p_norm = norm.cdf(x)
        npt.assert_allclose(p_ged, p_norm, atol=1e-4)


class TestGedInv:
    """Unit tests for gedinv."""

    def test_gedinv_median(self) -> None:
        """Quantile at p=0.5 should be 0 (symmetric)."""
        x = gedinv(np.array([0.5]), np.array([2.0]))
        npt.assert_allclose(x, 0.0, atol=ATOL)

    def test_gedinv_cdf_roundtrip(self) -> None:
        """CDF → Inv → CDF roundtrip should recover original probabilities."""
        x_orig = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        v = np.full_like(x_orig, 2.0)
        p = gedcdf(x_orig, v)
        x_recovered = gedinv(p, v)
        npt.assert_allclose(x_recovered, x_orig, atol=1e-5, rtol=1e-4)


class TestGedPdf:
    """Unit tests for gedpdf."""

    def test_gedpdf_positive(self) -> None:
        """PDF should be non-negative."""
        x = np.linspace(-3, 3, 20)
        v = np.full_like(x, 2.0)
        f = gedpdf(x, v)
        assert np.all(f >= 0), "PDF must be non-negative"

    def test_gedpdf_symmetric(self) -> None:
        """PDF should be symmetric around 0."""
        x = np.array([1.0, 2.0, 3.0])
        v = np.full_like(x, 2.0)
        f_pos = gedpdf(x, v)
        f_neg = gedpdf(-x, v)
        npt.assert_allclose(f_pos, f_neg, atol=ATOL)

    def test_gedpdf_peak_at_zero(self) -> None:
        """PDF should peak at x=0."""
        x = np.array([-1.0, 0.0, 1.0])
        v = np.full_like(x, 2.0)
        f = gedpdf(x, v)
        assert f[1] >= f[0] and f[1] >= f[2], "PDF should peak at x=0"


class TestGedLoglik:
    """Unit tests for gedloglik."""

    def test_gedloglik_returns_scalar(self) -> None:
        """Sum log-likelihood component should be a scalar."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        # gedloglik returns (sum_ll, per_obs_ll) tuple
        result = gedloglik(x, 0.0, 1.0, 2.0)
        if isinstance(result, tuple):
            ll_sum = result[0]
            assert np.isscalar(ll_sum) or isinstance(ll_sum, np.floating)
            assert np.isfinite(ll_sum)
        else:
            assert np.isfinite(result)

    def test_gedloglik_finite(self) -> None:
        """Log-likelihood should be finite for valid data."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        # gedloglik returns (sum_ll, per_obs_ll) tuple
        result = gedloglik(x, 0.0, 1.0, 2.0)
        ll = result[0] if isinstance(result, tuple) else result
        assert np.isfinite(ll), f"LL should be finite, got {ll}"


class TestGedRnd:
    """Unit tests for gedrnd."""

    def test_gedrnd_shape(self) -> None:
        """Random variates should have correct shape."""
        r = gedrnd(2.0, (100,))
        assert r.shape == (100,), f"Expected (100,), got {r.shape}"

    def test_gedrnd_mean_near_zero(self) -> None:
        """GED is symmetric — mean should be near 0."""
        r = gedrnd(2.0, (50000,))
        npt.assert_allclose(np.mean(r), 0.0, atol=0.05)


class TestGedFixtureParity:
    """Fixture parity tests for GED distribution."""

    def test_gedcdf_parity(self, distributions_fixture_dir) -> None:
        """Check CDF parity against MATLAB fixture."""
        from tests.conftest import load_fixture_npy
        p_expected = load_fixture_npy(distributions_fixture_dir, "gedcdf_p")
        x = load_fixture_npy(distributions_fixture_dir, "gedcdf_x")
        v = load_fixture_npy(distributions_fixture_dir, "gedcdf_v")
        p_actual = gedcdf(x, v)
        npt.assert_allclose(p_actual, p_expected, atol=ATOL, rtol=RTOL)

    def test_gedpdf_parity(self, distributions_fixture_dir) -> None:
        """Check PDF parity against MATLAB fixture."""
        from tests.conftest import load_fixture_npy
        f_expected = load_fixture_npy(distributions_fixture_dir, "gedpdf_f")
        x = load_fixture_npy(distributions_fixture_dir, "gedpdf_x")
        v = load_fixture_npy(distributions_fixture_dir, "gedpdf_v")
        f_actual = gedpdf(x, v)
        npt.assert_allclose(f_actual, f_expected, atol=ATOL, rtol=RTOL)
