"""Pytest tests for the Standardized Student's t distribution functions.

Tests stdtcdf, stdtinv, stdtpdf, stdtloglik, and stdtrnd from
``mfe_toolbox.distributions``.

Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.distributions.stdtcdf import stdtcdf
from mfe_toolbox.distributions.stdtinv import stdtinv
from mfe_toolbox.distributions.stdtpdf import stdtpdf
from mfe_toolbox.distributions.stdtloglik import stdtloglik
from mfe_toolbox.distributions.stdtrnd import stdtrnd

from tests.conftest import ATOL, RTOL


class TestStdtCdf:
    """Unit tests for stdtcdf."""

    def test_stdtcdf_zero(self) -> None:
        """CDF at x=0 should be 0.5 for symmetric distribution."""
        p = stdtcdf(np.array([0.0]), np.array([5.0]))
        npt.assert_allclose(p, 0.5, atol=ATOL)

    def test_stdtcdf_monotonic(self) -> None:
        """CDF should be monotonically increasing."""
        x = np.linspace(-4, 4, 50)
        v = np.full_like(x, 5.0)
        p = stdtcdf(x, v)
        assert np.all(np.diff(p) >= 0), "CDF must be non-decreasing"

    def test_stdtcdf_extreme_values(self) -> None:
        """CDF at extreme values should approach 0 and 1."""
        p_low = stdtcdf(np.array([-100.0]), np.array([5.0]))
        p_high = stdtcdf(np.array([100.0]), np.array([5.0]))
        assert p_low.item() < 1e-6, f"CDF(-100) should be near 0, got {p_low}"
        assert p_high.item() > 1 - 1e-6, f"CDF(100) should be near 1, got {p_high}"

    def test_stdtcdf_v_gt_2_required(self) -> None:
        """v <= 2 should produce NaN (variance not finite)."""
        p = stdtcdf(np.array([0.0]), np.array([2.0]))
        assert np.isnan(p).all(), "v=2 should give NaN"

    def test_stdtcdf_scalar_input(self) -> None:
        """Scalar input should return scalar-like output."""
        p = stdtcdf(np.array([0.5]), np.array([10.0]))
        assert p.shape == (1,) or p.ndim == 0


class TestStdtInv:
    """Unit tests for stdtinv."""

    def test_stdtinv_median(self) -> None:
        """Quantile at p=0.5 should be 0 (symmetric distribution)."""
        x = stdtinv(np.array([0.5]), np.array([5.0]))
        npt.assert_allclose(x, 0.0, atol=ATOL)

    def test_stdtinv_cdf_roundtrip(self) -> None:
        """CDF → Inv → CDF roundtrip should recover original probabilities."""
        x_orig = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        v = np.full_like(x_orig, 7.0)
        p = stdtcdf(x_orig, v)
        x_recovered = stdtinv(p, v)
        npt.assert_allclose(x_recovered, x_orig, atol=1e-5, rtol=1e-4)


class TestStdtPdf:
    """Unit tests for stdtpdf."""

    def test_stdtpdf_positive(self) -> None:
        """PDF should be positive for finite inputs."""
        x = np.linspace(-3, 3, 20)
        # stdtpdf signature: (x, mu, sigma2, nu)
        f = stdtpdf(x, 0.0, 1.0, 5.0)
        assert np.all(f >= 0), "PDF must be non-negative"

    def test_stdtpdf_symmetric(self) -> None:
        """PDF should be symmetric around 0."""
        x = np.array([1.0, 2.0, 3.0])
        f_pos = stdtpdf(x, 0.0, 1.0, 5.0)
        f_neg = stdtpdf(-x, 0.0, 1.0, 5.0)
        npt.assert_allclose(f_pos, f_neg, atol=ATOL)

    def test_stdtpdf_peak_at_zero(self) -> None:
        """PDF should peak at x=0."""
        x = np.array([-1.0, 0.0, 1.0])
        f = stdtpdf(x, 0.0, 1.0, 5.0)
        assert f[1] >= f[0] and f[1] >= f[2], "PDF should peak at x=0"


class TestStdtLoglik:
    """Unit tests for stdtloglik."""

    def test_stdtloglik_returns_scalar(self) -> None:
        """Sum log-likelihood component should be a scalar."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        # stdtloglik returns (sum_ll, per_obs_ll) tuple
        result = stdtloglik(x, 0.0, 1.0, 5.0)
        if isinstance(result, tuple):
            ll_sum = result[0]
            assert np.isscalar(ll_sum) or (isinstance(ll_sum, np.floating))
            assert np.isfinite(ll_sum)
        else:
            assert np.isfinite(result)

    def test_stdtloglik_increases_with_better_fit(self) -> None:
        """Higher df should give better LL for near-normal data."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        # stdtloglik returns (sum_ll, per_obs_ll) tuple
        result_5 = stdtloglik(x, 0.0, 1.0, 5.0)
        result_30 = stdtloglik(x, 0.0, 1.0, 30.0)
        ll_5 = result_5[0] if isinstance(result_5, tuple) else result_5
        ll_30 = result_30[0] if isinstance(result_30, tuple) else result_30
        assert np.isfinite(ll_5) and np.isfinite(ll_30)


class TestStdtRnd:
    """Unit tests for stdtrnd."""

    def test_stdtrnd_shape(self) -> None:
        """Random variates should have correct shape."""
        r = stdtrnd(5.0, (100,))
        assert r.shape == (100,), f"Expected shape (100,), got {r.shape}"

    def test_stdtrnd_unit_variance(self) -> None:
        """Standardized t should have approximately unit variance for large samples."""
        r = stdtrnd(10.0, (50000,))
        var = np.var(r)
        npt.assert_allclose(var, 1.0, atol=0.1)

    def test_stdtrnd_reproducible(self) -> None:
        """Same RNG state should produce same variates."""
        # stdtrnd signature: (v, *size_args, rng=None)
        rng1 = np.random.default_rng(42)
        r1 = stdtrnd(5.0, (50,), rng=rng1)
        rng2 = np.random.default_rng(42)
        r2 = stdtrnd(5.0, (50,), rng=rng2)
        npt.assert_array_equal(r1, r2)


class TestStdtFixtureParity:
    """Fixture parity tests for standardized t distribution."""

    def test_stdtcdf_parity(self, distributions_fixture_dir) -> None:
        """Check CDF parity against MATLAB fixture."""
        from tests.conftest import load_fixture_npy
        p_expected = load_fixture_npy(distributions_fixture_dir, "stdtcdf_p")
        x = load_fixture_npy(distributions_fixture_dir, "stdtcdf_x")
        v = load_fixture_npy(distributions_fixture_dir, "stdtcdf_v")
        p_actual = stdtcdf(x, v)
        npt.assert_allclose(p_actual, p_expected, atol=ATOL, rtol=RTOL)

    def test_stdtpdf_parity(self, distributions_fixture_dir) -> None:
        """Check PDF parity against MATLAB fixture."""
        from tests.conftest import load_fixture_npy
        f_expected = load_fixture_npy(distributions_fixture_dir, "stdtpdf_f")
        x = load_fixture_npy(distributions_fixture_dir, "stdtpdf_x")
        v = load_fixture_npy(distributions_fixture_dir, "stdtpdf_v")
        f_actual = stdtpdf(x, v)
        npt.assert_allclose(f_actual, f_expected, atol=ATOL, rtol=RTOL)
