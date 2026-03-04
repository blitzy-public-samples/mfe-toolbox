"""
Pytest test suite for mfe_toolbox.distributions.normloglik — normal log-likelihood.

Tests the univariate normal log-likelihood function which computes both
aggregate and per-observation log-likelihoods with optional mean and variance
parameters.  Covers:

- Default argument paths (1-arg, 2-arg, 3-arg)
- Scalar and vector mu / sigma2 inputs
- Sum consistency (LL == sum(lls))
- Independent reference comparison with scipy.stats.norm.logpdf
- Known closed-form values
- Edge-case error handling (non-column x, sigma2 <= 0, dimension mismatches)
- MATLAB parity via fixture comparison (normloglik.npy)

Source reference: distributions/normloglik.m (MFE Toolbox v4.0, Kevin Sheppard)

AAP Section 0.7.1 — Numerical parity contract:
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
"""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import scipy.stats as stats

from mfe_toolbox.distributions.normloglik import normloglik

# Import tolerance constants from conftest (auto-discovered by pytest)
from tests.conftest import ATOL, RTOL, load_fixture_npy


# ---------------------------------------------------------------------------
# Module-level Constants
# ---------------------------------------------------------------------------
# Known log-likelihood of standard normal at x=0:
#   lls = -0.5 * (log(2*pi) + log(1) + 0^2/1) = -0.5 * log(2*pi)
LOG_2PI = np.log(2.0 * np.pi)
KNOWN_LL_STD_NORMAL_AT_ZERO = -0.5 * LOG_2PI  # ≈ -0.9189385332046727


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng_local():
    """Module-scoped reproducible random generator (seed 42)."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def x_data_100(rng_local):
    """T=100 column vector of standard normal draws, shape (100, 1)."""
    return rng_local.standard_normal((100, 1))


@pytest.fixture(scope="module")
def x_data_1000(rng_local):
    """T=1000 column vector of standard normal draws, shape (1000, 1)."""
    return rng_local.standard_normal((1000, 1))


@pytest.fixture(scope="module")
def fixture_dir():
    """Resolve the distributions fixture directory.

    Respects the MFE_FIXTURE_DIR environment variable for CI overrides.
    Falls back to tests/fixtures/distributions relative to repo root.
    """
    env_dir = os.environ.get("MFE_FIXTURE_DIR")
    if env_dir:
        return Path(env_dir) / "distributions"
    return Path(__file__).resolve().parent.parent / "fixtures" / "distributions"


@pytest.fixture(scope="module")
def normloglik_fixture(fixture_dir):
    """Load the main normloglik parity fixture (normloglik.npy).

    Returns the fixture dict with keys like 'x', 'case1_LL', 'case1_lls', etc.
    Skips the test if the fixture file does not exist.
    """
    return load_fixture_npy(fixture_dir, "normloglik")


@pytest.fixture(scope="module")
def normloglik_x_dist(fixture_dir):
    """Load the normloglik_x_dist fixture (linspace test data)."""
    return load_fixture_npy(fixture_dir, "normloglik_x_dist")


@pytest.fixture(scope="module")
def normloglik_normll_out(fixture_dir):
    """Load the scalar LL output fixture."""
    return load_fixture_npy(fixture_dir, "normloglik_normll_out")


@pytest.fixture(scope="module")
def normloglik_normlls_out(fixture_dir):
    """Load the per-obs lls output fixture."""
    return load_fixture_npy(fixture_dir, "normloglik_normlls_out")


# ---------------------------------------------------------------------------
# Phase 2: Default Arguments Tests
# ---------------------------------------------------------------------------


class TestNormloglikDefaultArgs:
    """Tests for default argument handling (1-arg, 2-arg, 3-arg paths).

    Ref: normloglik.m:36-60 — nargin-based branching.
    """

    def test_normloglik_one_arg(self, x_data_100):
        """Only x provided; mu=0, sigma2=1 should be used internally.

        Ref: normloglik.m:36-37 — nargin==1 → sigma2=ones(T,K), no demeaning.
        """
        LL, lls = normloglik(x_data_100)

        # Verify shapes
        assert isinstance(LL, float), "LL must be a float scalar"
        assert isinstance(lls, np.ndarray), "lls must be np.ndarray"
        assert lls.shape == (100, 1), f"lls shape should be (100,1), got {lls.shape}"

        # Manually compute reference: mu=0, sigma2=1
        # lls_ref = -0.5 * (log(2*pi) + log(1) + x^2 / 1) = -0.5 * (log(2*pi) + x^2)
        lls_ref = -0.5 * (LOG_2PI + x_data_100 ** 2)
        LL_ref = float(np.sum(lls_ref))

        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL,
                            err_msg="lls mismatch for 1-arg call")
        npt.assert_allclose(LL, LL_ref, atol=ATOL, rtol=RTOL,
                            err_msg="LL mismatch for 1-arg call")

    def test_normloglik_two_args(self, x_data_100):
        """x and mu provided; sigma2=1 should be used internally.

        Ref: normloglik.m:38-44 — nargin==2 → validate mu, demean, sigma2=ones.
        """
        mu_val = 0.5
        LL, lls = normloglik(x_data_100, mu=mu_val)

        x_demeaned = x_data_100 - mu_val
        lls_ref = -0.5 * (LOG_2PI + x_demeaned ** 2)
        LL_ref = float(np.sum(lls_ref))

        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL,
                            err_msg="lls mismatch for 2-arg call")
        npt.assert_allclose(LL, LL_ref, atol=ATOL, rtol=RTOL,
                            err_msg="LL mismatch for 2-arg call")

    def test_normloglik_three_args(self, x_data_100):
        """x, mu, sigma2 all provided — full 3-arg path.

        Ref: normloglik.m:45-58 — nargin==3 → validate both, demean, use sigma2.
        """
        mu_val = 1.0
        sigma2_val = 2.0
        LL, lls = normloglik(x_data_100, mu=mu_val, sigma2=sigma2_val)

        x_demeaned = x_data_100 - mu_val
        lls_ref = -0.5 * (LOG_2PI + np.log(sigma2_val) + x_demeaned ** 2 / sigma2_val)
        LL_ref = float(np.sum(lls_ref))

        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL,
                            err_msg="lls mismatch for 3-arg call")
        npt.assert_allclose(LL, LL_ref, atol=ATOL, rtol=RTOL,
                            err_msg="LL mismatch for 3-arg call")


# ---------------------------------------------------------------------------
# Phase 3: Basic Functionality Tests
# ---------------------------------------------------------------------------


class TestNormloglikBasicFunctionality:
    """Core functional tests: scalar/vector parameters, sum consistency, references."""

    def test_normloglik_scalar_sigma2(self, x_data_100):
        """T=100, mu=0, sigma2=1.0; verify lls shape is (T, 1).

        Ref: normloglik.m:52-53 — scalar sigma2 → sigma2*ones(T,K).
        """
        LL, lls = normloglik(x_data_100, mu=0.0, sigma2=1.0)
        assert lls.shape == (100, 1)
        assert isinstance(LL, float)

    def test_normloglik_vector_sigma2(self, x_data_100):
        """T=100, sigma2 as T-length column vector.

        Ref: normloglik.m:54-55 — vector sigma2 with size(sigma2,1)==T.
        """
        T = x_data_100.shape[0]
        sigma2_vec = np.abs(x_data_100) + 0.1  # Positive variance vector, shape (T,1)
        LL, lls = normloglik(x_data_100, mu=0.0, sigma2=sigma2_vec)

        assert lls.shape == (T, 1)
        # Manual reference
        x_demeaned = x_data_100 - 0.0
        lls_ref = -0.5 * (LOG_2PI + np.log(sigma2_vec) + x_demeaned ** 2 / sigma2_vec)
        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL,
                            err_msg="lls mismatch for vector sigma2")
        npt.assert_allclose(LL, float(np.sum(lls_ref)), atol=ATOL, rtol=RTOL)

    def test_normloglik_sum_consistency(self, x_data_100):
        """Verify np.sum(lls) ≈ LL within atol.

        Ref: normloglik.m:70 — LL = sum(lls).
        """
        LL, lls = normloglik(x_data_100, mu=0.0, sigma2=1.5)
        npt.assert_allclose(float(np.sum(lls)), LL, atol=ATOL, rtol=RTOL,
                            err_msg="Sum consistency: sum(lls) != LL")

    def test_normloglik_standard_normal_reference(self, x_data_100):
        """Compare against scipy.stats.norm.logpdf as independent reference.

        For mu=0, sigma2=1:
            norm.logpdf(x, loc=0, scale=1) should match lls (flattened).
        """
        LL, lls = normloglik(x_data_100)

        # scipy.stats.norm.logpdf uses scale=std (not variance)
        scipy_lls = stats.norm.logpdf(x_data_100.ravel(), loc=0.0, scale=1.0)
        npt.assert_allclose(lls.ravel(), scipy_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Mismatch with scipy.stats.norm.logpdf reference")

    def test_normloglik_standard_normal_reference_nonunit_variance(self, x_data_100):
        """Compare against scipy.stats.norm.logpdf with sigma2=2.5.

        scale = sqrt(sigma2) for scipy.
        """
        sigma2_val = 2.5
        mu_val = 1.0
        LL, lls = normloglik(x_data_100, mu=mu_val, sigma2=sigma2_val)

        scipy_lls = stats.norm.logpdf(x_data_100.ravel(), loc=mu_val,
                                      scale=np.sqrt(sigma2_val))
        npt.assert_allclose(lls.ravel(), scipy_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Mismatch with scipy reference (sigma2=2.5)")

    def test_normloglik_known_values(self):
        """x=[0], mu=0, sigma2=1 → lls ≈ -0.9189385332.

        The log-density of N(0,1) at x=0 is -0.5 * log(2*pi).
        """
        x = np.array([[0.0]])
        LL, lls = normloglik(x)
        expected_ll = KNOWN_LL_STD_NORMAL_AT_ZERO

        npt.assert_allclose(lls[0, 0], expected_ll, atol=ATOL, rtol=RTOL,
                            err_msg="Known value at x=0 mismatch")
        npt.assert_allclose(LL, expected_ll, atol=ATOL, rtol=RTOL)

    def test_normloglik_known_values_nonzero_x(self):
        """x=[1], mu=0, sigma2=1 → lls = -0.5 * (log(2*pi) + 1) ≈ -1.4189385332."""
        x = np.array([[1.0]])
        LL, lls = normloglik(x)
        expected_ll = -0.5 * (LOG_2PI + 1.0)

        npt.assert_allclose(lls[0, 0], expected_ll, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(LL, expected_ll, atol=ATOL, rtol=RTOL)

    def test_normloglik_nonzero_mu(self, x_data_100):
        """mu=2.0 — verify demeaning is applied correctly.

        Ref: normloglik.m:43 — x = x - mu.
        """
        mu_val = 2.0
        LL, lls = normloglik(x_data_100, mu=mu_val)

        # Equivalent to passing (x - mu) with mu=0
        x_demeaned = x_data_100 - mu_val
        LL_ref, lls_ref = normloglik(x_demeaned)

        npt.assert_allclose(LL, LL_ref, atol=ATOL, rtol=RTOL,
                            err_msg="Demeaning: LL mismatch")
        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL,
                            err_msg="Demeaning: lls mismatch")

    def test_normloglik_vector_mu(self, x_data_100):
        """mu as T-length column vector — verify per-obs demeaning.

        Ref: normloglik.m:40 — length(mu)~=1 path.
        """
        T = x_data_100.shape[0]
        mu_vec = np.linspace(-1.0, 1.0, T).reshape(T, 1)
        LL, lls = normloglik(x_data_100, mu=mu_vec, sigma2=1.0)

        x_demeaned = x_data_100 - mu_vec
        lls_ref = -0.5 * (LOG_2PI + x_demeaned ** 2)
        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL,
                            err_msg="Vector mu: lls mismatch")
        npt.assert_allclose(LL, float(np.sum(lls_ref)), atol=ATOL, rtol=RTOL)

    def test_normloglik_vector_mu_and_sigma2(self, x_data_100):
        """Both mu and sigma2 as T-length column vectors."""
        T = x_data_100.shape[0]
        mu_vec = np.linspace(-1.0, 1.0, T).reshape(T, 1)
        sigma2_vec = np.linspace(0.5, 3.0, T).reshape(T, 1)
        LL, lls = normloglik(x_data_100, mu=mu_vec, sigma2=sigma2_vec)

        x_demeaned = x_data_100 - mu_vec
        lls_ref = -0.5 * (LOG_2PI + np.log(sigma2_vec) + x_demeaned ** 2 / sigma2_vec)
        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(LL, float(np.sum(lls_ref)), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Phase 4: Edge Cases and Validation
# ---------------------------------------------------------------------------


class TestNormloglikValidation:
    """Tests for input validation error paths.

    Ref: normloglik.m:32-55 — various error() calls.
    """

    def test_normloglik_column_required_1d(self):
        """Non-column x (1D array) → ValueError.

        Ref: normloglik.m:32-34 — if K~=1, error('x must be a column vector').
        Note: The Python implementation requires x.ndim==2 and x.shape[1]==1.
        """
        x_1d = np.array([1.0, 2.0, 3.0])  # shape (3,) — not a column
        with pytest.raises(ValueError, match="column vector"):
            normloglik(x_1d)

    def test_normloglik_column_required_row(self):
        """Row vector x of shape (1, T) → ValueError."""
        x_row = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3) — K != 1
        with pytest.raises(ValueError, match="column vector"):
            normloglik(x_row)

    def test_normloglik_column_required_matrix(self):
        """Matrix x of shape (T, 2) → ValueError."""
        x_mat = np.ones((10, 2))
        with pytest.raises(ValueError, match="column vector"):
            normloglik(x_mat)

    def test_normloglik_sigma2_positive_required_zero(self):
        """sigma2 = 0 → ValueError.

        Ref: normloglik.m:49-50 — if any(sigma2<=0), error(...).
        """
        x = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="positive"):
            normloglik(x, mu=0.0, sigma2=0.0)

    def test_normloglik_sigma2_positive_required_negative(self):
        """sigma2 = -1.0 → ValueError."""
        x = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="positive"):
            normloglik(x, mu=0.0, sigma2=-1.0)

    def test_normloglik_sigma2_positive_required_vector_with_zero(self):
        """sigma2 vector containing 0 → ValueError."""
        x = np.array([[1.0], [2.0], [3.0]])
        sigma2 = np.array([[1.0], [0.0], [1.0]])
        with pytest.raises(ValueError, match="positive"):
            normloglik(x, mu=0.0, sigma2=sigma2)

    def test_normloglik_mu_size_mismatch_raises(self):
        """mu with wrong size (not scalar, not T×1) → ValueError.

        Ref: normloglik.m:40-42 — if length(mu)~=1 && ~all(size(mu)==[T K]).
        """
        x = np.array([[1.0], [2.0], [3.0]])  # T=3
        mu_wrong = np.array([[1.0], [2.0]])   # T=2, not matching
        with pytest.raises(ValueError, match="mu must be either a scalar"):
            normloglik(x, mu=mu_wrong)

    def test_normloglik_sigma2_size_mismatch_raises(self):
        """sigma2 with wrong size (not scalar, not T×1) → ValueError.

        Ref: normloglik.m:54-55 — size(sigma2,1)~=T || size(sigma2,2)~=1.
        """
        x = np.array([[1.0], [2.0], [3.0]])  # T=3
        sigma2_wrong = np.array([[1.0], [2.0]])  # T=2, not matching
        with pytest.raises(ValueError, match="sigma2 must be a scalar"):
            normloglik(x, mu=0.0, sigma2=sigma2_wrong)

    def test_normloglik_sigma2_matrix_raises(self):
        """sigma2 as 2D matrix (T, 2) → ValueError."""
        x = np.array([[1.0], [2.0], [3.0]])  # T=3
        sigma2_mat = np.ones((3, 2))
        with pytest.raises(ValueError, match="sigma2 must be a scalar"):
            normloglik(x, mu=0.0, sigma2=sigma2_mat)

    def test_normloglik_mu_size_mismatch_three_args(self):
        """mu mismatch when all 3 args provided — error should still trigger.

        Ref: normloglik.m:46-48 — nargin==3 path also validates mu.
        """
        x = np.array([[1.0], [2.0], [3.0]])
        mu_wrong = np.array([[1.0], [2.0], [3.0], [4.0]])  # T=4
        with pytest.raises(ValueError, match="mu must be either a scalar"):
            normloglik(x, mu=mu_wrong, sigma2=1.0)


# ---------------------------------------------------------------------------
# Phase 5: Parity Tests — MATLAB Fixture Comparison
# ---------------------------------------------------------------------------


class TestNormloglikParity:
    """Parity tests against MATLAB-generated fixture data.

    Fixture file: tests/fixtures/distributions/normloglik.npy
    Generated by: scripts/generate_fixtures.m via Octave 8.4.0

    AAP Section 0.7.1: assert_allclose(actual, expected, atol=1e-6, rtol=1e-4).
    """

    def test_normloglik_parity_case1_default(self, normloglik_fixture):
        """Case 1: normloglik(x) — default mu=0, sigma2=1.

        Ref: normloglik.npy:case1_* keys.
        """
        fixture = normloglik_fixture.item() if normloglik_fixture.ndim == 0 else normloglik_fixture
        if isinstance(fixture, np.ndarray):
            fixture = fixture.item()

        x = fixture['x'].reshape(-1, 1)
        expected_LL = float(fixture['case1_LL'])
        expected_lls = fixture['case1_lls'].reshape(-1, 1)

        LL, lls = normloglik(x)

        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Parity Case 1 LL mismatch")
        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Parity Case 1 lls mismatch")

    def test_normloglik_parity_case2_explicit_mu(self, normloglik_fixture):
        """Case 2: normloglik(x, 0) — explicit mu=0, default sigma2=1.

        Ref: normloglik.npy:case2_* keys.
        """
        fixture = normloglik_fixture.item() if normloglik_fixture.ndim == 0 else normloglik_fixture
        if isinstance(fixture, np.ndarray):
            fixture = fixture.item()

        x = fixture['x'].reshape(-1, 1)
        mu = float(fixture['case2_mu'])
        expected_LL = float(fixture['case2_LL'])
        expected_lls = fixture['case2_lls'].reshape(-1, 1)

        LL, lls = normloglik(x, mu=mu)

        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Parity Case 2 LL mismatch")
        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Parity Case 2 lls mismatch")

    def test_normloglik_parity_case3_sigma2(self, normloglik_fixture):
        """Case 3: normloglik(x, 0, 1.5) — explicit mu=0, sigma2=1.5.

        Ref: normloglik.npy:case3_* keys.
        """
        fixture = normloglik_fixture.item() if normloglik_fixture.ndim == 0 else normloglik_fixture
        if isinstance(fixture, np.ndarray):
            fixture = fixture.item()

        x = fixture['x'].reshape(-1, 1)
        mu = float(fixture['case3_mu'])
        sigma2 = float(fixture['case3_sigma2'])
        expected_LL = float(fixture['case3_LL'])
        expected_lls = fixture['case3_lls'].reshape(-1, 1)

        LL, lls = normloglik(x, mu=mu, sigma2=sigma2)

        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Parity Case 3 LL mismatch")
        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Parity Case 3 lls mismatch")

    def test_normloglik_parity_all_LL(self, normloglik_fixture):
        """Verify all 3 test case LL values match the all_LL summary array."""
        fixture = normloglik_fixture.item() if normloglik_fixture.ndim == 0 else normloglik_fixture
        if isinstance(fixture, np.ndarray):
            fixture = fixture.item()

        x = fixture['x'].reshape(-1, 1)
        all_LL = fixture['all_LL']
        all_mu = fixture['all_mu']
        all_sigma2 = fixture['all_sigma2']

        for i in range(len(all_LL)):
            mu_i = float(all_mu[i])
            sigma2_i = float(all_sigma2[i])
            expected_LL_i = float(all_LL[i])

            # Handle default sigma2 case (sigma2=1 in fixture means no sigma2 arg)
            if sigma2_i == 1.0 and mu_i == 0.0 and i == 0:
                LL_i, _ = normloglik(x)
            elif sigma2_i == 1.0:
                LL_i, _ = normloglik(x, mu=mu_i)
            else:
                LL_i, _ = normloglik(x, mu=mu_i, sigma2=sigma2_i)

            npt.assert_allclose(LL_i, expected_LL_i, atol=ATOL, rtol=RTOL,
                                err_msg=f"Parity all_LL[{i}] mismatch")

    def test_normloglik_parity_linspace_data(
        self, normloglik_x_dist, normloglik_normll_out, normloglik_normlls_out
    ):
        """Verify normloglik against linspace(-3,3,50) fixture data.

        Uses the secondary fixtures: normloglik_x_dist.npy, normloglik_normll_out.npy,
        normloglik_normlls_out.npy.
        """
        x = normloglik_x_dist.reshape(-1, 1)
        expected_LL = float(normloglik_normll_out)
        expected_lls = normloglik_normlls_out.reshape(-1, 1)

        LL, lls = normloglik(x)

        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Linspace parity LL mismatch")
        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Linspace parity lls mismatch")


# ---------------------------------------------------------------------------
# Phase 6: Parametrized Tests
# ---------------------------------------------------------------------------


class TestNormloglikParametrized:
    """Parametrized tests for sigma2 values and output type verification."""

    @pytest.mark.parametrize("sigma2_val", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    def test_normloglik_parametrized_sigma2(self, x_data_100, sigma2_val):
        """Parametrize over sigma2 values — verify against scipy reference.

        For each sigma2, compare normloglik output to scipy.stats.norm.logpdf
        with scale=sqrt(sigma2).
        """
        LL, lls = normloglik(x_data_100, mu=0.0, sigma2=sigma2_val)

        scipy_lls = stats.norm.logpdf(
            x_data_100.ravel(), loc=0.0, scale=np.sqrt(sigma2_val)
        )
        npt.assert_allclose(lls.ravel(), scipy_lls, atol=ATOL, rtol=RTOL,
                            err_msg=f"Parametrized sigma2={sigma2_val}: lls mismatch")
        npt.assert_allclose(LL, float(np.sum(scipy_lls)), atol=ATOL, rtol=RTOL,
                            err_msg=f"Parametrized sigma2={sigma2_val}: LL mismatch")

    @pytest.mark.parametrize("mu_val", [-5.0, -1.0, 0.0, 0.5, 1.0, 3.0])
    def test_normloglik_parametrized_mu(self, x_data_100, mu_val):
        """Parametrize over mu values — verify against scipy reference."""
        LL, lls = normloglik(x_data_100, mu=mu_val, sigma2=1.0)

        scipy_lls = stats.norm.logpdf(x_data_100.ravel(), loc=mu_val, scale=1.0)
        npt.assert_allclose(lls.ravel(), scipy_lls, atol=ATOL, rtol=RTOL,
                            err_msg=f"Parametrized mu={mu_val}: lls mismatch")
        npt.assert_allclose(LL, float(np.sum(scipy_lls)), atol=ATOL, rtol=RTOL,
                            err_msg=f"Parametrized mu={mu_val}: LL mismatch")

    def test_normloglik_output_types(self, x_data_100):
        """Verify LL is float, lls is np.ndarray with correct shape."""
        LL, lls = normloglik(x_data_100)

        assert isinstance(LL, float), f"LL should be float, got {type(LL).__name__}"
        assert isinstance(lls, np.ndarray), f"lls should be ndarray, got {type(lls).__name__}"
        assert lls.shape == (100, 1), f"lls shape should be (100,1), got {lls.shape}"
        assert lls.dtype == np.float64, f"lls dtype should be float64, got {lls.dtype}"

    def test_normloglik_output_types_scalar_input(self):
        """Verify output types for single-element input."""
        x = np.array([[2.5]])
        LL, lls = normloglik(x, mu=0.0, sigma2=1.0)

        assert isinstance(LL, float)
        assert isinstance(lls, np.ndarray)
        assert lls.shape == (1, 1)


# ---------------------------------------------------------------------------
# Phase 7: Additional Edge Cases and Robustness
# ---------------------------------------------------------------------------


class TestNormloglikRobustness:
    """Additional robustness and edge-case tests."""

    def test_normloglik_large_T(self):
        """Large T=10000 — verify no numerical overflow or performance issues."""
        rng = np.random.default_rng(123)
        x = rng.standard_normal((10000, 1))
        LL, lls = normloglik(x)

        assert lls.shape == (10000, 1)
        assert np.isfinite(LL), "LL should be finite for standard normal data"
        assert np.all(np.isfinite(lls)), "All lls should be finite"

        # Cross-check with scipy
        scipy_lls = stats.norm.logpdf(x.ravel(), loc=0.0, scale=1.0)
        npt.assert_allclose(lls.ravel(), scipy_lls, atol=ATOL, rtol=RTOL)

    def test_normloglik_single_obs(self):
        """Single observation T=1."""
        x = np.array([[3.0]])
        LL, lls = normloglik(x, mu=1.0, sigma2=2.0)

        x_demeaned = 3.0 - 1.0  # 2.0
        expected_ll = -0.5 * (LOG_2PI + np.log(2.0) + 4.0 / 2.0)
        npt.assert_allclose(LL, expected_ll, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lls[0, 0], expected_ll, atol=ATOL, rtol=RTOL)

    def test_normloglik_very_small_sigma2(self):
        """Very small sigma2 — verify no crash (large negative lls expected)."""
        x = np.array([[1.0], [2.0]])
        LL, lls = normloglik(x, mu=0.0, sigma2=1e-10)

        # Very small sigma2 with non-zero x → very large negative lls
        assert np.all(np.isfinite(lls)), "lls should be finite even for small sigma2"
        assert LL < 0, "LL should be negative"

    def test_normloglik_very_large_sigma2(self):
        """Very large sigma2 — lls approach -0.5*log(2*pi*sigma2)."""
        x = np.array([[0.0], [0.0], [0.0]])
        sigma2_val = 1e10
        LL, lls = normloglik(x, mu=0.0, sigma2=sigma2_val)

        # At x=0, mu=0: lls = -0.5 * (log(2*pi) + log(sigma2) + 0)
        expected_ll = -0.5 * (LOG_2PI + np.log(sigma2_val))
        npt.assert_allclose(lls[0, 0], expected_ll, atol=ATOL, rtol=RTOL)

    def test_normloglik_mu_scalar_broadcast(self):
        """Scalar mu is broadcast to all observations.

        Ref: normloglik.m:40 — length(mu)==1 is allowed (scalar).
        """
        x = np.array([[1.0], [2.0], [3.0]])
        LL1, lls1 = normloglik(x, mu=0.5, sigma2=1.0)
        LL2, lls2 = normloglik(x, mu=np.array([[0.5]]), sigma2=1.0)

        # Scalar mu and (1,1) mu should give identical results
        npt.assert_allclose(LL1, LL2, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lls1, lls2, atol=ATOL, rtol=RTOL)

    def test_normloglik_sigma2_scalar_broadcast(self):
        """Scalar sigma2 is broadcast to all observations.

        Ref: normloglik.m:52-53 — length(sigma2)==1 → sigma2*ones(T,K).
        """
        x = np.array([[1.0], [2.0], [3.0]])
        LL1, lls1 = normloglik(x, mu=0.0, sigma2=2.0)
        # Equivalent: pass sigma2 as a vector of 2.0's
        sigma2_vec = 2.0 * np.ones((3, 1))
        LL2, lls2 = normloglik(x, mu=0.0, sigma2=sigma2_vec)

        npt.assert_allclose(LL1, LL2, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lls1, lls2, atol=ATOL, rtol=RTOL)

    def test_normloglik_negative_lls(self, x_data_100):
        """All per-observation log-likelihoods should be negative (log of probability < 1)."""
        LL, lls = normloglik(x_data_100)
        # For normal density, max log-pdf = -0.5*log(2*pi) ≈ -0.9189
        assert np.all(lls < 0), "All lls must be negative for normal distribution"
        assert LL < 0, "LL must be negative"

    def test_normloglik_symmetry(self):
        """normloglik(x) == normloglik(-x) when mu=0 (symmetric distribution)."""
        x = np.array([[1.0], [-2.0], [3.0], [-4.0], [5.0]])
        x_neg = -x

        LL1, lls1 = normloglik(x)
        LL2, lls2 = normloglik(x_neg)

        npt.assert_allclose(LL1, LL2, atol=ATOL, rtol=RTOL,
                            err_msg="Symmetry: LL should be identical for x and -x")
        # Per-obs lls should match elementwise (due to x^2 = (-x)^2)
        npt.assert_allclose(lls1, lls2, atol=ATOL, rtol=RTOL,
                            err_msg="Symmetry: lls should be identical for x and -x")

    def test_normloglik_mu_zero_equivalent(self, x_data_100):
        """Passing mu=0 should be equivalent to omitting mu.

        Ref: normloglik.m — nargin==1 (no mu) vs nargin==2 (mu=0).
        """
        LL_no_mu, lls_no_mu = normloglik(x_data_100)
        LL_mu_zero, lls_mu_zero = normloglik(x_data_100, mu=0.0)

        npt.assert_allclose(LL_no_mu, LL_mu_zero, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lls_no_mu, lls_mu_zero, atol=ATOL, rtol=RTOL)

    def test_normloglik_sigma2_one_equivalent(self, x_data_100):
        """Passing sigma2=1 should be equivalent to omitting sigma2.

        Ref: normloglik.m — nargin<=2 default sigma2=ones(T,K).
        """
        LL_no_s2, lls_no_s2 = normloglik(x_data_100, mu=0.0)
        LL_s2_one, lls_s2_one = normloglik(x_data_100, mu=0.0, sigma2=1.0)

        npt.assert_allclose(LL_no_s2, LL_s2_one, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lls_no_s2, lls_s2_one, atol=ATOL, rtol=RTOL)

    def test_normloglik_1d_mu_two_args(self):
        """1D mu array of shape (T,) in 2-arg path triggers reshape.

        Ref: normloglik.py:94-95 — if mu_arr.ndim==1 and shape[0]==T → reshape.
        """
        x = np.array([[1.0], [2.0], [3.0]])
        mu_1d = np.array([0.5, 0.5, 0.5])  # shape (3,) not (3,1)
        LL, lls = normloglik(x, mu=mu_1d)

        x_demeaned = x - 0.5
        lls_ref = -0.5 * (LOG_2PI + x_demeaned ** 2)
        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL)

    def test_normloglik_1d_mu_three_args(self):
        """1D mu array of shape (T,) in 3-arg path triggers reshape.

        Ref: normloglik.py:110-111 — if mu_arr.ndim==1 and shape[0]==T → reshape.
        """
        x = np.array([[1.0], [2.0], [3.0]])
        mu_1d = np.array([0.1, 0.2, 0.3])
        LL, lls = normloglik(x, mu=mu_1d, sigma2=1.0)

        x_demeaned = x - mu_1d.reshape(-1, 1)
        lls_ref = -0.5 * (LOG_2PI + x_demeaned ** 2)
        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL)

    def test_normloglik_1d_sigma2_three_args(self):
        """1D sigma2 array of shape (T,) in 3-arg path triggers reshape.

        Ref: normloglik.py:129-130 — if sigma2_arr.ndim==1 and shape[0]==T → reshape.
        """
        x = np.array([[1.0], [2.0], [3.0]])
        sigma2_1d = np.array([1.0, 2.0, 3.0])  # shape (3,)
        LL, lls = normloglik(x, mu=0.0, sigma2=sigma2_1d)

        sigma2_col = sigma2_1d.reshape(-1, 1)
        lls_ref = -0.5 * (LOG_2PI + np.log(sigma2_col) + x ** 2 / sigma2_col)
        npt.assert_allclose(lls, lls_ref, atol=ATOL, rtol=RTOL)
