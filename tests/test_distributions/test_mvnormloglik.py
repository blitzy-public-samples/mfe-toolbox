"""Parity and unit tests for mfe_toolbox.distributions.mvnormloglik.

Tests the multivariate normal log-likelihood function migrated from MATLAB
``distributions/mvnormloglik.m`` (MFE Toolbox, Version 4.0).

The function computes ``LL`` (scalar total log-likelihood) and ``lls`` (per-
observation log-likelihoods) for T observations of a K-dimensional
multivariate normal, with two code paths:

1. **Constant covariance** — ``sigma`` is K × K (Cholesky-based vectorised).
2. **Time-varying covariance** — ``sigma`` is K × K × T (per-obs loop).

Per AAP Section 0.7.1:
- ``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
- Both code paths must match MATLAB reference outputs to ±1e-6.
- Invalid inputs must raise ``ValueError`` (matching MATLAB ``error()``).
- Coverage ≥90 %, all tests pass with ``pytest -x --tb=short``.

Ref: distributions/mvnormloglik.m (103 lines)
Ref: mfe_toolbox/distributions/mvnormloglik.py (310 lines)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import scipy.stats as stats

from mfe_toolbox.distributions.mvnormloglik import mvnormloglik
from mfe_toolbox.distributions.normloglik import normloglik

# ---------------------------------------------------------------------------
# Numerical tolerance constants — mirrors tests/conftest.py ATOL/RTOL
# Per AAP Section 0.7.1: atol=1e-6, rtol=1e-4
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def rng_local() -> np.random.Generator:
    """Seeded random generator for reproducible test data (seed=42)."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def mvn_x(rng_local: np.random.Generator) -> np.ndarray:
    """T=100, K=3 standard-normal data matrix."""
    return rng_local.standard_normal((100, 3))


@pytest.fixture(scope="module")
def pd_sigma_3x3() -> np.ndarray:
    """A known 3×3 positive-definite covariance matrix.

    Constructed from a Cholesky factor to guarantee PD-ness.
    """
    L = np.array([
        [2.0, 0.0, 0.0],
        [0.3, 1.5, 0.0],
        [-0.4, 0.2, 1.8],
    ])
    return L @ L.T


@pytest.fixture(scope="module")
def sigma_3d_constant(pd_sigma_3x3: np.ndarray) -> np.ndarray:
    """K×K×T 3-D sigma where every slice is the same PD matrix (T=100)."""
    T = 100
    sigma_3d = np.zeros((3, 3, T))
    for t in range(T):
        sigma_3d[:, :, t] = pd_sigma_3x3
    return sigma_3d


@pytest.fixture(scope="module")
def sigma_3d_varying() -> np.ndarray:
    """K×K×T truly time-varying 3-D sigma (K=3, T=50).

    Each slice is a distinct positive-definite matrix generated from a
    random Cholesky factor.
    """
    rng = np.random.default_rng(99)
    T = 50
    K = 3
    sigma_3d = np.zeros((K, K, T))
    for t in range(T):
        L = np.eye(K) + 0.3 * np.tril(rng.standard_normal((K, K)))
        sigma_3d[:, :, t] = L @ L.T + 0.05 * np.eye(K)
    return sigma_3d


@pytest.fixture(scope="module")
def mvn_fixture(distributions_fixture_dir: Path) -> dict:
    """Load the mvnormloglik MATLAB-parity fixture from .npy file.

    Uses ``distributions_fixture_dir`` pytest fixture from conftest.py.
    If the fixture file does not exist the test is gracefully skipped.
    """
    path = distributions_fixture_dir / "mvnormloglik.npy"
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")
    return np.load(path, allow_pickle=True).item()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Constant Sigma (2-D) Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMvnormloglikConstantSigma:
    """Tests for the constant (K × K) covariance code path."""

    def test_mvnormloglik_identity_sigma(self, mvn_x: np.ndarray) -> None:
        """Identity covariance: result equals sum of K independent normals.

        When sigma = I_K the multivariate normal decomposes into K
        independent standard normals so the log-likelihood can be computed
        analytically as sum of univariate standard-normal log-likelihoods.

        Ref: mvnormloglik.m:87-93 — constant sigma path.
        """
        T, K = mvn_x.shape
        sigma = np.eye(K)
        LL, lls = mvnormloglik(mvn_x, sigma=sigma)

        # Expected: -0.5*(K*log(2*pi) + sum_k x_{t,k}^2 ) per observation
        expected_lls = -0.5 * (K * np.log(2.0 * np.pi) + np.sum(mvn_x ** 2, axis=1))
        expected_LL = np.sum(expected_lls)

        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Per-obs log-likelihoods with identity sigma")
        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Total log-likelihood with identity sigma")

    def test_mvnormloglik_general_sigma_2d(
        self, mvn_x: np.ndarray, pd_sigma_3x3: np.ndarray
    ) -> None:
        """General PD covariance: compare against manual Cholesky computation.

        Ref: mvnormloglik.m:88-92 — sigma^(-0.5) path.
        """
        T, K = mvn_x.shape
        sigma = pd_sigma_3x3
        LL, lls = mvnormloglik(mvn_x, sigma=sigma)

        # Manually compute via Cholesky
        L = np.linalg.cholesky(sigma)
        stdx = np.linalg.solve(L, mvn_x.T).T  # whitened observations
        _sign, logdet = np.linalg.slogdet(sigma)
        manual_lls = -0.5 * (
            K * np.log(2.0 * np.pi) + logdet + np.sum(stdx ** 2, axis=1)
        )
        manual_LL = np.sum(manual_lls)

        npt.assert_allclose(lls, manual_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Per-obs ll with general PD sigma")
        npt.assert_allclose(LL, manual_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Total ll with general PD sigma")

    def test_mvnormloglik_sum_consistency_2d(
        self, mvn_x: np.ndarray, pd_sigma_3x3: np.ndarray
    ) -> None:
        """Verify LL == sum(lls) for constant covariance.

        Ref: mvnormloglik.m:89 — LL aggregation.
        """
        LL, lls = mvnormloglik(mvn_x, sigma=pd_sigma_3x3)
        npt.assert_allclose(LL, np.sum(lls), atol=1e-10, rtol=0,
                            err_msg="LL must equal sum(lls)")

    def test_mvnormloglik_scipy_reference_2d(
        self, mvn_x: np.ndarray, pd_sigma_3x3: np.ndarray
    ) -> None:
        """Cross-validate against scipy.stats.multivariate_normal.logpdf.

        This is the primary independent reference check — scipy provides a
        fully independent implementation of the multivariate normal density.

        Ref: AAP Section 0.7.1 — numerical parity.
        """
        T, K = mvn_x.shape
        mu = np.zeros(K)
        sigma = pd_sigma_3x3

        LL, lls = mvnormloglik(mvn_x, sigma=sigma)

        # scipy reference: logpdf returns per-obs values
        ref_lls = stats.multivariate_normal.logpdf(mvn_x, mean=mu, cov=sigma)
        ref_LL = np.sum(ref_lls)

        npt.assert_allclose(lls, ref_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Per-obs lls vs scipy reference")
        npt.assert_allclose(LL, ref_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Total LL vs scipy reference")

    def test_mvnormloglik_nonzero_mu(self, mvn_x: np.ndarray) -> None:
        """Non-zero mean: verify demeaning is applied correctly.

        Ref: mvnormloglik.m:45-47, 55-57 — mu subtraction.
        """
        T, K = mvn_x.shape
        mu = np.array([1.0, -0.5, 2.0])
        sigma = np.eye(K)

        LL, lls = mvnormloglik(mvn_x, mu=mu, sigma=sigma)

        # Manual: demean then compute identity-sigma log-likelihood
        x_demeaned = mvn_x - np.tile(mu, (T, 1))
        expected_lls = -0.5 * (K * np.log(2.0 * np.pi) + np.sum(x_demeaned ** 2, axis=1))
        expected_LL = np.sum(expected_lls)

        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL)

    def test_mvnormloglik_nonzero_mu_with_sigma(
        self, mvn_x: np.ndarray, pd_sigma_3x3: np.ndarray
    ) -> None:
        """Non-zero mean with general PD sigma: scipy cross-check.

        Ref: mvnormloglik.m:50-57, 87-93.
        """
        T, K = mvn_x.shape
        mu = np.array([0.5, -1.0, 0.3])
        sigma = pd_sigma_3x3

        LL, lls = mvnormloglik(mvn_x, mu=mu, sigma=sigma)

        ref_lls = stats.multivariate_normal.logpdf(mvn_x, mean=mu, cov=sigma)
        ref_LL = np.sum(ref_lls)

        npt.assert_allclose(lls, ref_lls, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(LL, ref_LL, atol=ATOL, rtol=RTOL)

    def test_mvnormloglik_negative_loglik_values(
        self, mvn_x: np.ndarray, pd_sigma_3x3: np.ndarray
    ) -> None:
        """Log-likelihood values for continuous distributions can be any real.

        For the MVN, with identity sigma and standard data, most per-obs ll
        values should be negative (the density at any point is < 1 when K >= 2
        because the normalisation constant 1/((2*pi)^{K/2}) < 1).
        """
        LL, lls = mvnormloglik(mvn_x, sigma=np.eye(mvn_x.shape[1]))
        assert LL < 0, "Total LL for standard normal data should be negative"
        # The vast majority of per-obs values should be negative for K=3
        assert np.sum(lls < 0) > 0.8 * len(lls), (
            "Most per-obs ll values for K=3 standard normal should be negative"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Time-Varying Sigma (3-D) Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMvnormloglikTimeVaryingSigma:
    """Tests for the time-varying (K × K × T) covariance code path."""

    def test_mvnormloglik_3d_sigma_shape(
        self, sigma_3d_varying: np.ndarray
    ) -> None:
        """3-D sigma: check lls shape is (T,).

        Ref: mvnormloglik.m:95-100 — time-varying path.
        """
        T = sigma_3d_varying.shape[2]
        K = sigma_3d_varying.shape[0]
        rng = np.random.default_rng(77)
        x = rng.standard_normal((T, K))

        LL, lls = mvnormloglik(x, sigma=sigma_3d_varying)

        assert lls.shape == (T,), f"Expected lls shape (T,), got {lls.shape}"
        assert isinstance(LL, float), "LL must be a float scalar"

    def test_mvnormloglik_3d_constant_check(
        self, mvn_x: np.ndarray, pd_sigma_3x3: np.ndarray,
        sigma_3d_constant: np.ndarray
    ) -> None:
        """3-D sigma with identical slices must match 2-D sigma result.

        If sigma[:, :, t] == sigma_2d for all t, then the time-varying and
        constant code paths must produce identical output.

        Ref: mvnormloglik.m:87-93 vs 94-100 — both paths should agree.
        """
        LL_2d, lls_2d = mvnormloglik(mvn_x, sigma=pd_sigma_3x3)
        LL_3d, lls_3d = mvnormloglik(mvn_x, sigma=sigma_3d_constant)

        npt.assert_allclose(lls_3d, lls_2d, atol=ATOL, rtol=RTOL,
                            err_msg="Per-obs lls: 3D constant vs 2D sigma")
        npt.assert_allclose(LL_3d, LL_2d, atol=ATOL, rtol=RTOL,
                            err_msg="Total LL: 3D constant vs 2D sigma")

    def test_mvnormloglik_sum_consistency_3d(
        self, sigma_3d_varying: np.ndarray
    ) -> None:
        """Verify LL == sum(lls) for time-varying covariance.

        Ref: mvnormloglik.m:100 — LL=sum(lls).
        """
        T = sigma_3d_varying.shape[2]
        K = sigma_3d_varying.shape[0]
        rng = np.random.default_rng(88)
        x = rng.standard_normal((T, K))

        LL, lls = mvnormloglik(x, sigma=sigma_3d_varying)
        npt.assert_allclose(LL, np.sum(lls), atol=1e-10, rtol=0,
                            err_msg="LL must equal sum(lls) for 3D sigma")

    def test_mvnormloglik_3d_vs_loop(
        self, sigma_3d_varying: np.ndarray
    ) -> None:
        """Manually compute per-obs log-likelihood via explicit loop.

        This verifies the time-varying path by independently computing each
        observation's log-likelihood using the formula:
            lls[t] = -0.5 * (K*log(2*pi) + log(det(s)) + x_t' @ inv(s) @ x_t)

        Ref: mvnormloglik.m:97-98.
        """
        T = sigma_3d_varying.shape[2]
        K = sigma_3d_varying.shape[0]
        rng = np.random.default_rng(101)
        x = rng.standard_normal((T, K))

        LL, lls = mvnormloglik(x, sigma=sigma_3d_varying)

        # Manual loop computation
        manual_lls = np.zeros(T)
        for t in range(T):
            s = sigma_3d_varying[:, :, t]
            _sign, logdet = np.linalg.slogdet(s)
            quad = x[t, :] @ np.linalg.solve(s, x[t, :])
            manual_lls[t] = -0.5 * (K * np.log(2.0 * np.pi) + logdet + quad)

        npt.assert_allclose(lls, manual_lls, atol=ATOL, rtol=RTOL,
                            err_msg="3D sigma per-obs lls vs manual loop")
        npt.assert_allclose(LL, np.sum(manual_lls), atol=ATOL, rtol=RTOL,
                            err_msg="3D sigma total LL vs manual loop sum")

    def test_mvnormloglik_3d_scipy_reference(
        self, sigma_3d_varying: np.ndarray
    ) -> None:
        """Cross-validate 3-D sigma path against per-obs scipy.stats calls.

        Each observation uses its own sigma[:, :, t], so we call
        ``scipy.stats.multivariate_normal.logpdf`` independently per obs.
        """
        T = sigma_3d_varying.shape[2]
        K = sigma_3d_varying.shape[0]
        rng = np.random.default_rng(202)
        x = rng.standard_normal((T, K))

        LL, lls = mvnormloglik(x, sigma=sigma_3d_varying)

        ref_lls = np.zeros(T)
        for t in range(T):
            s = sigma_3d_varying[:, :, t]
            ref_lls[t] = stats.multivariate_normal.logpdf(
                x[t, :], mean=np.zeros(K), cov=s
            )

        npt.assert_allclose(lls, ref_lls, atol=ATOL, rtol=RTOL,
                            err_msg="3D sigma per-obs lls vs scipy loop")
        npt.assert_allclose(LL, np.sum(ref_lls), atol=ATOL, rtol=RTOL,
                            err_msg="3D sigma total LL vs scipy loop sum")

    def test_mvnormloglik_3d_with_mu(
        self, sigma_3d_varying: np.ndarray
    ) -> None:
        """3-D sigma with non-zero mean: verify correct demeaning per obs.

        Ref: mvnormloglik.m:50-57 — mu handling with 3-D sigma.
        """
        T = sigma_3d_varying.shape[2]
        K = sigma_3d_varying.shape[0]
        rng = np.random.default_rng(303)
        x = rng.standard_normal((T, K))
        mu = np.array([0.5, -1.0, 0.3])

        LL, lls = mvnormloglik(x, mu=mu, sigma=sigma_3d_varying)

        # Manual computation with demeaning
        x_dm = x - np.tile(mu, (T, 1))
        manual_lls = np.zeros(T)
        for t in range(T):
            s = sigma_3d_varying[:, :, t]
            _sign, logdet = np.linalg.slogdet(s)
            quad = x_dm[t, :] @ np.linalg.solve(s, x_dm[t, :])
            manual_lls[t] = -0.5 * (K * np.log(2.0 * np.pi) + logdet + quad)

        npt.assert_allclose(lls, manual_lls, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(LL, np.sum(manual_lls), atol=ATOL, rtol=RTOL)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Default Arguments
# ═══════════════════════════════════════════════════════════════════════════

class TestMvnormloglikDefaults:
    """Tests for default argument handling."""

    def test_mvnormloglik_defaults_mu(self, mvn_x: np.ndarray) -> None:
        """Omitting mu should produce same result as mu=0.

        Ref: mvnormloglik.m:36-38 — nargin==1 path.
        """
        LL_nomu, lls_nomu = mvnormloglik(mvn_x)
        LL_zero, lls_zero = mvnormloglik(mvn_x, mu=np.zeros(mvn_x.shape[1]))

        npt.assert_allclose(lls_nomu, lls_zero, atol=1e-12, rtol=0,
                            err_msg="Default mu should equal zero mu")
        npt.assert_allclose(LL_nomu, LL_zero, atol=1e-12, rtol=0)

    def test_mvnormloglik_defaults_sigma(self, mvn_x: np.ndarray) -> None:
        """Omitting sigma should produce same result as sigma=eye(K).

        Ref: mvnormloglik.m:37-38, 48-49 — sigma = eye(K).
        """
        K = mvn_x.shape[1]
        LL_default, lls_default = mvnormloglik(mvn_x)
        LL_eye, lls_eye = mvnormloglik(mvn_x, sigma=np.eye(K))

        npt.assert_allclose(lls_default, lls_eye, atol=1e-12, rtol=0,
                            err_msg="Default sigma should equal identity")
        npt.assert_allclose(LL_default, LL_eye, atol=1e-12, rtol=0)

    def test_mvnormloglik_scalar_mu_broadcast(self, mvn_x: np.ndarray) -> None:
        """1×K mu vector should be broadcast across all T observations.

        Ref: mvnormloglik.m:44-46 — mu=repmat(mu', T, 1).
        """
        T, K = mvn_x.shape
        mu_k = np.array([0.5, -1.0, 0.3])
        mu_txk = np.tile(mu_k, (T, 1))  # explicit T×K

        LL_k, lls_k = mvnormloglik(mvn_x, mu=mu_k)
        LL_txk, lls_txk = mvnormloglik(mvn_x, mu=mu_txk)

        npt.assert_allclose(lls_k, lls_txk, atol=1e-12, rtol=0,
                            err_msg="K-vector mu should match T×K tiled mu")
        npt.assert_allclose(LL_k, LL_txk, atol=1e-12, rtol=0)

    def test_mvnormloglik_column_mu(self, mvn_x: np.ndarray) -> None:
        """(K, 1) column mu should work the same as (K,) vector mu.

        Ref: mvnormloglik.m:41 — all(size(mu)==[K 1]).
        """
        K = mvn_x.shape[1]
        mu_flat = np.array([0.5, -1.0, 0.3])
        mu_col = mu_flat.reshape(K, 1)

        LL_flat, lls_flat = mvnormloglik(mvn_x, mu=mu_flat)
        LL_col, lls_col = mvnormloglik(mvn_x, mu=mu_col)

        npt.assert_allclose(lls_flat, lls_col, atol=1e-12, rtol=0,
                            err_msg="(K,1) column mu should match (K,) vector mu")
        npt.assert_allclose(LL_flat, LL_col, atol=1e-12, rtol=0)

    def test_mvnormloglik_only_x_and_sigma(self, mvn_x: np.ndarray) -> None:
        """Passing sigma without mu (mu=None): sigma used, mu=0.

        Ref: mvnormloglik.m:36-38 — sigma without mu defaults mu to 0.
        """
        K = mvn_x.shape[1]
        sigma = 2.0 * np.eye(K)

        LL_nomu, lls_nomu = mvnormloglik(mvn_x, sigma=sigma)
        LL_zero, lls_zero = mvnormloglik(mvn_x, mu=np.zeros(K), sigma=sigma)

        npt.assert_allclose(lls_nomu, lls_zero, atol=1e-12, rtol=0)
        npt.assert_allclose(LL_nomu, LL_zero, atol=1e-12, rtol=0)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Edge Cases and Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestMvnormloglikEdgeCases:
    """Edge cases, boundary conditions, and error handling tests."""

    def test_mvnormloglik_k1_matches_univariate(self) -> None:
        """K=1 multivariate case should match univariate normloglik.

        When K=1 and sigma = [[sigma2]], the multivariate normal reduces to
        the univariate normal with variance sigma2.  Compare against the
        dedicated ``normloglik`` function.

        Ref: mvnormloglik.m — general formula specialises to univariate.
        """
        rng = np.random.default_rng(55)
        T = 50
        x_2d = rng.standard_normal((T, 1))

        # mvnormloglik with K=1
        LL_mv, lls_mv = mvnormloglik(x_2d)

        # normloglik expects (T, 1) column vector
        LL_uni, lls_uni = normloglik(x_2d)

        # normloglik returns lls as (T, 1); mvnormloglik returns (T,)
        npt.assert_allclose(LL_mv, LL_uni, atol=ATOL, rtol=RTOL,
                            err_msg="K=1 mv LL should match univariate LL")
        npt.assert_allclose(lls_mv, lls_uni.flatten(), atol=ATOL, rtol=RTOL,
                            err_msg="K=1 mv lls should match univariate lls")

    def test_mvnormloglik_k1_with_variance(self) -> None:
        """K=1 with non-unit variance: compare mv and univariate.

        Sigma = [[4.0]] in mv should match sigma2=4.0 in univariate.
        Note: normloglik requires mu to be provided when sigma2 is given,
        so we pass mu=0 explicitly.
        """
        rng = np.random.default_rng(56)
        T = 50
        x_2d = rng.standard_normal((T, 1))
        sigma2 = 4.0

        LL_mv, lls_mv = mvnormloglik(x_2d, sigma=np.array([[sigma2]]))
        # normloglik requires mu when sigma2 is given
        LL_uni, lls_uni = normloglik(x_2d, mu=0.0, sigma2=sigma2)

        npt.assert_allclose(LL_mv, LL_uni, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lls_mv, lls_uni.flatten(), atol=ATOL, rtol=RTOL)

    def test_mvnormloglik_k2_scipy_reference(self) -> None:
        """K=2 case cross-checked with scipy.stats.multivariate_normal.

        Exercises a low-dimensional case distinct from the K=3 main tests.
        """
        rng = np.random.default_rng(60)
        T = 30
        K = 2
        x = rng.standard_normal((T, K))
        # Construct PD sigma
        A = np.array([[1.5, 0.0], [0.4, 1.2]])
        sigma = A @ A.T

        LL, lls = mvnormloglik(x, sigma=sigma)
        ref_lls = stats.multivariate_normal.logpdf(
            x, mean=np.zeros(K), cov=sigma
        )
        npt.assert_allclose(lls, ref_lls, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(LL, np.sum(ref_lls), atol=ATOL, rtol=RTOL)

    def test_mvnormloglik_large_k(self) -> None:
        """Larger K=10 case to exercise numerical stability.

        Large K tests that slogdet/cholesky code path handles larger
        matrices without numerical issues.
        """
        rng = np.random.default_rng(70)
        T = 40
        K = 10
        x = rng.standard_normal((T, K))
        # Random PD sigma
        A = np.eye(K) + 0.2 * np.tril(rng.standard_normal((K, K)))
        sigma = A @ A.T + 0.1 * np.eye(K)

        LL, lls = mvnormloglik(x, sigma=sigma)
        ref_lls = stats.multivariate_normal.logpdf(
            x, mean=np.zeros(K), cov=sigma
        )
        npt.assert_allclose(lls, ref_lls, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(LL, np.sum(ref_lls), atol=ATOL, rtol=RTOL)

    def test_mvnormloglik_single_observation(self) -> None:
        """T=1 single observation: should still work correctly.

        Boundary condition where T=1 — both 2D and 3D sigma paths.
        """
        rng = np.random.default_rng(75)
        K = 3
        x = rng.standard_normal((1, K))
        sigma = np.eye(K)

        LL, lls = mvnormloglik(x, sigma=sigma)
        assert lls.shape == (1,), f"Expected shape (1,), got {lls.shape}"
        npt.assert_allclose(LL, lls[0], atol=1e-12, rtol=0)

        # Compare with scipy
        ref = stats.multivariate_normal.logpdf(x[0], mean=np.zeros(K), cov=sigma)
        npt.assert_allclose(LL, ref, atol=ATOL, rtol=RTOL)

    def test_mvnormloglik_sigma_not_pd_raises(self) -> None:
        """Non-positive-definite sigma must raise ValueError.

        Ref: mvnormloglik.m:61 — min(eig(sigma))<=0.
        """
        rng = np.random.default_rng(80)
        T = 20
        K = 3
        x = rng.standard_normal((T, K))

        # Construct a non-PD matrix (has a negative eigenvalue)
        sigma_bad = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 1.0],
        ])

        with pytest.raises(ValueError, match="positive definite"):
            mvnormloglik(x, sigma=sigma_bad)

    def test_mvnormloglik_sigma_not_symmetric_raises(self) -> None:
        """Non-symmetric sigma must raise ValueError.

        Ref: mvnormloglik.m:61 — any(any(sigma~=sigma')).
        """
        rng = np.random.default_rng(81)
        T = 20
        K = 3
        x = rng.standard_normal((T, K))

        sigma_asym = np.eye(K)
        sigma_asym[0, 1] = 0.5  # break symmetry
        # sigma_asym[1, 0] remains 0, so sigma != sigma.T

        with pytest.raises(ValueError, match="positive definite"):
            mvnormloglik(x, sigma=sigma_asym)

    def test_mvnormloglik_dimension_mismatch_raises(self) -> None:
        """sigma dimensions incompatible with x must raise ValueError.

        Ref: mvnormloglik.m:61 — size checks.
        """
        rng = np.random.default_rng(82)
        T = 20
        K = 3
        x = rng.standard_normal((T, K))

        # sigma is (K+1) × (K+1) — mismatch
        sigma_wrong = np.eye(K + 1)

        with pytest.raises(ValueError, match="positive definite"):
            mvnormloglik(x, sigma=sigma_wrong)

    def test_mvnormloglik_3d_sigma_wrong_T_raises(self) -> None:
        """3-D sigma with wrong T dimension must raise ValueError.

        Ref: mvnormloglik.m:66 — size(sigma,3)~=T.
        """
        rng = np.random.default_rng(83)
        T = 20
        K = 3
        x = rng.standard_normal((T, K))

        # sigma is K×K×(T+5) — T mismatch
        sigma_bad = np.zeros((K, K, T + 5))
        for t in range(T + 5):
            sigma_bad[:, :, t] = np.eye(K)

        with pytest.raises(ValueError, match="positive definite"):
            mvnormloglik(x, sigma=sigma_bad)

    def test_mvnormloglik_mu_wrong_shape_raises(self) -> None:
        """mu with wrong shape must raise ValueError.

        Ref: mvnormloglik.m:41-42, 51-52 — mu conformability check.
        """
        rng = np.random.default_rng(84)
        T = 20
        K = 3
        x = rng.standard_normal((T, K))

        # mu has wrong number of elements
        mu_bad = np.array([1.0, 2.0])  # K=2, but data is K=3

        with pytest.raises(ValueError, match="[Mm][Uu]"):
            mvnormloglik(x, mu=mu_bad)

    def test_mvnormloglik_x_not_2d_raises(self) -> None:
        """x that is not 2-D must raise ValueError.

        Ref: mvnormloglik.m:32-34 — ndims(x)>2.
        """
        x_1d = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="T by K matrix"):
            mvnormloglik(x_1d)

    def test_mvnormloglik_3d_sigma_not_pd_raises(self) -> None:
        """3-D sigma with a non-PD slice must raise ValueError.

        Ref: mvnormloglik.m:69-74 — per-slice PD check.
        """
        rng = np.random.default_rng(85)
        T = 10
        K = 3
        x = rng.standard_normal((T, K))

        sigma_3d = np.zeros((K, K, T))
        for t in range(T):
            sigma_3d[:, :, t] = np.eye(K)
        # Make one slice non-PD
        sigma_3d[:, :, 5] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -0.1, 0.0],
            [0.0, 0.0, 1.0],
        ])

        with pytest.raises(ValueError, match="positive definite"):
            mvnormloglik(x, sigma=sigma_3d)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — MATLAB Parity Tests (Fixture-Based)
# ═══════════════════════════════════════════════════════════════════════════

class TestMvnormloglikParity:
    """MATLAB parity tests using pre-generated fixture data.

    Fixture file: tests/fixtures/distributions/mvnormloglik.npy
    Contains 4 test cases generated from MATLAB reference computation:
      case1: mvnormloglik(x) — no mu, sigma=eye(K)
      case2: mvnormloglik(x, mu_Kx1) — K×1 mean, sigma=eye(K)
      case3: mvnormloglik(x, mu_Kx1, sigma_KxK) — K×1 mean, K×K covariance
      case4: mvnormloglik(x, mu_TxK, sigma_KxK) — T×K mean, K×K covariance
    """

    @pytest.mark.parity
    def test_mvnormloglik_parity_case1(self, mvn_fixture: dict) -> None:
        """Parity case 1: mvnormloglik(x) — default mu and sigma.

        Ref: fixture case1 — no mu, sigma=eye(K).
        """
        x = mvn_fixture["x"]
        expected_LL = float(mvn_fixture["case1_LL"])
        expected_lls = mvn_fixture["case1_lls"]

        LL, lls = mvnormloglik(x)

        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Parity case1 LL")
        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Parity case1 lls")

    @pytest.mark.parity
    def test_mvnormloglik_parity_case2(self, mvn_fixture: dict) -> None:
        """Parity case 2: mvnormloglik(x, mu_K) — K×1 mean, sigma=eye(K).

        Ref: fixture case2 — has_mu=True, mu_shape=K, has_sigma=False.
        """
        x = mvn_fixture["x"]
        mu = mvn_fixture["mu_K"]
        expected_LL = float(mvn_fixture["case2_LL"])
        expected_lls = mvn_fixture["case2_lls"]

        LL, lls = mvnormloglik(x, mu=mu)

        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Parity case2 LL")
        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Parity case2 lls")

    @pytest.mark.parity
    def test_mvnormloglik_parity_case3(self, mvn_fixture: dict) -> None:
        """Parity case 3: mvnormloglik(x, mu_K, sigma_KxK).

        Ref: fixture case3 — K×1 mean, K×K covariance.
        """
        x = mvn_fixture["x"]
        mu = mvn_fixture["mu_K"]
        sigma = mvn_fixture["sigma_KxK"]
        expected_LL = float(mvn_fixture["case3_LL"])
        expected_lls = mvn_fixture["case3_lls"]

        LL, lls = mvnormloglik(x, mu=mu, sigma=sigma)

        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Parity case3 LL")
        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Parity case3 lls")

    @pytest.mark.parity
    def test_mvnormloglik_parity_case4(self, mvn_fixture: dict) -> None:
        """Parity case 4: mvnormloglik(x, mu_TxK, sigma_KxK).

        Ref: fixture case4 — T×K mean, K×K covariance.
        """
        x = mvn_fixture["x"]
        mu = mvn_fixture["mu_TxK"]
        sigma = mvn_fixture["sigma_KxK"]
        expected_LL = float(mvn_fixture["case4_LL"])
        expected_lls = mvn_fixture["case4_lls"]

        LL, lls = mvnormloglik(x, mu=mu, sigma=sigma)

        npt.assert_allclose(LL, expected_LL, atol=ATOL, rtol=RTOL,
                            err_msg="Parity case4 LL")
        npt.assert_allclose(lls, expected_lls, atol=ATOL, rtol=RTOL,
                            err_msg="Parity case4 lls")

    @pytest.mark.parity
    def test_mvnormloglik_parity_all_ll(self, mvn_fixture: dict) -> None:
        """Verify all 4 LL values match fixture all_LL vector.

        Cross-check that each case's total LL is consistent with the
        summary all_LL array stored in the fixture.
        """
        expected_all = mvn_fixture["all_LL"]
        x = mvn_fixture["x"]
        mu_k = mvn_fixture["mu_K"]
        mu_txk = mvn_fixture["mu_TxK"]
        sigma = mvn_fixture["sigma_KxK"]

        actual_all = np.zeros(4)
        actual_all[0], _ = mvnormloglik(x)
        actual_all[1], _ = mvnormloglik(x, mu=mu_k)
        actual_all[2], _ = mvnormloglik(x, mu=mu_k, sigma=sigma)
        actual_all[3], _ = mvnormloglik(x, mu=mu_txk, sigma=sigma)

        npt.assert_allclose(actual_all, expected_all, atol=ATOL, rtol=RTOL,
                            err_msg="Parity all_LL vector")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — Additional Robustness Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMvnormloglikRobustness:
    """Additional robustness and integration tests."""

    def test_mvnormloglik_return_types(self, mvn_x: np.ndarray) -> None:
        """Verify return types: LL is float, lls is numpy.ndarray."""
        LL, lls = mvnormloglik(mvn_x)
        assert isinstance(LL, float), f"LL should be float, got {type(LL)}"
        assert isinstance(lls, np.ndarray), (
            f"lls should be np.ndarray, got {type(lls)}"
        )

    def test_mvnormloglik_lls_dtype(self, mvn_x: np.ndarray) -> None:
        """Verify lls dtype is float64."""
        _, lls = mvnormloglik(mvn_x)
        assert lls.dtype == np.float64, f"lls dtype should be float64, got {lls.dtype}"

    def test_mvnormloglik_lls_length(self, mvn_x: np.ndarray) -> None:
        """Verify lls has T elements (one per observation)."""
        T = mvn_x.shape[0]
        _, lls = mvnormloglik(mvn_x)
        assert len(lls) == T, f"Expected {T} elements, got {len(lls)}"

    def test_mvnormloglik_diagonal_sigma_decompose(self) -> None:
        """Diagonal sigma: each dimension is independent.

        With diagonal sigma, the MVN log-lik equals the sum of K independent
        normal log-likelihoods with respective variances on the diagonal.
        """
        rng = np.random.default_rng(90)
        T = 60
        K = 3
        x = rng.standard_normal((T, K))
        variances = np.array([0.5, 2.0, 1.5])
        sigma_diag = np.diag(variances)

        LL, lls = mvnormloglik(x, sigma=sigma_diag)

        # Manually compute as sum of K independent normals
        manual_lls = np.zeros(T)
        for k in range(K):
            manual_lls += -0.5 * (
                np.log(2.0 * np.pi)
                + np.log(variances[k])
                + x[:, k] ** 2 / variances[k]
            )

        npt.assert_allclose(lls, manual_lls, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(LL, np.sum(manual_lls), atol=ATOL, rtol=RTOL)

    def test_mvnormloglik_scaled_identity(self) -> None:
        """sigma = c * I should scale like N(0, cI).

        For sigma = c * I_K, the log-det is K*log(c) and the quadratic form
        is (1/c)*sum(x^2).
        """
        rng = np.random.default_rng(91)
        T = 40
        K = 3
        x = rng.standard_normal((T, K))
        c = 3.0
        sigma = c * np.eye(K)

        LL, lls = mvnormloglik(x, sigma=sigma)
        ref_lls = stats.multivariate_normal.logpdf(
            x, mean=np.zeros(K), cov=sigma
        )
        npt.assert_allclose(lls, ref_lls, atol=ATOL, rtol=RTOL)

    def test_mvnormloglik_input_not_modified(self) -> None:
        """Verify that the function does not modify input arrays in-place.

        Original x, mu, sigma should remain unchanged after the call.
        """
        rng = np.random.default_rng(92)
        T = 30
        K = 3
        x_orig = rng.standard_normal((T, K)).copy()
        mu_orig = np.array([1.0, -0.5, 0.2]).copy()
        sigma_orig = np.eye(K).copy()

        x_copy = x_orig.copy()
        mu_copy = mu_orig.copy()
        sigma_copy = sigma_orig.copy()

        mvnormloglik(x_copy, mu=mu_copy, sigma=sigma_copy)

        npt.assert_array_equal(x_copy, x_orig, err_msg="x was modified")
        npt.assert_array_equal(mu_copy, mu_orig, err_msg="mu was modified")
        npt.assert_array_equal(sigma_copy, sigma_orig,
                               err_msg="sigma was modified")

    @pytest.mark.parametrize("K", [1, 2, 3, 5, 8])
    def test_mvnormloglik_parametrized_dimensions(self, K: int) -> None:
        """Parametrised test across several K values with identity sigma.

        Verifies the function works correctly for a range of dimensionalities.
        """
        rng = np.random.default_rng(100 + K)
        T = 50
        x = rng.standard_normal((T, K))

        LL, lls = mvnormloglik(x, sigma=np.eye(K))
        ref_lls = stats.multivariate_normal.logpdf(
            x, mean=np.zeros(K), cov=np.eye(K)
        )

        npt.assert_allclose(lls, ref_lls, atol=ATOL, rtol=RTOL,
                            err_msg=f"K={K} identity sigma")
        npt.assert_allclose(LL, np.sum(ref_lls), atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("T", [1, 5, 50, 200])
    def test_mvnormloglik_parametrized_samples(self, T: int) -> None:
        """Parametrised test across several T values (K=3 fixed).

        Verifies the function handles different sample sizes correctly.
        """
        rng = np.random.default_rng(200 + T)
        K = 3
        x = rng.standard_normal((T, K))

        LL, lls = mvnormloglik(x, sigma=np.eye(K))
        ref_lls = stats.multivariate_normal.logpdf(
            x, mean=np.zeros(K), cov=np.eye(K)
        )

        npt.assert_allclose(lls, ref_lls, atol=ATOL, rtol=RTOL,
                            err_msg=f"T={T} identity sigma")
