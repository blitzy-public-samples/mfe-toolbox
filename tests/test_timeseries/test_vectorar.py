"""
Comprehensive pytest tests for ``mfe_toolbox.timeseries.vectorar``.

Tests VAR(P) model estimation with 4 VCV estimation modes (het × uncorr
combinations).  Covers output structure, statistical properties, edge cases,
and fixture-based numerical parity against MATLAB/Octave reference outputs.

Source MATLAB Reference
-----------------------
- ``timeseries/vectorar.m`` by Kevin Sheppard (Revision 3.1, 3/1/2014)
- Function signature::

    [parameters, stderr, tstat, pval, const, conststd, r2, errors,
     s2, paramvec, vcv] = vectorar(y, constant, lags, het, uncorr)

The function estimates a Vector Autoregression and produces the parameter
variance-covariance matrix under 4 assumptions about error covariance:

1. Conditionally Homoskedastic and Uncorrelated (het=0, uncorr=1)
2. Conditionally Homoskedastic but Correlated (het=0, uncorr=0)
3. Heteroskedastic but Conditionally Uncorrelated (het=1, uncorr=1)
4. Heteroskedastic and Correlated (het=1, uncorr=0) [DEFAULT]

Per AAP Section 0.7.1: All migrated functions MUST pass
``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
against MATLAB-generated fixtures.
"""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.vectorar import vectorar

# ---------------------------------------------------------------------------
# Numerical Parity Tolerance Constants
# Per AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# ---------------------------------------------------------------------------
# Fixture Directory Resolution
# Per AAP Section 0.7.2: Optional MFE_FIXTURE_DIR override
# ---------------------------------------------------------------------------
_FIXTURE_BASE: Path = Path(__file__).resolve().parent.parent / "fixtures"
_TIMESERIES_FIXTURE_DIR: Path = Path(
    os.environ.get("MFE_FIXTURE_DIR", str(_FIXTURE_BASE))
) / "timeseries"

_VECTORAR_FIXTURE_PATH: Path = _TIMESERIES_FIXTURE_DIR / "vectorar.npy"
_HAS_FIXTURES: bool = _VECTORAR_FIXTURE_PATH.exists()


# ===================================================================
# Fixtures — Data Generation
# ===================================================================


@pytest.fixture
def var1_data():
    """Generate a bivariate VAR(1) DGP with known coefficient matrix.

    Uses a local RNG seed (12345) independent of conftest.rng to avoid
    session-order dependency.  The coefficient matrix A has spectral
    radius < 1 (eigenvalues ≈ 0.3 and 0.6) ensuring stationarity.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y, A)`` where ``y`` has shape ``(500, 2)`` and ``A`` is
        the ``(2, 2)`` true coefficient matrix.
    """
    rng = np.random.default_rng(12345)
    T, K = 500, 2
    A = np.array([[0.5, 0.1], [0.2, 0.4]])
    y = np.zeros((T, K))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.standard_normal(K)
    return y, A


@pytest.fixture
def trivariate_data():
    """Generate a trivariate VAR(1) DGP with known coefficient matrix.

    Returns
    -------
    np.ndarray
        ``(300, 3)`` array of trivariate VAR data.
    """
    rng = np.random.default_rng(54321)
    T, K = 300, 3
    # Coefficient matrix with spectral radius < 1
    A = np.array([
        [0.3, 0.05, -0.02],
        [0.1, 0.4, 0.03],
        [-0.05, 0.02, 0.35],
    ])
    y = np.zeros((T, K))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.standard_normal(K)
    return y


@pytest.fixture
def large_bivariate_data():
    """Generate a large bivariate dataset for precise estimation tests.

    Returns
    -------
    np.ndarray
        ``(2000, 2)`` array of bivariate data.
    """
    rng = np.random.default_rng(99999)
    T, K = 2000, 2
    A = np.array([[0.4, 0.05], [0.1, 0.3]])
    y = np.zeros((T, K))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.standard_normal(K)
    return y


@pytest.fixture(scope="module")
def fixture_data():
    """Load MATLAB reference fixture data if available.

    Returns
    -------
    dict or None
        Fixture dictionary with test cases, or ``None`` if fixture
        file is not found.
    """
    if not _HAS_FIXTURES:
        return None
    return np.load(_VECTORAR_FIXTURE_PATH, allow_pickle=True).item()


# ===================================================================
# Helper Functions
# ===================================================================


def _run_vectorar(y, constant, lags, het=1, uncorr=0):
    """Call vectorar and unpack the 11-element tuple for convenient access."""
    result = vectorar(y, constant, np.asarray(lags), het=het, uncorr=uncorr)
    return result


# ===================================================================
# Phase 2: Output Structure Tests
# ===================================================================


class TestOutputStructure:
    """Test that vectorar returns correct output types and shapes."""

    def test_vectorar_returns_eleven(self, var1_data):
        """Verify that vectorar returns a tuple of exactly 11 elements."""
        y, _ = var1_data
        result = vectorar(y, 1, np.array([1]))
        assert isinstance(result, tuple), "Return value must be a tuple"
        assert len(result) == 11, f"Expected 11 elements, got {len(result)}"

    def test_vectorar_parameters_structure(self, var1_data):
        """Verify ``parameters`` is a list of K×K matrices, one per lag."""
        y, _ = var1_data
        K = y.shape[1]
        parameters, *_ = vectorar(y, 1, np.array([1]))

        assert isinstance(parameters, list), "parameters must be a list"
        assert len(parameters) == 1, "VAR(1) parameters list length must be 1"
        assert parameters[0] is not None, "parameters[0] must be populated"
        assert parameters[0].shape == (K, K), (
            f"Expected parameter matrix shape ({K}, {K}), "
            f"got {parameters[0].shape}"
        )

    def test_vectorar_stderr_shape(self, var1_data):
        """Verify stderr has same structure as parameters, all positive."""
        y, _ = var1_data
        K = y.shape[1]
        _, stderr, *_ = vectorar(y, 1, np.array([1]))

        assert isinstance(stderr, list), "stderr must be a list"
        assert len(stderr) == 1, "VAR(1) stderr list length must be 1"
        assert stderr[0].shape == (K, K), (
            f"Expected stderr shape ({K}, {K}), got {stderr[0].shape}"
        )
        # All standard errors must be strictly positive
        assert np.all(stderr[0] > 0), "All standard errors must be positive"

    def test_vectorar_const_shape(self, var1_data):
        """Verify const is a K-length vector when constant=1."""
        y, _ = var1_data
        K = y.shape[1]
        result = vectorar(y, 1, np.array([1]))
        const = result[4]

        assert const is not None, "const must not be None when constant=1"
        assert const.shape == (K,), (
            f"Expected const shape ({K},), got {const.shape}"
        )

    def test_vectorar_r2_range(self, var1_data):
        """Verify all R² values are in [0, 1]."""
        y, _ = var1_data
        result = vectorar(y, 1, np.array([1]))
        r2 = result[6]

        assert r2.ndim == 1, "r2 must be 1-D"
        assert r2.shape[0] == y.shape[1], (
            f"r2 length {r2.shape[0]} != K={y.shape[1]}"
        )
        assert np.all(r2 >= 0.0), f"R² values must be >= 0, got min={r2.min()}"
        assert np.all(r2 <= 1.0), f"R² values must be <= 1, got max={r2.max()}"

    def test_vectorar_errors_shape(self, var1_data):
        """Verify errors matrix has shape (T - max(lags), K)."""
        y, _ = var1_data
        T, K = y.shape
        lags = np.array([1])
        m = int(np.max(lags))
        result = vectorar(y, 1, lags)
        errors = result[7]

        expected_rows = T - m
        assert errors.shape == (expected_rows, K), (
            f"Expected errors shape ({expected_rows}, {K}), "
            f"got {errors.shape}"
        )

    def test_vectorar_s2_symmetric(self, var1_data):
        """Verify s2 is a K×K symmetric matrix."""
        y, _ = var1_data
        K = y.shape[1]
        result = vectorar(y, 1, np.array([1]))
        s2 = result[8]

        assert s2.shape == (K, K), f"Expected s2 shape ({K}, {K}), got {s2.shape}"
        # Ref: vectorar.m:172 — s2 = errors' * errors / T
        npt.assert_allclose(s2, s2.T, atol=ATOL,
                            err_msg="s2 must be symmetric")

    def test_vectorar_s2_psd(self, var1_data):
        """Verify s2 is positive semi-definite (eigenvalues >= -ATOL)."""
        y, _ = var1_data
        result = vectorar(y, 1, np.array([1]))
        s2 = result[8]

        eigenvalues = np.linalg.eigh(s2)[0]
        assert np.all(eigenvalues >= -ATOL), (
            f"s2 must be positive semi-definite; "
            f"min eigenvalue = {eigenvalues.min()}"
        )

    def test_vectorar_vcv_symmetric(self, var1_data):
        """Verify the parameter VCV matrix is symmetric."""
        y, _ = var1_data
        result = vectorar(y, 1, np.array([1]))
        vcv = result[10]

        assert vcv.ndim == 2, "VCV must be 2-D"
        assert vcv.shape[0] == vcv.shape[1], "VCV must be square"
        npt.assert_allclose(vcv, vcv.T, atol=ATOL,
                            err_msg="VCV must be symmetric")

    def test_vectorar_paramvec_length(self, var1_data):
        """Verify paramvec has correct length: K * (K * p + constant)."""
        y, _ = var1_data
        K = y.shape[1]
        constant = 1
        lags = np.array([1])
        p = int(np.sum(lags > 0))  # number of positive lags
        # Ref: vectorar.m:41-48 — paramvec length is K * Np
        # where Np = K * p + constant
        expected_length = K * (K * p + constant)

        result = vectorar(y, constant, lags)
        paramvec = result[9]

        assert paramvec.ndim == 1, "paramvec must be 1-D"
        assert paramvec.shape[0] == expected_length, (
            f"Expected paramvec length {expected_length}, "
            f"got {paramvec.shape[0]}"
        )


# ===================================================================
# Phase 3: Statistical Properties Tests
# ===================================================================


class TestStatisticalProperties:
    """Test statistical correctness of VAR estimation."""

    def test_vectorar_known_dgp(self, var1_data):
        """VAR(1) with known A matrix: estimated params ≈ true A.

        With T=500, finite sample tolerance is generous (atol=0.15).
        The true DGP is y(t) = A @ y(t-1) + e(t) with:
            A = [[0.5, 0.1], [0.2, 0.4]]
        """
        y, A_true = var1_data
        parameters, *_ = vectorar(y, 1, np.array([1]))

        # parameters[0] is the estimated (K, K) coefficient matrix at lag 1
        A_hat = parameters[0]
        assert A_hat is not None, "Parameters at lag 1 must be populated"

        # Finite sample tolerance — with T=500, estimates should be
        # within ~0.15 of the true values
        npt.assert_allclose(
            A_hat, A_true, atol=0.15,
            err_msg="Estimated VAR(1) coefficients should approximate "
                    "true DGP parameters within finite sample tolerance"
        )

    def test_vectorar_no_constant(self, var1_data):
        """With constant=0, const and conststd must be None."""
        y, _ = var1_data
        result = vectorar(y, 0, np.array([1]))
        const = result[4]
        conststd = result[5]

        assert const is None, (
            "const must be None when constant=0"
        )
        assert conststd is None, (
            "conststd must be None when constant=0"
        )

    def test_vectorar_noncontiguous_lags(self, trivariate_data):
        """With lags=[1, 3], only lag 1 and lag 3 positions populated."""
        y = trivariate_data
        K = y.shape[1]
        lags = np.array([1, 3])
        parameters, stderr, tstat, pval, *_ = vectorar(y, 1, lags)

        # List length should be max(lags) = 3
        assert len(parameters) == 3, (
            f"parameters list length must be max(lags)=3, got {len(parameters)}"
        )

        # Lag 1 (index 0) — populated
        assert parameters[0] is not None, "parameters[0] (lag 1) must be populated"
        assert parameters[0].shape == (K, K)

        # Lag 2 (index 1) — NOT requested, must be None
        assert parameters[1] is None, "parameters[1] (lag 2) must be None"

        # Lag 3 (index 2) — populated
        assert parameters[2] is not None, "parameters[2] (lag 3) must be populated"
        assert parameters[2].shape == (K, K)

        # Same structure for stderr
        assert stderr[0] is not None
        assert stderr[1] is None
        assert stderr[2] is not None


# ===================================================================
# Phase 4: VCV Mode Tests (all 4 combinations)
# ===================================================================


class TestVCVModes:
    """Test all 4 VCV estimation modes (het × uncorr)."""

    def test_vectorar_het0_uncorr0(self, var1_data):
        """Homoskedastic correlated (het=0, uncorr=0)."""
        y, _ = var1_data
        result = _run_vectorar(y, 1, [1], het=0, uncorr=0)

        # VCV must be symmetric and well-defined
        vcv = result[10]
        npt.assert_allclose(vcv, vcv.T, atol=ATOL,
                            err_msg="VCV (het=0, uncorr=0) must be symmetric")
        # Diagonal must be positive (variances)
        assert np.all(np.diag(vcv) > 0), (
            "VCV diagonal (variances) must be positive"
        )

    def test_vectorar_het1_uncorr0(self, var1_data):
        """Heteroskedastic correlated (het=1, uncorr=0) — DEFAULT."""
        y, _ = var1_data
        result = _run_vectorar(y, 1, [1], het=1, uncorr=0)

        vcv = result[10]
        npt.assert_allclose(vcv, vcv.T, atol=ATOL,
                            err_msg="VCV (het=1, uncorr=0) must be symmetric")
        assert np.all(np.diag(vcv) > 0), (
            "VCV diagonal (variances) must be positive"
        )

    def test_vectorar_het0_uncorr1(self, var1_data):
        """Homoskedastic uncorrelated (het=0, uncorr=1)."""
        y, _ = var1_data
        K = y.shape[1]
        result = _run_vectorar(y, 1, [1], het=0, uncorr=1)

        vcv = result[10]
        npt.assert_allclose(vcv, vcv.T, atol=ATOL,
                            err_msg="VCV (het=0, uncorr=1) must be symmetric")
        assert np.all(np.diag(vcv) > 0), (
            "VCV diagonal (variances) must be positive"
        )

        # For uncorrelated mode, off-diagonal blocks should be zero
        # VCV is (K*Np, K*Np). For uncorr=1, cross-equation covariances = 0.
        Np = vcv.shape[0] // K
        for i in range(K):
            for j in range(K):
                if i != j:
                    block = vcv[i * Np:(i + 1) * Np,
                                j * Np:(j + 1) * Np]
                    npt.assert_allclose(
                        block, np.zeros_like(block), atol=ATOL,
                        err_msg=f"Off-diagonal block ({i},{j}) must be "
                                f"zero for uncorrelated VCV"
                    )

    def test_vectorar_het1_uncorr1(self, var1_data):
        """Heteroskedastic uncorrelated (het=1, uncorr=1)."""
        y, _ = var1_data
        K = y.shape[1]
        result = _run_vectorar(y, 1, [1], het=1, uncorr=1)

        vcv = result[10]
        npt.assert_allclose(vcv, vcv.T, atol=ATOL,
                            err_msg="VCV (het=1, uncorr=1) must be symmetric")
        assert np.all(np.diag(vcv) > 0), (
            "VCV diagonal (variances) must be positive"
        )

        # Uncorrelated: cross-equation blocks should be zero
        Np = vcv.shape[0] // K
        for i in range(K):
            for j in range(K):
                if i != j:
                    block = vcv[i * Np:(i + 1) * Np,
                                j * Np:(j + 1) * Np]
                    npt.assert_allclose(
                        block, np.zeros_like(block), atol=ATOL,
                        err_msg=f"Off-diagonal block ({i},{j}) must be "
                                f"zero for uncorrelated VCV"
                    )

    def test_vectorar_vcv_modes_differ(self, var1_data):
        """Different VCV modes produce different VCV matrices.

        The four VCV estimation assumptions (het × uncorr) should
        generally yield different VCV matrices for the same data.
        """
        y, _ = var1_data
        vcvs = {}
        for het in (0, 1):
            for uncorr in (0, 1):
                result = _run_vectorar(y, 1, [1], het=het, uncorr=uncorr)
                vcvs[(het, uncorr)] = result[10]

        # At least some pairs should differ
        pairs_differ = 0
        keys = list(vcvs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                diff = np.linalg.norm(vcvs[keys[i]] - vcvs[keys[j]])
                if diff > 1e-10:
                    pairs_differ += 1

        assert pairs_differ > 0, (
            "At least some VCV mode pairs should produce different results"
        )


# ===================================================================
# Phase 5: Edge Cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_vectorar_trivariate(self, trivariate_data):
        """K=3 multivariate data works correctly."""
        y = trivariate_data
        T, K = y.shape
        assert K == 3, "Sanity check: trivariate data must have K=3"

        result = vectorar(y, 1, np.array([1]))
        parameters, stderr, tstat, pval, const, conststd, r2, errors, s2, paramvec, vcv = result

        # Structural checks for K=3
        assert parameters[0].shape == (3, 3)
        assert stderr[0].shape == (3, 3)
        assert const.shape == (3,)
        assert r2.shape == (3,)
        assert errors.shape == (T - 1, 3)
        assert s2.shape == (3, 3)
        # paramvec: K * (K * p + constant) = 3 * (3*1 + 1) = 12
        assert paramvec.shape == (12,)
        assert vcv.shape == (12, 12)

    def test_vectorar_residual_orthogonality(self, var1_data):
        """X'errors ≈ 0 — the OLS orthogonality condition.

        The OLS residuals must be orthogonal to the regressor matrix X.
        We verify this by checking that the residuals are orthogonal
        to the column space of regressors.

        Since vectorar doesn't return X directly, we reconstruct it
        from paramvec, errors, and Y = X @ paramvec + errors.
        """
        y, _ = var1_data
        T, K = y.shape
        result = vectorar(y, 1, np.array([1]))
        _, _, _, _, _, _, _, errors, _, paramvec_flat, _ = result

        # Reconstruct X and Y
        # Y = y[1:, :] (lag 1, so m=1)
        m = 1
        Y = y[m:, :]

        # X = [ones, y_{t-1}] where y_{t-1} is the lag-1 matrix
        T_eff = T - m
        X = np.zeros((T_eff, 1 + K))
        X[:, 0] = 1.0  # constant
        X[:, 1:] = y[:T - m, :]  # lag 1

        # OLS orthogonality: X.T @ errors ≈ 0
        orth_check = X.T @ errors
        npt.assert_allclose(
            orth_check, np.zeros_like(orth_check), atol=1e-8,
            err_msg="X'errors must be approximately zero (OLS condition)"
        )

    def test_vectorar_pval_range(self, var1_data):
        """All p-values must be in [0, 1]."""
        y, _ = var1_data
        _, _, _, pval, _, _, _, _, _, _, _ = vectorar(y, 1, np.array([1]))

        for i, p in enumerate(pval):
            if p is not None:
                assert np.all(p >= 0.0), (
                    f"p-values at lag {i + 1} must be >= 0"
                )
                assert np.all(p <= 1.0), (
                    f"p-values at lag {i + 1} must be <= 1"
                )

    def test_vectorar_tstat_consistent(self, var1_data):
        """t-statistics = parameters / stderr for populated lag positions."""
        y, _ = var1_data
        parameters, stderr, tstat, *_ = vectorar(y, 1, np.array([1]))

        for i in range(len(parameters)):
            if parameters[i] is not None:
                expected_tstat = parameters[i] / stderr[i]
                npt.assert_allclose(
                    tstat[i], expected_tstat, atol=ATOL,
                    err_msg=f"tstat at lag {i + 1} must equal "
                            f"parameters / stderr"
                )

    def test_vectorar_s2_matches_errors(self, var1_data):
        """s2 == errors.T @ errors / T (original T, not T-m).

        Ref: vectorar.m:172 — s2=errors'*errors/T
        """
        y, _ = var1_data
        T = y.shape[0]
        result = vectorar(y, 1, np.array([1]))
        errors = result[7]
        s2 = result[8]

        # Ref: vectorar.m:172 — T is the ORIGINAL number of observations
        expected_s2 = errors.T @ errors / T
        npt.assert_allclose(s2, expected_s2, atol=ATOL,
                            err_msg="s2 must equal errors.T @ errors / T")

    def test_vectorar_var2(self, large_bivariate_data):
        """VAR(2) model — lags=[1, 2] — produces correct structure."""
        y = large_bivariate_data
        T, K = y.shape
        lags = np.array([1, 2])
        m = 2

        result = vectorar(y, 1, lags)
        parameters, stderr, tstat, pval, const, conststd, r2, errors, s2, paramvec, vcv = result

        # parameters should have length 2 (max lag = 2)
        assert len(parameters) == 2
        assert parameters[0] is not None, "lag 1 parameters must be populated"
        assert parameters[1] is not None, "lag 2 parameters must be populated"
        assert parameters[0].shape == (K, K)
        assert parameters[1].shape == (K, K)

        # Errors shape
        assert errors.shape == (T - m, K)

        # paramvec length: K * (K * 2 + constant) = 2 * (2*2 + 1) = 10
        assert paramvec.shape[0] == K * (K * 2 + 1)


# ===================================================================
# Phase 6: Parity Tests (MATLAB Fixture Comparison)
# ===================================================================


class TestFixtureParity:
    """Tests comparing Python output against MATLAB/Octave reference fixtures.

    These tests load precomputed reference data from
    ``tests/fixtures/timeseries/vectorar.npy`` and verify that the Python
    ``vectorar`` function produces identical results within ATOL/RTOL
    tolerance.

    All tests are skipped if fixture files are not available.
    """

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="MATLAB fixture file not found: vectorar.npy",
    )
    def test_vectorar_parity_default(self, fixture_data):
        """Parity: VAR(1) K=3, constant=1, het=1, uncorr=0 (default)."""
        assert fixture_data is not None
        case = fixture_data["var1_k3"]
        y = fixture_data["y"]

        constant = int(case["constant"])
        lags = np.asarray(case["lags"])
        het = int(case["het"])
        uncorr = int(case["uncorr"])

        result = vectorar(y, constant, lags, het=het, uncorr=uncorr)
        parameters, stderr, tstat, pval, const, conststd, r2, errors, s2, paramvec, vcv = result

        # Compare parameters at lag 1
        npt.assert_allclose(
            parameters[0], case["parameters_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: parameters lag 1 mismatch (default)"
        )

        # Compare standard errors at lag 1
        npt.assert_allclose(
            stderr[0], case["stderr_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: stderr lag 1 mismatch (default)"
        )

        # Compare t-statistics
        npt.assert_allclose(
            tstat[0], case["tstat_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: tstat lag 1 mismatch (default)"
        )

        # Compare p-values
        npt.assert_allclose(
            pval[0], case["pval_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: pval lag 1 mismatch (default)"
        )

        # Compare constants
        npt.assert_allclose(
            const, case["const"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: const mismatch (default)"
        )
        npt.assert_allclose(
            conststd, case["conststd"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: conststd mismatch (default)"
        )

        # Compare R²
        npt.assert_allclose(
            r2, case["r2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: r2 mismatch (default)"
        )

        # Compare error covariance
        npt.assert_allclose(
            s2, case["s2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: s2 mismatch (default)"
        )

        # Compare parameter vector (column-major ordering)
        npt.assert_allclose(
            paramvec, case["paramvec"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: paramvec mismatch (default)"
        )

        # Compare VCV
        npt.assert_allclose(
            vcv, case["vcv"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: vcv mismatch (default)"
        )

        # Compare errors (residuals)
        npt.assert_allclose(
            errors, case["errors"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: errors mismatch (default)"
        )

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="MATLAB fixture file not found: vectorar.npy",
    )
    def test_vectorar_parity_var2(self, fixture_data):
        """Parity: VAR(2) K=3, constant=1, het=1, uncorr=0."""
        assert fixture_data is not None
        case = fixture_data["var2_k3"]
        y = fixture_data["y"]

        constant = int(case["constant"])
        lags = np.asarray(case["lags"])
        het = int(case["het"])
        uncorr = int(case["uncorr"])

        result = vectorar(y, constant, lags, het=het, uncorr=uncorr)
        parameters, stderr, tstat, pval, const, conststd, r2, errors, s2, paramvec, vcv = result

        # Compare parameters at lag 1
        npt.assert_allclose(
            parameters[0], case["parameters_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: parameters lag 1 mismatch (VAR2)"
        )

        # Compare parameters at lag 2
        npt.assert_allclose(
            parameters[1], case["parameters_lag2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: parameters lag 2 mismatch (VAR2)"
        )

        # Compare stderr at lag 1 and lag 2
        npt.assert_allclose(
            stderr[0], case["stderr_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: stderr lag 1 mismatch (VAR2)"
        )
        npt.assert_allclose(
            stderr[1], case["stderr_lag2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: stderr lag 2 mismatch (VAR2)"
        )

        # Compare t-statistics at lag 1 and lag 2
        npt.assert_allclose(
            tstat[0], case["tstat_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: tstat lag 1 mismatch (VAR2)"
        )
        npt.assert_allclose(
            tstat[1], case["tstat_lag2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: tstat lag 2 mismatch (VAR2)"
        )

        # Compare constants, r2, s2, paramvec, vcv
        npt.assert_allclose(
            const, case["const"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: const mismatch (VAR2)"
        )
        npt.assert_allclose(
            r2, case["r2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: r2 mismatch (VAR2)"
        )
        npt.assert_allclose(
            s2, case["s2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: s2 mismatch (VAR2)"
        )
        npt.assert_allclose(
            paramvec, case["paramvec"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: paramvec mismatch (VAR2)"
        )
        npt.assert_allclose(
            vcv, case["vcv"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: vcv mismatch (VAR2)"
        )

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="MATLAB fixture file not found: vectorar.npy",
    )
    def test_vectorar_parity_no_constant(self, fixture_data):
        """Parity: VAR(1) K=3, constant=0, het=0, uncorr=0."""
        assert fixture_data is not None
        case = fixture_data["no_constant"]
        y = fixture_data["y"]

        constant = int(case["constant"])
        lags = np.asarray(case["lags"])
        het = int(case["het"])
        uncorr = int(case["uncorr"])

        result = vectorar(y, constant, lags, het=het, uncorr=uncorr)
        parameters, stderr, tstat, pval, const, conststd, r2, errors, s2, paramvec, vcv = result

        # Parameters at lag 1
        npt.assert_allclose(
            parameters[0], case["parameters_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: parameters lag 1 mismatch (no_constant)"
        )

        # Stderr at lag 1
        npt.assert_allclose(
            stderr[0], case["stderr_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: stderr lag 1 mismatch (no_constant)"
        )

        # t-stats
        npt.assert_allclose(
            tstat[0], case["tstat_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: tstat lag 1 mismatch (no_constant)"
        )

        # p-values
        npt.assert_allclose(
            pval[0], case["pval_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: pval lag 1 mismatch (no_constant)"
        )

        # const must be None when constant=0
        assert const is None, (
            "const must be None when constant=0 (parity)"
        )

        # R², s2, paramvec, vcv, errors
        npt.assert_allclose(
            r2, case["r2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: r2 mismatch (no_constant)"
        )
        npt.assert_allclose(
            s2, case["s2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: s2 mismatch (no_constant)"
        )
        npt.assert_allclose(
            paramvec, case["paramvec"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: paramvec mismatch (no_constant)"
        )
        npt.assert_allclose(
            vcv, case["vcv"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: vcv mismatch (no_constant)"
        )
        npt.assert_allclose(
            errors, case["errors"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: errors mismatch (no_constant)"
        )

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="MATLAB fixture file not found: vectorar.npy",
    )
    def test_vectorar_parity_uncorrelated(self, fixture_data):
        """Parity: VAR(1) K=3, constant=1, het=0, uncorr=1."""
        assert fixture_data is not None
        case = fixture_data["uncorrelated"]
        y = fixture_data["y"]

        constant = int(case["constant"])
        lags = np.asarray(case["lags"])
        het = int(case["het"])
        uncorr = int(case["uncorr"])

        result = vectorar(y, constant, lags, het=het, uncorr=uncorr)
        parameters, stderr, tstat, pval, const, conststd, r2, errors, s2, paramvec, vcv = result

        # Parameters at lag 1
        npt.assert_allclose(
            parameters[0], case["parameters_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: parameters lag 1 mismatch (uncorrelated)"
        )

        # Stderr
        npt.assert_allclose(
            stderr[0], case["stderr_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: stderr lag 1 mismatch (uncorrelated)"
        )

        # Constants
        npt.assert_allclose(
            const, case["const"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: const mismatch (uncorrelated)"
        )
        npt.assert_allclose(
            conststd, case["conststd"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: conststd mismatch (uncorrelated)"
        )

        # r2, s2, paramvec, vcv
        npt.assert_allclose(
            r2, case["r2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: r2 mismatch (uncorrelated)"
        )
        npt.assert_allclose(
            s2, case["s2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: s2 mismatch (uncorrelated)"
        )
        npt.assert_allclose(
            paramvec, case["paramvec"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: paramvec mismatch (uncorrelated)"
        )
        npt.assert_allclose(
            vcv, case["vcv"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: vcv mismatch (uncorrelated)"
        )

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="MATLAB fixture file not found: vectorar.npy",
    )
    def test_vectorar_parity_homoskedastic(self, fixture_data):
        """Parity: VAR(1) K=3, constant=1, het=0, uncorr=0."""
        assert fixture_data is not None
        case = fixture_data["homoskedastic"]
        y = fixture_data["y"]

        constant = int(case["constant"])
        lags = np.asarray(case["lags"])
        het = int(case["het"])
        uncorr = int(case["uncorr"])

        result = vectorar(y, constant, lags, het=het, uncorr=uncorr)
        parameters, stderr, tstat, pval, const, conststd, r2, errors, s2, paramvec, vcv = result

        # Parameters at lag 1
        npt.assert_allclose(
            parameters[0], case["parameters_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: parameters lag 1 mismatch (homoskedastic)"
        )

        # Stderr
        npt.assert_allclose(
            stderr[0], case["stderr_lag1"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: stderr lag 1 mismatch (homoskedastic)"
        )

        # Constants
        npt.assert_allclose(
            const, case["const"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: const mismatch (homoskedastic)"
        )
        npt.assert_allclose(
            conststd, case["conststd"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: conststd mismatch (homoskedastic)"
        )

        # r2, s2, paramvec, vcv
        npt.assert_allclose(
            r2, case["r2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: r2 mismatch (homoskedastic)"
        )
        npt.assert_allclose(
            s2, case["s2"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: s2 mismatch (homoskedastic)"
        )
        npt.assert_allclose(
            paramvec, case["paramvec"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: paramvec mismatch (homoskedastic)"
        )
        npt.assert_allclose(
            vcv, case["vcv"], atol=ATOL, rtol=RTOL,
            err_msg="Parity: vcv mismatch (homoskedastic)"
        )


# ===================================================================
# Additional Input Validation Tests
# ===================================================================


class TestInputValidation:
    """Test that vectorar raises appropriate errors on invalid inputs."""

    def test_vectorar_invalid_y_1d(self):
        """Raise ValueError for 1-D input y."""
        y = np.random.default_rng(1).standard_normal(100)
        with pytest.raises(ValueError, match="Y must be T by K"):
            vectorar(y, 1, np.array([1]))

    def test_vectorar_constant_column(self):
        """Raise ValueError when a column of y is constant."""
        rng = np.random.default_rng(2)
        y = rng.standard_normal((100, 2))
        y[:, 1] = 5.0  # constant column
        with pytest.raises(ValueError, match="constant"):
            vectorar(y, 1, np.array([1]))

    def test_vectorar_invalid_constant(self):
        """Raise ValueError for constant not in {0, 1}."""
        rng = np.random.default_rng(3)
        y = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match="CONSTANT"):
            vectorar(y, 2, np.array([1]))

    def test_vectorar_negative_lags(self):
        """Raise ValueError for negative lag values."""
        rng = np.random.default_rng(4)
        y = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match="nonnegative"):
            vectorar(y, 1, np.array([-1]))

    def test_vectorar_fractional_lags(self):
        """Raise ValueError for non-integer lag values."""
        rng = np.random.default_rng(5)
        y = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match="integers"):
            vectorar(y, 1, np.array([1.5]))

    def test_vectorar_duplicate_lags(self):
        """Raise ValueError for duplicate lag values."""
        rng = np.random.default_rng(6)
        y = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match="unique"):
            vectorar(y, 1, np.array([1, 1]))

    def test_vectorar_lag0_no_constant(self):
        """Raise ValueError for lags=0 without constant."""
        rng = np.random.default_rng(7)
        y = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match="constant must be included"):
            vectorar(y, 0, np.array([0]))

    def test_vectorar_invalid_het(self):
        """Raise ValueError for het not in {0, 1}."""
        rng = np.random.default_rng(8)
        y = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match="HET"):
            vectorar(y, 1, np.array([1]), het=2)

    def test_vectorar_invalid_uncorr(self):
        """Raise ValueError for uncorr not in {0, 1}."""
        rng = np.random.default_rng(9)
        y = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match="UNCORR"):
            vectorar(y, 1, np.array([1]), uncorr=3)
