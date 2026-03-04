"""
Pytest tests for mfe_toolbox.timeseries.olsnw — OLS with Newey-West HAC standard errors.

Migrated from timeseries/olsnw.m (Kevin Sheppard, MFE Toolbox v4.0).
Tests cover:
  - Output structure (7-tuple return)
  - Output shapes (b, tstat, vcvnw, yhat) for both c=0 and c=1
  - Statistical properties (coefficient estimation, R-squared, residual orthogonality)
  - Perfect-fit scenario (noiseless data)
  - White HC vs Newey-West HAC distinction (nwlags=0 vs nwlags>0)
  - Edge cases (single regressor, no constant)
  - MATLAB parity against Octave-generated fixtures (4 test cases)

Numerical tolerances per AAP Section 0.7.1:
    ATOL = 1e-6
    RTOL = 1e-4
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.olsnw import olsnw

# Conftest helpers — load_fixture_npy is a plain function imported directly;
# rng and timeseries_fixture_dir are pytest fixtures injected by the
# conftest.py session-scoped fixture infrastructure.
from tests.conftest import load_fixture_npy

# ---------------------------------------------------------------------------
# Tolerance constants (match AAP Section 0.7.1)
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data(rng):
    """Generate T=200 regression data (y, x) for olsnw tests.

    DGP: y = 1.0 + 2.0*x1 - 0.5*x2 + epsilon
    where epsilon ~ N(0, 1) i.i.d.

    Parameters
    ----------
    rng : numpy.random.Generator
        Session-scoped seeded RNG from conftest (seed=42).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y, x)`` where ``y`` has shape ``(200,)`` and ``x`` has shape
        ``(200, 2)``.
    """
    T = 200
    x = rng.standard_normal((T, 2))
    beta = np.array([1.0, 2.0, -0.5])  # const, b1, b2
    y = beta[0] + x @ beta[1:] + rng.standard_normal(T)
    return y, x


# ===========================================================================
# Phase 2: Output Structure and Shape Tests
# ===========================================================================


def test_olsnw_returns_seven(regression_data):
    """olsnw must return a 7-tuple.

    Ref: olsnw.m:1 — [b, tstat, s2, vcvnw, R2, Rbar, yhat] = olsnw(y,x,c,nwlags)
    """
    y, x = regression_data
    result = olsnw(y, x)
    assert isinstance(result, tuple), "Return value must be a tuple"
    assert len(result) == 7, f"Expected 7 elements, got {len(result)}"


def test_olsnw_b_shape_with_constant(regression_data):
    """With c=1, b should have K+1 elements (constant first).

    Ref: olsnw.m:17 — B is K(+1 if C=1) vector; constant is first element.
    """
    y, x = regression_data
    K = x.shape[1]
    b, *_ = olsnw(y, x, c=1)
    assert b.shape == (K + 1,), f"Expected b.shape=({K + 1},), got {b.shape}"


def test_olsnw_b_shape_no_constant(regression_data):
    """With c=0, b should have K elements (no intercept).

    Ref: olsnw.m:100-105 — c=0 skips constant column prepension.
    """
    y, x = regression_data
    K = x.shape[1]
    b, *_ = olsnw(y, x, c=0)
    assert b.shape == (K,), f"Expected b.shape=({K},), got {b.shape}"


def test_olsnw_tstat_shape(regression_data):
    """tstat must have the same length as b.

    Ref: olsnw.m:18 — TSTAT is K(+1) vector of t-statistics.
    """
    y, x = regression_data
    b, tstat, *_ = olsnw(y, x, c=1)
    assert tstat.shape == b.shape, (
        f"tstat.shape={tstat.shape} != b.shape={b.shape}"
    )


def test_olsnw_s2_positive(regression_data):
    """Error variance s2 must be strictly positive for noisy data.

    Ref: olsnw.m:19 — S2 is estimated error variance (Newey-West HAC).
    """
    y, x = regression_data
    _, _, s2, *_ = olsnw(y, x)
    assert s2 > 0, f"s2 must be positive, got {s2}"


def test_olsnw_vcvnw_symmetric(regression_data):
    """Newey-West VCV matrix must be symmetric.

    Ref: olsnw.m:119 — vcvnw = XpXi * covnw(scores) * XpXi / T is
    symmetric by construction (sandwich with symmetric covnw inner).
    """
    y, x = regression_data
    _, _, _, vcvnw, *_ = olsnw(y, x)
    npt.assert_allclose(
        vcvnw, vcvnw.T, atol=ATOL,
        err_msg="VCV matrix is not symmetric",
    )


def test_olsnw_vcvnw_psd(regression_data):
    """Newey-West VCV matrix must be positive semi-definite.

    The Bartlett kernel guarantees PSD for any finite NW lag length.
    """
    y, x = regression_data
    _, _, _, vcvnw, *_ = olsnw(y, x)
    eigenvalues = np.linalg.eigvalsh(vcvnw)
    assert np.all(eigenvalues >= -ATOL), (
        f"VCV has negative eigenvalues: {eigenvalues[eigenvalues < -ATOL]}"
    )


def test_olsnw_r2_range(regression_data):
    """R-squared must be in [0, 1] when a constant is included.

    Ref: olsnw.m:125-128 — Centered R-squared when c=1.
    """
    y, x = regression_data
    _, _, _, _, r2, *_ = olsnw(y, x, c=1)
    assert 0.0 <= r2 <= 1.0, f"R-squared out of range: {r2}"


def test_olsnw_rbar_leq_r2(regression_data):
    """Adjusted R-squared should be <= R-squared.

    Ref: olsnw.m:128 — Rbar = 1 - (SSR/SST)*(T-1)/(T-K); the
    degrees-of-freedom adjustment never exceeds R-squared.
    """
    y, x = regression_data
    _, _, _, _, r2, rbar, _ = olsnw(y, x, c=1)
    assert rbar <= r2 + ATOL, f"Rbar={rbar} > R2={r2}"


def test_olsnw_yhat_shape(regression_data):
    """Fitted values yhat must have the same length as y.

    Ref: olsnw.m:23 — YHAT is T-by-1 vector of fitted values.
    """
    y, x = regression_data
    *_, yhat = olsnw(y, x)
    assert yhat.shape == y.shape, (
        f"yhat.shape={yhat.shape} != y.shape={y.shape}"
    )


def test_olsnw_yhat_equals_xb(regression_data):
    """Fitted values must equal X*b (with constant column prepended when c=1).

    Ref: olsnw.m:111 — yhat = x * b where x includes the constant column.
    """
    y, x = regression_data
    T = y.shape[0]
    b, _, _, _, _, _, yhat = olsnw(y, x, c=1)
    X_full = np.column_stack([np.ones(T), x])
    npt.assert_allclose(
        yhat, X_full @ b, atol=ATOL,
        err_msg="yhat != X_full @ b",
    )


# ===========================================================================
# Phase 3: Statistical Properties Tests
# ===========================================================================


def test_olsnw_known_regression(regression_data):
    """Known DGP: estimated b should be close to true parameters.

    DGP: y = 1.0 + 2.0*x1 - 0.5*x2 + N(0,1)
    With T=200 and unit noise variance, OLS estimates converge to the true
    values. Using a generous tolerance of 0.5 (approximately 7 standard
    errors for T=200) ensures the test passes with overwhelming probability
    regardless of the specific RNG draw.
    """
    y, x = regression_data
    b, *_ = olsnw(y, x, c=1)
    true_beta = np.array([1.0, 2.0, -0.5])
    npt.assert_allclose(
        b, true_beta, atol=0.5,
        err_msg="OLS coefficients far from true DGP values",
    )


def test_olsnw_perfect_fit(rng):
    """When y = X*b exactly (no noise), R-squared must be 1 and s2 must be 0.

    Ref: olsnw.m:126-128 — With zero residuals, SSR=0 so R2=1.
    """
    T = 100
    x = rng.standard_normal((T, 2))
    b_true = np.array([3.0, 1.5, -2.0])  # const, b1, b2
    # Noiseless DGP — y is an exact linear combination of regressors
    y = b_true[0] + x @ b_true[1:]
    b, _, s2, _, r2, rbar, yhat = olsnw(y, x, c=1)
    assert r2 > 1.0 - ATOL, f"R-squared should be ~1 for perfect fit, got {r2}"
    assert abs(s2) < ATOL, f"s2 should be ~0 for perfect fit, got {s2}"
    npt.assert_allclose(
        b, b_true, atol=ATOL,
        err_msg="Coefficients should match true values exactly for noiseless DGP",
    )


def test_olsnw_white_vs_nw(regression_data):
    """White HC (nwlags=0) and NW HAC (nwlags>0) should yield different VCVs.

    Ref: olsnw.m:13-14 — nwlags=0 estimates White's HC; nwlags>0 adds
    autocovariance correction via the Bartlett kernel. Even for i.i.d. data,
    the finite-sample cross-product terms at non-zero lags are not exactly
    zero, so the two matrices should differ.
    """
    y, x = regression_data
    _, _, _, vcv_white, *_ = olsnw(y, x, nwlags=0)
    _, _, _, vcv_nw, *_ = olsnw(y, x, nwlags=5)
    # The two VCVs should generally differ for finite-sample data
    assert not np.allclose(vcv_white, vcv_nw, atol=ATOL), (
        "White HC and NW HAC VCVs should differ for nwlags > 0"
    )


def test_olsnw_residual_orthogonality(regression_data):
    """OLS normal equations: X'e approximately 0 where e = y - X*b.

    Ref: olsnw.m:108-113 — OLS via backslash (lstsq) guarantees that the
    residuals are orthogonal to the regressor matrix, i.e. X'*epsilon = 0.
    """
    y, x = regression_data
    T = y.shape[0]
    b, _, _, _, _, _, yhat = olsnw(y, x, c=1)
    epsilon = y - yhat
    X_full = np.column_stack([np.ones(T), x])
    orth = X_full.T @ epsilon
    npt.assert_allclose(
        orth, np.zeros(X_full.shape[1]), atol=1e-8,
        err_msg="OLS residuals not orthogonal to regressors",
    )


# ===========================================================================
# Phase 4: Edge Case Tests
# ===========================================================================


def test_olsnw_single_regressor(rng):
    """K=1 regressor with constant (total 2 parameters).

    Ref: olsnw.m handles K=1 plus constant column correctly.
    Verifies shapes and R-squared range for the simplest regression case.
    """
    T = 150
    x = rng.standard_normal((T, 1))
    y = 2.0 + 3.0 * x.ravel() + rng.standard_normal(T) * 0.5
    b, tstat, s2, vcvnw, r2, rbar, yhat = olsnw(y, x, c=1)
    # Shape checks
    assert b.shape == (2,), f"Expected b.shape=(2,), got {b.shape}"
    assert tstat.shape == (2,), f"Expected tstat.shape=(2,), got {tstat.shape}"
    assert vcvnw.shape == (2, 2), f"Expected vcvnw.shape=(2,2), got {vcvnw.shape}"
    assert yhat.shape == (T,), f"Expected yhat.shape=({T},), got {yhat.shape}"
    # Statistical property
    assert 0.0 <= r2 <= 1.0, f"R-squared out of range: {r2}"
    # VCV must be symmetric and PSD
    npt.assert_allclose(vcvnw, vcvnw.T, atol=ATOL)
    eigenvalues = np.linalg.eigvalsh(vcvnw)
    assert np.all(eigenvalues >= -ATOL), "VCV not PSD for single regressor case"


def test_olsnw_no_constant(rng):
    """c=0: no intercept; b has K elements.

    Ref: olsnw.m:100-105 — c=0 skips constant prepension; uses uncentered
    R-squared (y'y in denominator).
    """
    T = 200
    x = rng.standard_normal((T, 3))
    beta_true = np.array([1.0, -1.0, 0.5])
    y = x @ beta_true + rng.standard_normal(T) * 0.3
    b, tstat, s2, vcvnw, r2, rbar, yhat = olsnw(y, x, c=0)
    # Shape checks
    assert b.shape == (3,), f"Expected b.shape=(3,), got {b.shape}"
    assert tstat.shape == (3,), f"Expected tstat.shape=(3,), got {tstat.shape}"
    assert vcvnw.shape == (3, 3), f"Expected vcvnw.shape=(3,3), got {vcvnw.shape}"
    assert yhat.shape == (T,), f"Expected yhat.shape=({T},), got {yhat.shape}"
    # Residual orthogonality for no-constant case
    epsilon = y - yhat
    orth = x.T @ epsilon
    npt.assert_allclose(
        orth, np.zeros(3), atol=1e-8,
        err_msg="Residuals not orthogonal to X when c=0",
    )


# ===========================================================================
# Phase 5: MATLAB Parity Tests
# ===========================================================================


@pytest.mark.parity
@pytest.mark.requires_fixtures
class TestOlsnwParity:
    """Fixture-based MATLAB parity tests for olsnw.

    Compares Python implementation against reference outputs stored in
    ``tests/fixtures/timeseries/olsnw.npy``.

    Four test cases stored in the fixture dict:
      1. ``standard_constant`` — c=1, default NW bandwidth (auto)
      2. ``no_constant`` — c=0
      3. ``whites_hc`` — c=1, nwlags=0 (White's HC estimator)
      4. ``custom_nwlags`` — c=1, nwlags=10 (explicit lag length)

    Per AAP Section 0.7.1: every migrated function MUST pass
    ``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
    against MATLAB-generated fixtures.
    """

    @staticmethod
    def _run_parity(fixture: dict, prefix: str) -> None:
        """Run parity comparison for a single fixture case.

        Loads input data (y, x, c, nwlags) from the fixture dict, calls
        ``olsnw``, and compares all 7 outputs against stored reference values.

        Parameters
        ----------
        fixture : dict
            Fixture dictionary loaded from ``olsnw.npy``.
        prefix : str
            Key prefix identifying the test case (e.g. ``'standard_constant'``).
        """
        # Extract input data from fixture
        y = fixture[f"{prefix}_y"]
        x = fixture[f"{prefix}_x"]
        c = int(fixture[f"{prefix}_c"].item())
        nwlags = int(fixture[f"{prefix}_nwlags"].item())

        # Run Python implementation
        b, tstat, s2, vcvnw, r2, rbar, yhat = olsnw(y, x, c=c, nwlags=nwlags)

        # Extract reference outputs from fixture
        ref_b = fixture[f"{prefix}_b"]
        ref_tstat = fixture[f"{prefix}_tstat"]
        ref_s2 = float(fixture[f"{prefix}_s2"].item())
        ref_vcvnw = fixture[f"{prefix}_vcvnw"]
        ref_r2 = float(fixture[f"{prefix}_R2"].item())
        ref_rbar = float(fixture[f"{prefix}_Rbar"].item())
        ref_yhat = fixture[f"{prefix}_yhat"]

        # Assert numerical parity for all 7 outputs
        npt.assert_allclose(
            b, ref_b, atol=ATOL, rtol=RTOL,
            err_msg=f"{prefix}: b mismatch",
        )
        npt.assert_allclose(
            tstat, ref_tstat, atol=ATOL, rtol=RTOL,
            err_msg=f"{prefix}: tstat mismatch",
        )
        npt.assert_allclose(
            s2, ref_s2, atol=ATOL, rtol=RTOL,
            err_msg=f"{prefix}: s2 mismatch",
        )
        npt.assert_allclose(
            vcvnw, ref_vcvnw, atol=ATOL, rtol=RTOL,
            err_msg=f"{prefix}: vcvnw mismatch",
        )
        npt.assert_allclose(
            r2, ref_r2, atol=ATOL, rtol=RTOL,
            err_msg=f"{prefix}: R2 mismatch",
        )
        npt.assert_allclose(
            rbar, ref_rbar, atol=ATOL, rtol=RTOL,
            err_msg=f"{prefix}: Rbar mismatch",
        )
        npt.assert_allclose(
            yhat, ref_yhat, atol=ATOL, rtol=RTOL,
            err_msg=f"{prefix}: yhat mismatch",
        )

    def test_olsnw_parity_standard_constant(self, timeseries_fixture_dir):
        """Parity: standard regression with constant, default NW bandwidth.

        Fixture case ``standard_constant``: T=1000, K=2 regressors, c=1,
        nwlags=floor(1000^(1/3))=10.
        """
        raw = load_fixture_npy(timeseries_fixture_dir, "olsnw")
        fixture = raw.item()  # extract dict from 0-d pickle array
        self._run_parity(fixture, "standard_constant")

    def test_olsnw_parity_no_constant(self, timeseries_fixture_dir):
        """Parity: regression without constant (c=0).

        Fixture case ``no_constant``: T=1000, K=2 regressors, c=0,
        uses uncentered R-squared.
        """
        raw = load_fixture_npy(timeseries_fixture_dir, "olsnw")
        fixture = raw.item()
        self._run_parity(fixture, "no_constant")

    def test_olsnw_parity_whites_hc(self, timeseries_fixture_dir):
        """Parity: White's heteroskedasticity-consistent VCV (nwlags=0).

        Fixture case ``whites_hc``: T=1000, K=1 regressor, c=1, nwlags=0.
        The Bartlett kernel with zero lags reduces to White's HC.
        """
        raw = load_fixture_npy(timeseries_fixture_dir, "olsnw")
        fixture = raw.item()
        self._run_parity(fixture, "whites_hc")

    def test_olsnw_parity_custom_nwlags(self, timeseries_fixture_dir):
        """Parity: custom NW lag length (nwlags=10).

        Fixture case ``custom_nwlags``: T=1000, K=3 regressors, c=1,
        nwlags=10 (explicitly specified).
        """
        raw = load_fixture_npy(timeseries_fixture_dir, "olsnw")
        fixture = raw.item()
        self._run_parity(fixture, "custom_nwlags")
