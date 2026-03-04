"""Pytest test suite for mfe_toolbox.utility.covnw.

Comprehensive tests for the Newey-West (Bartlett kernel) HAC covariance
estimator.  Covers output shape, symmetry, default lag computation, Bartlett
kernel weights, demean behaviour, error handling, and MATLAB fixture parity.

Source Reference
----------------
utility/covnw.m (79 lines, Revision 3, 5/1/2007, Kevin Sheppard)

Key MATLAB behaviour verified:
- Accepts T×K data, optional nlag, optional demean flag
- Default nlag: ``min(floor(1.2 * T^(1/3)), T)``
- Bartlett weights: ``w(j) = 1 - j / (nlag + 1)`` for j = 1..nlag
- Demeans by default: subtracts column means before computing autocovariances
- Returns K×K positive semi-definite covariance matrix

Numerical parity target: ``numpy.testing.assert_allclose(atol=1e-6, rtol=1e-4)``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.covnw import covnw
from tests.conftest import ATOL, RTOL, assert_allclose, load_fixture_npy

# ---------------------------------------------------------------------------
# Tolerance constants — per AAP Section 0.7.1
# ---------------------------------------------------------------------------
_ATOL: float = ATOL  # 1e-6
_RTOL: float = RTOL  # 1e-4


# ===================================================================
# 1. test_covnw_white_noise
# ===================================================================


def test_covnw_white_noise(multivariate_data: np.ndarray) -> None:
    """i.i.d. white noise → HAC should approximate sample covariance.

    For i.i.d. data the autocovariance at all non-zero lags is zero in
    expectation, so the Newey-West estimate should be close to the
    standard sample covariance (X'X / T after demeaning), which is the
    same as ``nlag=0`` (White's HC estimator).

    Uses the ``multivariate_data`` session fixture (T=1000, K=3,
    demeaned) from ``conftest.py``.
    """
    data: np.ndarray = multivariate_data  # T=1000, K=3 demeaned i.i.d. data

    # Newey-West HAC with default lags
    S_nw: np.ndarray = covnw(data)

    # White's HC estimator (nlag=0) — equivalent to plain sample covariance
    S_white: np.ndarray = covnw(data, nw_lags=0)

    # Also compute via numpy.cov for cross-check (ddof=0 → 1/T normalisation)
    # Ref: MATLAB's covnw uses 1/T, not 1/(T-1)
    S_cov: np.ndarray = np.cov(data, rowvar=False, ddof=0)

    # For white noise, the HAC correction terms should be negligible
    npt.assert_allclose(
        S_nw, S_white, atol=0.15, rtol=0.15,
        err_msg="HAC for white noise should approximate sample covariance",
    )

    # nlag=0 should match numpy.cov with ddof=0 exactly (both demean, both 1/T)
    npt.assert_allclose(
        S_white, S_cov, atol=1e-12,
        err_msg="nlag=0 should equal np.cov(data, ddof=0) for demeaned data",
    )


# ===================================================================
# 2. test_covnw_known_covariance
# ===================================================================


def test_covnw_known_covariance() -> None:
    """Large T i.i.d. data with known population covariance.

    With very large T and i.i.d. observations drawn from N(0, I_K),
    the HAC covariance should converge to the identity matrix since the
    population covariance of standard normal vectors is I_K.
    """
    local_rng: np.random.Generator = np.random.default_rng(99)
    T: int = 50000
    K: int = 2
    data: np.ndarray = local_rng.standard_normal((T, K))

    S: np.ndarray = covnw(data)

    # For i.i.d. standard normal, population covariance is I_K.
    # With T=50000 convergence should be quite good.
    npt.assert_allclose(
        S, np.eye(K), atol=0.05, rtol=0.05,
        err_msg="HAC for large T iid should approximate population covariance",
    )


# ===================================================================
# 3. test_covnw_default_nlag
# ===================================================================


@pytest.mark.parametrize("T", [10, 50, 100, 200, 500, 1000, 5000])
def test_covnw_default_nlag(T: int) -> None:
    """Verify default nlag = min(floor(1.2 * T^(1/3)), T).

    Ref: covnw.m:41 — ``nlag = min(floor(1.2 * T^(1/3)), T)``

    For each T value, calling ``covnw`` with default nlag must produce
    the same result as calling it with the explicitly computed lag.
    """
    K: int = 2
    local_rng: np.random.Generator = np.random.default_rng(42 + T)
    data: np.ndarray = local_rng.standard_normal((T, K))

    # Compute the expected default nlag using the same formula as the
    # Python implementation (Ref: covnw.m:41, covnw.py:108)
    expected_nlag: int = int(np.minimum(np.floor(1.2 * T ** (1.0 / 3.0)), T))

    S_default: np.ndarray = covnw(data)
    S_explicit: np.ndarray = covnw(data, nw_lags=expected_nlag)

    npt.assert_array_equal(
        S_default, S_explicit,
        err_msg=f"Default nlag for T={T} should be {expected_nlag}",
    )


# ===================================================================
# 4. test_covnw_custom_nlag
# ===================================================================


def test_covnw_custom_nlag(rng: np.random.Generator) -> None:
    """Explicit nlag=5 should produce a different result from default.

    Using T=1000, the default nlag ≈ 12, so nlag=5 gives a different
    HAC estimate (fewer lags → different Bartlett weights).
    """
    T: int = 1000
    K: int = 3
    data: np.ndarray = rng.standard_normal((T, K))

    S_default: np.ndarray = covnw(data)
    S_lag5: np.ndarray = covnw(data, nw_lags=5)

    # Different lags must produce different results for non-trivial data
    assert not np.allclose(S_default, S_lag5, atol=1e-10), (
        "Custom nlag=5 should differ from the default nlag"
    )

    # Both must be K×K
    assert S_default.shape == (K, K)
    assert S_lag5.shape == (K, K)


# ===================================================================
# 5. test_covnw_nlag_zero
# ===================================================================


def test_covnw_nlag_zero(rng: np.random.Generator) -> None:
    """nlag=0 → standard sample covariance (no HAC correction).

    With zero lags the Newey-West estimator reduces to White's HC
    covariance: ``S = X'X / T`` (after demeaning).

    Ref: covnw.m:71 — ``V = data' * data / T`` (only lag-0 term when
    nlag = 0, the loop body on lines 72-76 never executes).
    """
    T: int = 500
    K: int = 3
    data: np.ndarray = rng.standard_normal((T, K))

    S_white: np.ndarray = covnw(data, nw_lags=0)

    # Manual computation: demean, then X'X / T
    data_dm: np.ndarray = data - data.mean(axis=0)
    S_manual: np.ndarray = data_dm.T @ data_dm / T

    # Zero-lag autocovariance contribution is exactly zero
    autocov_contribution: np.ndarray = np.zeros((K, K))

    npt.assert_allclose(
        S_white, S_manual + autocov_contribution, atol=1e-12,
        err_msg="nlag=0 should give X'X/T (after demeaning) with no lag terms",
    )


# ===================================================================
# 6. test_covnw_symmetric_output
# ===================================================================


def test_covnw_symmetric_output(rng: np.random.Generator) -> None:
    """Output covariance matrix must be symmetric: S == S.T.

    The Bartlett kernel HAC estimator produces symmetric output by
    construction: each lag contribution is ``Gamma_i + Gamma_i'``.

    Ref: covnw.m:74 — ``GplusGprime = Gammai + Gammai';``
    """
    T: int = 500
    K: int = 5
    data: np.ndarray = rng.standard_normal((T, K))

    S: np.ndarray = covnw(data)
    assert np.allclose(S, S.T, atol=1e-14), (
        "HAC covariance matrix must be symmetric"
    )

    # Also verify symmetry with an explicit non-default lag count
    S2: np.ndarray = covnw(data, nw_lags=3)
    assert np.allclose(S2, S2.T, atol=1e-14), (
        "HAC covariance matrix must be symmetric (nlag=3)"
    )


# ===================================================================
# 7. test_covnw_output_shape
# ===================================================================


@pytest.mark.parametrize("T,K", [
    (100, 1),
    (100, 2),
    (100, 5),
    (500, 3),
    (1000, 10),
    (50, 2),
])
def test_covnw_output_shape(T: int, K: int) -> None:
    """T×K input → K×K output.

    Verifies that the output has the correct square dimensions matching
    the number of columns in the input.
    """
    local_rng: np.random.Generator = np.random.default_rng(123)
    data: np.ndarray = local_rng.standard_normal((T, K))

    S: np.ndarray = covnw(data)

    assert isinstance(S, np.ndarray), "Output must be a numpy ndarray"
    assert S.shape == (K, K), f"Expected ({K},{K}), got {S.shape}"


# ===================================================================
# 8. test_covnw_demean_flag
# ===================================================================


def test_covnw_demean_flag(rng: np.random.Generator) -> None:
    """demean=True vs demean=False should give different results.

    When data has large non-zero column means, toggling the demean flag
    affects the outer products and thus the covariance estimate.
    """
    T: int = 500
    K: int = 3
    # Create data with large non-zero mean to ensure a clear difference
    offsets: np.ndarray = np.array([5.0, -3.0, 10.0])
    data: np.ndarray = rng.standard_normal((T, K)) + offsets

    S_demean: np.ndarray = covnw(data, nw_lags=5, demean=True)
    S_nodemean: np.ndarray = covnw(data, nw_lags=5, demean=False)

    # They must differ since data has non-zero mean
    assert not np.allclose(S_demean, S_nodemean, atol=1e-6), (
        "demean=True and demean=False should give different results "
        "for non-zero mean data"
    )

    # Both must still be symmetric
    assert np.allclose(S_demean, S_demean.T, atol=1e-14)
    assert np.allclose(S_nodemean, S_nodemean.T, atol=1e-14)

    # With demean=True, the result should match manually-demeaned data
    data_dm: np.ndarray = data - data.mean(axis=0)
    S_manual_dm: np.ndarray = covnw(data_dm, nw_lags=5, demean=False)
    npt.assert_allclose(
        S_demean, S_manual_dm, atol=1e-12,
        err_msg="demean=True should equal manual demeaning + demean=False",
    )


# ===================================================================
# 9. test_covnw_single_column
# ===================================================================


def test_covnw_single_column(rng: np.random.Generator) -> None:
    """K=1 input → 1×1 scalar variance output.

    For a single column the output should be a 1×1 matrix representing
    the HAC variance estimate.
    """
    T: int = 500
    data: np.ndarray = rng.standard_normal((T, 1))

    S: np.ndarray = covnw(data)

    assert S.shape == (1, 1), f"Expected (1,1), got {S.shape}"
    assert S[0, 0] > 0, "Variance should be positive for non-degenerate data"

    # With nlag=0, should match plain variance (1/T normalisation)
    data_dm: np.ndarray = data - data.mean(axis=0)
    var_manual: float = float((data_dm.T @ data_dm / T).item())
    S_nlag0: np.ndarray = covnw(data, nw_lags=0)
    npt.assert_allclose(
        S_nlag0[0, 0], var_manual, atol=1e-12,
        err_msg="K=1, nlag=0 should give scalar variance",
    )


# ===================================================================
# 10. test_covnw_non_matrix_raises
# ===================================================================


def test_covnw_non_matrix_raises() -> None:
    """Non-2D input should raise ValueError.

    Ref: covnw.m:58-60 — ``error('DATA must be a T by K matrix of data.')``
    """
    # 1-D array (ndim = 1)
    with pytest.raises(ValueError, match="scores must be a T by K matrix"):
        covnw(np.array([1.0, 2.0, 3.0]))

    # 3-D array (ndim = 3)
    with pytest.raises(ValueError, match="scores must be a T by K matrix"):
        covnw(np.zeros((10, 3, 2)))

    # 0-D scalar — np.asarray converts to ndim=0 array
    with pytest.raises(ValueError, match="scores must be a T by K matrix"):
        covnw(np.float64(1.0))


# ===================================================================
# 11. test_covnw_fixture_parity
# ===================================================================


def test_covnw_fixture_parity(utility_fixture_dir: Path) -> None:
    """MATLAB reference fixture parity test.

    Loads MATLAB-generated fixture data and verifies that the Python
    implementation produces numerically identical results within
    ``atol=1e-6``, ``rtol=1e-4`` per AAP Section 0.7.1.

    Fixture keys (from ``covnw.npy``):
    - ``data``          — (1000, 3) input matrix generated by Octave
    - ``V_auto``        — covnw output with automatic nlag selection
    - ``V_10``          — covnw output with nlag=10
    - ``V_10_nodemean`` — covnw output with nlag=10, demean=false
    - ``nlag_auto``     — the auto-selected lag value from Octave
    """
    arr: np.ndarray = load_fixture_npy(utility_fixture_dir, "covnw")
    fixture: dict = arr.item() if arr.shape == () else arr

    data: np.ndarray = fixture["data"]
    nlag_auto: int = int(fixture["nlag_auto"])

    # ---- Sub-test 1: covnw with the auto-detected nlag from Octave ----
    # Use the fixture's nlag value explicitly to avoid floating-point
    # discrepancies in the default lag formula between Python and Octave.
    S_auto: np.ndarray = covnw(data, nw_lags=nlag_auto)
    assert_allclose(
        S_auto, fixture["V_auto"],
        err_msg="covnw auto-lag result does not match MATLAB fixture",
    )

    # ---- Sub-test 2: covnw with nlag=10 ----
    S_10: np.ndarray = covnw(data, nw_lags=10)
    assert_allclose(
        S_10, fixture["V_10"],
        err_msg="covnw nlag=10 result does not match MATLAB fixture",
    )

    # ---- Sub-test 3: covnw with nlag=10, demean=False ----
    S_10_nodemean: np.ndarray = covnw(data, nw_lags=10, demean=False)
    assert_allclose(
        S_10_nodemean, fixture["V_10_nodemean"],
        err_msg="covnw nlag=10 nodemean does not match MATLAB fixture",
    )
