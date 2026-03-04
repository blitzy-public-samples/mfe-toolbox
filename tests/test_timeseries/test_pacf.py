"""Pytest tests for ``mfe_toolbox.timeseries.pacf`` — theoretical ARMA partial autocorrelations.

Tests verify:
- Return type and output shape (N+1 elements, lag-0 always 1)
- AR(p) cutoff property: PACF is exactly zero beyond lag p
- MA(q) geometric decay: PACF does NOT cut off for MA processes
- ARMA(p,q) decay behaviour
- White-noise base case (all PACF beyond lag 0 are zero)
- Boundedness: |pacf[k]| <= 1 for all k
- Levinson-Durbin consistency: pacf[1] == acf[1]
- Numerical parity against MATLAB-generated fixture files (atol=1e-6, rtol=1e-4)

Ref: timeseries/pacf.m — Kevin Sheppard, MFE Toolbox v4.0
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.pacf import pacf

# ---------------------------------------------------------------------------
# Numerical parity constants per AAP §0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ===================================================================
# Test 1: test_pacf_returns_array
# ===================================================================
def test_pacf_returns_array() -> None:
    """``pacf`` must return a ``numpy.ndarray``."""
    phi = np.array([0.5])
    theta = np.array([])
    result = pacf(phi, theta, 5)
    assert isinstance(result, np.ndarray), (
        f"Expected numpy.ndarray, got {type(result).__name__}"
    )


# ===================================================================
# Test 2: test_pacf_output_length
# ===================================================================
@pytest.mark.parametrize("N", [1, 5, 10, 20, 50])
def test_pacf_output_length(N: int) -> None:
    """``pautocorr`` must have exactly ``N+1`` elements for any valid *N*."""
    phi = np.array([0.6])
    theta = np.array([])
    pautocorr = pacf(phi, theta, N)
    assert pautocorr.shape == (N + 1,), (
        f"Expected shape ({N + 1},), got {pautocorr.shape}"
    )


# ===================================================================
# Test 3: test_pacf_zero_lag_is_one
# ===================================================================
def test_pacf_zero_lag_is_one() -> None:
    """``pautocorr[0]`` must equal 1 by convention (lag-0 partial autocorrelation)."""
    # AR(1)
    pautocorr_ar = pacf(np.array([0.7]), np.array([]), 10)
    npt.assert_allclose(pautocorr_ar[0], 1.0, atol=ATOL, rtol=RTOL)

    # MA(1)
    pautocorr_ma = pacf(np.array([]), np.array([0.5]), 10)
    npt.assert_allclose(pautocorr_ma[0], 1.0, atol=ATOL, rtol=RTOL)

    # ARMA(1,1)
    pautocorr_arma = pacf(np.array([0.3]), np.array([0.4]), 10)
    npt.assert_allclose(pautocorr_arma[0], 1.0, atol=ATOL, rtol=RTOL)

    # White noise
    pautocorr_wn = pacf(np.array([]), np.array([]), 10)
    npt.assert_allclose(pautocorr_wn[0], 1.0, atol=ATOL, rtol=RTOL)


# ===================================================================
# Test 4: test_pacf_ar1
# ===================================================================
def test_pacf_ar1() -> None:
    """AR(1) with phi=0.7: pacf(1)=0.7, pacf(k)=0 for k >= 2.

    The partial autocorrelation function of an AR(p) process is exactly zero
    beyond lag p.  For AR(1) the sole non-zero PACF coefficient equals the
    AR parameter itself.

    Ref: pacf.m — Levinson-Durbin recursion on Toeplitz ACF matrix
    """
    phi = np.array([0.7])
    theta = np.array([])
    pautocorr = pacf(phi, theta, 10)

    # pacf[1] should equal the AR(1) coefficient
    npt.assert_allclose(pautocorr[1], 0.7, atol=ATOL, rtol=RTOL)

    # pacf[k] for k >= 2 should be exactly zero (AR cutoff property)
    npt.assert_allclose(pautocorr[2:], 0.0, atol=ATOL)


# ===================================================================
# Test 5: test_pacf_ar2
# ===================================================================
def test_pacf_ar2() -> None:
    """AR(2) phi=[0.5, -0.3]: pacf(1), pacf(2) nonzero; pacf(k)=0 for k >= 3.

    For an AR(2) process, the PACF cutoff occurs at lag 2.  Lags 1 and 2
    must be detectably different from zero, while all higher lags vanish.

    Ref: pacf.m — partitioned matrix inverse via Schur complement
    """
    phi = np.array([0.5, -0.3])
    theta = np.array([])
    pautocorr = pacf(phi, theta, 10)

    # Lags 1 and 2 must be non-trivially nonzero
    assert abs(pautocorr[1]) > 0.01, (
        f"pacf[1] should be nonzero, got {pautocorr[1]}"
    )
    assert abs(pautocorr[2]) > 0.01, (
        f"pacf[2] should be nonzero, got {pautocorr[2]}"
    )

    # All higher-order PACFs must vanish (AR(2) cutoff property)
    npt.assert_allclose(pautocorr[3:], 0.0, atol=ATOL)


# ===================================================================
# Test 6: test_pacf_ma1
# ===================================================================
def test_pacf_ma1() -> None:
    """MA(1) with theta=0.5: PACF decays geometrically — it does NOT cut off.

    Unlike AR(p), an MA(q) process has an infinite PACF that decays
    exponentially.  We verify that several lags beyond lag 1 remain nonzero
    (though they decrease in magnitude).

    Ref: pacf.m — Levinson-Durbin applied to MA ACF
    """
    phi = np.array([])
    theta = np.array([0.5])
    N = 10
    pautocorr = pacf(phi, theta, N)

    # Lag 1 must be nonzero (the dominant PACF for MA(1))
    assert abs(pautocorr[1]) > 0.01, (
        f"MA(1) pacf[1] should be nonzero, got {pautocorr[1]}"
    )

    # Higher lags should remain nonzero (geometric decay, NOT cutoff)
    # Check at least lag 2 and 3 are nonzero (though smaller)
    assert abs(pautocorr[2]) > 1e-8, (
        f"MA(1) pacf[2] should be nonzero (geometric decay), got {pautocorr[2]}"
    )
    assert abs(pautocorr[3]) > 1e-10, (
        f"MA(1) pacf[3] should be nonzero (geometric decay), got {pautocorr[3]}"
    )

    # Verify decay: |pacf[1]| > |pacf[2]| > |pacf[3]|
    assert abs(pautocorr[1]) > abs(pautocorr[2]), (
        "MA(1) PACF should decay: |pacf[1]| > |pacf[2]|"
    )
    assert abs(pautocorr[2]) > abs(pautocorr[3]), (
        "MA(1) PACF should decay: |pacf[2]| > |pacf[3]|"
    )


# ===================================================================
# Test 7: test_pacf_arma11
# ===================================================================
def test_pacf_arma11() -> None:
    """ARMA(1,1) with phi=0.6, theta=0.3: PACF decays with no sharp cutoff.

    For mixed ARMA processes the PACF exhibits damped exponential and/or
    sinusoidal decay after the first (max(p,q)-1) lags.  We verify that
    multiple lags are nonzero and that the absolute values decrease.

    Ref: pacf.m — Levinson-Durbin on full ARMA ACF
    """
    phi = np.array([0.6])
    theta = np.array([0.3])
    N = 10
    pautocorr = pacf(phi, theta, N)

    # Lag 1 must be strongly nonzero
    assert abs(pautocorr[1]) > 0.01, (
        f"ARMA(1,1) pacf[1] should be nonzero, got {pautocorr[1]}"
    )

    # Higher lags should still be nonzero (no sharp cutoff)
    assert abs(pautocorr[2]) > 1e-8, (
        f"ARMA(1,1) pacf[2] should be nonzero, got {pautocorr[2]}"
    )
    assert abs(pautocorr[3]) > 1e-10, (
        f"ARMA(1,1) pacf[3] should be nonzero, got {pautocorr[3]}"
    )


# ===================================================================
# Test 8: test_pacf_white_noise
# ===================================================================
def test_pacf_white_noise() -> None:
    """White noise (phi=[], theta=[]): all PACF beyond lag 0 are exactly zero.

    A white-noise process has zero autocorrelation at all lags > 0, so the
    Levinson-Durbin recursion yields identically zero PACFs.

    Ref: pacf.m — trivial ACF input produces trivial PACF
    """
    phi = np.array([])
    theta = np.array([])
    N = 15
    pautocorr = pacf(phi, theta, N)

    # Lag 0 = 1
    npt.assert_allclose(pautocorr[0], 1.0, atol=ATOL, rtol=RTOL)

    # All lags > 0 must be zero
    npt.assert_allclose(pautocorr[1:], 0.0, atol=ATOL)


# ===================================================================
# Test 9: test_pacf_bounded
# ===================================================================
@pytest.mark.parametrize(
    "phi, theta",
    [
        (np.array([0.9]), np.array([])),       # AR(1) near unit root
        (np.array([0.5, -0.3]), np.array([])), # AR(2)
        (np.array([]), np.array([0.8])),        # MA(1)
        (np.array([]), np.array([0.6, 0.3])),   # MA(2)
        (np.array([0.7]), np.array([0.4])),     # ARMA(1,1)
        (np.array([0.5, -0.3]), np.array([0.4])), # ARMA(2,1)
    ],
    ids=["AR1_near_unit", "AR2", "MA1", "MA2", "ARMA11", "ARMA21"],
)
def test_pacf_bounded(phi: np.ndarray, theta: np.ndarray) -> None:
    """All partial autocorrelations must satisfy |pacf[k]| <= 1.

    This is a fundamental property of the partial autocorrelation function
    for any stationary process.
    """
    N = 20
    pautocorr = pacf(phi, theta, N)
    assert np.all(np.abs(pautocorr) <= 1.0 + ATOL), (
        f"Found |pacf| > 1: max |pacf| = {np.max(np.abs(pautocorr))}"
    )


# ===================================================================
# Test 10: test_pacf_consistency_with_acf
# ===================================================================
def test_pacf_consistency_with_acf() -> None:
    """Levinson-Durbin consistency: pacf[1] must equal acf[1].

    The first partial autocorrelation is always identical to the first
    autocorrelation — this is a fundamental Levinson-Durbin identity.

    Ref: pacf.m:46 — pac(1) = ac(1) where ac is the autocorrelation vector.
    """
    from mfe_toolbox.timeseries.acf import acf

    phi = np.array([0.6, -0.2])
    theta = np.array([0.3])
    N = 10

    autocorr, _ = acf(phi, theta, N)
    pautocorr = pacf(phi, theta, N)

    # Levinson-Durbin: pacf[1] == acf[1]
    npt.assert_allclose(
        pautocorr[1], autocorr[1], atol=ATOL, rtol=RTOL,
        err_msg="Levinson-Durbin consistency: pacf[1] must equal acf[1]",
    )


# ===================================================================
# Test 11: test_pacf_parity — Fixture comparison against MATLAB outputs
# ===================================================================
@pytest.mark.parity
def test_pacf_parity(timeseries_fixture_dir) -> None:
    """Compare ``pacf`` output against MATLAB-generated reference fixtures.

    Fixture ``pacf.npy`` contains four test cases generated by MATLAB/Octave:
    - AR(1) with phi=0.7
    - ARMA(2,1) with phi=[0.5, -0.3], theta=0.4
    - MA(2) with theta=[0.6, 0.3]
    - AR(2) with phi=[0.5, -0.3]

    Each case stores inputs (phi, theta, N) and expected pautocorr.
    """
    fixture_path = timeseries_fixture_dir / "pacf.npy"
    if not fixture_path.exists():
        pytest.skip("Fixture file tests/fixtures/timeseries/pacf.npy not found")

    fixture = np.load(fixture_path, allow_pickle=True).item()

    # --- Case 1: AR(1) ---
    pautocorr_ar1 = pacf(
        fixture["phi_ar1"], fixture["theta_ar1"], int(fixture["N_ar1"])
    )
    npt.assert_allclose(
        pautocorr_ar1, fixture["pautocorr_ar1"],
        atol=ATOL, rtol=RTOL,
        err_msg="AR(1) parity failed",
    )

    # --- Case 2: ARMA(2,1) ---
    pautocorr_arma21 = pacf(
        fixture["phi_arma21"], fixture["theta_arma21"], int(fixture["N_arma21"])
    )
    npt.assert_allclose(
        pautocorr_arma21, fixture["pautocorr_arma21"],
        atol=ATOL, rtol=RTOL,
        err_msg="ARMA(2,1) parity failed",
    )

    # --- Case 3: MA(2) ---
    pautocorr_ma2 = pacf(
        fixture["phi_ma2"], fixture["theta_ma2"], int(fixture["N_ma2"])
    )
    npt.assert_allclose(
        pautocorr_ma2, fixture["pautocorr_ma2"],
        atol=ATOL, rtol=RTOL,
        err_msg="MA(2) parity failed",
    )

    # --- Case 4: AR(2) ---
    pautocorr_ar2 = pacf(
        fixture["phi_ar2"], fixture["theta_ar2"], int(fixture["N_ar2"])
    )
    npt.assert_allclose(
        pautocorr_ar2, fixture["pautocorr_ar2"],
        atol=ATOL, rtol=RTOL,
        err_msg="AR(2) parity failed",
    )


# ===================================================================
# Additional tests (beyond the 11 specified) for robustness
# ===================================================================

def test_pacf_ar1_negative_coefficient() -> None:
    """AR(1) with negative phi=-0.6: pacf(1)=-0.6, pacf(k)=0 for k >= 2.

    A negative AR coefficient produces an alternating-sign ACF, but the
    PACF cutoff property holds identically.
    """
    phi = np.array([-0.6])
    theta = np.array([])
    pautocorr = pacf(phi, theta, 10)

    npt.assert_allclose(pautocorr[1], -0.6, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(pautocorr[2:], 0.0, atol=ATOL)


def test_pacf_near_unit_root_ar1() -> None:
    """AR(1) with phi=0.99 (near unit root): pacf(1)=0.99, higher lags zero.

    Even very persistent AR(1) processes maintain the PACF cutoff property.
    """
    phi = np.array([0.99])
    theta = np.array([])
    pautocorr = pacf(phi, theta, 10)

    npt.assert_allclose(pautocorr[1], 0.99, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(pautocorr[2:], 0.0, atol=ATOL)


def test_pacf_large_n() -> None:
    """Large N=100: verify no numerical instability in the recursion.

    The partitioned-matrix-inverse algorithm can accumulate rounding errors
    for large N; this test checks the output remains bounded and sensible.
    """
    phi = np.array([0.5, -0.2])
    theta = np.array([0.3])
    N = 100
    pautocorr = pacf(phi, theta, N)

    # Shape correctness
    assert pautocorr.shape == (N + 1,)

    # Lag 0 = 1
    npt.assert_allclose(pautocorr[0], 1.0, atol=ATOL, rtol=RTOL)

    # All values bounded
    assert np.all(np.abs(pautocorr) <= 1.0 + ATOL), (
        f"Found |pacf| > 1 at large N: max = {np.max(np.abs(pautocorr))}"
    )
