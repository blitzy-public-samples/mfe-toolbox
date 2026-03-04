"""Pytest tests for mfe_toolbox.timeseries.acf — theoretical ARMA autocorrelations.

Tests verify:
- Input validation (empty AR/MA, scalar vs vector, invalid N, invalid sigma2_e)
- Output shapes (autocorr: (N+1,), sigma2_y: scalar)
- Lag-0 autocorrelation equals 1.0 always
- Known analytic results for simple AR(1), MA(1), ARMA(1,1), AR(2)
- Sigma2_e scaling properties (scales variance, not autocorrelations)
- Numerical parity against MATLAB fixtures (atol=1e-6, rtol=1e-4)
- Edge cases: large N, near-unit-root AR(1), negative AR coefficient

Ref: timeseries/acf.m — Kevin Sheppard, MFE Toolbox v4.0
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.acf import acf

# ---------------------------------------------------------------------------
# Constants — AAP §0.7.1: parity tolerance ±1e-6
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ar1_params() -> dict:
    """AR(1) with phi=0.8: known autocorrelation rho(k) = 0.8^k."""
    return dict(phi=np.array([0.8]), theta=np.array([]), N=20, sigma2_e=1.0)


@pytest.fixture
def ma1_params() -> dict:
    """MA(1) with theta=0.5: rho(1) = theta/(1+theta^2), rho(k) = 0 for k >= 2."""
    return dict(phi=np.array([]), theta=np.array([0.5]), N=10, sigma2_e=1.0)


@pytest.fixture
def arma11_params() -> dict:
    """ARMA(1,1) with phi=0.7, theta=0.3."""
    return dict(phi=np.array([0.7]), theta=np.array([0.3]), N=15, sigma2_e=1.0)


# ===========================================================================
# Phase 3 — Input Validation Tests
# ===========================================================================

def test_acf_empty_ar_and_ma() -> None:
    """Both phi and theta empty → white noise: autocorr = [1, 0, ..., 0],
    sigma2_y = sigma2_e.

    Ref: acf.m — when p=0 and q=0, the only autocovariance is gamma(0) = sigma2_e.
    """
    N = 10
    sigma2_e = 2.5
    autocorr, sigma2_y = acf(
        phi=np.array([]), theta=np.array([]), N=N, sigma2_e=sigma2_e
    )
    expected_autocorr = np.zeros(N + 1)
    expected_autocorr[0] = 1.0
    npt.assert_allclose(autocorr, expected_autocorr, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(sigma2_y, sigma2_e, atol=ATOL, rtol=RTOL)


def test_acf_N_negative_raises() -> None:
    """N < 0 must raise ValueError.

    Ref: acf.m line ~40 — validates ``N >= 0``.  Note that N=0 is *allowed*
    by the MATLAB source (returns a single-element autocorrelation vector).
    """
    with pytest.raises(ValueError):
        acf(phi=np.array([0.5]), theta=np.array([]), N=-1, sigma2_e=1.0)


def test_acf_sigma2_e_negative_raises() -> None:
    """sigma2_e <= 0 must raise ValueError.

    Ref: acf.m — innovation variance must be strictly positive.
    """
    with pytest.raises(ValueError):
        acf(phi=np.array([0.5]), theta=np.array([]), N=5, sigma2_e=-1.0)
    with pytest.raises(ValueError):
        acf(phi=np.array([0.5]), theta=np.array([]), N=5, sigma2_e=0.0)


# ===========================================================================
# Phase 4 — Output Shape and Type Tests
# ===========================================================================

def test_acf_output_shapes(ar1_params: dict) -> None:
    """autocorr has shape (N+1,); sigma2_y is a scalar float."""
    autocorr, sigma2_y = acf(**ar1_params)
    N = ar1_params["N"]
    assert autocorr.shape == (N + 1,), (
        f"Expected autocorr shape ({N + 1},), got {autocorr.shape}"
    )
    assert isinstance(sigma2_y, (float, np.float64)), (
        f"sigma2_y should be scalar float, got {type(sigma2_y)}"
    )


def test_acf_returns_tuple(ar1_params: dict) -> None:
    """Function returns exactly two values (autocorr, sigma2_y)."""
    result = acf(**ar1_params)
    assert isinstance(result, tuple), "acf should return a tuple"
    assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"


def test_acf_lag0_always_one() -> None:
    """autocorr[0] == 1.0 for any valid input set.

    The lag-0 autocorrelation is always 1 by definition (rho(0) = gamma(0)/gamma(0)).
    Ref: acf.m — final normalization ``autocorr = autocov ./ autocov(1)`` guarantees this.
    """
    test_cases = [
        # (phi, theta, N, sigma2_e)
        (np.array([0.8]), np.array([]), 5, 1.0),
        (np.array([]), np.array([0.5]), 5, 1.0),
        (np.array([0.5, 0.3]), np.array([]), 5, 1.0),
        (np.array([0.7]), np.array([0.3]), 5, 1.0),
        (np.array([]), np.array([]), 5, 3.0),
        (np.array([0.99]), np.array([]), 5, 1.0),
        (np.array([-0.5]), np.array([]), 5, 1.0),
    ]
    for phi, theta, N, sigma2_e in test_cases:
        autocorr, _ = acf(phi=phi, theta=theta, N=N, sigma2_e=sigma2_e)
        npt.assert_allclose(
            autocorr[0], 1.0, atol=ATOL, rtol=RTOL,
            err_msg=f"Lag-0 autocorrelation should be 1.0 for phi={phi}, theta={theta}"
        )


# ===========================================================================
# Phase 5 — Analytic Verification Tests (CRITICAL)
# ===========================================================================

def test_acf_ar1_closed_form(ar1_params: dict) -> None:
    """AR(1) with phi=0.8: autocorr[k] = 0.8^k, sigma2_y = 1/(1-phi^2).

    Ref: acf.m — for AR(1) the Yule-Walker solution gives rho(k) = phi^k.
    """
    autocorr, sigma2_y = acf(**ar1_params)
    phi_val = 0.8
    N = ar1_params["N"]
    expected_autocorr = np.array([phi_val ** k for k in range(N + 1)])
    npt.assert_allclose(autocorr, expected_autocorr, atol=ATOL, rtol=RTOL)
    expected_var = 1.0 / (1.0 - phi_val ** 2)
    npt.assert_allclose(sigma2_y, expected_var, atol=ATOL, rtol=RTOL)


def test_acf_ma1_closed_form(ma1_params: dict) -> None:
    """MA(1) with theta=0.5: rho(0)=1, rho(1) = theta/(1+theta^2), rho(k)=0 for k>=2.

    sigma2_y = sigma2_e * (1 + theta^2).

    Ref: acf.m — MA(q) autocorrelations are zero beyond lag q.
    """
    autocorr, sigma2_y = acf(**ma1_params)
    theta_val = 0.5
    N = ma1_params["N"]

    # Build expected autocorrelation vector
    expected = np.zeros(N + 1)
    expected[0] = 1.0
    expected[1] = theta_val / (1.0 + theta_val ** 2)  # = 0.5 / 1.25 = 0.4
    # rho(k) = 0 for k >= 2 — already zeros

    npt.assert_allclose(autocorr, expected, atol=ATOL, rtol=RTOL)

    # Unconditional variance: sigma2_y = sigma2_e * (1 + theta^2)
    expected_var = 1.0 * (1.0 + theta_val ** 2)  # = 1.25
    npt.assert_allclose(sigma2_y, expected_var, atol=ATOL, rtol=RTOL)


def test_acf_white_noise() -> None:
    """phi=[], theta=[] → rho(0)=1, rho(k)=0 for k>=1, sigma2_y = sigma2_e.

    Ref: acf.m — the trivial ARMA(0,0) case.
    """
    N = 15
    sigma2_e = 4.0
    autocorr, sigma2_y = acf(
        phi=np.array([]), theta=np.array([]), N=N, sigma2_e=sigma2_e
    )
    expected = np.zeros(N + 1)
    expected[0] = 1.0
    npt.assert_allclose(autocorr, expected, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(sigma2_y, sigma2_e, atol=ATOL, rtol=RTOL)


def test_acf_ar2_stationary() -> None:
    """AR(2) with phi=[0.5, 0.3]: verify Yule-Walker recursive autocorrelations.

    For AR(2): y(t) = phi1*y(t-1) + phi2*y(t-2) + e(t)
    Yule-Walker:
        rho(1) = phi1 / (1 - phi2)
        rho(2) = phi1*rho(1) + phi2
        rho(k) = phi1*rho(k-1) + phi2*rho(k-2)  for k >= 3

    sigma2_y = sigma2_e / (1 - phi1*rho(1) - phi2*rho(2))

    Ref: acf.m — the general Yule-Walker extension routine.
    """
    phi1, phi2 = 0.5, 0.3
    N = 10
    autocorr, sigma2_y = acf(
        phi=np.array([phi1, phi2]), theta=np.array([]), N=N, sigma2_e=1.0
    )

    # Build expected autocorrelations via Yule-Walker recursion
    expected = np.zeros(N + 1)
    expected[0] = 1.0
    expected[1] = phi1 / (1.0 - phi2)
    expected[2] = phi1 * expected[1] + phi2
    for k in range(3, N + 1):
        expected[k] = phi1 * expected[k - 1] + phi2 * expected[k - 2]

    npt.assert_allclose(autocorr, expected, atol=ATOL, rtol=RTOL)

    # Verify unconditional variance
    expected_var = 1.0 / (1.0 - phi1 * expected[1] - phi2 * expected[2])
    npt.assert_allclose(sigma2_y, expected_var, atol=ATOL, rtol=RTOL)


def test_acf_arma11(arma11_params: dict) -> None:
    """ARMA(1,1) with phi=0.7, theta=0.3: verify known analytic formula.

    For ARMA(1,1): y(t) = phi*y(t-1) + e(t) + theta*e(t-1)
    gamma(0) = sigma2_e * (1 + 2*phi*theta + theta^2) / (1 - phi^2)
    gamma(1) = sigma2_e * (phi + theta)*(1 + phi*theta) / (1 - phi^2)
    gamma(k) = phi * gamma(k-1)  for k >= 2

    rho(1) = (phi + theta)*(1 + phi*theta) / (1 + 2*phi*theta + theta^2)
    rho(k) = phi^(k-1) * rho(1)  for k >= 2

    Ref: acf.m — combined AR/MA autocovariance system.
    """
    phi_val = 0.7
    theta_val = 0.3
    N = arma11_params["N"]

    autocorr, sigma2_y = acf(**arma11_params)

    # Analytic rho(1) for ARMA(1,1)
    rho1_num = (phi_val + theta_val) * (1.0 + phi_val * theta_val)
    rho1_den = 1.0 + 2.0 * phi_val * theta_val + theta_val ** 2
    rho1 = rho1_num / rho1_den

    # Build expected autocorrelation vector
    expected = np.zeros(N + 1)
    expected[0] = 1.0
    expected[1] = rho1
    for k in range(2, N + 1):
        expected[k] = phi_val * expected[k - 1]

    npt.assert_allclose(autocorr, expected, atol=ATOL, rtol=RTOL)

    # Unconditional variance for ARMA(1,1)
    expected_var = (1.0 + 2.0 * phi_val * theta_val + theta_val ** 2) / (
        1.0 - phi_val ** 2
    )
    npt.assert_allclose(sigma2_y, expected_var, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Phase 6 — Sigma2_e Scaling Tests
# ===========================================================================

def test_acf_sigma2_e_scaling() -> None:
    """Changing sigma2_e scales sigma2_y proportionally but does NOT change
    autocorrelations (since autocorrelations are normalized by gamma(0)).

    Ref: acf.m — final output: ``sigma2_y = sigma2_e * autocov(1)``,
    ``autocorr = autocov / autocov(1)``.
    """
    phi = np.array([0.6])
    theta = np.array([0.2])
    N = 10

    autocorr1, var1 = acf(phi=phi, theta=theta, N=N, sigma2_e=1.0)
    autocorr2, var2 = acf(phi=phi, theta=theta, N=N, sigma2_e=3.5)

    # Autocorrelations should be identical
    npt.assert_allclose(autocorr1, autocorr2, atol=ATOL, rtol=RTOL)

    # Variance should scale linearly with sigma2_e
    npt.assert_allclose(var2, var1 * 3.5, atol=ATOL, rtol=RTOL)


def test_acf_sigma2_e_default() -> None:
    """sigma2_e defaults to 1.0 when not specified.

    Ref: acf.m line ~29 — ``if nargin == 3, sigma2_e = 1``.
    """
    phi = np.array([0.5])
    theta = np.array([])
    N = 5

    autocorr_default, var_default = acf(phi=phi, theta=theta, N=N)
    autocorr_explicit, var_explicit = acf(
        phi=phi, theta=theta, N=N, sigma2_e=1.0
    )

    npt.assert_allclose(autocorr_default, autocorr_explicit, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(var_default, var_explicit, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Phase 7 — Parity Tests Against MATLAB Fixtures
# ===========================================================================

@pytest.mark.parity
def test_acf_parity_ar1(timeseries_fixture_dir) -> None:  # type: ignore[no-untyped-def]
    """Load AR(1) fixture generated by MATLAB/Octave and compare.

    Ref: AAP §0.7.1 — every migrated function MUST pass
    ``assert_allclose(atol=1e-6, rtol=1e-4)``.
    """
    fixture_path = timeseries_fixture_dir / "acf_ar1.npy"
    if not fixture_path.exists():
        pytest.skip("Fixture acf_ar1.npy not found")
    fixture = np.load(fixture_path, allow_pickle=True).item()
    autocorr, sigma2_y = acf(
        fixture["phi"], fixture["theta"], fixture["N"], fixture["sigma2_e"]
    )
    npt.assert_allclose(autocorr, fixture["autocorr"], atol=ATOL, rtol=RTOL)
    npt.assert_allclose(sigma2_y, fixture["sigma2_y"], atol=ATOL, rtol=RTOL)


@pytest.mark.parity
def test_acf_parity_arma11(timeseries_fixture_dir) -> None:  # type: ignore[no-untyped-def]
    """Load ARMA(1,1) fixture generated by MATLAB/Octave and compare.

    Ref: AAP §0.7.1 — numerical parity ±1e-6 against MATLAB reference.
    """
    fixture_path = timeseries_fixture_dir / "acf_arma11.npy"
    if not fixture_path.exists():
        pytest.skip("Fixture acf_arma11.npy not found")
    fixture = np.load(fixture_path, allow_pickle=True).item()
    autocorr, sigma2_y = acf(
        fixture["phi"], fixture["theta"], fixture["N"], fixture["sigma2_e"]
    )
    npt.assert_allclose(autocorr, fixture["autocorr"], atol=ATOL, rtol=RTOL)
    npt.assert_allclose(sigma2_y, fixture["sigma2_y"], atol=ATOL, rtol=RTOL)


# ===========================================================================
# Phase 8 — Edge Cases
# ===========================================================================

def test_acf_large_N() -> None:
    """N=500: verify no numerical instability in the Yule-Walker extension.

    For a stationary AR(1), autocorrelations decay geometrically and the
    tail values should approach zero without NaN or overflow.

    Ref: acf.m — the Yule-Walker extension loop for large N.
    """
    N = 500
    phi_val = 0.7
    autocorr, sigma2_y = acf(
        phi=np.array([phi_val]), theta=np.array([]), N=N, sigma2_e=1.0
    )
    assert autocorr.shape == (N + 1,)
    assert not np.any(np.isnan(autocorr)), "No NaN values expected"
    assert not np.any(np.isinf(autocorr)), "No Inf values expected"

    # Check the geometric decay pattern
    expected_tail = phi_val ** N
    npt.assert_allclose(autocorr[N], expected_tail, atol=ATOL, rtol=RTOL)

    # Variance should be well-defined
    expected_var = 1.0 / (1.0 - phi_val ** 2)
    npt.assert_allclose(sigma2_y, expected_var, atol=ATOL, rtol=RTOL)


def test_acf_near_unit_root_ar1() -> None:
    """phi=0.99: verify accuracy close to unit root.

    The process is still stationary (|phi| < 1) but variance is large and
    autocorrelations decay very slowly.

    Ref: acf.m — stationarity is checked via inverse_ar_roots(phi).
    """
    phi_val = 0.99
    N = 50
    autocorr, sigma2_y = acf(
        phi=np.array([phi_val]), theta=np.array([]), N=N, sigma2_e=1.0
    )

    # Verify geometric decay
    expected_autocorr = np.array([phi_val ** k for k in range(N + 1)])
    npt.assert_allclose(autocorr, expected_autocorr, atol=ATOL, rtol=RTOL)

    # Variance should be large: 1/(1-0.99^2) ≈ 50.25
    expected_var = 1.0 / (1.0 - phi_val ** 2)
    npt.assert_allclose(sigma2_y, expected_var, atol=ATOL, rtol=RTOL)
    assert sigma2_y > 50.0, "Near-unit-root variance should be large"


def test_acf_negative_ar_coefficient() -> None:
    """phi=-0.5: verify alternating-sign autocorrelations.

    For AR(1) with negative phi, rho(k) = phi^k = (-0.5)^k, so
    rho(1) < 0, rho(2) > 0, rho(3) < 0, etc.

    Ref: acf.m — the algorithm handles negative coefficients identically.
    """
    phi_val = -0.5
    N = 10
    autocorr, sigma2_y = acf(
        phi=np.array([phi_val]), theta=np.array([]), N=N, sigma2_e=1.0
    )
    expected_autocorr = np.array([phi_val ** k for k in range(N + 1)])
    npt.assert_allclose(autocorr, expected_autocorr, atol=ATOL, rtol=RTOL)

    # Verify alternating signs
    assert autocorr[1] < 0, "rho(1) should be negative for phi=-0.5"
    assert autocorr[2] > 0, "rho(2) should be positive for phi=-0.5"
    assert autocorr[3] < 0, "rho(3) should be negative for phi=-0.5"

    # Variance: 1/(1 - 0.25) = 4/3
    expected_var = 1.0 / (1.0 - phi_val ** 2)
    npt.assert_allclose(sigma2_y, expected_var, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Additional Tests — N=0 boundary, MA(2), ARMA(2,1)
# ===========================================================================

def test_acf_N_zero_valid() -> None:
    """N=0 is valid and returns autocorr=[1.0] and the unconditional variance.

    Ref: acf.m — MATLAB allows N=0, returning a single-element vector.
    The Python implementation mirrors this: only N < 0 raises ValueError.
    """
    autocorr, sigma2_y = acf(
        phi=np.array([0.5]), theta=np.array([]), N=0, sigma2_e=1.0
    )
    assert autocorr.shape == (1,), f"Expected shape (1,), got {autocorr.shape}"
    npt.assert_allclose(autocorr[0], 1.0, atol=ATOL, rtol=RTOL)
    # sigma2_y should equal 1/(1-0.25) = 4/3
    npt.assert_allclose(sigma2_y, 1.0 / (1.0 - 0.25), atol=ATOL, rtol=RTOL)


def test_acf_ma2_closed_form() -> None:
    """MA(2) with theta=[0.4, 0.2]: verify higher-order MA autocorrelation structure.

    For MA(2): y(t) = e(t) + theta1*e(t-1) + theta2*e(t-2)
    gamma(0) = sigma2_e * (1 + theta1^2 + theta2^2)
    gamma(1) = sigma2_e * (theta1 + theta1*theta2)
    gamma(2) = sigma2_e * theta2
    gamma(k) = 0  for k >= 3

    rho(k) = gamma(k) / gamma(0).
    """
    theta1, theta2 = 0.4, 0.2
    N = 8
    autocorr, sigma2_y = acf(
        phi=np.array([]), theta=np.array([theta1, theta2]), N=N, sigma2_e=1.0
    )

    gamma0 = 1.0 + theta1 ** 2 + theta2 ** 2  # 1 + 0.16 + 0.04 = 1.20
    gamma1 = theta1 + theta1 * theta2  # 0.4 + 0.08 = 0.48
    gamma2 = theta2  # 0.2

    expected = np.zeros(N + 1)
    expected[0] = 1.0
    expected[1] = gamma1 / gamma0
    expected[2] = gamma2 / gamma0
    # rho(k) = 0 for k >= 3

    npt.assert_allclose(autocorr, expected, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(sigma2_y, gamma0, atol=ATOL, rtol=RTOL)


def test_acf_arma21() -> None:
    """ARMA(2,1) with phi=[0.5, -0.2], theta=[0.3]: non-trivial mixed model.

    Verify fundamental properties rather than full closed-form:
    1. autocorr[0] = 1.0
    2. sigma2_y > 0
    3. All |autocorr[k]| <= 1
    4. For k >= max(p,q)+1, the recursion rho(k) = phi1*rho(k-1) + phi2*rho(k-2) holds
    """
    phi = np.array([0.5, -0.2])
    theta = np.array([0.3])
    N = 20
    autocorr, sigma2_y = acf(phi=phi, theta=theta, N=N, sigma2_e=1.0)

    # Basic properties
    npt.assert_allclose(autocorr[0], 1.0, atol=ATOL, rtol=RTOL)
    assert sigma2_y > 0, "Unconditional variance must be positive"
    assert np.all(np.abs(autocorr) <= 1.0 + ATOL), (
        "All autocorrelations must satisfy |rho(k)| <= 1"
    )

    # Yule-Walker recursion holds for k >= max(2,1)+1 = 3
    # rho(k) = phi1*rho(k-1) + phi2*rho(k-2)
    for k in range(3, N + 1):
        expected_k = phi[0] * autocorr[k - 1] + phi[1] * autocorr[k - 2]
        npt.assert_allclose(
            autocorr[k], expected_k, atol=ATOL, rtol=RTOL,
            err_msg=f"Yule-Walker recursion failed at lag {k}"
        )
