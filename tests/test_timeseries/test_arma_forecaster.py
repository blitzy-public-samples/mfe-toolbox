"""
Pytest parity and correctness tests for mfe_toolbox.timeseries.arma_forecaster.

This module tests the ARMA h-step ahead forecasting function, which was migrated
from:
  - timeseries/arma_forecaster.m  (Kevin Sheppard, Revision 3, 1/1/2007)

The test suite covers:
  1. Output structure (tuple of 4 arrays)
  2. Output shape (all T-length matching input y)
  3. NaN padding semantics (first r-1 of yhattph; first r-1 and last h of ytph)
  4. AR(1) one-step forecast correctness
  5. Forecast error definition (forerr = ytph - yhattph)
  6. MA(1) multi-step forecasting (MA dies out beyond lag)
  7. One-step ahead (h=1) forecasts
  8. Large forecast horizon (h=20) handling
  9. seregression scaling of ystd
  10. Constant-only model (no AR/MA)
  11-13. MATLAB parity tests via .npy fixtures (AR(1) h=1, ARMA(1,1) h=5, AR(2) h=3)

Per AAP Section 0.7.1:
  - Every migrated function MUST pass numpy.testing.assert_allclose(atol=1e-6, rtol=1e-4)
  - If a MATLAB function errors on invalid input, the Python equivalent must raise
    an equivalent exception

Ref: timeseries/arma_forecaster.m (Kevin Sheppard, Revision 3, 1/1/2007)
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.arma_forecaster import arma_forecaster

# ---------------------------------------------------------------------------
# Numerical parity tolerances per AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Pytest Fixtures — reproducible test data
# ---------------------------------------------------------------------------

@pytest.fixture
def rng_local():
    """Return a seeded random number generator for reproducible test data.

    Uses seed 42 to match fixture generation convention.
    """
    return np.random.default_rng(42)


@pytest.fixture
def ar1_data(rng_local):
    """Generate an AR(1) series: y[t] = 0.5 * y[t-1] + e[t], T=200.

    Returns
    -------
    dict
        Dictionary with keys 'y', 'phi', 'T' for use in AR(1) tests.
    """
    T = 200
    phi = 0.5
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + rng_local.standard_normal()
    return {"y": y, "phi": phi, "T": T}


@pytest.fixture
def arma11_data(rng_local):
    """Generate an ARMA(1,1) series: y[t] = 0.6*y[t-1] + e[t] + 0.3*e[t-1], T=300.

    Returns
    -------
    dict
        Dictionary with keys 'y', 'phi', 'theta', 'T'.
    """
    T = 300
    phi = 0.6
    theta = 0.3
    y = np.zeros(T)
    e = rng_local.standard_normal(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + e[t] + theta * e[t - 1]
    return {"y": y, "phi": phi, "theta": theta, "T": T, "e": e}


@pytest.fixture
def ma1_data(rng_local):
    """Generate a pure MA(1) series: y[t] = 0.1 + e[t] + 0.4*e[t-1], T=250.

    Returns
    -------
    dict
        Dictionary with keys 'y', 'mu', 'theta', 'T'.
    """
    T = 250
    mu = 0.1
    theta = 0.4
    e = rng_local.standard_normal(T)
    y = np.zeros(T)
    y[0] = mu + e[0]
    for t in range(1, T):
        y[t] = mu + e[t] + theta * e[t - 1]
    return {"y": y, "mu": mu, "theta": theta, "T": T}


# ---------------------------------------------------------------------------
# Test 1: Output structure
# ---------------------------------------------------------------------------

class TestArmaForecasterReturns:
    """Tests for arma_forecaster output structure and shapes."""

    def test_arma_forecaster_returns_four_outputs(self, ar1_data):
        """arma_forecaster returns a tuple of exactly 4 elements:
        (yhattph, ytph, forerr, ystd).
        """
        y = ar1_data["y"]
        phi = ar1_data["phi"]
        params = np.array([0.0, phi])  # [constant, phi]
        R = 150
        h = 1
        result = arma_forecaster(
            y, params, 1, np.array([1]), np.array([]), R, h
        )
        assert isinstance(result, tuple), "Expected a tuple return type"
        assert len(result) == 4, f"Expected 4 outputs, got {len(result)}"

    def test_arma_forecaster_output_shapes(self, ar1_data):
        """All array outputs (yhattph, ytph, forerr) must have length T
        matching the input y; ystd must be a scalar float.
        """
        y = ar1_data["y"]
        T = ar1_data["T"]
        phi = ar1_data["phi"]
        params = np.array([0.0, phi])
        R = 150
        h = 1
        yhattph, ytph, forerr, ystd = arma_forecaster(
            y, params, 1, np.array([1]), np.array([]), R, h
        )
        assert yhattph.shape == (T,), f"yhattph shape {yhattph.shape} != ({T},)"
        assert ytph.shape == (T,), f"ytph shape {ytph.shape} != ({T},)"
        assert forerr.shape == (T,), f"forerr shape {forerr.shape} != ({T},)"
        assert isinstance(ystd, float), f"ystd type {type(ystd)} != float"


# ---------------------------------------------------------------------------
# Test 2: NaN padding
# ---------------------------------------------------------------------------

class TestArmaForecasterNanPadding:
    """Tests for NaN padding semantics in arma_forecaster outputs."""

    def test_arma_forecaster_nan_padding_yhattph(self, ar1_data):
        """First r-1 elements of yhattph must be NaN; element at index r-1
        must have a valid forecast value.

        Ref: arma_forecaster.m:212-231 — MATLAB loop starts at t=r,
        so positions 1..(r-1) are NaN in 1-based indexing, mapping to
        Python 0-based indices 0..(r-2).
        """
        y = ar1_data["y"]
        phi = ar1_data["phi"]
        params = np.array([0.0, phi])
        R = 150
        h = 1
        yhattph, _, _, _ = arma_forecaster(
            y, params, 1, np.array([1]), np.array([]), R, h
        )
        # First R-1 elements are NaN
        assert np.all(np.isnan(yhattph[:R - 1])), (
            f"Expected first {R - 1} elements of yhattph to be NaN"
        )
        # Element at index R-1 should be a valid forecast
        assert not np.isnan(yhattph[R - 1]), (
            f"Element at index {R - 1} should be a valid forecast, not NaN"
        )

    def test_arma_forecaster_nan_padding_ytph(self, ar1_data):
        """First r-1 elements and last h elements of ytph must be NaN.

        Ref: arma_forecaster.m:233 —
        ytph = [NaN(r-1,1); yorig(r+h:T); NaN(h,1)]
        """
        y = ar1_data["y"]
        T = ar1_data["T"]
        phi = ar1_data["phi"]
        params = np.array([0.0, phi])
        R = 150
        h = 3
        _, ytph, _, _ = arma_forecaster(
            y, params, 1, np.array([1]), np.array([]), R, h
        )
        # First R-1 elements are NaN
        assert np.all(np.isnan(ytph[:R - 1])), (
            f"Expected first {R - 1} elements of ytph to be NaN"
        )
        # Last h elements are NaN (no data beyond T for comparison)
        assert np.all(np.isnan(ytph[-h:])), (
            f"Expected last {h} elements of ytph to be NaN"
        )
        # Middle section should have valid values
        valid_section = ytph[R - 1:T - h]
        assert not np.any(np.isnan(valid_section)), (
            "Middle section of ytph should not contain NaN"
        )

    def test_arma_forecaster_nan_padding_large_h(self, ar1_data):
        """With large h, more trailing NaN elements appear in ytph."""
        y = ar1_data["y"]
        T = ar1_data["T"]
        phi = ar1_data["phi"]
        params = np.array([0.0, phi])
        R = 100
        h = 20
        _, ytph, _, _ = arma_forecaster(
            y, params, 1, np.array([1]), np.array([]), R, h
        )
        # Last h=20 elements should be NaN
        assert np.all(np.isnan(ytph[-h:])), (
            f"Expected last {h} elements of ytph to be NaN"
        )
        # Number of valid values: T - (R-1) - h = T - R + 1 - h
        expected_valid = T - (R - 1) - h
        actual_valid = np.sum(~np.isnan(ytph))
        assert actual_valid == expected_valid, (
            f"Expected {expected_valid} valid ytph values, got {actual_valid}"
        )


# ---------------------------------------------------------------------------
# Test 3: AR(1) one-step forecast
# ---------------------------------------------------------------------------

def test_arma_forecaster_ar1_onestep(ar1_data):
    """AR(1) one-step ahead forecast: yhat(t+1|t) = constant + phi*y(t).

    For an AR(1) model with no MA component and h=1, the forecast at
    position t should be: constant + phi * y(t).

    Ref: arma_forecaster.m:226 — forward recursion with AR only.
    """
    y = ar1_data["y"]
    T = ar1_data["T"]
    phi = ar1_data["phi"]
    constant_val = 0.0
    params = np.array([constant_val, phi])  # [constant, phi]
    R = 150
    h = 1

    yhattph, ytph, forerr, ystd = arma_forecaster(
        y, params, 1, np.array([1]), np.array([]), R, h
    )

    # Verify shape
    assert yhattph.shape[0] == T

    # First R-1 values should be NaN
    assert np.all(np.isnan(yhattph[:R - 1]))

    # For AR(1) h=1, the forecast at position t is:
    # yhat(t+1|t) = constant + phi * y(t)
    # Check the valid forecast region
    for t_idx in range(R - 1, T):
        expected_forecast = constant_val + phi * y[t_idx]
        npt.assert_allclose(
            yhattph[t_idx], expected_forecast, atol=ATOL, rtol=RTOL,
            err_msg=f"AR(1) one-step forecast mismatch at index {t_idx}"
        )


# ---------------------------------------------------------------------------
# Test 4: Forecast error is the difference
# ---------------------------------------------------------------------------

def test_arma_forecaster_forecast_error_is_difference(ar1_data):
    """Forecast error forerr = ytph - yhattph at positions where both
    are non-NaN.

    Ref: arma_forecaster.m:237 — forerr = ytph - yhattph
    """
    y = ar1_data["y"]
    phi = ar1_data["phi"]
    params = np.array([0.0, phi])
    R = 150
    h = 1

    yhattph, ytph, forerr, _ = arma_forecaster(
        y, params, 1, np.array([1]), np.array([]), R, h
    )

    # Where both are not NaN, forerr == ytph - yhattph
    mask = ~np.isnan(ytph) & ~np.isnan(yhattph)
    assert np.sum(mask) > 0, "No valid positions to compare"
    npt.assert_allclose(
        forerr[mask], ytph[mask] - yhattph[mask], atol=1e-12,
        err_msg="forerr != ytph - yhattph"
    )


# ---------------------------------------------------------------------------
# Test 5: MA(1) multi-step forecast — MA dies out for h >= 2
# ---------------------------------------------------------------------------

def test_arma_forecaster_ma1_forecast(ma1_data):
    """For a pure MA(1) model with h >= 2, the MA component vanishes
    beyond the first step. Multi-step forecasts should converge to just
    the constant value.

    For MA(1): y[t] = mu + e[t] + theta*e[t-1]
    h=1: yhat(t+1|t) = mu + theta*e(t)
    h>=2: yhat(t+h|t) = mu  (MA component dies out)
    """
    y = ma1_data["y"]
    T = ma1_data["T"]
    mu = ma1_data["mu"]
    theta = ma1_data["theta"]
    # parameters = [constant, theta] since we have constant=1, q=[1], p=[]
    params = np.array([mu, theta])
    R = 180
    h = 5  # MA(1) effect vanishes for h >= 2

    yhattph, _, _, _ = arma_forecaster(
        y, params, 1, np.array([]), np.array([1]), R, h
    )

    # For h=5, the multi-step forecast should converge toward the constant
    # because the MA(1) component only contributes at h=1.
    # After multiple steps of forward recursion with AR=0 and MA errors=0,
    # the forecast is just the constant (mu).
    valid_mask = ~np.isnan(yhattph)
    forecasts = yhattph[valid_mask]

    # All multi-step forecasts should be approximately mu
    npt.assert_allclose(
        forecasts, mu, atol=1e-4, rtol=1e-3,
        err_msg="MA(1) h=5 forecasts should converge to the constant"
    )


# ---------------------------------------------------------------------------
# Test 6: h=1 one-step ahead
# ---------------------------------------------------------------------------

def test_arma_forecaster_h_equals_1(ar1_data):
    """One-step ahead forecast for AR(1) — verify basic structure.

    With h=1, the last element of ytph should be NaN (no y(T+1)),
    and ystd should equal seregression (since psi_0 = 1 for h=1).
    """
    y = ar1_data["y"]
    T = ar1_data["T"]
    phi = ar1_data["phi"]
    params = np.array([0.0, phi])
    R = 100
    h = 1

    yhattph, ytph, forerr, ystd = arma_forecaster(
        y, params, 1, np.array([1]), np.array([]), R, h
    )

    # ystd for h=1 should equal seregression (default 1.0)
    # because the impulse response at lag 0 is always 1:
    # ystd = sqrt(psi_0^2) * seregression = 1.0 * 1.0 = 1.0
    npt.assert_allclose(ystd, 1.0, atol=ATOL,
                        err_msg="ystd for h=1 should equal seregression (1.0)")

    # Last element of ytph is NaN (no data at T+1)
    assert np.isnan(ytph[-1]), "Last element of ytph should be NaN for h=1"

    # yhattph at last index should be a valid forecast (out-of-sample)
    assert not np.isnan(yhattph[-1]), (
        "Last element of yhattph should be a valid OOS forecast"
    )


# ---------------------------------------------------------------------------
# Test 7: Large forecast horizon
# ---------------------------------------------------------------------------

def test_arma_forecaster_large_h(ar1_data):
    """Large forecast horizon h=20 — verify output structure and trailing NaN.

    With h=20, the last 20 positions in ytph should be NaN.
    """
    y = ar1_data["y"]
    T = ar1_data["T"]
    phi = ar1_data["phi"]
    params = np.array([0.0, phi])
    R = 100
    h = 20

    yhattph, ytph, forerr, ystd = arma_forecaster(
        y, params, 1, np.array([1]), np.array([]), R, h
    )

    # Output shape matches T
    assert yhattph.shape == (T,)

    # Last h positions of ytph must be NaN
    assert np.all(np.isnan(ytph[-h:])), (
        f"Last {h} elements of ytph should be NaN"
    )

    # yhattph should have T - (R-1) valid forecasts
    valid_yhattph = np.sum(~np.isnan(yhattph))
    expected_valid = T - (R - 1)
    assert valid_yhattph == expected_valid, (
        f"Expected {expected_valid} valid forecasts, got {valid_yhattph}"
    )

    # ystd for h>1 should be > seregression due to accumulating uncertainty
    # For AR(1) with phi=0.5, impulse response: psi_k = phi^k
    # Var = sum(psi_k^2) * sigma^2 = (1 + phi^2 + phi^4 + ... + phi^{2(h-1)}) * 1
    assert ystd > 1.0, (
        "ystd for h=20 AR(1) should be > seregression(=1) due to accumulated uncertainty"
    )


# ---------------------------------------------------------------------------
# Test 8: seregression affects ystd
# ---------------------------------------------------------------------------

def test_arma_forecaster_seregression_affects_ystd(ar1_data):
    """Larger seregression must produce proportionally larger ystd.

    ystd = seregression * sqrt(sum(impulse_response^2)), so doubling
    seregression doubles ystd.
    """
    y = ar1_data["y"]
    phi = ar1_data["phi"]
    params = np.array([0.0, phi])
    R = 150
    h = 5

    _, _, _, ystd1 = arma_forecaster(
        y, params, 1, np.array([1]), np.array([]), R, h, seregression=1.0
    )
    _, _, _, ystd2 = arma_forecaster(
        y, params, 1, np.array([1]), np.array([]), R, h, seregression=2.0
    )
    _, _, _, ystd3 = arma_forecaster(
        y, params, 1, np.array([1]), np.array([]), R, h, seregression=0.5
    )

    # ystd should scale linearly with seregression
    npt.assert_allclose(ystd2, 2.0 * ystd1, atol=ATOL,
                        err_msg="ystd should double when seregression doubles")
    npt.assert_allclose(ystd3, 0.5 * ystd1, atol=ATOL,
                        err_msg="ystd should halve when seregression halves")

    # Larger seregression → strictly larger ystd
    assert ystd2 > ystd1, "seregression=2 should give larger ystd"
    assert ystd1 > ystd3, "seregression=1 should give larger ystd than 0.5"


# ---------------------------------------------------------------------------
# Test 9: Constant-only model
# ---------------------------------------------------------------------------

def test_arma_forecaster_constant_only_model(rng_local):
    """With p=[], q=[], the forecast should be just the constant value.

    The function internally inserts dummy AR(1)=0 and MA(1)=0, so the
    h-step ahead forecast reduces to: yhat(t+h|t) = constant.
    """
    T = 200
    y = rng_local.standard_normal(T)
    mu = np.mean(y)  # Use sample mean as constant estimate
    params = np.array([mu])  # [constant]
    R = 120
    h = 3

    yhattph, ytph, forerr, ystd = arma_forecaster(
        y, params, 1, np.array([]), np.array([]), R, h
    )

    # All valid forecasts should be approximately mu
    valid = ~np.isnan(yhattph)
    npt.assert_allclose(
        yhattph[valid], mu, atol=ATOL, rtol=RTOL,
        err_msg="Constant-only forecast should equal the constant"
    )


# ---------------------------------------------------------------------------
# Test 10: ARMA(1,1) output consistency
# ---------------------------------------------------------------------------

def test_arma_forecaster_arma11_structure(arma11_data):
    """ARMA(1,1) model — verify output structure and that forecasts are
    finite (no Inf/NaN in the valid region).
    """
    y = arma11_data["y"]
    T = arma11_data["T"]
    phi = arma11_data["phi"]
    theta = arma11_data["theta"]
    params = np.array([0.0, phi, theta])  # [constant=0, phi, theta]
    R = 200
    h = 3

    yhattph, ytph, forerr, ystd = arma_forecaster(
        y, params, 1, np.array([1]), np.array([1]), R, h
    )

    # Shape
    assert yhattph.shape == (T,)

    # Valid forecasts should be finite
    valid_yhattph = yhattph[~np.isnan(yhattph)]
    assert np.all(np.isfinite(valid_yhattph)), (
        "All valid ARMA(1,1) forecasts should be finite"
    )

    # ystd should be positive
    assert ystd > 0, "ystd must be positive"


# ---------------------------------------------------------------------------
# Test 11: No-constant model (constant=0)
# ---------------------------------------------------------------------------

def test_arma_forecaster_no_constant(ar1_data):
    """With constant=0, the model has no intercept term.

    Parameters vector has only the AR coefficient.
    """
    y = ar1_data["y"]
    T = ar1_data["T"]
    phi = ar1_data["phi"]
    params = np.array([phi])  # [phi] — no constant
    R = 150
    h = 1

    yhattph, ytph, forerr, ystd = arma_forecaster(
        y, params, 0, np.array([1]), np.array([]), R, h
    )

    # Verify shape
    assert yhattph.shape == (T,)

    # For AR(1) with no constant and h=1:
    # yhat(t+1|t) = phi * y(t)
    for t_idx in range(R - 1, T):
        expected_forecast = phi * y[t_idx]
        npt.assert_allclose(
            yhattph[t_idx], expected_forecast, atol=ATOL, rtol=RTOL,
            err_msg=f"No-constant AR(1) forecast mismatch at index {t_idx}"
        )


# ---------------------------------------------------------------------------
# Test 12: ystd computation for known AR(1) model
# ---------------------------------------------------------------------------

def test_arma_forecaster_ystd_ar1_analytical():
    """Verify ystd for AR(1) against analytical formula.

    For AR(1) with coefficient phi, the h-step forecast variance is:
        Var = sigma^2 * sum_{k=0}^{h-1} phi^{2k}
    so ystd = sigma * sqrt(sum_{k=0}^{h-1} phi^{2k}).
    """
    rng = np.random.default_rng(123)
    T = 500
    phi = 0.7
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + rng.standard_normal()

    sigma = 1.5
    params = np.array([0.0, phi])  # [constant=0, phi]
    R = 300
    h = 10

    _, _, _, ystd = arma_forecaster(
        y, params, 1, np.array([1]), np.array([]), R, h,
        seregression=sigma
    )

    # Analytical ystd for AR(1): sigma * sqrt(sum_{k=0}^{h-1} phi^{2k})
    impulse_response_sq = np.array([phi ** (2 * k) for k in range(h)])
    expected_ystd = sigma * np.sqrt(np.sum(impulse_response_sq))

    npt.assert_allclose(
        ystd, expected_ystd, atol=ATOL, rtol=RTOL,
        err_msg="ystd for AR(1) does not match analytical formula"
    )


# ---------------------------------------------------------------------------
# MATLAB Parity Tests — loaded from .npy fixtures
# ---------------------------------------------------------------------------

@pytest.mark.parity
def test_arma_forecaster_parity_ar1_h1(timeseries_fixture_dir):
    """MATLAB parity: AR(1) model with h=1 forecast horizon.

    Loads the ar1_h1 test case from the fixture file and compares all
    4 outputs against the MATLAB reference at ATOL=1e-6, RTOL=1e-4.
    """
    fixture_path = timeseries_fixture_dir / "arma_forecaster.npy"
    if not fixture_path.exists():
        pytest.skip("Fixture not found: arma_forecaster.npy")

    fixture = np.load(fixture_path, allow_pickle=True).item()

    y = fixture["y"]
    params = fixture["ar1_h1_parameters"]
    constant = int(fixture["ar1_h1_constant"])
    p = fixture["ar1_h1_p"]
    q = fixture["ar1_h1_q"]
    r = int(fixture["ar1_h1_r"])
    h = int(fixture["ar1_h1_h"])
    seregression = float(fixture["ar1_h1_seregression"])

    yhattph, ytph, forerr, ystd = arma_forecaster(
        y, params, constant, p, q, r, h, seregression=seregression
    )

    ref_yhattph = fixture["ar1_h1_yhattph"]
    ref_ytph = fixture["ar1_h1_ytph"]
    ref_forerr = fixture["ar1_h1_forerr"]
    ref_ystd = float(fixture["ar1_h1_ystd"])

    # Compare yhattph (non-NaN positions)
    mask_yhat = ~np.isnan(ref_yhattph)
    npt.assert_allclose(
        yhattph[mask_yhat], ref_yhattph[mask_yhat], atol=ATOL, rtol=RTOL,
        err_msg="AR(1) h=1 yhattph parity failure"
    )

    # Compare ytph (non-NaN positions)
    mask_yt = ~np.isnan(ref_ytph)
    npt.assert_allclose(
        ytph[mask_yt], ref_ytph[mask_yt], atol=ATOL, rtol=RTOL,
        err_msg="AR(1) h=1 ytph parity failure"
    )

    # Compare forerr (non-NaN positions)
    mask_fe = ~np.isnan(ref_forerr)
    npt.assert_allclose(
        forerr[mask_fe], ref_forerr[mask_fe], atol=ATOL, rtol=RTOL,
        err_msg="AR(1) h=1 forerr parity failure"
    )

    # Compare ystd (scalar)
    npt.assert_allclose(
        ystd, ref_ystd, atol=ATOL, rtol=RTOL,
        err_msg="AR(1) h=1 ystd parity failure"
    )


@pytest.mark.parity
def test_arma_forecaster_parity_arma11_h5(timeseries_fixture_dir):
    """MATLAB parity: ARMA(1,1) model with h=5 forecast horizon.

    Loads the arma11_h5 test case from the fixture file and compares all
    4 outputs against the MATLAB reference at ATOL=1e-6, RTOL=1e-4.
    """
    fixture_path = timeseries_fixture_dir / "arma_forecaster.npy"
    if not fixture_path.exists():
        pytest.skip("Fixture not found: arma_forecaster.npy")

    fixture = np.load(fixture_path, allow_pickle=True).item()

    y = fixture["y"]
    params = fixture["arma11_h5_parameters"]
    constant = int(fixture["arma11_h5_constant"])
    p = fixture["arma11_h5_p"]
    q = fixture["arma11_h5_q"]
    r = int(fixture["arma11_h5_r"])
    h = int(fixture["arma11_h5_h"])
    seregression = float(fixture["arma11_h5_seregression"])

    yhattph, ytph, forerr, ystd = arma_forecaster(
        y, params, constant, p, q, r, h, seregression=seregression
    )

    ref_yhattph = fixture["arma11_h5_yhattph"]
    ref_ytph = fixture["arma11_h5_ytph"]
    ref_forerr = fixture["arma11_h5_forerr"]
    ref_ystd = float(fixture["arma11_h5_ystd"])

    # Compare yhattph (non-NaN positions)
    mask_yhat = ~np.isnan(ref_yhattph)
    npt.assert_allclose(
        yhattph[mask_yhat], ref_yhattph[mask_yhat], atol=ATOL, rtol=RTOL,
        err_msg="ARMA(1,1) h=5 yhattph parity failure"
    )

    # Compare ytph (non-NaN positions)
    mask_yt = ~np.isnan(ref_ytph)
    npt.assert_allclose(
        ytph[mask_yt], ref_ytph[mask_yt], atol=ATOL, rtol=RTOL,
        err_msg="ARMA(1,1) h=5 ytph parity failure"
    )

    # Compare forerr (non-NaN positions)
    mask_fe = ~np.isnan(ref_forerr)
    npt.assert_allclose(
        forerr[mask_fe], ref_forerr[mask_fe], atol=ATOL, rtol=RTOL,
        err_msg="ARMA(1,1) h=5 forerr parity failure"
    )

    # Compare ystd (scalar)
    npt.assert_allclose(
        ystd, ref_ystd, atol=ATOL, rtol=RTOL,
        err_msg="ARMA(1,1) h=5 ystd parity failure"
    )


@pytest.mark.parity
def test_arma_forecaster_parity_ar2_h3(timeseries_fixture_dir):
    """MATLAB parity: AR(2) model with h=3 forecast horizon.

    Loads the ar2_h3 test case from the fixture file and compares all
    4 outputs against the MATLAB reference at ATOL=1e-6, RTOL=1e-4.
    """
    fixture_path = timeseries_fixture_dir / "arma_forecaster.npy"
    if not fixture_path.exists():
        pytest.skip("Fixture not found: arma_forecaster.npy")

    fixture = np.load(fixture_path, allow_pickle=True).item()

    y = fixture["y"]
    params = fixture["ar2_h3_parameters"]
    constant = int(fixture["ar2_h3_constant"])
    p = fixture["ar2_h3_p"]
    q = fixture["ar2_h3_q"]
    r = int(fixture["ar2_h3_r"])
    h = int(fixture["ar2_h3_h"])
    seregression = float(fixture["ar2_h3_seregression"])

    yhattph, ytph, forerr, ystd = arma_forecaster(
        y, params, constant, p, q, r, h, seregression=seregression
    )

    ref_yhattph = fixture["ar2_h3_yhattph"]
    ref_ytph = fixture["ar2_h3_ytph"]
    ref_forerr = fixture["ar2_h3_forerr"]
    ref_ystd = float(fixture["ar2_h3_ystd"])

    # Compare yhattph (non-NaN positions)
    mask_yhat = ~np.isnan(ref_yhattph)
    npt.assert_allclose(
        yhattph[mask_yhat], ref_yhattph[mask_yhat], atol=ATOL, rtol=RTOL,
        err_msg="AR(2) h=3 yhattph parity failure"
    )

    # Compare ytph (non-NaN positions)
    mask_yt = ~np.isnan(ref_ytph)
    npt.assert_allclose(
        ytph[mask_yt], ref_ytph[mask_yt], atol=ATOL, rtol=RTOL,
        err_msg="AR(2) h=3 ytph parity failure"
    )

    # Compare forerr (non-NaN positions)
    mask_fe = ~np.isnan(ref_forerr)
    npt.assert_allclose(
        forerr[mask_fe], ref_forerr[mask_fe], atol=ATOL, rtol=RTOL,
        err_msg="AR(2) h=3 forerr parity failure"
    )

    # Compare ystd (scalar)
    npt.assert_allclose(
        ystd, ref_ystd, atol=ATOL, rtol=RTOL,
        err_msg="AR(2) h=3 ystd parity failure"
    )


# ---------------------------------------------------------------------------
# Test 13: NaN pattern consistency across all parity fixture test cases
# ---------------------------------------------------------------------------

@pytest.mark.parity
def test_arma_forecaster_parity_nan_patterns(timeseries_fixture_dir):
    """Verify NaN patterns in Python output match MATLAB reference exactly
    across all fixture test cases.
    """
    fixture_path = timeseries_fixture_dir / "arma_forecaster.npy"
    if not fixture_path.exists():
        pytest.skip("Fixture not found: arma_forecaster.npy")

    fixture = np.load(fixture_path, allow_pickle=True).item()
    y = fixture["y"]

    for prefix in ("ar1_h1", "arma11_h5", "ar2_h3"):
        params = fixture[f"{prefix}_parameters"]
        constant = int(fixture[f"{prefix}_constant"])
        p = fixture[f"{prefix}_p"]
        q = fixture[f"{prefix}_q"]
        r = int(fixture[f"{prefix}_r"])
        h = int(fixture[f"{prefix}_h"])
        seregression = float(fixture[f"{prefix}_seregression"])

        yhattph, ytph, forerr, _ = arma_forecaster(
            y, params, constant, p, q, r, h, seregression=seregression
        )

        ref_yhattph = fixture[f"{prefix}_yhattph"]
        ref_ytph = fixture[f"{prefix}_ytph"]

        # NaN patterns must match exactly
        np.testing.assert_array_equal(
            np.isnan(yhattph), np.isnan(ref_yhattph),
            err_msg=f"{prefix}: yhattph NaN pattern mismatch"
        )
        np.testing.assert_array_equal(
            np.isnan(ytph), np.isnan(ref_ytph),
            err_msg=f"{prefix}: ytph NaN pattern mismatch"
        )
