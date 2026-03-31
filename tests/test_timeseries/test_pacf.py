"""Pytest tests for ``mfe_toolbox.timeseries.pacf`` — sample partial
autocorrelation function via Levinson-Durbin / Yule-Walker.

Tests verify:

* **Return type** — ``pacf`` returns a ``tuple[np.ndarray, np.ndarray]``.
* **Output shape** — ``pacf_vals.shape == (lags + 1,)`` and
  ``bounds.shape == (lags + 1,)``.
* **Lag-0 identity** — ``pacf_vals[0] == 1.0`` for any valid input.
* **White noise property** — sample PACFs are near zero for i.i.d. data.
* **AR(1) cutoff from data** — first PACF coefficient recovers the
  AR(1) parameter from simulated data.
* **Confidence bounds positivity** — all bounds are positive and equal
  ``1.96 / sqrt(T)``.
* **Boundedness** — ``|pacf_vals[k]| <= 1`` for all lags and data types.
* **MATLAB/Octave parity** — fixture comparison at ``atol=1e-6``.
* **Error validation** — ``ValueError`` for invalid inputs.
* **Mean-demeaning invariance** — results unchanged by mean shift.
* **Cumulative sum series parity** — fixture comparison for integrated
  series at ``atol=1e-6``.

Reference: ``timeseries/pacf.m`` — Kevin Sheppard, MFE Toolbox v4.0.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.pacf import pacf

# ---------------------------------------------------------------------------
# Numerical Parity Tolerance Constants
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# =========================================================================
# Test 1: Return Type Verification
# =========================================================================

def test_pacf_returns_tuple() -> None:
    """Verify ``pacf`` returns a tuple of two :class:`numpy.ndarray` objects.

    The refactored ``pacf`` function returns ``(pacf_vals, bounds)`` as a
    two-element tuple.  Both elements must be NumPy arrays.

    Ref: pacf.m — Levinson-Durbin recursion output construction.
    """
    rng = np.random.default_rng(42)
    y = rng.standard_normal(500)
    result = pacf(y, 20)

    # Return value is a tuple
    assert isinstance(result, tuple), (
        f"Expected tuple return type, got {type(result).__name__}"
    )
    # Exactly 2 elements
    assert len(result) == 2, (
        f"Expected tuple of length 2, got length {len(result)}"
    )
    # Both elements are ndarrays
    pacf_vals, bounds = result
    assert isinstance(pacf_vals, np.ndarray), (
        f"Expected pacf_vals to be np.ndarray, got {type(pacf_vals).__name__}"
    )
    assert isinstance(bounds, np.ndarray), (
        f"Expected bounds to be np.ndarray, got {type(bounds).__name__}"
    )


# =========================================================================
# Test 2: Output Shape Verification
# =========================================================================

@pytest.mark.parametrize("lags", [1, 5, 10, 20, 50])
def test_pacf_output_shape(lags: int) -> None:
    """Verify ``pacf_vals`` and ``bounds`` have shape ``(lags + 1,)``.

    Both the PACF values array and the confidence bounds array must have
    exactly ``lags + 1`` elements: index 0 for lag-0 (identity) and
    indices 1 through ``lags`` for the computed partial autocorrelations.

    Parameters
    ----------
    lags : int
        Number of lags to compute (parametrized: 1, 5, 10, 20, 50).
    """
    rng = np.random.default_rng(42)
    y = rng.standard_normal(500)
    pacf_vals, bounds = pacf(y, lags)

    assert pacf_vals.shape == (lags + 1,), (
        f"pacf_vals.shape={pacf_vals.shape}, expected ({lags + 1},)"
    )
    assert bounds.shape == (lags + 1,), (
        f"bounds.shape={bounds.shape}, expected ({lags + 1},)"
    )


# =========================================================================
# Test 3: Lag-0 Identity
# =========================================================================

def test_pacf_lag_zero_identity() -> None:
    """Verify ``pacf_vals[0] == 1.0`` for multiple data types.

    The partial autocorrelation at lag 0 is, by definition, 1.0 for any
    stationary or non-stationary sample input.  This tests white noise,
    AR(1), and random walk data to confirm the convention is preserved.
    """
    rng = np.random.default_rng(42)

    # White noise
    y_wn = rng.standard_normal(500)
    pacf_vals_wn, _ = pacf(y_wn, 10)
    npt.assert_allclose(
        pacf_vals_wn[0], 1.0, atol=ATOL, rtol=RTOL,
        err_msg="Lag-0 identity failed for white noise data",
    )

    # AR(1) simulated data (phi=0.5)
    T_ar = 1000
    y_ar = np.zeros(T_ar)
    for t in range(1, T_ar):
        y_ar[t] = 0.5 * y_ar[t - 1] + rng.standard_normal()
    pacf_vals_ar, _ = pacf(y_ar, 10)
    npt.assert_allclose(
        pacf_vals_ar[0], 1.0, atol=ATOL, rtol=RTOL,
        err_msg="Lag-0 identity failed for AR(1) data",
    )

    # Random walk (cumulative sum)
    y_rw = np.cumsum(rng.standard_normal(200))
    pacf_vals_rw, _ = pacf(y_rw, 10)
    npt.assert_allclose(
        pacf_vals_rw[0], 1.0, atol=ATOL, rtol=RTOL,
        err_msg="Lag-0 identity failed for random walk data",
    )


# =========================================================================
# Test 4: White Noise Property
# =========================================================================

def test_pacf_white_noise() -> None:
    """Verify sample PACFs are near zero for i.i.d. white noise.

    For white noise of length ``T=500``, the population PACF at all
    non-zero lags is 0.  The sample estimates should be small in
    magnitude.  We allow a relaxed threshold of ``3.0 / sqrt(T)`` and
    permit at most 2 out of 20 lags to exceed it (finite-sample
    tolerance).
    """
    rng = np.random.default_rng(42)
    y = rng.standard_normal(500)
    pacf_vals, bounds = pacf(y, 20)

    # Threshold for approximate zero: 3 standard errors under white noise
    T = len(y)
    threshold = 3.0 / np.sqrt(T)

    # Count how many lags exceed the threshold (lags 1..20)
    n_exceeding = np.sum(np.abs(pacf_vals[1:]) > threshold)
    assert n_exceeding <= 2, (
        f"Too many sample PACF values exceed the white noise threshold: "
        f"{n_exceeding} out of 20 exceed {threshold:.4f}. "
        f"PACF values: {pacf_vals[1:]}"
    )


# =========================================================================
# Test 5: AR(1) Data Cutoff
# =========================================================================

def test_pacf_ar1_data() -> None:
    """Verify PACF recovers AR(1) coefficient from simulated data.

    For an AR(1) process ``y[t] = 0.7 * y[t-1] + e[t]``, the
    theoretical PACF is:

    * ``pacf[1] = 0.7`` (the AR coefficient)
    * ``pacf[k] = 0`` for ``k >= 2``

    With ``T=5000`` samples, the sample estimate ``pacf_vals[1]`` should
    be close to ``0.7`` (``atol=0.05``) and higher-order PACFs should be
    near zero (``atol=0.1``).

    Ref: pacf.m — Levinson-Durbin recursion.
    """
    rng = np.random.default_rng(42)
    T = 5000
    phi = 0.7
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + rng.standard_normal()

    pacf_vals, bounds = pacf(y, 10)

    # First PACF should be near the true AR(1) coefficient
    npt.assert_allclose(
        pacf_vals[1], phi, atol=0.05,
        err_msg=(
            f"PACF at lag 1 should approximate AR(1) coefficient {phi}. "
            f"Got {pacf_vals[1]:.6f}"
        ),
    )

    # Higher-order PACFs should be near zero (finite-sample tolerance)
    npt.assert_allclose(
        pacf_vals[3:], 0.0, atol=0.1,
        err_msg=(
            "Higher-order PACFs for AR(1) data should be near zero. "
            f"Got max |pacf_vals[3:]| = {np.max(np.abs(pacf_vals[3:])):.6f}"
        ),
    )


# =========================================================================
# Test 6: Confidence Bounds Positivity
# =========================================================================

def test_pacf_bounds_positive() -> None:
    """Verify all confidence bounds are positive and match ``1.96 / sqrt(T)``.

    The asymptotic confidence bounds under the null hypothesis of white
    noise are ``1.96 / sqrt(T)`` at every lag, including lag 0.  All
    values must be strictly positive.
    """
    rng = np.random.default_rng(42)
    y = rng.standard_normal(500)
    pacf_vals, bounds = pacf(y, 20)

    # All bounds must be positive
    assert np.all(bounds > 0), (
        "All confidence bounds must be strictly positive. "
        f"Min bound = {np.min(bounds)}"
    )

    # Bounds must match the asymptotic formula exactly
    T = len(y)
    expected_bound = 1.96 / np.sqrt(T)
    npt.assert_allclose(
        bounds, expected_bound, atol=ATOL,
        err_msg=(
            f"Bounds should all equal 1.96/sqrt({T}) = {expected_bound:.10f}"
        ),
    )


# =========================================================================
# Test 7: Boundedness
# =========================================================================

@pytest.mark.parametrize("data_id", ["white_noise", "ar1_phi09", "random_walk"])
def test_pacf_bounded(data_id: str) -> None:
    """Verify ``|pacf_vals[k]| <= 1.0`` for multiple sample data types.

    Partial autocorrelations are correlation coefficients and must lie
    in the interval ``[-1, 1]`` (within numerical tolerance).

    Parameters
    ----------
    data_id : str
        Identifier for the data generation strategy (parametrized).
    """
    rng = np.random.default_rng(42)

    if data_id == "white_noise":
        y = rng.standard_normal(500)
        lags = 20
    elif data_id == "ar1_phi09":
        # Simulate AR(1) with phi=0.9
        T = 1000
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.9 * y[t - 1] + rng.standard_normal()
        lags = 20
    elif data_id == "random_walk":
        y = np.cumsum(rng.standard_normal(200))
        lags = 15
    else:
        raise ValueError(f"Unknown data_id: {data_id}")

    pacf_vals, _ = pacf(y, lags)
    assert np.all(np.abs(pacf_vals) <= 1.0 + ATOL), (
        f"PACF values must lie in [-1, 1].  "
        f"Max |pacf| = {np.max(np.abs(pacf_vals)):.10f} for data_id={data_id}"
    )


# =========================================================================
# Test 8: Parity Against MATLAB/Octave Fixtures
# =========================================================================

@pytest.mark.parity
def test_pacf_parity(timeseries_fixture_dir) -> None:
    """Compare Python sample PACF output against MATLAB/Octave reference.

    Loads fixture data from ``tests/fixtures/timeseries/pacf.npy``
    containing two test cases:

    * **Case 1:** ``randn(500, 1)`` with 20 lags — standard normal series.
    * **Case 2:** ``cumsum(randn(200, 1))`` with 15 lags — random walk.

    Asserts numerical parity at ``atol=1e-6`` and ``rtol=1e-4``.

    Parameters
    ----------
    timeseries_fixture_dir : Path
        Session-scoped fixture from ``conftest.py`` providing the path
        to ``tests/fixtures/timeseries/``.

    Ref: pacf.m — Levinson-Durbin recursion with biased autocovariance.
    """
    fixture_path = timeseries_fixture_dir / "pacf.npy"
    if not fixture_path.exists():
        pytest.skip(
            f"Fixture file not found: {fixture_path}. "
            "Run fixture generation first."
        )

    fixture = np.load(fixture_path, allow_pickle=True).item()

    # ------------------------------------------------------------------
    # Case 1: Standard normal series — randn(500,1) with 20 lags
    # ------------------------------------------------------------------
    y_case1 = np.asarray(fixture["y_case1"], dtype=np.float64).ravel()
    lags_case1 = int(fixture["lags_case1"])
    expected_pacf_case1 = np.asarray(
        fixture["pacf_case1"], dtype=np.float64
    ).ravel()
    expected_bounds_case1 = np.asarray(
        fixture["bounds_case1"], dtype=np.float64
    ).ravel()

    pacf_vals1, bounds1 = pacf(y_case1, lags_case1)

    npt.assert_allclose(
        pacf_vals1, expected_pacf_case1, atol=ATOL, rtol=RTOL,
        err_msg="PACF parity failed for Case 1 (randn 500, 20 lags)",
    )
    npt.assert_allclose(
        bounds1, expected_bounds_case1, atol=ATOL, rtol=RTOL,
        err_msg="Bounds parity failed for Case 1 (randn 500, 20 lags)",
    )

    # ------------------------------------------------------------------
    # Case 2: Cumulative sum (random walk) — cumsum(randn(200,1)), 15 lags
    # ------------------------------------------------------------------
    y_case2 = np.asarray(fixture["y_case2"], dtype=np.float64).ravel()
    lags_case2 = int(fixture["lags_case2"])
    expected_pacf_case2 = np.asarray(
        fixture["pacf_case2"], dtype=np.float64
    ).ravel()
    expected_bounds_case2 = np.asarray(
        fixture["bounds_case2"], dtype=np.float64
    ).ravel()

    pacf_vals2, bounds2 = pacf(y_case2, lags_case2)

    npt.assert_allclose(
        pacf_vals2, expected_pacf_case2, atol=ATOL, rtol=RTOL,
        err_msg="PACF parity failed for Case 2 (cumsum 200, 15 lags)",
    )
    npt.assert_allclose(
        bounds2, expected_bounds_case2, atol=ATOL, rtol=RTOL,
        err_msg="Bounds parity failed for Case 2 (cumsum 200, 15 lags)",
    )


# =========================================================================
# Test 9: Error — Lags Too Large
# =========================================================================

def test_pacf_error_lags_too_large() -> None:
    """Verify ``ValueError`` when ``lags >= len(y) / 2``.

    The safety threshold prevents numerical instability in the
    Levinson-Durbin recursion when too many lags are requested relative
    to the sample size.

    * ``lags = len(y) / 2`` must raise ``ValueError`` (boundary).
    * ``lags > len(y) / 2`` must raise ``ValueError``.
    * ``lags = len(y) / 2 - 1`` must NOT raise (valid boundary).
    """
    rng = np.random.default_rng(42)
    y = rng.standard_normal(100)

    # lags = 50 == len(y)/2 = 50 → should raise
    with pytest.raises(ValueError, match="lags must be less than"):
        pacf(y, 50)

    # lags = 60 > len(y)/2 = 50 → should raise
    with pytest.raises(ValueError, match="lags must be less than"):
        pacf(y, 60)

    # lags = 49 < len(y)/2 = 50 → should NOT raise
    pacf_vals, bounds = pacf(y, 49)
    assert pacf_vals.shape == (50,), (
        "pacf(y, 49) should return (50,)-shaped array for boundary case"
    )


# =========================================================================
# Test 10: Error — Invalid Input
# =========================================================================

def test_pacf_error_invalid_input() -> None:
    """Verify ``ValueError`` for various invalid inputs.

    Tests cover:

    * **Non-1D array:** 2-D square matrix ``(5, 5)`` must be rejected.
    * **Empty array:** zero-length array must be rejected.
    * **Non-numeric input:** string array must be rejected.
    * **Non-positive lags:** ``lags=0`` and ``lags=-1`` must be rejected.
    """
    # Non-1D input (square matrix)
    with pytest.raises(ValueError):
        pacf(np.ones((5, 5)), 2)

    # Empty input
    with pytest.raises(ValueError):
        pacf(np.array([]), 1)

    # Non-numeric input (string array)
    with pytest.raises((ValueError, TypeError)):
        pacf(np.array(["a", "b"]), 1)

    # Invalid lags: zero
    with pytest.raises(ValueError):
        pacf(np.ones(100), 0)

    # Invalid lags: negative
    with pytest.raises(ValueError):
        pacf(np.ones(100), -1)


# =========================================================================
# Test 11: Mean-Demeaning Invariance
# =========================================================================

def test_pacf_demeaning() -> None:
    """Verify PACF results are invariant to additive mean shift.

    The ``pacf`` function subtracts the sample mean internally, so
    ``pacf(y, lags)`` and ``pacf(y + C, lags)`` must produce identical
    PACF values and bounds for any constant ``C``.
    """
    rng = np.random.default_rng(42)
    y = rng.standard_normal(300)
    lags = 15

    pacf_vals1, bounds1 = pacf(y, lags)
    pacf_vals2, bounds2 = pacf(y + 100.0, lags)

    npt.assert_allclose(
        pacf_vals1, pacf_vals2, atol=ATOL,
        err_msg="PACF values should be invariant to mean shift (y + 100)",
    )
    npt.assert_allclose(
        bounds1, bounds2, atol=ATOL,
        err_msg="Bounds should be invariant to mean shift (same T)",
    )


# =========================================================================
# Test 12: Cumulative Sum Series Parity
# =========================================================================

@pytest.mark.parity
def test_pacf_cumsum_series(timeseries_fixture_dir) -> None:
    """Compare PACF of cumulative sum series against MATLAB/Octave reference.

    Loads the cumulative sum (random walk) test case from the fixture
    file and verifies parity at ``atol=1e-6``.  This is a separate
    dedicated test for the integrated-series case complementing
    ``test_pacf_parity`` Case 2.

    Parameters
    ----------
    timeseries_fixture_dir : Path
        Session-scoped fixture from ``conftest.py`` providing the path
        to ``tests/fixtures/timeseries/``.

    Ref: pacf.m — Levinson-Durbin recursion with biased autocovariance.
    """
    fixture_path = timeseries_fixture_dir / "pacf.npy"
    if not fixture_path.exists():
        pytest.skip(
            f"Fixture file not found: {fixture_path}. "
            "Run fixture generation first."
        )

    fixture = np.load(fixture_path, allow_pickle=True).item()

    # Extract cumulative sum case (Case 2)
    y_case2 = np.asarray(fixture["y_case2"], dtype=np.float64).ravel()
    lags_case2 = int(fixture["lags_case2"])
    expected_pacf_case2 = np.asarray(
        fixture["pacf_case2"], dtype=np.float64
    ).ravel()

    pacf_vals, bounds = pacf(y_case2, lags_case2)

    npt.assert_allclose(
        pacf_vals, expected_pacf_case2, atol=ATOL, rtol=RTOL,
        err_msg="PACF parity failed for cumsum series (200 points, 15 lags)",
    )

    # Verify bounds match expected formula for cumsum series
    expected_bound = 1.96 / np.sqrt(len(y_case2))
    npt.assert_allclose(
        bounds, expected_bound, atol=ATOL,
        err_msg="Bounds mismatch for cumsum series",
    )
