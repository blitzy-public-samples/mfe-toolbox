"""
Pytest tests for ``mfe_toolbox.timeseries.armaxfilter_core``.

Tests the inner-loop forward recursion for ARMAX estimation, verifying:
- Output structure (tuple of errors vector and SSE scalar)
- Output shapes and types
- ARX-only recursion (no MA terms)
- AR(1) recursive subtraction
- MA(1) error recursion
- Numerical stability clamping when |error| > 100000 * maxe
- SSE computation correctness
- Zero-initialization before maxq
- ARMA(1,1) combined recursion
- Numerical parity against MATLAB/Octave reference fixtures (±1e-6)

References
----------
- Source: timeseries/armaxfilter_core.m (MFE Toolbox Version 4.0)
- Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.armaxfilter_core import armaxfilter_core

# Tolerance constants per AAP Section 0.7.1
ATOL = 1e-6
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Test 1: Return structure
# ---------------------------------------------------------------------------
def test_armaxfilter_core_returns_two():
    """Verify armaxfilter_core returns a 2-tuple (errors, E)."""
    tau = 20
    regressand = np.ones(tau)
    regressors = np.zeros((tau, 1))
    arx_parameters = np.array([0.5])
    ma_parameters = np.array([])
    q = np.array([], dtype=np.int64)
    K = 1
    maxq = 0
    maxe = 10.0

    result = armaxfilter_core(
        regressand, regressors, arx_parameters, ma_parameters,
        q, K, tau, maxq, maxe,
    )

    # Must return exactly a 2-element tuple
    assert isinstance(result, tuple), "Return value must be a tuple"
    assert len(result) == 2, "Return tuple must have exactly 2 elements"

    errors, E = result
    assert isinstance(errors, np.ndarray), "First element (errors) must be numpy.ndarray"
    assert np.isscalar(E) or isinstance(E, (int, float, np.floating)), (
        "Second element (E) must be a scalar"
    )


# ---------------------------------------------------------------------------
# Test 2: Output shapes
# ---------------------------------------------------------------------------
def test_armaxfilter_core_output_shapes():
    """Verify errors has shape (tau,) and E is a scalar float."""
    tau = 50
    rng = np.random.default_rng(123)
    regressand = rng.standard_normal(tau)
    K = 2
    regressors = rng.standard_normal((tau, K))
    arx_parameters = np.array([0.3, -0.1])
    ma_parameters = np.array([0.2])
    q = np.array([1])
    maxq = 1
    maxe = 5.0

    errors, E = armaxfilter_core(
        regressand, regressors, arx_parameters, ma_parameters,
        q, K, tau, maxq, maxe,
    )

    assert errors.shape == (tau,), (
        f"errors shape should be ({tau},), got {errors.shape}"
    )
    assert errors.ndim == 1, "errors must be 1-dimensional"
    assert np.isscalar(E) or isinstance(E, (float, np.floating)), (
        f"E should be scalar, got type {type(E)}"
    )


# ---------------------------------------------------------------------------
# Test 3: No MA terms (pure ARX)
# ---------------------------------------------------------------------------
def test_armaxfilter_core_no_ma():
    """No MA terms: errors(t) = regressand(t) - regressors(t,:) @ arx_params."""
    tau = 30
    rng = np.random.default_rng(42)
    regressand = rng.standard_normal(tau)
    K = 3
    regressors = rng.standard_normal((tau, K))
    arx_parameters = np.array([0.5, -0.3, 0.1])
    ma_parameters = np.array([])
    q = np.array([], dtype=np.int64)
    maxq = 0
    maxe = 1000.0  # Large enough to avoid clamping

    errors, E = armaxfilter_core(
        regressand, regressors, arx_parameters, ma_parameters,
        q, K, tau, maxq, maxe,
    )

    # Compute expected errors manually: no MA means a simple linear residual
    expected_errors = regressand - regressors @ arx_parameters
    npt.assert_allclose(
        errors, expected_errors, atol=ATOL, rtol=RTOL,
        err_msg="No-MA errors should equal regressand - regressors @ arx_params",
    )

    # E = sum of all squared errors (maxq=0 ⇒ from index 0)
    expected_E = np.dot(expected_errors, expected_errors)
    npt.assert_allclose(
        E, expected_E, atol=ATOL, rtol=RTOL,
        err_msg="E should be SSE of all errors when maxq=0",
    )


# ---------------------------------------------------------------------------
# Test 4: AR(1) — verify recursive subtraction
# ---------------------------------------------------------------------------
def test_armaxfilter_core_ar_only():
    """AR(1): verify recursive subtraction with lagged regressor."""
    tau = 20
    rng = np.random.default_rng(99)
    y = rng.standard_normal(tau)

    # Build AR(1) regressor: column of lagged y values
    X = np.zeros((tau, 1))
    for t in range(1, tau):
        X[t, 0] = y[t - 1]

    phi = 0.7
    arx_parameters = np.array([phi])
    ma_parameters = np.array([])
    q = np.array([], dtype=np.int64)
    K = 1
    maxq = 1
    maxe = 100.0

    errors, E = armaxfilter_core(
        y, X, arx_parameters, ma_parameters, q, K, tau, maxq, maxe,
    )

    # Manually compute expected: errors[t] = y[t] - phi * y[t-1] for t >= maxq
    expected = np.zeros(tau)
    for t in range(maxq, tau):
        expected[t] = y[t] - phi * X[t, 0]

    npt.assert_allclose(
        errors, expected, atol=ATOL, rtol=RTOL,
        err_msg="AR(1) recursion should subtract lagged regressor term",
    )


# ---------------------------------------------------------------------------
# Test 5: MA(1) recursion
# ---------------------------------------------------------------------------
def test_armaxfilter_core_ma_recursion():
    """MA(1): errors(t) -= theta * errors(t-1) recursion with no regressors."""
    tau = 30
    rng = np.random.default_rng(77)
    regressand = rng.standard_normal(tau)
    regressors = np.zeros((tau, 0))
    arx_parameters = np.array([])
    theta = 0.4
    ma_parameters = np.array([theta])
    q = np.array([1])
    K = 0
    maxq = 1
    maxe = 1000.0  # Large enough to avoid clamping

    errors, E = armaxfilter_core(
        regressand, regressors, arx_parameters, ma_parameters,
        q, K, tau, maxq, maxe,
    )

    # Manually compute: errors[t] = regressand[t] - theta * errors[t-1]
    expected = np.zeros(tau)
    for t in range(maxq, tau):
        expected[t] = regressand[t] - theta * expected[t - 1]

    npt.assert_allclose(
        errors, expected, atol=ATOL, rtol=RTOL,
        err_msg="MA(1) recursion should subtract theta * errors(t-1)",
    )


# ---------------------------------------------------------------------------
# Test 6: Error clamping
# ---------------------------------------------------------------------------
def test_armaxfilter_core_error_clamping():
    """When error exceeds 100000*maxe, it's clamped to maxe*sign(error)."""
    tau = 10
    regressand = np.zeros(tau)
    regressand[5] = 1e12  # Huge positive value to trigger clamping
    regressors = np.zeros((tau, 0))
    arx_parameters = np.array([])
    ma_parameters = np.array([])
    q = np.array([], dtype=np.int64)
    K = 0
    maxq = 0
    maxe = 1.0

    errors, E = armaxfilter_core(
        regressand, regressors, arx_parameters, ma_parameters,
        q, K, tau, maxq, maxe,
    )

    # Error at position 5 should be clamped: 1e12 > 100000*1.0 = 100000
    # so errors[5] = maxe * sign(1e12) = 1.0
    assert abs(errors[5]) <= maxe + ATOL, (
        f"Clamped error should be <= maxe={maxe}, got {errors[5]}"
    )
    npt.assert_allclose(
        errors[5], maxe, atol=ATOL,
        err_msg="Positive clamped error should equal maxe",
    )


def test_armaxfilter_core_negative_clamping():
    """Verify negative clamping: huge negative error is clamped to -maxe."""
    tau = 10
    regressand = np.zeros(tau)
    regressand[3] = -5e11  # Huge negative value
    regressors = np.zeros((tau, 0))
    arx_parameters = np.array([])
    ma_parameters = np.array([])
    q = np.array([], dtype=np.int64)
    K = 0
    maxq = 0
    maxe = 2.0

    errors, E = armaxfilter_core(
        regressand, regressors, arx_parameters, ma_parameters,
        q, K, tau, maxq, maxe,
    )

    # Error at position 3 should be clamped to -maxe = -2.0
    npt.assert_allclose(
        errors[3], -maxe, atol=ATOL,
        err_msg="Negative clamped error should equal -maxe",
    )


def test_armaxfilter_core_no_clamping_within_threshold():
    """Verify errors within 100000*maxe are NOT clamped."""
    tau = 10
    maxe = 5.0
    threshold = 100000.0 * maxe  # 500000
    regressand = np.zeros(tau)
    regressand[4] = threshold * 0.99  # Just below threshold
    regressors = np.zeros((tau, 0))

    errors, E = armaxfilter_core(
        regressand, regressors, np.array([]), np.array([]),
        np.array([], dtype=np.int64), 0, tau, 0, maxe,
    )

    # Error should NOT be clamped — should equal the original regressand value
    npt.assert_allclose(
        errors[4], threshold * 0.99, atol=ATOL, rtol=RTOL,
        err_msg="Error within threshold should not be clamped",
    )


# ---------------------------------------------------------------------------
# Test 7: E is SSE
# ---------------------------------------------------------------------------
def test_armaxfilter_core_E_is_sse():
    """E equals sum of squared errors from errors[maxq:tau]."""
    tau = 40
    rng = np.random.default_rng(55)
    regressand = rng.standard_normal(tau)
    K = 2
    regressors = rng.standard_normal((tau, K))
    arx_parameters = np.array([0.3, -0.2])
    ma_parameters = np.array([0.1])
    q = np.array([1])
    maxq = 1
    maxe = 50.0

    errors, E = armaxfilter_core(
        regressand, regressors, arx_parameters, ma_parameters,
        q, K, tau, maxq, maxe,
    )

    # Manually compute SSE: E = dot(errors[maxq:tau], errors[maxq:tau])
    expected_E = np.sum(errors[maxq:tau] ** 2)
    npt.assert_allclose(
        E, expected_E, atol=ATOL, rtol=RTOL,
        err_msg="E should equal sum of squared errors from maxq to tau",
    )


# ---------------------------------------------------------------------------
# Test 8: Zeros before maxq
# ---------------------------------------------------------------------------
def test_armaxfilter_core_zeros_before_maxq():
    """Errors before index maxq remain zero (they are never computed)."""
    tau = 30
    rng = np.random.default_rng(66)
    regressand = rng.standard_normal(tau) + 5.0  # Non-zero data
    regressors = rng.standard_normal((tau, 1))
    arx_parameters = np.array([0.5])
    ma_parameters = np.array([0.3, -0.1])
    q = np.array([1, 2])
    K = 1
    maxq = 3  # First 3 positions should remain zero
    maxe = 100.0

    errors, E = armaxfilter_core(
        regressand, regressors, arx_parameters, ma_parameters,
        q, K, tau, maxq, maxe,
    )

    # errors[0:maxq] should all be exactly zero
    npt.assert_array_equal(
        errors[:maxq], np.zeros(maxq),
        err_msg="Errors before maxq must be exactly zero",
    )

    # Some errors after maxq should be non-zero with non-zero inputs
    assert np.any(errors[maxq:] != 0.0), (
        "Some errors after maxq should be non-zero with non-zero inputs"
    )


# ---------------------------------------------------------------------------
# Test 9: ARMA(1,1) combined recursion
# ---------------------------------------------------------------------------
def test_armaxfilter_core_arma11():
    """ARMA(1,1) recursion correctness: combined AR subtraction and MA feedback."""
    tau = 25
    rng = np.random.default_rng(88)
    y = rng.standard_normal(tau)

    # Build AR(1) regressor: lagged y values in column 0
    X = np.zeros((tau, 1))
    for t in range(1, tau):
        X[t, 0] = y[t - 1]

    phi = 0.6    # AR(1) coefficient
    theta = 0.3  # MA(1) coefficient
    arx_parameters = np.array([phi])
    ma_parameters = np.array([theta])
    q = np.array([1])
    K = 1
    maxq = 1
    maxe = 100.0

    errors, E = armaxfilter_core(
        y, X, arx_parameters, ma_parameters, q, K, tau, maxq, maxe,
    )

    # Manually compute ARMA(1,1) recursion
    expected = np.zeros(tau)
    for t in range(maxq, tau):
        expected[t] = y[t] - phi * X[t, 0] - theta * expected[t - 1]

    npt.assert_allclose(
        errors, expected, atol=ATOL, rtol=RTOL,
        err_msg="ARMA(1,1) recursion should combine AR and MA terms",
    )

    # Also verify E matches
    expected_E = np.dot(expected[maxq:tau], expected[maxq:tau])
    npt.assert_allclose(
        E, expected_E, atol=ATOL, rtol=RTOL,
        err_msg="ARMA(1,1) E should match manual SSE computation",
    )


# ---------------------------------------------------------------------------
# Test 10: Fixture parity against MATLAB/Octave reference outputs
# ---------------------------------------------------------------------------
@pytest.fixture
def fixture_data(timeseries_fixture_dir):
    """Load the armaxfilter_core fixture data dictionary.

    The fixture contains pre-computed MATLAB/Octave reference outputs for
    three scenarios (arma11, arma22, ma_only) plus metadata.
    """
    fixture_path = Path(timeseries_fixture_dir) / "armaxfilter_core.npy"
    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found: {fixture_path}")
    data = np.load(fixture_path, allow_pickle=True)
    # Handle 0-d array wrapping a dict (numpy save/load convention)
    if data.ndim == 0:
        return data.item()
    return data


@pytest.mark.parametrize("scenario", ["arma11", "arma22", "ma_only"])
def test_armaxfilter_core_parity(fixture_data, scenario):
    """Fixture comparison: verify numerical parity with MATLAB/Octave output.

    Per AAP Section 0.7.1, all migrated functions must pass
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
    against MATLAB-generated fixtures.
    """
    if scenario not in fixture_data:
        pytest.skip(f"Scenario '{scenario}' not found in fixture data")

    case = fixture_data[scenario]

    # Extract inputs — fixture uses MATLAB-style names (no underscores)
    regressand = case["regressand"]
    regressors = case["regressors"]
    arx_parameters = case["arxparameters"]
    ma_parameters = case["maparameters"]
    q_lags = case["q"]
    tau = int(case["tau"])
    K = int(case["K"])
    maxq = int(case["maxq"])
    maxe = float(case["maxe"])

    # Expected outputs from MATLAB/Octave
    expected_errors = case["errors"]
    expected_E = float(case["E"])

    # Run Python implementation
    errors, E = armaxfilter_core(
        regressand, regressors, arx_parameters, ma_parameters,
        q_lags, K, tau, maxq, maxe,
    )

    # Verify error vector parity within AAP-mandated tolerance
    npt.assert_allclose(
        errors, expected_errors, atol=ATOL, rtol=RTOL,
        err_msg=f"Errors parity failed for scenario '{scenario}'",
    )

    # Verify SSE parity
    npt.assert_allclose(
        E, expected_E, atol=ATOL, rtol=RTOL,
        err_msg=f"SSE parity failed for scenario '{scenario}'",
    )
