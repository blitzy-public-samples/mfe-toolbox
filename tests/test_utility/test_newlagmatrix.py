"""
Pytest tests for mfe_toolbox.utility.newlagmatrix.

Validates the lag matrix construction function migrated from
utility/newlagmatrix.m (Kevin Sheppard, Revision 5, Date: 12/1/2005).

Tests cover:
- Basic shape and output verification
- Known-value regression checks
- Constant column inclusion/exclusion
- Output shape formula: (T - nlags) × (K * nlags + c)
- Multi-column (K > 1) lag matrix construction
- Single lag (nlags=1) and larger lag (nlags=5) cases
- Manual lag matrix value comparison
- 1-D vector input handling
- Error conditions: non-positive nlags, nlags too large
- MATLAB fixture parity (atol=1e-6, rtol=1e-4) per AAP Section 0.7.1

References:
    Source MATLAB: utility/newlagmatrix.m
    Python target: mfe_toolbox/utility/newlagmatrix.py
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.newlagmatrix import newlagmatrix
from tests.conftest import ATOL, RTOL, load_fixture_npy, assert_allclose


# ---------------------------------------------------------------------------
# Test 1: Basic shape and output type
# ---------------------------------------------------------------------------
def test_newlagmatrix_basic():
    """T=10, K=1, nlags=2, c=0 → y_trimmed (8×1), x (8×2).

    Ref: newlagmatrix.m — basic usage: [y,x]=newlagmatrix(data,2,0)
    Verifies that the output shapes follow (T-nlags) rows with
    nlags columns per input variable and that results are numpy arrays.
    """
    data = np.arange(1.0, 11.0)  # T=10, K=1
    y_trimmed, x = newlagmatrix(data, 2, 0)

    # Shape checks: (T-nlags) × K for y, (T-nlags) × (K*nlags) for x
    assert isinstance(y_trimmed, np.ndarray), "y_trimmed must be ndarray"
    assert isinstance(x, np.ndarray), "x must be ndarray"
    assert y_trimmed.shape == (8, 1), (
        f"Expected y_trimmed shape (8, 1), got {y_trimmed.shape}"
    )
    assert x.shape == (8, 2), (
        f"Expected x shape (8, 2), got {x.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: Known values with simple sequential data
# ---------------------------------------------------------------------------
def test_newlagmatrix_known_values():
    """Use [1,2,3,4,5], nlags=1 → y_trimmed=[2,3,4,5], x=[[1],[2],[3],[4]].

    Ref: newlagmatrix.m:59-66 — lag matrix construction.
    With nlags=1, the most recent lag (t-1) is the only lag column.
    y_trimmed contains the contemporaneous values starting at index nlags.
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_trimmed, x = newlagmatrix(data, 1, 0)

    # Contemporaneous: data[1:] = [2, 3, 4, 5]
    expected_y = np.array([[2.0], [3.0], [4.0], [5.0]])
    # Lag-1: data[0:4] = [1, 2, 3, 4]
    expected_x = np.array([[1.0], [2.0], [3.0], [4.0]])

    npt.assert_allclose(y_trimmed, expected_y, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(x, expected_x, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 3: Constant column included (c=1)
# ---------------------------------------------------------------------------
def test_newlagmatrix_with_constant():
    """c=1 → first column of x is all ones.

    Ref: newlagmatrix.m:67-69 — if c==1, x=[ones(size(x,1),1) x]
    Verifies that when include_constant=1, the first column of x is a
    column of ones (intercept/constant term) and the remaining columns
    contain the lag values.
    """
    data = np.arange(1.0, 11.0)  # T=10
    nlags = 2
    y_trimmed, x = newlagmatrix(data, nlags, 1)

    # Output shape: (T-nlags) × (1 + K*nlags) = (8, 3)
    assert x.shape == (8, 3), f"Expected x shape (8, 3), got {x.shape}"

    # First column must be all ones
    expected_ones = np.ones((8, 1))
    npt.assert_allclose(x[:, 0:1], expected_ones, atol=ATOL, rtol=RTOL,
                        err_msg="First column should be all ones when c=1")

    # Remaining columns should be the lag values (same as without constant)
    _, x_no_const = newlagmatrix(data, nlags, 0)
    npt.assert_allclose(x[:, 1:], x_no_const, atol=ATOL, rtol=RTOL,
                        err_msg="Lag columns should match whether constant is present or not")


# ---------------------------------------------------------------------------
# Test 4: No constant column (c=0)
# ---------------------------------------------------------------------------
def test_newlagmatrix_without_constant():
    """c=0 → no constant column prepended, width is K*nlags.

    Ref: newlagmatrix.m — default c=0, no ones column.
    Verifies that when include_constant=0, the output x has exactly
    K*nlags columns (no leading ones column).
    """
    data = np.arange(1.0, 11.0)  # T=10, K=1
    nlags = 3
    y_trimmed, x = newlagmatrix(data, nlags, 0)

    # (T-nlags) × (K*nlags) = (7, 3)
    assert x.shape == (7, 3), f"Expected x shape (7, 3), got {x.shape}"

    # Verify no column is all ones (unless data happens to have ones, but
    # with arange(1..11) and nlags=3, no lag column will be all ones)
    for col_idx in range(x.shape[1]):
        assert not np.allclose(x[:, col_idx], np.ones(7)), (
            f"Column {col_idx} should not be all ones when c=0"
        )


# ---------------------------------------------------------------------------
# Test 5: Output shape formula verification
# ---------------------------------------------------------------------------
def test_newlagmatrix_output_shape():
    """Verify shape (T-nlags) × (K*nlags + c) for multiple configurations.

    Tests the shape formula across several (T, K, nlags, c) combinations
    to ensure the function always produces correctly sized outputs.
    """
    rng = np.random.default_rng(42)

    # Test configurations: (T, K, nlags, c)
    configs = [
        (20, 1, 3, 0),
        (20, 1, 3, 1),
        (50, 1, 5, 0),
        (50, 1, 5, 1),
        (100, 1, 10, 0),
        (100, 1, 10, 1),
        (30, 1, 1, 0),
        (30, 1, 1, 1),
    ]

    for T, K, nlags, c in configs:
        data = rng.standard_normal((T, K))
        y_trimmed, x = newlagmatrix(data, nlags, c)

        expected_y_rows = T - nlags
        expected_x_rows = T - nlags
        expected_x_cols = K * nlags + c

        assert y_trimmed.shape == (expected_y_rows, K), (
            f"Config (T={T}, K={K}, nlags={nlags}, c={c}): "
            f"Expected y shape ({expected_y_rows}, {K}), got {y_trimmed.shape}"
        )
        assert x.shape == (expected_x_rows, expected_x_cols), (
            f"Config (T={T}, K={K}, nlags={nlags}, c={c}): "
            f"Expected x shape ({expected_x_rows}, {expected_x_cols}), got {x.shape}"
        )


# ---------------------------------------------------------------------------
# Test 6: Multi-column input (K=3, nlags=2)
# ---------------------------------------------------------------------------
def test_newlagmatrix_multiple_columns():
    """K=3, nlags=2 → (T-2) × 6 without constant.

    Ref: newlagmatrix.m — for K>1, each lag block is K columns wide.
    Lag ordering: [y_{t-1}^{(1)}, y_{t-1}^{(2)}, y_{t-1}^{(3)},
                   y_{t-2}^{(1)}, y_{t-2}^{(2)}, y_{t-2}^{(3)}]
    """
    rng = np.random.default_rng(99)
    T, K, nlags = 15, 3, 2
    data = rng.standard_normal((T, K))

    y_trimmed, x = newlagmatrix(data, nlags, 0)

    # Shape: (T-nlags) × (K*nlags) = (13, 6)
    assert y_trimmed.shape == (T - nlags, K), (
        f"Expected y_trimmed shape ({T - nlags}, {K}), got {y_trimmed.shape}"
    )
    assert x.shape == (T - nlags, K * nlags), (
        f"Expected x shape ({T - nlags}, {K * nlags}), got {x.shape}"
    )

    # Verify lag ordering: first K columns are lag-1, next K columns are lag-2
    # Lag-1 block: data[nlags-1 : T-1, :] = data[1:14, :]
    expected_lag1 = data[nlags - 1: T - 1, :]  # (13, 3) — lag 1
    expected_lag2 = data[nlags - 2: T - 2, :]  # (13, 3) — lag 2

    npt.assert_allclose(x[:, 0:K], expected_lag1, atol=ATOL, rtol=RTOL,
                        err_msg="First K columns should be lag-1 values")
    npt.assert_allclose(x[:, K:2 * K], expected_lag2, atol=ATOL, rtol=RTOL,
                        err_msg="Next K columns should be lag-2 values")


# ---------------------------------------------------------------------------
# Test 7: Single lag (nlags=1)
# ---------------------------------------------------------------------------
def test_newlagmatrix_nlags_1():
    """Simplest lag case: nlags=1 → one lag column.

    Verifies that with a single lag, x contains exactly one column of
    values shifted by one position from y_trimmed.
    """
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    y_trimmed, x = newlagmatrix(data, 1, 0)

    # y_trimmed: data[1:] = [20, 30, 40, 50, 60]
    expected_y = np.array([[20.0], [30.0], [40.0], [50.0], [60.0]])
    # x (lag-1): data[0:5] = [10, 20, 30, 40, 50]
    expected_x = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])

    assert y_trimmed.shape == (5, 1)
    assert x.shape == (5, 1)
    npt.assert_allclose(y_trimmed, expected_y, atol=ATOL, rtol=RTOL)
    npt.assert_allclose(x, expected_x, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 8: Larger lag order (nlags=5)
# ---------------------------------------------------------------------------
def test_newlagmatrix_nlags_5():
    """Larger lag order: nlags=5 with T=20.

    Verifies correct shape and that each lag column has the expected
    shift pattern for a larger number of lags.
    """
    rng = np.random.default_rng(123)
    T = 20
    data = rng.standard_normal(T)
    nlags = 5

    y_trimmed, x = newlagmatrix(data, nlags, 0)

    # Shape: (T-nlags) × nlags = (15, 5)
    assert y_trimmed.shape == (T - nlags, 1), (
        f"Expected y shape ({T - nlags}, 1), got {y_trimmed.shape}"
    )
    assert x.shape == (T - nlags, nlags), (
        f"Expected x shape ({T - nlags}, {nlags}), got {x.shape}"
    )

    # Verify each lag column i (0-indexed) contains data shifted by (i+1)
    data_2d = data.reshape(-1, 1)
    for i in range(nlags):
        # Lag (i+1): data[nlags-(i+1) : T-(i+1)]
        lag_idx = i + 1
        expected_col = data_2d[nlags - lag_idx: T - lag_idx, 0]
        npt.assert_allclose(
            x[:, i], expected_col, atol=ATOL, rtol=RTOL,
            err_msg=f"Lag column {i} (lag {lag_idx}) values mismatch"
        )


# ---------------------------------------------------------------------------
# Test 9: Manual lag matrix construction comparison
# ---------------------------------------------------------------------------
def test_newlagmatrix_values_match_manual():
    """Manually construct lag matrix and compare with function output.

    Uses np.column_stack to build the expected lag matrix independently
    and verifies element-by-element parity.
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    nlags = 3
    T = len(data)
    data_2d = data.reshape(-1, 1)

    # Manually construct lag matrix: [lag_1, lag_2, lag_3]
    # lag_1 (t-1): data[2:7] = [3, 4, 5, 6, 7]
    # lag_2 (t-2): data[1:6] = [2, 3, 4, 5, 6]
    # lag_3 (t-3): data[0:5] = [1, 2, 3, 4, 5]
    lag1 = data_2d[nlags - 1: T - 1, :]   # [3,4,5,6,7]
    lag2 = data_2d[nlags - 2: T - 2, :]   # [2,3,4,5,6]
    lag3 = data_2d[nlags - 3: T - 3, :]   # [1,2,3,4,5]
    expected_x = np.column_stack([lag1, lag2, lag3])

    # Contemporaneous: data[3:] = [4, 5, 6, 7, 8]
    expected_y = data_2d[nlags:, :]

    y_trimmed, x = newlagmatrix(data, nlags, 0)

    npt.assert_allclose(y_trimmed, expected_y, atol=ATOL, rtol=RTOL,
                        err_msg="y_trimmed does not match manual construction")
    npt.assert_allclose(x, expected_x, atol=ATOL, rtol=RTOL,
                        err_msg="x does not match manually constructed lag matrix")

    # Also test with constant
    y_trimmed_c, x_c = newlagmatrix(data, nlags, 1)
    expected_x_c = np.column_stack([np.ones((5, 1)), expected_x])
    npt.assert_allclose(x_c, expected_x_c, atol=ATOL, rtol=RTOL,
                        err_msg="Constant-prepended x does not match manual construction")


# ---------------------------------------------------------------------------
# Test 10: Vector input handling — 1-D vs 2-D
# ---------------------------------------------------------------------------
def test_newlagmatrix_vector_input():
    """Column vector (T,) vs (T,1) should produce identical results.

    Ref: newlagmatrix.m:43-45 — MATLAB transposes row vectors.
    Python implementation reshapes 1-D to (T, 1) before processing.
    Both 1-D and 2-D column vectors should yield the same output.
    """
    rng = np.random.default_rng(77)
    T = 25
    nlags = 3

    data_1d = rng.standard_normal(T)        # shape (25,)
    data_2d = data_1d.reshape(-1, 1)         # shape (25, 1)

    y_1d, x_1d = newlagmatrix(data_1d, nlags, 1)
    y_2d, x_2d = newlagmatrix(data_2d, nlags, 1)

    # Both should produce identical results
    npt.assert_allclose(y_1d, y_2d, atol=ATOL, rtol=RTOL,
                        err_msg="1-D and 2-D inputs produce different y_trimmed")
    npt.assert_allclose(x_1d, x_2d, atol=ATOL, rtol=RTOL,
                        err_msg="1-D and 2-D inputs produce different x")

    # Verify shape is consistent: y should be (T-nlags, 1)
    assert y_1d.shape == (T - nlags, 1), (
        f"Expected y shape ({T - nlags}, 1), got {y_1d.shape}"
    )
    assert x_1d.shape == (T - nlags, nlags + 1), (
        f"Expected x shape ({T - nlags}, {nlags + 1}), got {x_1d.shape}"
    )


# ---------------------------------------------------------------------------
# Test 11: Non-positive nlags raises ValueError
# ---------------------------------------------------------------------------
def test_newlagmatrix_non_positive_nlags_raises():
    """Negative nlags → ValueError.

    Ref: newlagmatrix.m:51-53 — MATLAB: 'NLAGS must be a positive integer'
    The Python implementation raises ValueError for negative lags.
    Note: nlags=0 is a valid edge case in both MATLAB and Python
    (returns y unchanged, x as empty or ones column).
    """
    data = np.arange(1.0, 11.0)

    # Negative lags must raise ValueError
    with pytest.raises(ValueError):
        newlagmatrix(data, -1, 0)

    with pytest.raises(ValueError):
        newlagmatrix(data, -5, 0)

    with pytest.raises(ValueError):
        newlagmatrix(data, -100, 1)

    # Non-integer lags must raise ValueError
    with pytest.raises(ValueError):
        newlagmatrix(data, 2.5, 0)

    # nlags=0 is valid (edge case) — should NOT raise
    y_zero, x_zero = newlagmatrix(data, 0, 0)
    assert y_zero.shape[0] == 10, "nlags=0 should return all rows"


# ---------------------------------------------------------------------------
# Test 12: nlags too large raises ValueError
# ---------------------------------------------------------------------------
def test_newlagmatrix_nlags_too_large_raises():
    """nlags >= T → ValueError.

    Ref: newlagmatrix.py:119 — 'Number of lags must be less than the
    number of observations'. When nlags equals or exceeds T, there are
    insufficient observations to form the lag matrix.
    """
    data = np.arange(1.0, 6.0)  # T=5

    # nlags == T → ValueError
    with pytest.raises(ValueError):
        newlagmatrix(data, 5, 0)

    # nlags > T → ValueError
    with pytest.raises(ValueError):
        newlagmatrix(data, 10, 0)

    with pytest.raises(ValueError):
        newlagmatrix(data, 6, 1)

    # nlags == T-1 should work (returns 1 row)
    y_edge, x_edge = newlagmatrix(data, 4, 0)
    assert y_edge.shape == (1, 1), (
        f"Expected y shape (1, 1) for nlags=T-1, got {y_edge.shape}"
    )
    assert x_edge.shape == (1, 4), (
        f"Expected x shape (1, 4) for nlags=T-1, got {x_edge.shape}"
    )


# ---------------------------------------------------------------------------
# Test 13: MATLAB fixture parity
# ---------------------------------------------------------------------------
def test_newlagmatrix_fixture_parity(utility_fixture_dir):
    """Compare against MATLAB/Octave generated fixture outputs.

    Loads newlagmatrix.npy which contains multiple test cases generated
    by Octave (seed=42) with known inputs and expected [y, X] outputs.
    Each case is verified with atol=1e-6, rtol=1e-4 per AAP Section 0.7.1.

    Also tests the standalone nlm_out / nlm_T secondary fixtures.
    """
    # Load the main fixture file
    fixture_data = load_fixture_npy(utility_fixture_dir, "newlagmatrix")

    # The fixture is a dict with input arrays and test cases
    if isinstance(fixture_data, np.ndarray) and fixture_data.ndim == 0:
        fixture_dict = fixture_data.item()
    else:
        pytest.skip("Unexpected fixture format for newlagmatrix.npy")

    # Iterate over all test cases in the fixture
    case_keys = sorted([k for k in fixture_dict if k.startswith("case_")])
    assert len(case_keys) > 0, "No test cases found in fixture"

    for case_key in case_keys:
        case = fixture_dict[case_key]
        description = case.get("description", case_key)
        input_key = case["input_key"]
        nlags = int(case["nlags"])
        constant = int(case["constant"])
        expected_y = np.asarray(case["y"], dtype=np.float64)
        expected_x = np.asarray(case["X"], dtype=np.float64)

        # Retrieve the input data for this case
        input_data = np.asarray(fixture_dict[input_key], dtype=np.float64)

        # Run the Python function
        y_trimmed, x = newlagmatrix(input_data, nlags, constant)

        # Handle shape differences between MATLAB (1-D) and Python (2-D) outputs
        y_compare = y_trimmed.squeeze()
        x_compare = x

        # Special case: nlags=0, constant=0 → MATLAB returns empty [] as (0,0)
        # Python returns (T, 0) shaped array. Both are semantically "empty".
        if nlags == 0 and constant == 0:
            assert x.shape[1] == 0, (
                f"Case '{description}': nlags=0, c=0 should have 0 columns, "
                f"got {x.shape[1]}"
            )
            # Compare y only (x is empty in both implementations)
            assert_allclose(
                y_compare, expected_y,
                err_msg=f"Case '{description}': y_trimmed mismatch"
            )
            continue

        # Standard case: compare both y and x
        assert_allclose(
            y_compare, expected_y,
            err_msg=f"Case '{description}': y_trimmed mismatch"
        )

        # Squeeze x for comparison if MATLAB has fewer dimensions
        if expected_x.ndim == 1:
            x_compare = x_compare.squeeze()

        assert_allclose(
            x_compare, expected_x,
            err_msg=f"Case '{description}': X lag matrix mismatch"
        )

    # ----- Secondary fixtures: nlm_out and nlm_T -----
    # These are standalone y/X outputs from a T=1000, nlags=5, constant=1 run
    try:
        nlm_out = load_fixture_npy(utility_fixture_dir, "newlagmatrix_nlm_out")
        nlm_T = load_fixture_npy(utility_fixture_dir, "newlagmatrix_nlm_T")
    except Exception:
        # If secondary fixtures unavailable, skip gracefully
        return

    # Verify shape consistency
    assert nlm_out.shape[0] == nlm_T.shape[0], (
        "nlm_out and nlm_T row counts should match"
    )

    # nlm_T has constant column (first col is ones) + 5 lag columns = 6 cols
    if nlm_T.shape[1] == 6:
        # First column should be all ones (constant)
        npt.assert_allclose(
            nlm_T[:, 0], np.ones(nlm_T.shape[0]),
            atol=ATOL, rtol=RTOL,
            err_msg="Secondary fixture nlm_T first column should be ones"
        )
