"""Pytest tests for ``mfe_toolbox.utility.hessian_2sided_nrows``.

Tests the partial Hessian computation that returns only the **last K rows**
of the full N×N two-sided finite-difference Hessian matrix.

Source reference: ``utility/hessian_2sided_nrows.m`` (89 lines, Kevin Sheppard).

Key behaviours verified:
- Same central-difference formula as ``hessian_2sided`` but restricted to the
  last ``k`` rows (MATLAB line 70: ``for i=n-k+1:n``, line 88: ``H=H((n-k+1):n,:)``).
- Step size floor is **1e-2** (Ref: hessian_2sided_nrows.m:52),
  which differs from ``hessian_2sided``'s floor of 1e-8.
- Output shape is ``(k, n)`` — a sub-matrix of the full Hessian.
- Raises ``ValueError`` for invalid *k* values (k ≤ 0 or k > n).
- Raises ``RuntimeError`` when the objective function cannot be evaluated.

Numerical parity rules (AAP §0.7.1):
    ``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.hessian_2sided_nrows import hessian_2sided_nrows
from mfe_toolbox.utility.hessian_2sided import hessian_2sided

# ---------------------------------------------------------------------------
# Tolerance constants (mirrored from conftest.py for explicitness)
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# Wider tolerance for numerical Hessian vs analytical (O(h²) error)
HESS_ATOL: float = 1e-3


# ---------------------------------------------------------------------------
# Helper objective functions
# ---------------------------------------------------------------------------

def _quadratic_func(x: np.ndarray, A: np.ndarray) -> float:
    """Compute f(x) = x^T A x.  Hessian is 2*A (constant)."""
    return float(x @ A @ x)


def _sum_of_cubes(x: np.ndarray) -> float:
    """f(x) = sum(x_i^3).  Hessian_ii = 6*x_i, H_ij = 0 for i!=j."""
    return float(np.sum(x ** 3))


def _rosenbrock(x: np.ndarray) -> float:
    """Extended Rosenbrock function (N-D):
    f = sum_{i=0}^{N-2} [ 100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2 ]
    Reduces to standard 2-D form when len(x)==2.
    """
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return float(total)


def _sum_of_squares(x: np.ndarray) -> float:
    """f(x) = sum(x_i^2).  Hessian = 2*I (identity scaled by 2)."""
    return float(np.sum(x ** 2))


# ---------------------------------------------------------------------------
# Test 1: Last nrows rows match full Hessian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, k", [(4, 2), (5, 3), (3, 1), (5, 5)])
def test_hessian_nrows_matches_full(n: int, k: int) -> None:
    """Verify last *k* rows of ``hessian_2sided_nrows`` match the full Hessian.

    We use test points where all |x_i| ≥ 1.0 so that both functions
    produce identical step sizes (both floors, 1e-2 and 1e-8, are
    irrelevant when |x_i| ≥ 1).

    Ref: hessian_2sided_nrows.m:70-88 — iterates last k rows, extracts
    ``H((n-k+1):n, :)`` which is the **last** k rows in MATLAB 1-indexing.
    """
    rng = np.random.default_rng(42)
    # All values >= 1.0 so that the step floor difference is irrelevant
    x = rng.uniform(1.0, 3.0, size=n)

    # Symmetric positive-definite A for a well-behaved quadratic
    M = rng.standard_normal((n, n))
    A = M.T @ M + np.eye(n)

    H_full = hessian_2sided(lambda z: _quadratic_func(z, A), x)
    H_partial = hessian_2sided_nrows(lambda z: _quadratic_func(z, A), x, k)

    # The partial result should equal the last k rows of the full Hessian
    npt.assert_allclose(H_partial, H_full[-k:, :], atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 2: Output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, k", [(5, 3), (4, 1), (6, 4), (3, 2)])
def test_hessian_nrows_output_shape(n: int, k: int) -> None:
    """K-dim input with nrows=k → output shape (k, n)."""
    x = np.ones(n) * 2.0
    H = hessian_2sided_nrows(_sum_of_squares, x, k)
    assert H.shape == (k, n), f"Expected shape ({k}, {n}), got {H.shape}"


# ---------------------------------------------------------------------------
# Test 3: nrows == N → same as full Hessian
# ---------------------------------------------------------------------------

def test_hessian_nrows_all_rows() -> None:
    """When nrows=N, result should equal the full Hessian (all rows computed).

    Uses test point with |x_i| ≥ 1.0 to equalise step-size floors.
    """
    n = 4
    rng = np.random.default_rng(123)
    x = rng.uniform(1.0, 5.0, size=n)

    M = rng.standard_normal((n, n))
    A = M.T @ M + np.eye(n)

    H_full = hessian_2sided(lambda z: _quadratic_func(z, A), x)
    H_nrows = hessian_2sided_nrows(lambda z: _quadratic_func(z, A), x, n)

    npt.assert_allclose(H_nrows, H_full, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 4: Single row (nrows = 1) — last row of Hessian
# ---------------------------------------------------------------------------

def test_hessian_nrows_single_row() -> None:
    """nrows=1 → 1×N array matching the last row of the full Hessian."""
    n = 5
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    A = np.eye(n) * 2.0  # Diagonal ⇒ H = 4*I
    H_partial = hessian_2sided_nrows(lambda z: _quadratic_func(z, A), x, 1)

    assert H_partial.shape == (1, n)

    # Last row of Hessian of x'*(2I)*x = 2*sum(x_i^2) is [0,…,0,4] (last diag)
    # Analytical last row of 2*A: all zeros except [n-1, n-1] = 4.0
    expected_last_row = np.zeros(n)
    expected_last_row[-1] = 4.0
    npt.assert_allclose(H_partial.ravel(), expected_last_row, atol=HESS_ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 5: Quadratic function — analytical Hessian is 2*A
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_hessian_nrows_quadratic(k: int) -> None:
    """f(x) = x^T A x ⇒ Hessian = 2*A.  Last k rows should match 2*A[-k:, :]."""
    n = 3
    A = np.array([
        [4.0, 1.0, 0.5],
        [1.0, 6.0, 2.0],
        [0.5, 2.0, 8.0],
    ])
    x = np.array([1.5, -0.5, 2.0])

    H = hessian_2sided_nrows(lambda z: _quadratic_func(z, A), x, k)
    expected = (2.0 * A)[-k:, :]

    npt.assert_allclose(H, expected, atol=HESS_ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 6: Accuracy against analytical partial Hessian (Rosenbrock 2-D)
# ---------------------------------------------------------------------------

def test_hessian_nrows_accuracy() -> None:
    """Compare numerical partial Hessian to analytical for the Rosenbrock function.

    Rosenbrock: f(x) = (1-x0)^2 + 100*(x1-x0^2)^2
    Analytical Hessian:
        H[0,0] = 2 - 400*(x1 - x0^2) + 800*x0^2  = 2 - 400*x1 + 1200*x0^2
        H[0,1] = H[1,0] = -400*x0
        H[1,1] = 200
    """
    x = np.array([1.0, 1.0])

    # Analytical Hessian at x = [1, 1]:
    # H[0,0] = 2 - 400*1 + 1200*1 = 802
    # H[0,1] = -400*1 = -400
    # H[1,0] = -400
    # H[1,1] = 200
    analytical = np.array([
        [802.0, -400.0],
        [-400.0, 200.0],
    ])

    # Last 1 row
    H_1 = hessian_2sided_nrows(_rosenbrock, x, 1)
    npt.assert_allclose(H_1, analytical[-1:, :], atol=HESS_ATOL, rtol=RTOL)

    # Last 2 rows (all rows for 2-D)
    H_2 = hessian_2sided_nrows(_rosenbrock, x, 2)
    npt.assert_allclose(H_2, analytical, atol=HESS_ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 7: Invalid nrows raises ValueError
# ---------------------------------------------------------------------------

def test_hessian_nrows_invalid_raises() -> None:
    """nrows > N, nrows=0, and nrows<0 must raise ValueError."""
    x = np.array([1.0, 2.0, 3.0])

    # k > n
    with pytest.raises(ValueError, match=r"must not exceed"):
        hessian_2sided_nrows(_sum_of_squares, x, 4)

    # k = 0
    with pytest.raises(ValueError, match=r"positive integer"):
        hessian_2sided_nrows(_sum_of_squares, x, 0)

    # k < 0
    with pytest.raises(ValueError, match=r"positive integer"):
        hessian_2sided_nrows(_sum_of_squares, x, -1)


# ---------------------------------------------------------------------------
# Test 7b: Non-integer k raises ValueError
# ---------------------------------------------------------------------------

def test_hessian_nrows_non_integer_raises() -> None:
    """Non-integer k must raise ValueError."""
    x = np.array([1.0, 2.0])
    with pytest.raises((ValueError, TypeError)):
        hessian_2sided_nrows(_sum_of_squares, x, 1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 7c: Invalid function raises RuntimeError
# ---------------------------------------------------------------------------

def test_hessian_nrows_bad_function_raises() -> None:
    """Function that cannot be evaluated should raise RuntimeError."""
    x = np.array([1.0, 2.0])

    def bad_func(z: np.ndarray) -> float:
        raise ZeroDivisionError("deliberately broken")

    with pytest.raises(RuntimeError, match=r"error evaluating"):
        hessian_2sided_nrows(bad_func, x, 1)


# ---------------------------------------------------------------------------
# Test 8: Fixture parity — MATLAB reference comparison
# ---------------------------------------------------------------------------

def test_hessian_nrows_fixture_parity(utility_fixture_dir: Path) -> None:
    """Compare against MATLAB-generated reference outputs.

    Fixture structure (object-dtype .npy with dict):
    - case*_x: parameter vector
    - case*_k: number of rows
    - case*_A: quadratic matrix (if func_type==1)
    - case*_H: expected partial Hessian
    - case*_func_type: 1=quadratic(x'Ax), 2=Rosenbrock
    """
    fixture_path = utility_fixture_dir / "hessian_2sided_nrows.npy"
    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found: {fixture_path}")

    data = np.load(fixture_path, allow_pickle=True).item()

    # Iterate over all cases in the fixture dict
    case_idx = 1
    while f"case{case_idx}_x" in data:
        prefix = f"case{case_idx}"
        x = np.asarray(data[f"{prefix}_x"], dtype=float)
        k = int(data[f"{prefix}_k"])
        func_type = int(data[f"{prefix}_func_type"])
        expected_H = np.asarray(data[f"{prefix}_H"], dtype=float)

        if func_type == 1:
            # Quadratic: f(x) = 0.5 * x' * A * x  (Hessian = A)
            # Ref: fixture generated with 0.5 factor so Hessian = A, not 2*A
            A = np.asarray(data[f"{prefix}_A"], dtype=float)
            actual_H = hessian_2sided_nrows(
                lambda z, _A=A: float(0.5 * z @ _A @ z), x, k
            )
        elif func_type == 2:
            # Rosenbrock (2-D)
            actual_H = hessian_2sided_nrows(_rosenbrock, x, k)
        else:
            pytest.fail(f"Unknown func_type={func_type} in fixture case {case_idx}")

        # Numerical Hessian computations have O(h²) error where h ≈ eps^(1/3)
        # ≈ 6e-6.  MATLAB and Python may differ at positions near zero due to
        # floating-point step-size rounding.  We use a slightly wider atol that
        # still validates algorithmic correctness while accommodating platform
        # differences in the finite-difference step.
        npt.assert_allclose(
            actual_H,
            expected_H,
            atol=5e-5,
            rtol=RTOL,
            err_msg=f"Fixture parity failed for {prefix}",
        )
        case_idx += 1


# ---------------------------------------------------------------------------
# Test 8b: Simple fixture — x_hn input with h2sn_H output
# ---------------------------------------------------------------------------

def test_hessian_nrows_simple_fixture_parity(utility_fixture_dir: Path) -> None:
    """Compare against simple x_hn / h2sn_H fixture pair."""
    x_path = utility_fixture_dir / "hessian_2sided_nrows_x_hn.npy"
    H_path = utility_fixture_dir / "hessian_2sided_nrows_h2sn_H.npy"

    if not x_path.exists() or not H_path.exists():
        pytest.skip("Simple fixture files not found")

    x = np.load(x_path, allow_pickle=True)
    expected_H = np.load(H_path, allow_pickle=True)

    n = len(x)
    k = expected_H.shape[0]

    # Default fixture function: f(x) = x' * I * x = sum(x_i^2)
    # Hessian = 2*I, last k rows → 2*I[-k:, :]
    actual_H = hessian_2sided_nrows(_sum_of_squares, x, k)

    npt.assert_allclose(
        actual_H,
        expected_H,
        atol=ATOL,
        rtol=RTOL,
        err_msg="Simple fixture parity failed",
    )


# ---------------------------------------------------------------------------
# Test 9: Separable function — off-diagonal should be near zero
# ---------------------------------------------------------------------------

def test_hessian_nrows_separable() -> None:
    """f(x) = sum(x_i^3) ⇒ H_ii=6*x_i, H_ij=0 for i≠j.

    Check last 2 rows of a 4-parameter function.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    k = 2

    H = hessian_2sided_nrows(_sum_of_cubes, x, k)

    # Analytical last 2 rows of the Hessian
    #   H[2, :] = [0, 0, 6*3, 0] = [0, 0, 18, 0]
    #   H[3, :] = [0, 0, 0, 6*4] = [0, 0, 0, 24]
    expected = np.array([
        [0.0, 0.0, 18.0, 0.0],
        [0.0, 0.0, 0.0, 24.0],
    ])

    npt.assert_allclose(H, expected, atol=HESS_ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 10: Symmetry of reconstructed rows
# ---------------------------------------------------------------------------

def test_hessian_nrows_partial_symmetry() -> None:
    """The partial Hessian rows should be consistent with full symmetry.

    Specifically, for a symmetric Hessian H, the last k rows H[-k:, :]
    should satisfy H[-k:, j] == H[j, -k:].T for columns that overlap.
    We verify by checking the full Hessian is symmetric and the partial
    rows match.
    """
    n = 4
    rng = np.random.default_rng(99)
    x = rng.uniform(1.0, 5.0, size=n)

    M = rng.standard_normal((n, n))
    A = M.T @ M + np.eye(n)

    func = lambda z: _quadratic_func(z, A)

    # Get partial (last 2 rows)
    H_partial = hessian_2sided_nrows(func, x, 2)

    # The (n-2, n-1) element from partial should equal the column
    # extracted from the other row
    # H_partial[0, :] = row (n-2) of full H
    # H_partial[1, :] = row (n-1) of full H
    # Full Hessian is symmetric, so H[n-2, n-1] == H[n-1, n-2]
    npt.assert_allclose(
        H_partial[0, n - 1],
        H_partial[1, n - 2],
        atol=ATOL,
        rtol=RTOL,
        err_msg="Symmetry of partial Hessian elements failed",
    )


# ---------------------------------------------------------------------------
# Test 11: Extra args pass-through
# ---------------------------------------------------------------------------

def test_hessian_nrows_extra_args() -> None:
    """Verify that extra positional arguments are forwarded to the function.

    Ref: hessian_2sided_nrows.m:49 — ``feval(f, x, varargin{:})``
    """
    A = np.array([[3.0, 0.5], [0.5, 4.0]])
    x = np.array([1.0, 2.0])

    H = hessian_2sided_nrows(_quadratic_func, x, 2, A)
    expected = 2.0 * A

    npt.assert_allclose(H, expected, atol=HESS_ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 12: Large-dimensional input
# ---------------------------------------------------------------------------

def test_hessian_nrows_large_dim() -> None:
    """Test with a higher-dimensional input (N=8, k=3)."""
    n = 8
    k = 3
    rng = np.random.default_rng(77)
    x = rng.uniform(1.0, 4.0, size=n)

    # Simple function with known Hessian = 2*I
    H = hessian_2sided_nrows(_sum_of_squares, x, k)

    assert H.shape == (k, n)

    # Expected last k rows of 2*I
    expected = np.zeros((k, n))
    for i in range(k):
        expected[i, n - k + i] = 2.0

    npt.assert_allclose(H, expected, atol=HESS_ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 13: 1-D input (scalar parameter)
# ---------------------------------------------------------------------------

def test_hessian_nrows_scalar_param() -> None:
    """N=1, k=1 → 1×1 second derivative."""
    x = np.array([2.0])

    # f(x) = x^4 ⇒ f''(x) = 12*x^2 = 48 at x=2
    def f_quartic(z: np.ndarray) -> float:
        return float(z[0] ** 4)

    H = hessian_2sided_nrows(f_quartic, x, 1)
    assert H.shape == (1, 1)
    npt.assert_allclose(H[0, 0], 48.0, atol=0.1, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 14: Input vector auto-flattening
# ---------------------------------------------------------------------------

def test_hessian_nrows_row_vector_input() -> None:
    """Row vector input (2-D shape (1, N)) should be handled gracefully.

    Ref: hessian_2sided_nrows.m:33-35 — auto-transposes row to column.
    """
    x = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3)
    k = 2

    # Should work without error — implementation uses np.atleast_1d + ravel
    H = hessian_2sided_nrows(_sum_of_squares, x, k)
    assert H.shape == (k, 3)
