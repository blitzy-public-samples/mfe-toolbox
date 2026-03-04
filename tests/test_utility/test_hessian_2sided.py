"""Pytest tests for ``mfe_toolbox.utility.hessian_2sided``.

Tests the two-sided finite difference Hessian matrix computation against
analytical Hessians, known mathematical properties (symmetry, positive
semi-definiteness for convex functions), output shape contracts, step-size
formula verification, and MATLAB-generated fixture files.

Source reference: ``utility/hessian_2sided.m`` (82 lines, Kevin Sheppard).

Key behaviours verified:
- Step size: ``h = eps^(1/3) * max(|x|, 1e-8)``  — min step is 1e-8
  (different from gradient_2sided's 1e-2!).
- Diagonal: second-order central difference via double-step evaluations.
- Off-diagonal: cross-partial via forward/backward perturbation pairs.
- Returns K×K symmetric Hessian matrix.
- Symmetrised: ``H = (H + H') / 2``.

Numerical parity rules (AAP §0.7.1):
    ``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
    for MATLAB fixture parity.
    ``atol=1e-4`` for accuracy tests against analytical Hessians (O(h²) error).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.hessian_2sided import hessian_2sided
from tests.conftest import load_fixture_npy


# ---------------------------------------------------------------------------
# Tolerance constants (per AAP §0.7.1 and agent prompt)
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# Wider tolerance for numerical Hessian vs analytical (O(h²) error).
# The numerical Hessian error is O(h²) where h ≈ eps^(1/3) ≈ 6e-6, but for
# functions with large higher-order derivatives (e.g. Rosenbrock at non-optimum)
# the absolute error can exceed 1e-4.  Using both atol and rtol ensures that
# small-valued and large-valued Hessian entries are compared fairly.
HESS_ATOL: float = 1e-4
HESS_RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Helper objective functions
# ---------------------------------------------------------------------------

def _quadratic_xAx(x: np.ndarray, A: np.ndarray) -> float:
    """Compute f(x) = x^T A x.  Hessian is 2*A (constant)."""
    return float(x @ A @ x)


def _quadratic_half(x: np.ndarray, A: np.ndarray) -> float:
    """Compute f(x) = 0.5 * x^T A x.  Hessian is A (constant)."""
    return float(0.5 * x @ A @ x)


def _quadratic_half_with_linear(
    x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float
) -> float:
    """Compute f(x) = 0.5 * x^T A x + b^T x + c.  Hessian is A."""
    return float(0.5 * x @ A @ x + np.dot(b, x) + c)


def _separable_cubic(x: np.ndarray) -> float:
    """Separable cubic: f(x) = sum(x_i^3).  Hessian: diag(6*x_i)."""
    return float(np.sum(x ** 3))


def _rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function (generalised, 2-D default).

    f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    2-D analytical Hessian at (a, b):
        f_xx = 1200*a^2 - 400*b + 2
        f_xy = -400*a
        f_yy = 200
    """
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return float(total)


def _linear(x: np.ndarray, c: np.ndarray) -> float:
    """Linear function: f(x) = c^T x.  Hessian is zero matrix."""
    return float(np.dot(c, x))


def _sum_exp(x: np.ndarray) -> float:
    """Exponential sum: f(x) = sum(exp(x_i)).  Hessian: diag(exp(x_i))."""
    return float(np.sum(np.exp(x)))


def _sum_squares(x: np.ndarray) -> float:
    """Sum of squares: f(x) = sum(x_i^2).  Hessian: 2*I."""
    return float(np.sum(x ** 2))


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spd_matrix() -> np.ndarray:
    """Return a 3×3 symmetric positive-definite matrix for test use."""
    A = np.array([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 0.7],
        [0.5, 0.7, 2.0],
    ])
    return A


# ---------------------------------------------------------------------------
# Test 1: Quadratic Function — f(x) = x'Ax → Hessian = 2A
# ---------------------------------------------------------------------------

def test_hessian_quadratic() -> None:
    """f(x) = x'Ax → Hessian = 2A (constant, known analytically).

    Uses a 2×2 symmetric positive-definite matrix A.  The Hessian of x'Ax
    is exactly 2A at any point, so the numerical Hessian should match to
    within O(h²) accuracy.
    """
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    x0 = np.array([1.0, 2.0])

    H = hessian_2sided(lambda z: _quadratic_xAx(z, A), x0)

    expected = 2.0 * A
    npt.assert_allclose(
        H, expected, atol=HESS_ATOL, rtol=HESS_RTOL,
        err_msg="Hessian of x'Ax should equal 2A",
    )


# ---------------------------------------------------------------------------
# Test 2: Separable Function — f(x) = sum(x_i^3) → diag(6*x_i)
# ---------------------------------------------------------------------------

def test_hessian_separable() -> None:
    """f(x) = sum(x_i^3) → H_ii = 6*x_i, H_ij = 0 for i ≠ j.

    Separable functions have purely diagonal Hessians.  The analytical
    second derivative of x^3 is 6x, so the Hessian should be diag(6*x).
    """
    x0 = np.array([1.0, 2.0, -1.5])
    H = hessian_2sided(_separable_cubic, x0)

    expected = np.diag(6.0 * x0)
    npt.assert_allclose(
        H, expected, atol=HESS_ATOL, rtol=HESS_RTOL,
        err_msg="Hessian of sum(x^3) should equal diag(6*x)",
    )


# ---------------------------------------------------------------------------
# Test 3: Rosenbrock Function — known analytical Hessian
# ---------------------------------------------------------------------------

def test_hessian_rosenbrock() -> None:
    """Rosenbrock Hessian at point (1.5, 1.5) matches analytical formula.

    Analytical 2-D Rosenbrock Hessian at (a, b):
        [[1200*a² - 400*b + 2,  -400*a],
         [-400*a,                 200  ]]

    At (1.5, 1.5):
        f_xx = 1200*2.25 - 400*1.5 + 2 = 2102
        f_xy = -400*1.5 = -600
        f_yy = 200
    """
    x0 = np.array([1.5, 1.5])
    H = hessian_2sided(_rosenbrock, x0)

    a, b = x0
    expected = np.array([
        [1200.0 * a ** 2 - 400.0 * b + 2.0, -400.0 * a],
        [-400.0 * a, 200.0],
    ])
    npt.assert_allclose(
        H, expected, atol=HESS_ATOL, rtol=HESS_RTOL,
        err_msg="Rosenbrock Hessian mismatch at (1.5, 1.5)",
    )


# ---------------------------------------------------------------------------
# Test 4: Symmetry — H always equals H.T
# ---------------------------------------------------------------------------

def test_hessian_symmetric(rng: np.random.Generator) -> None:
    """Output Hessian is always symmetric (H == H.T) for any function.

    The implementation symmetrises by construction:
        ``H = (H + H.T) / 2``  (Ref: hessian_2sided.m:79)

    Uses a non-separable function with random evaluation point to ensure
    off-diagonal entries are non-trivial.
    """
    # Random SPD matrix for non-separable coupling
    raw = rng.standard_normal((4, 4))
    A = raw.T @ raw + np.eye(4)  # guaranteed SPD
    A = (A + A.T) / 2.0  # ensure exact symmetry of input A

    x0 = rng.standard_normal(4)
    H = hessian_2sided(lambda z: _quadratic_xAx(z, A), x0)

    # Exact symmetry: H[i,j] == H[j,i] (not just approximate)
    npt.assert_array_equal(
        H, H.T,
    )


# ---------------------------------------------------------------------------
# Test 5: Output Shape — K-dim input → K×K Hessian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [1, 2, 3, 5, 10])
def test_hessian_output_shape(K: int) -> None:
    """K-dim input produces K×K Hessian matrix.

    Parametrised over K = 1, 2, 3, 5, 10 to cover scalar, small, and
    medium-dimensional cases.
    """
    x0 = np.ones(K)
    H = hessian_2sided(_sum_squares, x0)

    assert H.shape == (K, K), (
        f"Expected Hessian shape ({K}, {K}), got {H.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: Scalar Input — K=1 → 1×1 second derivative
# ---------------------------------------------------------------------------

def test_hessian_scalar_input() -> None:
    """K=1 → 1×1 matrix containing the second derivative.

    f(x) = x^3 → f''(x) = 6x.  At x=2: f''(2) = 12.
    """
    def f_cubic(x: np.ndarray) -> float:
        return float(x[0] ** 3)

    x0 = np.array([2.0])
    H = hessian_2sided(f_cubic, x0)

    assert H.shape == (1, 1), f"Expected (1, 1), got {H.shape}"
    # f''(2) = 6*2 = 12
    assert H[0, 0] == pytest.approx(12.0, abs=HESS_ATOL), (
        f"f''(2) for x^3 should be 12.0, got {H[0, 0]}"
    )


# ---------------------------------------------------------------------------
# Test 7: Step Size Formula — h = eps^(1/3) * max(|x|, 1e-8)
# ---------------------------------------------------------------------------

def test_hessian_step_size() -> None:
    """Verify step size formula: h = eps^(1/3) * max(|x|, 1e-8).

    At x=0, the minimum step base is 1e-8.  This is different from
    gradient_2sided which uses a minimum step base of 1e-2.

    Verification strategy: use a recording wrapper to capture the exact
    perturbation sizes passed to the objective function, then compare
    against the expected step formula.
    """
    eps_val = np.finfo(float).eps

    # ---- Case 1: x = 0 → h = eps^(1/3) * 1e-8 ----
    x_zero = np.array([0.0])
    expected_h_zero = eps_val ** (1.0 / 3.0) * np.maximum(np.abs(0.0), 1e-8)
    # Ref: hessian_2sided.m:50-51 — precision trick
    xh_zero = x_zero[0] + expected_h_zero
    expected_h_zero_adj = xh_zero - x_zero[0]

    evals_zero: list[np.ndarray] = []

    def recording_zero(x: np.ndarray) -> float:
        evals_zero.append(x.copy())
        return float(np.sum(x ** 2))

    hessian_2sided(recording_zero, x_zero)

    # Extract perturbation magnitudes from x=0
    perturbs_zero = sorted({
        round(float(np.abs(ev[0])), 25)
        for ev in evals_zero
        if np.abs(ev[0]) > 0
    })

    assert len(perturbs_zero) > 0, "Expected non-zero perturbations at x=0"
    # The single-step perturbation should equal expected_h_zero_adj
    min_perturb_zero = perturbs_zero[0]
    npt.assert_allclose(
        min_perturb_zero, expected_h_zero_adj, rtol=1e-10,
        err_msg=(
            "Step at x=0 should be eps^(1/3)*1e-8 "
            "(hessian_2sided min base = 1e-8, NOT gradient's 1e-2)"
        ),
    )

    # ---- Case 2: x = 100 → h = eps^(1/3) * 100 ----
    x_large = np.array([100.0])
    expected_h_large = eps_val ** (1.0 / 3.0) * np.maximum(
        np.abs(x_large[0]), 1e-8
    )
    xh_large = x_large[0] + expected_h_large
    expected_h_large_adj = xh_large - x_large[0]

    evals_large: list[np.ndarray] = []

    def recording_large(x: np.ndarray) -> float:
        evals_large.append(x.copy())
        return float(np.sum(x ** 2))

    hessian_2sided(recording_large, x_large)

    perturbs_large = sorted({
        round(float(np.abs(ev[0] - 100.0)), 20)
        for ev in evals_large
        if np.abs(ev[0] - 100.0) > 0
    })

    assert len(perturbs_large) > 0, "Expected non-zero perturbations at x=100"
    min_perturb_large = perturbs_large[0]
    npt.assert_allclose(
        min_perturb_large, expected_h_large_adj, rtol=1e-6,
        err_msg="Step at x=100 should scale with |x|",
    )


# ---------------------------------------------------------------------------
# Test 8: Linear Function — f(x) = c'x → Hessian = 0
# ---------------------------------------------------------------------------

def test_hessian_linear_function() -> None:
    """f(x) = c'x → Hessian is the zero matrix.

    A purely linear function has zero second derivatives everywhere.
    The numerical Hessian is not exactly zero due to floating-point
    cancellation: the central-difference formula subtracts nearly equal
    function values, producing noise of order eps * |f(x)| / h² where
    h ≈ eps^(1/3).  This yields residuals on the order of 1e-5 to 1e-4.
    """
    c = np.array([1.5, -2.0, 0.7])
    x0 = np.array([1.0, -1.0, 0.5])

    H = hessian_2sided(lambda z: _linear(z, c), x0)

    # Use HESS_ATOL (1e-4) because the numerical second derivative of a
    # linear function is O(eps^(1/3)) due to subtractive cancellation
    npt.assert_allclose(
        H, np.zeros((3, 3)), atol=HESS_ATOL,
        err_msg="Hessian of linear function should be approximately zero",
    )


# ---------------------------------------------------------------------------
# Test 9: PSD for Convex Functions
# ---------------------------------------------------------------------------

def test_hessian_psd_for_convex() -> None:
    """For convex functions, the Hessian should be positive semi-definite.

    Tests two convex functions:
    1. f(x) = sum(x_i^2) — quadratic bowl, Hessian = 2I (PSD)
    2. f(x) = sum(exp(x_i)) — separable convex, Hessian = diag(exp(x_i)) (PSD)
    """
    x0 = np.array([1.0, -2.0, 3.0])

    # ---- Test 1: sum of squares ----
    H_sq = hessian_2sided(_sum_squares, x0)
    eigs_sq = np.linalg.eigvalsh(H_sq)
    assert np.all(eigs_sq >= -1e-10), (
        f"Hessian of sum(x^2) should be PSD; "
        f"min eigenvalue = {eigs_sq.min():.2e}"
    )

    # ---- Test 2: sum of exp ----
    H_exp = hessian_2sided(_sum_exp, x0)
    eigs_exp = np.linalg.eigvalsh(H_exp)
    assert np.all(eigs_exp >= -1e-10), (
        f"Hessian of sum(exp(x)) should be PSD; "
        f"min eigenvalue = {eigs_exp.min():.2e}"
    )

    # Verify eigenvalues are actually positive (strict PD for these functions)
    assert np.all(eigs_sq > 0.1), (
        "Hessian of sum(x^2) should be strictly positive definite"
    )
    assert np.all(eigs_exp > 0.01), (
        "Hessian of sum(exp(x)) should be strictly positive definite"
    )


# ---------------------------------------------------------------------------
# Test 10: Accuracy vs Analytical Hessian
# ---------------------------------------------------------------------------

def test_hessian_accuracy(spd_matrix: np.ndarray) -> None:
    """Compare numerical Hessian to analytical with atol=1e-4 (O(h²) error).

    Uses two functions with known analytical Hessians:
    1. f(x) = 0.5 * x'Ax + g0'x + c → Hessian = A
    2. Rosenbrock at the optimum (1, 1) → known analytical formula
    """
    A = spd_matrix  # 3×3 SPD matrix from fixture
    g0 = np.array([1.0, -0.5, 0.2])
    c = 3.0

    def f_quad(x: np.ndarray) -> float:
        return _quadratic_half_with_linear(x, A, g0, c)

    x0 = np.array([1.0, -1.0, 0.5])
    H_num = hessian_2sided(f_quad, x0)

    # Hessian of 0.5*x'Ax + g0'x + c is A (constant)
    npt.assert_allclose(
        H_num, A, atol=HESS_ATOL, rtol=HESS_RTOL,
        err_msg="Numerical Hessian of 0.5*x'Ax should match A",
    )

    # ---- Rosenbrock at optimum (1, 1) ----
    x_opt = np.array([1.0, 1.0])
    H_ros = hessian_2sided(_rosenbrock, x_opt)

    # At (1, 1): f_xx = 1200*1 - 400*1 + 2 = 802; f_xy = -400; f_yy = 200
    H_ros_analytic = np.array([
        [802.0, -400.0],
        [-400.0, 200.0],
    ])
    npt.assert_allclose(
        H_ros, H_ros_analytic, atol=HESS_ATOL, rtol=HESS_RTOL,
        err_msg="Rosenbrock Hessian at optimum should match analytical",
    )


# ---------------------------------------------------------------------------
# Test 11: Multidimensional — K=5 parameter vector
# ---------------------------------------------------------------------------

def test_hessian_multidim() -> None:
    """K=5 parameter vector produces correct 5×5 Hessian.

    Uses f(x) = x'Ax with a 5×5 SPD matrix to verify correctness in
    moderate dimensions.
    """
    K = 5
    # Build a known SPD matrix: I + 0.3*ones creates a diagonally dominant SPD
    A = np.eye(K) + 0.3 * np.ones((K, K))
    A = (A + A.T) / 2.0  # guarantee exact symmetry

    x0 = np.array([1.0, -0.5, 2.0, -1.0, 0.3])
    H = hessian_2sided(lambda z: _quadratic_xAx(z, A), x0)

    # Hessian of x'Ax = 2A
    expected = 2.0 * A
    assert H.shape == (K, K), f"Expected ({K}, {K}), got {H.shape}"
    npt.assert_allclose(
        H, expected, atol=HESS_ATOL, rtol=HESS_RTOL,
        err_msg="5-dimensional Hessian should match 2A",
    )

    # Verify symmetry
    npt.assert_array_equal(H, H.T)

    # Verify PSD (since 2A is PD when A is PD)
    eigenvalues = np.linalg.eigvalsh(H)
    assert np.all(eigenvalues > 0), (
        f"Hessian 2A should be positive definite; min eig = {eigenvalues.min()}"
    )


# ---------------------------------------------------------------------------
# Test 12: MATLAB Fixture Parity
# ---------------------------------------------------------------------------

def test_hessian_fixture_parity(utility_fixture_dir: Path) -> None:
    """MATLAB fixture comparison with atol=1e-6, rtol=1e-4.

    Loads the ``hessian_2sided.npy`` fixture (object-dtype dict with 5 cases)
    and verifies the Python implementation produces matching results for each
    fixture case.

    Fixture cases:
        case1: quadratic_2x2 (func_type=1, A, b, c provided)
        case2: sum_of_squares_3x3 (func_type=2)
        case3: rosenbrock_2x2 (func_type=3)
        case4: quadratic_4x4 (func_type=1, A provided)
        case5: exponential_sum_3x3 (func_type=4)
    """
    # Load fixture using shared helper (skips if file missing)
    fixture_data = load_fixture_npy(utility_fixture_dir, "hessian_2sided")

    # The fixture is stored as object-dtype array wrapping a dict
    if hasattr(fixture_data, "item"):
        fixture_data = fixture_data.item()

    # Iterate over all cases in the fixture dictionary
    case_idx = 1
    cases_tested = 0

    while f"case{case_idx}_description" in fixture_data:
        prefix = f"case{case_idx}"
        desc = str(fixture_data[f"{prefix}_description"])
        func_type = int(fixture_data[f"{prefix}_func_type"])
        x = np.asarray(fixture_data[f"{prefix}_x"], dtype=float).ravel()
        expected_H = np.asarray(fixture_data[f"{prefix}_H"], dtype=float)

        if func_type == 1:
            # Quadratic: f(x) = 0.5 * x'Ax + b'x + c → Hessian = A
            # Ref: fixture generated with 0.5 factor so Hessian = A, not 2*A
            A = np.asarray(fixture_data[f"{prefix}_A"], dtype=float)
            n = len(x)
            b_key = f"{prefix}_b"
            c_key = f"{prefix}_c"
            b = np.asarray(
                fixture_data.get(b_key, np.zeros(n)), dtype=float
            ).ravel()
            c_val = float(fixture_data.get(c_key, 0.0))

            H = hessian_2sided(
                lambda z, _A=A, _b=b, _c=c_val: _quadratic_half_with_linear(
                    z, _A, _b, _c
                ),
                x,
            )

        elif func_type == 2:
            # Sum of squares: f(x) = sum(x_i^2) → Hessian = 2I
            H = hessian_2sided(_sum_squares, x)

        elif func_type == 3:
            # Rosenbrock
            H = hessian_2sided(_rosenbrock, x)

        elif func_type == 4:
            # Exponential sum: f(x) = sum(exp(x_i)) → Hessian = diag(exp(x))
            H = hessian_2sided(_sum_exp, x)

        else:
            pytest.skip(
                f"Unknown func_type={func_type} in fixture case {case_idx}"
            )

        npt.assert_allclose(
            H,
            expected_H,
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"Fixture parity failed for {prefix}: {desc}",
        )
        cases_tested += 1
        case_idx += 1

    # Also test the simple fixture pair (x_hess / h2s_H)
    x_hess_path = utility_fixture_dir / "hessian_2sided_x_hess.npy"
    h2s_H_path = utility_fixture_dir / "hessian_2sided_h2s_H.npy"

    if x_hess_path.exists() and h2s_H_path.exists():
        x_simple = np.load(x_hess_path, allow_pickle=True)
        expected_H_simple = np.load(h2s_H_path, allow_pickle=True)

        # Default fixture function: f(x) = sum(x_i^2) → Hessian = 2I
        H_simple = hessian_2sided(_sum_squares, x_simple.ravel())

        npt.assert_allclose(
            H_simple,
            expected_H_simple,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Simple fixture parity failed (x_hess / h2s_H)",
        )
        cases_tested += 1

    assert cases_tested > 0, (
        "No fixture cases were tested — fixture file may be empty or malformed"
    )
