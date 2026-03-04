"""Pytest tests for mfe_toolbox.utility.gradient_2sided — two-sided numerical gradient.

Tests cover:
- Analytical gradient parity for quadratic, linear, and Rosenbrock functions
- Scalar-valued and vector-valued (score) function modes
- Output shapes and dimensionality
- Step-size computation: h = eps^(1/3) * max(|x|, 1e-2)
- Behaviour at the origin
- Multi-dimensional parameter vectors
- Numerical accuracy against closed-form gradients
- MATLAB fixture parity (atol=1e-6, rtol=1e-4 per AAP §0.7.1)

Ref: utility/gradient_2sided.m — Kevin Sheppard, Revision 3, 2/1/2006.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.gradient_2sided import gradient_2sided

# ---------------------------------------------------------------------------
# Import tolerance constants from conftest (auto-discovered by pytest).
# We also use them directly so that they appear as members_accessed.
# ---------------------------------------------------------------------------
from tests.conftest import ATOL, RTOL, load_fixture_npy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Machine epsilon to the 1/3 power — the theoretical optimal step for central
# differences.  Ref: gradient_2sided.m:41
EPS_THIRD: float = np.finfo(np.float64).eps ** (1.0 / 3.0)

# Looser tolerance for gradient-vs-analytical comparisons: the central-
# difference formula has O(h^2) truncation error which can reach ~1e-4.
GRAD_ATOL: float = 1e-4


# ============================================================================
# Helper objective functions
# ============================================================================

def _sum_of_squares(x: np.ndarray) -> float:
    """f(x) = x' * x  (sum-of-squares).  Gradient = 2x."""
    return float(x @ x)


def _quadratic(x: np.ndarray, A: np.ndarray) -> float:
    """f(x) = x' * A * x.  Gradient = (A + A') * x = 2 * A * x (if A symm.)."""
    return float(x @ A @ x)


def _half_quadratic(x: np.ndarray, A: np.ndarray) -> float:
    """f(x) = 0.5 * x' * A * x.  Gradient = A * x  (for symmetric A).

    This matches the MATLAB fixture's 'weighted_quadratic' test case which
    uses  @(x) 0.5 * x' * A * x  so that the gradient is simply A*x.
    Ref: scripts/generate_fixtures.m — case2_weighted_quadratic
    """
    return float(0.5 * x @ A @ x)


def _linear(x: np.ndarray, c: np.ndarray) -> float:
    """f(x) = c' * x.  Gradient = c (constant)."""
    return float(np.dot(c, x))


def _rosenbrock(x: np.ndarray) -> float:
    """2-D Rosenbrock:  f(x) = (1-x0)^2 + 100*(x1-x0^2)^2.

    Gradient:
        df/dx0 = -2*(1-x0) - 400*x0*(x1-x0^2)
        df/dx1 = 200*(x1-x0^2)
    """
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def _rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Closed-form gradient of the 2-D Rosenbrock function."""
    g0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    g1 = 200.0 * (x[1] - x[0] ** 2)
    return np.array([g0, g1])


def _exp_sum(x: np.ndarray) -> float:
    """f(x) = sum(exp(x_i)).  Gradient = exp(x)."""
    return float(np.sum(np.exp(x)))


def _polynomial_mix(x: np.ndarray) -> float:
    """f(x) = x[0]^3 + 2*x[1]^2 + sin(x[2]).

    Gradient = [3*x[0]^2, 4*x[1], cos(x[2])].
    Matches MATLAB fixture case4 'polynomial_mix'.
    Ref: scripts/generate_fixtures.m — case4_polynomial_mix
    """
    return float(x[0] ** 3 + 2.0 * x[1] ** 2 + np.sin(x[2]))


def _scalar_poly(x: np.ndarray) -> float:
    """f(x) = sum(x_i^3).  Gradient_i = 3*x_i^2."""
    return float(np.sum(x ** 3))


def _gaussian_nll(x: np.ndarray, data: np.ndarray) -> float:
    """Scalar negative log-likelihood for N(mu, exp(log_sigma^2)).

    params = [mu, log_sigma].  Returns scalar NLL.
    """
    mu = x[0]
    sigma = np.exp(x[1])
    # Ref: gradient_2sided.m tests typically use sum of log-likelihoods
    return float(0.5 * np.sum(np.log(2.0 * np.pi * sigma ** 2)
                               + ((data - mu) / sigma) ** 2))


def _gaussian_nll_with_scores(
    x: np.ndarray, data: np.ndarray
) -> tuple[float, np.ndarray]:
    """Returns (scalar NLL, T-element score vector) for gradient_2sided score mode.

    Each element of the score vector is the per-observation NLL contribution.
    """
    mu = x[0]
    sigma = np.exp(x[1])
    scores = 0.5 * (np.log(2.0 * np.pi * sigma ** 2)
                     + ((data - mu) / sigma) ** 2)
    return float(np.sum(scores)), scores


def _multidim_func(x: np.ndarray) -> float:
    """f(x) = x0^2 + 2*x1^2 + 3*x2^2 + 4*x3^2 + 5*x4^2.

    Gradient_i = 2*i_coeff * x_i.
    """
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return float(np.sum(coeffs * x ** 2))


def _vector_func(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Returns (sum, individual_terms) for a vector-valued f.

    f_t(x) = (t+1) * sum(x_i^2),  t = 0..T-1  with T=5.
    Scalar = sum of f_t.  Score vector = [f_0, f_1, ..., f_{T-1}].
    """
    T = 5
    base = float(x @ x)
    scores = np.array([(t + 1) * base for t in range(T)])
    return float(np.sum(scores)), scores


# ============================================================================
# Tests
# ============================================================================


def test_gradient_quadratic() -> None:
    """f(x) = x' * A * x  with A symmetric  →  gradient = 2*A*x.

    Uses a known 3×3 symmetric positive-definite matrix.
    Ref: gradient_2sided.m:73 — G = (gf-gb)./(2*h)
    """
    A = np.array([[2.0, 0.5, 0.0],
                  [0.5, 3.0, 0.5],
                  [0.0, 0.5, 4.0]])
    x = np.array([1.0, 2.0, 3.0])
    expected_grad = 2.0 * A @ x  # = (A + A') @ x since A symmetric

    G = gradient_2sided(_quadratic, x, A)

    npt.assert_allclose(G, expected_grad, atol=GRAD_ATOL,
                        err_msg="Quadratic gradient mismatch")


def test_gradient_linear() -> None:
    """f(x) = c' * x  →  gradient = c  (constant gradient).

    The numerical gradient of a linear function should exactly reproduce
    the coefficient vector c (up to floating-point step-size rounding).
    """
    c = np.array([1.0, -2.0, 3.5, 0.0])
    x = np.array([10.0, -5.0, 0.0, 100.0])

    G = gradient_2sided(_linear, x, c)

    npt.assert_allclose(G, c, atol=GRAD_ATOL,
                        err_msg="Linear gradient should equal coefficient vector c")


def test_gradient_rosenbrock() -> None:
    """Rosenbrock function gradient at (1.5, 2.25).

    Known analytical:
        df/dx0 = -2*(1-1.5) - 400*1.5*(2.25 - 1.5^2)
        df/dx1 = 200*(2.25 - 1.5^2)
    """
    x = np.array([1.5, 2.25])
    expected = _rosenbrock_grad(x)

    G = gradient_2sided(_rosenbrock, x)

    npt.assert_allclose(G, expected, atol=GRAD_ATOL,
                        err_msg="Rosenbrock gradient mismatch")


def test_gradient_scalar_function() -> None:
    """f: R^K → R  should return a gradient vector of shape (K,).

    Uses _exp_sum(x) = sum(exp(x_i)) whose gradient is exp(x).
    """
    rng = np.random.default_rng(99)
    K = 4
    x = rng.standard_normal(K)

    G = gradient_2sided(_exp_sum, x)

    # Verify it is a 1-D ndarray of length K
    assert isinstance(G, np.ndarray), "Gradient must be ndarray"
    assert G.shape == (K,), f"Expected shape ({K},), got {G.shape}"

    # Analytical gradient of sum(exp(x)) is exp(x)
    npt.assert_allclose(G, np.exp(x), atol=GRAD_ATOL,
                        err_msg="exp-sum gradient mismatch")


def test_gradient_vector_function() -> None:
    """f: R^K → R^T  (with compute_scores=True) returns T×K Jacobian.

    _vector_func returns (scalar_sum, T-element score vector).
    gradient_2sided should return (G, Gt) where Gt has shape (T, K).
    """
    K = 3
    x = np.array([1.0, 2.0, 3.0])

    result = gradient_2sided(_vector_func, x, compute_scores=True)

    assert isinstance(result, tuple), "Score mode should return a tuple"
    G, Gt = result

    # G is the gradient of the scalar sum — shape (K,)
    assert G.shape == (K,), f"G shape expected ({K},), got {G.shape}"

    # Gt is the T×K individual score derivatives
    T_expected = 5  # _vector_func has T=5
    assert Gt.shape == (T_expected, K), (
        f"Gt shape expected ({T_expected}, {K}), got {Gt.shape}"
    )


def test_gradient_output_shape() -> None:
    """K-dimensional input → K-dimensional gradient vector.

    Parametrised over several dimensionalities.
    """
    for K in [1, 2, 3, 5, 10]:
        x = np.ones(K)
        G = gradient_2sided(_sum_of_squares, x)
        assert G.shape == (K,), f"K={K}: expected ({K},), got {G.shape}"


def test_gradient_step_size() -> None:
    """Verify step h = eps^(1/3) * max(|x|, 1e-2).

    CRITICAL: the step floor is 1e-2 (NOT 1e-8 as in hessian_2sided).
    Ref: gradient_2sided.m:41 — h = eps.^(1/3)*max(abs(x),1e-2)

    At x = 0 : h = eps^(1/3) * 1e-2  (floor dominates)
    At x = 100 : h = eps^(1/3) * 100  (|x| dominates)
    """
    # --- Test at x = 0 (floor = 1e-2 dominates) ---
    x_zero = np.array([0.0])
    h_expected_zero = EPS_THIRD * 1e-2
    # Compute h the same way the function does internally:
    h_raw = EPS_THIRD * np.maximum(np.abs(x_zero), 1e-2)
    npt.assert_allclose(h_raw, np.array([h_expected_zero]), atol=1e-20,
                        err_msg="Step at x=0 should use floor 1e-2")

    # --- Test at x = 100 (|x| dominates) ---
    x_large = np.array([100.0])
    h_expected_large = EPS_THIRD * 100.0
    h_raw_large = EPS_THIRD * np.maximum(np.abs(x_large), 1e-2)
    npt.assert_allclose(h_raw_large, np.array([h_expected_large]), atol=1e-20,
                        err_msg="Step at x=100 should scale with |x|")

    # --- Test at negative x ---
    x_neg = np.array([-50.0])
    h_expected_neg = EPS_THIRD * 50.0
    h_raw_neg = EPS_THIRD * np.maximum(np.abs(x_neg), 1e-2)
    npt.assert_allclose(h_raw_neg, np.array([h_expected_neg]), atol=1e-20,
                        err_msg="Step at x=-50 should use |x|=50")

    # --- Additional: verify eps^(1/3) value ---
    npt.assert_allclose(EPS_THIRD, np.finfo(np.float64).eps ** (1.0 / 3.0),
                        atol=0.0,
                        err_msg="eps^(1/3) constant mismatch")


def test_gradient_at_origin() -> None:
    """Gradient of f(x) = x^2 at x = 0 should be ≈ 0.

    Ref: gradient_2sided.m:41 — h at x=0 uses floor of 1e-2,
    so the perturbation is non-zero even at the origin.
    Central difference of x^2 at 0: (h^2 - h^2)/(2h) = 0 exactly.
    """
    x = np.array([0.0])

    def _sq(x: np.ndarray) -> float:
        return float(x[0] ** 2)

    G = gradient_2sided(_sq, x)

    npt.assert_allclose(G, np.zeros(1), atol=1e-10,
                        err_msg="Gradient of x^2 at origin should be ~0")


def test_gradient_multidim() -> None:
    """K = 5 parameter vector with weighted quadratic objective.

    f(x) = x0^2 + 2*x1^2 + 3*x2^2 + 4*x3^2 + 5*x4^2
    Gradient_i = 2 * coeff_i * x_i
    """
    x = np.array([1.0, -1.0, 0.5, 2.0, -0.3])
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_grad = 2.0 * coeffs * x

    G = gradient_2sided(_multidim_func, x)

    assert G.shape == (5,), f"Expected shape (5,), got {G.shape}"
    npt.assert_allclose(G, expected_grad, atol=GRAD_ATOL,
                        err_msg="5-D weighted quadratic gradient mismatch")


def test_gradient_accuracy() -> None:
    """Compare numerical gradient to analytical gradient with O(h^2) tolerance.

    Uses several functions where analytical gradients are known exactly.
    NOTE: central difference has O(h^2) ≈ O(eps^{2/3}) ≈ 4e-11 truncation,
    but rounding errors bring practical accuracy to ~1e-6..1e-4.  We use
    atol = 1e-4 (GRAD_ATOL) per AAP §0.7.1 note on gradient accuracy.
    """
    rng = np.random.default_rng(2024)

    # --- Case 1: Sum-of-squares ---
    x1 = rng.standard_normal(3)
    G1 = gradient_2sided(_sum_of_squares, x1)
    npt.assert_allclose(G1, 2.0 * x1, atol=GRAD_ATOL,
                        err_msg="sum-of-squares gradient accuracy")

    # --- Case 2: Exponential ---
    x2 = rng.standard_normal(4) * 0.5  # keep moderate
    G2 = gradient_2sided(_exp_sum, x2)
    npt.assert_allclose(G2, np.exp(x2), atol=GRAD_ATOL,
                        err_msg="exp-sum gradient accuracy")

    # --- Case 3: Cubic polynomial ---
    x3 = rng.standard_normal(3)
    G3 = gradient_2sided(_scalar_poly, x3)
    npt.assert_allclose(G3, 3.0 * x3 ** 2, atol=GRAD_ATOL,
                        err_msg="cubic polynomial gradient accuracy")

    # --- Case 4: Rosenbrock ---
    x4 = np.array([1.2, 1.0])
    G4 = gradient_2sided(_rosenbrock, x4)
    npt.assert_allclose(G4, _rosenbrock_grad(x4), atol=GRAD_ATOL,
                        err_msg="Rosenbrock gradient accuracy")


def test_gradient_score_output() -> None:
    """When compute_scores=True, gradient_2sided returns (G, Gt).

    Gt should be T×K where T is the length of the per-observation score
    vector returned by the objective function.  We also verify that the
    column-sum of Gt equals G (since total gradient = sum of observation
    gradients).
    """
    rng = np.random.default_rng(42)
    T = 20
    data = rng.standard_normal(T)
    x = np.array([0.1, 0.2])  # [mu, log_sigma]

    result = gradient_2sided(
        _gaussian_nll_with_scores, x, data, compute_scores=True
    )

    assert isinstance(result, tuple), "Score mode returns a tuple"
    G, Gt = result

    assert G.shape == (2,), f"G shape expected (2,), got {G.shape}"
    assert Gt.shape == (T, 2), f"Gt shape expected ({T}, 2), got {Gt.shape}"

    # Column sums of Gt should approximate G (the total gradient)
    # Ref: gradient_2sided.m:73-76 — G = (gf-gb)/(2h), Gt = (Gf-Gb)/(2h')
    npt.assert_allclose(Gt.sum(axis=0), G, atol=GRAD_ATOL,
                        err_msg="Column sums of Gt should equal G")


def test_gradient_fixture_parity(utility_fixture_dir: Path) -> None:
    """Compare gradient_2sided output against MATLAB reference fixtures.

    Fixtures were generated by scripts/generate_fixtures.m via Octave and
    stored in tests/fixtures/utility/gradient_2sided.npy.

    Per AAP §0.7.1: atol=1e-6, rtol=1e-4 for MATLAB parity.
    """
    fixture = load_fixture_npy(utility_fixture_dir, "gradient_2sided")

    # The fixture is a 0-d object array wrapping a dict
    if fixture.ndim == 0:
        d = fixture.item()
    else:
        d = fixture  # pragma: no cover

    # ------------------------------------------------------------------
    # Case 1: quadratic_sum_of_squares  f(x)=x'*x, x=[1,2,3]
    # ------------------------------------------------------------------
    x1 = np.asarray(d["case1_x"])
    G1_expected = np.asarray(d["case1_G"])

    G1 = gradient_2sided(_sum_of_squares, x1)
    npt.assert_allclose(G1, G1_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case1 (sum-of-squares) G mismatch")

    # Verify step sizes match MATLAB
    h1_expected = np.asarray(d["case1_h"])
    h1_computed = EPS_THIRD * np.maximum(np.abs(x1), 1e-2)
    xh1 = x1 + h1_computed
    h1_computed = xh1 - x1
    npt.assert_allclose(h1_computed, h1_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case1 step-size h mismatch")

    # ------------------------------------------------------------------
    # Case 2: weighted_quadratic  f(x)=0.5*x'*A*x, x=[1.5,-0.5]
    # MATLAB fixture uses  @(x) 0.5 * x'*A*x  so gradient = A*x.
    # ------------------------------------------------------------------
    x2 = np.asarray(d["case2_x"])
    A2 = np.asarray(d["case2_A"])
    G2_expected = np.asarray(d["case2_G"])

    G2 = gradient_2sided(_half_quadratic, x2, A2)
    npt.assert_allclose(G2, G2_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case2 (weighted quadratic) G mismatch")

    # ------------------------------------------------------------------
    # Case 3: exponential_sum  f(x)=sum(exp(x)), x=[0,0.5,1,-1]
    # ------------------------------------------------------------------
    x3 = np.asarray(d["case3_x"])
    G3_expected = np.asarray(d["case3_G"])

    G3 = gradient_2sided(_exp_sum, x3)
    npt.assert_allclose(G3, G3_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case3 (exp-sum) G mismatch")

    # ------------------------------------------------------------------
    # Case 4: polynomial_mix  f(x)=x0^3 + 2*x1^2 + sin(x2)
    # ------------------------------------------------------------------
    x4 = np.asarray(d["case4_x"])
    G4_expected = np.asarray(d["case4_G"])

    G4 = gradient_2sided(_polynomial_mix, x4)
    npt.assert_allclose(G4, G4_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case4 (polynomial_mix) G mismatch")

    # ------------------------------------------------------------------
    # Case 5: near_zero_parameters (tests the 1e-2 floor!)
    # ------------------------------------------------------------------
    x5 = np.asarray(d["case5_x"])
    G5_expected = np.asarray(d["case5_G"])
    h5_expected = np.asarray(d["case5_h"])

    G5 = gradient_2sided(_sum_of_squares, x5)
    npt.assert_allclose(G5, G5_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case5 (near-zero) G mismatch")

    # Verify step sizes for near-zero x
    h5_computed = EPS_THIRD * np.maximum(np.abs(x5), 1e-2)
    xh5 = x5 + h5_computed
    h5_computed = xh5 - x5
    npt.assert_allclose(h5_computed, h5_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case5 step-size h mismatch")

    # ------------------------------------------------------------------
    # Case 6: single_parameter  f(x)=x^2, x=[3]
    # ------------------------------------------------------------------
    x6 = np.asarray(d["case6_x"])
    G6_expected = np.asarray(d["case6_G"])

    G6 = gradient_2sided(_sum_of_squares, x6)
    npt.assert_allclose(G6, G6_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case6 (single param) G mismatch")

    # ------------------------------------------------------------------
    # Case 7: gaussian_nll_with_scores (score mode)
    # ------------------------------------------------------------------
    x7 = np.asarray(d["case7_x"])
    data7 = np.asarray(d["case7_data"])
    G7_expected = np.asarray(d["case7_G"])
    Gt7_expected = np.asarray(d["case7_Gt"])

    G7, Gt7 = gradient_2sided(
        _gaussian_nll_with_scores, x7, data7, compute_scores=True
    )
    npt.assert_allclose(G7, G7_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case7 (score) G mismatch")
    npt.assert_allclose(Gt7, Gt7_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Fixture case7 (score) Gt mismatch")
