"""Pytest tests for ``mfe_toolbox.utility.robustvcv`` — robust (sandwich)
variance-covariance matrix estimator.

Validates the six-element return tuple ``(VCV, A, B, scores, hess,
gross_scores)`` against known analytical properties and, when available,
MATLAB-generated reference fixtures.

Test categories:
    - Output structure (count, shapes, types)
    - Matrix properties (symmetry, positive semi-definiteness)
    - Numerical accuracy (known MLE, analytical quadratic, single-parameter)
    - MATLAB fixture parity (atol=1e-6, rtol=1e-4)

Source reference: ``utility/robustvcv.m`` (82 lines, Kevin Sheppard)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.robustvcv import robustvcv

# ---------------------------------------------------------------------------
# Tolerance constants — imported from conftest but also defined locally
# as explicit documentation of the ±1e-6 parity contract.
# ---------------------------------------------------------------------------
from tests.conftest import ATOL, RTOL, load_fixture_npy

# ---------------------------------------------------------------------------
# Test helper: Normal log-likelihood function
# ---------------------------------------------------------------------------
# This helper is used across multiple test cases.  It matches the signature
# required by ``robustvcv``: fun(theta, *args) -> (scalar, T-array).
# Ref: Agent Action Plan specification for test_robustvcv.


def normal_loglik(
    params: np.ndarray,
    data: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Normal log-likelihood for location-log-scale model.

    Parameters
    ----------
    params : np.ndarray
        Length-2 vector ``[mu, log_sigma]``.
    data : np.ndarray
        T-length observation vector.

    Returns
    -------
    ll_total : float
        Sum of individual log-likelihoods.
    ll_individual : np.ndarray
        Per-observation log-likelihoods (length T).
    """
    mu = params[0]
    sigma = np.exp(params[1])
    # Ref: AAP normal_loglik helper — standard normal log-likelihood
    ll_individual = -0.5 * (np.log(2.0 * np.pi * sigma**2) + ((data - mu) / sigma) ** 2)
    ll_total = float(np.sum(ll_individual))
    return ll_total, ll_individual


def simple_quadratic_loglik(
    params: np.ndarray,
    data: np.ndarray,
) -> tuple[float, np.ndarray]:
    """A simple quadratic 'likelihood' with known analytical VCV.

    f(theta) = -0.5 * sum_t (y_t - theta)^2
    Individual: f_t = -0.5 * (y_t - theta)^2

    The Hessian is H = -T (constant), so A = H/T = -1.
    Scores: s_t = (y_t - theta), so B = Var(scores) = Var(y).
    VCV = A^{-1} B A^{-1} / T = 1 * Var(y) * 1 / T = Var(y) / T.

    Parameters
    ----------
    params : np.ndarray
        Length-1 parameter [theta].
    data : np.ndarray
        T-length observation vector.

    Returns
    -------
    ll_total : float
        Sum of individual log-likelihoods.
    ll_individual : np.ndarray
        Per-observation log-likelihoods (length T).
    """
    theta = params[0]
    residuals = data - theta
    ll_individual = -0.5 * residuals**2
    ll_total = float(np.sum(ll_individual))
    return ll_total, ll_individual


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_data() -> np.ndarray:
    """Generate reproducible normal data for robustvcv tests.

    Uses a fixed seed separate from the session-level ``rng`` to ensure
    robustvcv tests are self-contained and deterministic.
    """
    rng = np.random.default_rng(12345)
    return rng.standard_normal(500)


@pytest.fixture
def normal_mle(normal_data: np.ndarray) -> np.ndarray:
    """MLE for the normal location-log-scale model.

    theta_hat = [mean(data), log(std(data, ddof=1))]
    """
    mu_hat = float(np.mean(normal_data))
    # Ref: robustvcv.m — MATLAB std uses ddof=1 by default
    sigma_hat = float(np.std(normal_data, ddof=1))
    return np.array([mu_hat, np.log(sigma_hat)])


@pytest.fixture
def robustvcv_result(
    normal_data: np.ndarray,
    normal_mle: np.ndarray,
) -> tuple:
    """Run robustvcv on the normal likelihood at MLE and cache the result."""
    return robustvcv(normal_loglik, normal_mle, 0, normal_data)


# ---------------------------------------------------------------------------
# Test 1: Normal likelihood — VCV approximates Fisher information inverse
# ---------------------------------------------------------------------------


def test_robustvcv_normal_likelihood(
    normal_data: np.ndarray,
    normal_mle: np.ndarray,
    robustvcv_result: tuple,
) -> None:
    """At the MLE of a normal model the robust VCV should approximate the
    inverse of the Fisher information divided by T.

    For N(mu, sigma^2) with known MLE:
      - Var(mu_hat) ≈ sigma^2 / T
      - The (0,0) element of VCV should be close to sigma^2 / T.
    """
    VCV = robustvcv_result[0]
    T = len(normal_data)
    sigma_hat = np.exp(normal_mle[1])
    expected_var_mu = sigma_hat**2 / T

    # The sandwich estimator may differ slightly from the exact Fisher
    # information inverse, so we use a wider tolerance.
    npt.assert_allclose(
        VCV[0, 0],
        expected_var_mu,
        rtol=0.15,
        err_msg="VCV[0,0] should approximate sigma^2/T for normal model",
    )
    # VCV should be non-negative everywhere on diagonal
    assert VCV[0, 0] > 0, "VCV[0,0] must be positive"
    assert VCV[1, 1] > 0, "VCV[1,1] must be positive"


# ---------------------------------------------------------------------------
# Test 2: Output tuple length
# ---------------------------------------------------------------------------


def test_robustvcv_output_tuple(robustvcv_result: tuple) -> None:
    """robustvcv must return exactly 6 values:
    (VCV, A, B, scores, hess, gross_scores).
    """
    assert isinstance(robustvcv_result, tuple), "Return value must be a tuple"
    assert len(robustvcv_result) == 6, (
        f"Expected 6 return values, got {len(robustvcv_result)}"
    )


# ---------------------------------------------------------------------------
# Test 3: VCV shape
# ---------------------------------------------------------------------------


def test_robustvcv_vcv_shape(
    normal_mle: np.ndarray,
    robustvcv_result: tuple,
) -> None:
    """K parameters → K×K VCV matrix."""
    VCV = robustvcv_result[0]
    K = len(normal_mle)
    assert VCV.shape == (K, K), (
        f"VCV shape {VCV.shape} does not match expected ({K}, {K})"
    )


# ---------------------------------------------------------------------------
# Test 4: VCV symmetric
# ---------------------------------------------------------------------------


def test_robustvcv_vcv_symmetric(robustvcv_result: tuple) -> None:
    """VCV matrix must be symmetric: VCV == VCV.T within numerical precision."""
    VCV = robustvcv_result[0]
    assert np.allclose(VCV, VCV.T, atol=1e-12), (
        "VCV matrix is not symmetric"
    )


# ---------------------------------------------------------------------------
# Test 5: VCV positive semi-definite
# ---------------------------------------------------------------------------


def test_robustvcv_vcv_psd(robustvcv_result: tuple) -> None:
    """VCV must be positive semi-definite (all eigenvalues ≥ 0)."""
    VCV = robustvcv_result[0]
    eigenvalues = np.linalg.eigvalsh(VCV)
    assert np.all(eigenvalues >= -1e-10), (
        f"VCV is not PSD; min eigenvalue = {eigenvalues.min():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 6: Hessian shape
# ---------------------------------------------------------------------------


def test_robustvcv_hessian_shape(
    normal_mle: np.ndarray,
    robustvcv_result: tuple,
) -> None:
    """hess (5th output) should be K×K."""
    hess = robustvcv_result[4]
    K = len(normal_mle)
    assert hess.shape == (K, K), (
        f"hess shape {hess.shape} does not match expected ({K}, {K})"
    )


# ---------------------------------------------------------------------------
# Test 7: Scores shape
# ---------------------------------------------------------------------------


def test_robustvcv_scores_shape(
    normal_data: np.ndarray,
    normal_mle: np.ndarray,
    robustvcv_result: tuple,
) -> None:
    """scores (4th output) should be T×K."""
    scores = robustvcv_result[3]
    T = len(normal_data)
    K = len(normal_mle)
    assert scores.shape == (T, K), (
        f"scores shape {scores.shape} does not match expected ({T}, {K})"
    )


# ---------------------------------------------------------------------------
# Test 8: Simple quadratic — known analytical VCV
# ---------------------------------------------------------------------------


def test_robustvcv_simple_quadratic() -> None:
    """For f_t(theta) = -0.5*(y_t - theta)^2 at theta_hat = mean(y):

    - Hessian: H = -T, so A = H/T = -1
    - Scores: s_t = y_t - theta_hat => B = Var(s) = Var(y) (ddof=1)
    - VCV = A^{-1} B A^{-1} / T = Var(y) / T

    This provides a fully analytical check.

    Note: We shift the data to have a non-zero mean so that theta_hat is
    away from zero.  When theta ≈ 0, the step size
    ``h = max(|theta| * eps^(1/3), 1e-8)`` becomes 1e-8, which causes
    catastrophic cancellation in the numerical Hessian.  With theta ≈ 5,
    the step size is ≈ 3e-5, which yields accurate finite differences.
    """
    rng = np.random.default_rng(99)
    # Shift data so theta_hat ≈ 5.0 (avoids h = 1e-8 catastrophic cancellation)
    data = rng.standard_normal(1000) + 5.0
    theta_hat = np.array([np.mean(data)])

    VCV, A, B, scores, hess, gross_scores = robustvcv(
        simple_quadratic_loglik, theta_hat, 0, data,
    )

    T = len(data)

    # VCV should be a 1×1 matrix
    assert VCV.shape == (1, 1), f"VCV shape is {VCV.shape}, expected (1, 1)"

    # Analytical Hessian check: A ≈ -1 (Hessian of -0.5*sum(y-theta)^2 / T)
    # Ref: robustvcv.m:72 — A=hess/t where hess is the raw Hessian
    npt.assert_allclose(
        A[0, 0],
        -1.0,
        atol=1e-3,
        err_msg="A[0,0] should be approximately -1 for quadratic model",
    )

    # Expected VCV = Var(data, ddof=1) / T
    # Ref: robustvcv.m:77 — B=cov(scores) uses 1/(T-1) normalization
    # Note: Since the numerical Hessian has O(h^2) error, we use a wider rtol
    expected_var = np.var(data, ddof=1) / T
    npt.assert_allclose(
        VCV[0, 0],
        expected_var,
        rtol=0.1,
        err_msg="VCV[0,0] should be close to Var(data)/T for quadratic model",
    )


# ---------------------------------------------------------------------------
# Test 9: Single parameter (K=1)
# ---------------------------------------------------------------------------


def test_robustvcv_single_param() -> None:
    """K=1 → VCV is a 1×1 matrix (scalar variance)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(200)
    theta = np.array([np.mean(data)])

    VCV, A, B, scores, hess, gross_scores = robustvcv(
        simple_quadratic_loglik, theta, 0, data,
    )

    # Shape checks
    assert VCV.shape == (1, 1), f"VCV shape {VCV.shape} != (1, 1)"
    assert A.shape == (1, 1), f"A shape {A.shape} != (1, 1)"
    assert B.shape == (1, 1), f"B shape {B.shape} != (1, 1)"
    assert scores.shape == (200, 1), f"scores shape {scores.shape} != (200, 1)"
    assert hess.shape == (1, 1), f"hess shape {hess.shape} != (1, 1)"
    assert gross_scores.shape == (1,), f"gross_scores shape {gross_scores.shape} != (1,)"

    # VCV must be positive
    assert VCV[0, 0] > 0, "VCV[0,0] must be positive for single-param model"


# ---------------------------------------------------------------------------
# Test 10: A matrix is approximately -inv(H)/T
# ---------------------------------------------------------------------------


def test_robustvcv_a_matrix_is_inv_hessian(
    normal_data: np.ndarray,
    normal_mle: np.ndarray,
    robustvcv_result: tuple,
) -> None:
    """Verify the relationship A = hess = H_raw / T.

    From robustvcv.m:72-73:
        A = hess / t
        hess = A

    So both A and hess should be equal (the implementation assigns hess = A
    before returning). We also verify that A is invertible and that
    VCV = Ainv @ B @ Ainv / T.
    """
    VCV, A, B, scores, hess, gross_scores = robustvcv_result

    # A and hess should be identical (robustvcv.m:73: hess=A)
    npt.assert_array_equal(A, hess, err_msg="A and hess should be identical")

    # Verify VCV = Ainv @ B @ Ainv / T
    T = len(normal_data)
    Ainv = np.linalg.inv(A)
    reconstructed_vcv = (Ainv @ B @ Ainv) / T
    npt.assert_allclose(
        VCV,
        reconstructed_vcv,
        atol=1e-10,
        err_msg="VCV != Ainv @ B @ Ainv / T",
    )


# ---------------------------------------------------------------------------
# Test 11: MATLAB fixture parity
# ---------------------------------------------------------------------------


@pytest.mark.parity
@pytest.mark.requires_fixtures
def test_robustvcv_fixture_parity(utility_fixture_dir: Path) -> None:
    """Compare Python robustvcv output against MATLAB-generated fixture.

    Uses the shared ``load_fixture_npy`` helper with graceful skip when
    fixtures are not available.
    """
    # Attempt to load fixture inputs and outputs
    try:
        fixture_input = load_fixture_npy(utility_fixture_dir, "robustvcv")
    except Exception:
        pytest.skip("robustvcv fixture not available")

    # If fixture_input was skipped by load_fixture_npy, execution won't reach here.
    # The fixture may be a structured array/dict or a simple array.
    # Handle both cases gracefully.
    if fixture_input is None:
        pytest.skip("robustvcv fixture is None")

    # Try to load specific fixture files for robustvcv
    # Convention: robustvcv_<output_name>.npy
    vcv_fixture_path = utility_fixture_dir / "robustvcv_vcv.npy"
    if not vcv_fixture_path.exists():
        pytest.skip(
            f"Fixture file not found: {vcv_fixture_path}. "
            "Robustvcv requires Optimization Toolbox functions which may "
            "not be available in Octave fixture generation."
        )

    # If we reach here, load and compare specific outputs
    expected_vcv = np.load(vcv_fixture_path, allow_pickle=True)

    # Also need the input data that was used to generate the fixture
    input_data_path = utility_fixture_dir / "robustvcv_input_data.npy"
    input_theta_path = utility_fixture_dir / "robustvcv_input_theta.npy"
    if not input_data_path.exists() or not input_theta_path.exists():
        pytest.skip("robustvcv fixture input files not available")

    input_data = np.load(input_data_path, allow_pickle=True)
    input_theta = np.load(input_theta_path, allow_pickle=True)

    VCV, A, B, scores, hess, gross_scores = robustvcv(
        normal_loglik, input_theta, 0, input_data,
    )

    npt.assert_allclose(
        VCV,
        expected_vcv,
        atol=ATOL,
        rtol=RTOL,
        err_msg="VCV does not match MATLAB fixture (atol=1e-6, rtol=1e-4)",
    )


# ---------------------------------------------------------------------------
# Additional property tests for completeness
# ---------------------------------------------------------------------------


def test_robustvcv_gross_scores_shape(
    normal_mle: np.ndarray,
    robustvcv_result: tuple,
) -> None:
    """gross_scores (6th output) should be length-K 1-D array."""
    gross_scores = robustvcv_result[5]
    K = len(normal_mle)
    assert gross_scores.shape == (K,), (
        f"gross_scores shape {gross_scores.shape} != ({K},)"
    )


def test_robustvcv_b_matrix_shape(
    normal_mle: np.ndarray,
    robustvcv_result: tuple,
) -> None:
    """B (3rd output) should be K×K."""
    B = robustvcv_result[2]
    K = len(normal_mle)
    assert B.shape == (K, K), f"B shape {B.shape} != ({K}, {K})"


def test_robustvcv_b_matrix_symmetric(robustvcv_result: tuple) -> None:
    """B matrix (score covariance) must be symmetric."""
    B = robustvcv_result[2]
    assert np.allclose(B, B.T, atol=1e-12), "B matrix is not symmetric"


def test_robustvcv_scores_near_zero_at_mle(robustvcv_result: tuple) -> None:
    """At MLE, the mean of individual scores should be close to zero
    (first-order condition). gross_scores ≈ 0 at the optimum.
    """
    scores = robustvcv_result[3]
    # Mean score across observations should be near zero at MLE
    mean_scores = np.mean(scores, axis=0)
    # Use a moderate tolerance since numerical derivatives have O(h^2) error
    npt.assert_allclose(
        mean_scores,
        np.zeros_like(mean_scores),
        atol=0.5,
        err_msg="Mean scores should be approximately zero at MLE",
    )


def test_robustvcv_with_newey_west_lags() -> None:
    """Test robustvcv with nw > 0 (Newey-West HAC correction).

    The Newey-West estimator should return a valid VCV when nw > 0.
    With i.i.d. data, the NW result should be close to the White result.
    """
    rng = np.random.default_rng(77)
    data = rng.standard_normal(500)
    theta = np.array([np.mean(data)])

    # White (nw=0)
    VCV_white, _, _, _, _, _ = robustvcv(
        simple_quadratic_loglik, theta, 0, data,
    )
    # Newey-West (nw=3)
    VCV_nw, A_nw, B_nw, scores_nw, hess_nw, gs_nw = robustvcv(
        simple_quadratic_loglik, theta, 3, data,
    )

    # Both should be valid
    assert VCV_nw.shape == (1, 1), f"NW VCV shape {VCV_nw.shape} != (1, 1)"
    assert VCV_nw[0, 0] > 0, "NW VCV must be positive"

    # For i.i.d. data, both should be reasonably close
    npt.assert_allclose(
        VCV_nw[0, 0],
        VCV_white[0, 0],
        rtol=0.3,
        err_msg=(
            "For i.i.d. data, Newey-West VCV should be close to White VCV"
        ),
    )


def test_robustvcv_multivariate_normal() -> None:
    """Test robustvcv with a 3-parameter multivariate normal model.

    Verifies that robustvcv handles K > 2 parameters correctly with
    all output shapes and properties intact.
    """
    rng = np.random.default_rng(2024)
    T = 300
    K_data = 2
    data = rng.standard_normal((T, K_data))

    def mv_loglik(params: np.ndarray, data: np.ndarray) -> tuple[float, np.ndarray]:
        """Simple bivariate normal log-likelihood with 3 params: mu1, mu2, log_sigma."""
        mu1, mu2, log_sigma = params[0], params[1], params[2]
        sigma = np.exp(log_sigma)
        resid1 = data[:, 0] - mu1
        resid2 = data[:, 1] - mu2
        ll_indiv = (
            -np.log(2.0 * np.pi * sigma**2)
            - 0.5 * (resid1**2 + resid2**2) / sigma**2
        )
        return float(np.sum(ll_indiv)), ll_indiv

    theta = np.array([
        np.mean(data[:, 0]),
        np.mean(data[:, 1]),
        np.log(np.std(data, ddof=1)),
    ])

    VCV, A, B, scores, hess, gross_scores = robustvcv(
        mv_loglik, theta, 0, data,
    )

    K = 3
    assert VCV.shape == (K, K), f"VCV shape {VCV.shape} != ({K}, {K})"
    assert A.shape == (K, K), f"A shape {A.shape} != ({K}, {K})"
    assert B.shape == (K, K), f"B shape {B.shape} != ({K}, {K})"
    assert scores.shape == (T, K), f"scores shape {scores.shape} != ({T}, {K})"
    assert hess.shape == (K, K), f"hess shape {hess.shape} != ({K}, {K})"
    assert gross_scores.shape == (K,), f"gross_scores shape {gross_scores.shape} != ({K},)"

    # VCV should be symmetric and PSD
    assert np.allclose(VCV, VCV.T, atol=1e-10), "VCV not symmetric"
    eigvals = np.linalg.eigvalsh(VCV)
    assert np.all(eigvals >= -1e-10), (
        f"VCV not PSD: min eigenvalue = {eigvals.min():.2e}"
    )
