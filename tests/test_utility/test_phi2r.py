"""
Pytest tests for mfe_toolbox.utility.phi2r — converts K(K-1)/2 partial
correlation angles to a K×K positive-definite correlation matrix.

Source reference: utility/phi2r.m (38 lines, Kevin Sheppard)
Algorithm:
  - Builds upper Cholesky factor C from cos/sin of angles
  - Uses cumulative sine products: C = C * cumprod(S, axis=0)
  - Forms R = C' * C, normalizes to unit diagonal, symmetrizes

Tests validate:
  1. Zero angles → all-ones matrix (perfect correlation)
  2. Known 3×3 angle → correlation matrix
  3. Known 4×4 angle → correlation matrix
  4. Diagonal always exactly 1.0
  5. Output is symmetric
  6. Output is positive definite (all eigenvalues > 0)
  7. Off-diagonal entries in [-1, 1]
  8. Roundtrip phi2r(r2phi(R)) ≈ R
  9. Specific pi/2 angles → near-identity structure
 10. Output shape K×K from K(K-1)/2 input
 11. MATLAB fixture parity (atol=1e-6, rtol=1e-4)

Rules: atol=1e-6, rtol=1e-4 per AAP Section 0.7.1
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.phi2r import phi2r
from mfe_toolbox.utility.r2phi import r2phi
from mfe_toolbox.utility.cov2corr import cov2corr

# Import shared test infrastructure from conftest (auto-discovered by pytest for
# fixtures; explicit import for module-level constants and helper functions).
# Ref: AAP Section 0.7.1 — ATOL=1e-6, RTOL=1e-4 numerical parity tolerances.
from tests.conftest import ATOL, RTOL, load_fixture_npy, assert_allclose


# ---------------------------------------------------------------------------
# Helper: compute K from the length m = K(K-1)/2
# ---------------------------------------------------------------------------
def _k_from_m(m: int) -> int:
    """Recover K given m = K(K-1)/2."""
    # K^2 - K - 2m = 0  =>  K = (1 + sqrt(1 + 8m)) / 2
    K = int(0.5 * (1.0 + np.sqrt(1.0 + 8.0 * m)))
    assert K * (K - 1) // 2 == m, f"m={m} is not K(K-1)/2 for any integer K"
    return K


# ===================================================================
# 1. test_phi2r_zero_angles — All zeros → all-ones (perfect correlation)
# ===================================================================
@pytest.mark.parametrize("K", [3, 4, 5])
def test_phi2r_zero_angles(K: int) -> None:
    """All-zero angles produce an all-ones correlation matrix (perfect correlation).

    When all angles are 0:
      cos(0) = 1 fills the upper Cholesky factor off-diag with 1s,
      sin(0) = 0 zeros out the cumulative product rows below row 0.
    The resulting C matrix has only row 0 non-zero (all 1s), so
    R = C'*C produces a rank-1 all-ones matrix with diag = 1.
    This is the **maximum** correlation case, NOT identity.
    For identity (zero correlation), use pi/2 angles (see test 9).

    Ref: phi2r.m — traced through with phi=[0,...,0].
    """
    m = K * (K - 1) // 2
    phi = np.zeros(m)
    R = phi2r(phi)

    # All-zero angles → all correlations = 1 (perfect correlation)
    expected = np.ones((K, K))
    npt.assert_allclose(R, expected, atol=ATOL, rtol=RTOL)
    # Confirm diagonal is 1.0
    npt.assert_allclose(np.diag(R), np.ones(K), atol=ATOL, rtol=RTOL)
    # Confirm valid correlation matrix properties
    assert R.shape == (K, K)
    npt.assert_allclose(R, R.T, atol=ATOL, rtol=RTOL)


# ===================================================================
# 2. test_phi2r_3x3 — 3 angles → 3×3 correlation matrix
# ===================================================================
def test_phi2r_3x3() -> None:
    """Three angles produce a valid 3×3 correlation matrix.

    Uses known angle vector and verifies basic properties: shape,
    symmetry, unit diagonal, and positive definiteness.
    """
    phi = np.array([0.5, 1.2, 2.1])
    R = phi2r(phi)

    assert R.shape == (3, 3), f"Expected (3, 3), got {R.shape}"
    # Symmetry
    npt.assert_allclose(R, R.T, atol=ATOL, rtol=RTOL)
    # Unit diagonal
    npt.assert_allclose(np.diag(R), np.ones(3), atol=ATOL, rtol=RTOL)
    # Positive definite
    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"
    # Off-diagonal bounds
    offdiag = R[~np.eye(3, dtype=bool)]
    assert np.all(offdiag >= -1.0 - ATOL) and np.all(offdiag <= 1.0 + ATOL), \
        f"Off-diagonal out of [-1, 1]: {offdiag}"


# ===================================================================
# 3. test_phi2r_4x4 — 6 angles → 4×4 correlation matrix
# ===================================================================
def test_phi2r_4x4() -> None:
    """Six angles produce a valid 4×4 correlation matrix.

    Verifies shape, symmetry, unit diagonal, and positive definiteness.
    """
    phi = np.array([0.3, 0.7, 1.5, 2.0, 0.9, 1.8])
    R = phi2r(phi)

    assert R.shape == (4, 4), f"Expected (4, 4), got {R.shape}"
    # Symmetry
    npt.assert_allclose(R, R.T, atol=ATOL, rtol=RTOL)
    # Unit diagonal
    npt.assert_allclose(np.diag(R), np.ones(4), atol=ATOL, rtol=RTOL)
    # Positive definite
    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"


# ===================================================================
# 4. test_phi2r_diagonal_ones — Diagonal always exactly 1.0
# ===================================================================
@pytest.mark.parametrize("K", [3, 4, 5])
def test_phi2r_diagonal_ones(K: int) -> None:
    """Diagonal of the output correlation matrix must always be 1.0.

    The normalization step in phi2r forces diag(R) = 1 regardless
    of angle values.
    Ref: phi2r.m:33 — normalization by sqrt(diag(R)).
    """
    rng = np.random.default_rng(42 + K)
    m = K * (K - 1) // 2
    phi = rng.uniform(0.0, np.pi, size=m)
    R = phi2r(phi)

    npt.assert_allclose(np.diag(R), np.ones(K), atol=ATOL, rtol=RTOL)


# ===================================================================
# 5. test_phi2r_symmetric — Output is symmetric
# ===================================================================
@pytest.mark.parametrize("K", [3, 4, 5])
def test_phi2r_symmetric(K: int) -> None:
    """Output correlation matrix must be symmetric: R == R.T.

    The final symmetrization step (R + R.T) / 2 in phi2r guarantees
    this property.
    Ref: phi2r.m:35 — explicit symmetrization.
    """
    rng = np.random.default_rng(123 + K)
    m = K * (K - 1) // 2
    phi = rng.uniform(0.0, np.pi, size=m)
    R = phi2r(phi)

    npt.assert_allclose(R, R.T, atol=ATOL, rtol=RTOL)


# ===================================================================
# 6. test_phi2r_positive_definite — All eigenvalues > 0
# ===================================================================
@pytest.mark.parametrize("K", [3, 4, 5])
def test_phi2r_positive_definite(K: int) -> None:
    """Output must be positive definite (all eigenvalues > 0).

    By construction, R = C'*C where C has full column rank for generic
    angles, so R is positive definite.
    """
    rng = np.random.default_rng(456 + K)
    m = K * (K - 1) // 2
    phi = rng.uniform(0.1, np.pi - 0.1, size=m)  # Avoid boundary angles
    R = phi2r(phi)

    eigvals = np.linalg.eigvalsh(R)
    assert np.all(eigvals > 0), (
        f"K={K}: non-positive eigenvalues {eigvals}"
    )


# ===================================================================
# 7. test_phi2r_off_diag_in_bounds — Off-diagonal in [-1, 1]
# ===================================================================
@pytest.mark.parametrize("K", [3, 4, 5])
def test_phi2r_off_diag_in_bounds(K: int) -> None:
    """All off-diagonal entries of the correlation matrix lie in [-1, 1].

    Correlation coefficients are bounded by definition; the normalization
    step enforces this.
    """
    rng = np.random.default_rng(789 + K)
    m = K * (K - 1) // 2
    phi = rng.uniform(0.0, np.pi, size=m)
    R = phi2r(phi)

    mask = ~np.eye(K, dtype=bool)
    offdiag = R[mask]
    assert np.all(offdiag >= -1.0 - ATOL), (
        f"Off-diagonal < -1: min = {offdiag.min()}"
    )
    assert np.all(offdiag <= 1.0 + ATOL), (
        f"Off-diagonal > 1: max = {offdiag.max()}"
    )


# ===================================================================
# 8. test_phi2r_roundtrip_r2phi — phi2r(r2phi(R)) ≈ R
# ===================================================================
@pytest.mark.parametrize("K", [3, 4, 5])
def test_phi2r_roundtrip_r2phi(K: int) -> None:
    """Roundtrip: phi2r(r2phi(R)) must recover the original correlation R.

    Generate a valid correlation matrix from a random Wishart-like
    construction: A = randn(K,K), then R = cov2corr(A @ A.T)[1].
    Ref: AAP test spec — generate valid R from random Wishart.
    """
    rng = np.random.default_rng(2024 + K)
    A = rng.standard_normal((K, K))
    cov_matrix = A @ A.T

    # cov2corr returns (sigma, correl) — unpack the correlation matrix
    _, R_orig = cov2corr(cov_matrix)

    # Forward: R → phi → R_reconstructed
    phi = r2phi(R_orig)
    R_reconstructed = phi2r(phi)

    npt.assert_allclose(
        R_reconstructed, R_orig, atol=ATOL, rtol=RTOL,
        err_msg=f"Roundtrip phi2r(r2phi(R)) failed for K={K}"
    )


# ===================================================================
# 9. test_phi2r_specific_values — pi/2 angles → near-identity
# ===================================================================
def test_phi2r_specific_values() -> None:
    """Angles of pi/2 produce a near-identity correlation matrix.

    When all angles are pi/2:
      cos(pi/2) ≈ 0  and  sin(pi/2) = 1
    This makes the Cholesky factor nearly diagonal, so R ≈ I.
    Off-diagonal entries should be approximately zero (within
    floating-point precision of cos(pi/2) ≈ 6.12e-17).
    """
    # 3×3 case: K(K-1)/2 = 3 angles
    K = 3
    m = K * (K - 1) // 2
    phi = np.full(m, np.pi / 2.0)
    R = phi2r(phi)

    # Diagonal must be exactly 1.0
    npt.assert_allclose(np.diag(R), np.ones(K), atol=ATOL, rtol=RTOL)

    # Off-diagonal must be near zero (cos(pi/2) is ~6.12e-17, not exact 0)
    mask = ~np.eye(K, dtype=bool)
    offdiag = R[mask]
    npt.assert_allclose(offdiag, np.zeros_like(offdiag), atol=1e-10,
                        err_msg="pi/2 angles should produce near-identity R")


# ===================================================================
# 10. test_phi2r_output_shape — K(K-1)/2 angles → K×K matrix
# ===================================================================
@pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
def test_phi2r_output_shape(K: int) -> None:
    """K(K-1)/2 input angles must produce a K×K output matrix."""
    m = K * (K - 1) // 2
    phi = np.zeros(m)
    R = phi2r(phi)

    assert isinstance(R, np.ndarray), "Output must be numpy.ndarray"
    assert R.shape == (K, K), f"Expected ({K}, {K}), got {R.shape}"


# ===================================================================
# 11. test_phi2r_fixture_parity — MATLAB fixture comparison
# ===================================================================
def test_phi2r_fixture_parity(utility_fixture_dir) -> None:
    """Compare Python phi2r output against MATLAB-generated fixtures.

    Uses load_fixture_npy from conftest (auto-skips if fixture missing)
    and assert_allclose from conftest (default atol=1e-6, rtol=1e-4).

    Fixture file: tests/fixtures/utility/phi2r.npy
    Contains dict with keys:
      phi_k3, R_k3       — 3×3 standard (no transform)
      phi_k4, R_k4       — 4×4 standard
      phi_k5, R_k5       — 5×5 standard
      phi_k3_u, R_k3_t   — 3×3 with logistic transform
      phi_k4_u, R_k4_t   — 4×4 with logistic transform
      phi_k5_u, R_k5_t   — 5×5 with logistic transform
      phi_k3_half, R_k3_half — pi/2 angles → near-identity
      phi_k3_near0, R_k3_near0 — small angles → high correlation
    """
    # load_fixture_npy auto-skips via pytest.skip if file not found
    raw = load_fixture_npy(utility_fixture_dir, "phi2r")
    data = raw.item()  # Unwrap 0-d object array to dict

    # --- Standard phi2r (no transform) ---
    # 3×3
    R_py_k3 = phi2r(data["phi_k3"])
    assert_allclose(
        R_py_k3, data["R_k3"],
        err_msg="phi2r 3×3 standard parity failed"
    )

    # 4×4
    R_py_k4 = phi2r(data["phi_k4"])
    assert_allclose(
        R_py_k4, data["R_k4"],
        err_msg="phi2r 4×4 standard parity failed"
    )

    # 5×5
    R_py_k5 = phi2r(data["phi_k5"])
    assert_allclose(
        R_py_k5, data["R_k5"],
        err_msg="phi2r 5×5 standard parity failed"
    )

    # --- phi2r with logistic transform ---
    # 3×3 (transform=True)
    R_py_k3_t = phi2r(data["phi_k3_u"], transform=True)
    assert_allclose(
        R_py_k3_t, data["R_k3_t"],
        err_msg="phi2r 3×3 transform parity failed"
    )

    # 4×4 (transform=True)
    R_py_k4_t = phi2r(data["phi_k4_u"], transform=True)
    assert_allclose(
        R_py_k4_t, data["R_k4_t"],
        err_msg="phi2r 4×4 transform parity failed"
    )

    # 5×5 (transform=True)
    R_py_k5_t = phi2r(data["phi_k5_u"], transform=True)
    assert_allclose(
        R_py_k5_t, data["R_k5_t"],
        err_msg="phi2r 5×5 transform parity failed"
    )

    # --- Special cases ---
    # pi/2 angles → near-identity
    R_py_half = phi2r(data["phi_k3_half"])
    assert_allclose(
        R_py_half, data["R_k3_half"],
        err_msg="phi2r pi/2 angles parity failed"
    )

    # Near-zero angles → high correlation
    R_py_near0 = phi2r(data["phi_k3_near0"])
    assert_allclose(
        R_py_near0, data["R_k3_near0"],
        err_msg="phi2r near-zero angles parity failed"
    )
