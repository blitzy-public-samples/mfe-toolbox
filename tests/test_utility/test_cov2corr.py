"""Pytest tests for mfe_toolbox.utility.cov2corr — covariance-to-correlation conversion.

Tests verify numerical parity with MATLAB reference outputs (cov2corr.m, 40 lines)
using atol=1e-6, rtol=1e-4 per AAP Section 0.7.1.

Source: utility/cov2corr.m — Kevin Sheppard, Revision 1, 10/23/2012
Key MATLAB behaviour:
    2-D: sigma = sqrt(diag(cov)); correl = cov ./ (sigma*sigma');
    3-D: K×K×T loop — sigma(t,:) = sqrt(diag(cov(:,:,t))),
         correl(:,:,t) = cov(:,:,t) ./ (sigma(t,:)' * sigma(t,:));
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.cov2corr import cov2corr
from tests.conftest import ATOL, RTOL, load_fixture_npy


# ---------------------------------------------------------------------------
# Test 1: 2×2 covariance → correlation
# ---------------------------------------------------------------------------

def test_cov2corr_2x2() -> None:
    """2×2 covariance → correlation. Diagonal should be exactly 1.0.

    Ref: cov2corr.m:28-30 — 2-D path.
    MATLAB: C = [4 1.2; 1.2 9] → sigma = [2 3], R = [1 0.2; 0.2 1]
    """
    cov = np.array([[4.0, 1.2],
                     [1.2, 9.0]])
    sigma, correl = cov2corr(cov)

    expected_sigma = np.array([2.0, 3.0])
    expected_correl = np.array([[1.0, 0.2],
                                 [0.2, 1.0]])

    npt.assert_allclose(sigma, expected_sigma, atol=ATOL, rtol=RTOL,
                         err_msg="2x2 sigma mismatch")
    npt.assert_allclose(correl, expected_correl, atol=ATOL, rtol=RTOL,
                         err_msg="2x2 correl mismatch")
    # Diagonal must be exactly 1.0
    npt.assert_array_equal(np.diag(correl), np.ones(2),
                            err_msg="2x2 diagonal not exactly 1.0")


# ---------------------------------------------------------------------------
# Test 2: 3×3 covariance → correlation with known values
# ---------------------------------------------------------------------------

def test_cov2corr_3x3() -> None:
    """3×3 covariance → correlation with analytically known values.

    C = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
    sigma = [2, sqrt(5), sqrt(6)]
    R_ij = C_ij / (sigma_i * sigma_j)
    """
    cov = np.array([[4.0, 2.0, 1.0],
                     [2.0, 5.0, 3.0],
                     [1.0, 3.0, 6.0]])
    sigma, correl = cov2corr(cov)

    expected_sigma = np.array([2.0, np.sqrt(5.0), np.sqrt(6.0)])
    # R[0,1] = 2 / (2*sqrt(5)) = 1/sqrt(5) ≈ 0.4472136
    # R[0,2] = 1 / (2*sqrt(6)) ≈ 0.2041241
    # R[1,2] = 3 / (sqrt(5)*sqrt(6)) = 3/sqrt(30) ≈ 0.5477226
    expected_correl = cov / np.outer(expected_sigma, expected_sigma)
    np.fill_diagonal(expected_correl, 1.0)

    npt.assert_allclose(sigma, expected_sigma, atol=ATOL, rtol=RTOL,
                         err_msg="3x3 sigma mismatch")
    npt.assert_allclose(correl, expected_correl, atol=ATOL, rtol=RTOL,
                         err_msg="3x3 correl mismatch")


# ---------------------------------------------------------------------------
# Test 3: Identity → Identity (cov already normalized)
# ---------------------------------------------------------------------------

def test_cov2corr_identity() -> None:
    """Identity covariance (already unit variance) → identity correlation.

    Ref: cov2corr.m:29-30 — sigma = [1,1,...,1], correl = I.
    """
    for k in (2, 3, 5):
        cov = np.eye(k)
        sigma, correl = cov2corr(cov)

        npt.assert_allclose(sigma, np.ones(k), atol=ATOL, rtol=RTOL,
                             err_msg=f"Identity K={k}: sigma not ones")
        npt.assert_allclose(correl, np.eye(k), atol=ATOL, rtol=RTOL,
                             err_msg=f"Identity K={k}: correl not identity")


# ---------------------------------------------------------------------------
# Test 4: Diagonal matrix → Identity correlation (uncorrelated)
# ---------------------------------------------------------------------------

def test_cov2corr_diagonal() -> None:
    """Diagonal covariance (uncorrelated) → identity correlation.

    Ref: cov2corr.m:30 — off-diagonal of cov are zero, so R = I.
    """
    diag_vals = np.array([2.5, 0.5, 8.1])
    cov = np.diag(diag_vals)
    sigma, correl = cov2corr(cov)

    expected_sigma = np.sqrt(diag_vals)
    npt.assert_allclose(sigma, expected_sigma, atol=ATOL, rtol=RTOL,
                         err_msg="Diagonal cov: sigma mismatch")
    npt.assert_allclose(correl, np.eye(3), atol=ATOL, rtol=RTOL,
                         err_msg="Diagonal cov: correl not identity")


# ---------------------------------------------------------------------------
# Test 5: Output diagonal is always exactly 1.0
# ---------------------------------------------------------------------------

def test_cov2corr_diagonal_ones() -> None:
    """Output correlation diagonal is always exactly 1.0 for any valid input.

    Ref: cov2corr.m:30 — R = cov ./ (sigma*sigma') naturally gives R_ii = 1
    because cov_ii / (sigma_i^2) = sigma_i^2 / sigma_i^2 = 1.
    """
    rng = np.random.default_rng(42)
    # Generate a random positive-definite covariance via A'A
    for k in (2, 3, 5, 10):
        A = rng.standard_normal((k + 5, k))
        cov = A.T @ A  # guaranteed PSD
        _, correl = cov2corr(cov)
        # Floating-point division cov_ii / sigma_i^2 may not be *exactly* 1.0
        # but must be within machine epsilon of 1.0.
        npt.assert_allclose(
            np.diag(correl), np.ones(k), atol=1e-14, rtol=0,
            err_msg=f"K={k}: diagonal not close to 1.0"
        )


# ---------------------------------------------------------------------------
# Test 6: Output is symmetric
# ---------------------------------------------------------------------------

def test_cov2corr_symmetric() -> None:
    """Output correlation matrix must be symmetric (R == R.T).

    Ref: cov2corr.m:30 — if cov is symmetric and sigma is real,
    then cov ./ outer(sigma, sigma) is symmetric.
    """
    rng = np.random.default_rng(123)
    for k in (2, 3, 5):
        A = rng.standard_normal((k + 3, k))
        cov = A.T @ A
        _, correl = cov2corr(cov)
        npt.assert_allclose(correl, correl.T, atol=ATOL, rtol=RTOL,
                             err_msg=f"K={k}: correlation not symmetric")


# ---------------------------------------------------------------------------
# Test 7: All off-diagonal elements in [-1, 1]
# ---------------------------------------------------------------------------

def test_cov2corr_bounds() -> None:
    """All off-diagonal correlation elements must lie in [-1, 1].

    For a valid (positive semi-definite) covariance matrix, the Cauchy-Schwarz
    inequality guarantees |R_ij| <= 1.
    """
    rng = np.random.default_rng(777)
    for k in (3, 5, 8):
        A = rng.standard_normal((k + 5, k))
        cov = A.T @ A
        _, correl = cov2corr(cov)
        # Extract off-diagonal
        off_diag = correl[~np.eye(k, dtype=bool)]
        assert np.all(off_diag >= -1.0 - ATOL), \
            f"K={k}: off-diag < -1 found: min={off_diag.min()}"
        assert np.all(off_diag <= 1.0 + ATOL), \
            f"K={k}: off-diag > 1 found: max={off_diag.max()}"


# ---------------------------------------------------------------------------
# Test 8: 3-D time-varying covariance input
# ---------------------------------------------------------------------------

def test_cov2corr_3d_input() -> None:
    """K×K×T 3-D input → K×K×T correlation, each slice normalised.

    Ref: cov2corr.m:31-39 — loops over T slices.
    Verifies:
      - sigma shape is (T, K)
      - correl shape is (K, K, T)
      - each slice diagonal is 1.0
      - each slice is symmetric
    """
    rng = np.random.default_rng(99)
    K, T = 3, 5
    cov_3d = np.zeros((K, K, T))
    for t in range(T):
        A = rng.standard_normal((K + 3, K))
        cov_3d[:, :, t] = A.T @ A

    sigma, correl = cov2corr(cov_3d)

    # Shape checks — Ref: cov2corr.m:34-35
    assert sigma.shape == (T, K), f"sigma shape {sigma.shape} != ({T}, {K})"
    assert correl.shape == (K, K, T), f"correl shape {correl.shape} != ({K}, {K}, {T})"

    for t in range(T):
        # Diagonal of each slice is 1.0 (within floating-point precision)
        npt.assert_allclose(
            np.diag(correl[:, :, t]), np.ones(K), atol=1e-14, rtol=0,
            err_msg=f"t={t}: diagonal not close to 1.0"
        )
        # Each slice is symmetric
        npt.assert_allclose(
            correl[:, :, t], correl[:, :, t].T, atol=ATOL, rtol=RTOL,
            err_msg=f"t={t}: slice not symmetric"
        )
        # sigma matches sqrt of diagonal of cov slice
        expected_sigma_t = np.sqrt(np.diag(cov_3d[:, :, t]))
        npt.assert_allclose(
            sigma[t, :], expected_sigma_t, atol=ATOL, rtol=RTOL,
            err_msg=f"t={t}: sigma mismatch"
        )


# ---------------------------------------------------------------------------
# Test 9: Non-square input raises ValueError
# ---------------------------------------------------------------------------

def test_cov2corr_non_square_raises() -> None:
    """Non-square covariance matrix input must raise ValueError.

    Ref: cov2corr.py:84-88 — shape[0] != shape[1] → ValueError.
    """
    non_square = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="first two dimensions"):
        cov2corr(non_square)


# ---------------------------------------------------------------------------
# Test 10: Non-positive semidefinite handling
# ---------------------------------------------------------------------------

def test_cov2corr_non_psd_handling() -> None:
    """Test behaviour with a non-positive-semidefinite (non-PSD) input.

    cov2corr does NOT enforce PSD-ness — it simply divides by
    sqrt(diag(S))*sqrt(diag(S))'. With a non-PSD matrix that still has
    positive diagonal entries, the function should run without error but
    may produce off-diagonal |R_ij| > 1.

    If the diagonal has a zero or negative entry, sqrt produces NaN/complex
    and the result will contain NaN. We verify the function completes and
    produces a (K, K) result.
    """
    # Case 1: non-PSD but positive diagonal — function runs
    non_psd = np.array([[1.0, 5.0],
                         [5.0, 1.0]])  # eigenvalues: 6, -4
    sigma, correl = cov2corr(non_psd)
    assert sigma.shape == (2,), "sigma shape mismatch for non-PSD"
    assert correl.shape == (2, 2), "correl shape mismatch for non-PSD"
    # Off-diagonal will be 5/(1*1) = 5 — outside [-1,1]
    npt.assert_allclose(correl[0, 1], 5.0, atol=ATOL, rtol=RTOL,
                         err_msg="Non-PSD: expected R[0,1]=5.0")

    # Case 2: zero diagonal entry → sigma contains 0 → division by 0 → inf/nan
    zero_diag = np.array([[0.0, 1.0],
                           [1.0, 4.0]])
    sigma2, correl2 = cov2corr(zero_diag)
    assert sigma2.shape == (2,), "sigma shape mismatch for zero-diag"
    # sigma[0] = 0 → division by zero → inf or nan in correl
    assert np.isnan(correl2[0, 1]) or np.isinf(correl2[0, 1]), \
        "Zero-diag: expected inf or nan in off-diagonal"


# ---------------------------------------------------------------------------
# Test 11: Fixture parity (MATLAB reference comparison)
# ---------------------------------------------------------------------------

def test_cov2corr_fixture_parity(utility_fixture_dir: Path) -> None:
    """Compare Python cov2corr against MATLAB/Octave-generated fixtures.

    Loads the 'cov2corr' fixture file which contains multiple test cases
    generated by Octave's cov2corr function, and verifies sigma/correl
    parity with atol=1e-6, rtol=1e-4.
    """
    fixture = load_fixture_npy(utility_fixture_dir, "cov2corr")
    # fixture is an object-dtype ndarray wrapping a dict
    data = fixture.item() if fixture.ndim == 0 else fixture

    # --- Test case 1: 2×2 known ---
    sigma1, correl1 = cov2corr(data["test1_input_cov"])
    npt.assert_allclose(sigma1, data["test1_sigma"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test1 sigma mismatch")
    npt.assert_allclose(correl1, data["test1_correl"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test1 correl mismatch")

    # --- Test case 2: 3×3 random ---
    sigma2, correl2 = cov2corr(data["test2_input_cov"])
    npt.assert_allclose(sigma2, data["test2_sigma"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test2 sigma mismatch")
    npt.assert_allclose(correl2, data["test2_correl"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test2 correl mismatch")

    # --- Test case 3: 4×4 random ---
    sigma3, correl3 = cov2corr(data["test3_input_cov"])
    npt.assert_allclose(sigma3, data["test3_sigma"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test3 sigma mismatch")
    npt.assert_allclose(correl3, data["test3_correl"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test3 correl mismatch")

    # --- Test case 4: 5×5 random ---
    sigma4, correl4 = cov2corr(data["test4_input_cov"])
    npt.assert_allclose(sigma4, data["test4_sigma"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test4 sigma mismatch")
    npt.assert_allclose(correl4, data["test4_correl"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test4 correl mismatch")

    # --- Test case 5: identity ---
    sigma5, correl5 = cov2corr(data["test5_input_cov"])
    npt.assert_allclose(sigma5, data["test5_sigma"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test5 sigma mismatch")
    npt.assert_allclose(correl5, data["test5_correl"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test5 correl mismatch")

    # --- Test case 6: diagonal ---
    sigma6, correl6 = cov2corr(data["test6_input_cov"])
    npt.assert_allclose(sigma6, data["test6_sigma"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test6 sigma mismatch")
    npt.assert_allclose(correl6, data["test6_correl"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test6 correl mismatch")

    # --- Test case 7: 3×3×5 time-varying ---
    sigma7, correl7 = cov2corr(data["test7_input_cov"])
    npt.assert_allclose(sigma7, data["test7_sigma"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test7 sigma mismatch")
    npt.assert_allclose(correl7, data["test7_correl"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test7 correl mismatch")

    # --- Test case 8: 2×2×3 time-varying ---
    sigma8, correl8 = cov2corr(data["test8_input_cov"])
    npt.assert_allclose(sigma8, data["test8_sigma"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test8 sigma mismatch")
    npt.assert_allclose(correl8, data["test8_correl"], atol=ATOL, rtol=RTOL,
                         err_msg="Fixture test8 correl mismatch")
