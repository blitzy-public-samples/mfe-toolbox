"""Pytest test suite for mfe_toolbox.utility.r2phi — correlation matrix to angle vector.

Tests the conversion of K×K correlation matrices to K(K-1)/2 angle vectors
via Cholesky decomposition and inverse-cosine extraction.  The r2phi function
is the inverse of phi2r: given a valid correlation matrix R, it recovers the
angle parameterization used by phi2r.

Source reference: utility/r2phi.m (35 lines, Kevin Sheppard)
Numerical parity: atol=1e-6, rtol=1e-4 per AAP Section 0.7.1

Key behaviors from r2phi.m:
  - Input: K×K positive-definite correlation matrix R
  - Output: K(K-1)/2 vector of angles in [0, pi]
  - Uses Cholesky decomposition: X = chol(R) (MATLAB upper triangular)
  - Extracts angles via arccos from Cholesky rows with cumulative sine products
  - Identity matrix maps to all pi/2 angles (arccos(0) = pi/2)
  - MATLAB docstring says [0, 2*pi] but actual output is [0, pi] due to arccos range
    (r2phi.m:19 has a FIXME about needing both cos and sin inversion for full [0, 2*pi])

Translation notes (MATLAB → Python):
  - MATLAB chol(R) returns upper triangular; numpy.linalg.cholesky returns lower,
    so .T is applied in the Python implementation (r2phi.py:76)
  - MATLAB 1-indexed loops → Python 0-indexed loops (r2phi.py:96)
  - MATLAB column-major P(P>0) → Python ravel('F') then mask (r2phi.py:119)
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.r2phi import r2phi
from mfe_toolbox.utility.phi2r import phi2r
from tests.conftest import ATOL, RTOL, load_fixture_npy


# ---------------------------------------------------------------------------
# Helper: Build a valid random K×K correlation matrix (reproducible)
# ---------------------------------------------------------------------------

def _random_corr_matrix(k: int, seed: int = 42) -> np.ndarray:
    """Generate a valid K×K correlation matrix from a seeded random generator.

    Uses A @ A.T + eps*I construction to guarantee positive definiteness,
    then normalizes to unit diagonal.

    Parameters
    ----------
    k : int
        Matrix dimension.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        K×K valid correlation matrix.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((k, k))
    S = A @ A.T + 0.1 * np.eye(k)
    d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------------------
# Test 1: test_r2phi_identity — Identity matrix → pi/2 angles
# ---------------------------------------------------------------------------

class TestR2PhiIdentity:
    """Test r2phi behavior with identity correlation matrices.

    For an identity matrix, the Cholesky factor is also the identity.
    Off-diagonal elements of X are zero, and cumS values are 1, so
    P[i,j] = arccos(0/1) = pi/2 for all extracted angles.

    Ref: r2phi.m:20-31 — When R=eye(k), X=eye(k), all angles are pi/2.
    """

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_identity_values(self, k: int) -> None:
        """Identity K×K correlation matrix should produce all pi/2 angles."""
        R = np.eye(k)
        phi = r2phi(R)
        expected_length = k * (k - 1) // 2
        # Identity → all off-diagonal Cholesky elements are 0 → arccos(0) = pi/2
        expected = np.full(expected_length, np.pi / 2.0)

        assert phi.shape == (expected_length,), (
            f"Expected shape ({expected_length},) for K={k}, got {phi.shape}"
        )
        npt.assert_allclose(
            phi, expected, atol=ATOL, rtol=RTOL,
            err_msg=f"Identity K={k} should produce all pi/2 angles"
        )

    def test_identity_2x2(self) -> None:
        """2×2 identity specifically: 1 angle = pi/2."""
        R = np.eye(2)
        phi = r2phi(R)
        assert phi.shape == (1,)
        npt.assert_allclose(
            phi, np.array([np.pi / 2.0]), atol=ATOL, rtol=RTOL,
            err_msg="2×2 identity should give [pi/2]"
        )

    def test_identity_3x3(self) -> None:
        """3×3 identity specifically: 3 angles = [pi/2, pi/2, pi/2]."""
        R = np.eye(3)
        phi = r2phi(R)
        assert phi.shape == (3,)
        npt.assert_allclose(
            phi, np.array([np.pi / 2.0] * 3), atol=ATOL, rtol=RTOL,
            err_msg="3×3 identity should give [pi/2, pi/2, pi/2]"
        )


# ---------------------------------------------------------------------------
# Test 2: test_r2phi_3x3 — Known 3×3 correlation → known angles
# ---------------------------------------------------------------------------

class TestR2Phi3x3:
    """Test r2phi with known 3×3 correlation matrices from CSV fixtures.

    Verifies that the Python implementation produces identical angles to
    the MATLAB reference for several test cases.

    Ref: tests/fixtures/utility/r2phi.csv — MATLAB-generated reference values.
    """

    def test_3x3_positive_corr(self) -> None:
        """3×3 with positive correlations [0.5, 0.3, 0.4].

        Ref: r2phi.csv row '3x3_positive_corr'.
        Off-diagonal: R[0,1]=0.5, R[0,2]=0.3, R[1,2]=0.4
        Expected phi from MATLAB: [1.04719755, 1.26610367, 1.26336252]
        """
        R = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        phi = r2phi(R)
        # Expected from r2phi.csv row 3x3_positive_corr
        expected = np.array([
            1.0471975511965976e+00,
            1.2661036727794992e+00,
            1.2633625162200470e+00,
        ])
        assert phi.shape == (3,)
        npt.assert_allclose(
            phi, expected, atol=ATOL, rtol=RTOL,
            err_msg="3×3 positive correlation angles do not match MATLAB fixture"
        )

    def test_3x3_negative_corr(self) -> None:
        """3×3 with negative correlations [-0.6, -0.3, 0.2].

        Ref: r2phi.csv row '3x3_negative_corr'.
        Off-diagonal: R[0,1]=-0.6, R[0,2]=-0.3, R[1,2]=0.2
        """
        R = np.array([
            [1.0, -0.6, -0.3],
            [-0.6, 1.0, 0.2],
            [-0.3, 0.2, 1.0],
        ])
        phi = r2phi(R)
        expected = np.array([
            2.2142974355881808e+00,
            1.8754889808102941e+00,
            1.5445862050499302e+00,
        ])
        assert phi.shape == (3,)
        npt.assert_allclose(
            phi, expected, atol=ATOL, rtol=RTOL,
            err_msg="3×3 negative correlation angles do not match MATLAB fixture"
        )

    def test_3x3_mixed_corr(self) -> None:
        """3×3 with mixed correlations [0.7, -0.4, -0.1].

        Ref: r2phi.csv row '3x3_mixed_corr'.
        Off-diagonal: R[0,1]=0.7, R[0,2]=-0.4, R[1,2]=-0.1
        """
        R = np.array([
            [1.0, 0.7, -0.4],
            [0.7, 1.0, -0.1],
            [-0.4, -0.1, 1.0],
        ])
        phi = r2phi(R)
        expected = np.array([
            7.9539883018414359e-01,
            1.9823131728623846e+00,
            1.2921966923429480e+00,
        ])
        assert phi.shape == (3,)
        npt.assert_allclose(
            phi, expected, atol=ATOL, rtol=RTOL,
            err_msg="3×3 mixed correlation angles do not match MATLAB fixture"
        )

    def test_3x3_identity(self) -> None:
        """3×3 identity from CSV fixture: all angles = pi/2.

        Ref: r2phi.csv row '3x3_identity'.
        """
        R = np.eye(3)
        phi = r2phi(R)
        expected = np.array([
            1.5707963267948966e+00,
            1.5707963267948966e+00,
            1.5707963267948966e+00,
        ])
        assert phi.shape == (3,)
        npt.assert_allclose(
            phi, expected, atol=ATOL, rtol=RTOL,
            err_msg="3×3 identity angles do not match MATLAB fixture"
        )


# ---------------------------------------------------------------------------
# Test 3: test_r2phi_output_length — K×K → K(K-1)/2 elements
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7])
def test_r2phi_output_length(k: int) -> None:
    """Output vector length must equal K(K-1)/2 for K×K input.

    Ref: r2phi.m:32 — phi = P(P>0) extracts K(K-1)/2 positive elements
    from the angle matrix P.
    """
    R = _random_corr_matrix(k, seed=42 + k)
    phi = r2phi(R)
    expected_length = k * (k - 1) // 2

    assert isinstance(phi, np.ndarray), "Return type must be numpy.ndarray"
    assert phi.shape == (expected_length,), (
        f"Expected shape ({expected_length},) for K={k}, got {phi.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4: test_r2phi_angle_range — All angles in valid range
# ---------------------------------------------------------------------------

class TestR2PhiAngleRange:
    """Test that all output angles lie within valid bounds.

    Ref: r2phi.m:19 — FIXME comment notes output should be [0, 2*pi] but
    actual implementation only uses arccos which produces [0, pi].
    The Python implementation (r2phi.py:54) documents this as [0, pi].

    We test the strict [0, pi] range since that is what arccos produces.
    We also verify the weaker [0, 2*pi] range from the MATLAB docstring.
    """

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_angles_in_0_pi(self, k: int) -> None:
        """All angles must be in [0, pi] (arccos range)."""
        R = _random_corr_matrix(k, seed=100 + k)
        phi = r2phi(R)
        assert np.all(phi >= 0.0), f"Some angles < 0 for K={k}"
        assert np.all(phi <= np.pi + 1e-10), f"Some angles > pi for K={k}"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_angles_in_0_2pi(self, k: int) -> None:
        """All angles must be in [0, 2*pi] (per MATLAB docstring).

        This is a weaker check — the actual range is [0, pi] since only
        arccos is used. The MATLAB docstring (r2phi.m:11) says [0, 2*pi].
        """
        R = _random_corr_matrix(k, seed=200 + k)
        phi = r2phi(R)
        assert np.all(phi >= 0.0), f"Some angles < 0 for K={k}"
        assert np.all(phi <= 2.0 * np.pi + 1e-10), f"Some angles > 2*pi for K={k}"

    def test_extreme_positive_corr(self) -> None:
        """Near-perfect positive correlation → angle near 0."""
        R = np.array([[1.0, 0.99], [0.99, 1.0]])
        phi = r2phi(R)
        # arccos(0.99) ≈ 0.1415 which is close to 0
        assert phi[0] > 0.0, "Angle should be > 0"
        assert phi[0] < np.pi / 4.0, "Angle should be small for high positive corr"

    def test_extreme_negative_corr(self) -> None:
        """Near-perfect negative correlation → angle near pi."""
        R = np.array([[1.0, -0.99], [-0.99, 1.0]])
        phi = r2phi(R)
        # arccos(-0.99) ≈ 3.0000 which is close to pi
        assert phi[0] < np.pi, "Angle should be < pi"
        assert phi[0] > 3.0 * np.pi / 4.0, "Angle should be close to pi for high negative corr"

    def test_various_correlation_matrices(self) -> None:
        """Test angle ranges with multiple randomly generated correlations."""
        for seed in range(10):
            for k in [3, 4, 5]:
                R = _random_corr_matrix(k, seed=seed * 100 + k)
                phi = r2phi(R)
                assert np.all(phi >= 0.0), (
                    f"Negative angle found: seed={seed}, k={k}, phi={phi}"
                )
                assert np.all(phi <= np.pi + 1e-10), (
                    f"Angle > pi found: seed={seed}, k={k}, phi={phi}"
                )


# ---------------------------------------------------------------------------
# Test 5: test_r2phi_roundtrip_phi2r — Inverse relationship verification
# ---------------------------------------------------------------------------

class TestR2PhiRoundtrip:
    """Test the roundtrip relationship between r2phi and phi2r.

    Both directions must hold:
      1. phi2r(r2phi(R)) ≈ R — start with correlation, recover it
      2. r2phi(phi2r(angles)) ≈ angles — start with angles, recover them

    Ref: r2phi.m:16 — "See also PHI2R" documenting the inverse relationship.
    """

    def test_roundtrip_R_to_angles_to_R_3x3(self) -> None:
        """phi2r(r2phi(R)) ≈ R for known 3×3 correlation matrix."""
        R = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        phi = r2phi(R)
        R_reconstructed = phi2r(phi)
        npt.assert_allclose(
            R_reconstructed, R, atol=ATOL, rtol=RTOL,
            err_msg="phi2r(r2phi(R)) should recover original R"
        )

    def test_roundtrip_angles_to_R_to_angles_3(self) -> None:
        """r2phi(phi2r(angles)) ≈ angles for known 3-element angle vector."""
        angles = np.array([np.pi / 4, np.pi / 3, np.pi / 6])
        R = phi2r(angles)
        angles_recovered = r2phi(R)
        npt.assert_allclose(
            angles_recovered, angles, atol=ATOL, rtol=RTOL,
            err_msg="r2phi(phi2r(angles)) should recover original angles"
        )

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_roundtrip_R_to_angles_to_R_random(self, k: int) -> None:
        """phi2r(r2phi(R)) ≈ R for random K×K correlation matrices."""
        R = _random_corr_matrix(k, seed=300 + k)
        phi = r2phi(R)
        R_reconstructed = phi2r(phi)
        npt.assert_allclose(
            R_reconstructed, R, atol=ATOL, rtol=RTOL,
            err_msg=f"phi2r(r2phi(R)) roundtrip failed for K={k}"
        )

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_roundtrip_angles_to_R_to_angles_random(self, k: int) -> None:
        """r2phi(phi2r(angles)) ≈ angles for random angle vectors in (0, pi).

        Angles are generated strictly within (0, pi) to avoid boundary
        edge cases at 0 or pi where sine is zero.
        """
        rng = np.random.default_rng(400 + k)
        n_angles = k * (k - 1) // 2
        # Generate angles strictly in (0.1, pi-0.1) for numerical stability
        angles = rng.uniform(0.1, np.pi - 0.1, size=n_angles)
        R = phi2r(angles)
        angles_recovered = r2phi(R)
        npt.assert_allclose(
            angles_recovered, angles, atol=ATOL, rtol=RTOL,
            err_msg=f"r2phi(phi2r(angles)) roundtrip failed for K={k}"
        )

    def test_roundtrip_identity(self) -> None:
        """Roundtrip through identity: phi2r(r2phi(eye(3))) ≈ eye(3)."""
        R = np.eye(3)
        phi = r2phi(R)
        R_reconstructed = phi2r(phi)
        npt.assert_allclose(
            R_reconstructed, R, atol=ATOL, rtol=RTOL,
            err_msg="Roundtrip through identity should recover eye(3)"
        )

    def test_roundtrip_2x2_various_rho(self) -> None:
        """Roundtrip for 2×2 matrices with various correlation values."""
        rho_values = [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9]
        for rho in rho_values:
            R = np.array([[1.0, rho], [rho, 1.0]])
            phi = r2phi(R)
            R_back = phi2r(phi)
            npt.assert_allclose(
                R_back, R, atol=ATOL, rtol=RTOL,
                err_msg=f"Roundtrip failed for 2×2 with rho={rho}"
            )


# ---------------------------------------------------------------------------
# Test 6: test_r2phi_non_symmetric_raises — Non-symmetric matrix → error
# ---------------------------------------------------------------------------

class TestR2PhiErrorHandling:
    """Test that r2phi raises appropriate errors for invalid inputs.

    Ref: r2phi.py:66-72 — Input validation checks for 2-D, square, symmetric.
    Ref: r2phi.py:76 — numpy.linalg.cholesky raises LinAlgError for non-PSD.
    """

    def test_non_symmetric_raises(self) -> None:
        """Non-symmetric matrix should raise ValueError.

        Ref: r2phi.py:71-72 — checks np.allclose(R, R.T, atol=1e-10)
        """
        R = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            r2phi(R)

    def test_non_symmetric_3x3_raises(self) -> None:
        """3×3 non-symmetric matrix should raise ValueError."""
        R = np.array([
            [1.0, 0.5, 0.2],
            [0.4, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])
        with pytest.raises(ValueError, match="symmetric"):
            r2phi(R)

    def test_non_square_raises(self) -> None:
        """Non-square matrix should raise ValueError.

        Ref: r2phi.py:69-70 — checks R.shape[0] != R.shape[1]
        """
        R = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4]])
        with pytest.raises(ValueError, match="square"):
            r2phi(R)

    def test_1d_array_raises(self) -> None:
        """1-D array should raise ValueError.

        Ref: r2phi.py:67-68 — checks R.ndim != 2
        """
        with pytest.raises(ValueError, match="2-D"):
            r2phi(np.array([1.0, 0.5, 0.3]))

    def test_3d_array_raises(self) -> None:
        """3-D array should raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            r2phi(np.ones((2, 2, 2)))

    def test_non_psd_raises(self) -> None:
        """Non-positive-definite symmetric matrix should raise LinAlgError.

        Cholesky decomposition requires positive definiteness.
        Ref: r2phi.py:76 — np.linalg.cholesky(R) will raise LinAlgError.
        """
        # Symmetric but not positive definite: eigenvalues include negative
        R = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(np.linalg.LinAlgError):
            r2phi(R)

    def test_negative_definite_raises(self) -> None:
        """Negative definite matrix should raise LinAlgError."""
        R = np.array([[-1.0, 0.0], [0.0, -1.0]])
        with pytest.raises(np.linalg.LinAlgError):
            r2phi(R)

    def test_singular_matrix_raises(self) -> None:
        """Singular (rank-deficient) symmetric matrix should raise LinAlgError.

        A singular correlation matrix has at least one zero eigenvalue,
        so Cholesky will fail.
        """
        # Perfect correlation → rank 1, Cholesky fails
        R = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(np.linalg.LinAlgError):
            r2phi(R)

    def test_non_correlation_non_unit_diagonal(self) -> None:
        """Non-unit diagonal: r2phi does NOT validate diagonal values.

        The MATLAB source (r2phi.m) does not check for unit diagonal;
        it just performs Cholesky and extracts angles.  The Python
        implementation preserves this behavior — a symmetric PSD matrix
        with non-unit diagonal will succeed but produce angles that do
        not correspond to a valid correlation parameterization.

        This test verifies the function runs without error (matching MATLAB)
        rather than expecting an exception.
        """
        # Symmetric PSD matrix with diagonal != 1
        R = np.array([[2.0, 0.5], [0.5, 2.0]])
        # Should NOT raise — matches MATLAB behavior
        phi = r2phi(R)
        assert isinstance(phi, np.ndarray), "Should return ndarray"
        assert phi.shape == (1,), "2×2 should return 1 angle"


# ---------------------------------------------------------------------------
# Test 7: test_r2phi_2x2 — 2×2 matrix → 1 angle
# ---------------------------------------------------------------------------

class TestR2Phi2x2:
    """Test r2phi for 2×2 correlation matrices.

    For a 2×2 correlation matrix [[1, rho], [rho, 1]], the Cholesky factor
    (upper triangular) is [[1, rho], [0, sqrt(1-rho^2)]].

    The single angle extracted is:
        P[0,1] = arccos(X[0,1] / cumS[0,1]) = arccos(rho / 1) = arccos(rho)

    Ref: r2phi.m:28 — P(i,i+1:k) = acos(X(i,i+1:k)./cumS(i,i+1:k))
    """

    @pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.9, 0.99])
    def test_positive_rho(self, rho: float) -> None:
        """2×2 positive correlation: angle = arccos(rho)."""
        R = np.array([[1.0, rho], [rho, 1.0]])
        phi = r2phi(R)
        expected_angle = np.arccos(rho)
        assert phi.shape == (1,), "2×2 should return exactly 1 angle"
        npt.assert_allclose(
            phi[0], expected_angle, atol=ATOL, rtol=RTOL,
            err_msg=f"2×2 angle for rho={rho} should be arccos({rho})"
        )

    @pytest.mark.parametrize("rho", [-0.5, -0.75, -0.99])
    def test_negative_rho(self, rho: float) -> None:
        """2×2 negative correlation: angle = arccos(rho)."""
        R = np.array([[1.0, rho], [rho, 1.0]])
        phi = r2phi(R)
        expected_angle = np.arccos(rho)
        assert phi.shape == (1,), "2×2 should return exactly 1 angle"
        npt.assert_allclose(
            phi[0], expected_angle, atol=ATOL, rtol=RTOL,
            err_msg=f"2×2 angle for rho={rho} should be arccos({rho})"
        )

    def test_zero_correlation(self) -> None:
        """2×2 zero correlation: angle = arccos(0) = pi/2."""
        R = np.array([[1.0, 0.0], [0.0, 1.0]])
        phi = r2phi(R)
        npt.assert_allclose(
            phi[0], np.pi / 2.0, atol=ATOL, rtol=RTOL,
            err_msg="Zero correlation should give angle pi/2"
        )

    def test_2x2_csv_fixture_values(self) -> None:
        """Verify against all 2×2 test cases from MATLAB r2phi.csv fixture.

        Ref: tests/fixtures/utility/r2phi.csv — 2×2 rows.
        """
        # (rho, expected_angle) pairs from r2phi.csv
        test_cases = [
            (0.0, 1.5707963267948966e+00),      # 2x2_identity
            (0.25, 1.3181160716528180e+00),      # 2x2_r_pos_0.25
            (0.5, 1.0471975511965976e+00),       # 2x2_r_pos_0.50
            (0.9, 4.5102681179626236e-01),       # 2x2_r_pos_0.90
            (0.99, 1.4153947332442729e-01),      # 2x2_r_pos_0.99
            (-0.5, 2.0943951023931957e+00),      # 2x2_r_neg_0.50
            (-0.75, 2.4188584057763776e+00),     # 2x2_r_neg_0.75
            (-0.99, 3.0000531802653660e+00),     # 2x2_r_neg_0.99
        ]
        for rho, expected_angle in test_cases:
            R = np.array([[1.0, rho], [rho, 1.0]])
            phi = r2phi(R)
            npt.assert_allclose(
                phi[0], expected_angle, atol=ATOL, rtol=RTOL,
                err_msg=f"2×2 rho={rho}: angle mismatch with MATLAB fixture"
            )


# ---------------------------------------------------------------------------
# Test 8: test_r2phi_4x4_and_5x5 — Larger matrices
# ---------------------------------------------------------------------------

class TestR2PhiLargerMatrices:
    """Test r2phi for 4×4 and 5×5 matrices.

    For K >= 4, the column-major extraction order matters — this validates
    that the Python implementation correctly handles the MATLAB column-major
    P(P>0) extraction via ravel('F').

    Ref: r2phi.py:119 — P.ravel('F') for column-major flattening.

    Uses .npy fixture data for exact MATLAB parity (the CSV fixture element
    ordering for K >= 4 is implementation-specific). Roundtrip and structural
    properties are validated independently.
    """

    def test_4x4_output_shape_and_range(self) -> None:
        """4×4 random correlation: correct output shape and angle range."""
        R = _random_corr_matrix(4, seed=1234)
        phi = r2phi(R)
        assert phi.shape == (6,), f"4×4 should produce 6 angles, got {phi.shape}"
        assert np.all(phi >= 0.0), "Some angles < 0"
        assert np.all(phi <= np.pi + 1e-10), "Some angles > pi"

    def test_4x4_roundtrip(self) -> None:
        """4×4 roundtrip: phi2r(r2phi(R)) ≈ R."""
        R = np.array([
            [1.0, 0.3, -0.2, 0.1],
            [0.3, 1.0, 0.15, -0.1],
            [-0.2, 0.15, 1.0, 0.25],
            [0.1, -0.1, 0.25, 1.0],
        ])
        phi = r2phi(R)
        assert phi.shape == (6,)
        R_back = phi2r(phi)
        npt.assert_allclose(
            R_back, R, atol=ATOL, rtol=RTOL,
            err_msg="4×4 roundtrip phi2r(r2phi(R)) failed"
        )

    def test_5x5_output_shape_and_range(self) -> None:
        """5×5 random correlation: correct output shape and angle range."""
        R = _random_corr_matrix(5, seed=5678)
        phi = r2phi(R)
        assert phi.shape == (10,), f"5×5 should produce 10 angles, got {phi.shape}"
        assert np.all(phi >= 0.0), "Some angles < 0"
        assert np.all(phi <= np.pi + 1e-10), "Some angles > pi"

    def test_5x5_roundtrip(self) -> None:
        """5×5 roundtrip: phi2r(r2phi(R)) ≈ R."""
        R = np.array([
            [1.0, 0.2, -0.1, 0.15, 0.05],
            [0.2, 1.0, 0.12, -0.05, 0.1],
            [-0.1, 0.12, 1.0, 0.08, -0.03],
            [0.15, -0.05, 0.08, 1.0, 0.18],
            [0.05, 0.1, -0.03, 0.18, 1.0],
        ])
        phi = r2phi(R)
        assert phi.shape == (10,)
        R_back = phi2r(phi)
        npt.assert_allclose(
            R_back, R, atol=ATOL, rtol=RTOL,
            err_msg="5×5 roundtrip phi2r(r2phi(R)) failed"
        )

    def test_6x6_structure(self) -> None:
        """6×6: verify correct shape K(K-1)/2 = 15 and roundtrip."""
        R = _random_corr_matrix(6, seed=9999)
        phi = r2phi(R)
        assert phi.shape == (15,), f"6×6 should produce 15 angles, got {phi.shape}"
        assert np.all(phi >= 0.0) and np.all(phi <= np.pi + 1e-10)
        # Roundtrip
        R_back = phi2r(phi)
        npt.assert_allclose(
            R_back, R, atol=ATOL, rtol=RTOL,
            err_msg="6×6 roundtrip phi2r(r2phi(R)) failed"
        )


# ---------------------------------------------------------------------------
# Test 9: test_r2phi_fixture_parity — MATLAB .npy fixture comparison
# ---------------------------------------------------------------------------

# Determine if fixture files are available
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "utility"
_R2PHI_NPY_EXISTS = (_FIXTURE_DIR / "r2phi.npy").exists()


@pytest.mark.skipif(
    not _R2PHI_NPY_EXISTS,
    reason="Fixture file tests/fixtures/utility/r2phi.npy not found"
)
class TestR2PhiFixtureParity:
    """MATLAB fixture parity tests for r2phi.

    Loads the r2phi.npy fixture file containing MATLAB-generated reference
    correlation matrices and their angle decompositions. Verifies that the
    Python implementation matches MATLAB output to within atol=1e-6, rtol=1e-4.

    Fixture structure (dict):
      - R3, phi3, R3_roundtrip: 3×3 test case
      - R4, phi4, R4_roundtrip: 4×4 test case
      - R5, phi5, R5_roundtrip: 5×5 test case
    """

    @pytest.fixture(autouse=True)
    def _load_fixtures(self, utility_fixture_dir: Path) -> None:
        """Load the r2phi.npy fixture data."""
        self._data = load_fixture_npy(utility_fixture_dir, "r2phi")
        # Unwrap 0-d object array if needed
        if self._data.ndim == 0:
            self._data = self._data.item()

    def test_3x3_parity(self) -> None:
        """3×3: Python r2phi(R3) matches MATLAB phi3."""
        R3 = self._data["R3"]
        phi3_expected = self._data["phi3"]
        phi3_actual = r2phi(R3)
        npt.assert_allclose(
            phi3_actual, phi3_expected, atol=ATOL, rtol=RTOL,
            err_msg="3×3 fixture parity: r2phi(R3) != MATLAB phi3"
        )

    def test_4x4_parity(self) -> None:
        """4×4: Python r2phi(R4) matches MATLAB phi4."""
        R4 = self._data["R4"]
        phi4_expected = self._data["phi4"]
        phi4_actual = r2phi(R4)
        npt.assert_allclose(
            phi4_actual, phi4_expected, atol=ATOL, rtol=RTOL,
            err_msg="4×4 fixture parity: r2phi(R4) != MATLAB phi4"
        )

    def test_5x5_parity(self) -> None:
        """5×5: Python r2phi(R5) matches MATLAB phi5."""
        R5 = self._data["R5"]
        phi5_expected = self._data["phi5"]
        phi5_actual = r2phi(R5)
        npt.assert_allclose(
            phi5_actual, phi5_expected, atol=ATOL, rtol=RTOL,
            err_msg="5×5 fixture parity: r2phi(R5) != MATLAB phi5"
        )

    def test_3x3_roundtrip_parity(self) -> None:
        """3×3: phi2r(r2phi(R3)) matches MATLAB R3_roundtrip."""
        R3 = self._data["R3"]
        R3_roundtrip_expected = self._data["R3_roundtrip"]
        phi3 = r2phi(R3)
        R3_roundtrip_actual = phi2r(phi3)
        npt.assert_allclose(
            R3_roundtrip_actual, R3_roundtrip_expected, atol=ATOL, rtol=RTOL,
            err_msg="3×3 roundtrip parity: phi2r(r2phi(R3)) != MATLAB R3_roundtrip"
        )

    def test_4x4_roundtrip_parity(self) -> None:
        """4×4: phi2r(r2phi(R4)) matches MATLAB R4_roundtrip."""
        R4 = self._data["R4"]
        R4_roundtrip_expected = self._data["R4_roundtrip"]
        phi4 = r2phi(R4)
        R4_roundtrip_actual = phi2r(phi4)
        npt.assert_allclose(
            R4_roundtrip_actual, R4_roundtrip_expected, atol=ATOL, rtol=RTOL,
            err_msg="4×4 roundtrip parity: phi2r(r2phi(R4)) != MATLAB R4_roundtrip"
        )

    def test_5x5_roundtrip_parity(self) -> None:
        """5×5: phi2r(r2phi(R5)) matches MATLAB R5_roundtrip."""
        R5 = self._data["R5"]
        R5_roundtrip_expected = self._data["R5_roundtrip"]
        phi5 = r2phi(R5)
        R5_roundtrip_actual = phi2r(phi5)
        npt.assert_allclose(
            R5_roundtrip_actual, R5_roundtrip_expected, atol=ATOL, rtol=RTOL,
            err_msg="5×5 roundtrip parity: phi2r(r2phi(R5)) != MATLAB R5_roundtrip"
        )


# ---------------------------------------------------------------------------
# Test 10: test_r2phi_return_type — Return type validation
# ---------------------------------------------------------------------------

class TestR2PhiReturnType:
    """Test that r2phi always returns numpy.ndarray with correct dtype."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_return_type_is_ndarray(self, k: int) -> None:
        """Return value must be numpy.ndarray for all matrix sizes."""
        R = _random_corr_matrix(k, seed=500 + k)
        phi = r2phi(R)
        assert isinstance(phi, np.ndarray), (
            f"Expected numpy.ndarray return type, got {type(phi)}"
        )

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_return_dtype_is_float(self, k: int) -> None:
        """Return array dtype should be float64."""
        R = _random_corr_matrix(k, seed=600 + k)
        phi = r2phi(R)
        assert phi.dtype == np.float64, (
            f"Expected float64 dtype, got {phi.dtype}"
        )

    def test_return_is_1d(self) -> None:
        """Return array should be 1-D."""
        R = _random_corr_matrix(4, seed=700)
        phi = r2phi(R)
        assert phi.ndim == 1, f"Expected 1-D array, got {phi.ndim}-D"


# ---------------------------------------------------------------------------
# Test 11: test_r2phi_sample_corr_matrix — Using shared fixture
# ---------------------------------------------------------------------------

def test_r2phi_sample_corr_matrix(sample_corr_matrix: np.ndarray) -> None:
    """Test r2phi with the shared 4×4 sample_corr_matrix from local conftest.

    Verifies that r2phi produces a valid angle vector for the deterministic
    correlation matrix fixture and that the roundtrip recovers the original.
    """
    R = sample_corr_matrix
    k = R.shape[0]
    expected_length = k * (k - 1) // 2

    phi = r2phi(R)
    assert phi.shape == (expected_length,), (
        f"Expected {expected_length} angles for 4×4, got {phi.shape}"
    )
    # All angles in valid range
    assert np.all(phi >= 0.0), "Some angles < 0"
    assert np.all(phi <= np.pi + 1e-10), "Some angles > pi"

    # Roundtrip: phi2r(r2phi(R)) ≈ R
    R_back = phi2r(phi)
    npt.assert_allclose(
        R_back, R, atol=ATOL, rtol=RTOL,
        err_msg="Roundtrip phi2r(r2phi(sample_corr_matrix)) failed"
    )
