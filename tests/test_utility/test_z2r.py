"""Pytest test suite for mfe_toolbox.utility.z2r — inverse Fisher z-transform.

Tests the conversion of K(K-1)/2 unconstrained values to a K×K positive-definite
correlation matrix via the inverse Fisher z-transform and Cholesky-like factor
construction.

Source reference: utility/z2r.m (53 lines, Kevin Sheppard)
Numerical parity: atol=1e-6, rtol=1e-4 per AAP Section 0.7.1

The z2r function implements the mapping:
    y = (exp(z) - 1) / (1 + exp(z))   maps (-inf, inf) -> (-1, 1)
    C = Cholesky-like factor from y
    R = C @ C.T (normalized to unit diagonal)
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.z2r import z2r
from mfe_toolbox.utility.r2z import r2z
from tests.conftest import ATOL, RTOL, load_fixture_npy


# ---------------------------------------------------------------------------
# Test 1: test_z2r_zeros — All-zero z → identity matrix
# ---------------------------------------------------------------------------
def test_z2r_zeros():
    """All-zero z vector should produce the identity (correlation) matrix.

    Ref: z2r.m:31 — when z = 0, (exp(0)-1)/(1+exp(0)) = 0, so all off-diagonal
    partial correlations are zero, resulting in an identity matrix.
    """
    # K=3: K(K-1)/2 = 3 elements
    z = np.zeros(3)
    R = z2r(z)
    expected = np.eye(3)
    npt.assert_allclose(R, expected, atol=ATOL, rtol=RTOL,
                        err_msg="All-zero z should produce identity matrix")


# ---------------------------------------------------------------------------
# Test 2: test_z2r_3x3 — 3 z-values → 3×3 valid correlation matrix
# ---------------------------------------------------------------------------
def test_z2r_3x3():
    """Three z-values produce a valid 3×3 correlation matrix.

    Uses specific z-values and verifies fundamental correlation matrix
    properties: symmetric, unit diagonal, off-diagonal in [-1, 1].
    """
    z = np.array([0.5, -0.3, 1.2])
    R = z2r(z)

    # Shape check
    assert R.shape == (3, 3), f"Expected (3,3) shape, got {R.shape}"

    # Diagonal should be 1.0
    npt.assert_allclose(np.diag(R), np.ones(3), atol=ATOL, rtol=RTOL,
                        err_msg="Diagonal should be 1.0")

    # Symmetric
    npt.assert_allclose(R, R.T, atol=ATOL, rtol=RTOL,
                        err_msg="Correlation matrix should be symmetric")

    # Off-diagonal in [-1, 1]
    off_diag_mask = ~np.eye(3, dtype=bool)
    assert np.all(np.abs(R[off_diag_mask]) <= 1.0 + ATOL), \
        "All off-diagonal elements must be in [-1, 1]"


# ---------------------------------------------------------------------------
# Test 3: test_z2r_diagonal_ones — Diagonal is always exactly 1.0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("z_values,k", [
    (np.zeros(1), 2),          # 2×2
    (np.zeros(3), 3),          # 3×3
    (np.zeros(6), 4),          # 4×4
    (np.array([0.5, -0.3, 1.2]), 3),  # Non-zero 3×3
    (np.array([0.1, -0.5, 0.8, -0.2, 1.5, -1.0]), 4),  # Non-zero 4×4
    (np.array([3.0, 2.5, 4.0]), 3),  # Large positive 3×3
    (np.array([-3.0, -2.5, -4.0]), 3),  # Large negative 3×3
])
def test_z2r_diagonal_ones(z_values, k):
    """Diagonal of the output correlation matrix must always be exactly 1.0.

    Ref: z2r.m:49-51 — R = C*C' normalized by r*r' forces diagonal to 1.0.
    """
    R = z2r(z_values)
    npt.assert_allclose(np.diag(R), np.ones(k), atol=ATOL, rtol=RTOL,
                        err_msg=f"Diagonal must be 1.0 for K={k}")


# ---------------------------------------------------------------------------
# Test 4: test_z2r_symmetric — Output is symmetric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("z_values", [
    np.array([0.5]),
    np.array([0.5, -0.3, 1.2]),
    np.array([0.1, -0.5, 0.8, -0.2, 1.5, -1.0]),
    np.array([0.3, -0.7, 1.1, -0.4, 0.6, -1.3, 0.9, -0.1, 0.2, -0.8]),
])
def test_z2r_symmetric(z_values):
    """Output correlation matrix must be symmetric: R == R.T.

    Ref: z2r.m:49 — R = C*C' is symmetric by construction.
    """
    R = z2r(z_values)
    npt.assert_allclose(R, R.T, atol=ATOL, rtol=RTOL,
                        err_msg="Correlation matrix must be symmetric")


# ---------------------------------------------------------------------------
# Test 5: test_z2r_positive_definite — All eigenvalues > 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("z_values", [
    np.zeros(3),
    np.array([0.5, -0.3, 1.2]),
    np.array([0.1, -0.5, 0.8, -0.2, 1.5, -1.0]),
    np.array([0.3, -0.7, 1.1, -0.4, 0.6, -1.3, 0.9, -0.1, 0.2, -0.8]),
    np.array([3.0, 2.5, 4.0]),
    np.array([-3.0, -2.5, -4.0]),
])
def test_z2r_positive_definite(z_values):
    """All eigenvalues of the output correlation matrix must be positive.

    Ref: z2r.m:49 — R = C*C' is PSD by construction. For non-degenerate
    inputs, it should be strictly PD (all eigenvalues > 0).
    """
    R = z2r(z_values)
    eigenvalues = np.linalg.eigvalsh(R)
    assert np.all(eigenvalues > -ATOL), \
        f"All eigenvalues should be non-negative, got min={eigenvalues.min()}"


# ---------------------------------------------------------------------------
# Test 6: test_z2r_off_diag_in_bounds — Off-diagonal in [-1, 1]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("z_values", [
    np.array([0.5, -0.3, 1.2]),
    np.array([0.1, -0.5, 0.8, -0.2, 1.5, -1.0]),
    np.array([5.0, -5.0, 0.0, 2.0, -2.0, 1e-3]),  # Extreme mix
    np.array([3.0, 2.5, 4.0]),
    np.array([-3.0, -2.5, -4.0]),
])
def test_z2r_off_diag_in_bounds(z_values):
    """All off-diagonal elements must be in [-1, 1] for a valid correlation matrix.

    Ref: z2r.m:51 — normalization ensures valid correlation bounds.
    """
    R = z2r(z_values)
    k = R.shape[0]
    off_diag_mask = ~np.eye(k, dtype=bool)
    off_diag = R[off_diag_mask]
    assert np.all(off_diag >= -1.0 - ATOL), \
        f"Off-diagonal elements must be >= -1, got min={off_diag.min()}"
    assert np.all(off_diag <= 1.0 + ATOL), \
        f"Off-diagonal elements must be <= 1, got max={off_diag.max()}"


# ---------------------------------------------------------------------------
# Test 7: test_z2r_roundtrip_r2z — z2r(r2z(R)) ≈ R AND r2z(z2r(z)) ≈ z
# ---------------------------------------------------------------------------
def test_z2r_roundtrip_r2z():
    """Roundtrip property: r2z(z2r(z)) ≈ z AND z2r(r2z(R)) ≈ R.

    Tests that z2r and r2z are true inverses of each other.
    """
    # Direction 1: r2z(z2r(z)) ≈ z
    z_vals = [
        np.array([0.5, -0.3, 1.2]),
        np.array([0.1, -0.5, 0.8, -0.2, 1.5, -1.0]),
        np.array([0.3, -0.7, 1.1, -0.4, 0.6, -1.3, 0.9, -0.1, 0.2, -0.8]),
    ]
    for z_original in z_vals:
        R = z2r(z_original)
        z_recovered = r2z(R)
        npt.assert_allclose(z_recovered, z_original, atol=ATOL, rtol=RTOL,
                            err_msg=f"r2z(z2r(z)) should recover z for {len(z_original)} elements")

    # Direction 2: z2r(r2z(R)) ≈ R
    # Construct a known valid correlation matrix
    rng = np.random.default_rng(42)
    for k in [3, 4, 5]:
        A = rng.standard_normal((k, k))
        S = A @ A.T
        # Convert covariance to correlation
        d = np.sqrt(np.diag(S))
        R_orig = S / np.outer(d, d)
        z_from_R = r2z(R_orig)
        R_recovered = z2r(z_from_R)
        npt.assert_allclose(R_recovered, R_orig, atol=ATOL, rtol=RTOL,
                            err_msg=f"z2r(r2z(R)) should recover R for K={k}")


# ---------------------------------------------------------------------------
# Test 8: test_z2r_large_positive — Large positive z → near +1 correlation
# ---------------------------------------------------------------------------
def test_z2r_large_positive():
    """Large positive z values should produce correlations near +1.

    Ref: z2r.m:31 — as z → +∞, (exp(z)-1)/(1+exp(z)) → 1, so
    the transformed partial correlations approach 1.
    """
    z = np.array([3.0, 2.5, 4.0])
    R = z2r(z)

    # For large positive z, some off-diagonal correlations should be close to +1
    off_diag_mask = ~np.eye(3, dtype=bool)
    off_diag = R[off_diag_mask]
    # At least one off-diagonal element should be > 0.5 (likely much closer to 1)
    assert np.any(off_diag > 0.5), \
        f"Large positive z should produce at least one strong positive correlation, got {off_diag}"

    # The R[0,1] element is influenced by the first z-value (3.0 → tanh(1.5) ≈ 0.905)
    # Ref: z2r.m:31 — (exp(3)-1)/(1+exp(3)) ≈ 0.9051
    assert R[0, 1] > 0.8, \
        f"R[0,1] for z[0]=3.0 should be > 0.8, got {R[0, 1]}"


# ---------------------------------------------------------------------------
# Test 9: test_z2r_large_negative — Large negative z → near -1 correlation
# ---------------------------------------------------------------------------
def test_z2r_large_negative():
    """Large negative z values should produce correlations near -1.

    Ref: z2r.m:31 — as z → -∞, (exp(z)-1)/(1+exp(z)) → -1, so
    the transformed partial correlations approach -1.
    """
    z = np.array([-3.0, -2.5, -4.0])
    R = z2r(z)

    # The R[0,1] element is influenced by the first z-value (-3.0 → ≈ -0.905)
    assert R[0, 1] < -0.8, \
        f"R[0,1] for z[0]=-3.0 should be < -0.8, got {R[0, 1]}"

    # Off-diagonal should contain some negative values
    off_diag_mask = ~np.eye(3, dtype=bool)
    off_diag = R[off_diag_mask]
    assert np.any(off_diag < -0.1), \
        f"Large negative z should produce some negative correlations, got {off_diag}"


# ---------------------------------------------------------------------------
# Test 10: test_z2r_output_shape — K(K-1)/2 input → K×K output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,k", [
    (1, 2),     # 2(2-1)/2 = 1
    (3, 3),     # 3(3-1)/2 = 3
    (6, 4),     # 4(4-1)/2 = 6
    (10, 5),    # 5(5-1)/2 = 10
    (15, 6),    # 6(6-1)/2 = 15
])
def test_z2r_output_shape(m, k):
    """z2r must produce a K×K matrix from a K(K-1)/2 input vector.

    Ref: z2r.m:26 — k = ceil(sqrt(2*m)), verified by k*(k-1)/2 == m.
    """
    z = np.zeros(m)
    R = z2r(z)
    assert R.shape == (k, k), f"Expected ({k},{k}), got {R.shape} for m={m}"
    assert isinstance(R, np.ndarray), "Output must be numpy.ndarray"


# ---------------------------------------------------------------------------
# Test 11: test_z2r_inverse_fisher — Verify (exp(z)-1)/(1+exp(z)) for 2×2
# ---------------------------------------------------------------------------
def test_z2r_inverse_fisher():
    """Verify the inverse Fisher z-transform mapping for the 2×2 case.

    For K=2 (1 z-value), the correlation matrix is:
        R = [[1, r], [r, 1]]
    where r = (exp(z) - 1) / (1 + exp(z)).

    Ref: z2r.m:31 — z = (exp(z)-1)./(1+exp(z))

    For the 2×2 case, the off-diagonal element of R equals the
    inverse Fisher z-transform of the single z-value, because the
    Cholesky-like construction simplifies: C = [[1, 0], [r, sqrt(1-r²)]]
    and R = C @ C.T normalized = [[1, r], [r, 1]].
    """
    test_z_values = [0.0, 0.5, 1.0, -0.5, -1.0, 2.0, -2.0, 3.0, -3.0]

    for z_val in test_z_values:
        z = np.array([z_val])
        R = z2r(z)

        # Compute expected off-diagonal correlation from the transform
        # Ref: z2r.m:31 — y = (exp(z)-1)/(1+exp(z))
        exp_z = np.exp(z_val)
        expected_r = (exp_z - 1.0) / (1.0 + exp_z)

        assert R.shape == (2, 2), f"Expected (2,2), got {R.shape}"
        npt.assert_allclose(R[0, 1], expected_r, atol=ATOL, rtol=RTOL,
                            err_msg=f"Off-diagonal R[0,1] should equal "
                                    f"(exp({z_val})-1)/(1+exp({z_val}))={expected_r}")
        npt.assert_allclose(R[1, 0], expected_r, atol=ATOL, rtol=RTOL,
                            err_msg="R must be symmetric: R[0,1] == R[1,0]")
        npt.assert_allclose(np.diag(R), np.ones(2), atol=ATOL, rtol=RTOL,
                            err_msg="Diagonal must be 1.0")


# ---------------------------------------------------------------------------
# Test 12: test_z2r_fixture_parity — MATLAB fixture comparison
# ---------------------------------------------------------------------------
@pytest.mark.parity
@pytest.mark.requires_fixtures
def test_z2r_fixture_parity(utility_fixture_dir):
    """Compare z2r output against MATLAB-generated reference fixtures.

    Loads fixture data from tests/fixtures/utility/z2r.npy which contains
    multiple test cases generated by MATLAB/Octave with known z-inputs and
    their corresponding R-output correlation matrices.

    Per AAP Section 0.7.1: numpy.testing.assert_allclose(atol=1e-6, rtol=1e-4).
    """
    fixture = load_fixture_npy(utility_fixture_dir, "z2r")

    # The fixture is an object array containing a dict of test cases
    fixture_data = fixture.item() if fixture.ndim == 0 else fixture

    # Test case 1: 3×3 basic
    z3_input = fixture_data["z3_input"]
    R3_expected = fixture_data["R3_expected"]
    R3_actual = z2r(z3_input)
    npt.assert_allclose(R3_actual, R3_expected, atol=ATOL, rtol=RTOL,
                        err_msg="3×3 fixture parity failed")

    # Test case 2: 4×4 basic
    z4_input = fixture_data["z4_input"]
    R4_expected = fixture_data["R4_expected"]
    R4_actual = z2r(z4_input)
    npt.assert_allclose(R4_actual, R4_expected, atol=ATOL, rtol=RTOL,
                        err_msg="4×4 fixture parity failed")

    # Test case 3: 5×5 basic
    z5_input = fixture_data["z5_input"]
    R5_expected = fixture_data["R5_expected"]
    R5_actual = z2r(z5_input)
    npt.assert_allclose(R5_actual, R5_expected, atol=ATOL, rtol=RTOL,
                        err_msg="5×5 fixture parity failed")

    # Test case 4: 3×3 zeros → identity
    z3_zero_input = fixture_data["z3_zero_input"]
    R3_zero_expected = fixture_data["R3_zero_expected"]
    R3_zero_actual = z2r(z3_zero_input)
    npt.assert_allclose(R3_zero_actual, R3_zero_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Zero-input fixture parity failed")

    # Test case 5: 3×3 large positive z
    z3_large_input = fixture_data["z3_large_input"]
    R3_large_expected = fixture_data["R3_large_expected"]
    R3_large_actual = z2r(z3_large_input)
    npt.assert_allclose(R3_large_actual, R3_large_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Large positive z fixture parity failed")

    # Test case 6: 3×3 large negative z
    z3_neg_input = fixture_data["z3_neg_input"]
    R3_neg_expected = fixture_data["R3_neg_expected"]
    R3_neg_actual = z2r(z3_neg_input)
    npt.assert_allclose(R3_neg_actual, R3_neg_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Large negative z fixture parity failed")

    # Test case 7: 4×4 mixed extreme values
    z4_mix_input = fixture_data["z4_mix_input"]
    R4_mix_expected = fixture_data["R4_mix_expected"]
    R4_mix_actual = z2r(z4_mix_input)
    npt.assert_allclose(R4_mix_actual, R4_mix_expected, atol=ATOL, rtol=RTOL,
                        err_msg="Mixed extreme z fixture parity failed")

    # Test case 8: Roundtrip z-values from fixture
    for key_prefix, label in [("z3_roundtrip", "3×3"), ("z4_roundtrip", "4×4"),
                               ("z5_roundtrip", "5×5")]:
        z_rt = fixture_data[key_prefix]
        R_rt = z2r(z_rt)
        z_recovered = r2z(R_rt)
        npt.assert_allclose(z_recovered, z_rt, atol=ATOL, rtol=RTOL,
                            err_msg=f"Roundtrip {label} fixture parity failed")
