"""Pytest test suite for mfe_toolbox.utility.r2z — Fisher z-transform of correlation matrices.

Tests the conversion of K×K correlation matrices to K(K-1)/2 unconstrained
values via Cholesky decomposition and Fisher z-transform.

The r2z function implements the mapping:
    C2 = chol(R)  (lower-triangular Cholesky factor)
    C  = partial-correlation extraction from C2 (row-wise normalization)
    z  = log((C+1)/(1-C))  (Fisher z-transform on upper-triangular elements)

Source reference: utility/r2z.m (45 lines, Kevin Sheppard)
Numerical parity: atol=1e-6, rtol=1e-4 per AAP Section 0.7.1
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.r2z import r2z
from mfe_toolbox.utility.z2r import z2r
from tests.conftest import ATOL, RTOL, load_fixture_npy


# ---------------------------------------------------------------------------
# Test 1: test_r2z_identity — Identity matrix → all zeros
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_r2z_identity(k: int) -> None:
    """Identity correlation matrix should produce all-zero z-vector.

    Ref: r2z.m:25-37 — When R is an identity matrix, the Cholesky factor
    is also the identity. The partial correlations extracted are all zero,
    and Fisher z-transform of 0 is log((1+0)/(1-0)) = log(1) = 0.
    """
    R = np.eye(k)
    z = r2z(R)
    expected_length = k * (k - 1) // 2
    expected = np.zeros(expected_length)

    assert z.shape == (expected_length,), (
        f"Expected shape ({expected_length},) for K={k}, got {z.shape}"
    )
    npt.assert_allclose(
        z, expected, atol=ATOL, rtol=RTOL,
        err_msg=f"Identity K={k} should produce all-zero z-vector"
    )


# ---------------------------------------------------------------------------
# Test 2: test_r2z_3x3 — Known 3×3 correlation → known z-values
# ---------------------------------------------------------------------------
def test_r2z_3x3() -> None:
    """Known 3×3 correlation matrix should produce known z-values.

    Uses the 3x3_positive_corr test case from the MATLAB-generated CSV
    fixture.  The correlation matrix has off-diagonal elements
    R[0,1]=0.5, R[0,2]=0.3, R[1,2]=0.4 and the expected z-values are
    obtained from the MATLAB r2z function.
    """
    # Ref: r2z.csv — 3x3_positive_corr row
    # Off-diagonal elements in column-major upper-triangle order:
    #   R[0,1]=0.5, R[0,2]=0.3, R[1,2]=0.4
    R = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ])
    z = r2z(R)

    # Expected values from MATLAB r2z.csv fixture
    expected = np.array([
        1.0986122886681098e+00,
        6.4848903911667444e-01,
        5.9424070333690115e-01,
    ])

    assert z.shape == (3,), f"Expected shape (3,) for 3×3, got {z.shape}"
    npt.assert_allclose(
        z, expected, atol=ATOL, rtol=RTOL,
        err_msg="3×3 positive correlation z-values do not match MATLAB fixture"
    )


# ---------------------------------------------------------------------------
# Test 3: test_r2z_output_length — K×K → K(K-1)/2 elements
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 7])
def test_r2z_output_length(k: int) -> None:
    """Output vector length must equal K(K-1)/2 for K×K input.

    Ref: r2z.m:36 — Extraction of strictly upper triangular elements from
    a K×K matrix yields exactly K(K-1)/2 values.
    """
    # Construct a valid correlation matrix via random generation
    rng = np.random.default_rng(42 + k)
    A = rng.standard_normal((k, k))
    S = A @ A.T + 0.1 * np.eye(k)
    d = np.sqrt(np.diag(S))
    R = S / np.outer(d, d)
    np.fill_diagonal(R, 1.0)

    z = r2z(R)
    expected_length = k * (k - 1) // 2
    assert len(z) == expected_length, (
        f"Expected {expected_length} elements for K={k}, got {len(z)}"
    )


# ---------------------------------------------------------------------------
# Test 4: test_r2z_unconstrained_range — Output can be any real number
# ---------------------------------------------------------------------------
def test_r2z_unconstrained_range() -> None:
    """Output values should be finite and unbounded (not restricted to [-1,1]).

    The Fisher z-transform maps (-1,1) to (-inf,inf), so outputs can be any
    finite real number.  Near-perfect correlations produce large magnitudes.
    """
    # High positive correlation → large positive z
    R_high = np.array([[1.0, 0.99], [0.99, 1.0]])
    z_high = r2z(R_high)
    assert np.all(np.isfinite(z_high)), "z-values must be finite"
    assert np.abs(z_high[0]) > 2.0, (
        "Near-perfect correlation (0.99) should produce |z| > 2"
    )

    # Low correlation → small z
    R_low = np.array([[1.0, 0.01], [0.01, 1.0]])
    z_low = r2z(R_low)
    assert np.all(np.isfinite(z_low)), "z-values must be finite for small correlations"
    assert np.abs(z_low[0]) < 0.5, "Small correlation should produce small z"

    # Negative correlation → negative z
    R_neg = np.array([[1.0, -0.8], [-0.8, 1.0]])
    z_neg = r2z(R_neg)
    assert np.all(np.isfinite(z_neg)), "z-values must be finite for negative correlations"
    assert z_neg[0] < -1.0, "Strong negative correlation should produce z < -1"

    # Moderate 3×3 case
    R_mod = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ])
    z_mod = r2z(R_mod)
    assert np.all(np.isfinite(z_mod)), "All z-values must be finite for 3×3"


# ---------------------------------------------------------------------------
# Test 5: test_r2z_roundtrip_z2r — z2r(r2z(R)) ≈ R AND r2z(z2r(z)) ≈ z
# ---------------------------------------------------------------------------
def test_r2z_roundtrip_z2r() -> None:
    """Roundtrip: z2r(r2z(R)) ≈ R for valid R, AND r2z(z2r(z)) ≈ z for valid z.

    Tests that r2z and z2r are true inverses of each other in both directions.
    """
    # Direction 1: z2r(r2z(R)) ≈ R
    # Use several valid correlation matrices of different sizes
    rng = np.random.default_rng(99)
    for k in [2, 3, 4, 5]:
        A = rng.standard_normal((k, k))
        S = A @ A.T + 0.1 * np.eye(k)
        d = np.sqrt(np.diag(S))
        R_orig = S / np.outer(d, d)
        np.fill_diagonal(R_orig, 1.0)

        z = r2z(R_orig)
        R_recovered = z2r(z)
        npt.assert_allclose(
            R_recovered, R_orig, atol=ATOL, rtol=RTOL,
            err_msg=f"z2r(r2z(R)) roundtrip failed for K={k}"
        )

    # Direction 2: r2z(z2r(z)) ≈ z
    z_test_vectors = [
        np.array([0.5]),                             # K=2
        np.array([0.5, -0.3, 1.2]),                  # K=3
        np.array([0.1, -0.5, 0.8, -0.2, 1.5, -1.0]),  # K=4
    ]
    for z_original in z_test_vectors:
        R_from_z = z2r(z_original)
        z_recovered = r2z(R_from_z)
        npt.assert_allclose(
            z_recovered, z_original, atol=ATOL, rtol=RTOL,
            err_msg=(
                f"r2z(z2r(z)) roundtrip failed for "
                f"{len(z_original)}-element z-vector"
            )
        )


# ---------------------------------------------------------------------------
# Test 6: test_r2z_high_correlation — Near-perfect correlation → large z
# ---------------------------------------------------------------------------
def test_r2z_high_correlation() -> None:
    """Near-perfect positive correlation (0.99) should produce a large z-value.

    Ref: r2z.m:37 — Fisher z-transform: z = log((1+r)/(1-r)).
    For r=0.99: z = log(199) ≈ 5.293.  For 2×2 matrices, the partial
    correlation equals the raw correlation, so the z-value matches the
    Fisher z-transform directly.
    """
    # Ref: r2z.csv — 2x2_r_pos_0.99 row
    R = np.array([[1.0, 0.99], [0.99, 1.0]])
    z = r2z(R)

    # Analytical Fisher z-transform value
    expected_z = np.log((1.0 + 0.99) / (1.0 - 0.99))
    # CSV reference: 5.2933048247244914
    assert z[0] > 2.0, f"z for r=0.99 should be large, got {z[0]}"
    npt.assert_allclose(
        z[0], expected_z, atol=ATOL, rtol=RTOL,
        err_msg="High correlation z-value does not match Fisher z formula"
    )

    # Also test 0.90
    R_90 = np.array([[1.0, 0.90], [0.90, 1.0]])
    z_90 = r2z(R_90)
    expected_z_90 = np.log((1.0 + 0.90) / (1.0 - 0.90))
    npt.assert_allclose(
        z_90[0], expected_z_90, atol=ATOL, rtol=RTOL,
        err_msg="z for r=0.90 does not match Fisher z formula"
    )


# ---------------------------------------------------------------------------
# Test 7: test_r2z_zero_correlation — Zero off-diagonal → z ≈ 0
# ---------------------------------------------------------------------------
def test_r2z_zero_correlation() -> None:
    """Zero off-diagonal correlation elements should produce z ≈ 0.

    Ref: r2z.m:37 — Fisher z-transform of 0: log((1+0)/(1-0)) = log(1) = 0.
    """
    # 2×2 zero correlation
    R_2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    z_2 = r2z(R_2)
    npt.assert_allclose(
        z_2, np.zeros(1), atol=ATOL, rtol=RTOL,
        err_msg="2×2 zero correlation should produce z=0"
    )

    # 3×3 zero correlation (identity)
    R_3 = np.eye(3)
    z_3 = r2z(R_3)
    npt.assert_allclose(
        z_3, np.zeros(3), atol=ATOL, rtol=RTOL,
        err_msg="3×3 zero correlation (identity) should produce z=0 vector"
    )

    # 4×4 zero correlation (identity)
    R_4 = np.eye(4)
    z_4 = r2z(R_4)
    npt.assert_allclose(
        z_4, np.zeros(6), atol=ATOL, rtol=RTOL,
        err_msg="4×4 zero correlation (identity) should produce z=0 vector"
    )


# ---------------------------------------------------------------------------
# Test 8: test_r2z_negative_correlation — Negative correlation → negative z
# ---------------------------------------------------------------------------
def test_r2z_negative_correlation() -> None:
    """Negative correlation should produce negative z-values.

    Ref: r2z.m:37 — For r < 0, (1+r)/(1-r) < 1, so log(...) < 0.
    """
    # 2×2 negative correlation r = -0.5
    R_neg_50 = np.array([[1.0, -0.5], [-0.5, 1.0]])
    z_neg_50 = r2z(R_neg_50)
    assert z_neg_50[0] < 0, (
        f"z for r=-0.5 should be negative, got {z_neg_50[0]}"
    )
    # CSV reference: -1.0986122886681098
    expected_neg_50 = np.log((1.0 - 0.5) / (1.0 + 0.5))
    npt.assert_allclose(
        z_neg_50[0], expected_neg_50, atol=ATOL, rtol=RTOL,
        err_msg="z for r=-0.5 does not match Fisher z formula"
    )

    # 2×2 strong negative correlation r = -0.75
    R_neg_75 = np.array([[1.0, -0.75], [-0.75, 1.0]])
    z_neg_75 = r2z(R_neg_75)
    assert z_neg_75[0] < z_neg_50[0], (
        "Stronger negative correlation should produce more negative z"
    )
    expected_neg_75 = np.log((1.0 - 0.75) / (1.0 + 0.75))
    npt.assert_allclose(
        z_neg_75[0], expected_neg_75, atol=ATOL, rtol=RTOL,
        err_msg="z for r=-0.75 does not match Fisher z formula"
    )

    # 2×2 extreme negative correlation r = -0.99
    R_neg_99 = np.array([[1.0, -0.99], [-0.99, 1.0]])
    z_neg_99 = r2z(R_neg_99)
    assert z_neg_99[0] < -2.0, (
        f"z for r=-0.99 should be strongly negative, got {z_neg_99[0]}"
    )


# ---------------------------------------------------------------------------
# Test 9: test_r2z_non_correlation_raises — Invalid input → error
# ---------------------------------------------------------------------------
def test_r2z_non_correlation_raises() -> None:
    """Invalid inputs should raise appropriate exceptions.

    Ref: r2z.py — Validates 2D square input; non-PD causes Cholesky failure.
    """
    # Non-square matrix → ValueError
    with pytest.raises(ValueError, match="square"):
        r2z(np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4]]))

    # 1D array → ValueError
    with pytest.raises(ValueError, match="2-dimensional"):
        r2z(np.array([1.0, 0.5, 1.0]))

    # Not positive definite (off-diagonal magnitude > 1) → LinAlgError
    bad_R = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        r2z(bad_R)

    # 3D array → ValueError
    with pytest.raises(ValueError, match="2-dimensional"):
        r2z(np.ones((3, 3, 3)))

    # Negative definite matrix → LinAlgError
    neg_def = np.array([[1.0, 1.5], [1.5, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        r2z(neg_def)


# ---------------------------------------------------------------------------
# Test 10: test_r2z_2x2 — 2×2 → 1 z-value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r,expected_z", [
    (0.25, 5.1082562376599072e-01),
    (0.50, 1.0986122886681098e+00),
    (0.90, 2.9444389791664407e+00),
    (-0.50, -1.0986122886681098e+00),
    (-0.75, -1.9459101490553135e+00),
])
def test_r2z_2x2(r: float, expected_z: float) -> None:
    """2×2 correlation matrix should produce exactly 1 z-value.

    Ref: r2z.csv — uses MATLAB-generated reference z-values for 2×2 inputs.
    For 2×2 matrices the partial correlation equals the raw correlation,
    so the output is exactly the Fisher z-transform of the single
    off-diagonal element.
    """
    R = np.array([[1.0, r], [r, 1.0]])
    z = r2z(R)

    assert z.shape == (1,), f"Expected shape (1,) for 2×2, got {z.shape}"
    npt.assert_allclose(
        z[0], expected_z, atol=ATOL, rtol=RTOL,
        err_msg=f"2×2 z-value for r={r} does not match MATLAB reference"
    )


# ---------------------------------------------------------------------------
# Test 11: test_r2z_fisher_transform — z = log((1+r)/(1-r)) for 2×2
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r", [
    0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99,
    -0.1, -0.25, -0.5, -0.75, -0.9, -0.99,
])
def test_r2z_fisher_transform(r: float) -> None:
    """Verify z = log((1+r)/(1-r)) relationship for 2×2 correlation matrices.

    For 2×2 matrices R = [[1, r], [r, 1]], the Cholesky decomposition yields
    a partial correlation equal to r itself.  The Fisher z-transform then
    gives z = log((1+r)/(1-r)).

    Ref: r2z.m:37 — Fisher z-transform formula.
    """
    R = np.array([[1.0, r], [r, 1.0]])
    z = r2z(R)

    # Analytical Fisher z-transform
    expected = np.log((1.0 + r) / (1.0 - r))
    npt.assert_allclose(
        z[0], expected, atol=ATOL, rtol=RTOL,
        err_msg=(
            f"Fisher z-transform mismatch for r={r}: "
            f"got {z[0]}, expected {expected}"
        )
    )


# ---------------------------------------------------------------------------
# Test 12: test_r2z_fixture_parity — MATLAB fixture comparison
# ---------------------------------------------------------------------------
@pytest.mark.parity
@pytest.mark.requires_fixtures
def test_r2z_fixture_parity(utility_fixture_dir) -> None:
    """Verify r2z output against MATLAB-generated fixture data.

    Loads the r2z.npy fixture produced by scripts/generate_fixtures.m and
    scripts/convert_fixtures.py, then compares Python r2z output against
    the MATLAB reference for correlation matrices of size 3×3, 4×4, 5×5,
    and a near-identity 3×3 matrix.

    Per AAP Section 0.7.1: atol=1e-6, rtol=1e-4.
    """
    fixture = load_fixture_npy(utility_fixture_dir, "r2z")
    # The fixture is a 0-d object array containing a dict
    data = fixture.item()

    # Define R/z pairs to test
    test_cases = [
        ("R3", "z3", "3×3 random correlation"),
        ("R4", "z4", "4×4 random correlation"),
        ("R5", "z5", "5×5 random correlation"),
        ("R3_identity_like", "z3_identity_like", "3×3 near-identity correlation"),
    ]

    for r_key, z_key, description in test_cases:
        R_input = data[r_key]
        expected_z = data[z_key]

        actual_z = r2z(R_input)
        npt.assert_allclose(
            actual_z, expected_z, atol=ATOL, rtol=RTOL,
            err_msg=(
                f"MATLAB parity failure for {description} "
                f"({r_key} → {z_key})"
            )
        )

    # Additionally verify roundtrip from fixture: z2r(r2z(R)) ≈ R_roundtrip
    roundtrip_pairs = [
        ("R3", "R3_roundtrip", "3×3"),
        ("R4", "R4_roundtrip", "4×4"),
        ("R5", "R5_roundtrip", "5×5"),
        ("R3_identity_like", "R3_identity_like_roundtrip", "3×3 near-identity"),
    ]

    for r_key, rt_key, description in roundtrip_pairs:
        R_input = data[r_key]
        R_roundtrip_expected = data[rt_key]

        z_computed = r2z(R_input)
        R_roundtrip_actual = z2r(z_computed)
        npt.assert_allclose(
            R_roundtrip_actual, R_roundtrip_expected, atol=ATOL, rtol=RTOL,
            err_msg=(
                f"Roundtrip parity failure for {description}: "
                f"z2r(r2z({r_key})) != {rt_key}"
            )
        )
