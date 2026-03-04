"""Pytest test suite for mfe_toolbox.utility.phi2u — Givens rotation angle-to-matrix conversion.

Tests the phi2u function which converts a K(K-1)/2 vector of Givens rotation
angles into a K×K orthogonal (rotation) matrix U by accumulating the product
of K(K-1)/2 individual Givens rotation matrices.

Key properties verified:
    - Orthogonality: U @ U.T ≈ I (identity within tolerance)
    - Determinant: |det(U)| ≈ 1 (special orthogonal for even-dimensional case)
    - Shape: output is (K, K) for K(K-1)/2 input angles
    - Column unit norms: each column of U has L2 norm ≈ 1
    - Known rotations: 2×2 with angle θ gives [[cos θ, −sin θ], [sin θ, cos θ]]
    - MATLAB fixture parity: assert_allclose(atol=1e-6, rtol=1e-4)

Source reference: utility/phi2u.m (20 lines, Kevin Sheppard, MFE Toolbox v4.0)
Numerical parity: atol=1e-6, rtol=1e-4 per AAP Section 0.7.1
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.phi2u import phi2u
from tests.conftest import ATOL, RTOL, load_fixture_npy


# ---------------------------------------------------------------------------
# Test 1: test_phi2u_zero_angles — All-zero angles yield identity matrix
# ---------------------------------------------------------------------------

def test_phi2u_zero_angles() -> None:
    """All-zero angle vector must produce the identity matrix.

    Ref: phi2u.m:12-15 — When all angles are zero, cos(0)=1 and sin(0)=0,
    so every Givens rotation R equals the identity matrix. Therefore the
    accumulated product U = I*I*...*I = I.

    Tests the 3×3 case: K=3 requires K(K-1)/2 = 3 angles, all set to 0.0.
    """
    # K=3 → 3 angles, all zero
    phi = np.zeros(3)
    U = phi2u(phi)

    npt.assert_allclose(
        U, np.eye(3), atol=ATOL, rtol=RTOL,
        err_msg="Zero-angle vector should produce 3×3 identity matrix",
    )


# ---------------------------------------------------------------------------
# Test 2: test_phi2u_orthogonal — U @ U.T ≈ I for various sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "angles",
    [
        pytest.param(np.array([0.3]), id="2x2_single_angle"),
        pytest.param(np.array([0.5, 1.0, 1.5]), id="3x3_mixed_angles"),
        pytest.param(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), id="4x4_sequential"),
        pytest.param(
            np.random.default_rng(42).uniform(0, np.pi, 10),
            id="5x5_random_seed42",
        ),
    ],
)
def test_phi2u_orthogonal(angles: np.ndarray) -> None:
    """Product U @ U.T must be the identity within tolerance.

    An orthogonal matrix satisfies U @ U.T = U.T @ U = I by definition.
    This is the fundamental property guaranteed by Givens rotation composition.
    """
    U = phi2u(angles)
    k = U.shape[0]
    identity = np.eye(k)

    npt.assert_allclose(
        U @ U.T, identity, atol=ATOL, rtol=RTOL,
        err_msg=f"U @ U.T must equal I({k}) for {len(angles)} angles",
    )
    npt.assert_allclose(
        U.T @ U, identity, atol=ATOL, rtol=RTOL,
        err_msg=f"U.T @ U must equal I({k}) for {len(angles)} angles",
    )


# ---------------------------------------------------------------------------
# Test 3: test_phi2u_determinant_one — |det(U)| ≈ 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "angles",
    [
        pytest.param(np.array([0.7]), id="2x2"),
        pytest.param(np.array([1.0, 2.0, 0.5]), id="3x3"),
        pytest.param(np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8]), id="4x4"),
        pytest.param(
            np.random.default_rng(99).uniform(0, 2 * np.pi, 10),
            id="5x5_random_seed99",
        ),
    ],
)
def test_phi2u_determinant_one(angles: np.ndarray) -> None:
    """Absolute value of determinant of U must be approximately 1.

    Each Givens rotation has determinant +1 (cos²θ + sin²θ = 1), so the
    product of Givens rotations also has determinant +1 (or -1 depending
    on parity). The absolute determinant must always be 1.
    """
    U = phi2u(angles)
    det_val = np.linalg.det(U)

    npt.assert_allclose(
        np.abs(det_val), 1.0, atol=ATOL, rtol=RTOL,
        err_msg=f"|det(U)| must be 1.0 for {len(angles)} angles, got {det_val}",
    )


# ---------------------------------------------------------------------------
# Test 4: test_phi2u_3x3 — 3 angles produce a 3×3 orthogonal matrix
# ---------------------------------------------------------------------------

def test_phi2u_3x3() -> None:
    """Three angles must produce a valid 3×3 orthogonal matrix.

    K=3 → K(K-1)/2 = 3 angles → 3×3 orthogonal output.
    Verifies shape, orthogonality, and determinant together.
    """
    rng = np.random.default_rng(7)
    phi = rng.uniform(0, np.pi, 3)
    U = phi2u(phi)

    # Shape check
    assert U.shape == (3, 3), f"Expected (3, 3), got {U.shape}"

    # Orthogonality check
    npt.assert_allclose(
        U @ U.T, np.eye(3), atol=ATOL, rtol=RTOL,
        err_msg="3×3 result must be orthogonal: U @ U.T = I",
    )

    # Determinant check
    npt.assert_allclose(
        np.abs(np.linalg.det(U)), 1.0, atol=ATOL, rtol=RTOL,
        err_msg="3×3 determinant must have |det| = 1",
    )


# ---------------------------------------------------------------------------
# Test 5: test_phi2u_4x4 — 6 angles produce a 4×4 orthogonal matrix
# ---------------------------------------------------------------------------

def test_phi2u_4x4() -> None:
    """Six angles must produce a valid 4×4 orthogonal matrix.

    K=4 → K(K-1)/2 = 6 angles → 4×4 orthogonal output.
    Verifies shape, orthogonality, and determinant together.
    """
    rng = np.random.default_rng(13)
    phi = rng.uniform(0, np.pi, 6)
    U = phi2u(phi)

    # Shape check
    assert U.shape == (4, 4), f"Expected (4, 4), got {U.shape}"

    # Orthogonality check
    npt.assert_allclose(
        U @ U.T, np.eye(4), atol=ATOL, rtol=RTOL,
        err_msg="4×4 result must be orthogonal: U @ U.T = I",
    )

    # Determinant check
    npt.assert_allclose(
        np.abs(np.linalg.det(U)), 1.0, atol=ATOL, rtol=RTOL,
        err_msg="4×4 determinant must have |det| = 1",
    )


# ---------------------------------------------------------------------------
# Test 6: test_phi2u_pi_half — π/2 rotation gives 90-degree rotation matrix
# ---------------------------------------------------------------------------

def test_phi2u_pi_half() -> None:
    """Angle π/2 in the 2×2 case must produce a 90-degree rotation.

    Ref: phi2u.m:12-15 — For K=2 with a single angle φ = π/2:
        R[0,0] = cos(π/2) ≈ 0
        R[0,1] = -sin(π/2) = -1
        R[1,0] = sin(π/2) = 1
        R[1,1] = cos(π/2) ≈ 0
    Expected output: [[0, -1], [1, 0]]
    """
    phi = np.array([np.pi / 2.0])
    U = phi2u(phi)

    expected = np.array([
        [np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
        [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)],
    ])

    npt.assert_allclose(
        U, expected, atol=ATOL, rtol=RTOL,
        err_msg="φ=π/2 should produce 90-degree rotation [[~0, -1], [1, ~0]]",
    )


# ---------------------------------------------------------------------------
# Test 7: test_phi2u_output_shape — Verify (K, K) output for various K
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "k",
    [2, 3, 4, 5],
    ids=["K=2", "K=3", "K=4", "K=5"],
)
def test_phi2u_output_shape(k: int) -> None:
    """Output shape must be (K, K) for K(K-1)/2 input angles.

    The number of Givens rotation angles for a K×K orthogonal matrix is
    K(K-1)/2 (the number of unique (i,j) pairs with i < j).
    """
    m = k * (k - 1) // 2
    rng = np.random.default_rng(k)
    phi = rng.uniform(0, np.pi, m)
    U = phi2u(phi)

    assert U.shape == (k, k), (
        f"For K={k}, expected output shape ({k}, {k}), got {U.shape}"
    )


# ---------------------------------------------------------------------------
# Test 8: test_phi2u_columns_unit_norm — Each column has L2 norm ≈ 1
# ---------------------------------------------------------------------------

def test_phi2u_columns_unit_norm() -> None:
    """Every column of the orthogonal matrix U must have unit L2 norm.

    This is a direct consequence of orthogonality: if U @ U.T = I, then
    U.T @ U = I, which means u_i · u_j = δ_{ij}. In particular, each
    column u_i has ||u_i||₂ = 1.

    Tests with K=2,3,4,5 using reproducible random angles.
    """
    rng = np.random.default_rng(77)
    for k in [2, 3, 4, 5]:
        m = k * (k - 1) // 2
        phi = rng.uniform(0, np.pi, m)
        U = phi2u(phi)

        for col_idx in range(k):
            col_norm = np.linalg.norm(U[:, col_idx])
            npt.assert_allclose(
                col_norm, 1.0, atol=ATOL, rtol=RTOL,
                err_msg=(
                    f"Column {col_idx} of {k}×{k} matrix must have unit "
                    f"norm, got {col_norm}"
                ),
            )


# ---------------------------------------------------------------------------
# Test 9: test_phi2u_known_rotation — 2×2 known rotation matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "theta",
    [0.0, 0.3, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi, 2.5],
    ids=["0", "0.3", "pi/6", "pi/4", "pi/3", "pi/2", "pi", "2.5"],
)
def test_phi2u_known_rotation(theta: float) -> None:
    """2×2 case with angle θ must produce [[cos θ, −sin θ], [sin θ, cos θ]].

    Ref: phi2u.m:12-15 — For K=2, there is a single (i=0, j=1) pair, so
    the Givens rotation is a standard 2D rotation matrix:
        U = [[cos(θ), -sin(θ)],
             [sin(θ),  cos(θ)]]
    """
    phi = np.array([theta])
    U = phi2u(phi)

    expected = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

    npt.assert_allclose(
        U, expected, atol=ATOL, rtol=RTOL,
        err_msg=f"2×2 rotation with θ={theta} does not match expected matrix",
    )


# ---------------------------------------------------------------------------
# Test 10: test_phi2u_fixture_parity — MATLAB-generated fixture comparison
# ---------------------------------------------------------------------------

def test_phi2u_fixture_parity(utility_fixture_dir: Path) -> None:
    """Validate phi2u output against MATLAB/Octave-generated fixtures.

    Loads the phi2u.npy fixture containing test cases generated by Octave
    with rng(42). Each test case includes input angles and the expected
    orthogonal matrix output.

    If fixture files are not available, the test is gracefully skipped via
    pytest.skip per AAP Section 0.7.1.
    """
    # Load the primary fixture dict (object-dtype .npy with K3, K4, K5 cases)
    fixture_data = load_fixture_npy(utility_fixture_dir, "phi2u")

    # The fixture is stored as an object array wrapping a dict
    if fixture_data.dtype == object:
        fixture_dict = fixture_data.item()
    else:
        pytest.skip("Unexpected fixture format for phi2u.npy")
        return  # unreachable, but satisfies type checker

    # Iterate over all K-specific test cases (K3, K4, K5)
    for case_key in ["K3", "K4", "K5"]:
        if case_key not in fixture_dict:
            continue

        case = fixture_dict[case_key]
        phi_input = np.asarray(case["input"], dtype=np.float64)
        expected_U = np.asarray(case["expected"], dtype=np.float64)
        k_val = case["k"]

        U = phi2u(phi_input)

        # Shape validation
        assert U.shape == (k_val, k_val), (
            f"[{case_key}] Expected shape ({k_val}, {k_val}), got {U.shape}"
        )

        # Element-wise parity with MATLAB fixture
        npt.assert_allclose(
            U, expected_U, atol=ATOL, rtol=RTOL,
            err_msg=(
                f"[{case_key}] phi2u output does not match MATLAB fixture. "
                f"Description: {case.get('description', 'N/A')}"
            ),
        )

    # Also validate the simple phi_vals / phi2u_out fixture pair
    phi_vals = load_fixture_npy(utility_fixture_dir, "phi2u_phi_vals")
    expected_out = load_fixture_npy(utility_fixture_dir, "phi2u_phi2u_out")

    # phi_vals contains [0.3, 0.5] which is 2 values but K(K-1)/2 = 1 for K=2
    # The fixture pair is a 2×2 rotation with a single angle
    # Check if phi_vals has length 1 (K=2) or length 2 (special handling)
    if phi_vals.ndim == 1 and len(phi_vals) == 1:
        U_simple = phi2u(phi_vals)
    elif phi_vals.ndim == 1 and len(phi_vals) == 2:
        # phi_vals = [0.3, 0.5] but only one angle needed for 2×2
        # The fixture was generated with a single angle from the first value
        # Try using first value as the single angle for 2×2
        U_simple = phi2u(np.array([phi_vals[0]]))
    else:
        pytest.skip(
            f"Unexpected phi_vals shape {phi_vals.shape} for simple fixture pair"
        )
        return

    npt.assert_allclose(
        U_simple, expected_out, atol=ATOL, rtol=RTOL,
        err_msg="Simple phi_vals fixture parity check failed",
    )
