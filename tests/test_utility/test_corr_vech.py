"""
Pytest tests for ``mfe_toolbox.utility.corr_vech``.

Tests the correlation half-vectorization function which extracts the
OFF-DIAGONAL lower triangular elements of a symmetric K×K correlation matrix
into a K(K-1)/2 vector, using column-major (Fortran) ordering to match
MATLAB's boolean-indexed extraction convention.

CRITICAL: ``corr_vech`` extracts K(K-1)/2 elements (off-diagonal only),
NOT K(K+1)/2 (which would include diagonal).  This is because correlation
matrix diagonals are always 1 and carry no information.

Migrated from: utility/corr_vech.m (39 lines)
Author: Kevin Sheppard

Test coverage:
    1.  test_corr_vech_3x3                 — 3×3 correlation → 3-element vector
    2.  test_corr_vech_4x4                 — 4×4 → 6-element vector
    3.  test_corr_vech_identity            — identity matrix → all zeros
    4.  test_corr_vech_output_length       — K×K → K(K-1)/2 elements
    5.  test_corr_vech_non_square_raises   — non-square input → ValueError
    6.  test_corr_vech_non_symmetric_raises — non-symmetric input → ValueError
    7.  test_corr_vech_corr_ivech_roundtrip — corr_ivech(corr_vech(R)) == R
    8.  test_corr_vech_column_major_order  — explicit column-major order check
    9.  test_corr_vech_fixture_parity      — MATLAB/Octave fixture comparison
    10. test_corr_vech_2x2                 — 2×2 → 1-element vector
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.corr_vech import corr_vech
from mfe_toolbox.utility.corr_ivech import corr_ivech
from tests.conftest import ATOL, RTOL, assert_allclose, load_fixture_npy


# ---------------------------------------------------------------------------
# Test 1: 3×3 correlation matrix → 3-element vector
# ---------------------------------------------------------------------------

def test_corr_vech_3x3() -> None:
    """3×3 correlation matrix produces a 3-element vector (K(K-1)/2 = 3).

    Ref: corr_vech.m:37-38 — sel = ~triu(true(k)); stackedData = matrixData(sel);
    Column-major off-diagonal extraction order for 3×3:
      Column 0: (1,0), (2,0)
      Column 1: (2,1)
    So for R = [[1, 0.5, 0.3], [0.5, 1, 0.7], [0.3, 0.7, 1]]:
      result = [R[1,0], R[2,0], R[2,1]] = [0.5, 0.3, 0.7]
    """
    R = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.7],
        [0.3, 0.7, 1.0],
    ])
    expected = np.array([0.5, 0.3, 0.7])
    result = corr_vech(R)

    assert result.ndim == 1, "Output must be 1-dimensional"
    assert result.shape[0] == 3, "3×3 matrix should yield 3 elements"
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                        err_msg="3×3 corr_vech mismatch")


# ---------------------------------------------------------------------------
# Test 2: 4×4 correlation matrix → 6-element vector
# ---------------------------------------------------------------------------

def test_corr_vech_4x4() -> None:
    """4×4 correlation matrix produces a 6-element vector (K(K-1)/2 = 6).

    Ref: corr_vech.m — column-major extraction for 4×4:
      Column 0: (1,0), (2,0), (3,0)
      Column 1: (2,1), (3,1)
      Column 2: (3,2)

    Uses a hand-constructed correlation matrix with distinct off-diagonal
    elements so ordering can be verified unambiguously.
    """
    R = np.array([
        [1.0, 0.1, 0.2, 0.3],
        [0.1, 1.0, 0.4, 0.5],
        [0.2, 0.4, 1.0, 0.6],
        [0.3, 0.5, 0.6, 1.0],
    ])
    # Column-major off-diagonal: (1,0)=0.1, (2,0)=0.2, (3,0)=0.3,
    #                             (2,1)=0.4, (3,1)=0.5, (3,2)=0.6
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    result = corr_vech(R)

    assert result.ndim == 1, "Output must be 1-dimensional"
    assert result.shape[0] == 6, "4×4 matrix should yield 6 elements"
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                        err_msg="4×4 corr_vech mismatch")


# ---------------------------------------------------------------------------
# Test 3: Identity matrix → all zeros
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [2, 3, 4, 5, 8])
def test_corr_vech_identity(K: int) -> None:
    """Identity matrix (all off-diag = 0) → zero vector of length K(K-1)/2.

    The identity matrix is a valid correlation matrix with all pairwise
    correlations equal to zero.  The output vector should contain all zeros.
    """
    I = np.eye(K)
    result = corr_vech(I)

    expected_len = K * (K - 1) // 2
    assert result.shape == (expected_len,), (
        f"Identity K={K}: expected shape ({expected_len},), got {result.shape}"
    )
    npt.assert_array_equal(result, np.zeros(expected_len),
                           err_msg=f"Identity K={K} should produce all zeros")


# ---------------------------------------------------------------------------
# Test 4: Output length is K(K-1)/2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [2, 3, 4, 5, 6, 7, 10])
def test_corr_vech_output_length(K: int) -> None:
    """K×K symmetric matrix → K(K-1)/2-element output vector.

    Ref: corr_vech.m — the selector ~triu(true(k)) picks exactly K(K-1)/2
    strictly lower triangular elements, excluding the K diagonal entries.
    """
    # Construct a valid symmetric matrix (identity suffices for length check)
    R = np.eye(K)
    result = corr_vech(R)

    expected_len = K * (K - 1) // 2
    assert len(result) == expected_len, (
        f"K={K}: expected {expected_len} elements, got {len(result)}"
    )


# ---------------------------------------------------------------------------
# Test 5: Non-square matrix → ValueError
# ---------------------------------------------------------------------------

def test_corr_vech_non_square_raises() -> None:
    """Non-square input must raise ValueError.

    Ref: corr_vech.m:31 — if k~=l ... error('MATRIXDATA must be a symmetric matrix')
    The Python equivalent raises ValueError when rows != cols.
    """
    non_square = np.array([
        [1.0, 0.5],
        [0.5, 1.0],
        [0.3, 0.7],
    ])
    with pytest.raises(ValueError):
        corr_vech(non_square)


# ---------------------------------------------------------------------------
# Test 6: Non-symmetric matrix → ValueError
# ---------------------------------------------------------------------------

def test_corr_vech_non_symmetric_raises() -> None:
    """Non-symmetric square matrix must raise ValueError.

    Ref: corr_vech.m:31 — any(any(matrixData~=matrixData')) triggers error.
    The Python equivalent uses np.allclose for floating-point tolerance.
    """
    non_sym = np.array([
        [1.0, 0.5, 0.3],
        [0.6, 1.0, 0.7],   # (1,0)=0.6 ≠ (0,1)=0.5 → not symmetric
        [0.3, 0.7, 1.0],
    ])
    with pytest.raises(ValueError):
        corr_vech(non_sym)


# ---------------------------------------------------------------------------
# Test 7: Roundtrip with corr_ivech
# ---------------------------------------------------------------------------

def test_corr_vech_corr_ivech_roundtrip() -> None:
    """corr_ivech(corr_vech(R)) == R for valid correlation matrices.

    Tests the fundamental identity: vectorizing a correlation matrix and
    then reconstructing it should yield the original matrix (within
    floating-point tolerance).  Tested for K = 3, 4, 5.
    """
    # 3×3 roundtrip
    R3 = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.7],
        [0.3, 0.7, 1.0],
    ])
    v3 = corr_vech(R3)
    R3_reconstructed = corr_ivech(v3)
    npt.assert_allclose(R3_reconstructed, R3, atol=ATOL, rtol=RTOL,
                        err_msg="3×3 roundtrip failed")

    # 4×4 roundtrip
    R4 = np.array([
        [1.0, 0.1, 0.2, 0.3],
        [0.1, 1.0, 0.4, 0.5],
        [0.2, 0.4, 1.0, 0.6],
        [0.3, 0.5, 0.6, 1.0],
    ])
    v4 = corr_vech(R4)
    R4_reconstructed = corr_ivech(v4)
    npt.assert_allclose(R4_reconstructed, R4, atol=ATOL, rtol=RTOL,
                        err_msg="4×4 roundtrip failed")

    # 5×5 roundtrip with sample_corr_matrix-style construction
    A5 = np.array([
        [2.0, 0.5, 0.3, 0.1, 0.2],
        [0.5, 3.0, 0.7, 0.2, 0.4],
        [0.3, 0.7, 1.5, 0.4, 0.6],
        [0.1, 0.2, 0.4, 2.5, 0.3],
        [0.2, 0.4, 0.6, 0.3, 1.8],
    ])
    spd5 = A5 @ A5.T + 0.1 * np.eye(5)
    d5 = np.sqrt(np.diag(spd5))
    R5 = spd5 / np.outer(d5, d5)
    np.fill_diagonal(R5, 1.0)

    v5 = corr_vech(R5)
    R5_reconstructed = corr_ivech(v5)
    npt.assert_allclose(R5_reconstructed, R5, atol=ATOL, rtol=RTOL,
                        err_msg="5×5 roundtrip failed")


# ---------------------------------------------------------------------------
# Test 8: Column-major ordering verification
# ---------------------------------------------------------------------------

def test_corr_vech_column_major_order() -> None:
    """Explicit verification of column-major (Fortran) extraction order.

    Ref: corr_vech.m:37-38 — MATLAB's boolean indexing with sel = ~triu(true(k))
    extracts elements in column-major order:
      For 4×4:
        Column 0: (1,0), (2,0), (3,0)
        Column 1: (2,1), (3,1)
        Column 2: (3,2)

    Constructs a matrix with element values encoding their position to
    make the ordering unambiguous.
    """
    # Create a 4×4 symmetric matrix with distinct off-diagonal values
    # Each off-diagonal element is set to a unique value encoding (row, col)
    R = np.eye(4)
    # Lower triangle (column-major order expected):
    #   (1,0)=0.10, (2,0)=0.20, (3,0)=0.30
    #   (2,1)=0.21, (3,1)=0.31
    #   (3,2)=0.32
    R[1, 0] = R[0, 1] = 0.10
    R[2, 0] = R[0, 2] = 0.20
    R[3, 0] = R[0, 3] = 0.30
    R[2, 1] = R[1, 2] = 0.21
    R[3, 1] = R[1, 3] = 0.31
    R[3, 2] = R[2, 3] = 0.32

    result = corr_vech(R)

    # Expected column-major ordering: down column 0, then column 1, then column 2
    expected = np.array([0.10, 0.20, 0.30, 0.21, 0.31, 0.32])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                        err_msg="Column-major ordering mismatch")


# ---------------------------------------------------------------------------
# Test 9: MATLAB/Octave fixture parity
# ---------------------------------------------------------------------------

def test_corr_vech_fixture_parity(utility_fixture_dir: Path) -> None:
    """Compare corr_vech output against MATLAB/Octave-generated fixtures.

    Loads the fixture file 'corr_vech.npy' which contains:
      - input_R3, expected_v3: 3×3 case
      - input_R4, expected_v4: 4×4 case
      - input_R5, expected_v5: 5×5 case
      - input_R_hand, expected_v_hand: hand-constructed 3×3 case

    Also checks the separate corr_vech_R_corr.npy / corr_vech_corr_vech_out.npy
    fixture pair.

    All comparisons use atol=1e-6, rtol=1e-4 per AAP Section 0.7.1.
    """
    # Load the main fixture dictionary
    fixture_data = load_fixture_npy(utility_fixture_dir, "corr_vech")

    # The fixture is stored as a 0-d object array wrapping a dict
    if fixture_data.dtype == object and fixture_data.ndim == 0:
        fixture_dict = fixture_data.item()
    else:
        fixture_dict = fixture_data

    # ---- 3×3 case ----
    if "input_R3" in fixture_dict and "expected_v3" in fixture_dict:
        input_R3 = np.asarray(fixture_dict["input_R3"], dtype=np.float64)
        expected_v3 = np.asarray(fixture_dict["expected_v3"], dtype=np.float64)
        result_v3 = corr_vech(input_R3)
        assert_allclose(result_v3, expected_v3,
                        err_msg="Fixture parity failed: 3×3 case")

    # ---- 4×4 case ----
    if "input_R4" in fixture_dict and "expected_v4" in fixture_dict:
        input_R4 = np.asarray(fixture_dict["input_R4"], dtype=np.float64)
        expected_v4 = np.asarray(fixture_dict["expected_v4"], dtype=np.float64)
        result_v4 = corr_vech(input_R4)
        assert_allclose(result_v4, expected_v4,
                        err_msg="Fixture parity failed: 4×4 case")

    # ---- 5×5 case ----
    if "input_R5" in fixture_dict and "expected_v5" in fixture_dict:
        input_R5 = np.asarray(fixture_dict["input_R5"], dtype=np.float64)
        expected_v5 = np.asarray(fixture_dict["expected_v5"], dtype=np.float64)
        result_v5 = corr_vech(input_R5)
        assert_allclose(result_v5, expected_v5,
                        err_msg="Fixture parity failed: 5×5 case")

    # ---- Hand-constructed case ----
    if "input_R_hand" in fixture_dict and "expected_v_hand" in fixture_dict:
        input_R_hand = np.asarray(fixture_dict["input_R_hand"], dtype=np.float64)
        expected_v_hand = np.asarray(fixture_dict["expected_v_hand"], dtype=np.float64)
        result_v_hand = corr_vech(input_R_hand)
        assert_allclose(result_v_hand, expected_v_hand,
                        err_msg="Fixture parity failed: hand-constructed case")

    # ---- Separate fixture pair ----
    R_corr_path = utility_fixture_dir / "corr_vech_R_corr.npy"
    out_path = utility_fixture_dir / "corr_vech_corr_vech_out.npy"
    if R_corr_path.exists() and out_path.exists():
        R_corr = np.load(R_corr_path, allow_pickle=True)
        expected_out = np.load(out_path, allow_pickle=True)
        result_out = corr_vech(np.asarray(R_corr, dtype=np.float64))
        assert_allclose(result_out, np.asarray(expected_out, dtype=np.float64),
                        err_msg="Fixture parity failed: corr_vech_R_corr pair")


# ---------------------------------------------------------------------------
# Test 10: 2×2 correlation matrix → 1-element vector
# ---------------------------------------------------------------------------

def test_corr_vech_2x2() -> None:
    """2×2 correlation matrix produces a single-element vector (K(K-1)/2 = 1).

    Ref: corr_vech.m — smallest valid case.
    The 2×2 correlation matrix [[1, r], [r, 1]] has one off-diagonal element.
    """
    R = np.array([
        [1.0, 0.8],
        [0.8, 1.0],
    ])
    expected = np.array([0.8])
    result = corr_vech(R)

    assert result.ndim == 1, "Output must be 1-dimensional"
    assert result.shape[0] == 1, "2×2 matrix should yield 1 element"
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                        err_msg="2×2 corr_vech mismatch")

    # Additional: test with negative correlation
    R_neg = np.array([
        [1.0, -0.6],
        [-0.6, 1.0],
    ])
    expected_neg = np.array([-0.6])
    result_neg = corr_vech(R_neg)
    npt.assert_allclose(result_neg, expected_neg, atol=ATOL, rtol=RTOL,
                        err_msg="2×2 negative correlation mismatch")
