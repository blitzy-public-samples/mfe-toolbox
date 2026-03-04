"""
Pytest tests for ``mfe_toolbox.utility.corr_ivech``.

Tests the inverse correlation half-vectorization function which reconstructs
a symmetric K×K correlation matrix from a K(K-1)/2 vector of off-diagonal
elements using column-major ordering with unit diagonal.

Migrated from: utility/corr_ivech.m (51 lines)
Author: Kevin Sheppard

Test coverage:
    1.  test_corr_ivech_1element      — single-element vector → 2×2 matrix
    2.  test_corr_ivech_3elements     — 3-element vector → 3×3 correlation matrix
    3.  test_corr_ivech_diagonal_ones — diagonal always exactly 1.0
    4.  test_corr_ivech_symmetric     — output is symmetric
    5.  test_corr_ivech_output_shape  — K(K-1)/2 input → K×K output
    6.  test_corr_ivech_zeros         — all-zero input → identity matrix
    7.  test_corr_ivech_invalid_length_raises — non-conformable length → ValueError
    8.  test_corr_ivech_non_vector_raises     — 2-D matrix input → ValueError
    9.  test_corr_ivech_corr_vech_roundtrip   — corr_vech(corr_ivech(v)) == v
    10. test_corr_ivech_column_major_fill     — verify column-major fill order
    11. test_corr_ivech_fixture_parity        — MATLAB/Octave fixture comparison
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.corr_ivech import corr_ivech
from mfe_toolbox.utility.corr_vech import corr_vech
from tests.conftest import ATOL, RTOL, assert_allclose, load_fixture_npy


# ---------------------------------------------------------------------------
# Test 1: Single-element vector → 2×2 correlation matrix
# ---------------------------------------------------------------------------

def test_corr_ivech_1element() -> None:
    """[0.5] → 2×2: [[1, 0.5], [0.5, 1]].

    Ref: corr_ivech.m — simplest case where K=2, K(K-1)/2 = 1.
    The single off-diagonal element fills both (0,1) and (1,0).
    """
    v = np.array([0.5])
    expected = np.array([
        [1.0, 0.5],
        [0.5, 1.0],
    ])
    result = corr_ivech(v)
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                        err_msg="1-element corr_ivech mismatch")


# ---------------------------------------------------------------------------
# Test 2: 3-element vector → 3×3 correlation matrix
# ---------------------------------------------------------------------------

def test_corr_ivech_3elements() -> None:
    """3-element vector → 3×3 correlation matrix.

    Ref: corr_ivech.m — K=3, K(K-1)/2 = 3.
    With column-major fill ordering, elements map as:
      v[0] → (1,0) and (0,1)
      v[1] → (2,0) and (0,2)
      v[2] → (2,1) and (1,2)
    """
    v = np.array([0.5, 0.3, 0.7])
    expected = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.7],
        [0.3, 0.7, 1.0],
    ])
    result = corr_ivech(v)
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                        err_msg="3-element corr_ivech mismatch")


# ---------------------------------------------------------------------------
# Test 3: Diagonal always exactly 1.0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_elements,K", [
    (1, 2),    # K=2
    (3, 3),    # K=3
    (6, 4),    # K=4
    (10, 5),   # K=5
])
def test_corr_ivech_diagonal_ones(n_elements: int, K: int) -> None:
    """Output diagonal is always exactly 1.0 for various matrix sizes.

    Ref: corr_ivech.m:51 — matrixData = matrixData + matrixData' + eye(K)
    The eye(K) guarantees unit diagonal regardless of input values.
    """
    rng = np.random.default_rng(123)
    v = rng.uniform(-0.9, 0.9, size=n_elements)
    result = corr_ivech(v)
    npt.assert_array_equal(np.diag(result), np.ones(K),
                           err_msg=f"Diagonal not all ones for K={K}")


# ---------------------------------------------------------------------------
# Test 4: Output is symmetric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_elements", [1, 3, 6, 10, 15])
def test_corr_ivech_symmetric(n_elements: int) -> None:
    """Output matrix is symmetric: result == result.T.

    Ref: corr_ivech.m:51 — matrixData + matrixData' construction
    guarantees symmetry by design.
    """
    rng = np.random.default_rng(456)
    v = rng.uniform(-0.9, 0.9, size=n_elements)
    result = corr_ivech(v)
    assert np.allclose(result, result.T, atol=1e-15), (
        f"Output matrix is not symmetric for {n_elements}-element input"
    )


# ---------------------------------------------------------------------------
# Test 5: Output shape K(K-1)/2 → K×K
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_elements,K", [
    (1, 2),
    (3, 3),
    (6, 4),
    (10, 5),
    (15, 6),
    (21, 7),
])
def test_corr_ivech_output_shape(n_elements: int, K: int) -> None:
    """K(K-1)/2 input vector → K×K output matrix.

    Ref: corr_ivech.m:38-39 — K = (-1 + sqrt(1 + 8*K2)) / 2 + 1
    Verifies the dimension formula is correct for various sizes.
    """
    v = np.zeros(n_elements)
    result = corr_ivech(v)
    assert result.shape == (K, K), (
        f"Expected shape ({K}, {K}), got {result.shape} "
        f"for {n_elements}-element input"
    )


# ---------------------------------------------------------------------------
# Test 6: All-zero input → identity matrix
# ---------------------------------------------------------------------------

def test_corr_ivech_zeros() -> None:
    """All-zero input vector → identity matrix.

    When all off-diagonal correlations are zero, the result must be I_K.
    Ref: corr_ivech.m:48-51 — zeros(K) + zeros(K)' + eye(K) = eye(K).
    """
    # Test for K=3 (3 zero elements)
    v = np.zeros(3)
    result = corr_ivech(v)
    npt.assert_allclose(result, np.eye(3), atol=ATOL, rtol=RTOL,
                        err_msg="All-zero input should produce identity matrix")

    # Test for K=4 (6 zero elements)
    v4 = np.zeros(6)
    result4 = corr_ivech(v4)
    npt.assert_allclose(result4, np.eye(4), atol=ATOL, rtol=RTOL,
                        err_msg="All-zero 6-element input should produce 4×4 identity")


# ---------------------------------------------------------------------------
# Test 7: Invalid vector length → ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_length", [2, 4, 5, 7, 8, 9, 11, 12, 13, 14])
def test_corr_ivech_invalid_length_raises(bad_length: int) -> None:
    """Invalid vector length (not K(K-1)/2 for any integer K) → ValueError.

    Ref: corr_ivech.m:41-44 — if floor(K)~=K, error(...)
    Valid lengths are: 1, 3, 6, 10, 15, 21, ...
    """
    v = np.zeros(bad_length)
    with pytest.raises(ValueError):
        corr_ivech(v)


# ---------------------------------------------------------------------------
# Test 8: 2-D matrix input → ValueError
# ---------------------------------------------------------------------------

def test_corr_ivech_non_vector_raises() -> None:
    """2-D matrix input (both dims > 1) → ValueError.

    Ref: corr_ivech.m:34-36 — if size(stackedData,2) ~= 1, error(...)
    The function only accepts 1-D vectors (or row/column vectors).
    """
    # 2×2 matrix — clearly not a vector
    v_2d = np.array([[0.5, 0.3], [0.4, 0.6]])
    with pytest.raises(ValueError):
        corr_ivech(v_2d)

    # 3×2 matrix — also not a vector
    v_3x2 = np.ones((3, 2))
    with pytest.raises(ValueError):
        corr_ivech(v_3x2)


# ---------------------------------------------------------------------------
# Test 9: Roundtrip corr_vech(corr_ivech(v)) == v
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_elements", [1, 3, 6, 10, 15])
def test_corr_ivech_corr_vech_roundtrip(n_elements: int) -> None:
    """Roundtrip: corr_vech(corr_ivech(v)) recovers the original vector v.

    Ref: corr_ivech.m See also: corr_vech
    This verifies that corr_ivech and corr_vech are true inverses,
    maintaining column-major ordering consistency.
    """
    rng = np.random.default_rng(789)
    v = rng.uniform(-0.9, 0.9, size=n_elements)

    # Forward: vector → matrix
    mat = corr_ivech(v)

    # Backward: matrix → vector
    v_recovered = corr_vech(mat)

    npt.assert_allclose(v_recovered, v, atol=ATOL, rtol=RTOL,
                        err_msg=f"Roundtrip failed for {n_elements}-element vector")


# ---------------------------------------------------------------------------
# Test 10: Column-major fill ordering with distinct values
# ---------------------------------------------------------------------------

def test_corr_ivech_column_major_fill() -> None:
    """Verify column-major fill order with distinct values.

    CRITICAL: MATLAB fills the lower triangle in column-major order:
    For K=4, ~triu(true(4)) selects positions in this order:
      (1,0), (2,0), (3,0), (2,1), (3,1), (3,2)  [0-based]

    Ref: corr_ivech.m:49-50 — loc = ~triu(true(K)); matrixData(loc) = stackedData
    MATLAB's logical indexing traverses columns first (column-major).

    With input v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
      v[0]=0.1 → (1,0) and (0,1)
      v[1]=0.2 → (2,0) and (0,2)
      v[2]=0.3 → (3,0) and (0,3)
      v[3]=0.4 → (2,1) and (1,2)
      v[4]=0.5 → (3,1) and (1,3)
      v[5]=0.6 → (3,2) and (2,3)
    """
    v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    expected = np.array([
        [1.0, 0.1, 0.2, 0.3],
        [0.1, 1.0, 0.4, 0.5],
        [0.2, 0.4, 1.0, 0.6],
        [0.3, 0.5, 0.6, 1.0],
    ])
    result = corr_ivech(v)
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                        err_msg="Column-major fill order mismatch for K=4")

    # Also test K=5 with 10 distinct elements
    v5 = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -0.1])
    expected5 = np.array([
        [1.0,   0.1,  -0.2,  0.3, -0.4],
        [0.1,   1.0,   0.5, -0.6,  0.7],
        [-0.2,  0.5,   1.0, -0.8,  0.9],
        [0.3,  -0.6,  -0.8,  1.0, -0.1],
        [-0.4,  0.7,   0.9, -0.1,  1.0],
    ])
    result5 = corr_ivech(v5)
    npt.assert_allclose(result5, expected5, atol=ATOL, rtol=RTOL,
                        err_msg="Column-major fill order mismatch for K=5")


# ---------------------------------------------------------------------------
# Test 11: Fixture parity — MATLAB/Octave reference comparison
# ---------------------------------------------------------------------------

def test_corr_ivech_fixture_parity(utility_fixture_dir: Path) -> None:
    """Fixture comparison against MATLAB/Octave generated reference outputs.

    Loads the corr_ivech.npy fixture file containing multiple test cases
    generated by Octave and verifies that the Python implementation
    produces identical results within tolerance (atol=1e-6, rtol=1e-4).

    Per AAP Section 0.7.1: Every migrated function MUST pass
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
    against MATLAB-generated fixtures.
    """
    fixture = load_fixture_npy(utility_fixture_dir, "corr_ivech")

    # The fixture is a 0-d object array wrapping a dict of test cases
    if fixture.ndim == 0:
        data = fixture.item()
    else:
        data = fixture

    # Test case 1: K=2 (1 element)
    if "test1_input_vector" in data:
        v1 = np.asarray(data["test1_input_vector"])
        expected1 = np.asarray(data["test1_expected_matrix"])
        result1 = corr_ivech(v1)
        assert_allclose(result1, expected1,
                        err_msg="Fixture parity: test1 (K=2) mismatch")

    # Test case 2: K=3 (3 elements)
    if "test2_input_vector" in data:
        v2 = np.asarray(data["test2_input_vector"])
        expected2 = np.asarray(data["test2_expected_matrix"])
        result2 = corr_ivech(v2)
        assert_allclose(result2, expected2,
                        err_msg="Fixture parity: test2 (K=3) mismatch")

    # Test case 3: K=4 (6 elements)
    if "test3_input_vector" in data:
        v3 = np.asarray(data["test3_input_vector"])
        expected3 = np.asarray(data["test3_expected_matrix"])
        result3 = corr_ivech(v3)
        assert_allclose(result3, expected3,
                        err_msg="Fixture parity: test3 (K=4) mismatch")

    # Test case 4: K=5 (10 elements)
    if "test4_input_vector" in data:
        v4 = np.asarray(data["test4_input_vector"])
        expected4 = np.asarray(data["test4_expected_matrix"])
        result4 = corr_ivech(v4)
        assert_allclose(result4, expected4,
                        err_msg="Fixture parity: test4 (K=5) mismatch")

    # Test case 5: Zero vector (identity matrix)
    if "test5_input_vector" in data:
        v5 = np.asarray(data["test5_input_vector"])
        expected5 = np.asarray(data["test5_expected_matrix"])
        result5 = corr_ivech(v5)
        assert_allclose(result5, expected5,
                        err_msg="Fixture parity: test5 (zeros) mismatch")

    # Test case 6: Roundtrip via corr_vech → corr_ivech
    if "test6_input_vector" in data:
        v6 = np.asarray(data["test6_input_vector"])
        expected6 = np.asarray(data["test6_expected_matrix"])
        result6 = corr_ivech(v6)
        assert_allclose(result6, expected6,
                        err_msg="Fixture parity: test6 (roundtrip) mismatch")

    # Test case 7: Negative correlation (K=2)
    if "test7_input_vector" in data:
        v7 = np.asarray(data["test7_input_vector"])
        expected7 = np.asarray(data["test7_expected_matrix"])
        result7 = corr_ivech(v7)
        assert_allclose(result7, expected7,
                        err_msg="Fixture parity: test7 (negative corr) mismatch")

    # Also test with the separate fixture files if available
    cv_input_path = utility_fixture_dir / "corr_ivech_cv_input.npy"
    cv_out_path = utility_fixture_dir / "corr_ivech_corr_ivech_out.npy"
    if cv_input_path.exists() and cv_out_path.exists():
        cv_input = np.load(cv_input_path, allow_pickle=True)
        cv_expected = np.load(cv_out_path, allow_pickle=True)
        cv_result = corr_ivech(cv_input)
        assert_allclose(cv_result, cv_expected,
                        err_msg="Fixture parity: separate fixture files mismatch")
