"""
Pytest tests for ``mfe_toolbox.utility.ivech`` — Inverse Half-Vectorization.

Validates that ``ivech()`` reconstructs a symmetric K×K matrix from a
K(K+1)/2 element vector, matching the MATLAB ``ivech.m`` behavior:
column-major lower-triangle fill via ``tril(true(K))``, then symmetrisation
via ``matrixData + matrixData' - diag(diag(matrixData))``.

Source Reference
----------------
MATLAB source: ``utility/ivech.m`` (57 lines, Revision 3, Date: 2/1/2008)
Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk

AAP Requirements
----------------
- Numerical parity: ``numpy.testing.assert_allclose(atol=1e-6, rtol=1e-4)``
- Coverage ≥90%
- Error handling parity with MATLAB
- Every test documents what MATLAB behavior it verifies
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.ivech import ivech
from mfe_toolbox.utility.vech import vech


# ---------------------------------------------------------------------------
# Tolerance constants (per AAP Section 0.7.1 and conftest.py)
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ===================================================================
# 1. Core Functionality Tests
# ===================================================================


def test_ivech_1element():
    """Verify ivech produces a 1×1 matrix from a single-element vector.

    Ref: ivech.m:49-55 — trivial K=1 case: zeros(1), fill with scalar,
    symmetrise is identity.
    MATLAB: ivech(5) → [[5]]
    """
    v = np.array([5.0])
    result = ivech(v)
    expected = np.array([[5.0]])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)
    assert result.shape == (1, 1), f"Expected shape (1,1), got {result.shape}"


def test_ivech_3elements():
    """Verify ivech produces a 2×2 symmetric matrix from 3 elements.

    Ref: ivech.m:52-53 — column-major fill via tril(true(2)):
      (0,0)=1, (1,0)=2, (1,1)=3
    Ref: ivech.m:54-55 — symmetrise: mat + mat' - diag(diag(mat))
    MATLAB: ivech([1; 0.5; 2]) → [[1, 0.5]; [0.5, 2]]
    """
    v = np.array([1.0, 0.5, 2.0])
    result = ivech(v)
    expected = np.array([[1.0, 0.5],
                         [0.5, 2.0]])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_ivech_6elements():
    """Verify ivech produces a 3×3 symmetric matrix from 6 elements.

    CRITICAL: Tests column-major fill order matching MATLAB tril(true(3)):
      Column 0: (0,0)=1, (1,0)=2, (2,0)=3
      Column 1: (1,1)=4, (2,1)=5
      Column 2: (2,2)=6
    Ref: ivech.m:52-53 — matrixData(tril(true(K))) = stackedData
    Ref: ivech.m:54-55 — symmetrise
    MATLAB: ivech([1;2;3;4;5;6]) → [[1,2,3];[2,4,5];[3,5,6]]
    """
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = ivech(v)
    expected = np.array([[1.0, 2.0, 3.0],
                         [2.0, 4.0, 5.0],
                         [3.0, 5.0, 6.0]])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_ivech_10elements():
    """Verify ivech produces a 4×4 symmetric matrix from 10 elements.

    Ref: ivech.m:52-55 — column-major fill for K=4:
      Col 0: (0,0), (1,0), (2,0), (3,0)
      Col 1: (1,1), (2,1), (3,1)
      Col 2: (2,2), (3,2)
      Col 3: (3,3)
    MATLAB: Tests with mixed positive/negative/zero values.
    """
    v = np.array([3.0, 0.5, -0.3, 0.7, 2.5, 1.2, -0.4, 1.8, 0.6, 2.0])
    result = ivech(v)
    expected = np.array([
        [3.0, 0.5, -0.3, 0.7],
        [0.5, 2.5, 1.2, -0.4],
        [-0.3, 1.2, 1.8, 0.6],
        [0.7, -0.4, 0.6, 2.0]
    ])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)
    assert result.shape == (4, 4), f"Expected shape (4,4), got {result.shape}"
    # Verify symmetry explicitly
    npt.assert_array_equal(result, result.T)


def test_ivech_15elements():
    """Verify ivech produces a 5×5 symmetric matrix from 15 elements.

    Ref: ivech.m:52-55 — column-major fill for K=5.
    Tests with larger matrix to ensure scaling works correctly.
    """
    v = np.array([6.0, 1.0, -2.0, 0.5, 3.0,
                  8.0, -1.5, 0.3, 4.0,
                  7.0, 2.5, -0.8,
                  5.0, 1.2,
                  9.0])
    result = ivech(v)
    expected = np.array([
        [6.0, 1.0, -2.0, 0.5, 3.0],
        [1.0, 8.0, -1.5, 0.3, 4.0],
        [-2.0, -1.5, 7.0, 2.5, -0.8],
        [0.5, 0.3, 2.5, 5.0, 1.2],
        [3.0, 4.0, -0.8, 1.2, 9.0]
    ])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)
    assert result.shape == (5, 5), f"Expected shape (5,5), got {result.shape}"


def test_ivech_identity_roundtrip():
    """Verify ivech produces the 3×3 identity from the vech of identity.

    Ref: ivech.m — identity matrix is its own vech/ivech roundtrip.
    MATLAB: ivech([1;0;0;1;0;1]) → eye(3)
    """
    v_identity = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    result = ivech(v_identity)
    expected = np.eye(3)
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_ivech_negative_diagonals():
    """Verify ivech handles negative diagonal elements correctly.

    Ref: ivech.m:54-55 — symmetrisation works regardless of sign.
    MATLAB: ivech([-2.5; 1.3; -0.7; 3.1; 0.4; -1.8])
    """
    v = np.array([-2.5, 1.3, -0.7, 3.1, 0.4, -1.8])
    result = ivech(v)
    expected = np.array([
        [-2.5, 1.3, -0.7],
        [1.3, 3.1, 0.4],
        [-0.7, 0.4, -1.8]
    ])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


# ===================================================================
# 2. Property Tests
# ===================================================================


@pytest.mark.parametrize("n_elements", [1, 3, 6, 10, 15, 21])
def test_ivech_output_symmetric(n_elements):
    """Verify ivech output is always symmetric for valid input sizes.

    Ref: ivech.m:54-55 — matrixData + matrixData' - diag(diag(matrixData))
    guarantees symmetry for any valid input.
    Uses distinct random values to ensure non-trivial symmetry test.
    """
    rng = np.random.default_rng(12345 + n_elements)
    v = rng.standard_normal(n_elements)
    result = ivech(v)
    npt.assert_allclose(
        result, result.T, atol=ATOL,
        err_msg=f"Output not symmetric for {n_elements}-element input"
    )


@pytest.mark.parametrize("K,n_elements", [
    (1, 1),
    (2, 3),
    (3, 6),
    (4, 10),
    (5, 15),
    (6, 21),
])
def test_ivech_output_shape(K, n_elements):
    """Verify ivech output has shape K×K for input of length K(K+1)/2.

    Ref: ivech.m:49 — matrixData = zeros(K) creates K×K output.
    Tests the quadratic formula dimension recovery: K = (-1+sqrt(1+8*K2))/2.
    """
    v = np.arange(1.0, n_elements + 1.0)
    result = ivech(v)
    assert result.shape == (K, K), (
        f"Expected shape ({K},{K}) for {n_elements}-element input, "
        f"got {result.shape}"
    )


def test_ivech_column_major_fill():
    """CRITICAL: Verify column-major fill order matches MATLAB tril(true(K)).

    This is THE critical test ensuring Python matches MATLAB convention.

    Ref: ivech.m:52-53 — pl = tril(true(K)); matrixData(pl) = stackedData
    MATLAB fills the lower triangle in COLUMN-major order:
      For K=3, v=[1,2,3,4,5,6]:
        Column 0: (0,0)=1, (1,0)=2, (2,0)=3
        Column 1: (1,1)=4, (2,1)=5
        Column 2: (2,2)=6

    After symmetrisation (mat + mat' - diag(diag(mat))):
      [[1, 2, 3],
       [2, 4, 5],
       [3, 5, 6]]

    If row-major fill were used instead (WRONG), you'd get:
      (0,0)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5, (2,2)=6
    Which after symmetrisation gives:
      [[1, 2, 4],
       [2, 3, 5],
       [4, 5, 6]]
    — a DIFFERENT matrix.
    """
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = ivech(v)

    # Expected: column-major fill, then symmetrised
    expected_column_major = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ])

    # Row-major fill would give a DIFFERENT result — verify we DON'T get that
    wrong_row_major = np.array([
        [1.0, 2.0, 4.0],
        [2.0, 3.0, 5.0],
        [4.0, 5.0, 6.0]
    ])

    npt.assert_allclose(result, expected_column_major, atol=ATOL, rtol=RTOL,
                        err_msg="Column-major fill order does not match MATLAB")
    # Verify explicitly that we do NOT match the row-major result
    assert not np.allclose(result, wrong_row_major, atol=ATOL), (
        "ivech produced row-major fill result — should be column-major"
    )


def test_ivech_column_major_fill_4x4():
    """Verify column-major fill order for a 4×4 matrix.

    Ref: ivech.m:52-53 — column-major fill for K=4:
      Col 0: (0,0)=1, (1,0)=2, (2,0)=3, (3,0)=4
      Col 1: (1,1)=5, (2,1)=6, (3,1)=7
      Col 2: (2,2)=8, (3,2)=9
      Col 3: (3,3)=10
    """
    v = np.arange(1.0, 11.0)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = ivech(v)

    # Column-major fill → symmetrise
    expected = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 5.0, 6.0, 7.0],
        [3.0, 6.0, 8.0, 9.0],
        [4.0, 7.0, 9.0, 10.0]
    ])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_ivech_diagonal_elements():
    """Verify diagonal elements are placed correctly.

    Ref: ivech.m:54 — diag_matrixData = diag(diag(matrixData))
    The diagonal elements in column-major order for K=3 are at positions
    0, 3, 5 in the input vector (i.e., (0,0), (1,1), (2,2)).
    """
    v = np.array([10.0, 0.0, 0.0, 20.0, 0.0, 30.0])
    result = ivech(v)
    # Only diagonal should be non-zero
    expected_diag = np.array([10.0, 20.0, 30.0])
    npt.assert_allclose(np.diag(result), expected_diag, atol=ATOL, rtol=RTOL)
    # Off-diagonal should be zero
    off_diag_mask = ~np.eye(3, dtype=bool)
    npt.assert_allclose(result[off_diag_mask], 0.0, atol=ATOL)


# ===================================================================
# 3. Input Handling Tests
# ===================================================================


def test_ivech_row_vector_input():
    """Verify row vector (1, N) is auto-handled to produce same result.

    Ref: ivech.m:29-31 — if size(stackedData,2) > size(stackedData,1),
    stackedData = stackedData'. The Python implementation uses ravel()
    which handles both row and column vector shapes.
    """
    v_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    v_row = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])  # shape (1, 6)

    result_1d = ivech(v_1d)
    result_row = ivech(v_row)

    npt.assert_allclose(result_row, result_1d, atol=ATOL, rtol=RTOL,
                        err_msg="Row vector input should produce same result as 1D")


def test_ivech_column_vector_input():
    """Verify column vector (N, 1) is auto-handled to produce same result.

    Ref: ivech.m:29-31 — MATLAB column vectors are the native case.
    The Python implementation uses ravel() for both orientations.
    """
    v_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    v_col = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])  # shape (6, 1)

    result_1d = ivech(v_1d)
    result_col = ivech(v_col)

    npt.assert_allclose(result_col, result_1d, atol=ATOL, rtol=RTOL,
                        err_msg="Column vector input should produce same result as 1D")


@pytest.mark.parametrize("invalid_length", [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16])
def test_ivech_invalid_length_raises(invalid_length):
    """Verify ValueError is raised for non-triangular input lengths.

    Ref: ivech.m:38-43 — K = (-1+sqrt(1+8*K2))/2; if floor(K) ~= K,
    error('... must be conformable to the inverse vech operation.')

    Valid triangular numbers are K(K+1)/2: 1, 3, 6, 10, 15, 21, ...
    All other lengths must raise ValueError.
    """
    v = np.ones(invalid_length)
    with pytest.raises(ValueError, match="conformable"):
        ivech(v)


def test_ivech_empty_input_raises():
    """Verify ValueError for empty array input.

    Ref: ivech.m:38-43 — K2=0 gives K=0, floor(0)==0 but 0*(0+1)/2=0,
    which is conformable but produces empty matrix. The Python implementation
    should handle this as either producing a (0,0) array or raising ValueError.
    """
    v = np.array([])
    # K = (-1 + sqrt(1 + 0))/2 = 0, so K*(K+1)/2 = 0 == len(v)
    # The implementation should either produce a (0,0) array or handle gracefully
    result = ivech(v)
    assert result.shape == (0, 0), f"Expected shape (0,0) for empty input, got {result.shape}"


def test_ivech_integer_input():
    """Verify ivech works with integer arrays (implicit float conversion).

    Ref: ivech.m — MATLAB handles both integer and double inputs seamlessly.
    Python implementation should accept integer arrays via numpy coercion.
    """
    v_int = np.array([1, 2, 3, 4, 5, 6])
    result = ivech(v_int)
    expected = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_ivech_list_input():
    """Verify ivech accepts Python list inputs.

    The Python implementation should handle list inputs via np.atleast_1d().
    """
    v_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result = ivech(v_list)
    expected = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


# ===================================================================
# 4. Roundtrip Tests
# ===================================================================


def test_ivech_vech_roundtrip_forward():
    """Verify vech(ivech(v)) == v (forward roundtrip).

    Given a valid vector v of length K(K+1)/2, applying ivech to get
    a symmetric matrix S, then vech to extract back, should recover
    the original vector.

    Ref: ivech.m and vech.m are documented inverses of each other.
    """
    rng = np.random.default_rng(42)
    for K in [2, 3, 4, 5, 6]:
        n = K * (K + 1) // 2
        v = rng.standard_normal(n)
        S = ivech(v)
        # vech returns (N, 1) column vector; flatten for comparison
        v_back = vech(S).ravel()
        npt.assert_allclose(
            v_back, v, atol=ATOL, rtol=RTOL,
            err_msg=f"Forward roundtrip failed for K={K}"
        )


def test_ivech_vech_roundtrip_reverse():
    """Verify ivech(vech(S)) == S (reverse roundtrip).

    Given a random symmetric matrix S, applying vech to extract the
    lower triangle, then ivech to reconstruct, should recover the
    original symmetric matrix.

    Ref: ivech.m and vech.m are documented inverses of each other.
    """
    rng = np.random.default_rng(42)
    for K in [2, 3, 4, 5, 6]:
        # Create a random symmetric matrix
        A = rng.standard_normal((K, K))
        S = (A + A.T) / 2.0  # ensure symmetry

        v = vech(S)
        S_back = ivech(v)
        npt.assert_allclose(
            S_back, S, atol=ATOL, rtol=RTOL,
            err_msg=f"Reverse roundtrip failed for K={K}"
        )


def test_ivech_vech_roundtrip_identity():
    """Verify roundtrip with identity matrices of various sizes.

    Identity matrices have a particularly simple vech representation
    and should roundtrip exactly.
    """
    for K in [1, 2, 3, 4, 5]:
        S = np.eye(K)
        v = vech(S)
        S_back = ivech(v)
        npt.assert_allclose(
            S_back, S, atol=ATOL, rtol=RTOL,
            err_msg=f"Identity roundtrip failed for K={K}"
        )


def test_ivech_vech_roundtrip_covariance():
    """Verify roundtrip with a realistic positive-definite covariance matrix.

    Covariance matrices are the primary use case for vech/ivech in the MFE
    Toolbox (multivariate GARCH models, BEKK, DCC, etc.).
    """
    rng = np.random.default_rng(99)
    K = 4
    # Generate a positive-definite covariance matrix: S = A'A / T
    A = rng.standard_normal((100, K))
    S = (A.T @ A) / 100.0

    v = vech(S)
    S_back = ivech(v)
    npt.assert_allclose(
        S_back, S, atol=ATOL, rtol=RTOL,
        err_msg="Covariance matrix roundtrip failed"
    )
    # Verify the reconstructed matrix is still symmetric
    npt.assert_allclose(S_back, S_back.T, atol=ATOL)


# ===================================================================
# 5. Fixture Parity Tests
# ===================================================================


# Path to fixture files for conditional skipping
_FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "utility"
_FIXTURE_NPY = _FIXTURE_DIR / "ivech.npy"
_HAS_FIXTURES = _FIXTURE_NPY.exists()


@pytest.mark.skipif(not _HAS_FIXTURES, reason="Fixture file not found")
class TestIvechFixtureParity:
    """MATLAB/Octave fixture parity tests for ivech.

    Loads reference inputs and expected outputs generated by Octave 8.4.0
    from tests/fixtures/utility/ivech.npy and compares Python ivech()
    results against MATLAB reference with atol=1e-6, rtol=1e-4.
    """

    @pytest.fixture(autouse=True)
    def _load_fixtures(self):
        """Load the ivech fixture data once for the class."""
        self.fixture = np.load(_FIXTURE_NPY, allow_pickle=True).item()

    def test_fixture_test1_1x1(self):
        """Fixture parity: K=1 trivial scalar case."""
        v = self.fixture["test1_input_vector"]
        expected = self.fixture["test1_expected_matrix"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: test1 (K=1)")

    def test_fixture_test2_2x2(self):
        """Fixture parity: K=2 simple case."""
        v = self.fixture["test2_input_vector"]
        expected = self.fixture["test2_expected_matrix"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: test2 (K=2)")

    def test_fixture_test3_3x3_sequential(self):
        """Fixture parity: K=3 sequential values [1..6] — column-major fill."""
        v = self.fixture["test3_input_vector"]
        expected = self.fixture["test3_expected_matrix"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: test3 (K=3 sequential)")

    def test_fixture_test4_4x4_mixed_sign(self):
        """Fixture parity: K=4 mixed positive/negative values."""
        v = self.fixture["test4_input_vector"]
        expected = self.fixture["test4_expected_matrix"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: test4 (K=4)")

    def test_fixture_test5_5x5_mixed_sign(self):
        """Fixture parity: K=5 mixed positive/negative values."""
        v = self.fixture["test5_input_vector"]
        expected = self.fixture["test5_expected_matrix"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: test5 (K=5)")

    def test_fixture_test6_identity(self):
        """Fixture parity: K=3 identity roundtrip."""
        v = self.fixture["test6_input_vector"]
        expected = self.fixture["test6_expected_matrix"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: test6 (identity)")

    def test_fixture_test7_negative_diagonals(self):
        """Fixture parity: K=3 negative diagonal elements."""
        v = self.fixture["test7_input_vector"]
        expected = self.fixture["test7_expected_matrix"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: test7 (negative diag)")

    def test_fixture_named_3x3(self):
        """Fixture parity: named input_vector_6 / expected_matrix_3x3."""
        v = self.fixture["input_vector_6"]
        expected = self.fixture["expected_matrix_3x3"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: named 3x3")

    def test_fixture_named_4x4(self):
        """Fixture parity: named input_vector_10 / expected_matrix_4x4."""
        v = self.fixture["input_vector_10"]
        expected = self.fixture["expected_matrix_4x4"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: named 4x4")

    def test_fixture_named_5x5(self):
        """Fixture parity: named input_vector_15 / expected_matrix_5x5."""
        v = self.fixture["input_vector_15"]
        expected = self.fixture["expected_matrix_5x5"]
        result = ivech(v)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="Fixture parity failed: named 5x5")


@pytest.mark.skipif(
    not (_FIXTURE_DIR / "ivech_ivech_input.npy").exists(),
    reason="Separate ivech fixture files not found"
)
def test_ivech_separate_fixture_parity():
    """Fixture parity using separate ivech_ivech_input.npy / ivech_ivech_out.npy.

    These are standalone fixture files (not part of the dict-based fixture).
    MATLAB: ivech(input) → expected output.
    """
    v = np.load(_FIXTURE_DIR / "ivech_ivech_input.npy", allow_pickle=True)
    expected = np.load(_FIXTURE_DIR / "ivech_ivech_out.npy", allow_pickle=True)
    result = ivech(v)
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                        err_msg="Separate fixture parity failed: ivech_ivech_input → ivech_ivech_out")


# ===================================================================
# 6. Edge Case and Robustness Tests
# ===================================================================


def test_ivech_all_zeros():
    """Verify ivech of all-zeros vector produces all-zeros matrix.

    Ref: ivech.m:49 — zeros(K) initialisation means all-zeros input
    produces all-zeros output without any numerical drift.
    """
    for K in [1, 2, 3, 4, 5]:
        n = K * (K + 1) // 2
        v = np.zeros(n)
        result = ivech(v)
        expected = np.zeros((K, K))
        npt.assert_array_equal(result, expected)


def test_ivech_all_ones():
    """Verify ivech of all-ones vector produces all-ones matrix.

    Since all lower-triangle elements are 1, after symmetrisation every
    element of the K×K matrix should be 1.
    """
    for K in [1, 2, 3, 4]:
        n = K * (K + 1) // 2
        v = np.ones(n)
        result = ivech(v)
        expected = np.ones((K, K))
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_ivech_large_values():
    """Verify ivech handles very large values without overflow.

    Financial covariance matrices can contain large eigenvalues; ivech
    should not introduce numerical issues with large inputs.
    """
    v = np.array([1e12, 5e11, 3e11, 8e11, 2e11, 7e11])
    result = ivech(v)
    assert result.shape == (3, 3)
    npt.assert_allclose(result, result.T, atol=1.0,
                        err_msg="Symmetry violated with large values")


def test_ivech_small_values():
    """Verify ivech handles very small values without underflow.

    High-frequency realized volatility estimates can be very small.
    """
    v = np.array([1e-12, 5e-13, 3e-13, 8e-13, 2e-13, 7e-13])
    result = ivech(v)
    assert result.shape == (3, 3)
    npt.assert_allclose(result, result.T, atol=1e-18,
                        err_msg="Symmetry violated with small values")


def test_ivech_output_dtype_preserves_float64():
    """Verify output dtype is float64 for float64 input.

    Ref: MATLAB's default is double-precision; Python implementation
    should preserve float64 dtype throughout.
    """
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = ivech(v)
    assert result.dtype == np.float64, (
        f"Expected float64 output, got {result.dtype}"
    )
