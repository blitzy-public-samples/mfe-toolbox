"""
Pytest tests for ``mfe_toolbox.utility.vech`` — half-vectorization operator.

Tests cover:
- Basic functionality with known 2×2, 3×3, 4×4 symmetric matrices
- Output length verification parametrized across matrix sizes
- Edge cases: 1×1 scalar, identity, zeros, negative values
- Error handling: non-square, non-symmetric, wrong dimensionality
- Round-trip parity: ``ivech(vech(S)) == S`` for random symmetric matrices
- MATLAB/Octave fixture parity: comparison against generated reference outputs

All numerical comparisons use ``numpy.testing.assert_allclose`` with
``atol=1e-6``, ``rtol=1e-4`` per AAP Section 0.7.1.

Column-major ordering — CRITICAL:
    MATLAB's ``matrixData(tril(true(k)))`` extracts elements in column-major
    (Fortran) order.  For a 3×3 matrix the extraction order is:
        Col 0: (0,0), (1,0), (2,0)
        Col 1: (1,1), (2,1)
        Col 2: (2,2)
    yielding [mat[0,0], mat[1,0], mat[2,0], mat[1,1], mat[2,1], mat[2,2]].
    All expected values in these tests follow this ordering.

Source reference: utility/vech.m (38 lines), Author: Kevin Sheppard
"""

import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

from mfe_toolbox.utility.vech import vech
from mfe_toolbox.utility.ivech import ivech
from tests.conftest import ATOL, RTOL, load_fixture_npy


# ---------------------------------------------------------------------------
# Basic Functionality Tests
# ---------------------------------------------------------------------------


class TestVechBasicFunctionality:
    """Tests for basic vech operation on known symmetric matrices."""

    def test_vech_2x2(self):
        """Test vech with 2×2 symmetric matrix → 3-element vector.

        MATLAB column-major extraction order for 2×2:
            Col 0: (0,0)=1, (1,0)=2
            Col 1: (1,1)=3
        Expected: [1, 2, 3]

        Ref: vech.m:37-38 — sel = tril(true(k)); stackedData = matrixData(sel);
        """
        mat = np.array([[1.0, 2.0],
                        [2.0, 3.0]])
        result = vech(mat)
        expected = np.array([1.0, 2.0, 3.0])
        # vech returns column vector shape (3, 1); flatten for comparison
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)

    def test_vech_3x3(self):
        """Test vech with 3×3 symmetric matrix → 6-element vector.

        MATLAB column-major extraction order for 3×3:
            Col 0: (0,0)=1, (1,0)=2, (2,0)=3
            Col 1: (1,1)=5, (2,1)=6
            Col 2: (2,2)=9
        Expected: [1, 2, 3, 5, 6, 9]
        """
        mat = np.array([[1.0, 2.0, 3.0],
                        [2.0, 5.0, 6.0],
                        [3.0, 6.0, 9.0]])
        result = vech(mat)
        expected = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 9.0])
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)

    def test_vech_4x4(self):
        """Test vech with 4×4 identity matrix → 10-element vector.

        MATLAB column-major extraction order for eye(4):
            Col 0: (0,0)=1, (1,0)=0, (2,0)=0, (3,0)=0
            Col 1: (1,1)=1, (2,1)=0, (3,1)=0
            Col 2: (2,2)=1, (3,2)=0
            Col 3: (3,3)=1
        Expected: [1, 0, 0, 0, 1, 0, 0, 1, 0, 1]
        """
        mat = np.eye(4)
        result = vech(mat)
        expected = np.array([1.0, 0.0, 0.0, 0.0,
                             1.0, 0.0, 0.0,
                             1.0, 0.0,
                             1.0])
        assert result.size == 10
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 10])
    def test_vech_output_length(self, k):
        """For a K×K matrix, vech output has exactly K*(K+1)/2 elements.

        Ref: vech.m — STACKEDDATA is a K(K+1)/2 vector of stacked data.
        """
        mat = np.eye(k)  # Identity is always symmetric
        result = vech(mat)
        expected_length = k * (k + 1) // 2
        assert result.size == expected_length, (
            f"Expected {expected_length} elements for K={k}, got {result.size}"
        )


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestVechEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_vech_1x1(self):
        """Scalar matrix [[5.0]] → [5.0].

        The smallest possible symmetric matrix has K=1 and K*(K+1)/2 = 1.
        """
        mat = np.array([[5.0]])
        result = vech(mat)
        expected = np.array([5.0])
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)

    def test_vech_identity_3x3(self):
        """Identity 3×3 → ones at diagonal positions, zeros elsewhere.

        Column-major extraction of eye(3):
            Col 0: (0,0)=1, (1,0)=0, (2,0)=0
            Col 1: (1,1)=1, (2,1)=0
            Col 2: (2,2)=1
        Expected: [1, 0, 0, 1, 0, 1]
        """
        mat = np.eye(3)
        result = vech(mat)
        expected = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_vech_zeros(self, k):
        """All-zero K×K matrix → all-zero vector of length K*(K+1)/2."""
        mat = np.zeros((k, k))
        result = vech(mat)
        expected = np.zeros(k * (k + 1) // 2)
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)

    def test_vech_negative_values(self):
        """Matrix with negative elements extracts correctly.

        [[-1, -2], [-2, -4]] → [-1, -2, -4]
        Column-major: (0,0)=-1, (1,0)=-2, (1,1)=-4
        """
        mat = np.array([[-1.0, -2.0],
                        [-2.0, -4.0]])
        result = vech(mat)
        expected = np.array([-1.0, -2.0, -4.0])
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)

    def test_vech_ones_matrix(self):
        """All-ones symmetric matrix → all-ones vector.

        ones(K) is trivially symmetric; all extracted elements are 1.0.
        """
        k = 3
        mat = np.ones((k, k))
        result = vech(mat)
        expected = np.ones(k * (k + 1) // 2)
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)

    def test_vech_return_type_is_ndarray(self):
        """Verify vech returns numpy.ndarray (not list or other type).

        Per AAP Section 0.7.1: All return types must be numpy.ndarray.
        """
        mat = np.eye(3)
        result = vech(mat)
        assert isinstance(result, np.ndarray)

    def test_vech_column_vector_shape(self):
        """Verify vech returns column vector with shape (K*(K+1)/2, 1).

        Ref: MATLAB vech returns a column vector; Python implementation
        mirrors this with reshape(-1, 1).
        """
        k = 3
        mat = np.eye(k)
        result = vech(mat)
        expected_rows = k * (k + 1) // 2
        assert result.shape == (expected_rows, 1), (
            f"Expected shape ({expected_rows}, 1), got {result.shape}"
        )

    def test_vech_large_matrix(self):
        """Test vech with a larger (10×10) random symmetric positive-definite matrix."""
        rng = np.random.default_rng(99999)
        A = rng.standard_normal((10, 10))
        mat = A @ A.T  # Guaranteed symmetric positive-definite
        result = vech(mat)
        expected_length = 10 * 11 // 2  # = 55
        assert result.size == expected_length
        assert result.shape == (expected_length, 1)

    def test_vech_float32_input(self):
        """Verify vech handles float32 input without error."""
        mat = np.eye(3, dtype=np.float32)
        result = vech(mat)
        assert result.size == 6

    def test_vech_integer_input(self):
        """Verify vech handles integer matrix input via np.asarray coercion."""
        mat = np.array([[1, 2], [2, 4]])
        result = vech(mat)
        expected = np.array([1, 2, 4], dtype=float)
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Column-Major Ordering Verification
# ---------------------------------------------------------------------------


class TestVechColumnMajorOrdering:
    """Explicitly verify column-major extraction order matches MATLAB.

    MATLAB extracts lower-triangular elements using
    ``sel = tril(true(k)); stackedData = matrixData(sel);``
    which traverses down each column first (Fortran/column-major order).

    These tests use matrices with distinct element values to unambiguously
    verify the extraction ordering.
    """

    def test_column_major_3x3_distinct_values(self):
        """3×3 matrix with all distinct lower-triangle elements.

        Matrix:
            [[10, 20, 30],
             [20, 50, 60],
             [30, 60, 90]]

        Column-major extraction:
            Col 0: (0,0)=10, (1,0)=20, (2,0)=30
            Col 1: (1,1)=50, (2,1)=60
            Col 2: (2,2)=90
        Expected: [10, 20, 30, 50, 60, 90]
        """
        mat = np.array([[10.0, 20.0, 30.0],
                        [20.0, 50.0, 60.0],
                        [30.0, 60.0, 90.0]])
        result = vech(mat)
        expected = np.array([10.0, 20.0, 30.0, 50.0, 60.0, 90.0])
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)

    def test_column_major_4x4_sequential(self):
        """4×4 symmetric matrix constructed from sequential lower-tri values.

        Lower-triangle column-major indices for 4×4:
            Col 0: (0,0), (1,0), (2,0), (3,0) → values 1, 2, 3, 4
            Col 1: (1,1), (2,1), (3,1)        → values 5, 6, 7
            Col 2: (2,2), (3,2)               → values 8, 9
            Col 3: (3,3)                       → value 10

        Build symmetric matrix and verify extraction order = [1..10].
        """
        k = 4
        # Build lower triangle in column-major order
        mat = np.zeros((k, k))
        val = 1.0
        for col in range(k):
            for row in range(col, k):
                mat[row, col] = val
                mat[col, row] = val  # Symmetrise
                val += 1.0

        result = vech(mat)
        expected = np.arange(1.0, 11.0)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        npt.assert_allclose(result.ravel(), expected, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestVechErrorHandling:
    """Tests for input validation and error raising.

    Per AAP Section 0.7.1: If a MATLAB function errors on invalid input,
    the Python equivalent MUST raise an equivalent exception.
    Ref: vech.m:31 — if k~=l || any(any(matrixData~=matrixData')) → error(...)
    """

    def test_vech_non_square_raises(self):
        """Non-square matrix (3×2) raises ValueError.

        Ref: vech.m:31 — if k~=l → error('MATRIXDATA must be a symmetric matrix')
        """
        mat = np.ones((3, 2))
        with pytest.raises(ValueError, match="non-square|symmetric"):
            vech(mat)

    def test_vech_non_symmetric_raises(self):
        """Non-symmetric square matrix raises ValueError.

        Ref: vech.m:31 — any(any(matrixData~=matrixData')) → error(...)
        """
        mat = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
        with pytest.raises(ValueError, match="symmetric"):
            vech(mat)

    def test_vech_non_symmetric_3x3_raises(self):
        """Non-symmetric 3×3 matrix raises ValueError.

        Only one off-diagonal pair differs: mat[0,1]=2, mat[1,0]=99.
        """
        mat = np.array([[1.0, 2.0, 3.0],
                        [99.0, 5.0, 6.0],
                        [3.0, 6.0, 9.0]])
        with pytest.raises(ValueError, match="symmetric"):
            vech(mat)

    def test_vech_1d_array_raises(self):
        """1-D array input raises ValueError (not 2-D).

        Ref: vech.m expects a 2-D matrix; 1-D input is invalid.
        """
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            vech(arr)

    def test_vech_3d_array_raises(self):
        """3-D array input raises ValueError (not 2-D)."""
        arr = np.ones((2, 2, 2))
        with pytest.raises(ValueError):
            vech(arr)

    def test_vech_tall_rectangular_raises(self):
        """Tall rectangular matrix (4×2) raises ValueError."""
        mat = np.ones((4, 2))
        with pytest.raises(ValueError):
            vech(mat)

    def test_vech_wide_rectangular_raises(self):
        """Wide rectangular matrix (2×4) raises ValueError."""
        mat = np.ones((2, 4))
        with pytest.raises(ValueError):
            vech(mat)


# ---------------------------------------------------------------------------
# Round-Trip Parity Tests
# ---------------------------------------------------------------------------


class TestVechIvechRoundTrip:
    """Round-trip tests: ``ivech(vech(S)) == S`` for random symmetric matrices.

    Uses deterministic seed for reproducibility.
    """

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_vech_ivech_roundtrip_random(self, k):
        """ivech(vech(S)) reconstructs the original symmetric K×K matrix.

        Generates a random symmetric matrix using seed 12345 and verifies
        that the round-trip vech → ivech produces the original matrix
        within numerical tolerance.
        """
        rng = np.random.default_rng(12345)
        A = rng.standard_normal((k, k))
        S = (A + A.T) / 2.0  # Symmetric matrix
        v = vech(S)
        S_reconstructed = ivech(v)
        npt.assert_allclose(S_reconstructed, S, atol=ATOL, rtol=RTOL,
                            err_msg=f"Round-trip failed for K={k}")

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_vech_ivech_roundtrip_positive_definite(self, k):
        """ivech(vech(S)) == S for random positive-definite K×K matrix.

        Constructs S = A @ A.T to guarantee positive definiteness.
        """
        rng = np.random.default_rng(54321)
        A = rng.standard_normal((k, k))
        S = A @ A.T  # Symmetric positive-definite
        v = vech(S)
        S_reconstructed = ivech(v)
        npt.assert_allclose(S_reconstructed, S, atol=ATOL, rtol=RTOL,
                            err_msg=f"PD round-trip failed for K={k}")

    def test_vech_ivech_roundtrip_identity(self):
        """ivech(vech(I)) == I for identity matrix."""
        k = 4
        I = np.eye(k)
        v = vech(I)
        I_reconstructed = ivech(v)
        npt.assert_allclose(I_reconstructed, I, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# MATLAB/Octave Fixture Parity Tests
# ---------------------------------------------------------------------------


class TestVechFixtureParity:
    """Tests against MATLAB/Octave-generated reference fixture files.

    Per AAP Section 0.7.1: Every migrated function MUST pass
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
    against MATLAB-generated fixtures.

    Fixture files in tests/fixtures/utility/:
    - vech.npy: Object array with dict containing multiple test cases
      (input_matrix_2x2, expected_output_2x2, input_matrix_3x3,
       expected_output_3x3, input_matrix (4×4), expected_output,
       input_matrix_5x5, expected_output_5x5)
    - vech_A_sym.npy: 3×3 symmetric test matrix
    - vech_vech_out.npy: Expected vech output for vech_A_sym
    """

    def test_vech_fixture_2x2(self, utility_fixture_dir):
        """Test vech against MATLAB reference: 2×2 case from vech.npy."""
        fixture = load_fixture_npy(utility_fixture_dir, "vech")
        data = fixture.item() if fixture.dtype == object else None
        if data is None or not isinstance(data, dict):
            pytest.skip("Legacy fixture format not recognized as dict")

        if "input_matrix_2x2" not in data or "expected_output_2x2" not in data:
            pytest.skip("2×2 test case not found in fixture")

        result = vech(data["input_matrix_2x2"])
        npt.assert_allclose(
            result.ravel(), data["expected_output_2x2"],
            atol=ATOL, rtol=RTOL,
            err_msg="MATLAB parity failed for 2×2 fixture case"
        )

    def test_vech_fixture_3x3(self, utility_fixture_dir):
        """Test vech against MATLAB reference: 3×3 case from vech.npy."""
        fixture = load_fixture_npy(utility_fixture_dir, "vech")
        data = fixture.item() if fixture.dtype == object else None
        if data is None or not isinstance(data, dict):
            pytest.skip("Legacy fixture format not recognized as dict")

        if "input_matrix_3x3" not in data or "expected_output_3x3" not in data:
            pytest.skip("3×3 test case not found in fixture")

        result = vech(data["input_matrix_3x3"])
        npt.assert_allclose(
            result.ravel(), data["expected_output_3x3"],
            atol=ATOL, rtol=RTOL,
            err_msg="MATLAB parity failed for 3×3 fixture case"
        )

    def test_vech_fixture_4x4_primary(self, utility_fixture_dir):
        """Test vech against MATLAB reference: primary 4×4 case from vech.npy.

        This is the primary test case generated with rng(42) in Octave.
        """
        fixture = load_fixture_npy(utility_fixture_dir, "vech")
        data = fixture.item() if fixture.dtype == object else None
        if data is None or not isinstance(data, dict):
            pytest.skip("Legacy fixture format not recognized as dict")

        if "input_matrix" not in data or "expected_output" not in data:
            pytest.skip("Primary 4×4 test case not found in fixture")

        result = vech(data["input_matrix"])
        npt.assert_allclose(
            result.ravel(), data["expected_output"],
            atol=ATOL, rtol=RTOL,
            err_msg="MATLAB parity failed for primary 4×4 fixture case"
        )

    def test_vech_fixture_5x5(self, utility_fixture_dir):
        """Test vech against MATLAB reference: 5×5 case from vech.npy."""
        fixture = load_fixture_npy(utility_fixture_dir, "vech")
        data = fixture.item() if fixture.dtype == object else None
        if data is None or not isinstance(data, dict):
            pytest.skip("Legacy fixture format not recognized as dict")

        if "input_matrix_5x5" not in data or "expected_output_5x5" not in data:
            pytest.skip("5×5 test case not found in fixture")

        result = vech(data["input_matrix_5x5"])
        npt.assert_allclose(
            result.ravel(), data["expected_output_5x5"],
            atol=ATOL, rtol=RTOL,
            err_msg="MATLAB parity failed for 5×5 fixture case"
        )

    def test_vech_fixture_a_sym(self, utility_fixture_dir):
        """Test vech against vech_A_sym.npy input / vech_vech_out.npy output.

        These are separate fixture files generated by the fixture conversion
        pipeline (scripts/convert_fixtures.py).
        """
        input_data = load_fixture_npy(utility_fixture_dir, "vech_A_sym")
        expected_output = load_fixture_npy(utility_fixture_dir, "vech_vech_out")

        result = vech(input_data)
        npt.assert_allclose(
            result.ravel(), expected_output.ravel(),
            atol=ATOL, rtol=RTOL,
            err_msg="MATLAB parity failed for vech_A_sym fixture"
        )

    def test_vech_fixture_all_cases_output_lengths(self, utility_fixture_dir):
        """Verify output lengths for all fixture cases match K*(K+1)/2."""
        fixture = load_fixture_npy(utility_fixture_dir, "vech")
        data = fixture.item() if fixture.dtype == object else None
        if data is None or not isinstance(data, dict):
            pytest.skip("Legacy fixture format not recognized as dict")

        test_cases = [
            ("input_matrix_2x2", "expected_output_2x2"),
            ("input_matrix_3x3", "expected_output_3x3"),
            ("input_matrix", "expected_output"),
            ("input_matrix_5x5", "expected_output_5x5"),
        ]
        for input_key, output_key in test_cases:
            if input_key in data and output_key in data:
                mat = data[input_key]
                k = mat.shape[0]
                expected_len = k * (k + 1) // 2
                result = vech(mat)
                assert result.size == expected_len, (
                    f"Output length mismatch for {input_key}: "
                    f"expected {expected_len}, got {result.size}"
                )
                assert len(data[output_key]) == expected_len, (
                    f"Fixture length mismatch for {output_key}: "
                    f"expected {expected_len}, got {len(data[output_key])}"
                )
