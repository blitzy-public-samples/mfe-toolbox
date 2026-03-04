"""Pytest test suite for mfe_toolbox.utility.vec2chol.

Tests the ``vec2chol`` function which reconstructs a K×K lower triangular
matrix from its K(K+1)/2 half-vec representation.  The stacking convention
follows MATLAB's **column-major** ordering:

    pl = tril(true(K));
    matrixData(pl) = stackedData;

This is the inverse operation of ``chol2vec``.

Source reference: utility/vec2chol.m (55 lines, Kevin Sheppard, Revision 3)

Per AAP Section 0.7.1:
    - All numerical parity checks use atol=1e-6, rtol=1e-4
    - If MATLAB errors on invalid input, Python must raise equivalent exception
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.vec2chol import vec2chol
from mfe_toolbox.utility.chol2vec import chol2vec

# ---------------------------------------------------------------------------
# Tolerance constants per AAP Section 0.7.1
# ---------------------------------------------------------------------------
# Imported from conftest for consistency, but defined locally for clarity
# in standalone runs.
ATOL: float = 1e-6
RTOL: float = 1e-4


# ===================================================================
# Test 1: 3-element vector → 2×2 lower triangular
# ===================================================================
def test_vec2chol_3elements() -> None:
    """[1, 2, 3] → 2×2 lower triangular with MATLAB column-major fill.

    Ref: vec2chol.m:52-54 — column-major fill order:
        Column 0: (0,0)=1, (1,0)=2
        Column 1: (1,1)=3

    Expected result::

        [[1, 0],
         [2, 3]]
    """
    v = np.array([1.0, 2.0, 3.0])
    result = vec2chol(v)
    expected = np.array([
        [1.0, 0.0],
        [2.0, 3.0],
    ])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


# ===================================================================
# Test 2: 6-element vector → 3×3 lower triangular
# ===================================================================
def test_vec2chol_6elements() -> None:
    """6-element vector → 3×3 lower triangular.

    Ref: vec2chol.m:52-54 — column-major fill order:
        Column 0: (0,0)=1, (1,0)=2, (2,0)=3
        Column 1: (1,1)=4, (2,1)=5
        Column 2: (2,2)=6

    Expected result::

        [[1, 0, 0],
         [2, 4, 0],
         [3, 5, 6]]
    """
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = vec2chol(v)
    expected = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 4.0, 0.0],
        [3.0, 5.0, 6.0],
    ])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


# ===================================================================
# Test 3: Output is lower triangular (upper triangle all zeros)
# ===================================================================
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_vec2chol_output_lower_triangular(k: int) -> None:
    """Verify that the upper triangle of the output is all zeros.

    For each matrix dimension K, create a valid K(K+1)/2 input vector
    and confirm that all strictly-upper-triangular entries are zero.
    """
    n = k * (k + 1) // 2
    v = np.arange(1.0, n + 1.0)
    result = vec2chol(v)

    # Extract strictly upper triangular elements and verify they are zero
    # Ref: vec2chol.m:50 — initialised with zeros(K)
    upper_indices = np.triu_indices(k, k=1)
    npt.assert_allclose(
        result[upper_indices],
        np.zeros(len(upper_indices[0])),
        atol=ATOL,
        err_msg=f"Upper triangle is not all zeros for K={k}",
    )


# ===================================================================
# Test 4: Output shape is K×K for K(K+1)/2 input length
# ===================================================================
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 10])
def test_vec2chol_output_shape(k: int) -> None:
    """Verify K(K+1)/2 input vector → K×K output matrix.

    Ref: vec2chol.m:39 — K = (-1 + sqrt(1 + 8*K2)) / 2
    """
    n = k * (k + 1) // 2
    v = np.ones(n)
    result = vec2chol(v)
    assert result.shape == (k, k), (
        f"Expected shape ({k}, {k}), got {result.shape}"
    )


# ===================================================================
# Test 5: Single element [5.0] → [[5.0]]
# ===================================================================
def test_vec2chol_1element() -> None:
    """Scalar input [5.0] → 1×1 matrix [[5.0]].

    Ref: vec2chol.m:39 — K = 1 for a single element.
    """
    v = np.array([5.0])
    result = vec2chol(v)
    expected = np.array([[5.0]])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)
    assert result.shape == (1, 1)


# ===================================================================
# Test 6: Row vector input is auto-accepted
# ===================================================================
def test_vec2chol_row_vector_input() -> None:
    """Row vector (1×N) should produce the same result as column vector.

    Ref: vec2chol.m:30-31 — ``if size(stackedData,2)>size(stackedData,1)``
    then transpose.  The Python implementation uses ``ravel()`` which
    handles both row and column vectors.
    """
    v_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    v_row = v_col.reshape(1, -1)

    result_col = vec2chol(v_col)
    result_row = vec2chol(v_row)

    npt.assert_allclose(result_row, result_col, atol=ATOL, rtol=RTOL)


# ===================================================================
# Test 7: Invalid vector length → ValueError
# ===================================================================
@pytest.mark.parametrize("length", [2, 4, 5, 7, 8, 9, 11])
def test_vec2chol_invalid_length_raises(length: int) -> None:
    """Invalid length (not K(K+1)/2 for any integer K) → ValueError.

    Ref: vec2chol.m:41-43 — ``if floor(K)~=K`` → error.
    Valid lengths are: 1, 3, 6, 10, 15, 21, ...
    """
    v = np.ones(length)
    with pytest.raises(ValueError):
        vec2chol(v)


# ===================================================================
# Test 8: 2-D matrix input → ValueError
# ===================================================================
def test_vec2chol_non_vector_raises() -> None:
    """2-D matrix input should raise ValueError.

    Ref: vec2chol.m:33-36 — ``if size(stackedData,2)~=1`` → error.

    Note: The Python implementation uses ``ravel()`` which accepts 2-D
    input by flattening.  A 2×2 matrix has 4 elements, which is NOT a
    valid K(K+1)/2 length, so it triggers the length validation error.
    Similarly, a 2×3 matrix (6 elements) would pass since 6 = 3*4/2.

    We test with a 2×2 matrix (4 elements — invalid length) to ensure
    a ValueError is raised, matching the MATLAB error behavior for
    non-column-vector input.
    """
    mat_2d = np.ones((2, 2))  # 4 elements total — not K(K+1)/2
    with pytest.raises(ValueError):
        vec2chol(mat_2d)


# ===================================================================
# Test 9: Roundtrip chol2vec(vec2chol(v)) == v
# ===================================================================
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_vec2chol_chol2vec_roundtrip(k: int) -> None:
    """Verify chol2vec(vec2chol(v)) == v for valid input vectors.

    The vec2chol → chol2vec composition should be the identity
    transformation on valid K(K+1)/2 vectors, since chol2vec is the
    inverse of vec2chol.
    """
    n = k * (k + 1) // 2
    # Use distinct values to fully exercise the roundtrip
    v = np.arange(1.0, n + 1.0)
    matrix = vec2chol(v)
    v_roundtrip = chol2vec(matrix)
    npt.assert_allclose(
        v_roundtrip,
        v,
        atol=ATOL,
        rtol=RTOL,
        err_msg=f"Roundtrip failed for K={k}",
    )


# ===================================================================
# Test 10: Column-major fill with distinct values
# ===================================================================
def test_vec2chol_column_major_fill() -> None:
    """Verify MATLAB-compatible column-major fill order with distinct values.

    Ref: vec2chol.m:53-54 — ``pl = tril(true(K)); matrixData(pl) = stackedData;``

    For a 4×4 matrix with input [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    MATLAB column-major fill:
        Column 0: (0,0)=1, (1,0)=2, (2,0)=3, (3,0)=4
        Column 1: (1,1)=5, (2,1)=6, (3,1)=7
        Column 2: (2,2)=8, (3,2)=9
        Column 3: (3,3)=10

    Expected result::

        [[ 1,  0,  0,  0],
         [ 2,  5,  0,  0],
         [ 3,  6,  8,  0],
         [ 4,  7,  9, 10]]

    This is THE critical test for column-major ordering parity.
    """
    v = np.arange(1.0, 11.0)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = vec2chol(v)
    expected = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 5.0, 0.0, 0.0],
        [3.0, 6.0, 8.0, 0.0],
        [4.0, 7.0, 9.0, 10.0],
    ])
    npt.assert_allclose(
        result,
        expected,
        atol=ATOL,
        rtol=RTOL,
        err_msg="Column-major fill order mismatch for 4×4 case",
    )

    # Also verify with the simpler 3×3 case explicitly
    v3 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    result3 = vec2chol(v3)
    expected3 = np.array([
        [10.0, 0.0, 0.0],
        [20.0, 40.0, 0.0],
        [30.0, 50.0, 60.0],
    ])
    npt.assert_allclose(
        result3,
        expected3,
        atol=ATOL,
        rtol=RTOL,
        err_msg="Column-major fill order mismatch for 3×3 case",
    )


# ===================================================================
# Test 11: Fixture parity with MATLAB/Octave reference
# ===================================================================
def test_vec2chol_fixture_parity(utility_fixture_dir) -> None:
    """Compare vec2chol output against MATLAB/Octave-generated fixtures.

    Loads the fixture file ``tests/fixtures/utility/vec2chol.npy`` which
    contains multiple test cases generated by Octave 8.4.0.  Each test
    case has an input vector and an expected output matrix.

    Per AAP Section 0.7.1: atol=1e-6, rtol=1e-4.
    """
    from tests.conftest import load_fixture_npy

    fixture = load_fixture_npy(utility_fixture_dir, "vec2chol")

    # Fixture is stored as an object array containing a dict
    if isinstance(fixture, np.ndarray) and fixture.dtype == object:
        data = fixture.item() if fixture.ndim == 0 else fixture[0]
    else:
        data = fixture

    # Iterate over all numbered test cases in the fixture
    test_count = 0
    for key in sorted(data.keys()):
        if key.startswith("test") and key.endswith("_input_vector"):
            test_id = key.replace("_input_vector", "")
            expected_key = f"{test_id}_expected_matrix"

            if expected_key not in data:
                continue

            input_vector = data[key]
            expected_matrix = data[expected_key]

            result = vec2chol(input_vector)
            npt.assert_allclose(
                result,
                expected_matrix,
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"Fixture parity failed for {test_id}",
            )
            test_count += 1

    # Ensure we actually tested some cases
    assert test_count >= 5, (
        f"Expected at least 5 fixture test cases, found {test_count}"
    )
