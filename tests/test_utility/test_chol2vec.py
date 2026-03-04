"""Pytest test suite for mfe_toolbox.utility.chol2vec.

Tests the ``chol2vec`` function which extracts the lower triangular elements
of a K×K matrix into a K(K+1)/2 length vector using **column-major**
(Fortran) ordering — matching MATLAB's ``matrixData(sel)`` convention where
``sel = tril(true(k))``.

Source reference: utility/chol2vec.m (40 lines, Kevin Sheppard, Revision 3)

Key MATLAB behaviour:
    sel = tril(true(k));
    stackedData = matrixData(sel);

MATLAB's logical indexing extracts column-by-column:
    Column 0 lower-tri elements first, then column 1, etc.

Per AAP Section 0.7.1:
    - All numerical parity checks use atol=1e-6, rtol=1e-4
    - If MATLAB errors on invalid input, Python must raise equivalent exception
    - Every migrated function MUST pass numpy.testing.assert_allclose against
      MATLAB-generated fixtures

See Also
--------
mfe_toolbox.utility.vec2chol : Inverse operation (vector → Cholesky factor).
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.chol2vec import chol2vec
from mfe_toolbox.utility.vec2chol import vec2chol

# ---------------------------------------------------------------------------
# Tolerance constants per AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ===================================================================
# Test 1: 2×2 lower triangular — basic functionality
# ===================================================================
def test_chol2vec_2x2() -> None:
    """Extract half-vec from a 2×2 lower triangular matrix.

    Input::

        L = [[2.5, 0.0],
             [1.3, 0.8]]

    Ref: chol2vec.m:38-39 — MATLAB column-major extraction via logical indexing.
    MATLAB column-major order extracts:
        Column 0: L[0,0]=2.5, L[1,0]=1.3
        Column 1: L[1,1]=0.8
    Expected output: [2.5, 1.3, 0.8]
    """
    L = np.array([
        [2.5, 0.0],
        [1.3, 0.8],
    ])
    result = chol2vec(L)
    expected = np.array([2.5, 1.3, 0.8])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)
    # Verify return type is numpy.ndarray
    assert isinstance(result, np.ndarray)


# ===================================================================
# Test 2: 3×3 lower triangular — verifies column-major for 6 elements
# ===================================================================
def test_chol2vec_3x3() -> None:
    """Extract half-vec from a 3×3 lower triangular matrix.

    Input::

        L = [[ 1.5,  0.0,  0.0],
             [ 0.3,  2.1,  0.0],
             [-0.7,  0.5,  1.8]]

    Ref: chol2vec.m:38-39 — MATLAB column-major extraction.
    CRITICAL: MATLAB matrixData(sel) extracts column-by-column:
        Column 0: L[0,0]=1.5, L[1,0]=0.3, L[2,0]=-0.7
        Column 1: L[1,1]=2.1, L[2,1]=0.5
        Column 2: L[2,2]=1.8
    Expected output: [1.5, 0.3, -0.7, 2.1, 0.5, 1.8]
    """
    L = np.array([
        [1.5, 0.0, 0.0],
        [0.3, 2.1, 0.0],
        [-0.7, 0.5, 1.8],
    ])
    result = chol2vec(L)
    expected = np.array([1.5, 0.3, -0.7, 2.1, 0.5, 1.8])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)
    assert isinstance(result, np.ndarray)


# ===================================================================
# Test 3: Identity matrix — diagonal-only values
# ===================================================================
@pytest.mark.parametrize("k", [3, 4, 5])
def test_chol2vec_identity(k: int) -> None:
    """Extract half-vec from K×K identity matrix.

    For identity matrices, the output vector contains ones at diagonal
    positions and zeros at off-diagonal positions.

    Ref: chol2vec.m:38-39 — column-major extraction of tril(eye(K)).
    For K=3: chol2vec(eye(3)) → [1, 0, 0, 1, 0, 1]
        Column 0: (0,0)=1, (1,0)=0, (2,0)=0
        Column 1: (1,1)=1, (2,1)=0
        Column 2: (2,2)=1

    For K=4: chol2vec(eye(4)) → [1, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    For K=5: chol2vec(eye(5)) → [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    """
    L = np.eye(k)
    result = chol2vec(L)

    # Build expected vector in column-major order for an identity matrix
    # Diagonal entry in column j is at position: sum_{i=0}^{j-1}(K-i) + 0
    #   = j*K - j*(j-1)/2
    n = k * (k + 1) // 2
    expected = np.zeros(n)
    # Ref: chol2vec.m:14-19 — stacking pattern showing diagonal positions
    idx = 0
    for col in range(k):
        # Number of lower-tri elements in this column = K - col
        n_col = k - col
        # First element of this column is the diagonal entry (row == col)
        expected[idx] = 1.0
        idx += n_col

    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


# ===================================================================
# Test 4: Output length — K*(K+1)/2 for various K
# ===================================================================
@pytest.mark.parametrize("k", [2, 3, 4, 5, 10])
def test_chol2vec_output_length(k: int) -> None:
    """Verify output vector length equals K*(K+1)/2.

    For each K, construct a K×K lower triangular matrix using np.tril
    on a random matrix and verify the output length.

    Ref: chol2vec.m:11 — STACKEDDATA - A K(K+1)/2 vector of stacked data.
    """
    # Create a deterministic lower-triangular matrix
    rng = np.random.default_rng(12345 + k)
    M = np.tril(rng.standard_normal((k, k)))
    result = chol2vec(M)

    expected_length = k * (k + 1) // 2
    assert len(result) == expected_length, (
        f"Expected length {expected_length} for K={k}, got {len(result)}"
    )
    # Also check that result is 1-D
    assert result.ndim == 1


# ===================================================================
# Test 5: Non-square matrix raises ValueError
# ===================================================================
def test_chol2vec_non_square_raises() -> None:
    """Non-square input must raise ValueError.

    Ref: chol2vec.m:32 — ``if k~=l`` → error('MATRIXDATA must be a lower
    triangular matrix')

    Per AAP Section 0.7.1: If MATLAB errors on invalid input, the Python
    equivalent must raise an equivalent exception.
    """
    non_square = np.array([
        [1.0, 0.0],
        [2.0, 3.0],
        [4.0, 5.0],
    ])
    with pytest.raises(ValueError):
        chol2vec(non_square)


# ===================================================================
# Test 6: Non-lower-triangular matrix raises ValueError
# ===================================================================
def test_chol2vec_non_lower_tri_raises() -> None:
    """Matrix with non-zero upper triangular elements must raise ValueError.

    Ref: chol2vec.m:31-32 — ``pl = ~tril(true(k)); if ... any(matrixData(pl)~=0)``
    → error('MATRIXDATA must be a lower triangular matrix')

    Per AAP Section 0.7.1: If MATLAB errors on invalid input, the Python
    equivalent must raise an equivalent exception.
    """
    # 2×2 with non-zero in upper triangle
    non_lower = np.array([
        [1.0, 0.5],
        [2.0, 3.0],
    ])
    with pytest.raises(ValueError):
        chol2vec(non_lower)

    # 3×3 with non-zero in upper triangle
    non_lower_3x3 = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [4.0, 5.0, 6.0],
    ])
    # The above is actually lower triangular — let's make it non-lower
    non_lower_3x3_bad = np.array([
        [1.0, 0.1, 0.0],
        [2.0, 3.0, 0.0],
        [4.0, 5.0, 6.0],
    ])
    with pytest.raises(ValueError):
        chol2vec(non_lower_3x3_bad)


# ===================================================================
# Test 7: Roundtrip chol2vec → vec2chol
# ===================================================================
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_chol2vec_vec2chol_roundtrip(k: int) -> None:
    """Verify vec2chol(chol2vec(L)) recovers L for lower triangular L.

    This tests the inverse relationship between chol2vec and vec2chol.
    For several K values, create a known lower triangular matrix, compute
    its half-vec, reconstruct via vec2chol, and verify exact recovery.

    Ref: chol2vec.m:21 — 'See also vec2chol' — these are inverse operations.
    """
    rng = np.random.default_rng(42 + k)
    # Create a lower triangular matrix with distinct non-zero entries
    raw = rng.standard_normal((k, k)) * 2.0
    L = np.tril(raw)
    # Ensure diagonal is positive (mimics Cholesky factor)
    np.fill_diagonal(L, np.abs(np.diag(L)) + 0.5)

    # Forward: L → vector
    v = chol2vec(L)
    assert v.shape == (k * (k + 1) // 2,)

    # Inverse: vector → L_reconstructed
    L_reconstructed = vec2chol(v)

    # Verify roundtrip parity
    npt.assert_allclose(L_reconstructed, L, atol=ATOL, rtol=RTOL)


# ===================================================================
# Test 8: Column-major ordering — CRITICAL parity test
# ===================================================================
def test_chol2vec_column_major_order() -> None:
    """Verify column-major extraction order with all-distinct values.

    This is the CRITICAL test for MATLAB column-major ordering parity.

    Input: 4×4 lower triangular with values 1-10::

        L = [[ 1,  0,  0,  0],
             [ 2,  5,  0,  0],
             [ 3,  6,  8,  0],
             [ 4,  7,  9, 10]]

    Ref: chol2vec.m:38-39 — MATLAB column-major extraction:
        Column 0: L[0,0]=1, L[1,0]=2, L[2,0]=3, L[3,0]=4
        Column 1: L[1,1]=5, L[2,1]=6, L[3,1]=7
        Column 2: L[2,2]=8, L[3,2]=9
        Column 3: L[3,3]=10

    Expected: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    This ordering is an unambiguous test because every value is unique,
    so any ordering error is immediately detectable.
    """
    L = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 5.0, 0.0, 0.0],
        [3.0, 6.0, 8.0, 0.0],
        [4.0, 7.0, 9.0, 10.0],
    ])
    result = chol2vec(L)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    # Additional verification: 3×3 with distinct values
    # Ref: chol2vec.m:14-19 — stacking pattern in docstring
    L3 = np.array([
        [10.0, 0.0, 0.0],
        [20.0, 40.0, 0.0],
        [30.0, 50.0, 60.0],
    ])
    result3 = chol2vec(L3)
    # Column-major: col0=[10,20,30], col1=[40,50], col2=[60]
    expected3 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    npt.assert_allclose(result3, expected3, atol=ATOL, rtol=RTOL)


# ===================================================================
# Test 9: Fixture parity — MATLAB reference comparison
# ===================================================================
def test_chol2vec_fixture_parity(utility_fixture_dir: Path) -> None:
    """Compare chol2vec output against MATLAB/Octave reference fixtures.

    Loads the fixture file ``tests/fixtures/utility/chol2vec.npy`` containing
    12 test cases generated by Octave 8.4.0 from the original MATLAB
    ``utility/chol2vec.m``.

    Each test case contains ``testN_input_matrix`` and ``testN_expected_vector``
    entries which are validated against the Python implementation at the
    standard tolerance (atol=1e-6, rtol=1e-4) per AAP Section 0.7.1.

    The fixture is skipped if the file does not exist (e.g. in CI before
    fixture generation).
    """
    fixture_path = utility_fixture_dir / "chol2vec.npy"
    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found: {fixture_path}")

    # Load the fixture — object-dtype array wrapping a dict
    fixture_data = np.load(fixture_path, allow_pickle=True).item()

    # Iterate over all numbered test cases in the fixture
    test_idx = 1
    while True:
        input_key = f"test{test_idx}_input_matrix"
        expected_key = f"test{test_idx}_expected_vector"

        if input_key not in fixture_data or expected_key not in fixture_data:
            break

        input_matrix = np.asarray(fixture_data[input_key], dtype=float)
        expected_vector = np.asarray(fixture_data[expected_key], dtype=float)

        result = chol2vec(input_matrix)

        npt.assert_allclose(
            result,
            expected_vector,
            atol=ATOL,
            rtol=RTOL,
            err_msg=(
                f"Fixture parity failure for test case {test_idx}: "
                f"K={input_matrix.shape[0]}"
            ),
        )

        test_idx += 1

    # Ensure we actually ran at least one test case
    assert test_idx > 1, "No test cases found in fixture file"
