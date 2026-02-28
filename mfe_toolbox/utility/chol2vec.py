"""
Transform a lower triangular Cholesky factor into its half-vec representation.

This module provides the ``chol2vec`` function which extracts the lower
triangular elements of a K×K matrix into a K(K+1)/2 length vector, stacking
elements column-by-column (matching MATLAB column-major convention).

Migrated from: utility/chol2vec.m
Author: Kevin Sheppard (original MATLAB), migrated to Python 3.12

See Also
--------
mfe_toolbox.utility.vec2chol : Inverse operation — vector to Cholesky factor.

Notes
-----
MATLAB uses column-major (Fortran) order for logical indexing, so the
stacking convention is:

.. code-block:: text

    [ data(1)  0          0          ...  0
      data(2)  data(K+1)  0          ...  0
      data(3)  data(K+2)  data(2K)   ...  0
      ...      ...        ...        ...  0
      data(K)  data(2K-1) ...        ...  data(K(K+1)/2) ]

In Python, this is achieved by extracting the upper triangle of the
**transposed** matrix, which replicates MATLAB's column-major extraction
of the lower triangle from the original matrix.
"""

import numpy as np


def chol2vec(matrix_data: np.ndarray) -> np.ndarray:
    """
    Transform a lower triangular matrix to its half-vec representation.

    Extracts the lower triangular elements (including the diagonal) of a
    square lower triangular matrix and stacks them into a one-dimensional
    vector in column-major order, matching the MATLAB convention used in
    the original MFE Toolbox.

    Parameters
    ----------
    matrix_data : numpy.ndarray
        K × K lower triangular matrix (all elements above the main diagonal
        must be exactly zero).

    Returns
    -------
    stacked_data : numpy.ndarray
        One-dimensional array of length K(K+1)/2 containing the lower
        triangular elements stacked column-by-column.

    Raises
    ------
    ValueError
        If ``matrix_data`` is not two-dimensional.
    ValueError
        If ``matrix_data`` is not square.
    ValueError
        If ``matrix_data`` is not lower triangular (i.e., any element above
        the main diagonal is non-zero).

    Examples
    --------
    >>> import numpy as np
    >>> L = np.array([[1.0, 0.0, 0.0],
    ...               [2.0, 3.0, 0.0],
    ...               [4.0, 5.0, 6.0]])
    >>> chol2vec(L)
    array([1., 2., 4., 3., 5., 6.])

    Notes
    -----
    The element ordering follows MATLAB's column-major convention:

    - First K elements come from column 1 (rows 1..K)
    - Next K-1 elements from column 2 (rows 2..K)
    - And so on until the last element from column K (row K)

    This is implemented in Python by extracting the upper triangle of the
    transposed matrix using ``np.triu_indices``, which traverses row-by-row
    in the transposed layout — equivalent to column-by-column in the
    original layout.

    References
    ----------
    Ref: chol2vec.m — Kevin Sheppard, Revision 3, 2/1/2008

    See Also
    --------
    mfe_toolbox.utility.vec2chol : Inverse operation.
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: chol2vec.m:30-34 — MATLAB checks size and upper triangle zeros
    # ------------------------------------------------------------------

    # Ensure input is a numpy array
    matrix_data = np.asarray(matrix_data, dtype=float)

    # Check that input is 2-D
    if matrix_data.ndim != 2:
        raise ValueError(
            "MATRIXDATA must be a 2-dimensional array, "
            f"got {matrix_data.ndim} dimensions"
        )

    # Ref: chol2vec.m:30 — [k,l] = size(matrixData)
    k, l = matrix_data.shape  # noqa: E741 (using 'l' to match MATLAB variable name)

    # Ref: chol2vec.m:32 — if k~=l
    if k != l:
        raise ValueError(
            "MATRIXDATA must be a square matrix, "
            f"got shape ({k}, {l})"
        )

    # Ref: chol2vec.m:31-32 — pl = ~tril(true(k)); if any(matrixData(pl)~=0)
    # Build boolean masks mirroring MATLAB's tril(true(k)) pattern.
    # np.ones creates the all-true matrix; np.tril extracts the lower triangle.
    lower_mask = np.tril(np.ones((k, k), dtype=bool))
    upper_mask = ~lower_mask  # Ref: chol2vec.m:31 — pl = ~tril(true(k))
    if not np.allclose(matrix_data[upper_mask], 0.0, atol=0.0, rtol=0.0):
        raise ValueError("MATRIXDATA must be a lower triangular matrix")

    # ------------------------------------------------------------------
    # Extract lower triangular elements in column-major order
    # Ref: chol2vec.m:38-39 — sel = tril(true(k)); stackedData = matrixData(sel)
    # ------------------------------------------------------------------
    # MATLAB's logical indexing extracts in column-major (Fortran) order.
    # In Python, np.tril_indices extracts row-by-row (C order), which gives
    # a different element ordering. To replicate MATLAB's column-major
    # extraction of the lower triangle, we extract the upper triangle of
    # the transposed matrix: L.T is upper-triangular, and np.triu_indices
    # traverses it row-by-row, which corresponds to column-by-column in L.
    # Ref: chol2vec.m:38 — sel = tril(true(k))
    # Ref: chol2vec.m:39 — stackedData = matrixData(sel)
    stacked_data = matrix_data.T[np.triu_indices(k)]

    return stacked_data
