"""
Half-vectorization (vech) operator for symmetric matrices.

Extracts the lower triangular elements (including the diagonal) of a square
symmetric matrix into a column vector using column-major (Fortran) ordering,
matching the standard mathematical definition of the vech operator and
maintaining numerical parity with the original MATLAB implementation.

This module is one of the most foundational utility functions in the MFE Toolbox,
imported by multivariate GARCH models, correlation utilities, and many other
modules throughout the package.

Migrated from: utility/vech.m — Author: Kevin Sheppard
"""

import numpy as np


def vech(mat: np.ndarray) -> np.ndarray:
    """
    Half-vectorization of a symmetric matrix.

    Extracts the lower triangular elements (including diagonal) of a square
    symmetric matrix into a K(K+1)/2 x 1 column vector, using column-major
    ordering to match MATLAB's extraction convention.

    Parameters
    ----------
    mat : numpy.ndarray
        K x K square symmetric matrix. Input is converted to a numpy array
        via ``np.asarray`` if not already one. Symmetry is verified using
        ``numpy.allclose`` for numerical robustness with floating-point data.

    Returns
    -------
    v : numpy.ndarray
        K(K+1)/2 x 1 column vector containing the lower triangular elements
        extracted in column-major order.

    Raises
    ------
    ValueError
        If the input is not a 2-D array, is not square, or is not symmetric
        (within floating-point tolerance).

    See Also
    --------
    ivech : Inverse half-vectorization, reconstructing a symmetric matrix
        from a vech vector.

    Notes
    -----
    Elements are extracted in column-major order to match the standard
    mathematical definition of the vech operator and to maintain parity
    with the MATLAB implementation (``vech.m``).

    For a 3x3 symmetric matrix, the stacking order is::

        [[a, b, c],         vech
         [b, e, f],   ------------->   [a, b, c, e, f, i]
         [c, f, i]]

    This corresponds to extracting column-by-column from the lower triangle:

    - Column 0: (0,0), (1,0), (2,0) → a, b, c
    - Column 1: (1,1), (2,1)        → e, f
    - Column 2: (2,2)               → i

    The implementation uses ``numpy.triu_indices`` with swapped row/column
    indices to achieve column-major lower-triangle extraction without
    explicit Fortran-order raveling.

    References
    ----------
    .. [1] Kevin Sheppard, "MFE Toolbox", utility/vech.m, Revision 3,
       Date: 2/1/2008.

    Examples
    --------
    >>> import numpy as np
    >>> S = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]])
    >>> vech(S)
    array([[1],
           [2],
           [3],
           [5],
           [6],
           [9]])
    """
    # Convert input to numpy array for flexibility with list/tuple inputs
    # Ref: vech.m uses MATLAB native matrix; Python equivalent is np.asarray
    mat = np.asarray(mat)

    # --- Input Validation ---
    # Ref: vech.m:30 — [k,l] = size(matrixData); validates 2-D
    if mat.ndim != 2:
        raise ValueError(
            'MATRIXDATA must be a 2-D square symmetric matrix, '
            f'but received array with {mat.ndim} dimension(s).'
        )

    # Ref: vech.m:31 — if k~=l; validates squareness
    k = mat.shape[0]
    if k != mat.shape[1]:
        raise ValueError(
            'MATRIXDATA must be a symmetric matrix. '
            f'Received non-square matrix with shape {mat.shape}.'
        )

    # Ref: vech.m:31 — any(any(matrixData~=matrixData')); validates symmetry
    # Use np.allclose for numerical robustness instead of exact equality,
    # since floating-point operations can introduce tiny asymmetries that
    # MATLAB's exact comparison would also reject.
    if k > 0 and not np.allclose(mat, mat.T):
        raise ValueError(
            'MATRIXDATA must be a symmetric matrix. '
            'The input matrix is not symmetric within floating-point tolerance.'
        )

    # --- Extract Lower Triangle in Column-Major Order ---
    # Ref: vech.m:37 — sel = tril(true(k))
    # Ref: vech.m:38 — stackedData = matrixData(sel)
    #
    # MATLAB's boolean indexing extracts elements in column-major order
    # (down each column first). Python/NumPy's boolean indexing extracts
    # in row-major order (across each row first). To replicate MATLAB's
    # column-major extraction:
    #
    # We use np.triu_indices(k) which returns the upper triangle indices
    # in row-major order: for K=3, rows=[0,0,0,1,1,2], cols=[0,1,2,1,2,2].
    # Swapping these (using cols as row indices and rows as col indices)
    # yields the lower triangle in column-major order:
    # rows_lower=[0,1,2,1,2,2], cols_lower=[0,0,0,1,1,2]
    # which matches MATLAB's extraction pattern.
    # Ref: vech.m:38 — column-major extraction of lower triangle
    triu_row_indices, triu_col_indices = np.triu_indices(k)
    # Swap row/col to get lower-triangle column-major ordering
    v = mat[triu_col_indices, triu_row_indices]

    # Return as column vector to match MATLAB output shape (K*(K+1)/2 x 1)
    # Ref: vech.m returns column vector per MATLAB convention
    return v.reshape(-1, 1)
