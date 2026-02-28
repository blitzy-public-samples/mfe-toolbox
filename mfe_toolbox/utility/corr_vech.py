"""
Correlation matrix half-vectorization (off-diagonal).

Extracts the strictly lower triangular elements of a symmetric correlation
matrix into a K(K-1)/2 vector, matching MATLAB column-major extraction order.

Migrated from: utility/corr_vech.m
Author: Kevin Sheppard
Original revision: 3, Date: 2/1/2008

See Also
--------
mfe_toolbox.utility.vech : Half-vectorization including diagonal.
mfe_toolbox.utility.corr_ivech : Inverse operation (vector to correlation matrix).
"""

import numpy as np


def corr_vech(matrix_data: np.ndarray) -> np.ndarray:
    """
    Transform a correlation matrix into its off-diagonal half-vec representation.

    Extracts the K(K-1)/2 off-diagonal elements from the strictly lower
    triangular portion of a K x K symmetric correlation matrix, stacked in
    column-major (Fortran) order to match the MATLAB convention.

    Parameters
    ----------
    matrix_data : numpy.ndarray
        K x K symmetric correlation matrix.  The matrix must be square and
        symmetric (validated via ``numpy.allclose``).

    Returns
    -------
    stacked_data : numpy.ndarray
        1-D array of length K(K-1)/2 containing the off-diagonal lower
        triangular elements in column-major order.

    Raises
    ------
    ValueError
        If ``matrix_data`` is not 2-dimensional.
    ValueError
        If ``matrix_data`` is not square.
    ValueError
        If ``matrix_data`` is not symmetric (within floating-point tolerance).

    Notes
    -----
    The stacking convention follows the MATLAB original where the output
    vector maps back to the correlation matrix as::

        [ 1         data[0]    data[1]     ...  data[K-2]
          data[0]   1          data[K-1]   ...  ...
          data[1]   data[K-1]  1           ...  ...
          ...       ...        ...         ...  data[K(K-1)/2 - 1]
          data[K-2] ...        ...         ...  1                  ]

    MATLAB extracts elements in column-major order (down each column first).
    To replicate this in NumPy, both the matrix and the boolean selector are
    ravelled with ``order='F'`` before applying the mask.

    Examples
    --------
    >>> import numpy as np
    >>> C = np.array([[1.0, 0.5, 0.3],
    ...               [0.5, 1.0, 0.7],
    ...               [0.3, 0.7, 1.0]])
    >>> corr_vech(C)
    array([0.5, 0.3, 0.7])
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: corr_vech.m:30 — [k,l] = size(matrixData);
    # ------------------------------------------------------------------
    if matrix_data.ndim != 2:
        raise ValueError(
            "MATRIX_DATA must be a 2-dimensional array, "
            f"got ndim={matrix_data.ndim}."
        )

    k, l = matrix_data.shape  # Ref: corr_vech.m:30

    # Ref: corr_vech.m:31 — if k~=l || any(any(matrixData~=matrixData'))
    if k != l:
        raise ValueError(
            f"MATRIX_DATA must be a square matrix, got shape ({k}, {l})."
        )

    # Symmetry check using np.allclose to accommodate floating-point rounding.
    # Ref: corr_vech.m:31 — matrixData~=matrixData' (exact in MATLAB, but
    # allclose is the appropriate Python analogue for IEEE 754 floats).
    if not np.allclose(matrix_data, matrix_data.T):
        raise ValueError("MATRIX_DATA must be a symmetric matrix.")

    # ------------------------------------------------------------------
    # Strictly lower triangular element extraction
    # Ref: corr_vech.m:37 — sel = ~triu(true(k));
    # ------------------------------------------------------------------
    # Build a boolean mask for the strictly lower triangle (excludes diagonal).
    # np.triu(np.ones(..., dtype=bool)) mirrors MATLAB triu(true(k)).
    sel = ~np.triu(np.ones((k, k), dtype=bool))  # Ref: corr_vech.m:37

    # Ref: corr_vech.m:38 — stackedData = matrixData(sel);
    # CRITICAL: MATLAB logical indexing extracts elements in column-major
    # (Fortran) order, while NumPy boolean indexing uses row-major (C) order.
    # To reproduce MATLAB behaviour, ravel both the data and the selector in
    # Fortran order before applying the mask.
    stacked_data = matrix_data.ravel('F')[sel.ravel('F')]

    return stacked_data
