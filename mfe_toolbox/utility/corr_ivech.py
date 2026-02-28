"""
Reconstruct a symmetric correlation matrix from its off-diagonal half-vec representation.

This module provides the ``corr_ivech`` function, which is the inverse of
``corr_vech``: given a K(K-1)/2 vector of off-diagonal correlation elements,
it reassembles the full K×K symmetric correlation matrix with unit diagonal.

Migrated from: utility/corr_ivech.m
Author: Kevin Sheppard
"""

import numpy as np


def corr_ivech(stacked_data: np.ndarray) -> np.ndarray:
    """
    Transform a vector into a symmetric correlation matrix.

    Reconstructs a K×K symmetric correlation matrix from its K(K-1)/2
    off-diagonal elements stored in lower-triangular column-major order
    (matching MATLAB's ``~triu(true(K))`` boolean indexing convention).

    Parameters
    ----------
    stacked_data : numpy.ndarray
        K(K-1)/2 vector of off-diagonal correlation elements.  Row vectors
        and higher-dimensional arrays that can be ravelled to a 1-D vector
        are accepted.

    Returns
    -------
    matrix_data : numpy.ndarray
        K × K symmetric correlation matrix with ones on the diagonal.

    Raises
    ------
    ValueError
        If ``stacked_data`` is a 2-D matrix with both dimensions > 1,
        or if its length is not conformable to K(K-1)/2 for some
        integer K ≥ 2.

    Notes
    -----
    The data is stacked according to the pattern (ref: corr_ivech.m:14-19)::

        [  1         data[0]    data[1]      ...  data[K-2]
           data[0]   1          data[K-1]    ...  ...
           data[1]   data[K-1]  1            ...  ...
           ...       ...        ...          ...  data[K(K-1)/2-1]
           data[K-2] ...        ...          ...  1                ]

    See Also
    --------
    corr_vech : Extract off-diagonal half-vec from a correlation matrix.
    ivech : General inverse half-vectorization (includes diagonal).

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.corr_ivech import corr_ivech
    >>> v = np.array([0.5, 0.3, 0.4])
    >>> corr_ivech(v)
    array([[1. , 0.5, 0.3],
           [0.5, 1. , 0.4],
           [0.3, 0.4, 1. ]])
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: corr_ivech.m:30-36 — auto-transpose row vector to column;
    # reject matrices with both dimensions > 1.
    # ------------------------------------------------------------------
    stacked_data = np.atleast_1d(stacked_data)

    # Ref: corr_ivech.m:34-36 — MATLAB checks size(stackedData,2)~=1 after
    # a possible transpose.  In Python we reject genuine 2-D matrices.
    if stacked_data.ndim >= 2 and min(stacked_data.shape) > 1:
        raise ValueError("STACKED_DATA must be a vector, not a matrix.")

    # Flatten to 1-D (handles row vectors, column vectors, and scalars)
    # Ref: corr_ivech.m:30-31 — row→column transpose
    stacked_data = stacked_data.ravel()

    # ------------------------------------------------------------------
    # Compute K from the vector length K2 = K*(K-1)/2
    # Solving: K^2 - K - 2*K2 = 0  =>  K = (1 + sqrt(1 + 8*K2)) / 2
    # The +1 offset vs regular ivech accounts for the missing diagonal
    # elements in the *correlation* half-vec.
    # Ref: corr_ivech.m:38-39
    # ------------------------------------------------------------------
    K2 = len(stacked_data)
    K_float = (-1.0 + np.sqrt(1.0 + 8.0 * K2)) / 2.0 + 1.0
    K = int(np.floor(K_float))

    # Ref: corr_ivech.m:41-44 — validate K is an exact integer
    if K != K_float:
        raise ValueError(
            "The number of elements in STACKED_DATA must be conformable to "
            "the inverse vech operation for a correlation matrix "
            "(i.e. K*(K-1)/2 for some integer K >= 2)."
        )

    # ------------------------------------------------------------------
    # Matrix reconstruction
    # Ref: corr_ivech.m:48-51
    # ------------------------------------------------------------------
    matrix_data = np.zeros((K, K))

    # CRITICAL COLUMN-MAJOR ORDERING FIX (corr_ivech.m:49-50):
    # MATLAB: loc = ~triu(true(K)); matrixData(loc) = stackedData;
    # MATLAB fills the lower triangle in *column-major* order:
    #   (1,0),(2,0),(3,0),...,(2,1),(3,1),...,(K-1,K-2)   [0-based]
    #
    # np.triu_indices(K, k=1) returns upper-triangle (row, col) pairs
    # in row-major order:
    #   (0,1),(0,2),(0,3),...,(1,2),(1,3),...,(K-2,K-1)
    # Swapping row↔col maps these to lower-triangle positions that
    # traverse columns first, exactly matching MATLAB's column-major
    # boolean-indexing order.
    rows, cols = np.triu_indices(K, k=1)
    matrix_data[cols, rows] = stacked_data

    # Ref: corr_ivech.m:51 — symmetrize and place unit diagonal
    matrix_data = matrix_data + matrix_data.T + np.eye(K)

    return matrix_data
