"""
Vector to lower-triangular Cholesky matrix reconstruction.

Reconstructs a K x K lower triangular matrix from its K(K+1)/2 half-vec
representation. This is the inverse operation of chol2vec.

The stacking convention follows MATLAB column-major ordering:
    [ data[0]    0         0         ...  0
      data[1]    data[K]   0         ...  0
      data[2]    data[K+1] data[2K-1] ... 0
      ...        ...       ...       ...  ...
      data[K-1]  data[2K-2] ...      ...  data[K(K+1)/2-1] ]

Notes
-----
Migrated from utility/vec2chol.m (MATLAB MFE Toolbox v4.0).
Author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 2/1/2008
"""

import numpy as np


def vec2chol(stacked_data: np.ndarray) -> np.ndarray:
    """
    Transform half-vec representation to lower triangular matrix.

    Reconstructs a K x K lower triangular matrix from a K(K+1)/2 vector
    of stacked elements, using the same column-major fill order as MATLAB's
    ``matrixData(tril(true(K))) = stackedData`` operation.

    Parameters
    ----------
    stacked_data : numpy.ndarray
        K(K+1)/2 vector of stacked data.  Row vectors and column vectors
        are both accepted (the input is automatically raveled).

    Returns
    -------
    matrix_data : numpy.ndarray
        K x K lower triangular matrix with the stacked data placed in the
        lower triangle (including diagonal) in column-major order.

    Raises
    ------
    ValueError
        If *stacked_data* is not a vector (i.e., has more than one non-
        singleton dimension after squeezing).
    ValueError
        If the length of *stacked_data* is not a valid K(K+1)/2 for some
        positive integer K.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> vec2chol(v)
    array([[1., 0., 0.],
           [2., 4., 0.],
           [3., 5., 6.]])

    See Also
    --------
    chol2vec : Inverse operation — extract half-vec from a lower triangular
        matrix.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    # Ref: vec2chol.m:30-31 — auto-transpose row vector to column vector
    stacked_data = np.atleast_1d(stacked_data).ravel()

    # Ref: vec2chol.m:33-36 — ensure input is effectively 1-D
    # After ravel() the array is always 1-D, but we guard against
    # non-numeric or empty inputs as a defensive measure.
    if stacked_data.ndim != 1:
        raise ValueError("STACKED_DATA must be a column vector.")

    # Ref: vec2chol.m:38-39 — derive matrix dimension K from vector length
    # K2 = K*(K+1)/2  =>  K = (-1 + sqrt(1 + 8*K2)) / 2
    k2 = len(stacked_data)
    k_float = (-1.0 + np.sqrt(1.0 + 8.0 * k2)) / 2.0

    # Ref: vec2chol.m:41-43 — verify K is a positive integer
    k = int(round(k_float))
    if abs(k_float - k) > 1e-10 or k < 1:
        raise ValueError(
            "The number of elements in STACKED_DATA must be conformable to "
            "the inverse chol2vec operation (length must equal K*(K+1)/2 for "
            "some positive integer K)."
        )

    # ------------------------------------------------------------------
    # Matrix reconstruction
    # ------------------------------------------------------------------
    # Ref: vec2chol.m:50 — initialise output
    # Ref: vec2chol.m:52-54 — MATLAB uses:
    #   pl = tril(true(K));
    #   matrixData(pl) = stackedData;
    #
    # MATLAB logical indexing fills in *column-major* order, so elements
    # are placed column-by-column down the lower triangle:
    #   (0,0), (1,0), (2,0), ..., (1,1), (2,1), ..., (K-1,K-1)
    #
    # Python boolean indexing fills in *row-major* order, so a direct
    # np.tril mask would produce a different element placement.
    #
    # Correct approach: fill an upper-triangular mask in row-major order
    # (which visits the same relative positions that column-major visits
    # in the lower triangle), then transpose to obtain the lower-triangular
    # result with identical element placement.
    temp = np.zeros((k, k))
    # Ref: vec2chol.m:53 — tril(true(K)) equivalent via upper-tri + transpose
    pu = np.triu(np.ones((k, k), dtype=bool))
    temp[pu] = stacked_data
    matrix_data = temp.T

    return matrix_data
