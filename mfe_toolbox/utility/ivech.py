"""
Inverse half-vectorization (ivech) operator.

Reconstructs a symmetric K×K matrix from a K(K+1)/2 element vector,
using column-major lower-triangle fill order to match MATLAB's behavior.

This is the inverse operation of :func:`mfe_toolbox.utility.vech.vech`.

Source Reference
----------------
Migrated from ``utility/ivech.m`` (57 lines).
Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3, Date: 2/1/2008
"""

import numpy as np


def ivech(v):
    """
    Inverse half-vectorization: reconstruct a symmetric matrix from a vector.

    Given a vector ``v`` of length K(K+1)/2, reconstruct the unique K×K
    symmetric matrix whose lower-triangular elements (read column-by-column,
    i.e. MATLAB / Fortran order) are the elements of ``v``.

    Parameters
    ----------
    v : array_like
        A K(K+1)/2 element vector (1-D, column, or row).  Row vectors and
        2-D column vectors are automatically flattened.

    Returns
    -------
    mat : numpy.ndarray
        K × K symmetric matrix such that ``mat[i, j] == mat[j, i]`` for
        all valid *i*, *j*.

    Raises
    ------
    ValueError
        If the length of *v* is not a valid triangular number K(K+1)/2 for
        some positive integer K, or if the input is not convertible to a
        1-D numeric array.

    See Also
    --------
    mfe_toolbox.utility.vech.vech : Forward half-vectorization.

    Notes
    -----
    **Column-major ordering (MATLAB parity):**

    MATLAB's ``matrixData(tril(true(K))) = stackedData`` fills the lower
    triangle in column-major (Fortran) order.  NumPy's default boolean /
    fancy indexing fills in row-major (C) order.  To guarantee numerical
    parity (±1e-6) with the MATLAB reference, this implementation
    explicitly constructs column-major-sorted index arrays via
    :func:`numpy.lexsort`.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.ivech import ivech
    >>> v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> ivech(v)
    array([[1., 2., 3.],
           [2., 4., 5.],
           [3., 5., 6.]])
    """
    # ------------------------------------------------------------------
    # Phase 1 — Input Validation  (Ref: ivech.m lines 29-43)
    # ------------------------------------------------------------------

    # Ref: ivech.m:29-31 — auto-transpose row vector to column vector.
    # In Python we flatten any shape to 1-D for uniform handling.
    v = np.atleast_1d(v).ravel()  # ensures 1-D numpy array

    # Ref: ivech.m:33-35 — MATLAB checks size(stackedData,2) ~= 1.
    # After ravel() ndim is guaranteed to be 1, but guard defensively.
    if v.ndim != 1:
        raise ValueError("Input must be a 1-D vector or convertible to one.")

    # Ref: ivech.m:37 — vector length
    K2 = len(v)

    # Ref: ivech.m:38 — recover matrix dimension K from K(K+1)/2 = K2
    # Using the quadratic formula: K = (-1 + sqrt(1 + 8*K2)) / 2
    K = int((-1 + np.sqrt(1 + 8 * K2)) / 2)

    # Ref: ivech.m:40-43 — validate that K is an exact integer solution,
    # i.e. the vector length is a valid triangular number.
    if K * (K + 1) // 2 != K2:
        raise ValueError(
            "The number of elements in the input vector must be conformable "
            "to the inverse vech operation (i.e. K*(K+1)/2 for integer K). "
            f"Received length {K2}."
        )

    # ------------------------------------------------------------------
    # Phase 2 — Lower-Triangle Fill in Column-Major Order
    #            (Ref: ivech.m lines 49-53)
    # ------------------------------------------------------------------

    # Ref: ivech.m:49 — initialise K×K output with zeros
    mat = np.zeros((K, K))

    # Ref: ivech.m:52-53 —
    #   pl = tril(true(K));
    #   matrixData(pl) = stackedData;
    #
    # CRITICAL: MATLAB fills the lower triangle in COLUMN-major order
    # (column 0 top-to-bottom, then column 1, …).  NumPy's tril_indices
    # returns indices in ROW-major order (row 0 left-to-right, then row 1, …).
    # We must re-sort to column-major order using lexsort so that element
    # placement matches MATLAB exactly.
    #
    # Example for K=3, v=[1,2,3,4,5,6]:
    #   Column-major (MATLAB): (0,0)=1  (1,0)=2  (2,0)=3  (1,1)=4  (2,1)=5  (2,2)=6
    #   Row-major    (NumPy) : (0,0)=1  (1,0)=2  (1,1)=3  (2,0)=4  (2,1)=5  (2,2)=6
    #   These yield DIFFERENT symmetric matrices after symmetrisation.
    rows, cols = np.tril_indices(K)
    col_major_order = np.lexsort((rows, cols))  # sort by cols first, then rows
    mat[rows[col_major_order], cols[col_major_order]] = v

    # ------------------------------------------------------------------
    # Phase 3 — Symmetrise  (Ref: ivech.m lines 54-55)
    # ------------------------------------------------------------------

    # Ref: ivech.m:54 — extract diagonal as a diagonal matrix
    diag_mat = np.diag(np.diag(mat))

    # Ref: ivech.m:55 — copy lower triangle to upper triangle,
    # subtracting the double-counted diagonal.
    mat = mat + mat.T - diag_mat

    return mat
