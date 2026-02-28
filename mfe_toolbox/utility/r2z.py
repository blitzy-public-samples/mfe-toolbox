"""
Correlation matrix to unconstrained vector via Fisher z-transform.

Transforms a K x K correlation matrix R into a K(K-1)/2 unconstrained vector z
using Cholesky factorization to extract partial correlations, then applying
the Fisher z-transform (log-odds) to map from (-1, 1) to (-inf, inf).

This is the forward transformation used for unconstrained optimization of
correlation matrices. The inverse transformation is provided by z2r.

Author: Kevin Sheppard
Reference: utility/r2z.m (Version 4.0, MFE Toolbox)
See also: z2r, r2phi, phi2r
"""

import numpy as np


def r2z(R: np.ndarray) -> np.ndarray:
    """
    Transform correlation matrix to unconstrained vector via Fisher z-transform.

    Uses Cholesky factorization to extract partial correlations from a K x K
    correlation matrix, then applies the Fisher z-transform (log-odds) to
    map each partial correlation from (-1, 1) to (-inf, inf), producing a
    K(K-1)/2 unconstrained vector suitable for unconstrained optimization.

    Parameters
    ----------
    R : numpy.ndarray
        K x K symmetric positive-definite correlation matrix with ones on
        the diagonal.

    Returns
    -------
    z : numpy.ndarray
        K(K-1)/2 vector of unconstrained values in (-inf, inf).

    Raises
    ------
    ValueError
        If R is not a 2-dimensional square array.
    numpy.linalg.LinAlgError
        If R is not positive definite (Cholesky decomposition fails).

    Notes
    -----
    The transformation proceeds in three steps:

    1. Cholesky decomposition: R = L @ L.T where L is lower triangular.
    2. Extract partial correlations from the Cholesky factor by normalizing
       each row element by the remaining variance.
    3. Apply Fisher z-transform: z = log((x + 1) / (1 - x)) to map
       partial correlations from (-1, 1) to (-inf, inf).

    The extraction of the strictly upper triangular elements from the
    transposed partial correlation matrix uses column-major (Fortran) ordering
    to match MATLAB's element extraction convention.

    The inverse transformation is implemented in ``z2r``.

    References
    ----------
    .. [1] Kevin Sheppard, MFE Toolbox utility/r2z.m, Revision 1, 3/27/2012.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> z = r2z(R)
    >>> z.shape
    (1,)
    """
    # Convert input and validate
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("R must be a 2-dimensional array.")
    if R.shape[0] != R.shape[1]:
        raise ValueError(
            f"R must be a square matrix, got shape {R.shape}."
        )

    # Ref: r2z.m:23 — get matrix dimension
    k = R.shape[0]

    # Handle trivial case: 1x1 correlation matrix has no off-diagonal elements
    if k == 1:
        return np.array([], dtype=np.float64)

    # Ref: r2z.m:24 — allocate partial correlation matrix
    C = np.zeros((k, k), dtype=np.float64)

    # Ref: r2z.m:25 — Cholesky decomposition
    # MATLAB chol(R)' transposes upper to get lower triangular.
    # NumPy np.linalg.cholesky returns lower triangular directly.
    C2 = np.linalg.cholesky(R)

    # Ref: r2z.m:27-33 — Extract partial correlations from Cholesky factor
    # MATLAB loop: for i=2:k, rem=1; for j=i-1:-1:1 (1-based indexing)
    # Python equivalent: for i in range(1, k); for j in range(i-1, -1, -1)
    for i in range(1, k):  # Ref: r2z.m:27 — MATLAB i=2:k maps to Python i=1:k-1
        rem = 1.0  # Ref: r2z.m:28 — remaining variance initialized to 1
        for j in range(i - 1, -1, -1):  # Ref: r2z.m:29 — MATLAB j=i-1:-1:1 maps to Python j=i-1 downto 0
            # Ref: r2z.m:30 — normalize Cholesky element by remaining std dev
            C[i, j] = C2[i, j] / np.sqrt(rem)
            # Ref: r2z.m:31 — update remaining variance using original Cholesky element
            rem = rem - C2[i, j] ** 2

    # Ref: r2z.m:35 — transpose: moves lower-triangular values to upper triangle
    C = C.T

    # Ref: r2z.m:36 — extract strictly upper triangular elements
    # CRITICAL: MATLAB C(~tril(true(k))) extracts in column-major (Fortran) order.
    # Must use .ravel('F') to match MATLAB's column-major element extraction.
    mask = ~np.tril(np.ones((k, k), dtype=bool))
    z = C.ravel('F')[mask.ravel('F')]

    # Ref: r2z.m:37 — Fisher z-transform (log-odds) maps (-1, 1) to (-inf, inf)
    z = np.log((z + 1.0) / (1.0 - z))

    return z
