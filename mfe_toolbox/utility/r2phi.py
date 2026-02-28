"""
Correlation matrix to angle vector transformation.

Migrated from utility/r2phi.m (32 lines) — Author: Kevin Sheppard.

Transforms a K x K correlation matrix to a K(K-1)/2 vector of angles in [0, pi]
via Cholesky decomposition and cumulative sine/cosine inversion.

See Also
--------
phi2r : Inverse transformation from angles to correlation matrix.
z2r : Unconstrained vector to correlation matrix via Fisher z-transform.
r2z : Correlation matrix to unconstrained vector via Fisher z-transform.
"""

import numpy as np


def r2phi(R: np.ndarray) -> np.ndarray:
    """
    Transform a correlation matrix to an angle vector.

    Extracts the Cholesky factor of the correlation matrix and recovers angles
    in [0, pi] via inverse cosine and cumulative sine products. The resulting
    angle vector parameterizes the correlation matrix and has K(K-1)/2 elements.

    Parameters
    ----------
    R : numpy.ndarray
        K x K symmetric positive-definite correlation matrix with unit diagonal.

    Returns
    -------
    phi : numpy.ndarray
        K(K-1)/2 vector of angles in [0, 2*pi].

    Raises
    ------
    ValueError
        If R is not a 2-D square matrix, not symmetric, or not positive definite.

    Notes
    -----
    The transformation is based on the Cholesky decomposition of R.
    Each off-diagonal element of the upper-triangular Cholesky factor is
    expressed as a product of cosines and sines of the angle parameters.
    The angles are extracted by inverting these relationships via arccos.

    MATLAB equivalent: ``phi = r2phi(R)`` from utility/r2phi.m.

    It is necessary to invert both cos and sin to identify where in [0, 2*pi]
    the angle is (see comment in r2phi.m:19 — FIXME note in original source).

    References
    ----------
    See phi2r for the inverse transformation from angles to correlation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> phi = r2phi(R)
    """
    # --- Input validation ---
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("R must be a 2-D array.")
    if R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix.")
    if not np.allclose(R, R.T, atol=1e-10):
        raise ValueError("R must be a symmetric matrix.")

    # Ref: r2phi.m:20 — MATLAB chol(R) returns upper triangular factor;
    # numpy.linalg.cholesky returns lower triangular, so transpose to match.
    X = np.linalg.cholesky(R).T  # Upper triangular Cholesky factor

    # Ref: r2phi.m:21 — k = length(R) → k = len(R)
    k = len(R)

    # Ref: r2phi.m:22 — S = zeros(k); initialize sine accumulation matrix
    S = np.zeros((k, k))

    # Ref: r2phi.m:24 — P = zeros(k); initialize angle matrix
    P = np.zeros((k, k))

    # Ref: r2phi.m:25 — S(1,:) = 1; first row of S is all ones (base case
    # for cumulative product: product of zero sine terms is 1)
    S[0, :] = 1.0

    # Ref: r2phi.m:26 — cumS = S; cumulative product of S rows
    cumS = S.copy()

    # Ref: r2phi.m:27-31 — Angle extraction loop.
    # MATLAB: for i=1:k-1 (1-indexed) → Python: for i in range(k-1) (0-indexed)
    for i in range(k - 1):
        # Ref: r2phi.m:28 — P(i,i+1:k) = acos(X(i,i+1:k)./cumS(i,i+1:k))
        # MATLAB 1-indexed i uses rows 1..k-1 and columns i+1..k.
        # Python 0-indexed i uses rows 0..k-2 and columns i+1..k-1.
        # Recover angles via arccos of the ratio of Cholesky element to
        # the cumulative product of previous sines.
        P[i, i + 1:k] = np.arccos(X[i, i + 1:k] / cumS[i, i + 1:k])

        # Ref: r2phi.m:29 — S(i+1,i+1:k) = sin(P(i,i+1:k))
        # Compute sine of the recovered angle for this row.
        S[i + 1, i + 1:k] = np.sin(P[i, i + 1:k])

        # Ref: r2phi.m:30 — cumS(i+1,i+1:k) = prod(S(1:i+1,i+1:k))
        # Update cumulative product of sines for the next iteration.
        # MATLAB: prod(S(1:i+1, ...)) — product down rows 1 through i+1 (1-indexed)
        # Python: np.prod(S[:i+2, ...], axis=0) — product of rows 0 through i+1 (0-indexed)
        cumS[i + 1, i + 1:k] = np.prod(S[:i + 2, i + 1:k], axis=0)

    # Ref: r2phi.m:32 — phi = P(P>0)
    # CRITICAL: MATLAB extracts elements from P in column-major (Fortran) order
    # when using logical indexing P(P>0). Python's P[P>0] extracts in row-major
    # (C) order. To match MATLAB's column-major extraction, we ravel P in
    # Fortran order and then apply the boolean mask on the raveled array.
    P_flat = P.ravel('F')  # Column-major flattening to match MATLAB P(P>0)
    phi = P_flat[P_flat > 0]

    return phi
