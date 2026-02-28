"""
Inverse Fisher z-transform from unconstrained vector to correlation matrix.

Transforms a K(K-1)/2 unconstrained vector into a valid K x K correlation matrix
using the inverse Fisher z-transform followed by a cumulative Cholesky-like
factor construction. This is the inverse operation of :func:`r2z`.

Migrated from: utility/z2r.m
Author: Kevin Sheppard
Revision: 1, Date: 3/27/2012

Notes
-----
The unconstrained values are mapped to the correlation matrix through:

    y = (exp(Z) - 1) / (1 + exp(Z))        # maps (-inf, inf) -> (-1, 1)
    C(i,j) = y(i,j) * sqrt(1 - sum(C(i, j+1:k)^2))   # for i=2,...,k, j=i-1:-1:1
    R = C @ C'                               # correlation matrix

See Also
--------
r2z : Correlation matrix to unconstrained vector (forward transform).
r2phi : Correlation matrix to partial correlations.
phi2r : Partial correlations to correlation matrix.
"""

import numpy as np


def z2r(z: np.ndarray) -> np.ndarray:
    """
    Transform unconstrained vector to correlation matrix.

    Converts a K(K-1)/2 vector of unconstrained real values into a valid
    K x K positive semi-definite correlation matrix via inverse Fisher
    z-transform and Cholesky-like factor construction.

    Parameters
    ----------
    z : numpy.ndarray
        K(K-1)/2 vector of unconstrained values in (-inf, inf). The length
        must satisfy ``k * (k - 1) // 2 == len(z)`` for some integer k >= 2.

    Returns
    -------
    R : numpy.ndarray
        K x K symmetric positive semi-definite correlation matrix with ones
        on the diagonal.

    Raises
    ------
    ValueError
        If the length of `z` is not a valid K(K-1)/2 triangular number.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.z2r import z2r
    >>> z = np.array([0.0, 0.5, -0.3])
    >>> R = z2r(z)
    >>> R.shape
    (3, 3)
    >>> np.allclose(np.diag(R), 1.0)
    True
    """
    # Ensure input is a numpy array with float type for numerical operations
    z = np.asarray(z, dtype=np.float64).ravel()

    # --- Input validation --- Ref: z2r.m:25-29
    m = len(z)  # Ref: z2r.m:25 — m = length(z)
    k = int(np.ceil(np.sqrt(2.0 * m)))  # Ref: z2r.m:26 — k = ceil(sqrt(2*m))

    # Ref: z2r.m:27-28 — Validate that z has exactly k*(k-1)/2 elements
    if k * (k - 1) // 2 != m:
        raise ValueError(
            f"Incorrect number of elements in z. Got {m} elements, which does "
            f"not correspond to k*(k-1)/2 for any integer k."
        )

    # --- Inverse Fisher z-transform --- Ref: z2r.m:31
    # Maps (-inf, inf) -> (-1, 1):  y = (exp(z) - 1) / (1 + exp(z))
    # This is algebraically equivalent to 2*sigmoid(z) - 1 or tanh(z/2),
    # but we follow the MATLAB formula exactly for numerical parity.
    exp_z = np.exp(z)
    z_transformed = (exp_z - 1.0) / (1.0 + exp_z)

    # --- Initialize Cholesky-like factor matrix --- Ref: z2r.m:30
    C = np.zeros((k, k))

    # --- Fill lower triangle with transformed values --- Ref: z2r.m:32-37
    # MATLAB count starts at 1 (1-indexed); Python count starts at 0 (0-indexed)
    count = 0  # Ref: z2r.m:32 — count = 1 (MATLAB) → count = 0 (Python)
    for i in range(1, k):  # Ref: z2r.m:33 — for i=2:k → for i in range(1, k)
        for j in range(0, i):  # Ref: z2r.m:34 — for j=1:i-1 → for j in range(0, i)
            C[i, j] = z_transformed[count]  # Ref: z2r.m:35 — C(i,j) = z(count)
            count += 1  # Ref: z2r.m:36 — count = count + 1

    # --- Construct Cholesky-like factor --- Ref: z2r.m:39-47
    C[0, 0] = 1.0  # Ref: z2r.m:39 — C(1,1) = 1 (MATLAB 1-indexed → Python 0-indexed)

    for i in range(1, k):  # Ref: z2r.m:40 — for i=2:k → for i in range(1, k)
        rem = 1.0  # Ref: z2r.m:41 — rem = 1

        # Ref: z2r.m:42 — for j=i-1:-1:1 → for j in range(i-1, -1, -1)
        # MATLAB j goes from i-1 down to 1 (1-indexed), Python j goes from i-1 down to 0 (0-indexed)
        for j in range(i - 1, -1, -1):
            C[i, j] = C[i, j] * np.sqrt(rem)  # Ref: z2r.m:43 — C(i,j) = C(i,j)*sqrt(rem)
            rem = rem - C[i, j] ** 2  # Ref: z2r.m:44 — rem = rem - C(i,j)^2

        C[i, i] = np.sqrt(rem)  # Ref: z2r.m:46 — C(i,i) = sqrt(rem) — diagonal element

    # --- Compute correlation matrix --- Ref: z2r.m:49-51
    R = C @ C.T  # Ref: z2r.m:49 — R = C*C'

    # Normalize to ensure unit diagonal (handles floating-point accumulation)
    r = np.sqrt(np.diag(R))  # Ref: z2r.m:50 — r = sqrt(diag(R))
    R = R / np.outer(r, r)  # Ref: z2r.m:51 — R = R ./ (r*r')

    return R
