"""
Transform angles to an orthogonal matrix via Givens rotation products.

Migrated from utility/phi2u.m (Version 4.0, Kevin Sheppard).
This module provides the phi2u function which constructs a K x K orthogonal
(rotation) matrix U from a K(K-1)/2 vector of angles by accumulating the
product of K(K-1)/2 individual Givens rotation matrices.

Each Givens rotation operates in the (i, j) plane, rotating by the angle
phi[count] using cos and sin values placed at positions (i,i), (i,j),
(j,i), and (j,j) of an identity matrix.
"""

import numpy as np


def phi2u(phi):
    """
    Transform angles to an orthogonal matrix via Givens rotations.

    Constructs an orthogonal matrix U as the product of K(K-1)/2 Givens
    rotation matrices, one for each pair (i, j) with i < j. The input
    angle vector phi must have length exactly K(K-1)/2 for some integer K.

    Parameters
    ----------
    phi : numpy.ndarray
        K(K-1)/2 vector of angles (in radians). The length must be
        a valid triangular number of the form K(K-1)/2 for some
        positive integer K >= 2, or zero-length for K = 0 or K = 1.

    Returns
    -------
    U : numpy.ndarray
        K x K orthogonal matrix satisfying U @ U.T ≈ I (product of
        Givens rotations). When phi has length 0, returns a 1x1 identity.

    Raises
    ------
    ValueError
        If phi is not a 1-D array or its length is not a valid K(K-1)/2
        triangular number.

    Notes
    -----
    The orthogonal matrix U is built by iterating over all unique pairs
    (i, j) with 0 <= i < j < K. For each pair, a Givens rotation matrix
    R is constructed as an identity with the four modified entries::

        R[i, i] = cos(phi[count])
        R[i, j] = -sin(phi[count])
        R[j, i] = sin(phi[count])
        R[j, j] = cos(phi[count])

    and U is updated as U = U @ R.

    References
    ----------
    Migrated from phi2u.m in the MFE Toolbox (Kevin Sheppard, University
    of Oxford).

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.phi2u import phi2u
    >>> phi = np.array([0.5, 1.0, 1.5])
    >>> U = phi2u(phi)
    >>> np.allclose(U @ U.T, np.eye(3), atol=1e-12)
    True
    """
    # Convert input to 1-D numpy array for consistent handling
    phi = np.asarray(phi, dtype=np.float64).ravel()

    # Ref: phi2u.m:3 — m = length(phi); MATLAB length returns max dimension
    m = len(phi)

    # Handle edge case: empty angle vector yields a 1x1 identity
    if m == 0:
        return np.eye(1)

    # Ref: phi2u.m:4 — k = ceil(sqrt(2*m)); recover matrix dimension K from
    # the triangular number K(K-1)/2 = m
    k = int(np.ceil(np.sqrt(2.0 * m)))

    # Validate that m is a valid triangular number K(K-1)/2
    # Ref: phi2u.m does not validate, but Python should raise on mismatch
    expected_m = k * (k - 1) // 2
    if expected_m != m:
        raise ValueError(
            f"Length of phi ({m}) is not a valid K(K-1)/2 triangular number. "
            f"Nearest K={k} would require {expected_m} elements."
        )

    # Ref: phi2u.m:5 — U = eye(k); initialize orthogonal matrix as identity
    U = np.eye(k)

    # Ref: phi2u.m:6-7 — c = cos(phi); s = sin(phi);
    # Pre-compute cosine and sine values for all angles
    c = np.cos(phi)
    s = np.sin(phi)

    # Ref: phi2u.m:8 — count = 1; MATLAB 1-indexed counter → Python 0-indexed
    count = 0

    # Ref: phi2u.m:9-17 — Double loop constructing Givens rotations
    # MATLAB: for i=1:k, for j=i+1:k → Python: for i in range(k), for j in range(i+1, k)
    for i in range(k):
        for j in range(i + 1, k):
            # Ref: phi2u.m:11 — R = eye(k); fresh identity for each rotation
            R = np.eye(k)

            # Ref: phi2u.m:12-15 — Set the four Givens rotation entries
            # R(i,i) = c(count); R(i,j) = -s(count);
            # R(j,i) = s(count); R(j,j) = c(count);
            R[i, i] = c[count]
            R[i, j] = -s[count]
            R[j, i] = s[count]
            R[j, j] = c[count]

            # Ref: phi2u.m:16 — U = U*R; accumulate rotation product
            U = U @ R

            # Ref: phi2u.m:17 — count = count + 1; advance to next angle
            count += 1

    return U
