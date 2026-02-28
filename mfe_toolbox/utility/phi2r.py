"""
Transform a vector of angles to a correlation matrix.

Migrated from utility/phi2r.m — Author: Kevin Sheppard
Part of the MFE Toolbox (Version 4.0) MATLAB-to-Python migration.

This module provides the phi2r function which constructs a valid correlation
matrix from a vector of K(K-1)/2 angles in [0, pi] (or unconstrained values
when using the logistic transform). The algorithm builds the correlation matrix
via cumulative sine/cosine products, ensuring the result is symmetric with
unit diagonal and positive semi-definite.

See Also
--------
mfe_toolbox.utility.r2phi : Inverse operation (correlation matrix to angles)
mfe_toolbox.utility.z2r : Fisher z-transform to correlation
mfe_toolbox.utility.r2z : Correlation to Fisher z-transform
"""

import numpy as np


def phi2r(phi, transform=False):
    """
    Transform a vector of angles to a correlation matrix.

    Constructs a K x K correlation matrix from K(K-1)/2 angles using
    cumulative sine/cosine products. Optionally applies a logistic transform
    to map unconstrained real values into the [0, 2*pi) angle space.

    Parameters
    ----------
    phi : numpy.ndarray
        K(K-1)/2 vector of angles in [0, pi]. If ``transform=True``, values
        are unconstrained real numbers that are first mapped to [0, 2*pi)
        via a logistic (sigmoid) transform.
    transform : bool, optional
        If True, apply logistic transform before constructing the correlation
        matrix: ``phi = 2 * pi * exp(phi) / (1 + exp(phi))``.
        Default is False.

    Returns
    -------
    R : numpy.ndarray
        K x K symmetric correlation matrix with unit diagonal and
        positive semi-definite structure.

    Raises
    ------
    ValueError
        If the length of ``phi`` does not equal K(K-1)/2 for any positive
        integer K.
    TypeError
        If ``phi`` is not array-like or cannot be converted to float64.

    Notes
    -----
    The algorithm constructs the correlation matrix using the following steps:

    1. Compute cosines (C) and sines (S) of the input angles.
    2. Arrange them into upper-triangular selection patterns.
    3. Multiply C element-wise by the column-wise cumulative product of S.
    4. Form R = C' * C and normalize to ensure unit diagonal.
    5. Enforce exact symmetry via averaging: R = (R + R') / 2.

    The parameterization ensures that any vector of angles in [0, pi] maps
    to a valid correlation matrix. The optional logistic transform allows
    unconstrained optimization over the angle parameters.

    **MATLAB Column-Major Fill Order (Critical Implementation Detail):**
    MATLAB's logical indexing fills matrices in column-major (Fortran) order,
    while NumPy uses row-major (C) order. For matrices of dimension K >= 4,
    these orderings differ. This implementation uses a transpose-based
    approach to replicate MATLAB's column-major fill order exactly, ensuring
    numerical parity to within ±1e-6.

    References
    ----------
    .. [1] Sheppard, K. (2009). MFE Toolbox, Version 4.0.
       utility/phi2r.m

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.phi2r import phi2r

    Construct a 3x3 correlation matrix from 3 angles:

    >>> angles = np.array([np.pi/4, np.pi/3, np.pi/6])
    >>> R = phi2r(angles)
    >>> R.shape
    (3, 3)
    >>> np.allclose(np.diag(R), 1.0)
    True
    >>> np.allclose(R, R.T)
    True

    Use logistic transform for unconstrained parameterization:

    >>> phi_unc = np.array([0.5, -0.3, 1.2])
    >>> R_unc = phi2r(phi_unc, transform=True)
    >>> np.allclose(np.diag(R_unc), 1.0)
    True
    """
    # Convert input to 1D float64 numpy array
    # Ref: phi2r.m uses MATLAB's automatic type handling
    try:
        phi = np.asarray(phi, dtype=np.float64).ravel()
    except (ValueError, TypeError) as exc:
        raise TypeError(
            "phi must be array-like and convertible to float64."
        ) from exc

    # Ref: phi2r.m:17-18 — Optional logistic transform maps unconstrained reals
    # to [0, 2*pi): phi = 2*pi*exp(phi)./(1+exp(phi))
    if transform:
        phi = 2.0 * np.pi * np.exp(phi) / (1.0 + np.exp(phi))

    # Ref: phi2r.m:21 — Number of angle parameters
    m = len(phi)

    # Handle empty input: return 0x0 correlation matrix
    # Ref: phi2r.m — MATLAB returns empty matrix for empty phi
    if m == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Ref: phi2r.m:22 — Compute matrix dimension K from K(K-1)/2 = m
    k = int(np.ceil(np.sqrt(2.0 * m)))

    # Validate that m equals exactly K*(K-1)/2
    # MATLAB does not validate explicitly, but would error on index mismatch
    expected_m = k * (k - 1) // 2
    if m != expected_m:
        raise ValueError(
            f"phi must have length K*(K-1)/2 for some positive integer K. "
            f"Got length {m}, which does not equal {k}*({k}-1)/2 = {expected_m}."
        )

    # Ref: phi2r.m:23 — Strictly upper triangular selection mask for C
    # ~tril(ones(k)) selects elements above the main diagonal
    Csel = ~np.tril(np.ones((k, k), dtype=bool))

    # Ref: phi2r.m:24-25 — Upper triangular (incl. diagonal) mask for S,
    # with entire first row excluded
    Ssel = np.triu(np.ones((k, k), dtype=bool))
    Ssel[0, :] = False  # Ref: phi2r.m:25 — first row excluded

    # Ref: phi2r.m:26 — Initialize C as identity matrix
    C = np.eye(k, dtype=np.float64)

    # Ref: phi2r.m:27-28 — Initialize S as zeros with first row all ones
    S = np.zeros((k, k), dtype=np.float64)
    S[0, :] = 1.0

    # Ref: phi2r.m:29-30 — Compute cosines and sines of angle vector
    c = np.cos(phi)
    s = np.sin(phi)

    # Ref: phi2r.m:31-32 — Fill C and S matrices using logical indexing
    #
    # CRITICAL: MATLAB logical indexing fills in column-major (Fortran) order,
    # traversing down each column before moving to the next column.
    # NumPy boolean indexing fills in row-major (C) order, traversing across
    # each row before moving to the next row.
    #
    # For K >= 4, these orderings are DIFFERENT. Example for K=4 strictly
    # upper triangular mask:
    #   MATLAB order: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3)
    #   NumPy order:  (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    #
    # Solution: Transpose matrices and masks, fill with NumPy's row-major
    # indexing (which now traverses original columns), then transpose back.
    # This exactly replicates MATLAB's column-major fill behavior.

    # Fill C with cosines in MATLAB column-major order — Ref: phi2r.m:31
    C_T = C.T.copy()
    C_T[Csel.T] = c
    C = C_T.T

    # Fill S with sines in MATLAB column-major order — Ref: phi2r.m:32
    S_T = S.T.copy()
    S_T[Ssel.T] = s
    S = S_T.T

    # Ref: phi2r.m:34 — Element-wise multiply C by column-wise cumulative product of S
    # MATLAB's cumprod(S) computes cumulative product along first dimension (columns),
    # which corresponds to axis=0 in NumPy
    C = C * np.cumprod(S, axis=0)

    # Ref: phi2r.m:35 — Form raw correlation-like matrix: R = C' * C
    R = C.T @ C

    # Ref: phi2r.m:36-37 — Normalize to ensure unit diagonal
    # r = sqrt(diag(R)); R = R ./ (r * r')
    r = np.sqrt(np.diag(R))
    R = R / np.outer(r, r)

    # Ref: phi2r.m:38 — Enforce exact symmetry by averaging with transpose
    R = (R + R.T) / 2.0

    return R
