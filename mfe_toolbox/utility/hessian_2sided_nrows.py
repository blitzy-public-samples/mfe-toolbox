"""
Compute the last K rows of a two-sided finite difference Hessian matrix.

This module provides a partial Hessian computation that only evaluates the last
K rows of the full N×N Hessian, avoiding unnecessary function evaluations when
only a subset of second derivatives is needed (e.g., in block-coordinate
optimization).

Migrated from utility/hessian_2sided_nrows.m (89 lines).
Original code from COMPECON toolbox [www4.ncsu.edu/~pfackler], modified by
James P. LeSage and Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk).

See Also
--------
hessian_2sided : Full N×N two-sided finite difference Hessian.
"""

import numpy as np


def hessian_2sided_nrows(f, x, k, *args):
    """
    Compute the last K rows of a two-sided finite difference Hessian.

    Evaluates only the double perturbations needed for the last K rows of the
    Hessian matrix, reducing the total number of function evaluations from
    O(N^2) to O(K*N) compared to computing the full Hessian.

    Parameters
    ----------
    f : callable
        Objective function with signature ``scalar = f(x, *args)``.
    x : numpy.ndarray
        N-element parameter vector.
    k : int
        Number of rows to compute (last k rows of the Hessian).
    *args : tuple
        Additional positional arguments passed through to *f*.

    Returns
    -------
    H : numpy.ndarray
        K × N partial Hessian matrix (the last k rows of the full Hessian).

    Raises
    ------
    ValueError
        If *x* is not a 1-D vector, if *k* is not a positive integer, or
        if *k* exceeds the length of *x*.
    RuntimeError
        If *f* cannot be evaluated at the initial parameter vector.

    Notes
    -----
    The central-difference formula for element (i, j) of the Hessian is:

    .. math::

        H_{ij} = \\frac{f(x+e_i+e_j) - f(x+e_i) - f(x+e_j) + 2f(x)
                       - f(x-e_i) - f(x-e_j) + f(x-e_i-e_j)}{2 h_i h_j}

    The step size uses ``eps^(1/3) * max(|x_i|, 1e-2)`` (Ref:
    hessian_2sided_nrows.m:52), which matches the gradient step-size
    convention rather than the full-Hessian convention of 1e-8.

    References
    ----------
    .. [1] Miranda, M.J. and Fackler, P.L., COMPECON toolbox.
    .. [2] Sheppard, K., MFE Toolbox Version 4.0, 2009.
    """
    # ------------------------------------------------------------------
    # Input handling — Ref: hessian_2sided_nrows.m:33-39
    # ------------------------------------------------------------------
    # Ensure x is a flat 1-D numpy array of floats.
    x = np.atleast_1d(x).ravel().astype(float, copy=False)
    n = len(x)

    # Validate k
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("K must be a positive integer.")
    if k > n:
        raise ValueError(
            f"K ({k}) must not exceed the number of parameters N ({n})."
        )

    # ------------------------------------------------------------------
    # Validate that f evaluates correctly — Ref: hessian_2sided_nrows.m:41-46
    # ------------------------------------------------------------------
    try:
        f(x, *args)
    except Exception as exc:
        raise RuntimeError(
            "There was an error evaluating the function.  Please check the "
            f"arguments.  The specific error was: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Base function evaluation — Ref: hessian_2sided_nrows.m:49
    # ------------------------------------------------------------------
    fx = f(x, *args)

    # ------------------------------------------------------------------
    # Step-size computation — Ref: hessian_2sided_nrows.m:52-54
    # Note: minimum clamp is 1e-2 (same as gradient_2sided), NOT 1e-8
    # ------------------------------------------------------------------
    h = np.finfo(float).eps ** (1.0 / 3.0) * np.maximum(np.abs(x), 1e-2)
    # Ref: hessian_2sided_nrows.m:53-54 — floating-point precision trick
    xh = x + h
    h = xh - x

    # ------------------------------------------------------------------
    # Diagonal perturbation matrix — Ref: hessian_2sided_nrows.m:56
    # MATLAB: ee = sparse(1:n,1:n,h,n,n); Python: dense np.diag is fine.
    # ------------------------------------------------------------------
    ee = np.diag(h)

    # ------------------------------------------------------------------
    # Single perturbations (forward / backward) for ALL n parameters
    # Ref: hessian_2sided_nrows.m:59-64
    # ------------------------------------------------------------------
    gp = np.zeros(n)
    gm = np.zeros(n)
    for i in range(n):
        # Ref: hessian_2sided_nrows.m:62 — MATLAB 1-indexed; Python 0-indexed
        gp[i] = f(x + ee[:, i], *args)
        gm[i] = f(x - ee[:, i], *args)

    # ------------------------------------------------------------------
    # Outer product of step sizes — Ref: hessian_2sided_nrows.m:66
    # MATLAB: hh = h*h' (outer product); Python: np.outer
    # ------------------------------------------------------------------
    hh = np.outer(h, h)

    # ------------------------------------------------------------------
    # NaN-initialized matrices for double perturbations
    # Ref: hessian_2sided_nrows.m:67-68
    # ------------------------------------------------------------------
    Hp = np.full((n, n), np.nan)
    Hm = np.full((n, n), np.nan)

    # ------------------------------------------------------------------
    # Double perturbations — ONLY for last k rows (efficiency)
    # Ref: hessian_2sided_nrows.m:70-77
    # MATLAB: for i=n-k+1:n (1-indexed) → Python: for i in range(n-k, n)
    # Inner loop covers ALL columns (j=1:n → j in range(n))
    # ------------------------------------------------------------------
    for i in range(n - k, n):
        for j in range(n):
            # Ref: hessian_2sided_nrows.m:72 — double forward perturbation
            Hp[i, j] = f(x + ee[:, i] + ee[:, j], *args)
            # Ref: hessian_2sided_nrows.m:73 — symmetry copy
            Hp[j, i] = Hp[i, j]
            # Ref: hessian_2sided_nrows.m:74 — double backward perturbation
            Hm[i, j] = f(x - ee[:, i] - ee[:, j], *args)
            # Ref: hessian_2sided_nrows.m:75 — symmetry copy
            Hm[j, i] = Hm[i, j]

    # ------------------------------------------------------------------
    # Hessian assembly — only for last k rows
    # Ref: hessian_2sided_nrows.m:79-85
    # Central-difference formula:
    #   H(i,j) = (Hp(i,j) - gp(i) - gp(j) + 2*fx - gm(i) - gm(j) + Hm(i,j))
    #            / (2 * hh(i,j))
    # ------------------------------------------------------------------
    H = np.zeros((n, n))
    for i in range(n - k, n):
        for j in range(n):
            # Ref: hessian_2sided_nrows.m:82
            H[i, j] = (
                Hp[i, j] - gp[i] - gp[j] + fx + fx
                - gm[i] - gm[j] + Hm[i, j]
            ) / (hh[i, j] * 2.0)
            # Ref: hessian_2sided_nrows.m:83 — symmetry copy into full matrix
            H[j, i] = H[i, j]

    # ------------------------------------------------------------------
    # Extract only the last k rows — Ref: hessian_2sided_nrows.m:88
    # MATLAB: H = H((n-k+1):n, :) (1-indexed) → Python: H[n-k:n, :]
    # ------------------------------------------------------------------
    H = H[n - k: n, :]

    return H
