"""
Two-sided finite difference Hessian matrix computation.

Computes the numerical Hessian of a scalar-valued function using two-sided
(central) finite differences. This provides more accurate second derivative
estimates than one-sided methods by canceling odd-order error terms.

Originally from the COMPECON toolbox (www4.ncsu.edu/~pfackler).
Documentation modified by James P. LeSage, Dept of Economics,
University of Toledo.

Further modified (to do 2-sided numerical derivatives, rather than 1-sided) by:
Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Migrated from MATLAB to Python as part of the MFE Toolbox migration.
Source: utility/hessian_2sided.m (82 lines)
"""

import numpy as np


def hessian_2sided(f, x, *args):
    """
    Compute the two-sided finite difference Hessian matrix.

    Evaluates the numerical Hessian of a scalar-valued function ``f`` at
    parameter vector ``x`` using central finite differences with step sizes
    proportional to ``eps**(1/3)``.

    Parameters
    ----------
    f : callable
        Scalar-valued objective function with signature ``f(x, *args)``.
        Must accept a 1-D numpy array as first argument and return a scalar.
    x : array_like
        Parameter vector of length N at which to evaluate the Hessian.
    *args : tuple
        Additional positional arguments passed through to ``f``.

    Returns
    -------
    H : numpy.ndarray
        Symmetric N x N Hessian matrix (second partial derivatives).

    Raises
    ------
    ValueError
        If ``x`` is empty or if ``f`` cannot be evaluated at ``x``.

    Notes
    -----
    The central difference Hessian formula used is:

    .. math::

        H_{ij} = \\frac{f(x+e_i+e_j) - f(x+e_i) - f(x+e_j) + 2f(x)
                       - f(x-e_i) - f(x-e_j) + f(x-e_i-e_j)}{2 h_i h_j}

    where :math:`e_i` is a vector with :math:`h_i` in position *i* and zeros
    elsewhere, and :math:`h_i = \\varepsilon^{1/3} \\max(|x_i|, 10^{-8})`.

    The step size exponent 1/3 is optimal for central differences in the
    presence of machine-precision rounding error (compared to 1/2 for
    one-sided differences).

    References
    ----------
    .. [1] Miranda, M.J. and Fackler, P.L. (2002). Applied Computational
       Economics and Finance. MIT Press (COMPECON toolbox).

    Examples
    --------
    >>> import numpy as np
    >>> def quadratic(x):
    ...     A = np.array([[2.0, 1.0], [1.0, 3.0]])
    ...     return 0.5 * x @ A @ x
    >>> H = hessian_2sided(quadratic, np.array([1.0, 2.0]))
    >>> np.allclose(H, np.array([[2.0, 1.0], [1.0, 3.0]]), atol=1e-6)
    True
    """
    # Ref: hessian_2sided.m:32 — size(x,1) gets parameter count
    # Convert to 1-D float array; ravel() handles both column and row vectors
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = len(x)

    # Validate non-empty parameter vector
    if n == 0:
        raise ValueError("X must be a non-empty parameter vector.")

    # Ref: hessian_2sided.m:34-36 — MATLAB validates x is a column vector;
    # Python equivalent: ravel() above ensures 1-D, so shape is always valid

    # Ref: hessian_2sided.m:38-43 — Validate that f can be evaluated at x
    try:
        f(x, *args)
    except Exception as e:
        # Ref: hessian_2sided.m:41 — preserve error message format from MATLAB
        raise ValueError(
            "There was an error evaluating the function. "
            "Please check the arguments. "
            f"The specific error was: {e}"
        ) from e

    # Ref: hessian_2sided.m:46 — Base function evaluation at x
    fx = f(x, *args)

    # ---- Step Size Computation ----
    # Ref: hessian_2sided.m:49 — eps^(1/3) * max(|x|, 1e-8)
    # CRITICAL: minimum step base is 1e-8 (NOT 1e-2 as in gradient_2sided.m)
    # The exponent 1/3 is optimal for central (2-sided) finite differences
    h = np.finfo(float).eps ** (1.0 / 3.0) * np.maximum(np.abs(x), 1e-8)

    # Ref: hessian_2sided.m:50-51 — Numerical precision trick:
    # Adding then subtracting ensures h is exactly representable in floating point
    xh = x + h
    h = xh - x

    # Ref: hessian_2sided.m:52 — Create diagonal perturbation matrix
    # MATLAB: sparse(1:n,1:n,h,n,n); Python: np.diag is sufficient for
    # typical parameter vector sizes (no need for scipy.sparse)
    ee = np.diag(h)

    # ---- Single Perturbation Forward/Backward Evaluations ----
    # Ref: hessian_2sided.m:55-56 — Initialize gradient-like vectors
    gp = np.zeros(n)
    gm = np.zeros(n)

    # Ref: hessian_2sided.m:57-60 — MATLAB 1-indexed loop; Python 0-indexed
    for i in range(n):
        gp[i] = f(x + ee[:, i], *args)  # Ref: hessian_2sided.m:58
        gm[i] = f(x - ee[:, i], *args)  # Ref: hessian_2sided.m:59

    # ---- Double Perturbation Evaluations ----
    # Ref: hessian_2sided.m:62 — MATLAB: h*h' (outer product of step sizes)
    hh = np.outer(h, h)

    # Ref: hessian_2sided.m:63-64 — MATLAB: NaN*ones(n) initializes NaN matrices
    Hp = np.full((n, n), np.nan)
    Hm = np.full((n, n), np.nan)

    # Ref: hessian_2sided.m:66-72 — Double loop for upper triangle (0-indexed)
    # Exploits symmetry: only computes (i, j) for j >= i, then mirrors
    for i in range(n):
        for j in range(i, n):
            Hp[i, j] = f(x + ee[:, i] + ee[:, j], *args)  # Ref: hessian_2sided.m:68
            Hp[j, i] = Hp[i, j]                             # Ref: hessian_2sided.m:69
            Hm[i, j] = f(x - ee[:, i] - ee[:, j], *args)  # Ref: hessian_2sided.m:70
            Hm[j, i] = Hm[i, j]                             # Ref: hessian_2sided.m:71

    # ---- Hessian Assembly ----
    # Ref: hessian_2sided.m:75 — Initialize output Hessian matrix
    H = np.zeros((n, n))

    # Ref: hessian_2sided.m:76-81 — Central difference Hessian formula
    # Formula: H(i,j) = (Hp(i,j) - gp(i) - gp(j) + 2*fx - gm(i) - gm(j) + Hm(i,j)) / (2 * hh(i,j))
    # MATLAB: /hh(i,j)/2 is (value / hh(i,j)) / 2 = value / (2 * hh(i,j))
    for i in range(n):
        for j in range(i, n):
            # Ref: hessian_2sided.m:78 — central difference Hessian formula
            H[i, j] = (
                Hp[i, j] - gp[i] - gp[j] + fx + fx - gm[i] - gm[j] + Hm[i, j]
            ) / (hh[i, j] * 2.0)
            # Ref: hessian_2sided.m:79 — enforce symmetry
            H[j, i] = H[i, j]

    return H
