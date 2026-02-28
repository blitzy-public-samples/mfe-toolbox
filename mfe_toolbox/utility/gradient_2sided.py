"""
Two-sided finite difference gradient computation.

Migrated from utility/gradient_2sided.m (77 lines) — Originally from COMPECON
toolbox [www4.ncsu.edu/~pfackler], documentation by James P. LeSage
(University of Toledo), further modified to perform 2-sided numerical
derivatives by Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk).

This module provides the :func:`gradient_2sided` function which computes a
central-difference gradient and, optionally, a T × M individual score matrix
used for robust covariance estimation.

Notes
-----
The step-size formula ``eps^(1/3) * max(|x|, 1e-2)`` is the classical
choice for central differences (see Dennis & Schnabel, *Numerical Methods
for Unconstrained Optimization*, 1996).

Revision: 3    Date: 2/1/2006 (original MATLAB)
"""

import numpy as np


def gradient_2sided(f, x, *args, compute_scores=False):
    """Compute the 2-sided finite difference gradient of a scalar function.

    Parameters
    ----------
    f : callable
        Objective function.  When *compute_scores* is ``False`` the call
        signature is ``fval = f(x, *args)`` and *fval* must be a scalar.
        When *compute_scores* is ``True`` the call signature is
        ``(fval, scores) = f(x, *args)`` where *scores* is a T-element
        array of individual log-likelihood contributions (or equivalent).
    x : array_like
        M × 1 parameter vector.
    *args : tuple
        Additional positional arguments forwarded to *f*.
    compute_scores : bool, optional
        If ``True``, also compute the T × M matrix of individual scores
        (numerical partial derivatives of each observation's contribution).
        Default ``False``.

    Returns
    -------
    G : numpy.ndarray
        M-element gradient vector (central-difference approximation).
    Gt : numpy.ndarray, optional
        T × M matrix of individual scores.  Only returned when
        *compute_scores* is ``True``.

    Raises
    ------
    TypeError
        If *f* is not callable.
    ValueError
        If *x* cannot be converted to a finite 1-D float array.

    Examples
    --------
    >>> import numpy as np
    >>> def quad(x): return float(x @ x)
    >>> G = gradient_2sided(quad, np.array([1.0, 2.0, 3.0]))
    >>> np.allclose(G, [2.0, 4.0, 6.0], atol=1e-6)
    True
    """

    # ------------------------------------------------------------------
    # Phase 1: Input handling
    # Ref: gradient_2sided.m:35-37 — MATLAB transposes row to column;
    #      Python equivalent: force 1-D float64 via atleast_1d + ravel.
    # ------------------------------------------------------------------
    if not callable(f):
        raise TypeError("f must be a callable function.")

    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()

    # Ref: gradient_2sided.m:39 — M = size(x,1)
    M = len(x)
    if M == 0:
        raise ValueError("Parameter vector x must have at least one element.")

    # ------------------------------------------------------------------
    # Phase 2: Step size computation
    # Ref: gradient_2sided.m:41 — h = eps.^(1/3) * max(abs(x), 1e-2)
    # The classical choice for central differences: O(h^2) truncation
    # balanced against O(eps/h) rounding, optimal at h ~ eps^(1/3).
    # ------------------------------------------------------------------
    eps_third = np.finfo(float).eps ** (1.0 / 3.0)
    h = eps_third * np.maximum(np.abs(x), 1e-2)

    # Ref: gradient_2sided.m:42-43 — xh = x + h; h = xh - x
    # This trick guarantees that the perturbation is exactly
    # representable in floating point (avoids catastrophic cancellation).
    xh = x + h
    h = xh - x  # re-derive h from the rounded sum

    # ------------------------------------------------------------------
    # Phase 3: Perturbation matrix
    # Ref: gradient_2sided.m:45 — ee = diag(h)
    # Each column ee[:, i] is a unit perturbation in direction i scaled
    # by the step size h[i].
    # ------------------------------------------------------------------
    ee = np.diag(h)

    # ------------------------------------------------------------------
    # Phase 4 & 5: Forward and backward evaluations
    # ------------------------------------------------------------------
    # Ref: gradient_2sided.m:46-47 — gf = zeros(M,1); gb = zeros(M,1)
    gf = np.zeros(M)
    gb = np.zeros(M)

    if compute_scores:
        # Ref: gradient_2sided.m:49-53 — Probe first perturbation to
        # discover T (number of individual likelihood contributions).
        # MATLAB: [temp1, temp2] = feval(f, x + ee(:,1), varargin{:})
        # Python uses 0-indexed column ee[:, 0].
        temp1, temp2 = f(x + ee[:, 0], *args)
        temp2 = np.atleast_1d(np.asarray(temp2, dtype=float)).ravel()
        T = len(temp2)

        # Ref: gradient_2sided.m:52-53 — Gf = zeros(T, M); Gb = zeros(T, M)
        Gf = np.zeros((T, M))
        Gb = np.zeros((T, M))

        # Store the already-computed first forward evaluation
        gf[0] = float(temp1)
        Gf[:, 0] = temp2

        # ---- Forward step loop (remaining parameters) ----
        # Ref: gradient_2sided.m:56-62 — for i=1:M (1-indexed)
        # Python 0-indexed; i=0 already handled above.
        for i in range(1, M):
            # Ref: gradient_2sided.m:60 — [gf(i), Gf(:,i)] = feval(...)
            val, scores_i = f(x + ee[:, i], *args)
            gf[i] = float(val)
            Gf[:, i] = np.atleast_1d(np.asarray(scores_i, dtype=float)).ravel()

        # ---- Backward step loop ----
        # Ref: gradient_2sided.m:64-71 — for i=1:M (backward)
        for i in range(M):
            # Ref: gradient_2sided.m:69 — [gb(i), Gb(:,i)] = feval(...)
            val, scores_i = f(x - ee[:, i], *args)
            gb[i] = float(val)
            Gb[:, i] = np.atleast_1d(np.asarray(scores_i, dtype=float)).ravel()
    else:
        # ---- Forward step loop (scalar-only) ----
        # Ref: gradient_2sided.m:56-62 — for i=1:M, gf(i) = feval(...)
        for i in range(M):
            # Ref: gradient_2sided.m:58 — gf(i) = feval(f, x+ee(:,i), ...)
            gf[i] = float(f(x + ee[:, i], *args))

        # ---- Backward step loop (scalar-only) ----
        # Ref: gradient_2sided.m:64-71 — for i=1:M, gb(i) = feval(...)
        for i in range(M):
            # Ref: gradient_2sided.m:67 — gb(i) = feval(f, x-ee(:,i), ...)
            gb[i] = float(f(x - ee[:, i], *args))

    # ------------------------------------------------------------------
    # Phase 6: Gradient computation (central difference)
    # Ref: gradient_2sided.m:73 — G = (gf - gb) ./ (2*h)
    # ------------------------------------------------------------------
    G = (gf - gb) / (2.0 * h)

    # ------------------------------------------------------------------
    # Phase 7: Return values
    # ------------------------------------------------------------------
    if compute_scores:
        # Ref: gradient_2sided.m:74-76 — Gt = (Gf-Gb) ./ repmat(2*h', T, 1)
        # NumPy broadcasting replaces MATLAB's repmat(2*h', T, 1):
        # h[np.newaxis, :] has shape (1, M), so division broadcasts over
        # the (T, M) numerator automatically.
        Gt = (Gf - Gb) / (2.0 * h[np.newaxis, :])
        return G, Gt
    else:
        return G
