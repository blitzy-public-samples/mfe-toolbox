"""
Simplified TARCH/GJR-GARCH conditional variance recursion with Numba JIT acceleration.

This module provides a straightforward pure-Python alternative to :func:`tarch_core`
for computing conditional variances in TARCH(P,O,Q) processes.  The simplified loop
structure uses explicit index-bounds checking at every lag access rather than
pre-initialising the first *m* elements, making the recursion easier to audit at the
cost of marginally higher per-iteration overhead.

The module serves as both a reference implementation and a fallback when the
optimised ``tarch_core`` (translated from C MEX ``mex_source/tarch_core.c``) is
unavailable or when clarity is preferred over speed.

Migrated from ``univariate/tarch_core_simple.m`` (Kevin Sheppard, MFE Toolbox v4.0,
Revision 4, 4/12/2012).

Notes
-----
* ``@numba.jit(nopython=True, cache=True)`` is mandatory per the migration rules –
  no *object-mode* fallback is permitted.
* Numerical parity target: ±1e-6 against MATLAB reference outputs.
"""

import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def tarch_core_simple(
    fdata: np.ndarray,
    fIdata: np.ndarray,
    parameters: np.ndarray,
    back_cast: float,
    p: int,
    o: int,
    q: int,
    m: int,
    T: int,
    tarch_type: int,
) -> np.ndarray:
    """Compute conditional variances for a TARCH(P,O,Q) process using a
    simplified recursion.

    This is a simpler alternative to :func:`tarch_core` that iterates over
    every time-step from ``t = 0`` and checks array-bounds for each lag
    individually, falling back to *back_cast* when the index is negative.
    The optimised version instead pre-fills the first *m* elements with
    *back_cast* and starts the recursion from index *m*.

    The recursion computes::

        g(h(t)) = omega
                  + sum_{j=1}^{p} alpha_j * f(r_{t-j})
                  + sum_{j=1}^{o} gamma_j * I_{t-j<0} * f(r_{t-j})
                  + sum_{j=1}^{q} beta_j  * g(h(t-j))

    where

    * ``f(x) = |x|``,  ``g(x) = sqrt(x)`` when *tarch_type* = 1  (AVGARCH)
    * ``f(x) = x²``,   ``g(x) = x``       when *tarch_type* = 2  (standard TARCH/GJR)

    Parameters
    ----------
    fdata : np.ndarray
        ``(T,)`` array of transformed data.  For *tarch_type* = 1 this is
        ``|data|``; for *tarch_type* = 2 this is ``data ** 2``.
    fIdata : np.ndarray
        ``(T,)`` array of asymmetric indicator-weighted transformed data,
        i.e. ``fdata * (data < 0)``.
    parameters : np.ndarray
        ``(1 + p + o + q,)`` parameter vector laid out as
        ``[omega, alpha_1 … alpha_p, gamma_1 … gamma_o, beta_1 … beta_q]``.
    back_cast : float
        Back-cast value used for any lag that references a pre-sample period.
        Typically the unconditional variance (or its square-root when
        *tarch_type* = 1).
    p : int
        Positive integer — number of symmetric innovation lags (ARCH order).
    o : int
        Non-negative integer — number of asymmetric innovation lags
        (leverage/threshold order).
    q : int
        Non-negative integer — number of conditional-variance lags
        (GARCH order).
    m : int
        ``max(p, o, q)`` — maximum lag order.  Included for interface
        compatibility with :func:`tarch_core`; the simplified recursion does
        not use it directly because it performs per-access bounds checks.
    T : int
        Number of observations (length of *fdata* and *fIdata*).
    tarch_type : int
        Variance model type:

        * 1 — AVGARCH: model evolves in absolute values; the output array
          is element-wise squared at the end to convert from
          ``sqrt(h(t))`` to ``h(t)``.
        * 2 — Standard TARCH / GJR-GARCH: model evolves directly in
          variances.

    Returns
    -------
    ht : np.ndarray
        ``(T,)`` array of conditional variances ``h(t)``.

    See Also
    --------
    tarch_core : Optimised TARCH recursion (translates C MEX
        ``mex_source/tarch_core.c``).

    Notes
    -----
    Ref: ``tarch_core_simple.m`` — Kevin Sheppard, University of Oxford.

    MATLAB uses 1-based indexing; this Python version uses 0-based arrays.

    * ``tarch_core_simple.m:53`` — MATLAB ``for i=1:T`` maps to
      ``for t in range(T)``.
    * ``tarch_core_simple.m:56`` — MATLAB condition ``(i-j)>0`` maps to
      ``(t - 1 - j) >= 0`` in 0-based Python.
    * ``tarch_core_simple.m:79-80`` — Final squaring for *tarch_type* = 1.
    """
    # Ref: tarch_core_simple.m:44 — Allocate output array (MATLAB: ht = zeros(T,1))
    ht = np.zeros(T)

    # ------------------------------------------------------------------
    # Main recursion loop
    # Ref: tarch_core_simple.m:53-76
    # ------------------------------------------------------------------
    for t in range(T):
        # Ref: tarch_core_simple.m:54 — Constant (omega)
        # MATLAB: ht(i) = parameters(1)  [1-based]
        # Python: parameters[0]           [0-based]
        ht[t] = parameters[0]

        # ---- ARCH (symmetric innovation) terms ----
        # Ref: tarch_core_simple.m:55-61
        # MATLAB: for j=1:p → parameters(j+1)*fdata(i-j)
        # Python: for j=0..p-1 → parameters[1+j]*fdata[t-1-j]
        for j in range(p):
            if (t - 1 - j) >= 0:
                ht[t] = ht[t] + parameters[1 + j] * fdata[t - 1 - j]
            else:
                ht[t] = ht[t] + parameters[1 + j] * back_cast

        # ---- Asymmetric / leverage terms ----
        # Ref: tarch_core_simple.m:62-68
        # MATLAB: for j=1:o → parameters(j+p+1)*fIdata(i-j)
        # Python: for j=0..o-1 → parameters[1+p+j]*fIdata[t-1-j]
        for j in range(o):
            if (t - 1 - j) >= 0:
                ht[t] = ht[t] + parameters[1 + p + j] * fIdata[t - 1 - j]
            else:
                ht[t] = ht[t] + parameters[1 + p + j] * back_cast

        # ---- GARCH (conditional-variance lag) terms ----
        # Ref: tarch_core_simple.m:69-75
        # MATLAB: for j=1:q → parameters(j+p+o+1)*ht(i-j)
        # Python: for j=0..q-1 → parameters[1+p+o+j]*ht[t-1-j]
        for j in range(q):
            if (t - 1 - j) >= 0:
                ht[t] = ht[t] + parameters[1 + p + o + j] * ht[t - 1 - j]
            else:
                ht[t] = ht[t] + parameters[1 + p + o + j] * back_cast

    # ------------------------------------------------------------------
    # Post-processing: convert from absolute-value space to variance space
    # Ref: tarch_core_simple.m:79-80 — "if tarch_type==1, ht=ht.^2; end"
    #
    # When tarch_type == 1 the recursion evolves in g(h) = sqrt(h), i.e.
    # in absolute-value units.  Squaring converts back to variances:
    #   h(t) = [sqrt(h(t))]^2
    #
    # np.abs() ensures numerical robustness: if a recursion value drifted
    # slightly negative due to floating-point arithmetic the square still
    # yields a valid non-negative variance (|x|^2 == x^2 for all real x,
    # but abs communicates intent).
    # ------------------------------------------------------------------
    if tarch_type == 1:
        for t in range(T):
            abs_val = np.abs(ht[t])
            ht[t] = abs_val * abs_val

    return ht
