"""
ARMAX residual (error) computation with Numba JIT acceleration.

Migrated from:
  - timeseries/armaxerrors.m  (MATLAB wrapper, Kevin Sheppard, Revision 3, 10/19/2009)
  - mex_source/armaxerrors.c  (C MEX kernel, Kevin Sheppard, Revision 3, 10/16/2009)

This module computes errors from an ARMAX model for use in a least-squares
optimizer.  The inner recursion loop is JIT-compiled via Numba in nopython
mode, replacing the original C MEX acceleration kernel
(``mex_source/armaxerrors.c``) with equivalent performance characteristics
(10–100× speedup over pure Python loops).

The function preserves the exact numerical behaviour of the C kernel,
including zero burn-in for the first *m* elements, column-major to
row-major exogenous variable layout adjustment, and post-loop sigma
normalisation.

Public API
----------
armaxerrors(parameters, p, q, constant, y, x, m, sigma) -> numpy.ndarray
    Compute ARMAX residuals (errors).

_armaxerrors_core(parameters, p, q, constant, y, x, m, sigma, T, np_val, nq, k) -> numpy.ndarray
    Numba JIT inner loop (exposed for direct testing / downstream JIT chains).
"""

import numpy as np
import numba


# ---------------------------------------------------------------------------
# Numba JIT inner function — direct translation of armaxerror_core() from
# mex_source/armaxerrors.c lines 18-70.
# ---------------------------------------------------------------------------
@numba.jit(nopython=True, cache=True)
def _armaxerrors_core(
    parameters: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    constant: int,
    y: np.ndarray,
    x: np.ndarray,
    m: int,
    sigma: np.ndarray,
    T: int,
    np_val: int,
    nq: int,
    k: int,
) -> np.ndarray:
    """
    Inner loop for ARMAX residual computation.

    Translated from ``mex_source/armaxerrors.c:armaxerror_core`` (lines 18-70).
    All array parameters must be contiguous float64 arrays.  Scalar parameters
    must be plain Python/Numba-compatible ints.

    Parameters
    ----------
    parameters : ndarray, shape (n_params,)
        ARMAX parameter vector laid out as
        ``[constant?, AR_1 … AR_np, X_1 … X_k, MA_1 … MA_nq]``.
    p : ndarray, shape (np_val,)
        AR lag indices (stored as float64; cast to int for indexing).
    q : ndarray, shape (nq,)
        MA lag indices (stored as float64; cast to int for indexing).
    constant : int
        1 if the model includes a constant term, 0 otherwise.
    y : ndarray, shape (T,)
        Observed data, augmented with ``max(max(Q)-max(P), 0)`` leading zeros.
    x : ndarray, shape (T, max(k, 1))
        Exogenous regressors in **row-major** layout.  When ``k == 0`` the
        exogenous loop is skipped; the array is still passed for Numba type
        stability.
    m : int
        Number of burn-in observations (first *m* errors are set to zero).
    sigma : ndarray, shape (T,)
        Conditional standard deviations used for GLS-style normalisation.
    T : int
        Total number of observations (``len(y)``).
    np_val : int
        Number of AR terms (``len(p)``).
    nq : int
        Number of MA terms (``len(q)``).
    k : int
        Number of exogenous regressors (columns of *x* actually used).

    Returns
    -------
    ndarray, shape (T,)
        Error vector.  The first *m* elements are zero; subsequent elements
        contain the ARMAX residuals divided by *sigma*.
    """
    # Allocate output error vector
    e = np.zeros(T)

    # ------------------------------------------------------------------
    # Ref: armaxerrors.c:40-43 — zero burn-in for first m elements.
    # The np.zeros call above already initialises to zero, but we keep
    # the explicit loop to mirror the C source exactly and guarantee
    # clarity of intent.
    # ------------------------------------------------------------------
    for t in range(m):
        e[t] = 0.0

    # ------------------------------------------------------------------
    # Ref: armaxerrors.c:44-65 — main recursion.
    # MATLAB 1-based loop ``for t = m+1 : T`` maps to C/Python 0-based
    # ``for t in range(m, T)``.
    # ------------------------------------------------------------------
    for t in range(m, T):
        # Ref: armaxerrors.c:46 — initialise with observed value
        e[t] = y[t]

        # Ref: armaxerrors.c:47-50 — constant subtraction
        if constant:
            e[t] -= parameters[0]

        # Ref: armaxerrors.c:51-54 — AR term loop, 0-based C indexing.
        # p[i] is a float64 (MATLAB stores lag indices as doubles);
        # cast to int for safe array indexing.
        for i in range(np_val):
            e[t] -= parameters[constant + i] * y[t - int(p[i])]

        # Ref: armaxerrors.c:55-59 — exogenous terms.
        # C MEX uses column-major offset ``x[t + T*i]`` because MATLAB
        # stores matrices column-major.  Python/NumPy uses row-major
        # layout, so the equivalent access is ``x[t, i]``.
        for i in range(k):
            e[t] -= parameters[constant + np_val + i] * x[t, i]

        # Ref: armaxerrors.c:60-64 — MA term recursion with lagged errors.
        # q[i] is a float64 lag index; cast to int for indexing.
        for i in range(nq):
            e[t] -= parameters[constant + np_val + k + i] * e[t - int(q[i])]

        # Ref: armaxerrors.c:64 — ``e[t] = e[t];`` is a no-op in the C
        # source (and line 51 of armaxerrors.m).  Omitted in Python.

    # ------------------------------------------------------------------
    # Ref: armaxerrors.c:66-69 — post-loop sigma normalisation.
    # The C code divides only elements from index m to T-1 by sigma.
    # (The MATLAB .m file divides the entire vector, but for valid
    # inputs the first m elements are zero so the result is identical.)
    # ------------------------------------------------------------------
    for t in range(m, T):
        e[t] = e[t] / sigma[t]

    return e


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------
def armaxerrors(
    parameters,
    p,
    q,
    constant,
    y,
    x,
    m,
    sigma,
):
    """
    Compute errors from an ARMAX model for use in a least-squares optimizer.

    This function is the Python equivalent of ``timeseries/armaxerrors.m``,
    accelerated via Numba JIT compilation (replacing
    ``mex_source/armaxerrors.c``).

    The ARMAX model is::

        y(t) = const + Σ_i φ_i · y(t - p_i) + Σ_j β_j · x(t, j)
                     + Σ_l θ_l · e(t - q_l) + e(t)

    so that the error (residual) at each time step is computed by
    rearranging the equation above.

    Parameters
    ----------
    parameters : array_like
        1-D parameter vector laid out as
        ``[constant?, AR_1 … AR_np, X_1 … X_k, MA_1 … MA_nq]``
        where ``constant?`` is present only when *constant* == 1.
    p : array_like
        Lag indices of the AR component (may be empty).  MATLAB convention
        stores these as doubles; they are cast to ``int`` inside the JIT
        kernel for array indexing.
    q : array_like
        Lag indices of the MA component (may be empty).
    constant : int
        ``1`` to include a constant term, ``0`` to exclude.
    y : array_like
        Observed data augmented with ``max(max(Q) - max(P), 0)`` leading
        zeros.  Must be 1-D.
    x : array_like
        Exogenous regressors augmented with the same number of leading-zero
        rows as *y*.  If there are no exogenous variables, pass a 1-D array
        (e.g. ``np.zeros(T)``); the exogenous loop will be skipped.
    m : int
        Index of the first element used in the recursive residual
        calculation.  The first *m* errors are set to zero (burn-in).
    sigma : array_like
        Conditional standard deviations with the same length as *y*, used
        for GLS-style error normalisation.

    Returns
    -------
    numpy.ndarray
        1-D error vector with the same length as *y*.  The first *m*
        elements are zero; subsequent elements are the sigma-normalised
        ARMAX residuals.

    Notes
    -----
    * The inner recursion is compiled once by Numba on the first call and
      cached for subsequent invocations (``@numba.jit(nopython=True,
      cache=True)``).
    * Ref: timeseries/armaxerrors.m — MATLAB wrapper.
    * Ref: mex_source/armaxerrors.c — C MEX kernel replaced by Numba JIT.

    See Also
    --------
    mfe_toolbox.timeseries.armaxfilter_likelihood : Uses ``armaxerrors``
        to evaluate the Gaussian log-likelihood.
    mfe_toolbox.timeseries.arma_forecaster : Uses ``armaxerrors`` to
        obtain the innovation history for forecasting.
    """
    # ------------------------------------------------------------------
    # Input coercion — ensure all arrays are contiguous float64 for Numba
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()

    # Derive dimension scalars
    T = y.shape[0]
    np_val = p.shape[0]
    nq = q.shape[0]

    # ------------------------------------------------------------------
    # Exogenous regressor handling:
    #   • 2-D input  → k = number of columns (actual regressors)
    #   • 1-D input  → k = 0 (no exogenous), reshape to (T, 1) dummy
    #                   for Numba type stability
    #
    # Ref: armaxerrors.c:102 — ``k = mxGetN(prhs[5])`` gets column count.
    # ------------------------------------------------------------------
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        # No exogenous regressors — create a dummy 2-D array.
        # k stays 0 so the exogenous loop inside _armaxerrors_core is
        # never entered.
        k = 0
        x = x.reshape(-1, 1)
    elif x.ndim == 2:
        k = x.shape[1]
    else:
        raise ValueError(
            "x must be a 1-D or 2-D array. "
            f"Received array with ndim={x.ndim}."
        )

    # Guarantee contiguous memory layout for Numba
    x = np.ascontiguousarray(x, dtype=np.float64)

    # Coerce scalar arguments to Python int for Numba nopython mode
    constant_int = int(constant)
    m_int = int(m)

    # ------------------------------------------------------------------
    # Delegate to JIT-compiled inner loop
    # ------------------------------------------------------------------
    return _armaxerrors_core(
        parameters, p, q, constant_int, y, x, m_int, sigma, T, np_val, nq, k
    )
