"""
SARIMAX residual (error) computation.

Migrated from sandbox/sarimax_errors.m (MFE Toolbox Version 4.0,
Kevin Sheppard, Revision 3, 4/1/2004).

This module provides a lightweight bridge between the seasonal ARMA
parameter specification and the core ARMAX residual computation engine.
It converts seasonal ARMA coefficients to their expanded nonseasonal form
via :func:`~mfe_toolbox.sandbox.sarma2arma.sarma2arma`, recombines them
with the exogenous parameter block, and delegates to
:func:`~mfe_toolbox.timeseries.armaxerrors.armaxerrors` for the actual
recursive residual calculation.

Public API
----------
sarimax_errors(parameters, p, q, constant, seasonal, y, x, sigma) -> numpy.ndarray
    Compute SARIMAX residuals by converting seasonal parameters and
    delegating to the ARMAX residual engine.
"""

from __future__ import annotations

import numpy as np

from mfe_toolbox.sandbox.sarma2arma import sarma2arma
from mfe_toolbox.timeseries.armaxerrors import armaxerrors


def sarimax_errors(
    parameters: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    constant: int,
    seasonal: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Compute SARIMAX recursive residuals.

    Adapts seasonal parameterizations for the shared
    :func:`~mfe_toolbox.timeseries.armaxerrors.armaxerrors` recursive
    residual calculator.  The function extracts the ARMA portion of the
    combined parameter vector, expands seasonal AR and MA components into
    their nonseasonal equivalents via
    :func:`~mfe_toolbox.sandbox.sarma2arma.sarma2arma`, recombines the
    exogenous and expanded ARMA parameters, and delegates to
    ``armaxerrors`` for the actual recursion.

    Parameters
    ----------
    parameters : np.ndarray
        Combined 1-D parameter vector ordered as
        ``[exogenous_params (length m), arma_params]`` where
        *arma_params* may include seasonal AR and MA components that
        will be expanded by ``sarma2arma``.
    p : np.ndarray
        1-D integer array of non-seasonal AR lag indices
        (e.g. ``[1, 2]`` for an AR(2) process).  May be empty.
    q : np.ndarray
        1-D integer array of non-seasonal MA lag indices
        (e.g. ``[1]`` for an MA(1) process).  May be empty.
    constant : int
        ``1`` if the model contains a constant term, ``0`` otherwise.
    seasonal : np.ndarray
        Seasonal specification array of shape ``(K, 4)`` with columns:

        * Column 0: seasonal AR orders ``Sp``
        * Column 1: *(reserved / unused)*
        * Column 2: seasonal MA orders ``Sq``
        * Column 3: seasonal period lengths ``S``
    y : np.ndarray
        T-by-1 dependent variable vector.
    x : np.ndarray
        T-by-m exogenous variable matrix (including the constant column
        if present).
    sigma : np.ndarray
        T-by-1 conditional standard deviation vector used for GLS-style
        error normalisation inside ``armaxerrors``.

    Returns
    -------
    np.ndarray
        T-by-1 recursive residual vector (sigma-normalised).

    Raises
    ------
    ValueError
        If *x* is not 2-D, or if *parameters* is too short for the
        combined exogenous + seasonal ARMA specification.

    See Also
    --------
    mfe_toolbox.sandbox.sarma2arma.sarma2arma :
        Seasonal-to-nonseasonal ARMA parameter conversion.
    mfe_toolbox.timeseries.armaxerrors.armaxerrors :
        Numba JIT-accelerated ARMAX recursive residual computation.

    Notes
    -----
    Ref: sandbox/sarimax_errors.m lines 1–35 (Kevin Sheppard, Revision 3).

    The MATLAB source uses 1-based indexing throughout; all index
    translations are annotated with ``# Ref:`` comments referencing the
    original source line numbers.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.sandbox.sarimax_errors import sarimax_errors
    >>> T = 100
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal(T)
    >>> x = np.ones((T, 1))
    >>> sigma = np.ones(T)
    >>> p = np.array([1])
    >>> q = np.array([1])
    >>> seasonal = np.array([[1, 0, 1, 12]])
    >>> # parameters: [const_coef, ar1, sar1, ma1, sma1]
    >>> params = np.array([0.0, 0.5, 0.3, 0.4, 0.2])
    >>> e = sarimax_errors(params, p, q, 0, seasonal, y, x, sigma)
    >>> e.shape == (T,)
    True
    """
    # ------------------------------------------------------------------
    # Input coercion — ensure arrays are proper numpy float64 1-D arrays
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)

    # Validate x is 2-D so that x.shape[1] is meaningful
    if x.ndim == 1:
        # If a 1-D array is passed, treat it as a single-column matrix
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError(
            f"x must be a 1-D or 2-D array; received ndim={x.ndim}."
        )

    # ------------------------------------------------------------------
    # Ref: sarimax_errors.m:32 — m = size(x,2)
    # Number of exogenous/constant columns in x.
    # MATLAB size(x,2) → Python x.shape[1] (0-indexed second dimension).
    # ------------------------------------------------------------------
    m: int = x.shape[1]

    # ------------------------------------------------------------------
    # Ref: sarimax_errors.m:33 —
    #   [armaParameters, p, q] = sarma2arma(
    #       parameters(m+1:length(parameters)), p, q, seasonal)
    #
    # MATLAB 1-indexed slice parameters(m+1:end) extracts elements from
    # position m+1 to end.  In Python 0-indexed slicing, the equivalent
    # is parameters[m:].
    # ------------------------------------------------------------------
    arma_parameters, p_new, q_new = sarma2arma(
        parameters[m:], p, q, seasonal
    )

    # ------------------------------------------------------------------
    # Ref: sarimax_errors.m:34 —
    #   parameters = [parameters(1:m); armaParameters]
    #
    # MATLAB vertical concatenation [a ; b] for column vectors.
    # In Python, both slices are 1-D arrays, so np.concatenate along
    # the default axis=0 produces the equivalent 1-D result.
    # MATLAB parameters(1:m) (1-indexed) → Python parameters[:m] (0-indexed).
    # ------------------------------------------------------------------
    parameters = np.concatenate([parameters[:m], arma_parameters])

    # ------------------------------------------------------------------
    # Burn-in padding: Python safety addition — NOT present in MATLAB
    # original (sarimax_errors.m:35).  The MATLAB source relies on
    # pre-padded data from the caller (armaxfilter.m) so that AR/MA lags
    # never index before the array start.  In a standalone call the MATLAB
    # C MEX (armaxerrors.c) would read uninitialized memory (undefined
    # behaviour in C).  This padding ensures correct zero-initial-condition
    # semantics in Python, preventing negative array indexing when
    # max_lag > burn (the exogenous column count m).
    # Ref: sarimax_errors.m:35 + mex_source/armaxerrors.c:40-65
    # ------------------------------------------------------------------
    max_lag: int = 0
    if len(p_new) > 0:
        max_lag = max(max_lag, int(np.max(p_new)))
    if len(q_new) > 0:
        max_lag = max(max_lag, int(np.max(q_new)))

    T_orig: int = y.shape[0]
    burn: int = m  # start with exogenous-column count

    if max_lag > burn:
        pad = max_lag - burn
        y = np.concatenate([np.zeros(pad), y])
        sigma = np.concatenate([np.ones(pad), sigma])
        if x.shape[1] > 0:
            x = np.vstack([np.zeros((pad, x.shape[1])), x])
        else:
            x = np.empty((y.shape[0], 0))
        burn = max_lag

    # ------------------------------------------------------------------
    # Ref: sarimax_errors.m:35 —
    #   e = armaxerrors(parameters, p, q, constant, y, x, m, sigma)
    #
    # Direct delegation to the ARMAX residual engine.  All arguments pass
    # through; p and q have been updated by sarma2arma to reflect the
    # expanded nonseasonal lag structure.
    # ------------------------------------------------------------------
    e: np.ndarray = armaxerrors(
        parameters, p_new, q_new, constant, y, x, burn, sigma
    )

    # Trim the padded prefix so the returned array matches T_orig
    if e.shape[0] > T_orig:
        e = e[e.shape[0] - T_orig:]

    return e
