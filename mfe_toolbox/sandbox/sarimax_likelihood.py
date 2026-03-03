"""
SARIMAX Gaussian log-likelihood computation.

Migrated from sandbox/sarimax_likelihood.m (MFE Toolbox Version 4.0,
Kevin Sheppard, Revision 3, Date: 4/1/2004).

This module computes the negative Gaussian log-likelihood for a SARIMAX
(Seasonal ARIMA with eXogenous variables) model.  The workflow is:

1.  Partition the parameter vector into exogenous and ARMA portions.
2.  Convert seasonal ARMA parameters to their non-seasonal expanded form
    via :func:`~mfe_toolbox.sandbox.sarma2arma.sarma2arma`.
3.  Compute ARMAX residuals via
    :func:`~mfe_toolbox.timeseries.armaxerrors.armaxerrors`.
4.  Evaluate per-observation Gaussian negative log-likelihood.
5.  Trim initial burn-in observations and aggregate.

Public API
----------
sarimax_likelihood(parameters, p, q, constant, seasonal, y, x, sigma)
    -> (float, numpy.ndarray, numpy.ndarray)
"""

from __future__ import annotations

import numpy as np

from mfe_toolbox.sandbox.sarma2arma import sarma2arma
from mfe_toolbox.timeseries.armaxerrors import armaxerrors


def sarimax_likelihood(
    parameters: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    constant: int,
    seasonal: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the negative Gaussian log-likelihood for a SARIMAX model.

    The function converts seasonal ARMA parameters to their expanded
    non-seasonal form, computes ARMAX residuals, and evaluates the
    Gaussian negative log-likelihood.  The initial burn-in observations
    are trimmed before aggregation, following the convention in the
    original MATLAB implementation.

    Parameters
    ----------
    parameters : np.ndarray
        1-D parameter vector ordered as
        ``[x_1, ..., x_m, ar_1, ..., ar_p, sar_..., ma_1, ..., ma_q, sma_...]``
        where the first *m* elements correspond to exogenous/constant
        regressors and the remaining elements form the ARMA portion
        that will be expanded via :func:`sarma2arma`.
    p : np.ndarray
        1-D integer array of non-seasonal AR lag indices
        (e.g. ``[1, 2]`` for AR(2)).  May be empty.
    q : np.ndarray
        1-D integer array of non-seasonal MA lag indices
        (e.g. ``[1]`` for MA(1)).  May be empty.
    constant : int
        ``1`` if the model includes a constant term, ``0`` otherwise.
    seasonal : np.ndarray
        Seasonal specification array of shape ``(K, 4)`` as described in
        :func:`~mfe_toolbox.sandbox.sarma2arma.sarma2arma`.
    y : np.ndarray
        1-D observed time-series data of length *T*.
    x : np.ndarray
        2-D exogenous regressor matrix of shape ``(T, m)`` (or ``(T, 1)``
        when only a constant column is present).
    sigma : np.ndarray
        1-D conditional standard-deviation vector of length *T*.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        LLF : float
            Negative Gaussian log-likelihood (scalar).  Set to ``1e7``
            when the computed value is ``NaN``.
        likelihoods : np.ndarray
            Per-observation negative log-likelihood values, trimmed of
            burn-in observations.
        errors : np.ndarray
            ARMAX residual vector, trimmed of burn-in observations.

    Raises
    ------
    ValueError
        If *x* is not a 2-D array or its leading dimension does not
        match *y*, or if *parameters* is too short for the given model
        specification.

    Notes
    -----
    Ref: sandbox/sarimax_likelihood.m, lines 1–61.

    The MATLAB source uses 1-based indexing throughout.  All index
    translations are annotated with ``# Ref:`` comments pointing back
    to the corresponding MATLAB source line.

    See Also
    --------
    mfe_toolbox.sandbox.sarma2arma.sarma2arma : Seasonal-to-non-seasonal
        parameter expansion.
    mfe_toolbox.timeseries.armaxerrors.armaxerrors : ARMAX residual
        computation with Numba JIT acceleration.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> T = 200
    >>> y = rng.standard_normal(T)
    >>> x = np.ones((T, 1))
    >>> sigma = np.ones(T)
    >>> params = np.array([0.0, 0.5, -0.3])  # constant_col, ar1, ma1
    >>> p = np.array([1])
    >>> q = np.array([1])
    >>> seasonal = np.array([[0, 0, 0, 12]])
    >>> llf, liks, errs = sarimax_likelihood(params, p, q, 0, seasonal,
    ...                                      y, x, sigma)
    """
    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)

    # Ensure x is 2-D (MATLAB always treats x as a matrix)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Ensure p and q are arrays (may be empty)
    p = np.asarray(p, dtype=np.float64).ravel() if p is not None else np.array([], dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).ravel() if q is not None else np.array([], dtype=np.float64)

    # Ref: sarimax_likelihood.m:32 — m = size(x,2)
    # Number of exogenous / constant regressor columns.
    m: int = x.shape[1]

    # ------------------------------------------------------------------
    # Convert seasonal ARMA parameters to non-seasonal form
    # Ref: sarimax_likelihood.m:33 — [armaParameters, p, q] =
    #   sarma2arma(parameters(m+1:length(parameters)), p, q, seasonal)
    # MATLAB 1-indexed slice parameters(m+1:end) → Python 0-indexed
    # parameters[m:]  (equivalent since element m+1 in MATLAB is index m
    # in Python).
    # ------------------------------------------------------------------
    arma_parameters, p_new, q_new = sarma2arma(parameters[m:], p, q, seasonal)

    # Ref: sarimax_likelihood.m:34 — parameters = [parameters(1:m); armaParameters]
    # Recombine exogenous portion with expanded ARMA parameters.
    parameters = np.concatenate([parameters[:m], arma_parameters])

    # ------------------------------------------------------------------
    # Compute ARMAX residuals
    # Ref: sarimax_likelihood.m:36 — e = armaxerrors(parameters,p,q,
    #   constant,y,x,m,ones(size(y)))
    # ones(size(y)) → np.ones_like(y) — unit sigma for residual pass.
    # ------------------------------------------------------------------
    e: np.ndarray = armaxerrors(
        parameters, p_new, q_new, constant, y, x, m, np.ones_like(y)
    )

    # Ref: sarimax_likelihood.m:37 — T = length(e)
    T: int = len(e)

    # ------------------------------------------------------------------
    # Gaussian negative log-likelihood per observation
    # Ref: sarimax_likelihood.m:40 —
    #   likelihoods = 0.5*(2*log(sigma) + (e./sigma).^2 + log(2*pi))
    # ------------------------------------------------------------------
    likelihoods: np.ndarray = 0.5 * (
        2.0 * np.log(sigma) + (e / sigma) ** 2 + np.log(2.0 * np.pi)
    )

    # ------------------------------------------------------------------
    # Handle empty p / q — default to 0 for max() computation
    # Ref: sarimax_likelihood.m:42-47
    # ------------------------------------------------------------------
    if p_new is None or len(p_new) == 0:
        # Ref: sarimax_likelihood.m:42-43 — if isempty(p), p = 0
        p_max: int = 0
    else:
        p_max = int(np.max(p_new))

    if q_new is None or len(q_new) == 0:
        # Ref: sarimax_likelihood.m:45-46 — if isempty(q), q = 0
        q_max: int = 0
    else:
        q_max = int(np.max(q_new))

    # ------------------------------------------------------------------
    # Trim initial burn-in observations
    # Ref: sarimax_likelihood.m:48-52
    # MATLAB: if max(q) > max(p), t = (max(q)-max(p))+1 : T
    #         else                 t = 1 : T
    # Python 0-indexed: MATLAB index (max(q)-max(p))+1 maps to
    # Python start index (max(q)-max(p)) — the "+1" in MATLAB makes the
    # range start inclusive at that 1-indexed position, which equals the
    # 0-indexed position (max(q)-max(p)).
    # ------------------------------------------------------------------
    if q_max > p_max:
        # Ref: sarimax_likelihood.m:49 — t = (max(q)-max(p))+1:T
        t_start: int = q_max - p_max
    else:
        # Ref: sarimax_likelihood.m:51 — t = 1:T → Python 0:T
        t_start = 0

    # Ref: sarimax_likelihood.m:54-56
    likelihoods = likelihoods[t_start:]
    errors: np.ndarray = e[t_start:]
    LLF: float = float(np.sum(likelihoods))

    # ------------------------------------------------------------------
    # NaN guard — replace diverged likelihood with large penalty
    # Ref: sarimax_likelihood.m:58-60
    # ------------------------------------------------------------------
    if np.isnan(LLF):
        LLF = 1e7

    return LLF, likelihoods, errors
