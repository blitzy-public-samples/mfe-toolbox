"""
ARMAX Gaussian log-likelihood computation.

Migrated from:
  - timeseries/armaxfilter_likelihood.m (Kevin Sheppard, Revision 3, Date: 4/1/2004)

This module evaluates the Gaussian log-likelihood for ARMAX models.  It
delegates to :func:`armaxerrors` to obtain the residual (error) series,
then computes per-observation log-likelihoods under the assumption of
Gaussian innovations with heteroscedastic conditional standard deviations
supplied via the *sigma* vector.

The negative log-likelihood is returned so that it can be passed directly
to ``scipy.optimize.minimize`` for maximum-likelihood estimation.

Public API
----------
armaxfilter_likelihood(parameters, p, q, constant, y, x, m, sigma)
    -> (llf, likelihoods, errors)

See Also
--------
mfe_toolbox.timeseries.armaxerrors : Computes the ARMAX residuals used
    as input to this likelihood function.
mfe_toolbox.timeseries.armaxfilter : Driver that wraps this likelihood
    inside a ``scipy.optimize.minimize`` call.
"""

import numpy as np

from mfe_toolbox.timeseries.armaxerrors import armaxerrors


def armaxfilter_likelihood(parameters, p, q, constant, y, x, m, sigma):
    """
    Compute the Gaussian log-likelihood for an ARMAX model.

    Evaluates the negative Gaussian log-likelihood for an ARMAX(p, q) model
    with optional exogenous regressors.  The residuals are obtained via
    :func:`armaxerrors` (called with unit sigma to retrieve raw, un-normalised
    residuals) and then standardised by the supplied conditional standard
    deviations *sigma*.

    Parameters
    ----------
    parameters : array_like
        1-D parameter vector laid out as
        ``[constant?, AR_1 … AR_np, X_1 … X_k, MA_1 … MA_nq]``
        where the leading constant element is present only when
        *constant* == 1.
    p : array_like
        Vector of AR lag indices (may be empty or zero-length).
    q : array_like
        Vector of MA lag indices (may be empty or zero-length).
    constant : int
        ``1`` to include a constant term in the model, ``0`` to exclude.
    y : array_like
        Dependent variable (T observations).
    x : array_like
        Exogenous regressors (T × k), excluding the constant.  If there are
        no exogenous variables, pass a 1-D array of zeros with length T.
    m : int
        Index to the first element used in the recursive residual
        calculation.  The first *m* residuals are treated as burn-in.
    sigma : array_like
        T × 1 vector of conditional standard deviations.

    Returns
    -------
    llf : float
        Negative log-likelihood.  Returns ``1e7`` when the computed value
        is ``NaN`` (numerical safeguard for the optimiser).
    likelihoods : numpy.ndarray
        1-D array of per-observation log-likelihood contributions (after
        burn-in trimming).
    errors : numpy.ndarray
        1-D array of model residuals (after burn-in trimming).

    Notes
    -----
    * The function mirrors the MATLAB implementation exactly, including
      the burn-in trimming logic when ``max(q) > max(p)`` and the NaN
      fallback to ``1e7``.
    * Ref: armaxfilter_likelihood.m — MATLAB source.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.timeseries.armaxfilter_likelihood import (
    ...     armaxfilter_likelihood,
    ... )
    >>> y = np.random.default_rng(42).standard_normal(100)
    >>> sigma = np.ones(100)
    >>> params = np.array([0.0])  # constant-only model
    >>> llf, lls, errs = armaxfilter_likelihood(
    ...     params, np.array([]), np.array([]), 1, y, np.zeros(100), 0, sigma
    ... )
    """
    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------
    # Ref: armaxfilter_likelihood.m:32 — ensure y and sigma are float64
    # so that np.ones_like(y) produces a float64 unit-sigma vector.
    y = np.asarray(y, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Compute raw ARMAX residuals
    # Ref: armaxfilter_likelihood.m:32
    #   e = armaxerrors(parameters,p,q,constant,y,x,m,ones(size(y)));
    # We pass unit sigma (ones_like) so that armaxerrors returns the raw
    # (un-normalised) error series.
    # ------------------------------------------------------------------
    e = armaxerrors(parameters, p, q, constant, y, x, m, np.ones_like(y))
    T = len(e)

    # ------------------------------------------------------------------
    # Determine maximum AR and MA lags
    # Ref: armaxfilter_likelihood.m:35-40
    #   if isempty(p), p = 0; end
    #   if isempty(q), q = 0; end
    # ------------------------------------------------------------------
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    q_arr = np.asarray(q, dtype=np.float64).ravel()

    if len(p_arr) == 0:
        # Ref: armaxfilter_likelihood.m:36 — MATLAB sets empty p to 0
        p_max = 0
    else:
        p_max = int(np.max(p_arr))

    if len(q_arr) == 0:
        # Ref: armaxfilter_likelihood.m:39 — MATLAB sets empty q to 0
        q_max = 0
    else:
        q_max = int(np.max(q_arr))

    # ------------------------------------------------------------------
    # Trim burn-in observations when MA order exceeds AR order
    # Ref: armaxfilter_likelihood.m:41-45
    #   MATLAB 1-based: t = (max(q)-max(p))+1:T
    #   Python 0-based: slice(q_max - p_max, T)
    # ------------------------------------------------------------------
    if q_max > p_max:
        # Ref: armaxfilter_likelihood.m:42 — MATLAB 1-indexed offset
        t_slice = slice(q_max - p_max, T)
    else:
        # Ref: armaxfilter_likelihood.m:44 — use all observations
        t_slice = slice(0, T)

    # ------------------------------------------------------------------
    # Standardise errors and estimate residual variance
    # Ref: armaxfilter_likelihood.m:47 — stde = e./sigma
    # Ref: armaxfilter_likelihood.m:48 — sigma2 = stde(t)'*stde(t)/length(t)
    # ------------------------------------------------------------------
    stde = e / sigma
    stde_trimmed = stde[t_slice]
    sigma2 = np.dot(stde_trimmed, stde_trimmed) / len(stde_trimmed)

    # ------------------------------------------------------------------
    # Per-observation Gaussian log-likelihood
    # Ref: armaxfilter_likelihood.m:51
    #   likelihoods = 0.5*(2*log(sigma) + log(sigma2) + stde.^2./sigma2
    #                      + log(2*pi));
    # Computed on full-length vectors, then trimmed below.
    # ------------------------------------------------------------------
    likelihoods = 0.5 * (
        2.0 * np.log(sigma)
        + np.log(sigma2)
        + stde ** 2 / sigma2
        + np.log(2.0 * np.pi)
    )

    # ------------------------------------------------------------------
    # Trim likelihoods and errors to valid observation window
    # Ref: armaxfilter_likelihood.m:53-54
    # ------------------------------------------------------------------
    likelihoods = likelihoods[t_slice]
    errors = e[t_slice]

    # ------------------------------------------------------------------
    # Sum for total negative log-likelihood
    # Ref: armaxfilter_likelihood.m:55
    # ------------------------------------------------------------------
    llf = float(np.sum(likelihoods))

    # ------------------------------------------------------------------
    # NaN safeguard — return large penalty for optimiser
    # Ref: armaxfilter_likelihood.m:57-58
    #   if isnan(LLF), LLF=1e7; end
    # ------------------------------------------------------------------
    if np.isnan(llf):
        llf = 1e7

    return llf, likelihoods, errors
