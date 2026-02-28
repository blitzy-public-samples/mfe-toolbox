"""
Newey-West HAC long-run covariance estimator with Bartlett kernel weights.

This module provides the ``covnw`` function which computes a heteroskedasticity
and autocorrelation consistent (HAC) covariance matrix using the Bartlett
(Newey-West) kernel.  When the number of lags is zero the estimator reduces to
White's heteroskedasticity-consistent covariance.

Migrated from: utility/covnw.m (79 lines)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3  Date: 5/1/2007
"""

import numpy as np


def covnw(scores: np.ndarray, nw_lags: int | None = None, demean: bool = True) -> np.ndarray:
    """
    Newey-West heteroskedasticity and autocorrelation consistent (HAC)
    covariance estimator.

    Computes the long-run covariance matrix using Bartlett (Newey-West) kernel
    weights.  When ``nw_lags=0`` this reduces to White's heteroskedasticity-
    consistent covariance estimator.

    Parameters
    ----------
    scores : numpy.ndarray
        T × K matrix of scores (moment conditions, e.g. residuals × instruments).
    nw_lags : int or None, optional
        Number of Newey-West lags.  If ``None`` (default), the bandwidth is
        automatically selected as ``min(floor(1.2 * T**(1/3)), T)``.  Set to 0
        for White's heteroskedasticity-consistent covariance.
    demean : bool, optional
        If ``True`` (default), subtract column means before computing the
        covariance.  If ``False``, use scores as-is.

    Returns
    -------
    S : numpy.ndarray
        K × K positive semi-definite long-run covariance matrix.

    Raises
    ------
    ValueError
        If *scores* is not a 2-D array, *nw_lags* is not a non-negative
        integer or is ≥ T, or *demean* is not boolean-like.

    Notes
    -----
    The Bartlett kernel weight for lag *i* is

    .. math::

        w_i = 1 - \\frac{i}{\\text{nw\\_lags} + 1}

    which linearly decreases from 1 (lag 0) to
    :math:`1 / (\\text{nw\\_lags} + 1)` at the maximum lag.

    The estimator computes:

    .. math::

        S = \\frac{1}{T} X^\\top X
            + \\sum_{i=1}^{\\text{nw\\_lags}} w_i
              \\bigl(\\Gamma_i + \\Gamma_i^\\top\\bigr)

    where :math:`\\Gamma_i = \\frac{1}{T} X_{i+1:T}^\\top X_{1:T-i}`.

    References
    ----------
    Newey, W. K. and West, K. D. (1987).  "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    *Econometrica*, 55(3), 703-708.

    See Also
    --------
    mfe_toolbox.utility.covvar : Covariance of variance estimator.
    mfe_toolbox.utility.robustvcv : Robust sandwich variance-covariance.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((100, 3))
    >>> S = covnw(data)                          # automatic bandwidth
    >>> S_white = covnw(data, nw_lags=0)         # White's HC
    >>> S_nw10 = covnw(data, nw_lags=10, demean=False)  # 10 lags, no demean
    """
    # ------------------------------------------------------------------
    # Input Checking  (Ref: covnw.m:36-63)
    # ------------------------------------------------------------------
    # Ensure numpy array
    if not isinstance(scores, np.ndarray):
        scores = np.asarray(scores, dtype=np.float64)

    # Ref: covnw.m:58-60 — DATA must be a T by K matrix
    if scores.ndim != 2:
        raise ValueError("scores must be a T by K matrix of data.")

    # Ref: covnw.m:39 — T=size(data,1)
    T: int = scores.shape[0]
    K: int = scores.shape[1]

    # Ref: covnw.m:40-41,46-48 — automatic bandwidth when nlag not provided
    # nlag = min(floor(1.2 * T^(1/3)), T)
    if nw_lags is None:
        nw_lags = int(np.minimum(np.floor(1.2 * T ** (1.0 / 3.0)), T))

    # Ref: covnw.m:55-57 — nlag must be a non-negative integer
    # Accept float values that are exact integers (e.g. 3.0)
    if isinstance(nw_lags, (float, np.floating)):
        if nw_lags != np.floor(nw_lags) or nw_lags < 0:
            raise ValueError("nw_lags must be a non-negative integer.")
        nw_lags = int(nw_lags)
    elif isinstance(nw_lags, (int, np.integer)):
        if nw_lags < 0:
            raise ValueError("nw_lags must be a non-negative integer.")
        nw_lags = int(nw_lags)
    else:
        raise ValueError("nw_lags must be a non-negative integer.")

    # Ref: covnw.m:50-62 — nw_lags must be less than T for meaningful computation
    if nw_lags >= T:
        raise ValueError(
            "nw_lags must be less than the number of observations T."
        )

    # Ref: covnw.m:52-54 — DEMEAN must be logical true or false
    if demean not in (True, False, 0, 1):
        raise ValueError("demean must be either logical true or false.")

    # ------------------------------------------------------------------
    # Demeaning  (Ref: covnw.m:64-66)
    # ------------------------------------------------------------------
    if demean:
        # Ref: covnw.m:65 — data=data-repmat(mean(data),T,1)
        # numpy broadcasting replaces repmat; axis=0 gives column means
        scores = scores - np.mean(scores, axis=0)

    # ------------------------------------------------------------------
    # Newey-West covariance estimation  (Ref: covnw.m:68-76)
    # ------------------------------------------------------------------
    # Ref: covnw.m:71 — White's heteroskedastic-consistent covariance (lag 0)
    # The Bartlett weight for lag 0 is w(1) = (nlag+1)/(nlag+1) = 1
    S: np.ndarray = scores.T @ scores / T

    # Ref: covnw.m:72-76 — Bartlett-weighted autocovariance lags
    for i in range(1, nw_lags + 1):
        # Ref: covnw.m:69 — Bartlett weight: w=(nlag+1-(0:nlag))./(nlag+1)
        # For lag i: w(i+1) = (nlag+1-i)/(nlag+1) = 1 - i/(nlag+1)
        w: float = 1.0 - i / (nw_lags + 1.0)

        # Ref: covnw.m:73 — Gammai=(data((i+1):T,:)'*data(1:T-i,:))/T
        # MATLAB 1-indexed data(i+1:T,:) → Python 0-indexed scores[i:]
        # MATLAB 1-indexed data(1:T-i,:) → Python 0-indexed scores[:T-i]
        Gamma_i: np.ndarray = scores[i:].T @ scores[:T - i] / T

        # Ref: covnw.m:74-75 — Symmetrize and accumulate weighted autocovariance
        S = S + w * (Gamma_i + Gamma_i.T)

    return S
