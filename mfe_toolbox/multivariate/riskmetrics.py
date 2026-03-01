"""
RiskMetrics EWMA (Exponentially Weighted Moving Average) covariance estimation.

Migrated from multivariate/riskmetrics.m (82 lines) — Kevin Sheppard, Revision 3, 03/10/2011.

Computes the RiskMetrics EWMA conditional covariance model:
    H(t) = (1 - lambda) * r(t-1)' * r(t-1) + lambda * H(t-1)

This is a pure filtering/recursion operation — no optimization is required.
The standard RiskMetrics decay parameter is lambda = 0.94.

References
----------
RiskMetrics — Technical Document, J.P. Morgan/Reuters, 1996.
"""

import numpy as np


def riskmetrics(data, lambda_param=0.94, back_cast=None):
    """
    Compute the RiskMetrics or other EWMA covariance estimator.

    Parameters
    ----------
    data : numpy.ndarray
        A T-by-K matrix of zero-mean residuals, or a K-by-K-by-T array of
        covariance estimators (e.g., realized covariance matrices).
    lambda_param : float, optional
        EWMA smoothing parameter, must satisfy 0 < lambda_param < 1.
        Default is 0.94 (the standard RiskMetrics value).
        Note: named ``lambda_param`` because ``lambda`` is a reserved
        Python keyword.
    back_cast : numpy.ndarray or None, optional
        A K-by-K positive semi-definite matrix to use as the initial
        covariance value. If None (default), a backward EWMA of the
        initial observations is used.

    Returns
    -------
    Ht : numpy.ndarray
        A K-by-K-by-T array of conditional covariance matrices.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100, 3)
    >>> Ht = riskmetrics(data, 0.94)
    >>> Ht_with_backcast = riskmetrics(data, 0.94, np.cov(data.T))

    Notes
    -----
    The conditional covariance follows the EWMA recursion:

        H(t) = (1 - lambda) * r(t-1)' * r(t-1) + lambda * H(t-1)

    When ``data`` is provided as a T-by-K matrix of returns, outer products
    r(t)' * r(t) are computed internally before running the recursion.
    """
    # -----------------------------------------------------------------------
    # Input validation
    # Ref: riskmetrics.m:35-42 — nargin switch replaced by Python defaults
    # -----------------------------------------------------------------------
    data = np.asarray(data, dtype=np.float64)

    # Ref: riskmetrics.m:45-52 — Convert 2D data to K×K×T outer product array
    if data.ndim == 2:
        # data is T×K: compute outer products for each time step
        T, k = data.shape
        temp = np.zeros((k, k, T))
        for t in range(T):
            # Ref: riskmetrics.m:49 — data(t,:)'*data(t,:) → np.outer
            temp[:, :, t] = np.outer(data[t, :], data[t, :])
        data = temp
    elif data.ndim == 3:
        k = data.shape[0]
        T = data.shape[2]
    else:
        raise ValueError('DATA must be a T-by-K matrix or a K-by-K-by-T 3D array.')

    # Ref: riskmetrics.m:54-56 — lambda validation
    if lambda_param <= 0 or lambda_param >= 1:
        raise ValueError('LAMBDA must be between 0 and 1.')

    # Ref: riskmetrics.m:58-67 — Backward EWMA backcast when back_cast is None
    if back_cast is None:
        # Compute the number of initial observations to use for backcasting
        # Ref: riskmetrics.m:59 — endPoint = max(min(floor(log(.01)/log(lambda)),T),k)
        end_point = int(max(
            min(np.floor(np.log(0.01) / np.log(lambda_param)), T),
            k
        ))
        # Ref: riskmetrics.m:60 — weights = (1-lambda).*lambda.^(0:endPoint-1)
        weights = (1.0 - lambda_param) * lambda_param ** np.arange(end_point)
        # Ref: riskmetrics.m:61 — weights = weights/sum(weights)
        weights = weights / np.sum(weights)
        # Ref: riskmetrics.m:62-65 — backward EWMA of initial observations
        back_cast = np.zeros((k, k))
        for i in range(end_point):
            # Ref: riskmetrics.m:64 — backCast = backCast + weights(i)*data(:,:,i)
            # MATLAB 1-indexed i → Python 0-indexed i (already correct here)
            back_cast = back_cast + weights[i] * data[:, :, i]
    else:
        back_cast = np.asarray(back_cast, dtype=np.float64)

    # Ref: riskmetrics.m:69 — backCast = (backCast+backCast)/2
    # Note: In the original MATLAB source this is (backCast+backCast)/2 (no transpose),
    # which is mathematically an identity operation. Preserved for exact fidelity.
    back_cast = (back_cast + back_cast) / 2.0

    # Ref: riskmetrics.m:70-72 — Check that backcast is positive semi-definite
    eigvals = np.linalg.eig(back_cast)[0]
    if np.min(np.real(eigvals)) < 0:
        raise ValueError('BACKCAST must be positive semidefinite if provided.')

    # -----------------------------------------------------------------------
    # EWMA Recursion
    # Ref: riskmetrics.m:77-81
    # -----------------------------------------------------------------------
    Ht = np.zeros((k, k, T))
    # Ref: riskmetrics.m:78 — Ht(:,:,1) = backCast
    Ht[:, :, 0] = back_cast
    # Ref: riskmetrics.m:79-81 — for i=2:T, Ht(:,:,i) = (1-lambda)*data(:,:,i-1) + lambda*Ht(:,:,i-1)
    # MATLAB 1-indexed i=2:T → Python 0-indexed i=1:T-1
    for i in range(1, T):
        Ht[:, :, i] = ((1.0 - lambda_param) * data[:, :, i - 1]
                        + lambda_param * Ht[:, :, i - 1])

    return Ht
