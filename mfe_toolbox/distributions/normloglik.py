"""
Normal distribution log-likelihood computation.

Computes the log-likelihood and per-observation log-likelihoods for
normally distributed data with optional mean and variance parameters.

Migrated from distributions/normloglik.m (MFE Toolbox Version 4.0).

References
----------
[1] Casella, G. and Berger, R.L. (1990) 'Statistical Inference'.
"""

import numpy as np

__all__ = ['normloglik']


def normloglik(
    x: np.ndarray,
    mu: np.ndarray | float | None = None,
    sigma2: np.ndarray | float | None = None,
) -> tuple[float, np.ndarray]:
    """
    Compute the log-likelihood for the normal distribution.

    Parameters
    ----------
    x : np.ndarray
        Normal random variables. Must be a column vector of shape (T, 1).
    mu : np.ndarray or float or None, optional
        Mean of x. Either a scalar or an array of shape (T, 1).
        If None, mu is treated as 0 (no demeaning is performed).
        Ref: normloglik.m — nargin==1 path skips mu subtraction.
    sigma2 : np.ndarray or float or None, optional
        Variance of x. Either a positive scalar or an array of shape (T, 1)
        with all positive elements. If None, sigma2 defaults to ones (unit
        variance). Ref: normloglik.m — nargin<=2 sets sigma2=ones(T,K).

    Returns
    -------
    LL : float
        Total log-likelihood evaluated at x, equal to sum(lls).
    lls : np.ndarray
        Vector of per-observation log-likelihoods with shape (T, 1).

    Raises
    ------
    ValueError
        If x is not a column vector (shape (T, 1) with second dimension == 1).
        If mu is not a scalar and not conformable with x.
        If sigma2 contains non-positive elements.
        If sigma2 is not a scalar and has incompatible dimensions with x.

    Notes
    -----
    The per-observation log-likelihood is computed as:

        lls_t = -0.5 * (log(2*pi) + log(sigma2_t) + (x_t - mu_t)^2 / sigma2_t)

    and the total log-likelihood is LL = sum(lls).

    Numerical parity with the MATLAB implementation is maintained to ±1e-6.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1.0], [2.0], [3.0]])
    >>> LL, lls = normloglik(x)
    >>> LL  # doctest: +SKIP
    -7.2568...
    """
    # Convert x to numpy array and validate column vector shape
    # Ref: normloglik.m:27 — [T,K]=size(x)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 1:
        # Ref: normloglik.m:32-34 — if K~=1, error('x must be a column vector')
        raise ValueError('x must be a column vector')

    T = x.shape[0]
    # Ref: normloglik.m:27 — K is always 1 at this point

    if mu is None and sigma2 is None:
        # Ref: normloglik.m:36-37 — nargin==1: sigma2=ones(T,K), no mu subtraction
        sigma2_arr = np.ones((T, 1), dtype=np.float64)
        # x remains unchanged (mu effectively 0)
    elif sigma2 is None:
        # Ref: normloglik.m:38-44 — nargin==2: validate mu, demean, sigma2=ones(T,K)
        mu_arr = np.asarray(mu, dtype=np.float64)

        # Ref: normloglik.m:40 — if length(mu)~=1 && ~all(size(mu)==[T K])
        if mu_arr.size != 1 and mu_arr.shape != (T, 1):
            # Allow scalar or (T,) shape by reshaping, but check conformability
            if mu_arr.ndim == 1 and mu_arr.shape[0] == T:
                mu_arr = mu_arr.reshape(T, 1)
            else:
                raise ValueError(
                    'mu must be either a scalar or the same size as X'
                )

        # Ref: normloglik.m:43 — x=x-mu
        x = x - mu_arr
        sigma2_arr = np.ones((T, 1), dtype=np.float64)
    else:
        # Ref: normloglik.m:45-58 — nargin==3: validate mu, validate sigma2, demean
        # Validate mu conformability
        # Ref: normloglik.m:46-48
        mu_arr = np.asarray(mu, dtype=np.float64)
        if mu_arr.size != 1 and mu_arr.shape != (T, 1):
            if mu_arr.ndim == 1 and mu_arr.shape[0] == T:
                mu_arr = mu_arr.reshape(T, 1)
            else:
                raise ValueError(
                    'mu must be either a scalar or the same size as X'
                )

        # Validate sigma2 positivity
        # Ref: normloglik.m:49-50
        sigma2_arr = np.asarray(sigma2, dtype=np.float64)
        if np.any(sigma2_arr <= 0):
            raise ValueError('sigma2 must contain only positive elements')

        # Ref: normloglik.m:52-56 — handle scalar or vector sigma2
        if sigma2_arr.size == 1:
            # Ref: normloglik.m:52-53 — if length(sigma2)==1, sigma2=sigma2*ones(T,K)
            sigma2_arr = sigma2_arr.flat[0] * np.ones((T, 1), dtype=np.float64)
        else:
            # Ref: normloglik.m:54-55 — validate size(sigma2,1)==T and size(sigma2,2)==1
            if sigma2_arr.ndim == 1 and sigma2_arr.shape[0] == T:
                sigma2_arr = sigma2_arr.reshape(T, 1)
            elif sigma2_arr.shape != (T, 1):
                raise ValueError(
                    'sigma2 must be a scalar or a vector with the same '
                    'dimensions as x'
                )

        # Ref: normloglik.m:57 — x=x-mu
        x = x - mu_arr

    # Compute per-observation log-likelihoods
    # Ref: normloglik.m:68 — lls = -0.5*(log(2*pi) + log(sigma2) + x.^2./sigma2)
    lls = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2_arr) + x ** 2 / sigma2_arr)

    # Compute aggregate log-likelihood
    # Ref: normloglik.m:70 — LL = sum(lls)
    LL = float(np.sum(lls))

    return LL, lls
