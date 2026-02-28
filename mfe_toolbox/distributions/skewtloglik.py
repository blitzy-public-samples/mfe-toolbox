"""
Log-likelihood for Hansen's (1994) Skewed Student's t distribution.

This module computes the log-likelihood of Hansen's Skewed Student's t
distribution, which is widely used in GARCH model estimation to capture
both heavy tails and asymmetry in financial return distributions.

The implementation follows Hansen, B.E. (1994), "Autoregressive Conditional
Density Estimation," International Economic Review, 35(3), 705-730.

Migrated from: distributions/skewtloglik.m (MFE Toolbox v4.0)
"""

import numpy
from scipy.special import gammaln

__all__ = ['skewtloglik']


def skewtloglik(x, mu, sigma2, v, lambda_):
    """
    Compute log-likelihood of Hansen's (1994) Skewed Student's t distribution.

    Parameters
    ----------
    x : array_like
        Observations, either scalar or 1-D array of length T.
    mu : float or array_like
        Mean of x. Either a scalar or array of same shape as x.
    sigma2 : float or array_like
        Variance of x. Either a positive scalar or array of same shape as x.
        All elements must be strictly positive.
    v : float
        Degree of freedom parameter. Must be a scalar strictly greater than 2.
    lambda_ : float
        Asymmetry parameter. Must be a scalar strictly between -1 and 1
        (exclusive). Named ``lambda_`` because ``lambda`` is a Python
        reserved keyword.

    Returns
    -------
    LL : float
        Total log-likelihood evaluated at x.
    lls : numpy.ndarray
        Array of shape (T,) containing individual log-likelihood contributions
        for each observation. ``numpy.sum(lls)`` is approximately equal to
        ``LL``.

    Raises
    ------
    ValueError
        If ``v <= 2``, ``|lambda_| >= 1``, any element of ``sigma2 <= 0``,
        or input dimensions are incompatible.

    Notes
    -----
    The density is parameterized following Hansen (1994) with constants:

    - ``c = exp(gammaln((v+1)/2) - gammaln(v/2) - 0.5*log(pi*(v-2)))``
    - ``a = 4 * lambda_ * c * (v-2) / (v-1)``
    - ``b = sqrt(1 + 3*lambda_**2 - a**2)``

    The piecewise log-likelihood splits at ``stdresid = -a/b`` where the
    lower tail uses scale ``(1 - lambda_)`` and the upper tail uses
    ``(1 + lambda_)``.

    When ``lambda_ = 0``, the distribution reduces to the standardized
    Student's t distribution.

    References
    ----------
    Hansen, B.E. (1994), "Autoregressive Conditional Density Estimation,"
    International Economic Review, 35(3), 705-730.

    See Also
    --------
    skewtcdf : CDF of Hansen's Skewed t distribution.
    skewtinv : Quantile function of Hansen's Skewed t distribution.
    skewtrnd : Random variates from Hansen's Skewed t distribution.
    skewtpdf : PDF of Hansen's Skewed t distribution.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.1, -0.2, 0.5, -0.3, 0.0])
    >>> LL, lls = skewtloglik(x, 0.0, 1.0, 5.0, 0.1)
    >>> isinstance(LL, float)
    True
    >>> lls.shape
    (5,)
    """
    # ---------------------------------------------------------------
    # Input conversion
    # Ref: skewtloglik.m:30 — [T,K]=size(x); MATLAB expects column vector
    # ---------------------------------------------------------------
    x = numpy.asarray(x, dtype=numpy.float64).flatten()
    T = len(x)

    # ---------------------------------------------------------------
    # Input validation
    # Ref: skewtloglik.m:35-56 — parameter checking block
    # ---------------------------------------------------------------
    # Ref: skewtloglik.m:51-52 — V must be a scalar greater than 2
    if numpy.isscalar(v):
        v = float(v)
    else:
        raise ValueError('V must be a scalar greater than 2')
    if v <= 2.0:
        raise ValueError('V must be a scalar greater than 2')

    # Ref: skewtloglik.m:54-55 — LAMBDA must be a scalar between -1 and 1
    if numpy.isscalar(lambda_):
        lambda_ = float(lambda_)
    else:
        raise ValueError('LAMBDA must be a scalar between -1 and 1')
    if lambda_ <= -1.0 or lambda_ >= 1.0:
        raise ValueError('LAMBDA must be a scalar between -1 and 1')

    # Convert mu to array for broadcasting
    # Ref: skewtloglik.m:40-41 — mu can be scalar or same size as x
    mu = numpy.asarray(mu, dtype=numpy.float64).flatten()
    if len(mu) == 1:
        mu = numpy.full(T, mu[0])
    elif len(mu) != T:
        raise ValueError('mu must be either a scalar or the same size as X')

    # Ref: skewtloglik.m:43-44 — sigma2 must contain only positive elements
    sigma2 = numpy.asarray(sigma2, dtype=numpy.float64).flatten()
    if not numpy.all(sigma2 > 0):
        raise ValueError('sigma2 must contain only positive elements')

    # Ref: skewtloglik.m:46-50 — broadcast sigma2 if scalar
    if len(sigma2) == 1:
        sigma2 = numpy.full(T, sigma2[0])
    elif len(sigma2) != T:
        raise ValueError(
            'sigma2 must be a scalar or a vector with the same dimensions as X'
        )

    # ---------------------------------------------------------------
    # Demean observations
    # Ref: skewtloglik.m:57 — x = x - mu
    # ---------------------------------------------------------------
    x = x - mu

    # ---------------------------------------------------------------
    # Compute Hansen (1994) skewed-t log-constants
    # ---------------------------------------------------------------
    # Ref: skewtloglik.m:66 — logc = gammaln((v+1)/2) - gammaln(v/2)
    #                                - 0.5*log(pi*(v-2))
    # gammaln from scipy.special replaces MATLAB built-in gammaln
    logc = gammaln((v + 1.0) / 2.0) - gammaln(v / 2.0) \
        - 0.5 * numpy.log(numpy.pi * (v - 2.0))

    # Ref: skewtloglik.m:67 — c = exp(logc)
    c = numpy.exp(logc)

    # Ref: skewtloglik.m:69 — a = 4*lambda.*c.*((v-2)./(v-1))
    a = 4.0 * lambda_ * c * ((v - 2.0) / (v - 1.0))

    # Ref: skewtloglik.m:70 — logb = 0.5*log(1 + 3*lambda.^2 - a.^2)
    logb = 0.5 * numpy.log(1.0 + 3.0 * lambda_ ** 2 - a ** 2)

    # Ref: skewtloglik.m:71 — b = exp(logb)
    b = numpy.exp(logb)

    # ---------------------------------------------------------------
    # Standardized residuals
    # Ref: skewtloglik.m:73 — stdresid = x ./ sqrt(sigma2)
    # Note: x is already demeaned at this point (line 57)
    # ---------------------------------------------------------------
    stdresid = x / numpy.sqrt(sigma2)

    # ---------------------------------------------------------------
    # Piecewise log-likelihood evaluation
    # Ref: skewtloglik.m:75-76 — Boolean partition at threshold -a/b
    # MATLAB find1/find2 are logical indices; Python uses boolean masks
    # ---------------------------------------------------------------
    threshold = -a / b
    find1 = stdresid < threshold   # lower tail mask
    find2 = ~find1                 # upper tail mask (stdresid >= threshold)

    # Ref: skewtloglik.m:77 — Lower tail log-likelihood contributions
    # LL1 = logb + logc - (v+1)/2 * log(1 + 1/(v-2) *
    #        ((b*stdresid(find1)+a)/(1-lambda))^2)
    LL1 = logb + logc - (v + 1.0) / 2.0 * numpy.log(
        1.0 + 1.0 / (v - 2.0) * (
            (b * stdresid[find1] + a) / (1.0 - lambda_)
        ) ** 2
    )

    # Ref: skewtloglik.m:78 — Upper tail log-likelihood contributions
    # LL2 = logb + logc - (v+1)/2 * log(1 + 1/(v-2) *
    #        ((b*stdresid(find2)+a)/(1+lambda))^2)
    LL2 = logb + logc - (v + 1.0) / 2.0 * numpy.log(
        1.0 + 1.0 / (v - 2.0) * (
            (b * stdresid[find2] + a) / (1.0 + lambda_)
        ) ** 2
    )

    # ---------------------------------------------------------------
    # Total log-likelihood
    # Ref: skewtloglik.m:79 — LL = sum(LL1) + sum(LL2) - 0.5*sum(log(sigma2))
    # The sigma2 correction is applied separately from the density terms
    # ---------------------------------------------------------------
    LL = float(
        numpy.sum(LL1) + numpy.sum(LL2)
        - 0.5 * numpy.sum(numpy.log(sigma2))
    )

    # ---------------------------------------------------------------
    # Individual log-likelihoods per observation
    # Ref: skewtloglik.m:83-85 — LLS(find1) = LL1 - 0.5*log(sigma2(find1))
    #                             LLS(find2) = LL2 - 0.5*log(sigma2(find2))
    # Each observation's contribution includes -0.5*log(sigma2_t)
    # ---------------------------------------------------------------
    lls = numpy.zeros(T, dtype=numpy.float64)
    lls[find1] = LL1 - 0.5 * numpy.log(sigma2[find1])
    lls[find2] = LL2 - 0.5 * numpy.log(sigma2[find2])

    return LL, lls
