"""
Log-likelihood of the Standardized Student's t distribution.

Computes the total and individual (per-observation) log-likelihood values
for the standardized Student's t distribution parameterized by location
(mu), scale (sigma2), and degrees of freedom (nu).  This function is the
primary objective used by GARCH model estimators when the innovation
distribution is assumed to follow a standardized t.

Migrated from: distributions/stdtloglik.m — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)

References
----------
Cassella and Berger (1990) 'Statistical Inference'
"""

import numpy
from scipy.special import gammaln


def stdtloglik(x, mu, sigma2, nu):
    """
    Compute the log-likelihood of the Standardized Student's t distribution.

    Parameters
    ----------
    x : array_like
        Standardized T random variables.  Must be interpretable as a 1-D
        array of length T (i.e. a column vector in the original MATLAB
        convention).
    mu : float or array_like
        Mean of *x*.  Either a scalar (applied to every observation) or a
        1-D array of the same length as *x*.
    sigma2 : float or array_like
        Variance of *x*.  Either a positive scalar (broadcast to every
        observation) or a 1-D array of the same length as *x* with all
        elements strictly positive.
    nu : float
        Degrees-of-freedom parameter.  Must be a scalar strictly greater
        than 2 so that the variance of the standardized distribution is
        finite.

    Returns
    -------
    LL : float
        Total (summed) log-likelihood evaluated at *x*.
    lls : numpy.ndarray
        1-D array of length T containing the individual log-likelihood
        contribution of each observation.

    Raises
    ------
    ValueError
        If *x* is empty, *mu* has incompatible length, *sigma2* contains
        non-positive values or has incompatible length, or *nu* is not a
        scalar greater than 2.

    Notes
    -----
    The standardized Student's t density is

    .. math::

        f(x; \\nu) = \\frac{\\Gamma((\\nu+1)/2)}
                          {\\Gamma(\\nu/2)\\,\\sqrt{\\pi(\\nu-2)}}
                     \\left(1 + \\frac{x^{2}}{\\nu - 2}\\right)
                     ^{-(\\nu+1)/2}

    The distribution is standardized so that its variance equals 1 when
    *sigma2* = 1.  The requirement *nu* > 2 ensures the variance exists.

    The total log-likelihood ``LL`` equals ``numpy.sum(lls)`` identically.

    See Also
    --------
    stdtcdf : CDF of the standardized Student's t distribution.
    stdtinv : Quantile function of the standardized Student's t.
    stdtrnd : Random variate generation for the standardized Student's t.
    stdtpdf : PDF of the standardized Student's t distribution.
    """

    # --- Input conversion --------------------------------------------------
    # Ref: stdtloglik.m:30 — [T,K]=size(x); MATLAB expects column (T×1)
    x = numpy.asarray(x, dtype=numpy.float64).flatten()
    T = len(x)

    if T == 0:
        raise ValueError('X must be a non-empty array')

    # --- Validate nu -------------------------------------------------------
    # Ref: stdtloglik.m:51-53 — nu must be scalar > 2
    if not numpy.isscalar(nu):
        raise ValueError('nu must be a scalar greater than 2')
    nu = float(nu)
    if nu <= 2.0:
        raise ValueError('nu must be a scalar greater than 2')

    # --- Validate and broadcast mu -----------------------------------------
    # Ref: stdtloglik.m:40-41 — mu scalar or same size as x
    mu_arr = numpy.asarray(mu, dtype=numpy.float64).flatten()
    if mu_arr.size == 1:
        # Scalar mu — numpy broadcasting handles subtraction below
        mu_val = mu_arr[0]
    elif mu_arr.size == T:
        mu_val = mu_arr
    else:
        raise ValueError('mu must be either a scalar or the same size as X')

    # --- Validate and broadcast sigma2 -------------------------------------
    # Ref: stdtloglik.m:43-50
    sigma2 = numpy.asarray(sigma2, dtype=numpy.float64).flatten()
    if numpy.any(sigma2 <= 0.0):
        raise ValueError('sigma2 must contain only positive elements')
    if sigma2.size == 1:
        # Broadcast scalar sigma2 to full vector — Ref: stdtloglik.m:47
        sigma2 = numpy.full(T, sigma2[0])
    elif sigma2.size != T:
        raise ValueError(
            'sigma2 must be a scalar or a vector with the same dimensions as X'
        )

    # --- Demean x ----------------------------------------------------------
    # Ref: stdtloglik.m:54 — x = x - mu
    x = x - mu_val

    # --- Compute individual log-likelihoods --------------------------------
    # Ref: stdtloglik.m:68-69
    #   lls = gammaln(0.5*(nu+1)) - gammaln(nu/2) - 0.5*log(pi*(nu-2))
    #         - 0.5*log(sigma2) - ((nu+1)/2)*log(1 + x^2/(sigma2*(nu-2)))
    #
    # gammaln() maps directly to scipy.special.gammaln().
    # MATLAB x.^2 → Python x**2.
    # MATLAB log() → numpy.log().
    # MATLAB pi → numpy.pi.
    lls = (gammaln(0.5 * (nu + 1.0))
           - gammaln(0.5 * nu)
           - 0.5 * numpy.log(numpy.pi * (nu - 2.0))
           - 0.5 * numpy.log(sigma2)
           - ((nu + 1.0) / 2.0) * numpy.log(
               1.0 + x ** 2 / (sigma2 * (nu - 2.0))))

    # --- Compute total log-likelihood --------------------------------------
    # Ref: stdtloglik.m:63-64
    LL = float(numpy.sum(lls))

    return LL, lls
