"""
Probability Density Function (PDF) for the Standardized Student's t distribution.

This module computes the PDF of a Student's t distribution that has been
standardized to have unit variance (when nu > 2), with location parameter mu
and scale parameter sigma2 (variance).

The standardized t distribution is commonly used in GARCH-family models for
modeling heavy-tailed return distributions in financial econometrics.

MATLAB Source Reference
-----------------------
Migrated from ``distributions/stdtpdf.m`` (MFE Toolbox v4.0, Kevin Sheppard).

**Bug Fix Note**: The original MATLAB source (stdtpdf.m) contains a bug on line 60
where ``(x-mu)`` is used after ``x`` has already been demeaned on line 53
(``x = x - mu``). This effectively double-subtracts mu. The Python version
corrects this by using ``x`` (already demeaned) in the PDF formula instead of
``(x - mu)``.

References
----------
[1] Cassella, G. and Berger, R.L. (1990) 'Statistical Inference', Wadsworth.

See Also
--------
stdtcdf : CDF of the standardized Student's t distribution
stdtinv : Inverse CDF (quantile function) of the standardized Student's t
stdtrnd : Random variate generation from the standardized Student's t
stdtloglik : Log-likelihood of the standardized Student's t distribution
"""

import numpy
from scipy.special import gammaln

__all__ = ['stdtpdf']


def stdtpdf(x, mu, sigma2, nu):
    """
    Compute the PDF of the Standardized Student's t distribution.

    Evaluates the probability density function of the standardized Student's t
    distribution with location ``mu``, variance ``sigma2``, and ``nu`` degrees
    of freedom. The distribution is parameterized so that when sigma2 = 1 and
    mu = 0, the variance of the distribution equals 1 (for nu > 2).

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF. Will be flattened to a 1-D array
        of length T.
    mu : scalar or array_like
        Location parameter (mean). If scalar, applied uniformly to all
        observations. If array, must have length T matching ``x``.
    sigma2 : scalar or array_like
        Variance (scale squared) parameter. Must be strictly positive.
        If scalar, broadcast to length T. If array, must have length T.
    nu : scalar
        Degrees of freedom. Must be a scalar strictly greater than 2.

    Returns
    -------
    y : numpy.ndarray
        PDF values evaluated at each element of ``x``. Shape is ``(T,)``
        where T = len(x).

    Raises
    ------
    ValueError
        If ``nu`` is not a scalar or is <= 2.
        If any element of ``sigma2`` is <= 0.
        If ``sigma2`` is an array whose length does not match ``x``.
        If ``mu`` is an array whose length does not match ``x``.
        If exactly 4 arguments are not provided (enforced by signature).

    Notes
    -----
    The PDF of the standardized Student's t is:

    .. math::

        f(x; \\mu, \\sigma^2, \\nu) = \\frac{\\Gamma((\\nu+1)/2)}{\\Gamma(\\nu/2)
        \\sqrt{\\pi (\\nu-2) \\sigma^2}}
        \\left(1 + \\frac{(x - \\mu)^2}{\\sigma^2 (\\nu - 2)}\\right)^{-(\\nu+1)/2}

    **MATLAB Bug Correction**: The original MATLAB implementation (stdtpdf.m)
    demeaned ``x`` on line 53 (``x = x - mu``) but then used ``(x - mu)`` again
    in the PDF formula on line 60, effectively double-subtracting ``mu``. This
    Python version correctly uses the already-demeaned ``x`` in the formula.
    Ref: stdtpdf.m:53,60.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.distributions.stdtpdf import stdtpdf
    >>> x = np.array([0.0, 1.0, -1.0])
    >>> y = stdtpdf(x, 0.0, 1.0, 5.0)
    """
    # -----------------------------------------------------------------
    # Input conversion
    # Ref: stdtpdf.m:29 — [T,K]=size(x); MATLAB enforces column vector
    # -----------------------------------------------------------------
    x = numpy.asarray(x, dtype=numpy.float64).flatten()
    T = len(x)

    # -----------------------------------------------------------------
    # Validate nu (degrees of freedom): must be scalar and > 2
    # Ref: stdtpdf.m:50-51 — if length(nu)>1 || nu<=2, error(...)
    # -----------------------------------------------------------------
    if not numpy.isscalar(nu):
        raise ValueError('nu must be a scalar greater than 2')
    nu = float(nu)
    if nu <= 2.0:
        raise ValueError('nu must be a scalar greater than 2')

    # -----------------------------------------------------------------
    # Validate and broadcast mu: scalar or array of length T
    # Ref: stdtpdf.m:39-41 — if length(mu)~=1 && ~all(size(mu)==[T K])
    # -----------------------------------------------------------------
    mu = numpy.asarray(mu, dtype=numpy.float64).flatten()
    if mu.size == 1:
        # Scalar mu — broadcast via numpy operations (no explicit expansion needed)
        mu = mu[0]  # extract scalar for efficient subtraction
    elif mu.size != T:
        raise ValueError(
            'mu must be either a scalar or the same size as X'
        )

    # -----------------------------------------------------------------
    # Validate sigma2: all elements must be > 0
    # Ref: stdtpdf.m:42-43 — if any(sigma2<=0), error(...)
    # -----------------------------------------------------------------
    sigma2 = numpy.asarray(sigma2, dtype=numpy.float64).flatten()
    if numpy.any(sigma2 <= 0.0):
        raise ValueError('sigma2 must contain only positive elements')

    # -----------------------------------------------------------------
    # Broadcast sigma2: scalar → array of length T, or verify length
    # Ref: stdtpdf.m:45-49 — if length(sigma2)==1, sigma2=sigma2*ones(T,K)
    # -----------------------------------------------------------------
    if sigma2.size == 1:
        # Ref: stdtpdf.m:46 — sigma2 = sigma2*ones(T,K)
        sigma2 = numpy.full(T, sigma2[0], dtype=numpy.float64)
    elif sigma2.size != T:
        raise ValueError(
            'sigma2 must be a scalar or a vector with the same dimensions as X'
        )

    # -----------------------------------------------------------------
    # Demean x by subtracting mu
    # Ref: stdtpdf.m:53 — x = x - mu
    # -----------------------------------------------------------------
    x = x - mu

    # -----------------------------------------------------------------
    # Compute the normalizing constant using the log-gamma function
    # Ref: stdtpdf.m:59 — constant = exp(gammaln(0.5*(nu+1)) - gammaln(0.5*nu))
    # gammaln(x) → scipy.special.gammaln(x) — direct MATLAB replacement
    # -----------------------------------------------------------------
    constant = numpy.exp(
        gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
    )

    # -----------------------------------------------------------------
    # Compute the standardized t PDF
    # Ref: stdtpdf.m:60 — MATLAB bug: double-subtracted mu; Python correctly
    # uses demeaned x.
    #
    # MATLAB original (buggy):
    #   y = constant ./ sqrt(pi*(nu-2)*sigma2) .* (1 + (x-mu).^2 / (sigma2*(nu-2))) .^ (-(nu+1)/2)
    # Corrected formula (uses x which is already demeaned):
    #   y = constant / sqrt(pi*(nu-2)*sigma2) * (1 + x^2 / (sigma2*(nu-2))) ^ (-(nu+1)/2)
    # -----------------------------------------------------------------
    y = (
        constant
        / numpy.sqrt(numpy.pi * (nu - 2.0) * sigma2)
        * (1.0 + x ** 2.0 / (sigma2 * (nu - 2.0))) ** (-(nu + 1.0) / 2.0)
    )

    return y
