"""
Generalized Error Distribution (GED) log-likelihood computation.

Computes the log-likelihood of the Generalized Error Distribution,
used in GARCH model estimation with GED-distributed innovations.

Migrated from: distributions/gedloglik.m (MFE Toolbox, Version 4.0)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Original revision: 1, Date: 9/1/2005

References
----------
.. [1] Tadikamalla (1980), "Random Sampling from the Generalized Error
       Distribution", J. Am. Stat. Assoc. (75)
.. [2] Nelson (1991), "Conditional Heteroskedasticity in Asset Returns:
       A New Approach", Econometrica

See Also
--------
gedcdf : GED cumulative distribution function
gedinv : GED inverse (quantile) function
gedrnd : GED random variate generation
gedpdf : GED probability density function
"""

import numpy  # Core numerical array operations — Ref: external_imports schema
from scipy.special import gammaln  # gammaln(x) → scipy.special.gammaln(x), direct MATLAB mapping


def gedloglik(x, mu, sigma2, v):
    """
    Compute the log-likelihood of the Generalized Error Distribution (GED).

    The GED density is:

        f(x, v) = [v / (lambda * 2^(1+1/v) * Gamma(1/v))]
                   * exp(-0.5 * |x / lambda|^v)

    where lambda = [2^(-2/v) * Gamma(1/v) / Gamma(3/v)]^0.5

    Parameters
    ----------
    x : array_like
        T observations (data vector). Accepts any shape; internally flattened
        to a 1-D array of length T.
        Ref: gedloglik.m:8 — "Standardized T random variables"
    mu : scalar or array_like
        Mean of x. Either a scalar (broadcast to all observations) or an
        array of length T matching x.
        Ref: gedloglik.m:9
    sigma2 : scalar or array_like
        Variance of x. Must contain only positive values. Either a scalar
        (broadcast to length T) or an array of length T.
        Ref: gedloglik.m:10
    v : scalar
        Shape (degree-of-freedom) parameter. Must be a scalar strictly
        greater than 1.
        Ref: gedloglik.m:11

    Returns
    -------
    LL : numpy.float64
        Total log-likelihood (scalar sum of individual log-likelihoods).
    lls : numpy.ndarray
        Array of length T containing individual observation log-likelihoods.

    Raises
    ------
    ValueError
        If x is empty, mu length mismatches x, sigma2 contains non-positive
        elements, sigma2 length mismatches x, or v is not a scalar > 1.

    Notes
    -----
    - MATLAB ``nargin==4`` check → all 4 parameters are required in the Python
      signature (no optional arguments).
      Ref: gedloglik.m:41,57-58
    - MATLAB ``nargout>1`` check → Python always computes both LL and lls
      (cheap overhead).
      Ref: gedloglik.m:71
    - Column vector enforcement uses ``.flatten()`` to handle both row and
      column input shapes.
      Ref: gedloglik.m:37-38

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.distributions.gedloglik import gedloglik
    >>> rng = np.random.default_rng(42)
    >>> x = rng.standard_normal(100)
    >>> LL, lls = gedloglik(x, 0.0, 1.0, 2.0)
    >>> print(f"Total LL: {LL:.4f}, len(lls): {len(lls)}")
    """
    # ---------------------------------------------------------------
    # Input Validation
    # ---------------------------------------------------------------

    # Convert x to float64 1-D array — handles column/row/scalar inputs
    # Ref: gedloglik.m:32 — [T,K]=size(x); uses column vector convention
    x = numpy.asarray(x, dtype=numpy.float64).flatten()
    T = len(x)  # Ref: gedloglik.m:32

    if T == 0:
        raise ValueError("x must be a non-empty array")

    # Validate v: must be scalar and > 1
    # Ref: gedloglik.m:53-54 — if length(v)>1 || v<=1 → error(...)
    # error('msg') → raise ValueError('msg')
    if not numpy.isscalar(v) or v <= 1:
        raise ValueError("V must be a scalar greater than 1")

    # Convert and validate mu
    # Ref: gedloglik.m:42-43 — mu must be scalar or same size as x
    mu = numpy.asarray(mu, dtype=numpy.float64)
    if mu.ndim > 0:
        mu = mu.flatten()
        if mu.size != 1 and mu.size != T:
            raise ValueError(
                "mu must be either a scalar or the same size as x"
            )
        # If mu is a single-element array, treat as scalar for broadcasting
        if mu.size == 1:
            mu = mu[0]

    # Convert and validate sigma2
    # Ref: gedloglik.m:45-46 — if any(sigma2<=0) → error(...)
    sigma2 = numpy.asarray(sigma2, dtype=numpy.float64)
    if sigma2.ndim > 0:
        sigma2 = sigma2.flatten()

    if numpy.any(sigma2 <= 0):
        raise ValueError("sigma2 must contain only positive elements")

    # Broadcast sigma2: scalar → array of length T
    # Ref: gedloglik.m:48-49 — if length(sigma2)==1 → sigma2=sigma2*ones(T,K)
    # ones(T,K) → numpy.full(T, float(sigma2)) for broadcasting
    if numpy.isscalar(sigma2) or sigma2.size == 1:
        sigma2_val = float(sigma2) if numpy.isscalar(sigma2) else float(sigma2.flat[0])
        sigma2 = numpy.full(T, sigma2_val, dtype=numpy.float64)
    else:
        # Ref: gedloglik.m:50-51 — if size(sigma2,1)~=T || size(sigma2,2)~=1 → error(...)
        if sigma2.size != T:
            raise ValueError(
                "sigma2 must be a scalar or a vector with the same dimensions as x"
            )

    # Demean x by mu
    # Ref: gedloglik.m:56 — x = x - mu
    x = x - mu

    # ---------------------------------------------------------------
    # Log-Likelihood Computation
    # ---------------------------------------------------------------

    # Compute log of the lambda scale factor
    # Ref: gedloglik.m:65 — logl = 0.5*(-2/v*log(2) + gammaln(1/v) - gammaln(3/v))
    # gammaln(x) → scipy.special.gammaln(x) — direct MATLAB mapping
    v_float = float(v)
    logl = 0.5 * (
        -2.0 / v_float * numpy.log(2.0)
        + gammaln(1.0 / v_float)
        - gammaln(3.0 / v_float)
    )

    # Lambda scale factor
    # Ref: gedloglik.m:66 — l = exp(logl)
    l = numpy.exp(logl)

    # Standardized absolute residuals raised to power v
    # Ref: gedloglik.m:68,72 — abs(x./(sqrt(sigma2)*l)).^v
    # abs(x).^v → numpy.abs(x) ** v — element-wise power
    abs_std_residuals_v = numpy.abs(x / (numpy.sqrt(sigma2) * l)) ** v_float

    # Individual observation log-likelihoods
    # Ref: gedloglik.m:72 — LLS = log(v) - logl - gammaln(1/v) - (1+1/v)*log(2)
    #                              - 0.5*log(sigma2) - 0.5*(abs(x./(sqrt(sigma2)*l))).^v
    lls = (
        numpy.log(v_float)
        - logl
        - gammaln(1.0 / v_float)
        - (1.0 + 1.0 / v_float) * numpy.log(2.0)
        - 0.5 * numpy.log(sigma2)
        - 0.5 * abs_std_residuals_v
    )

    # Total log-likelihood: sum of individual contributions
    # Ref: gedloglik.m:67-68 — LL computed as T*(...) - 0.5*sum(...) - 0.5*sum(...)
    # Equivalent to numpy.sum(lls) by algebraic identity
    LL = numpy.sum(lls)

    return LL, lls
