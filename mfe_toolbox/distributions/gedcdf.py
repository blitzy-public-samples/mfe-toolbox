"""
Cumulative Distribution Function (CDF) of the Generalized Error Distribution (GED).

Migrated from distributions/gedcdf.m (MFE Toolbox Version 4.0).

A scalar GED random variable with variance normalized to 1 has probability
density given by::

    f(x, v) = [v / (lda * 2^(1+1/v) * gamma(1/v))] * exp(-0.5 * |x/lda|^v)
    lda = [2^(-2/v) * gamma(1/v) / gamma(3/v)]^0.5

where v > 1 is the shape parameter.

The CDF is computed via the relationship between the GED and the incomplete
gamma function::

    For x >= 0: P(x) = 0.5 * (1 + gamma_cdf((|x * scalex|)^v, 1/v))
    For x <  0: P(x) = 0.5 * (1 - gamma_cdf((|x * scalex|)^v, 1/v))

    where scalex = sqrt(gamma(3/v) / gamma(1/v))

Uses ``scipy.stats.gamma.cdf`` (replaces MATLAB ``gamcdf``) and
``scipy.special.gamma`` (replaces MATLAB ``gamma``).

Authors:
    Ivana Komunjer (original MATLAB, komunjer@hss.caltech.edu)
    Modifications: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
    Revision: 3    Date: 8/1/2005

References
----------
.. [1] Tadikamalla (1980), J.Am.Stat.Assoc. (75)
.. [2] Nelson (1991), Econometrica

See Also
--------
gedpdf : GED probability density function
gedinv : GED inverse CDF (quantile function)
gedrnd : GED random variate generation
gedloglik : GED log-likelihood

MATLAB-to-Python migration notes:
- gamcdf(x, a) → scipy.stats.gamma.cdf(x, a) with scale=1 (default)
- gamma(x)    → scipy.special.gamma(x) (gamma function, not distribution)
- size(x)     → numpy.array(x.shape) for iscompatible call
- find(cond)  → boolean indexing
- zeros(size(x)) → numpy.zeros(x.shape, dtype=numpy.float64)
- nargin check → Python function signature enforcing exactly 2 args
- p(v<1) = NaN → numpy.where(v<1, nan, v) + p[isnan(v)] = nan
- 1-based indexing → 0-based boolean indexing
"""

import numpy
import scipy.special
import scipy.stats

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['gedcdf']


def gedcdf(x, v):
    """
    Compute the CDF of the Generalized Error Distribution (GED).

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the GED CDF. Can be scalar, 1-D, or
        multi-dimensional array.
    v : array_like
        Shape (degree of freedom) parameter of the GED. Must satisfy v > 1
        for valid results. Either scalar or same shape as ``x``.

    Returns
    -------
    p : numpy.ndarray
        CDF values evaluated at ``x``. Same shape as ``x`` (after possible
        scalar-to-1D promotion). Returns ``NaN`` at positions where ``v < 1``.

    Raises
    ------
    ValueError
        If ``v`` is not compatible with the shape of ``x`` (i.e., ``v`` is
        neither scalar nor the same shape as ``x``).

    Notes
    -----
    The GED shape parameter ``v`` controls the tail thickness:

    - ``v = 2`` recovers the standard normal distribution.
    - ``v < 2`` produces heavier tails than normal.
    - ``v > 2`` produces lighter tails than normal.
    - ``v = 1`` gives the Laplace (double exponential) distribution.
    - ``v < 1`` is not a valid GED and returns ``NaN``.

    The CDF at ``x = 0`` is always 0.5 (symmetric distribution centered at 0).

    The implementation mirrors the MATLAB ``gedcdf.m`` algorithm:

    1. Validate parameter compatibility via :func:`iscompatible`.
    2. Compute the GED standardization scale:
       ``scalex = sqrt(gamma(3/v) / gamma(1/v))``
    3. For ``x >= 0``: ``p = 0.5 * (1 + gamma_cdf((|x*scalex|)^v, 1/v))``
    4. For ``x < 0``:  ``p = 0.5 * (1 - gamma_cdf((|x*scalex|)^v, 1/v))``
    5. Clamp ``p`` to ``[0, 1]`` to prevent round-off artifacts.

    Ref: gedcdf.m — MFE Toolbox v4.0 by Kevin Sheppard.

    Examples
    --------
    Scalar evaluation:

    >>> import numpy as np
    >>> p = gedcdf(0.0, 2.0)
    >>> np.isclose(p, 0.5, atol=1e-10)
    True

    Array evaluation with v=2 (normal distribution):

    >>> x = np.array([-1.0, 0.0, 1.0])
    >>> p = gedcdf(x, 2.0)
    >>> np.allclose(p, [0.158655, 0.5, 0.841345], atol=1e-4)
    True

    Invalid shape parameter returns NaN:

    >>> p = gedcdf(0.5, 0.5)
    >>> np.isnan(p)
    True
    """
    # Ref: gedcdf.m:36-38 — nargin~=2 check replaced by Python function
    # signature enforcing exactly 2 positional arguments.

    # Convert x to numpy float64 array
    # Ref: gedcdf.m expects matrix input; numpy.asarray handles conversion
    x = numpy.asarray(x, dtype=numpy.float64)

    # Handle scalar input: MATLAB treats all values as at least 2-D matrices,
    # so a scalar x has size(x) = [1,1]. In Python, a 0-D array has shape ()
    # which causes issues with iscompatible's size parsing. We promote 0-D to
    # 1-D for consistent processing.
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x.reshape(1)  # Promote 0-D to 1-D array of length 1

    # Ref: gedcdf.m:40 — [err, errtext, sizeOut, v] = iscompatible(1,v,size(x))
    # MATLAB size(x) returns a row vector of dimension sizes.
    # Python numpy.array(x.shape) produces the equivalent 1-D integer array.
    # iscompatible with narg=1 treats v as the single parameter and the
    # remaining argument as the requested output size specification.
    err, errtext, size_out, v = iscompatible(1, v, numpy.array(x.shape))

    # Ref: gedcdf.m:42-44 — if err, error(errtext), end
    if err:
        raise ValueError(errtext)

    # Ensure v is a properly shaped float64 array after broadcasting
    # Ref: iscompatible broadcasts scalar v to match x.shape
    v = numpy.asarray(v, dtype=numpy.float64)

    # Ref: gedcdf.m:47 — p = zeros(size(x));
    # Initialize output CDF array to zero.
    p = numpy.zeros(x.shape, dtype=numpy.float64)

    # Ref: gedcdf.m:51 — v(v<1) = NaN; Return NaN if v is outside its limits.
    # In MATLAB, this sets elements of v to NaN where v < 1, and separately
    # p(v<1) = NaN sets output to NaN. In Python, we replace invalid v values
    # with NaN and track via numpy.isnan for subsequent masking.
    invalid_v = v < 1
    v = numpy.where(invalid_v, numpy.nan, v)
    p[invalid_v] = numpy.nan

    # Ref: gedcdf.m:48 — scalex = (gamma(3./v)./gamma(1./v)).^0.5
    # Compute the GED standardization scale factor.
    # scipy.special.gamma replaces MATLAB's gamma() function.
    # Where v is NaN, scalex will also be NaN (no error raised).
    scalex = numpy.sqrt(
        scipy.special.gamma(3.0 / v) / scipy.special.gamma(1.0 / v)
    )

    # ---- Upper branch: x >= 0 and v is valid ----
    # Ref: gedcdf.m:53 — ku = find(x >= 0 & ~(v < 1));
    # Python: boolean indexing replaces MATLAB find() pattern.
    # After replacing v<1 with NaN, we use ~isnan(v) to identify valid entries.
    ku = (x >= 0) & (~numpy.isnan(v))
    if numpy.any(ku):
        vku = v[ku]
        xku = x[ku]
        scalexku = scalex[ku]
        # Ref: gedcdf.m:58 — p(ku) = 0.5*(1 + gamcdf((abs(xku.*scalexku)).^vku, 1./vku))
        # MATLAB gamcdf(x, a) → scipy.stats.gamma.cdf(x, a) with scale=1 (default).
        # The gamma CDF shape parameter 'a' here is 1/v.
        p[ku] = 0.5 * (1.0 + scipy.stats.gamma.cdf(
            (numpy.abs(xku * scalexku)) ** vku, 1.0 / vku
        ))

    # ---- Lower branch: x < 0 and v is valid ----
    # Ref: gedcdf.m:61 — kl = find(x < 0 & ~(v < 1));
    kl = (x < 0) & (~numpy.isnan(v))
    if numpy.any(kl):
        vkl = v[kl]
        xkl = x[kl]
        scalexkl = scalex[kl]
        # Ref: gedcdf.m:66 — p(kl) = 0.5*(1 - gamcdf((abs(xkl.*scalexkl)).^vkl, 1./vkl))
        p[kl] = 0.5 * (1.0 - scipy.stats.gamma.cdf(
            (numpy.abs(xkl * scalexkl)) ** vkl, 1.0 / vkl
        ))

    # Ref: gedcdf.m:70-71 — Make sure that round-off errors never make P
    # less than 0 or greater than 1.
    # NaN values are unaffected because NaN comparisons return False.
    p[p < 0] = 0.0
    p[p > 1] = 1.0

    return p
