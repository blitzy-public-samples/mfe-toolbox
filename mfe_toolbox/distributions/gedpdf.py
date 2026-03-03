"""
Probability Density Function (PDF) for the Generalized Error Distribution (GED).

Migrated from distributions/gedpdf.m (MFE Toolbox Version 4.0).
Computes the GED density at specified points for given shape parameter values.
The GED with shape parameter V=2 reduces to the standard normal distribution.

A scalar GED random variable with variance normalized to 1 has probability
density given by:

    f(x, V) = V * exp(-0.5 * |x / lda|^V) / (lda * 2^(1 + 1/V) * gamma(1/V))
    lda = [2^(-2/V) * gamma(1/V) / gamma(3/V)]^0.5

where V >= 1 is the shape parameter and lda is the scale factor ensuring
unit variance.

Author: Ivana Komunjer (original MATLAB), Kevin Sheppard (modifications)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 8/1/2005

MATLAB-to-Python migration notes:
- gamma(x) → scipy.special.gamma(x)
- 2.^(-2./v) → 2.0 ** (-2.0 / v) (element-wise power in numpy)
- find(v >= 1) → boolean indexing (v >= 1)
- zeros(size(x)) → numpy.zeros(x.shape)
- error('msg') → raise ValueError('msg')
- nargin check → Python enforces via required positional args

References
----------
[1] Tadikamalla (1980), "On Simulating Non-Normal Distributions",
    J.Am.Stat.Assoc. (75)
[2] Nelson (1991), "Conditional Heteroskedasticity in Asset Returns:
    A New Approach", Econometrica

See Also
--------
gedcdf : GED cumulative distribution function
gedinv : GED inverse (quantile) function
gedrnd : GED random variate generation
gedloglik : GED log-likelihood
"""

import numpy as np
from scipy.special import gamma

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['gedpdf']


def gedpdf(x, v):
    """
    Compute the PDF of the Generalized Error Distribution (GED).

    Evaluates the GED probability density function at each point in ``x``
    for the specified shape parameter ``v``. The GED is parameterized so
    that its variance is normalized to 1.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the GED probability density. Can be
        a scalar, 1-D array, or N-D array of float64 values.
    v : array_like
        Shape (tail-thickness) parameter of the GED. Must be >= 1 for
        valid output. Can be a scalar (broadcast to match ``x``) or an
        array of the same shape as ``x``.

    Returns
    -------
    y : numpy.ndarray
        GED probability density values evaluated at ``x``. Has the same
        shape as ``x``. Elements corresponding to ``v < 1`` are set to
        ``numpy.nan``.

    Raises
    ------
    ValueError
        If ``v`` cannot be broadcast to the shape of ``x`` (incompatible
        sizes), as determined by :func:`iscompatible`.

    Notes
    -----
    The GED PDF with unit variance is defined as:

    .. math::

        f(x, V) = \\frac{V}{\\lambda \\cdot 2^{1+1/V} \\cdot \\Gamma(1/V)}
                   \\exp\\left(-\\frac{1}{2} \\left|\\frac{x}{\\lambda}\\right|^V\\right)

    where the scale factor :math:`\\lambda` is:

    .. math::

        \\lambda = \\left[\\frac{2^{-2/V} \\cdot \\Gamma(1/V)}{\\Gamma(3/V)}\\right]^{1/2}

    Special cases:
    - V = 1: Laplace (double-exponential) distribution
    - V = 2: Standard normal distribution
    - V → ∞: Uniform distribution on [-√3, √3]

    The function faithfully reproduces the MATLAB ``gedpdf.m`` numerical
    behavior to ±1e-6 tolerance.

    References
    ----------
    .. [1] Tadikamalla (1980), J.Am.Stat.Assoc. (75)
    .. [2] Nelson (1991), Econometrica

    Examples
    --------
    Evaluate GED PDF at a single point with shape parameter v=2 (normal):

    >>> import numpy as np
    >>> y = gedpdf(0.0, 2.0)
    >>> np.allclose(y, 1.0 / np.sqrt(2.0 * np.pi), atol=1e-10)
    True

    Evaluate at multiple points:

    >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> y = gedpdf(x, 2.0)
    >>> y.shape
    (5,)

    Invalid shape parameter returns NaN:

    >>> y = gedpdf(0.0, 0.5)
    >>> np.isnan(y[0])
    True
    """
    # Convert x to numpy float64 array for consistent numerical operations
    # Ref: gedpdf.m — MATLAB implicitly stores x as double
    x = np.asarray(x, dtype=np.float64)

    # Handle 0-d (scalar) input: MATLAB's size(scalar) returns [1,1],
    # but numpy scalar has shape (). Reshape to (1,) for iscompatible
    # compatibility, which expects at least a 1-d size specification.
    # Ref: MATLAB treats all scalars as 1x1 matrices
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x.reshape(1)

    # Validate and broadcast shape parameter v to match x dimensions
    # Ref: gedpdf.m:39 — [err, errtext, sizeOut, v] = iscompatible(1,v,size(x))
    # narg=1 means one parameter (v) to validate against size(x)
    err, errtext, size_out, v = iscompatible(1, v, np.array(x.shape))

    # Raise ValueError on incompatible parameter/size combinations
    # Ref: gedpdf.m:41-43 — if err, error(errtext), end
    if err:
        raise ValueError(errtext)

    # Initialize output array to zeros
    # Ref: gedpdf.m:46 — y = zeros(size(x))
    y = np.zeros(x.shape, dtype=np.float64)

    # Set output to NaN where shape parameter is below the valid limit (v >= 1)
    # Ref: gedpdf.m:49 — y(v < 1) = NaN
    # Note: NaN comparisons return False in both MATLAB and numpy, so
    # NaN values in v do NOT trigger this condition (matching MATLAB behavior)
    invalid_mask = v < 1
    if np.any(invalid_mask):
        y[invalid_mask] = np.nan

    # Boolean mask for valid shape parameter values (v >= 1)
    # Ref: gedpdf.m:51 — k = find(v >= 1)
    # Python uses boolean indexing instead of MATLAB's find() index list
    k = v >= 1

    # Compute GED PDF only for valid parameter values
    # Ref: gedpdf.m:53 — if any(k)
    if np.any(k):
        # Extract valid parameter and data subsets
        # Ref: gedpdf.m:52 — vk = v(k); xk = x(k)
        vk = v[k]
        xk = x[k]

        # Compute the GED lambda scale factor ensuring unit variance
        # lda = [2^(-2/V) * gamma(1/V) / gamma(3/V)]^(1/2)
        # Ref: gedpdf.m:54 — ldak = ((2.^(-2./vk)).*(gamma(1./vk))./(gamma(3./vk))).^(0.5)
        # Translation: .^ → **, .* → *, ./ → /  (numpy element-wise by default)
        ldak = np.sqrt(
            (2.0 ** (-2.0 / vk)) * gamma(1.0 / vk) / gamma(3.0 / vk)
        )

        # Compute the GED PDF using the standard formula
        # f(x,V) = V * exp(-0.5 * |x/lda|^V) / (lda * gamma(1/V) * 2^(1+1/V))
        # Ref: gedpdf.m:55 — y(k) = vk.*exp(-0.5*((abs(xk./ldak)).^vk))
        #                            ./(ldak.*gamma(1./vk).*(2.^(1+1./vk)))
        numerator = vk * np.exp(-0.5 * (np.abs(xk / ldak)) ** vk)
        denominator = ldak * gamma(1.0 / vk) * (2.0 ** (1.0 + 1.0 / vk))
        y[k] = numerator / denominator

    return y
