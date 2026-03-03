"""
Inverse CDF (quantile function) of Hansen's (1994) Skewed Student's t distribution.

Migrated from distributions/skewtinv.m (MFE Toolbox Version 4.0).
Maps probabilities in [0, 1] to quantiles of the Skewed-t distribution
parameterized by degrees of freedom V and asymmetry parameter LAMBDA.

The piecewise inversion splits at the threshold p = (1 - lambda) / 2, using the
standard Student's t quantile function (scipy.stats.t.ppf) on each piece with
Hansen's (1994) location-scale reparameterization.

Author: Andrew Patton (original MATLAB)
        a.patton@lse.ac.uk
Modifications Copyright: Kevin Sheppard
        kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005

MATLAB-to-Python migration notes:
- tinv(p, v) → scipy.stats.t.ppf(p, v)
- gamma(x) → scipy.special.gamma(x)
- find(condition) → boolean indexing directly
- repmat(NaN, sizeOut) → np.full(size_out, np.nan)
- lambda → lambda_ (Python reserved keyword)
- MATLAB 1-based indexing → Python 0-based (boolean masks used, no offset needed)
- MATLAB size(p) returns [1, 1] for scalars → Python np.asarray(scalar).shape is ()
"""

import numpy as np
from scipy.special import gamma
from scipy.stats import t as t_dist

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['skewtinv']


def skewtinv(p, v, lambda_):
    """
    Inverse CDF (quantile function) of Hansen's (1994) Skewed Student's t distribution.

    Maps probabilities in [0, 1] to quantiles of the Skewed-t distribution with
    ``v`` degrees of freedom and asymmetry parameter ``lambda_``.

    Parameters
    ----------
    p : array_like
        Probabilities to be inverted, values in [0, 1]. Can be a scalar, 1-D
        array, or multi-dimensional array.
    v : array_like
        Degrees of freedom parameter(s). Must satisfy ``v > 2`` for valid
        output; elements with ``v <= 2`` produce NaN. Scalar or array of the
        same shape as ``p``.
    lambda_ : array_like
        Skewness (asymmetry) parameter(s). Must satisfy
        ``-0.99 < lambda_ < 0.99`` for valid output; elements outside this
        range produce NaN. Scalar or array of the same shape as ``p``.

    Returns
    -------
    x : numpy.ndarray
        Skewed-t distributed quantiles corresponding to the input probabilities.
        Shape matches the broadcast-compatible shape of ``p``, ``v``, and
        ``lambda_``. Elements are NaN where ``v <= 2`` or
        ``|lambda_| >= 0.99``.

    Raises
    ------
    ValueError
        If the sizes of ``v`` and ``lambda_`` are not compatible with the shape
        of ``p``.

    Notes
    -----
    This function implements the inverse CDF of the skewed Student's t
    distribution as defined in Hansen (1994) [1]_. The distribution is
    parameterized by degrees of freedom ``v`` and asymmetry parameter
    ``lambda_``.

    The Hansen (1994) constants are computed as:

    .. math::

        c = \\frac{\\Gamma((v+1)/2)}{\\sqrt{\\pi(v-2)}\\,\\Gamma(v/2)}

        a = 4\\lambda\\,c\\,\\frac{v-2}{v-1}

        b = \\sqrt{1 + 3\\lambda^2 - a^2}

    The piecewise inversion formula splits at the threshold
    ``p = (1 - lambda_) / 2``:

    - For ``p < (1 - lambda_) / 2`` (left tail):

      .. math::

          x = \\frac{1 - \\lambda}{b}\\sqrt{\\frac{v-2}{v}}\\,
              t^{-1}\\!\\left(\\frac{p}{1-\\lambda},\\,v\\right)
              - \\frac{a}{b}

    - For ``p >= (1 - lambda_) / 2`` (right tail):

      .. math::

          x = \\frac{1 + \\lambda}{b}\\sqrt{\\frac{v-2}{v}}\\,
              t^{-1}\\!\\left(\\frac{1}{2} + \\frac{1}{1+\\lambda}
              \\left(p - \\frac{1-\\lambda}{2}\\right),\\,v\\right)
              - \\frac{a}{b}

    where :math:`t^{-1}(\\cdot, v)` is the standard Student's t quantile
    function with ``v`` degrees of freedom.

    References
    ----------
    .. [1] Hansen, B.E. (1994), "Autoregressive Conditional Density Estimation,"
           International Economic Review, 35, 705-730.

    See Also
    --------
    skewtcdf : CDF of Hansen's skewed t.
    skewtpdf : PDF of Hansen's skewed t.
    skewtrnd : Random variates from Hansen's skewed t.
    skewtloglik : Log-likelihood of Hansen's skewed t.

    Examples
    --------
    Scalar inputs (symmetric case, lambda_=0 → median at 0):

    >>> import numpy as np
    >>> from mfe_toolbox.distributions.skewtinv import skewtinv
    >>> x = skewtinv(0.5, 5.0, 0.0)
    >>> np.abs(x.flat[0]) < 1e-10
    True

    Array of probabilities:

    >>> x = skewtinv(np.array([0.1, 0.5, 0.9]), 5.0, 0.0)
    >>> x.shape
    (3,)

    NaN for invalid degrees of freedom:

    >>> x = skewtinv(0.5, 2.0, 0.0)
    >>> np.isnan(x.flat[0])
    True
    """
    # Ref: skewtinv.m:31-33 — nargin~=3 check enforced by Python function signature

    # Convert p to a numpy float64 array for shape inspection
    p_arr = np.asarray(p, dtype=np.float64)

    # Ref: skewtinv.m:35 — [err, errtext, sizeOut, v, lambda] = iscompatible(2,v,lambda,size(p))
    # MATLAB size(scalar) returns [1, 1]; Python np.asarray(scalar).shape returns ().
    # We pass the shape as a single argument (vector) to iscompatible.
    p_shape = p_arr.shape if p_arr.ndim > 0 else (1, 1)

    result = iscompatible(2, v, lambda_, p_shape)
    err = result[0]
    errtext = result[1]
    size_out = result[2]
    # Ref: skewtinv.m:35 — v and lambda are broadcast to sizeOut by iscompatible
    # Use np.array to ensure writable arrays (we will mutate them with NaN below)
    v = np.array(result[3], dtype=np.float64)
    lambda_ = np.array(result[4], dtype=np.float64)

    # Ref: skewtinv.m:37-39 — if err, error(errtext)
    if err:
        raise ValueError(errtext)

    # Ref: skewtinv.m:41 — v(v<=2) = NaN
    v[v <= 2] = np.nan
    # Ref: skewtinv.m:42 — lambda(lambda<-.99 | lambda>.99) = NaN
    lambda_[(lambda_ < -0.99) | (lambda_ > 0.99)] = np.nan

    # Suppress expected floating-point warnings from NaN propagation through
    # gamma/sqrt when v or lambda_ contain NaN (invalid parameters).
    with np.errstate(invalid='ignore', divide='ignore'):
        # Ref: skewtinv.m:44 — c = gamma((v+1)/2) ./ (sqrt(pi*(v-2)) .* gamma(v/2))
        c = gamma((v + 1) / 2) / (np.sqrt(np.pi * (v - 2)) * gamma(v / 2))

        # Ref: skewtinv.m:45 — a = 4*lambda.*c.*((v-2)./(v-1))
        a = 4 * lambda_ * c * ((v - 2) / (v - 1))

        # Ref: skewtinv.m:46 — b = sqrt(1 + 3*lambda.^2 - a.^2)
        b = np.sqrt(1 + 3 * lambda_ ** 2 - a ** 2)

    # Broadcast p to match the output size for element-wise comparison
    # Ref: skewtinv.m:48-49 — find() operates on arrays of matching shape
    p_broad = np.broadcast_to(p_arr, size_out).copy()

    # Ref: skewtinv.m:48 — f1 = find(p < ((1-lambda)/2))
    f1 = p_broad < ((1 - lambda_) / 2)
    # Ref: skewtinv.m:49 — f2 = find(p >= ((1-lambda)/2))
    f2 = p_broad >= ((1 - lambda_) / 2)

    # Ref: skewtinv.m:53 — x = repmat(NaN, sizeOut)
    x = np.full(size_out, np.nan)

    # Ref: skewtinv.m:51 — Left piece inversion (lower tail)
    # inv1 = (1-lambda(f1))./b(f1).*sqrt((v(f1)-2)./v(f1))
    #         .* tinv(p(f1)./(1-lambda(f1)), v(f1)) - a(f1)./b(f1)
    if np.any(f1):
        lam_f1 = lambda_[f1]
        v_f1 = v[f1]
        b_f1 = b[f1]
        a_f1 = a[f1]
        p_f1 = p_broad[f1]

        # Ref: skewtinv.m:51 — tinv(p(f1)./(1-lambda(f1)), v(f1))
        # Replaced by scipy.stats.t.ppf
        inv1 = ((1 - lam_f1) / b_f1
                * np.sqrt((v_f1 - 2) / v_f1)
                * t_dist.ppf(p_f1 / (1 - lam_f1), v_f1)
                - a_f1 / b_f1)
        # Ref: skewtinv.m:54 — x(f1) = inv1
        x[f1] = inv1

    # Ref: skewtinv.m:52 — Right piece inversion (upper tail)
    # inv2 = (1+lambda(f2))./b(f2).*sqrt((v(f2)-2)./v(f2))
    #         .* tinv(0.5 + 1./(1+lambda(f2)).*(p(f2)-(1-lambda(f2))./2), v(f2))
    #         - a(f2)./b(f2)
    if np.any(f2):
        lam_f2 = lambda_[f2]
        v_f2 = v[f2]
        b_f2 = b[f2]
        a_f2 = a[f2]
        p_f2 = p_broad[f2]

        # Ref: skewtinv.m:52 — tinv argument:
        # 0.5 + 1./(1+lambda(f2)) .* (p(f2) - (1-lambda(f2))./2)
        tinv_arg = 0.5 + 1.0 / (1 + lam_f2) * (p_f2 - (1 - lam_f2) / 2)

        # Ref: skewtinv.m:52 — tinv(..., v(f2))
        # Replaced by scipy.stats.t.ppf
        inv2 = ((1 + lam_f2) / b_f2
                * np.sqrt((v_f2 - 2) / v_f2)
                * t_dist.ppf(tinv_arg, v_f2)
                - a_f2 / b_f2)
        # Ref: skewtinv.m:55 — x(f2) = inv2
        x[f2] = inv2

    return x
