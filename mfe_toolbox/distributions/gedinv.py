"""
Inverse CDF (quantile function) of the Generalized Error Distribution (GED).

Migrated from distributions/gedinv.m (MFE Toolbox Version 4.0).
Computes the quantile function for the GED using Tadikamalla's (1980) method,
which exploits the relationship between the GED and the Gamma distribution.
A scalar GED random variable with variance normalized to 1 has probability
density:

    f(x, v) = [v / (lda * 2^(1+1/v) * Gamma(1/v))] * exp(-0.5 * |x/lda|^v)
    lda = [2^(-2/v) * Gamma(1/v) / Gamma(3/v)]^0.5

where v >= 1 is the shape parameter.  The inverse CDF is computed via the
relationship Y = |X|^v ~ Gamma(1/v), enabling use of the Gamma inverse CDF
(scipy.stats.gamma.ppf) as the core computational kernel.

Author: Ivana Komunjer (original MATLAB, komunjer@hss.caltech.edu)
Modifications: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005
MATLAB-to-Python migration: Blitzy Platform

References
----------
[1] Tadikamalla (1980), "Random Sampling from the Exponential Power
    Distribution", Journal of the American Statistical Association,
    75(371), 683-686.
[2] Nelson (1991), "Conditional Heteroskedasticity in Asset Returns:
    A New Approach", Econometrica, 59(2), 347-370.
"""

import numpy
from scipy.special import gamma as gamma_func
from scipy.stats import gamma as gamma_dist

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['gedinv']


def gedinv(p, v):
    """
    Inverse CDF (quantile function) of the Generalized Error Distribution.

    Maps probabilities in [0, 1] to quantiles of a GED with shape parameter
    ``v`` and unit variance normalization.  Uses Tadikamalla's (1980) method
    based on the Gamma distribution inverse CDF.

    Parameters
    ----------
    p : array_like
        Probability values to be inverted.  Must lie in [0, 1].
    v : array_like
        Shape parameter(s) of the GED.  Must be >= 1 for valid output.
        Can be a scalar (broadcast to match ``p``) or an array with the
        same shape as ``p``.

    Returns
    -------
    x : numpy.ndarray
        GED quantiles corresponding to the input probabilities.

        * ``numpy.nan`` is returned where ``v < 1`` or ``p`` is outside
          [0, 1].
        * ``0.0`` is returned where ``p == 0`` (with valid ``v``).
        * ``numpy.inf`` is returned where ``p == 1`` (with valid ``v``).

    Raises
    ------
    ValueError
        If ``p`` and ``v`` have incompatible shapes that cannot be broadcast
        to a common output size.

    Notes
    -----
    The algorithm decomposes the GED quantile computation into two branches
    (below and above the median) using the identity that for a GED with
    shape ``v``, the variable Y = |X|^v follows a Gamma(1/v) distribution.
    The quantile is then:

    * Lower branch (``p < 0.5``):
      ``x = -(gaminv(1 - 2p, 1/v))^(1/v) / scalex``
    * Upper branch (``p >= 0.5``):
      ``x = (gaminv(2p - 1, 1/v))^(1/v) / scalex``

    where ``scalex = sqrt(Gamma(3/v) / Gamma(1/v))`` is the standardization
    factor ensuring unit variance, and ``gaminv`` is the Gamma distribution
    inverse CDF.

    The MATLAB ``gaminv(p, a)`` function maps directly to
    ``scipy.stats.gamma.ppf(p, a)`` with the default scale parameter of 1.

    See Also
    --------
    gedcdf : CDF of the GED.
    gedpdf : PDF of the GED.
    gedrnd : Random variates from the GED.
    gedloglik : Log-likelihood of the GED.
    """
    # ----------------------------------------------------------------
    # Step 1: Convert input probability to a numpy float64 array
    # Ref: gedinv.m — MATLAB implicitly treats inputs as double matrices
    # ----------------------------------------------------------------
    p = numpy.asarray(p, dtype=numpy.float64)

    # MATLAB treats scalars as 1x1 matrices; Python has 0-D arrays.
    # Ensure at least 1-D so that shape can be passed to iscompatible.
    # Ref: MATLAB size(scalar) returns [1, 1]; numpy scalar.shape is ()
    scalar_input = (p.ndim == 0)
    if scalar_input:
        p = p.reshape(1)

    # ----------------------------------------------------------------
    # Step 2: Validate and broadcast v to match p's shape
    # Ref: gedinv.m:40 — [err, errtext, sizeOut, v] = iscompatible(1, v, size(p))
    # iscompatible(narg=1, v, numpy.array(p.shape)) checks that v is
    # compatible with p's shape and broadcasts v to size_out.
    # ----------------------------------------------------------------
    err, errtext, size_out, v = iscompatible(1, v, numpy.array(p.shape))

    # Ref: gedinv.m:42-44 — if err, error(errtext), end
    if err:
        raise ValueError(errtext)

    # Ensure v is a writable float64 array for subsequent operations
    v = numpy.asarray(v, dtype=numpy.float64).copy()

    # ----------------------------------------------------------------
    # Step 3: Mask invalid shape parameter values
    # Ref: gedinv.m:50 — v < 1 included in the invalid-entry mask
    # Using numpy.where to set v to NaN where v < 1; this causes NaN
    # propagation through scalex and the final division, ensuring
    # output is NaN for entries with invalid shape parameter.
    # ----------------------------------------------------------------
    v = numpy.where(v < 1, numpy.nan, v)

    # ----------------------------------------------------------------
    # Step 4: Initialize output array to zeros
    # Ref: gedinv.m:47 — x = zeros(size(p))
    # MATLAB initializes x to zeros; entries for p==0 with valid v
    # remain at zero after all branch assignments.
    # ----------------------------------------------------------------
    x = numpy.zeros(p.shape, dtype=numpy.float64)

    # ----------------------------------------------------------------
    # Step 5: Compute the GED standardization scale factor
    # scalex = sqrt(Gamma(3/v) / Gamma(1/v))
    # Ref: gedinv.m:48 — scalex = (gamma(3./v)./gamma(1./v)).^0.5
    #
    # Initialize scalex to NaN to avoid computing gamma for NaN v values.
    # Only compute for entries where v is valid (not NaN after masking).
    # ----------------------------------------------------------------
    scalex = numpy.full(v.shape, numpy.nan, dtype=numpy.float64)
    valid_v_mask = ~numpy.isnan(v)

    if numpy.any(valid_v_mask):
        v_valid = v[valid_v_mask]
        # Ref: gedinv.m:48 — gamma(3./v)./gamma(1./v) uses MATLAB's built-in
        # gamma(), replaced here by scipy.special.gamma
        scalex[valid_v_mask] = numpy.sqrt(
            gamma_func(3.0 / v_valid) / gamma_func(1.0 / v_valid)
        )

    # ----------------------------------------------------------------
    # Step 6: Set NaN for out-of-range probabilities or invalid v
    # Ref: gedinv.m:50-54 — k = find(p<0 | p>1 | v < 1); x(k) = NaN
    # Since v was already set to NaN for v<1, numpy.isnan(v) captures
    # the v<1 condition.
    # ----------------------------------------------------------------
    invalid = numpy.isnan(v) | (p < 0) | (p > 1)
    x[invalid] = numpy.nan

    # ----------------------------------------------------------------
    # Step 7: Boundary case — p == 0 with valid v → x = 0
    # Ref: gedinv.m:57-60 — k0 = find(p == 0 & v >= 1); x(k0) = zeros(...)
    # x is already initialized to zero, so no action needed for p==0.
    # After final division by scalex: 0.0 / scalex = 0.0, preserving
    # the boundary value.
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Step 8: Boundary case — p == 1 with valid v → x = +Inf
    # Ref: gedinv.m:62-66 — k1 = find(p == 1 & v >= 1); x(k1) = Inf
    # After final division by scalex: Inf / scalex = Inf.
    # ----------------------------------------------------------------
    k1 = (p == 1) & valid_v_mask
    x[k1] = numpy.inf

    # ----------------------------------------------------------------
    # Step 9: Lower branch — Tadikamalla's method for p < 0.5
    # Uses the fact that Y = |X|^v has a Gamma(1/v) distribution.
    # For p < 0.5, the quantile is negative (left tail):
    #   x = -(gaminv(1 - 2p, 1/v))^(1/v)
    # Ref: gedinv.m:70-77
    # ----------------------------------------------------------------
    kl = (p > 0) & (p < 0.5) & valid_v_mask
    if numpy.any(kl):
        pkl = p[kl]
        vkl = v[kl]
        # Ref: gedinv.m:74 — xkl = gaminv(1-2*pkl, 1./vkl)
        # gaminv(p, a) → scipy.stats.gamma.ppf(p, a) with scale=1
        xkl = gamma_dist.ppf(1.0 - 2.0 * pkl, 1.0 / vkl)
        # Ref: gedinv.m:75 — xkl = (-1)*xkl.^(1./vkl)
        xkl = (-1.0) * xkl ** (1.0 / vkl)
        x[kl] = xkl

    # ----------------------------------------------------------------
    # Step 10: Upper branch — Tadikamalla's method for p >= 0.5
    # For p >= 0.5, the quantile is non-negative (right tail):
    #   x = (gaminv(2p - 1, 1/v))^(1/v)
    # Ref: gedinv.m:79-86
    # ----------------------------------------------------------------
    ku = (p >= 0.5) & (p < 1) & valid_v_mask
    if numpy.any(ku):
        pku = p[ku]
        vku = v[ku]
        # Ref: gedinv.m:83 — xku = gaminv(2*pku-1, 1./vku)
        xku = gamma_dist.ppf(2.0 * pku - 1.0, 1.0 / vku)
        # Ref: gedinv.m:84 — xku = xku.^(1./vku)
        xku = xku ** (1.0 / vku)
        x[ku] = xku

    # ----------------------------------------------------------------
    # Step 11: Standardize the quantiles by dividing by the scale factor
    # Ref: gedinv.m:89 — x = x./scalex
    # NaN entries in scalex (from invalid v) propagate NaN through the
    # division.  Boundary values are preserved: 0/finite = 0,
    # Inf/finite = Inf.
    # ----------------------------------------------------------------
    with numpy.errstate(invalid='ignore'):
        x = x / scalex

    return x
