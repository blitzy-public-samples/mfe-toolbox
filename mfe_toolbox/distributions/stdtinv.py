"""
Inverse CDF (quantile function) of the Standardized Student's t distribution.

Migrated from distributions/stdtinv.m (MFE Toolbox Version 4.0).
Computes quantiles of the standardized Student's t distribution which has
unit variance for degrees of freedom v > 2.  The standardized t distribution
is the Student's t distribution rescaled by dividing by sqrt(v / (v - 2))
so that the resulting distribution has variance equal to 1.

The implementation uses scipy.stats.t.ppf for the raw t inverse CDF and then
applies the standardization factor.  For v <= 2, the variance of the standard
t is undefined (infinite or non-existent), so the result is NaN.

Author: Kevin Sheppard (original MATLAB)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005

MATLAB-to-Python migration notes:
- tinv(p, v) → scipy.stats.t.ppf(p, v) — direct 1:1 mapping
- nargin check replaced by Python function signature enforcement
- MATLAB error() → Python raise ValueError()
- MATLAB size(p) → numpy.array(p.shape) for iscompatible call
- MATLAB 1-indexed arrays → Python 0-indexed (no index changes needed here)
- MATLAB implicit scalar expansion → explicit numpy broadcasting via iscompatible
- Division by zero / sqrt of negative handled via numpy.errstate context manager
"""

import numpy
import scipy.stats

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['stdtinv']


def stdtinv(p, v):
    """
    Inverse CDF (quantile function) of the Standardized Student's t distribution.

    Maps probabilities in [0, 1] to quantiles of a standardized Student's t
    distribution with ``v`` degrees of freedom.  The standardized t has unit
    variance (for v > 2) and is obtained by dividing the ordinary t quantile
    by ``sqrt(v / (v - 2))``.

    Parameters
    ----------
    p : array_like
        Probability values in [0, 1] at which to evaluate the inverse CDF.
        Can be a scalar, 1-D array, or N-D array.
    v : array_like
        Degrees of freedom parameter.  Must be either a scalar or an array
        with the same shape as ``p``.  The standardized t distribution requires
        ``v > 2`` for the variance to exist; quantiles for ``v <= 2`` are
        returned as ``NaN``.

    Returns
    -------
    x : numpy.ndarray
        Quantiles of the standardized Student's t distribution corresponding
        to the input probabilities ``p``.  Has the same shape as ``p`` (or the
        common broadcast shape of ``p`` and ``v``).

    Raises
    ------
    ValueError
        If ``p`` and ``v`` have incompatible shapes that cannot be broadcast
        to a common size.

    Notes
    -----
    The standardized Student's t distribution with ``v`` degrees of freedom is
    defined as T / sqrt(v / (v - 2)) where T ~ t(v).  This rescaling yields
    unit variance.

    For ``v <= 2``, the variance of the t distribution is infinite (v == 2) or
    undefined (v < 2), so the standardization factor is meaningless and the
    function returns ``NaN``.

    The relationship ``stdtinv(stdtcdf(x, v), v) ≈ x`` (to numerical
    precision) holds for ``v > 2``.

    References
    ----------
    [1] Casella, G. and Berger, R. L. (1990). *Statistical Inference*.
        Wadsworth & Brooks/Cole.

    See Also
    --------
    stdtcdf : CDF of the standardized Student's t distribution.
    stdtpdf : PDF of the standardized Student's t distribution.
    stdtrnd : Random variates from the standardized Student's t distribution.
    stdtloglik : Log-likelihood of the standardized Student's t distribution.

    Examples
    --------
    Scalar inputs:

    >>> import numpy
    >>> x = stdtinv(0.5, 5)
    >>> numpy.testing.assert_allclose(x, 0.0, atol=1e-10)

    Array inputs:

    >>> p = numpy.array([0.025, 0.5, 0.975])
    >>> x = stdtinv(p, 10)
    >>> x.shape
    (3,)
    """
    # Convert p to numpy float64 array
    # Ref: stdtinv.m uses MATLAB's implicit type handling; Python needs explicit conversion
    p = numpy.asarray(p, dtype=numpy.float64)

    # Track whether input was a scalar (0-dim array) for output shape restoration
    # Ref: MATLAB always operates on at least 1x1 matrices; Python scalars have shape ()
    scalar_input = (p.ndim == 0)
    if scalar_input:
        # Reshape to 1-D so p.shape is (1,) instead of () for iscompatible
        # Ref: MATLAB size(scalar) returns [1, 1]; we ensure non-empty shape
        p = p.reshape(1)

    # Validate and broadcast v to match p's shape
    # Ref: stdtinv.m:33 — [err, errtext, sizeOut, v] = iscompatible(1, v, size(p))
    err, errtext, size_out, v = iscompatible(1, v, numpy.array(p.shape))

    # Ref: stdtinv.m:34-36 — if err, error(errtext), end
    if err:
        raise ValueError(errtext)

    # Ensure v is a float64 array for consistent arithmetic
    v = numpy.asarray(v, dtype=numpy.float64)

    # Compute the standard (non-standardized) t-distribution inverse CDF
    # Ref: stdtinv.m:38 — x = tinv(p, v)
    # scipy.stats.t.ppf(p, df) is the direct replacement for MATLAB's tinv(p, v)
    x = scipy.stats.t.ppf(p, v)

    # Compute the standard deviation of the t(v) distribution: sqrt(v / (v - 2))
    # This is the factor needed to standardize to unit variance.
    # For v <= 2, this involves division by zero (v == 2) or sqrt of a negative
    # number (v < 2), so we suppress warnings and let numpy produce inf/nan,
    # then explicitly set those cases to NaN below.
    # Ref: stdtinv.m:40 — stdev = sqrt(v ./ (v - 2))
    with numpy.errstate(divide='ignore', invalid='ignore'):
        stdev = numpy.sqrt(v / (v - 2.0))

    # For v <= 2, the variance of t(v) is undefined, so the standardized
    # quantile is meaningless.  Set stdev to NaN to propagate NaN into x.
    # Ref: stdtinv.m:41 — stdev(v <= 2) = NaN
    stdev = numpy.where(v <= 2, numpy.nan, stdev)

    # Divide by stdev to standardize the t quantile to unit variance
    # Ref: stdtinv.m:42 — x = x ./ stdev
    x = x / stdev

    # Restore scalar output shape if the input was scalar
    if scalar_input:
        x = x.squeeze()

    return x
