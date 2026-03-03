"""
Cumulative Distribution Function (CDF) of the Standardized Student's t distribution.

Migrated from distributions/stdtcdf.m (MFE Toolbox Version 4.0).

The standardized Student's t distribution has unit variance, unlike the standard
Student's t distribution whose variance is v/(v-2). This module rescales the
input from the standardized t scale to the standard t scale before evaluating
the CDF via ``scipy.stats.t.cdf``.

The transformation is:
    x_standard = x * sqrt(v / (v - 2))
    p = t_cdf(x_standard, v)

where ``v`` is the degrees-of-freedom parameter and must satisfy ``v > 2``
for the variance to be finite (and therefore for the standardized distribution
to be well-defined).

Author: Kevin Sheppard (original MATLAB)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 8/1/2005

References
----------
[1] Casella, G. and Berger, R.L. (1990) 'Statistical Inference'

See Also
--------
mfe_toolbox.distributions.stdtpdf : Standardized t PDF
mfe_toolbox.distributions.stdtinv : Standardized t inverse CDF (quantile)
mfe_toolbox.distributions.stdtrnd : Standardized t random variates
mfe_toolbox.distributions.stdtloglik : Standardized t log-likelihood

MATLAB-to-Python migration notes:
- tcdf(x, v) → scipy.stats.t.cdf(x, v) — direct mapping
- sqrt(v./(v-2)) → numpy.sqrt(v / (v - 2.0))
- size(x) → numpy.array(x.shape) for iscompatible call
- nargin check → Python function signature enforces 2 required arguments
- MATLAB error() → Python raise ValueError()
- stdev(v<=2) = NaN → numpy.where(v <= 2, numpy.nan, stdev)
"""

import numpy
from scipy import stats

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['stdtcdf']


def stdtcdf(x, v):
    """
    Compute the CDF of the Standardized Student's t distribution.

    The standardized Student's t distribution is the standard t distribution
    rescaled to have unit variance. For degrees of freedom ``v > 2``, the
    standard t distribution has variance ``v / (v - 2)``, so the standardized
    version divides by ``sqrt(v / (v - 2))``. This function accepts values
    on the standardized scale and returns CDF probabilities.

    Parameters
    ----------
    x : array_like
        Standardized t random variables at which to evaluate the CDF.
        Can be scalar, 1-D array, or N-D array.
    v : array_like
        Degrees of freedom parameter(s). Must be either a scalar or an array
        with the same shape as ``x``. Values must satisfy ``v > 2`` for the
        standardized distribution to be defined; when ``v <= 2``, ``NaN`` is
        returned.

    Returns
    -------
    p : numpy.ndarray
        Cumulative distribution function evaluated at each element of ``x``.
        Same shape as ``x`` (after broadcasting ``v`` if scalar). Values are
        in the range [0, 1] for valid ``v > 2``; ``NaN`` for ``v <= 2``.

    Raises
    ------
    ValueError
        If ``v`` cannot be broadcast to the shape of ``x`` (via
        ``iscompatible``), or if any required input is empty.

    Notes
    -----
    The computation proceeds in three steps (Ref: stdtcdf.m:38-42):

    1. Compute the standard deviation of the standard t(v):
       ``stdev = sqrt(v / (v - 2))``
    2. Rescale from standardized to standard t scale:
       ``x_scaled = x * stdev``
    3. Evaluate the standard t CDF:
       ``p = t.cdf(x_scaled, v)``

    For ``v <= 2``, the variance of the t distribution is undefined (infinite
    or non-existent), so the standardized distribution cannot be defined.
    The function returns ``NaN`` for these cases, consistent with the original
    MATLAB implementation.

    Examples
    --------
    Scalar evaluation at the median (CDF(0) = 0.5 for any valid v):

    >>> import numpy
    >>> p = stdtcdf(0.0, 5.0)
    >>> numpy.testing.assert_allclose(p, 0.5, atol=1e-10)

    Array evaluation:

    >>> x = numpy.array([-1.0, 0.0, 1.0])
    >>> p = stdtcdf(x, 10.0)
    >>> assert p[0] < 0.5
    >>> assert numpy.isclose(p[1], 0.5, atol=1e-10)
    >>> assert p[2] > 0.5

    Invalid degrees of freedom (v <= 2) returns NaN:

    >>> p = stdtcdf(0.0, 2.0)
    >>> assert numpy.isnan(p)
    """
    # Convert x to float64 numpy array for consistent numerical operations
    # Ref: stdtcdf.m uses MATLAB's implicit double conversion
    x = numpy.asarray(x, dtype=numpy.float64)

    # Track original dimensionality to restore output shape for scalar inputs
    # MATLAB treats scalars as 1×1 matrices (size(scalar) → [1, 1]);
    # NumPy scalars have shape () which produces an empty array when passed
    # to numpy.array(). We use atleast_1d to ensure a valid non-empty shape.
    x_ndim_orig = x.ndim
    if x.ndim == 0:
        # Ref: stdtcdf.m:32 — MATLAB size(scalar) returns [1 1]; in Python,
        # scalar has shape () which becomes empty array. Promote to 1-D.
        x = x.reshape(1)

    # Validate and broadcast v to match x's shape
    # Ref: stdtcdf.m:32 — [err, errtext, sizeOut, v] = iscompatible(1, v, size(x))
    # narg=1 means one parameter (v); size spec is x.shape
    err, errtext, size_out, v = iscompatible(1, v, numpy.array(x.shape))

    # Ref: stdtcdf.m:33-35 — if err, error(errtext), end
    if err:
        raise ValueError(errtext)

    # Ensure v is a float64 array after broadcasting for consistent arithmetic
    v = numpy.asarray(v, dtype=numpy.float64)

    # Compute the standard deviation of the standard t(v) distribution
    # For t(v), Var = v/(v-2) when v > 2, so stdev = sqrt(v/(v-2))
    # Ref: stdtcdf.m:38 — stdev = sqrt(v./(v-2))
    # Suppress expected warnings: v=2 causes divide-by-zero (Inf), v<2 causes
    # negative under sqrt (NaN). Both are handled explicitly by the where() below.
    with numpy.errstate(divide='ignore', invalid='ignore'):
        stdev = numpy.sqrt(v / (v - 2.0))

    # For v <= 2, the variance is undefined (infinite for v=2, non-existent for v<2)
    # Set stdev to NaN so that the resulting CDF is also NaN
    # Ref: stdtcdf.m:39 — stdev(v<=2) = NaN
    stdev = numpy.where(v <= 2, numpy.nan, stdev)

    # Rescale x from standardized (unit variance) scale to standard t scale
    # If x ~ standardized_t(v), then x * stdev ~ t(v)
    # Ref: stdtcdf.m:40 — x = x.*stdev
    x_scaled = x * stdev

    # Evaluate the CDF of the standard Student's t distribution
    # Ref: stdtcdf.m:42 — p = tcdf(x, v)
    # scipy.stats.t.cdf(x, df) computes the CDF of the standard t with df degrees of freedom
    p = stats.t.cdf(x_scaled, v)

    # Restore scalar output if original input was scalar (0-D)
    # Ref: MATLAB returns a 1×1 scalar for scalar inputs; we match by squeezing
    if x_ndim_orig == 0:
        p = p.squeeze()

    return p
