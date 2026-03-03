"""
Probability Density Function (PDF) of Hansen's (1994) Skewed Student's t distribution.

Migrated from distributions/skewtpdf.m (MFE Toolbox Version 4.0).

The skewed Student's t distribution generalizes the standard Student's t
distribution by introducing an asymmetry parameter lambda_ in (-1, 1).
When lambda_ = 0, the distribution reduces to the standardized Student's t.

Author: Andrew Patton (original MATLAB, a.patton@lse.ac.uk)
Modifications Copyright: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 2    Date: 12/31/2001

MATLAB-to-Python migration notes:
- ``lambda`` is a Python reserved keyword; parameter renamed to ``lambda_``
- MATLAB ``gamma()`` → ``scipy.special.gamma()``
- MATLAB ``.^`` → Python ``**`` power operator
- MATLAB element-wise ``.*`` → Python ``*`` (numpy arrays use element-wise by default)
- MATLAB NaN propagation preserved via numpy NaN semantics (IEEE 754)
- MATLAB ``size(x)`` returns ``[1,1]`` for scalars; handled via ``numpy.atleast_1d``
- MATLAB ``error()`` → Python ``raise ValueError()``
- MATLAB ``nargin`` check → Python enforces via function signature (TypeError on missing)
- MATLAB 1-indexed loops → not applicable (vectorized operations only)
"""

import numpy
from scipy.special import gamma

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['skewtpdf']


def skewtpdf(x, v, lambda_):
    """
    Compute the PDF of Hansen's (1994) Skewed Student's t distribution.

    Parameters
    ----------
    x : array_like
        Standardized t random variables at which to evaluate the PDF.
    v : array_like
        Degree of freedom parameter(s). Must satisfy v > 2.
        Can be scalar or array matching the shape of ``x``.
    lambda_ : array_like
        Asymmetry (skewness) parameter(s). Must satisfy -1 < lambda\\_ < 1.
        Can be scalar or array matching the shape of ``x``.

    Returns
    -------
    y : numpy.ndarray
        PDF values evaluated at ``x``. Returns NaN where ``v <= 2`` or
        ``|lambda_| >= 1``.

    Raises
    ------
    ValueError
        If parameters have incompatible shapes that cannot be broadcast
        to the shape of ``x``.

    Notes
    -----
    Implements Hansen's (1994) skewed Student's t distribution PDF:

    .. math::

        c = \\frac{\\Gamma((v+1)/2)}{\\sqrt{\\pi(v-2)} \\, \\Gamma(v/2)}

        a = 4 \\lambda \\, c \\, \\frac{v-2}{v-1}

        b = \\sqrt{1 + 3\\lambda^2 - a^2}

    Piecewise density:

    .. math::

        f(x) = b \\, c \\left(1 + \\frac{1}{v-2}
               \\left(\\frac{bx+a}{1-\\lambda}\\right)^2\\right)^{-(v+1)/2}
               \\quad \\text{for } x < -a/b

        f(x) = b \\, c \\left(1 + \\frac{1}{v-2}
               \\left(\\frac{bx+a}{1+\\lambda}\\right)^2\\right)^{-(v+1)/2}
               \\quad \\text{for } x \\ge -a/b

    When ``lambda_ = 0``, the distribution reduces to the standardized
    Student's t distribution with ``v`` degrees of freedom.

    The PDF is non-negative everywhere and integrates to 1 for valid
    parameter values (``v > 2``, ``|lambda_| < 1``).

    References
    ----------
    [1] Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation."
        International Economic Review, 35(3), 705-730.

    See Also
    --------
    skewtcdf : CDF of Hansen's skewed t distribution.
    skewtinv : Quantile function of Hansen's skewed t distribution.
    skewtrnd : Random variates from Hansen's skewed t distribution.
    skewtloglik : Log-likelihood of Hansen's skewed t distribution.

    Examples
    --------
    Evaluate PDF at a single point with zero skewness (reduces to
    standardized t):

    >>> import numpy
    >>> y = skewtpdf(0.0, 5.0, 0.0)

    Evaluate PDF at multiple points with positive skewness:

    >>> y = skewtpdf(numpy.array([-1.0, 0.0, 1.0]), 5.0, 0.5)
    """
    # Ref: skewtpdf.m:31-33 — nargin check
    # In Python, missing arguments raise TypeError automatically from the
    # function signature; no explicit nargin check needed.

    # ---- Convert inputs to numpy float64 arrays ----
    # Ref: skewtpdf.m:35 — implicit input handling
    x = numpy.asarray(x, dtype=numpy.float64)

    # Track original dimensionality to restore output shape for scalar inputs
    original_shape = x.shape

    # Ensure x is at least 1-d for iscompatible shape passing.
    # Ref: In MATLAB, size(scalar) = [1,1]; all arrays are at least 2-d.
    # numpy.atleast_1d converts scalar (shape ()) to array (shape (1,)).
    x = numpy.atleast_1d(x)

    v = numpy.asarray(v, dtype=numpy.float64)
    lambda_ = numpy.asarray(lambda_, dtype=numpy.float64)

    # ---- Invalidate out-of-range parameters with NaN ----
    # Ref: skewtpdf.m:35 — v(v<2) = NaN
    # Degrees of freedom must be > 2 for a well-defined density.
    # Values at exactly v=2 are allowed through but produce NaN/Inf via
    # division by (v-2)=0, matching MATLAB behavior.
    v = numpy.where(v < 2, numpy.nan, v)

    # Ref: skewtpdf.m:36 — lambda(lambda<-1 | lambda>1) = NaN
    # Asymmetry parameter must be in (-1, 1). Boundary values ±1 pass through
    # but produce division by zero in the piecewise branches, matching MATLAB.
    # Uses numpy.abs for the absolute-value test.
    lambda_ = numpy.where(numpy.abs(lambda_) > 1, numpy.nan, lambda_)

    # ---- Shape compatibility check and parameter broadcasting ----
    # Ref: skewtpdf.m:38 —
    #   [err, errtext, sizeOut, v, lambda] = iscompatible(2, v, lambda, size(x))
    # iscompatible validates that v and lambda_ are either scalar or match
    # x's shape, then broadcasts scalars to the output size.
    err, errtext, size_out, v, lambda_ = iscompatible(
        2, v, lambda_, numpy.array(x.shape)
    )

    # Ref: skewtpdf.m:40-42 — if err; error(errtext); end
    if err:
        raise ValueError(errtext)

    # Broadcast x to match the output size determined by iscompatible.
    # v and lambda_ are already broadcast by iscompatible; x may need
    # broadcasting if it was scalar and v/lambda_ defined the output shape.
    # .copy() ensures the array is writable (broadcast_to returns read-only).
    x_bc = numpy.broadcast_to(x, size_out).copy()

    # ---- Initialize output with NaN (safety default for invalid params) ----
    # Ref: skewtpdf.m implied — invalid v/lambda produce NaN output.
    # numpy.full pre-fills with NaN so any uncomputed or edge-case element
    # defaults to NaN, providing a safety net alongside NaN propagation.
    y = numpy.full(size_out, numpy.nan, dtype=numpy.float64)

    # ---- Compute Hansen (1994) skewed t PDF ----
    # NaN propagation: invalid parameters (set to NaN above) flow through
    # all arithmetic operations, producing NaN in the output via IEEE 754.
    # numpy.errstate suppresses harmless RuntimeWarnings from NaN arithmetic
    # (e.g., sqrt(NaN), NaN/NaN, division by zero when v=2 or lambda=±1).
    with numpy.errstate(invalid='ignore', divide='ignore'):
        # Ref: skewtpdf.m:44 —
        #   c = gamma((v+1)/2) ./ (sqrt(pi*(v-2)) .* gamma(v/2))
        # Normalizing constant of the skewed t density.
        # gamma() here is scipy.special.gamma (the Gamma function).
        c = gamma((v + 1.0) / 2.0) / (
            numpy.sqrt(numpy.pi * (v - 2.0)) * gamma(v / 2.0)
        )

        # Ref: skewtpdf.m:45 — a = 4*lambda.*c.*((v-2)./(v-1))
        # Location shift parameter ensuring the distribution has zero mean.
        a = 4.0 * lambda_ * c * ((v - 2.0) / (v - 1.0))

        # Ref: skewtpdf.m:46 — b = sqrt(1 + 3*lambda.^2 - a.^2)
        # Scale parameter ensuring unit variance.
        b = numpy.sqrt(1.0 + 3.0 * lambda_ ** 2 - a ** 2)

        # Ref: skewtpdf.m:48 —
        #   y1 = b.*c.*(1 + 1./(v-2).*((b.*x+a)./(1-lambda)).^2).^(-(v+1)/2)
        # Left branch of the piecewise PDF (for x < -a/b).
        # Uses (1-lambda) in the denominator for left-skew scaling.
        y1 = (
            b * c
            * (
                1.0
                + 1.0 / (v - 2.0)
                * ((b * x_bc + a) / (1.0 - lambda_)) ** 2
            )
            ** (-(v + 1.0) / 2.0)
        )

        # Ref: skewtpdf.m:49 —
        #   y2 = b.*c.*(1 + 1./(v-2).*((b.*x+a)./(1+lambda)).^2).^(-(v+1)/2)
        # Right branch of the piecewise PDF (for x >= -a/b).
        # Uses (1+lambda) in the denominator for right-skew scaling.
        y2 = (
            b * c
            * (
                1.0
                + 1.0 / (v - 2.0)
                * ((b * x_bc + a) / (1.0 + lambda_)) ** 2
            )
            ** (-(v + 1.0) / 2.0)
        )

        # Ref: skewtpdf.m:50 — y = y1.*(x<(-a./b)) + y2.*(x>=(-a./b))
        # Piecewise combination: left branch below the threshold -a/b,
        # right branch at or above the threshold.
        #
        # NaN propagation for invalid parameters:
        # When v or lambda_ is NaN, c/a/b are NaN, so y1 and y2 are NaN.
        # The threshold (-a/b) is NaN, so both comparisons (x < NaN) and
        # (x >= NaN) return False. Thus y = NaN*0 + NaN*0 = NaN + NaN = NaN
        # (IEEE 754: NaN * 0 = NaN), correctly producing NaN output.
        # The numpy.full NaN initialization serves as a safety net for any
        # unforeseen edge case.
        threshold = -a / b
        y[:] = y1 * (x_bc < threshold) + y2 * (x_bc >= threshold)

    # Restore output shape to match original input dimensionality.
    # If x was originally scalar (shape ()), atleast_1d made it (1,);
    # reshape back to scalar for consistency with scipy conventions.
    if original_shape == ():
        y = y.reshape(original_shape)

    return y
