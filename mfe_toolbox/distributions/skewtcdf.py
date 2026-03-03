"""
Cumulative Distribution Function (CDF) of Hansen's (1994) Skewed Student's t distribution.

Migrated from distributions/skewtcdf.m (MFE Toolbox Version 4.0).
Computes the CDF of the skewed t distribution parameterized by degrees of freedom
``v`` (>2) and asymmetry parameter ``lambda_`` (|lambda_| < 1) following
Hansen, B.E. (1994), "Autoregressive Conditional Density Estimation",
International Economic Review, 35(3), 705-730.

Author: Andrew Patton (original MATLAB), Kevin Sheppard (modifications)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 2    Date: 12/31/2001

MATLAB-to-Python migration notes:
- ``lambda`` renamed to ``lambda_`` to avoid Python keyword conflict
- MATLAB ``tcdf(x, v)`` replaced by ``scipy.stats.t.cdf(x, v)``
- MATLAB ``gamma(x)`` replaced by ``scipy.special.gamma(x)``
- MATLAB ``size(x)`` replaced by ``numpy.array(x.shape)`` for iscompatible call
- Vectorized Boolean mask multiplication preserved from MATLAB original
  (Ref: skewtcdf.m:48-49)
- numpy.errstate used to suppress IEEE 754 warnings from intermediate
  computations on invalid parameter entries (v <= 2 or |lambda_| >= 1)
  which correctly propagate NaN through the result
"""

import numpy
import scipy.special
import scipy.stats

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['skewtcdf']


def skewtcdf(x, v, lambda_):
    """
    Compute the CDF of Hansen's (1994) Skewed Student's t distribution.

    Parameters
    ----------
    x : array_like
        Standardized t random variables at which to evaluate the CDF.
    v : array_like
        Degrees of freedom parameter(s). Must satisfy ``v > 2``.
        Can be scalar or array matching the shape of ``x``.
    lambda_ : array_like
        Asymmetry (skewness) parameter(s). Must satisfy ``-1 < lambda_ < 1``.
        Can be scalar or array matching the shape of ``x``.

    Returns
    -------
    p : numpy.ndarray
        CDF values evaluated at ``x``.  Entries where parameters are invalid
        (``v <= 2`` or ``|lambda_| >= 1``) are ``NaN``, produced by natural
        IEEE 754 NaN propagation matching MATLAB behavior.

    Raises
    ------
    ValueError
        If ``x``, ``v``, and ``lambda_`` have incompatible shapes (as
        determined by :func:`iscompatible`).

    Notes
    -----
    Hansen's (1994) skewed Student's t distribution is defined by the
    following constants computed from the degrees of freedom ``v`` and
    asymmetry parameter ``lambda_``:

    .. math::

        c = \\frac{\\Gamma((v+1)/2)}{\\sqrt{\\pi(v-2)} \\, \\Gamma(v/2)}

        a = 4 \\lambda \\, c \\, \\frac{v-2}{v-1}

        b = \\sqrt{1 + 3 \\lambda^2 - a^2}

    The CDF is computed piecewise (Ref: skewtcdf.m:48-49):

    For ``x < -a/b``:

    .. math::

        P(x) = (1-\\lambda) \\, F_t\\!\\left(
            \\frac{(bx+a)}{(1-\\lambda)} \\sqrt{\\frac{v}{v-2}},\\; v\\right)

    For ``x \\geq -a/b``:

    .. math::

        P(x) = \\frac{1-\\lambda}{2} + (1+\\lambda)\\!\\left(
            F_t\\!\\left(
                \\frac{(bx+a)}{(1+\\lambda)} \\sqrt{\\frac{v}{v-2}},\\; v
            \\right) - \\frac{1}{2}\\right)

    where :math:`F_t(\\cdot, v)` is the standard Student's t CDF with
    ``v`` degrees of freedom.  The :math:`\\sqrt{v/(v-2)}` factor converts
    from unit-variance standardized t to the standard Student's t used by
    ``scipy.stats.t.cdf``.

    When ``lambda_ = 0`` the distribution reduces to the standardized
    Student's t.

    References
    ----------
    .. [1] Hansen, B.E. (1994). "Autoregressive Conditional Density
       Estimation." *International Economic Review*, 35(3), 705-730.

    See Also
    --------
    skewtpdf : PDF of Hansen's skewed t distribution.
    skewtinv : Inverse CDF (quantile function) of Hansen's skewed t.
    skewtrnd : Random variates from Hansen's skewed t distribution.
    skewtloglik : Log-likelihood of Hansen's skewed t distribution.

    Examples
    --------
    >>> import numpy
    >>> p = skewtcdf(0.0, 5.0, 0.0)
    >>> numpy.testing.assert_allclose(p, 0.5, atol=1e-10)

    >>> p = skewtcdf(numpy.array([-1.0, 0.0, 1.0]), 5.0, 0.3)
    >>> p.shape
    (3,)
    """
    # ------------------------------------------------------------------
    # Input conversion
    # Ref: skewtcdf.m — MATLAB implicitly handles type promotion
    # ------------------------------------------------------------------
    x = numpy.asarray(x, dtype=numpy.float64)
    scalar_input = x.ndim == 0
    if scalar_input:
        # Ref: In MATLAB, size(scalar) = [1, 1]. Ensure at least 1-d for
        # iscompatible and consistent array operations.
        x = x.reshape((1,))

    # ------------------------------------------------------------------
    # Shape validation and parameter broadcasting
    # Ref: skewtcdf.m:35 — [err, errtext, sizeOut, v, lambda] =
    #                        iscompatible(2, v, lambda, size(x))
    # ------------------------------------------------------------------
    err, errtext, size_out, v, lambda_ = iscompatible(
        2, v, lambda_, numpy.array(x.shape)
    )
    if err:
        raise ValueError(errtext)

    # Ensure broadcasted parameters are float64 ndarrays
    v = numpy.asarray(v, dtype=numpy.float64)
    lambda_ = numpy.asarray(lambda_, dtype=numpy.float64)

    # ------------------------------------------------------------------
    # Hansen (1994) constants and piecewise CDF computation
    #
    # Invalid parameters (v <= 2, |lambda_| >= 1) naturally produce NaN
    # through IEEE 754 arithmetic (sqrt of negative, division by zero),
    # exactly matching MATLAB behavior.  numpy.errstate suppresses the
    # harmless runtime warnings from these intermediate operations.
    # ------------------------------------------------------------------
    with numpy.errstate(invalid='ignore', divide='ignore'):
        # Normalizing constant c
        # Ref: skewtcdf.m:41 — c = gamma((v+1)/2) ./ (sqrt(pi*(v-2)) .* gamma(v/2))
        c = scipy.special.gamma((v + 1.0) / 2.0) / (
            numpy.sqrt(numpy.pi * (v - 2.0)) * scipy.special.gamma(v / 2.0)
        )

        # Location shift parameter a
        # Ref: skewtcdf.m:42 — a = 4*lambda.*c.*((v-2)./(v-1))
        a = 4.0 * lambda_ * c * ((v - 2.0) / (v - 1.0))

        # Scale factor b
        # Ref: skewtcdf.m:43 — b = sqrt(1 + 3*lambda.^2 - a.^2)
        b = numpy.sqrt(1.0 + 3.0 * lambda_ ** 2 - a ** 2)

        # Standardized arguments for Student's t CDF
        # Ref: skewtcdf.m:45 — y1 = (b.*x+a)./(1-lambda).*sqrt(v./(v-2))
        # Ref: skewtcdf.m:46 — y2 = (b.*x+a)./(1+lambda).*sqrt(v./(v-2))
        # The sqrt(v/(v-2)) converts from the unit-variance standardized t
        # (Hansen's parameterization) to the standard t used by tcdf / t.cdf.
        std_factor = numpy.sqrt(v / (v - 2.0))
        y1 = (b * x + a) / (1.0 - lambda_) * std_factor
        y2 = (b * x + a) / (1.0 + lambda_) * std_factor

        # Piecewise CDF via vectorized Boolean-mask multiplication
        # Ref: skewtcdf.m:48 — p = (1-lambda).*tcdf(y1,v).*(x<-a./b)
        # Ref: skewtcdf.m:49 — p = p + (x>=-a./b).*((1-lambda)/2 ...
        #                            + (1+lambda).*(tcdf(y2,v)-0.5))
        #
        # tcdf(y, v) in MATLAB maps directly to scipy.stats.t.cdf(y, v).
        # Boolean masks as 0/1 multipliers faithfully replicate the MATLAB
        # vectorized branching.  For invalid parameters, NaN propagates
        # through both branches identically to MATLAB (0*NaN = NaN in
        # IEEE 754).
        lower_mask = (x < -a / b)
        p = (1.0 - lambda_) * scipy.stats.t.cdf(y1, v) * lower_mask

        upper_mask = ~lower_mask
        p = p + upper_mask * (
            (1.0 - lambda_) / 2.0
            + (1.0 + lambda_) * (scipy.stats.t.cdf(y2, v) - 0.5)
        )

    # ------------------------------------------------------------------
    # Return scalar for scalar input, array otherwise
    # ------------------------------------------------------------------
    if scalar_input:
        return p.squeeze()
    return p
