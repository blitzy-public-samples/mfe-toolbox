"""
Generate random variates from a Standardized Student's t distribution.

Migrated from distributions/stdtrnd.m (MFE Toolbox Version 4.0).
The standardized Student's t distribution has unit variance for degrees of
freedom v > 2.  Standard Student's t variates are divided by their
theoretical standard deviation sqrt(v / (v - 2)) to obtain unit-variance
variates.

The function signature mirrors the MATLAB original:
    r = stdtrnd(v)             → scalar output if v scalar
    r = stdtrnd(v, S1)         → S1×S1 output
    r = stdtrnd(v, S1, S2, …)  → S1×S2×… output
    r = stdtrnd(v, [S1, S2, …])→ S1×S2×… output

Author: Kevin Sheppard (original MATLAB)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005

MATLAB-to-Python migration notes:
- trnd(v, sizeOut) → rng.standard_t(v, size=tuple(size_out))
  NumPy's Generator.standard_t generates Student's t variates.
  Ref: stdtrnd.m:48
- sqrt(v./(v-2)) → numpy.sqrt(v / (v - 2.0))  direct mapping.
  Ref: stdtrnd.m:49
- stdev(v<=2) = NaN → numpy.where(v <= 2, numpy.nan, stdev)
  Ref: stdtrnd.m:50
- MATLAB 1-indexed arrays → Python 0-indexed (no index changes needed
  for this module because all operations are element-wise).
"""

import numpy

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['stdtrnd']


def stdtrnd(v, *size_args, rng=None):
    """
    Generate random variates from a Standardized Student's t distribution.

    The standardized Student's t distribution rescales the ordinary
    Student's t distribution so that its variance equals 1 for v > 2.
    Draws from t(v) are divided by sqrt(v / (v - 2)).  When v <= 2 the
    variance of the t distribution is infinite or undefined, so the
    output is ``NaN``.

    Parameters
    ----------
    v : array_like
        Degrees-of-freedom parameter.  Can be a scalar or an array.
        Must satisfy v > 2 for a well-defined standardized distribution;
        when v <= 2 the corresponding output entries are ``NaN``.
    *size_args : int or array_like of int, optional
        Requested output size.  Accepted forms:

        * No size args: output shape matches ``v`` (scalar → scalar).
        * Single scalar ``S1``: output shape ``(S1,)``.
        * Multiple scalars ``S1, S2, …``: output shape ``(S1, S2, …)``.
        * Single vector ``[S1, S2, …]``: output shape ``(S1, S2, …)``.

        When ``v`` is non-scalar **and** size args are provided, the
        shape of ``v`` must equal the requested size.
    rng : numpy.random.Generator or None, optional
        Random number generator instance.  If ``None`` (default), a new
        ``numpy.random.default_rng()`` instance is created.

    Returns
    -------
    r : numpy.ndarray
        Standardized Student's t distributed random variates with unit
        variance (for v > 2).  Elements where v <= 2 are ``NaN``.

    Raises
    ------
    ValueError
        If ``v`` is empty, if no arguments are provided, or if the
        requested size is incompatible with the shape of ``v``.

    Notes
    -----
    Uses ``numpy.random.Generator.standard_t`` to draw from the ordinary
    Student's t(v) distribution and then divides by the theoretical
    standard deviation ``sqrt(v / (v - 2))``.

    References
    ----------
    [1] Casella, G. and Berger, R.L. (1990) *Statistical Inference*.
        Wadsworth & Brooks/Cole.

    See Also
    --------
    stdtpdf : Standardized Student's t probability density function.
    stdtcdf : Standardized Student's t cumulative distribution function.
    stdtinv : Standardized Student's t inverse CDF (quantile function).
    stdtloglik : Standardized Student's t log-likelihood.

    Examples
    --------
    Scalar degrees of freedom, default output size (scalar):

    >>> import numpy
    >>> r = stdtrnd(5.0, rng=numpy.random.default_rng(42))
    >>> r.shape
    (1, 1)

    Explicit output size:

    >>> r = stdtrnd(10.0, 3, 4, rng=numpy.random.default_rng(42))
    >>> r.shape
    (3, 4)

    Array degrees of freedom (per-element df):

    >>> v = numpy.array([5.0, 10.0, 30.0])
    >>> r = stdtrnd(v, rng=numpy.random.default_rng(42))
    >>> r.shape
    (3,)

    Values where v <= 2 produce NaN:

    >>> r = stdtrnd(numpy.array([1.5, 5.0]), rng=numpy.random.default_rng(0))
    >>> numpy.isnan(r[0])
    True
    """
    # ------------------------------------------------------------------
    # Input validation via iscompatible (1 parameter: v)
    # Ref: stdtrnd.m:38-40  — if nargin<1 error(...)
    # Ref: stdtrnd.m:42     — [err, errtext, sizeOut, v] = iscompatible(1, v, varargin{:})
    # ------------------------------------------------------------------
    result = iscompatible(1, v, *size_args)
    # Unpack: error flag, error message, output size, broadcasted v
    err = result[0]
    errtext = result[1]
    size_out = result[2]
    v = result[3]  # v has been broadcast to size_out by iscompatible

    # Ref: stdtrnd.m:44-46 — if err error(errtext) end
    if err:
        raise ValueError(errtext)

    # ------------------------------------------------------------------
    # Instantiate random number generator if not provided
    # ------------------------------------------------------------------
    if rng is None:
        rng = numpy.random.default_rng()

    # ------------------------------------------------------------------
    # Generate standard Student's t random variates
    # Ref: stdtrnd.m:48 — r = trnd(v, sizeOut)
    # NumPy's standard_t takes df (array-broadcastable) and size.
    # Since v is already broadcast to size_out by iscompatible, the
    # shapes are compatible.
    # ------------------------------------------------------------------
    r = rng.standard_t(v, size=tuple(size_out))

    # ------------------------------------------------------------------
    # Compute standard deviation of the t(v) distribution
    # The variance of t(v) is v / (v - 2) for v > 2.
    # Ref: stdtrnd.m:49 — stdev = sqrt(v ./ (v - 2))
    # Ref: stdtrnd.m:50 — stdev(v <= 2) = NaN
    #
    # When v <= 2 the division v/(v-2) may produce inf (v==2) or a
    # negative value (v<2), triggering numpy RuntimeWarnings.  These are
    # expected and handled by the subsequent numpy.where NaN assignment,
    # so we suppress them with errstate.
    # ------------------------------------------------------------------
    with numpy.errstate(divide='ignore', invalid='ignore'):
        stdev = numpy.sqrt(v / (v - 2.0))

    # Set stdev to NaN where v <= 2 (variance is undefined/infinite)
    stdev = numpy.where(v <= 2, numpy.nan, stdev)

    # ------------------------------------------------------------------
    # Standardize: divide by stdev to obtain unit-variance variates
    # Ref: stdtrnd.m:51 — r = r ./ stdev
    # ------------------------------------------------------------------
    r = r / stdev

    return r
