"""
Generate random variates from the Generalized Error Distribution (GED).

Migrated from distributions/gedrnd.m (MFE Toolbox Version 4.0).

The GED with shape parameter v >= 1 has probability density:

    f(x, v) = [v / (lambda * 2^(1+1/v) * Gamma(1/v))] *
              exp(-0.5 * |x / lambda|^v)

    lambda = [2^(-2/v) * Gamma(1/v) / Gamma(3/v)]^0.5

The algorithm exploits the relation: if X ~ GED(v), then |X|^v ~ Gamma(1/v)
(Tadikamalla, 1980). GAMRND does the computational work.

References
----------
[1] Tadikamalla (1980), J.Am.Stat.Assoc. (75)
[2] Nelson (1991), Econometrica

Authors
-------
Original MATLAB: Ivana Komunjer (komunjer@hss.caltech.edu)
Modifications: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

MATLAB-to-Python migration notes:
- gamrnd(shape, scale, size) -> rng.gamma(shape, scale, size=size)
  Ref: gedrnd.m:58
- gamma(x) -> scipy.special.gamma(x)
  Ref: gedrnd.m:63
- rand(sizeOut) > 0.5 -> rng.random(size) > 0.5
  Ref: gedrnd.m:60
- v(v<1) = NaN -> numpy.where(v < 1, numpy.nan, v)
  Ref: gedrnd.m:49 — MATLAB in-place assignment; Python creates new array
- MATLAB 1-indexed arrays -> Python 0-indexed (no index changes needed here)
"""

import numpy
from scipy.special import gamma as _gamma_func

from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = ['gedrnd']


def gedrnd(v, *size_args, rng=None):
    """
    Generate random variates from the Generalized Error Distribution (GED).

    Produces an array of GED-distributed random variates with unit variance.
    The shape parameter ``v`` controls the tail thickness (v=2 yields normal,
    v=1 yields double-exponential/Laplace). Values with v < 1 are invalid
    and produce NaN in the output.

    Parameters
    ----------
    v : array_like
        Shape (degrees of freedom) parameter of the GED. Must satisfy v >= 1.
        Can be a scalar or an array. Elements where v < 1 produce NaN in
        the corresponding output positions.
    *size_args : int or array_like, optional
        Output size specification, following MATLAB conventions:

        - Omitted: output shape matches shape of ``v``. If ``v`` is scalar,
          output shape is ``(1, 1)``.
        - Single int ``S``: output is ``(S,)`` shaped.
        - Multiple ints ``S1, S2, ...``: output is ``(S1, S2, ...)`` shaped.
        - Single array ``[S1, S2, ...]``: output is ``(S1, S2, ...)`` shaped.

        If ``v`` is non-scalar and size arguments are provided, the shape of
        ``v`` must equal the requested output shape.
    rng : numpy.random.Generator, optional
        NumPy random number generator instance for reproducibility.
        If ``None`` (default), a new generator is created via
        ``numpy.random.default_rng()``.

    Returns
    -------
    r : numpy.ndarray
        Array of GED-distributed random variates with unit variance and the
        shape determined by ``v`` and ``size_args``.

    Raises
    ------
    ValueError
        If ``v`` and the requested output size are incompatible shapes,
        or if required inputs are empty.

    Notes
    -----
    The algorithm uses the identity: if X ~ GED(v), then |X|^v ~ Gamma(1/v)
    (Tadikamalla, 1980). The procedure is:

    1. Draw ``g`` from Gamma(1/v, 1).
    2. Compute ``r = g^(1/v)``.
    3. Assign a random sign (+1 or -1) with equal probability.
    4. Standardize by dividing by ``sqrt(Gamma(3/v) / Gamma(1/v))`` to
       achieve unit variance.

    When v = 2, the GED reduces to the standard normal distribution.
    When v = 1, the GED is the Laplace (double exponential) distribution.

    See Also
    --------
    gedpdf : GED probability density function.
    gedcdf : GED cumulative distribution function.
    gedinv : GED inverse CDF (quantile function).
    gedloglik : GED log-likelihood function.

    Examples
    --------
    Generate a single GED(2) random variate (equivalent to standard normal):

    >>> import numpy as np
    >>> from mfe_toolbox.distributions.gedrnd import gedrnd
    >>> r = gedrnd(2.0, rng=np.random.default_rng(42))
    >>> r.shape
    (1, 1)

    Generate a 3x4 array of GED(1.5) variates:

    >>> r = gedrnd(1.5, 3, 4, rng=np.random.default_rng(42))
    >>> r.shape
    (3, 4)

    Invalid shape parameter v < 1 produces NaN:

    >>> r = gedrnd(0.5)
    >>> np.isnan(r).all()
    True
    """
    # Convert v to float64 numpy array for consistent numerical operations
    v = numpy.asarray(v, dtype=numpy.float64)

    # Ref: gedrnd.m:49 — Invalidate v < 1 by setting to NaN
    # MATLAB: v(v<1)=NaN;
    # This is done BEFORE iscompatible, matching MATLAB ordering
    v = numpy.where(v < 1, numpy.nan, v)

    # Ref: gedrnd.m:51 — Validate shape compatibility and broadcast v
    # MATLAB: [err, errtext, sizeOut] = iscompatible(1, v, varargin{:});
    # Python iscompatible always returns broadcasted params (no nargout concept)
    result = iscompatible(1, v, *size_args)
    err = result[0]
    errtext = result[1]
    size_out = result[2]
    # Ref: gedrnd.m:51 — Python iscompatible returns broadcasted v as 4th element
    v = result[3]

    # Ref: gedrnd.m:53-55 — Raise error if shapes are incompatible
    # MATLAB: if err, error(errtext), end
    if err:
        raise ValueError(errtext)

    # Initialize random number generator if not provided
    # Allows reproducible results when rng is seeded externally
    if rng is None:
        rng = numpy.random.default_rng()

    # Identify positions with invalid v (NaN from v < 1 check above)
    # These positions will produce NaN in the final output
    nan_mask = numpy.isnan(v)

    # Create a computation-safe copy of v: replace NaN with 1.0 to avoid
    # errors in numpy.random.Generator.gamma() which does not accept NaN.
    # The final output at NaN positions is overwritten with NaN after computation.
    v_safe = numpy.where(nan_mask, 1.0, v)

    # Ref: gedrnd.m:58 — Generate gamma random variates
    # MATLAB: r = gamrnd(1./v, 1, sizeOut);
    # numpy.random.Generator.gamma(shape, scale, size) matches MATLAB gamrnd
    r = rng.gamma(1.0 / v_safe, 1.0, size=tuple(size_out))

    # Ref: gedrnd.m:59 — Power transform: if |X|^v ~ Gamma(1/v), then
    # |X| = Gamma(1/v)^(1/v)
    # MATLAB: r = r.^(1./v);
    r = r ** (1.0 / v_safe)

    # Ref: gedrnd.m:60 — Generate random sign +1 or -1 with equal probability
    # MATLAB: rndsgn = 2*((rand(sizeOut)>0.5)-0.5);
    # This produces -1.0 when uniform <= 0.5, and +1.0 when uniform > 0.5
    rndsgn = 2.0 * (rng.random(tuple(size_out)) > 0.5).astype(numpy.float64) - 1.0

    # Ref: gedrnd.m:61 — Apply random sign to make distribution symmetric
    # MATLAB: r = r.*rndsgn;
    r = r * rndsgn

    # Ref: gedrnd.m:63 — Compute standardization scale factor for unit variance
    # MATLAB: scalex = (gamma(3./v)./gamma(1./v)).^0.5;
    # This ensures Var(X) = 1 for the GED distribution
    scalex = numpy.sqrt(_gamma_func(3.0 / v_safe) / _gamma_func(1.0 / v_safe))

    # Ref: gedrnd.m:64 — Standardize the random variates
    # MATLAB: r = r./scalex;
    r = r / scalex

    # Overwrite positions where v < 1 (invalid) with NaN
    # This faithfully reproduces MATLAB behavior where NaN propagates through
    # gamrnd and arithmetic operations for invalid v values
    if numpy.any(nan_mask):
        r = numpy.where(nan_mask, numpy.nan, r)

    return r
