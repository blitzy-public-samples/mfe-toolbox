"""
Random variate generator for Hansen's (1994) Skewed Student's t distribution.

Migrated from distributions/skewtrnd.m (MFE Toolbox Version 4.0).
Generates random samples from the skewed-t distribution parameterized by
degrees of freedom V and asymmetry parameter LAMBDA using the inverse
transform method: uniform variates are mapped through the skewed-t inverse
CDF (skewtinv).

Author: Andrew Patton (original MATLAB)
        a.patton@lse.ac.uk
Modifications Copyright: Kevin Sheppard
        kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005

MATLAB-to-Python migration notes:
- rand(sizeOut) → rng.random(tuple(size_out)) using numpy.random.default_rng() per AAP
- iscompatible(2,v,lambda,varargin{:}) → iscompatible(2, v, lambda_, *size_args)
- Parameter 'lambda' → renamed to 'lambda_' (Python reserved keyword)
- nargin < 2 check → enforced by Python function signature requiring v and lambda_
- error('msg') → raise ValueError('msg') per AAP MATLAB→Python translation rules
- 0-based indexing not relevant here (no direct array indexing)
"""

import numpy as np

from mfe_toolbox.distributions.iscompatible import iscompatible
from mfe_toolbox.distributions.skewtinv import skewtinv

__all__ = ['skewtrnd']


def skewtrnd(v, lambda_, *size_args, rng=None):
    """
    Generate random variates from Hansen's (1994) Skewed Student's t distribution.

    Uses the inverse transform method: generates uniform(0,1) variates and maps
    them through the skewed-t inverse CDF (``skewtinv``).

    Parameters
    ----------
    v : array_like
        Degrees of freedom parameter. Must satisfy ``v > 2``. Can be scalar
        or array; if array, its shape must be compatible with ``lambda_`` and
        any explicit size specification.
    lambda_ : array_like
        Skewness (asymmetry) parameter. Must satisfy ``-0.99 < lambda_ < 0.99``.
        Can be scalar or array; if array, its shape must be compatible with
        ``v`` and any explicit size specification.
    *size_args : int or array_like, optional
        Output size specification. Accepted formats:

        - No size args: output shape is inferred from ``v`` and ``lambda_``
          (scalar inputs produce a single scalar output).
        - Single array/list ``[S1, S2, ..., SN]``: output shape is
          ``(S1, S2, ..., SN)``.
        - Multiple scalar ints ``S1, S2, ..., SN``: output shape is
          ``(S1, S2, ..., SN)``.

        If ``v`` or ``lambda_`` is non-scalar and size args are provided, the
        parameter shapes must match the requested size.
    rng : numpy.random.Generator, optional
        NumPy random number generator instance for reproducibility.  If
        ``None`` (default), a new generator is created via
        ``numpy.random.default_rng()``.

    Returns
    -------
    r : numpy.ndarray
        Skewed-t distributed random variates with shape determined by the
        broadcast-compatible combination of ``v``, ``lambda_``, and any
        explicit size specification.

    Raises
    ------
    ValueError
        If the shapes of ``v`` and ``lambda_`` are not mutually compatible,
        or if they are incompatible with the requested output size.

    Notes
    -----
    The Hansen (1994) skewed-t distribution is parameterized by degrees of
    freedom ``v`` (controls tail heaviness) and asymmetry parameter ``lambda_``
    (controls skewness direction and magnitude).

    The inverse transform sampling method is used:

    1. Generate uniform(0, 1) random variates ``u`` of the target output shape.
    2. Map ``u`` through the inverse CDF: ``r = skewtinv(u, v, lambda_)``.

    This guarantees that the resulting samples follow the skewed-t distribution
    exactly, subject to the numerical precision of ``skewtinv``.

    References
    ----------
    .. [1] Hansen, B.E. (1994), "Autoregressive Conditional Density Estimation,"
           International Economic Review, 35, 705-730.

    See Also
    --------
    skewtpdf : PDF of Hansen's skewed t.
    skewtcdf : CDF of Hansen's skewed t.
    skewtinv : Inverse CDF (quantile function) of Hansen's skewed t.
    skewtloglik : Log-likelihood of Hansen's skewed t.

    Examples
    --------
    Generate a single skewed-t variate with v=5, lambda_=0.3:

    >>> import numpy as np
    >>> from mfe_toolbox.distributions.skewtrnd import skewtrnd
    >>> r = skewtrnd(5.0, 0.3, rng=np.random.default_rng(42))
    >>> isinstance(r, np.ndarray)
    True

    Generate a 3x4 array of skewed-t variates:

    >>> r = skewtrnd(5.0, 0.3, 3, 4, rng=np.random.default_rng(42))
    >>> r.shape
    (3, 4)

    Generate variates with array parameters:

    >>> v_arr = np.array([5.0, 10.0, 20.0])
    >>> lam_arr = np.array([0.1, -0.2, 0.3])
    >>> r = skewtrnd(v_arr, lam_arr, rng=np.random.default_rng(42))
    >>> r.shape
    (3,)
    """
    # Ref: skewtrnd.m:41-43 — nargin < 2 check
    # Enforced by Python function signature: v and lambda_ are required positional args.

    # Ref: skewtrnd.m:45 — [err, errtext, sizeOut] = iscompatible(2,v,lambda,varargin{:})
    # Call iscompatible with 2 parameters (v, lambda_) plus any size specification.
    # iscompatible returns (err, errtext, size_out, v_broadcast, lambda_broadcast).
    # MATLAB original only captures (err, errtext, sizeOut) — broadcast params not used
    # because skewtinv handles its own broadcasting internally.
    result = iscompatible(2, v, lambda_, *size_args)
    err = result[0]
    errtext = result[1]
    size_out = result[2]
    # Ref: skewtrnd.m:47-49 — if err, error(errtext)
    if err:
        raise ValueError(errtext)

    # Ref: skewtrnd.m:50 — u = rand(sizeOut)
    # AAP rule: use numpy.random.default_rng() instead of MATLAB rand()
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(tuple(size_out))

    # Ref: skewtrnd.m:51 — r = skewtinv(u, v, lambda)
    # Pass original (non-broadcast) v and lambda_ to skewtinv, which handles
    # its own shape validation and broadcasting internally via iscompatible.
    r = skewtinv(u, v, lambda_)

    return r
