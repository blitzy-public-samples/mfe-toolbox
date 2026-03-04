"""
Simulate APARCH(P,O,Q) time series with power parameter delta.

Migrated from univariate/aparch_simulate.m (MFE Toolbox Version 4.0).

The Asymmetric Power ARCH (APARCH) model generalizes the standard GARCH
model by introducing a power parameter (delta) for the conditional variance
equation, allowing the model to capture asymmetric effects and power
transformations of volatility.

The conditional variance equation is:

    h(t)^(delta/2) = omega
        + alpha(1)*(|r(t-1)| + gamma(1)*r(t-1))^delta + ...
          + alpha(p)*(|r(t-p)| + gamma(p)*r(t-p))^delta
        + beta(1)*h(t-1)^(delta/2) + ... + beta(q)*h(t-q)^(delta/2)

Special cases:
    delta=2, gamma=0        -> GARCH(P,Q)
    delta=1, gamma=0        -> AVGARCH
    delta=2, gamma!=0       -> GJR-GARCH
    delta=1, gamma!=0       -> TARCH

Author: Kevin Sheppard (original MATLAB)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005

MATLAB-to-Python migration notes:
- randn(t,1) -> numpy.random.default_rng().standard_normal(t)
  Ref: aparch_simulate.m:141
- MATLAB 1-indexed loops -> Python 0-indexed. The recursion loop indices
  are carefully adjusted: MATLAB for t=(m+1):T with data(t-j) becomes
  Python for i in range(m, total_len) with data[i-j] (j still 1-based).
  Ref: aparch_simulate.m:179-193
- MATLAB ceil(rand(2000,1)*t) -> rng.integers(0, t_len, size=2000)
  Ref: aparch_simulate.m:158
- MATLAB error() -> raise ValueError()
  Ref: aparch_simulate.m throughout
- MATLAB warning('UCSD_GARCH:nonstationary',...) -> warnings.warn(...)
  Ref: aparch_simulate.m:120
"""

import warnings

import numpy

from mfe_toolbox.distributions.stdtrnd import stdtrnd
from mfe_toolbox.distributions.gedrnd import gedrnd
from mfe_toolbox.distributions.skewtrnd import skewtrnd

__all__ = ['aparch_simulate']

# Number of burn-in observations to minimize starting bias
# Ref: aparch_simulate.m:44 — "This program generates 2000 more than required"
_BURN_IN = 2000


def aparch_simulate(t, parameters, p, o, q, error_type='NORMAL'):
    """
    Simulate an APARCH(P,O,Q) time series with multiple error distributions.

    Generates a simulated time series from an Asymmetric Power ARCH model
    with power parameter delta. The function supports Normal, Student's t,
    Generalized Error Distribution (GED), and Hansen's Skewed-t innovations.

    Parameters
    ----------
    t : int or numpy.ndarray
        If scalar (int or float): length of the time series to be simulated.
        If array: a (T,) or (T,1) vector of user-supplied random innovations.
        When user-supplied, 2000 resampled burn-in observations are prepended
        to reduce starting bias.
    parameters : numpy.ndarray
        Parameter vector of length (2 + p + o + q + extrap) where extrap is
        0 for NORMAL, 1 for STUDENTST/GED, 2 for SKEWT. The ordering is::

            [omega, alpha(1), ..., alpha(p), gamma(1), ..., gamma(o),
             beta(1), ..., beta(q), delta, [nu, [lambda_]]]

        Constraints:
            - omega > 0
            - alpha(i) > 0 for all i
            - -1 < gamma(i) < 1 for all i
            - beta(j) >= 0 for all j
            - delta > 0
            - nu > 2 for STUDENTST, nu > 1 for GED
            - -1 < lambda_ < 1 for SKEWT
    p : int
        Positive integer: number of symmetric innovation lags.
    o : int
        Non-negative integer: number of asymmetric innovation lags.
        Must satisfy o <= p.
    q : int
        Non-negative integer: number of conditional variance lags
        (0 for ARCH-only model).
    error_type : str, optional
        Error distribution for innovations. One of:

        - ``'NORMAL'`` (default): Gaussian innovations
        - ``'STUDENTST'``: Standardized Student's t innovations
        - ``'GED'``: Generalized Error Distribution innovations
        - ``'SKEWT'``: Hansen's Skewed Student's t innovations

    Returns
    -------
    simulatedata : numpy.ndarray
        Simulated time series of shape (T,), where T is the requested length
        (or the length of user-supplied random numbers).
    ht : numpy.ndarray
        Conditional variances of shape (T,) corresponding to ``simulatedata``.

    Raises
    ------
    ValueError
        If inputs are invalid: wrong error_type, o > p, wrong parameter
        vector length, or invalid p/o/q values.

    Notes
    -----
    A burn-in period of 2000 observations is used to minimize starting bias
    in the simulated series. These observations are discarded before returning
    results.

    The APARCH model nests several popular GARCH variants as special cases:

    - GARCH(P,Q): delta=2, o=0
    - AVGARCH: delta=1, o=0
    - GJR-GARCH(P,O,Q): delta=2, o>0
    - TARCH(P,O,Q): delta=1, o>0

    See Also
    --------
    aparch : APARCH model estimation.
    tarch_simulate : TARCH/GJR-GARCH simulation.
    egarch_simulate : EGARCH simulation.

    Examples
    --------
    Simulate a GARCH(1,1) (delta=2, no asymmetry):

    >>> import numpy as np
    >>> data, ht = aparch_simulate(1000, np.array([0.1, 0.1, 0.85, 2.0]), 1, 0, 1)
    >>> len(data)
    1000

    Simulate an APARCH(1,1,1) with asymmetry:

    >>> data, ht = aparch_simulate(1000, np.array([0.1, 0.1, -0.1, 0.8, 0.8]), 1, 1, 1)
    >>> len(data)
    1000

    Simulate an APARCH(1,1,1) with Student's t innovations:

    >>> data, ht = aparch_simulate(
    ...     1000, np.array([0.1, 0.1, -0.1, 0.85, 2.0, 6.0]), 1, 1, 1, 'STUDENTST'
    ... )
    >>> len(data)
    1000
    """
    # ------------------------------------------------------------------
    # Input Checking
    # Ref: aparch_simulate.m:70-76 — nargin check and error_type default
    # ------------------------------------------------------------------
    if error_type is None or error_type == '':
        error_type = 'NORMAL'

    error_type = error_type.upper()

    # Ref: aparch_simulate.m:82-92 — determine extra parameters based on error type
    if error_type == 'NORMAL':
        extrap = 0
    elif error_type == 'STUDENTST':
        extrap = 1
    elif error_type == 'GED':
        extrap = 1
    elif error_type == 'SKEWT':
        extrap = 2
    else:
        raise ValueError('Unknown error type')

    # Ref: aparch_simulate.m:94-96 — O must be <= P
    if o > p:
        raise ValueError('O must be less than P')

    # Ref: aparch_simulate.m:98-103 — ensure parameters is a column vector
    # with the correct number of elements
    parameters = numpy.asarray(parameters, dtype=numpy.float64).ravel()
    expected_length = 2 + p + o + q + extrap
    if len(parameters) != expected_length:
        raise ValueError(
            'parameters must be a column vector with the correct number of parameters.'
        )

    # ------------------------------------------------------------------
    # Separate the parameters
    # Ref: aparch_simulate.m:106-110
    # ------------------------------------------------------------------
    omega = parameters[0]
    # Ref: aparch_simulate.m:107 — alpha=parameters(2:p+1) [MATLAB 1-indexed]
    alpha = parameters[1:p + 1]
    # Ref: aparch_simulate.m:108 — gamma=parameters(p+2:p+o+1) [MATLAB 1-indexed]
    gamma_params = parameters[p + 1:p + o + 1]
    # Ref: aparch_simulate.m:109 — beta=parameters(p+o+2:p+o+q+1) [MATLAB 1-indexed]
    beta = parameters[p + o + 1:p + o + q + 1]
    # Ref: aparch_simulate.m:110 — delta = parameters(p+o+q+2) [MATLAB 1-indexed]
    delta = parameters[p + o + q + 1]

    # ------------------------------------------------------------------
    # Stationarity check
    # Ref: aparch_simulate.m:112-124
    # ------------------------------------------------------------------
    if o > 0:
        # Ref: aparch_simulate.m:113 — gamma2=[gamma;zeros(p-o,1)]
        gamma2 = numpy.concatenate([gamma_params, numpy.zeros(p - o)])
        sumag = 0.5 * numpy.sum(alpha * gamma2)
    else:
        sumag = 0.0

    # Ref: aparch_simulate.m:119-124 — stationarity warning
    if (numpy.sum(alpha) + sumag + numpy.sum(beta)) >= 1.0:
        warnings.warn(
            'PARAMETERS may be in the non-stationary space, '
            'be sure to check that H is not inf.',
            stacklevel=2
        )
        nonstationary = True
    else:
        nonstationary = False

    # ------------------------------------------------------------------
    # Validate p, o, q are valid scalars
    # Ref: aparch_simulate.m:126-128
    # ------------------------------------------------------------------
    if p < 1 or o < 0 or q < 0:
        raise ValueError(
            'P, O and Q must be scalars with P positive and O and Q non-negative'
        )

    # ------------------------------------------------------------------
    # Validate t shape
    # Ref: aparch_simulate.m:130-132
    # ------------------------------------------------------------------
    t_input = numpy.asarray(t, dtype=numpy.float64)
    if t_input.ndim == 2 and t_input.shape[1] != 1:
        raise ValueError(
            'T must be either a positive scalar or a vector of random numbers.'
        )

    # ------------------------------------------------------------------
    # Initialize random numbers
    # Ref: aparch_simulate.m:138-161
    # ------------------------------------------------------------------
    rng = numpy.random.default_rng()

    if numpy.isscalar(t) or (t_input.ndim == 0):
        # Ref: aparch_simulate.m:139 — t=t+2000 (burn-in)
        t_len = int(t_input.item()) + _BURN_IN

        # Ref: aparch_simulate.m:140-154 — generate distribution-specific innovations
        if error_type == 'NORMAL':
            # Ref: aparch_simulate.m:141 — RandomNums=randn(t,1)
            random_nums = rng.standard_normal(t_len)
        elif error_type == 'STUDENTST':
            # Ref: aparch_simulate.m:143-144 — nu=parameters(3+o+p+q); stdtrnd(nu,t,1)
            nu = parameters[2 + o + p + q]
            random_nums = stdtrnd(nu, t_len, 1).ravel()
        elif error_type == 'GED':
            # Ref: aparch_simulate.m:146-147 — nu=parameters(3+o+p+q); gedrnd(nu,t,1)
            nu = parameters[2 + o + p + q]
            random_nums = gedrnd(nu, t_len, 1).ravel()
        elif error_type == 'SKEWT':
            # Ref: aparch_simulate.m:149-151
            # nu=parameters(3+o+p+q); lambda=parameters(4+o+p+q)
            nu = parameters[2 + o + p + q]
            lambda_ = parameters[3 + o + p + q]
            random_nums = skewtrnd(nu, lambda_, t_len, 1).ravel()
        else:
            raise ValueError('Unknown error type')
    else:
        # Ref: aparch_simulate.m:156-161 — user-supplied random numbers
        random_nums = t_input.ravel().copy()
        t_len = len(random_nums)
        # Ref: aparch_simulate.m:158 — seeds=ceil(rand(2000,1)*t)
        # MATLAB ceil(rand*N) gives 1..N; Python integers(0, N) gives 0..N-1
        seeds = rng.integers(0, t_len, size=_BURN_IN)
        # Ref: aparch_simulate.m:159 — RandomNums=[RandomNums(seeds);RandomNums]
        random_nums = numpy.concatenate([random_nums[seeds], random_nums])
        t_len = len(random_nums)

    # ------------------------------------------------------------------
    # Back-cast initialization
    # Ref: aparch_simulate.m:163-176
    # ------------------------------------------------------------------
    m = max(p, o, q)
    # Ref: aparch_simulate.m:165 — RandomNums=[zeros(m,1);RandomNums]
    random_nums = numpy.concatenate([numpy.zeros(m), random_nums])

    # Ref: aparch_simulate.m:167-171 — unconditional standard deviation
    # Note: the MATLAB code uses the nonstationary branch to compute UncondStd
    # from the formula, and the stationary branch defaults to 1. This matches
    # the original code logic exactly.
    if nonstationary:
        denom = 1.0 - numpy.sum(alpha) - numpy.sum(beta)
        if denom > 0.0:
            uncond_std = numpy.sqrt(omega / denom)
        else:
            # Ref: aparch_simulate.m:168 — MATLAB sqrt(negative) returns complex;
            # Python returns NaN. Using 1.0 fallback for truly non-stationary case
            # where sum(alpha)+sum(beta)>=1. Burn-in absorbs initialization bias.
            uncond_std = 1.0
    else:
        # Ref: aparch_simulate.m:170-171
        uncond_std = 1.0

    # Ref: aparch_simulate.m:173-176 — initialize arrays
    total_len = t_len + m
    # Ref: aparch_simulate.m:173 — hdelta=UncondStd.^delta*ones(t+m,1)
    hdelta = numpy.ones(total_len) * (uncond_std ** delta)
    # Ref: aparch_simulate.m:174 — h=UncondStd.^delta*ones(t+m,1)
    h = numpy.ones(total_len) * (uncond_std ** delta)
    # Ref: aparch_simulate.m:175 — data=UncondStd*ones(t+m,1)
    data = numpy.ones(total_len) * uncond_std

    # ------------------------------------------------------------------
    # APARCH recursion
    # Ref: aparch_simulate.m:179-193
    # Note on indexing: MATLAB uses 1-based indexing with loop t=(m+1):T.
    # Python uses 0-based indexing with loop i=m..(total_len-1).
    # The offset cancels in data[i-j] where j is kept 1-based.
    # Ref: aparch_simulate.m:179 — for t = (m + 1):T
    # ------------------------------------------------------------------
    for i in range(m, total_len):
        # Ref: aparch_simulate.m:180 — hdelta(t) = omega
        hdelta[i] = omega

        # Ref: aparch_simulate.m:181-187 — alpha and gamma terms
        for j in range(1, p + 1):
            # Ref: aparch_simulate.m:182 — if o>=j
            if o >= j:
                # Ref: aparch_simulate.m:183
                # hdelta(t) = hdelta(t) + alpha(j)*(abs(data(t-j))+gamma(j)*data(t-j))^delta
                hdelta[i] += alpha[j - 1] * (
                    numpy.abs(data[i - j]) + gamma_params[j - 1] * data[i - j]
                ) ** delta
            else:
                # Ref: aparch_simulate.m:185
                # hdelta(t) = hdelta(t) + alpha(j)*abs(data(t-j))^delta
                hdelta[i] += alpha[j - 1] * numpy.abs(data[i - j]) ** delta

        # Ref: aparch_simulate.m:188-190 — beta terms
        for j in range(1, q + 1):
            # Ref: aparch_simulate.m:189 — hdelta(t) = hdelta(t) + beta(j)*hdelta(t-j)
            hdelta[i] += beta[j - 1] * hdelta[i - j]

        # Ref: aparch_simulate.m:191 — h(t) = hdelta(t)^(2/delta)
        h[i] = hdelta[i] ** (2.0 / delta)

        # Ref: aparch_simulate.m:192 — data(t) = sqrt(h(t))*RandomNums(t)
        data[i] = numpy.sqrt(h[i]) * random_nums[i]

    # ------------------------------------------------------------------
    # Extract output, removing burn-in and back-cast
    # Ref: aparch_simulate.m:195 — simulatedata=data((m+1+2000):T)
    # MATLAB (m+1+2000) in 1-indexed -> Python (m+2000) in 0-indexed
    # Ref: aparch_simulate.m:196 — ht=h(m+1+2000:T)
    # ------------------------------------------------------------------
    simulatedata = data[m + _BURN_IN:]
    ht = h[m + _BURN_IN:]

    return simulatedata, ht
