"""
Simulate AGARCH(P,Q) and NAGARCH(P,Q) time series with multiple error distributions.

Migrated from univariate/agarch_simulate.m (MFE Toolbox Version 4.0).

The conditional variance h(t) of an AGARCH(P,Q) process is given by:

    h(t) = omega
           + alpha(1)*(r_{t-1} - gamma)^2 + ... + alpha(p)*(r_{t-p} - gamma)^2
           + beta(1)*h(t-1) + ... + beta(q)*h(t-q)

The conditional variance h(t) of a NAGARCH(P,Q) process is given by:

    h(t) = omega
           + alpha(1)*(r_{t-1} - gamma*sqrt(h(t-1)))^2
           + ... + alpha(p)*(r_{t-p} - gamma*sqrt(h(t-p)))^2
           + beta(1)*h(t-1) + ... + beta(q)*h(t-q)

The simulation generates 2000 extra observations as burn-in to minimize starting bias,
then discards them from the returned output.

Author: Kevin Sheppard (original MATLAB)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/12/2009

MATLAB-to-Python migration notes:
- randn(T,1) -> numpy.random.default_rng().standard_normal(T)
  Ref: agarch_simulate.m:115
- stdtrnd(nu,t,1) -> stdtrnd(nu, t, rng=rng).ravel()
  Ref: agarch_simulate.m:118
- gedrnd(nu,t,1) -> gedrnd(nu, t, rng=rng).ravel()
  Ref: agarch_simulate.m:120-121
- skewtrnd(nu,lambda,t,1) -> skewtrnd(nu, lambda_, t, rng=rng).ravel()
  Ref: agarch_simulate.m:124-125
- MATLAB 1-indexed loops -> Python 0-indexed loops
  Ref: agarch_simulate.m:156-173
- ceil(rand(2000,1)*t) -> rng.integers(0, t_len, size=2000)
  Ref: agarch_simulate.m:132 — MATLAB produces 1-based indices; Python 0-based
- warning('UCSD_GARCH:nonstationary', ...) -> warnings.warn(...)
  Ref: agarch_simulate.m:91
- error(...) -> raise ValueError(...)
  Ref: agarch_simulate.m throughout
"""

import warnings

import numpy as np

from mfe_toolbox.distributions.gedrnd import gedrnd
from mfe_toolbox.distributions.skewtrnd import skewtrnd
from mfe_toolbox.distributions.stdtrnd import stdtrnd

__all__ = ['agarch_simulate']

# Number of burn-in observations to minimize starting bias
# Ref: agarch_simulate.m:42 — "generates 2000 more than required"
_BURN_IN = 2000


def agarch_simulate(
    t: int | np.ndarray,
    parameters: np.ndarray,
    p: int,
    q: int,
    model_type: str = 'AGARCH',
    error_type: str = 'NORMAL',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate an AGARCH(P,Q) or NAGARCH(P,Q) time series.

    Parameters
    ----------
    t : int or np.ndarray
        If scalar (int or length-1 array), specifies the number of observations
        to simulate. If a 1-D array with length > 1, the values are used
        directly as user-supplied standardized random innovations (replacing
        internal random number generation).
    parameters : np.ndarray
        Parameter vector of length ``2 + p + q + extrap`` where *extrap* depends
        on the error distribution:

        * ``'NORMAL'``: extrap = 0 → ``[omega, alpha(1..p), gamma, beta(1..q)]``
        * ``'STUDENTST'``: extrap = 1 → ``[..., nu]``
        * ``'GED'``: extrap = 1 → ``[..., nu]``
        * ``'SKEWT'``: extrap = 2 → ``[..., nu, lambda]``

        Layout (0-indexed):

        - ``parameters[0]`` = omega  (intercept, > 0)
        - ``parameters[1:p+1]`` = alpha  (p ARCH coefficients, >= 0)
        - ``parameters[p+1]`` = gamma  (asymmetry parameter)
        - ``parameters[p+2:p+q+2]`` = beta  (q GARCH coefficients, >= 0)
        - ``parameters[p+q+2]`` = nu  (shape, for STUDENTST / GED / SKEWT)
        - ``parameters[p+q+3]`` = lambda  (skewness, for SKEWT only)
    p : int
        Positive integer — number of symmetric innovation (ARCH) lags.
    q : int
        Non-negative integer — number of conditional variance (GARCH) lags.
    model_type : str, optional
        Variance model specification. One of:

        * ``'AGARCH'``  — Asymmetric GARCH (Engle 1990) **[default]**
        * ``'NAGARCH'`` — Nonlinear Asymmetric GARCH (Engle & Ng 1993)
    error_type : str, optional
        Innovation distribution. One of:

        * ``'NORMAL'``    — Gaussian innovations **[default]**
        * ``'STUDENTST'`` — Standardized Student's t
        * ``'GED'``       — Generalized Error Distribution
        * ``'SKEWT'``     — Hansen's Skewed Student's t

    Returns
    -------
    simulatedata : np.ndarray
        1-D array of simulated returns of length equal to the requested *t*
        (scalar case) or ``len(t)`` (vector case).
    ht : np.ndarray
        1-D array of conditional variances corresponding to *simulatedata*.

    Raises
    ------
    ValueError
        If inputs fail any validation check (wrong parameter count, invalid
        model_type, invalid error_type, invalid p/q, etc.).

    Warns
    -----
    UserWarning
        When ``sum(alpha) + sum(beta) >= 1``, indicating the process lies
        in the non-stationary parameter space.

    See Also
    --------
    agarch : AGARCH/NAGARCH estimation driver.

    Examples
    --------
    Simulate 1000 observations from an AGARCH(1,1) with normal errors:

    >>> import numpy as np
    >>> params = np.array([0.01, 0.05, 0.02, 0.90])
    >>> sim, ht = agarch_simulate(1000, params, 1, 1)
    >>> sim.shape
    (1000,)
    >>> ht.shape
    (1000,)
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: agarch_simulate.m:55-100
    # ------------------------------------------------------------------

    # Validate error_type and determine number of extra distribution params
    # Ref: agarch_simulate.m:66-80
    if error_type is None or error_type == '':
        error_type = 'NORMAL'

    error_type_upper = error_type.upper()
    if error_type_upper == 'NORMAL':
        extrap = 0
    elif error_type_upper == 'STUDENTST':
        extrap = 1
    elif error_type_upper == 'GED':
        extrap = 1
    elif error_type_upper == 'SKEWT':
        extrap = 2
    else:
        raise ValueError('Unknown error type')

    # Validate model_type
    # Ref: agarch_simulate.m:82-84
    if model_type is None or model_type == '':
        model_type = 'AGARCH'

    model_type_upper = model_type.upper()
    if model_type_upper not in ('AGARCH', 'NAGARCH'):
        raise ValueError("MODEL_TYPE must be either 'AGARCH' or 'NAGARCH'")

    # Validate p and q
    # Ref: agarch_simulate.m:94-96
    if not np.isscalar(p) or not np.isscalar(q):
        raise ValueError(
            'P and Q must be scalars with P positive and Q non-negative'
        )
    p = int(p)
    q = int(q)
    if p < 1 or q < 0:
        raise ValueError(
            'P and Q must be scalars with P positive and Q non-negative'
        )

    # Convert and validate parameters
    # Ref: agarch_simulate.m:86-88
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    expected_len = 2 + p + q + extrap
    if len(parameters) != expected_len:
        raise ValueError(
            'PARAMETERS must be a column vector with the correct number of '
            'parameters.'
        )

    # Stationarity check: sum(alpha) + sum(beta) >= 1
    # Ref: agarch_simulate.m:90-92
    # alpha occupies indices 1..p (0-based), beta occupies indices p+2..p+q+1 (0-based)
    alpha_sum = np.sum(parameters[1 : p + 1])
    beta_sum = np.sum(parameters[p + 2 : p + q + 2])
    if (alpha_sum + beta_sum) >= 1.0:
        warnings.warn(
            'PARAMETERS are in the non-stationary space, be sure to check '
            'that H is not inf.',
            stacklevel=2,
        )

    # Validate t shape
    # Ref: agarch_simulate.m:98-100
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim > 1:
        # Ref: agarch_simulate.m:98 — size(t,2)~=1 check for column-vector compliance
        if t_arr.ndim == 2 and t_arr.shape[1] == 1:
            t_arr = t_arr.ravel()
        else:
            raise ValueError(
                'T must be either a positive scalar or a vector of random '
                'numbers.'
            )

    # ------------------------------------------------------------------
    # Extract model parameters
    # Ref: agarch_simulate.m:106-109 — 0-based indexing adjustment
    # ------------------------------------------------------------------
    omega = parameters[0]
    # Ref: agarch_simulate.m:107 — alpha=parameters(2:p+1) (MATLAB 1-based)
    alpha = parameters[1 : p + 1]
    # Ref: agarch_simulate.m:108 — gamma=parameters(p+2) (MATLAB 1-based)
    gamma_param = parameters[p + 1]
    # Ref: agarch_simulate.m:109 — beta=parameters(p+3:p+q+2) (MATLAB 1-based)
    beta = parameters[p + 2 : p + q + 2]

    # ------------------------------------------------------------------
    # Random number generation
    # Ref: agarch_simulate.m:112-135
    # ------------------------------------------------------------------
    rng = np.random.default_rng()

    # Determine if t is scalar (simulation length) or vector (user-supplied RNs)
    if t_arr.size == 1:
        # Scalar case: t specifies number of observations to generate
        # Ref: agarch_simulate.m:113-128 — add burn-in and generate RNs
        t_len = int(t_arr.item()) + _BURN_IN
        if error_type_upper == 'NORMAL':
            # Ref: agarch_simulate.m:115 — RandomNums=randn(t,1)
            random_nums = rng.standard_normal(t_len)
        elif error_type_upper == 'STUDENTST':
            # Ref: agarch_simulate.m:117-118 — nu=parameters(3+p+q)
            nu = parameters[2 + p + q]
            random_nums = stdtrnd(nu, t_len, rng=rng).ravel()
        elif error_type_upper == 'GED':
            # Ref: agarch_simulate.m:119-121 — nu=parameters(3+p+q)
            nu = parameters[2 + p + q]
            random_nums = gedrnd(nu, t_len, rng=rng).ravel()
        elif error_type_upper == 'SKEWT':
            # Ref: agarch_simulate.m:122-125 — nu, lambda
            nu = parameters[2 + p + q]
            lambda_ = parameters[3 + p + q]
            random_nums = skewtrnd(nu, lambda_, t_len, rng=rng).ravel()
        else:
            # Defensive guard — should never reach here after validation above
            raise ValueError('Unknown error type')
    else:
        # Vector case: t provides user-supplied random numbers
        # Ref: agarch_simulate.m:130-135
        random_nums = t_arr.ravel().copy()
        orig_len = len(random_nums)
        # Ref: agarch_simulate.m:132 — seeds=ceil(rand(2000,1)*t)
        # MATLAB ceil(rand*n) gives 1..n (1-based); Python integers 0..n-1 (0-based)
        seeds = rng.integers(0, orig_len, size=_BURN_IN)
        # Ref: agarch_simulate.m:133 — RandomNums=[RandomNums(seeds);RandomNums]
        random_nums = np.concatenate([random_nums[seeds], random_nums])
        t_len = len(random_nums)

    # ------------------------------------------------------------------
    # Prepare arrays for the recursion
    # Ref: agarch_simulate.m:137-151
    # ------------------------------------------------------------------
    m = max(p, q)

    # Ref: agarch_simulate.m:139 — RandomNums=[zeros(m,1);RandomNums]
    random_nums = np.concatenate([np.zeros(m), random_nums])

    # Compute unconditional standard deviation
    # Ref: agarch_simulate.m:141-146
    persistence = np.sum(alpha) + np.sum(beta)
    if (1.0 - persistence) > 0.0:
        # Ref: agarch_simulate.m:143 — approximate unconditional std
        uncond_std = np.sqrt(omega / (1.0 - persistence))
    else:
        # Non-stationary case
        uncond_std = 1.0

    total_len = t_len + m

    # Ref: agarch_simulate.m:148-151 — initialize h, data, shock arrays
    h = uncond_std ** 2 * np.ones(total_len)
    data = uncond_std * np.ones(total_len)
    shock = np.zeros(total_len)

    # ------------------------------------------------------------------
    # Repack parameter vector for the recursion dot product
    # Ref: agarch_simulate.m:154 — parameters=[omega;alpha;beta]
    # This is used in: h(t) = [omega alpha beta] * [1; shock_lags; h_lags]
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # AGARCH / NAGARCH variance recursion
    # Ref: agarch_simulate.m:155-173
    #
    # MATLAB loop runs t=(m+1):T with 1-based indexing.
    # Python equivalent: i in range(m, total_len) with 0-based indexing.
    # ------------------------------------------------------------------
    if model_type_upper == 'AGARCH':
        # Initialize shock for the first m periods
        # Ref: agarch_simulate.m:156-158
        for i in range(m):
            shock[i] = (uncond_std - gamma_param) ** 2

        # Main recursion
        # Ref: agarch_simulate.m:159-163
        for i in range(m, total_len):
            # h(t) = omega + sum_j(alpha_j * shock(t-j)) + sum_j(beta_j * h(t-j))
            val = omega
            for j in range(p):
                # Ref: agarch_simulate.m:160 — shock(t-(1:p)) → shock[i-1], shock[i-2], ...
                val += alpha[j] * shock[i - 1 - j]
            for j in range(q):
                # Ref: agarch_simulate.m:160 — h(t-(1:q)) → h[i-1], h[i-2], ...
                val += beta[j] * h[i - 1 - j]
            h[i] = val
            # Ref: agarch_simulate.m:161 — data(t)=RandomNums(t)*sqrt(h(t))
            data[i] = random_nums[i] * np.sqrt(h[i])
            # Ref: agarch_simulate.m:162 — shock(t)=(data(t)-gamma)^2
            shock[i] = (data[i] - gamma_param) ** 2
    else:
        # NAGARCH model
        # Initialize shock for the first m periods
        # Ref: agarch_simulate.m:165-167
        for i in range(m):
            shock[i] = (uncond_std - gamma_param * uncond_std) ** 2

        # Main recursion
        # Ref: agarch_simulate.m:168-172
        for i in range(m, total_len):
            val = omega
            for j in range(p):
                val += alpha[j] * shock[i - 1 - j]
            for j in range(q):
                val += beta[j] * h[i - 1 - j]
            h[i] = val
            # Ref: agarch_simulate.m:170 — data(t)=RandomNums(t)*sqrt(h(t))
            data[i] = random_nums[i] * np.sqrt(h[i])
            # Ref: agarch_simulate.m:171 — shock(t)=(data(t)-gamma*sqrt(h(t)))^2
            shock[i] = (data[i] - gamma_param * np.sqrt(h[i])) ** 2

    # ------------------------------------------------------------------
    # Output: discard burn-in and prepended zeros
    # Ref: agarch_simulate.m:174-175
    # MATLAB: simulatedata=data((m+1+2000):T)  (1-based, inclusive)
    # Python: data[m + 2000 : total_len]        (0-based, exclusive end)
    # ------------------------------------------------------------------
    simulatedata = data[m + _BURN_IN:]
    ht = h[m + _BURN_IN:]

    return simulatedata, ht
