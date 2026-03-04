"""
FIGARCH(Q,D,P) time series simulation with multiple error distributions.

Simulates a FIGARCH(Q,D,P) process (for P in {0,1} and Q in {0,1}) using
the truncated ARCH(infinity) weight representation.  The conditional
variance of a FIGARCH(1,d,1) process is modeled as:

    h(t) = omega + [1 - beta*L - phi*L*(1-L)^d] * epsilon^2(t) + beta * h(t-1)

which is estimated using the ARCH(infinity) representation:

    h(t) = omega + sum_{i=1}^{truncLag} lambda(i) * epsilon^2(t-i)

where lambda(i) is a function of the fractional differencing parameter d,
the AR parameter phi, and the MA parameter beta.  These weights are computed
by :func:`figarch_weights`.

Supported error distributions:
    - ``'NORMAL'``    — Gaussian innovations (default)
    - ``'STUDENTST'`` — Standardized Student's t distributed errors
    - ``'GED'``       — Generalized Error Distribution errors
    - ``'SKEWT'``     — Hansen's (1994) Skewed Student's t errors

Migrated from: univariate/figarch_simulate.m (Version 4.0, Kevin Sheppard)

See Also
--------
figarch : FIGARCH model driver
figarch_weights : FIGARCH ARCH(infinity) truncation weights
figarch_likelihood : FIGARCH log-likelihood
figarch_parameter_check : FIGARCH parameter validation
figarch_transform : FIGARCH parameter transformation
figarch_itransform : FIGARCH inverse parameter transformation
"""

# Copyright: Kevin Sheppard
# kevin.sheppard@economics.ox.ac.uk
# Revision: 1    Date: 7/18/2009
# Python migration: MATLAB-to-Python 3.12

import numpy as np

from mfe_toolbox.univariate.figarch_weights import figarch_weights
from mfe_toolbox.distributions.stdtrnd import stdtrnd
from mfe_toolbox.distributions.gedrnd import gedrnd
from mfe_toolbox.distributions.skewtrnd import skewtrnd


def figarch_simulate(
    t: int | np.ndarray,
    parameters: np.ndarray,
    p: int,
    q: int,
    error_type: str = 'NORMAL',
    truncLag: int = 1000,
    bcLength: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a FIGARCH(Q,D,P) time series using ARCH(infinity) truncated weights.

    Parameters
    ----------
    t : int or np.ndarray
        If scalar (int): the number of time-series observations to simulate.
        If array (np.ndarray): a T-element vector of user-supplied random
        innovations (e.g. ``numpy.random.default_rng().standard_normal(T)``).
        When an array is provided, its values are used directly as the
        innovation sequence and ``bcLength`` additional bootstrap-resampled
        values are prepended for burn-in bias reduction.
    parameters : np.ndarray
        Parameter vector whose length depends on the model specification
        and error distribution:

        * FIGARCH(0,d,0) NORMAL: ``[omega, d]``  (2 elements)
        * FIGARCH(1,d,0) NORMAL: ``[omega, phi, d]``  (3 elements)
        * FIGARCH(0,d,1) NORMAL: ``[omega, d, beta]``  (3 elements)
        * FIGARCH(1,d,1) NORMAL: ``[omega, phi, d, beta]``  (4 elements)

        For non-normal distributions, append extra shape parameters:

        * STUDENTST: append ``[nu]``  (degrees of freedom, nu > 2)
        * GED:       append ``[nu]``  (shape parameter, nu >= 1)
        * SKEWT:     append ``[nu, lambda_]``  (df and asymmetry parameter)

        The parameter constraints are:

        * omega > 0
        * phi >= 0, beta >= 0
        * phi + d - beta >= 0
        * phi < lambda_check[1] / lambda_check[0]  where
          ``lambda_check = figarch_weights([d], 0, 0, 2)``
    p : int
        0 or 1 indicating whether the autoregressive (phi) term is present.
    q : int
        0 or 1 indicating whether the moving average (beta) term is present.
    error_type : str, optional
        Innovation distribution.  One of ``'NORMAL'``, ``'STUDENTST'``,
        ``'GED'``, ``'SKEWT'``.  Default is ``'NORMAL'``.
    truncLag : int, optional
        Truncation lag for the ARCH(infinity) weight representation.
        Controls how many past squared innovations are used in the
        conditional variance recursion.  Default is 1000.
    bcLength : int, optional
        Number of extra burn-in observations generated (and later discarded)
        to reduce start-up bias in the conditional variance.
        Default is 1000.

    Returns
    -------
    simulatedata : np.ndarray
        Simulated FIGARCH time series of length equal to the original ``t``
        (scalar input) or ``len(t)`` (array input).  Shape is ``(T,)``.
    ht : np.ndarray
        Conditional variance series corresponding to ``simulatedata``.
        Shape is ``(T,)``.

    Raises
    ------
    ValueError
        If inputs violate parameter constraints, dimensions are incorrect,
        or the error distribution type is unrecognized.

    Notes
    -----
    The simulation uses a truncated ARCH(infinity) representation where the
    conditional variance at each time step depends on ``truncLag`` past
    values.  To reduce initialization bias, ``bcLength`` extra observations
    are generated and discarded from the beginning of the simulation.

    The FIGARCH weights ``lambda`` can be obtained separately by calling
    :func:`figarch_weights` with the model parameters.

    References
    ----------
    Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
    Fractionally integrated generalized autoregressive conditional
    heteroskedasticity.  Journal of Econometrics, 74(1), 3-30.

    Examples
    --------
    FIGARCH(0,d,0) simulation with 2500 observations:

    >>> import numpy as np
    >>> from mfe_toolbox.univariate.figarch_simulate import figarch_simulate
    >>> data, ht = figarch_simulate(2500, np.array([0.1, 0.42]), 0, 0)
    >>> data.shape
    (2500,)

    FIGARCH(1,d,1) simulation:

    >>> data, ht = figarch_simulate(2500, np.array([0.1, 0.1, 0.42, 0.4]), 1, 1)

    FIGARCH(0,d,0) with Student's t errors:

    >>> data, ht = figarch_simulate(
    ...     2500, np.array([0.1, 0.42, 6.0]), 0, 0, error_type='STUDENTST')

    FIGARCH(0,d,0) with user-supplied random numbers:

    >>> rng = np.random.default_rng(42)
    >>> innovations = rng.standard_normal(1000)
    >>> data, ht = figarch_simulate(innovations, np.array([0.1, 0.42]), 0, 0)
    >>> data.shape
    (1000,)
    """
    # ======================================================================
    # Input Validation
    # Ref: figarch_simulate.m:58-105 — Input Checking block
    # ======================================================================

    # --- Validate error_type and determine number of extra distribution params ---
    # Ref: figarch_simulate.m:77-91
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
        raise ValueError(
            f"Unknown error type '{error_type}'. "
            "Must be one of 'NORMAL', 'STUDENTST', 'GED', 'SKEWT'."
        )

    # --- Validate p and q ---
    # Ref: figarch_simulate.m:100-102
    # Validated before parameter length check since p and q determine expected length
    if p not in (0, 1):
        raise ValueError("P must be 0 or 1.")
    if q not in (0, 1):
        raise ValueError("Q must be 0 or 1.")

    # --- Validate and reshape parameters ---
    # Ref: figarch_simulate.m:93-98
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    expected_len = 2 + p + q + extrap
    if parameters.size != expected_len:
        raise ValueError(
            f"PARAMETERS must have {expected_len} elements for the specified "
            f"model (p={p}, q={q}) and error type ('{error_type_upper}'). "
            f"Got {parameters.size} elements."
        )

    # --- Validate t shape ---
    # Ref: figarch_simulate.m:104-106
    t_input = np.asarray(t)
    if t_input.ndim > 1:
        # Ref: figarch_simulate.m:104 — size(t,2)~=1 check; flatten for Python
        t_input = t_input.ravel()

    # ======================================================================
    # Parameter Extraction
    # Ref: figarch_simulate.m:108-121
    # ======================================================================

    # Ref: figarch_simulate.m:109 — omega is always the first parameter
    omega = parameters[0]

    # Ref: figarch_simulate.m:110-116 — extract phi and d based on p
    if p:
        # Ref: figarch_simulate.m:111-112 — MATLAB parameters(2), parameters(3)
        # Python 0-indexed: parameters[1], parameters[2]
        phi = parameters[1]
        d = parameters[2]
    else:
        phi = 0.0
        # Ref: figarch_simulate.m:115 — when no phi, d is at MATLAB index 2
        # Python 0-indexed: parameters[1]
        d = parameters[1]

    # Ref: figarch_simulate.m:117-121 — extract beta based on q
    if q:
        # Ref: figarch_simulate.m:118 — MATLAB parameters(3+p)
        # Python 0-indexed: parameters[2+p]
        beta = parameters[2 + p]
    else:
        beta = 0.0

    # ======================================================================
    # Parameter Constraint Validation
    # Ref: figarch_simulate.m:122-134
    # ======================================================================

    # Ref: figarch_simulate.m:122-124
    if omega < 0:
        raise ValueError("omega must be positive.")

    # Ref: figarch_simulate.m:125-127
    if phi < 0 or beta < 0:
        raise ValueError("phi and beta must be non-negative.")

    # Ref: figarch_simulate.m:128-130
    if phi + d - beta < 0:
        raise ValueError("phi + d - beta must be non-negative.")

    # Ref: figarch_simulate.m:131-134 — validate phi upper bound using figarch_weights
    # Compute initial weights with d only (p=0, q=0, 2 lags) for the check
    lam_check = figarch_weights(np.array([d]), 0, 0, 2)
    if lam_check[0] != 0.0 and phi > lam_check[1] / lam_check[0]:
        raise ValueError(
            "phi must be less than lambda(2)/lambda(1) from a call to "
            "figarch_weights(d, 0, 0, 2). "
            f"Got phi={phi}, lambda(2)/lambda(1)={lam_check[1] / lam_check[0]:.6f}."
        )

    # --- Handle empty/None truncLag and bcLength ---
    # Ref: figarch_simulate.m:136-142
    if truncLag is None or truncLag <= 0:
        truncLag = 1000
    if bcLength is None or bcLength <= 0:
        bcLength = 1000

    # ======================================================================
    # Random Number Generation
    # Ref: figarch_simulate.m:148-172
    # ======================================================================

    # Determine if t is a scalar (number of observations) or a vector (user innovations)
    t_arr = np.asarray(t_input, dtype=np.float64).ravel()

    # Ref: figarch_simulate.m:149 — isscalar(t) check
    if t_arr.ndim == 0 or t_arr.size == 1:
        # Scalar case: t is the desired series length
        # Ref: figarch_simulate.m:150 — t = t + bcLength
        original_t = int(t_arr.flat[0])
        if original_t <= 0:
            raise ValueError("T must be a positive scalar or a vector of random numbers.")
        t_len = original_t + bcLength

        # Generate random innovations based on error distribution
        rng = np.random.default_rng()
        if error_type_upper == 'NORMAL':
            # Ref: figarch_simulate.m:152 — RandomNums = randn(t, 1)
            random_nums = rng.standard_normal(t_len)
        elif error_type_upper == 'STUDENTST':
            # Ref: figarch_simulate.m:154-155 — nu=parameters(3+p+q); stdtrnd(nu,t,1)
            # Python 0-indexed: parameters[2+p+q]
            nu = parameters[2 + p + q]
            random_nums = stdtrnd(nu, t_len, 1, rng=rng).ravel()
        elif error_type_upper == 'GED':
            # Ref: figarch_simulate.m:157 — gedrnd(nu, t, 1)
            nu = parameters[2 + p + q]
            random_nums = gedrnd(nu, t_len, 1, rng=rng).ravel()
        elif error_type_upper == 'SKEWT':
            # Ref: figarch_simulate.m:160-162 — skewtrnd(nu, lambda, t, 1)
            nu = parameters[2 + p + q]
            skew_lambda = parameters[3 + p + q]
            random_nums = skewtrnd(nu, skew_lambda, t_len, 1, rng=rng).ravel()
        else:
            # Should not reach here due to earlier validation, but defensive
            raise ValueError(f"Unknown error type '{error_type_upper}'.")
    else:
        # Vector case: user-supplied random numbers
        # Ref: figarch_simulate.m:167-171
        random_nums_input = t_arr.copy()
        original_t = random_nums_input.size
        # Ref: figarch_simulate.m:169 — seeds = ceil(rand(bcLength,1)*t)
        # Generate bootstrap indices for burn-in prepending
        # Ref: MATLAB ceil(rand*N) produces integers 1..N; Python equivalent is randint(0,N)
        rng = np.random.default_rng()
        seeds = rng.integers(0, original_t, size=bcLength)
        # Ref: figarch_simulate.m:170 — RandomNums=[RandomNums(seeds);RandomNums]
        random_nums = np.concatenate([random_nums_input[seeds], random_nums_input])
        t_len = random_nums.size

    # ======================================================================
    # Compute FIGARCH ARCH(infinity) Weights
    # Ref: figarch_simulate.m:173 — lambda = figarch_weights(parameters(2:2+p+q),p,q,truncLag)
    # ======================================================================

    # Ref: figarch_simulate.m:173 — MATLAB parameters(2:2+p+q) is 1-indexed range
    # Python 0-indexed: parameters[1:2+p+q] gives elements at indices 1..(1+p+q)
    # which is (1+p+q) elements = the model params excluding omega and dist params
    weights_params = parameters[1:2 + p + q].copy()
    lam_weights = figarch_weights(weights_params, p, q, truncLag)

    # ======================================================================
    # Back-cast Initialization
    # Ref: figarch_simulate.m:175-179
    # ======================================================================

    lam_sum = np.sum(lam_weights)
    # Ref: figarch_simulate.m:176 — if sum(lambda)<1: backCast = omega/(1-sum(lambda))
    if lam_sum < 1.0:
        back_cast = omega / (1.0 - lam_sum)
    else:
        # Ref: figarch_simulate.m:178 — backCast = omega/.01
        back_cast = omega / 0.01

    # ======================================================================
    # Array Setup for Simulation
    # Ref: figarch_simulate.m:181-185
    # ======================================================================

    total_len = truncLag + t_len

    # Ref: figarch_simulate.m:181 — r2=[backCast*ones(truncLag,1);zeros(size(RandomNums))]
    # r2 stores epsilon^2 * h products; initialized with back-cast for history
    r2 = np.zeros(total_len)
    r2[:truncLag] = back_cast

    # Ref: figarch_simulate.m:182 — e2=[zeros(truncLag,1);RandomNums.^2]
    e2 = np.zeros(total_len)
    e2[truncLag:] = random_nums ** 2

    # Ref: figarch_simulate.m:183 — e=[zeros(truncLag,1);RandomNums]
    e = np.zeros(total_len)
    e[truncLag:] = random_nums

    # Ref: figarch_simulate.m:184 — data=[zeros(truncLag,1);zeros(size(RandomNums))]
    data = np.zeros(total_len)

    # Ref: figarch_simulate.m:185 — h = zeros(size(e2))
    h = np.zeros(total_len)

    # ======================================================================
    # FIGARCH Variance Recursion Loop
    # Ref: figarch_simulate.m:187-191
    #
    # MATLAB loop (1-indexed):
    #   for i = truncLag+1 : t+truncLag
    #       h(i) = omega + lambda' * r2(i-1:-1:i-truncLag)
    #       r2(i) = e2(i) * h(i)
    #       data(i) = e(i) * sqrt(h(i))
    #   end
    #
    # Python loop (0-indexed):
    #   for i in range(truncLag, truncLag + t_len):
    #       h[i] = omega + dot(lam_weights, r2[i-truncLag:i][::-1])
    #       r2[i] = e2[i] * h[i]
    #       data[i] = e[i] * sqrt(h[i])
    #
    # Optimization: pre-reverse the weight vector to avoid reversing the
    # r2 slice at each iteration.  Instead of dot(lam, r2[slice][::-1]),
    # compute dot(lam_rev, r2[slice]) which reads memory forward.
    # ======================================================================

    # Pre-reverse weights for efficient dot product in the loop
    lam_weights_rev = lam_weights[::-1].copy()

    for i in range(truncLag, truncLag + t_len):
        # Ref: figarch_simulate.m:188 — h(i) = omega + lambda'*r2(i-1:-1:i-truncLag)
        # Using reversed weights: dot(lam_rev, r2[i-truncLag:i])
        # r2[i-truncLag:i] has exactly truncLag elements
        h[i] = omega + np.dot(lam_weights_rev, r2[i - truncLag:i])

        # Ref: figarch_simulate.m:189 — r2(i) = e2(i) * h(i)
        r2[i] = e2[i] * h[i]

        # Ref: figarch_simulate.m:190 — data(i) = e(i) * sqrt(h(i))
        data[i] = e[i] * np.sqrt(h[i])

    # ======================================================================
    # Discard Burn-in and Extract Output
    # Ref: figarch_simulate.m:194-196
    #
    # MATLAB (1-indexed):
    #   tau = (truncLag+1+bcLength):(truncLag+t)
    #   simulatedata = data(tau)
    #   ht = h(tau)
    #
    # Python (0-indexed):
    #   start = truncLag + bcLength
    #   end   = truncLag + t_len
    #   (t_len - bcLength == original_t, which is the user-requested length)
    # ======================================================================

    start_idx = truncLag + bcLength
    end_idx = truncLag + t_len
    simulatedata = data[start_idx:end_idx]
    ht = h[start_idx:end_idx]

    return simulatedata, ht
