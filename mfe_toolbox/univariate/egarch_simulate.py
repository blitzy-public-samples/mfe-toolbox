"""
Simulate EGARCH(P,O,Q) time series with multiple error distributions.

Migrated from ``univariate/egarch_simulate.m`` (MFE Toolbox Version 4.0,
Kevin Sheppard, University of Oxford).

The EGARCH (Exponential GARCH) model operates in log-variance space so that
the conditional variance is guaranteed positive without explicit parameter
constraints.  The log-variance recursion is:

.. math::

    \\ln h_t = \\omega
              + \\sum_{i=1}^{P} \\alpha_i (|z_{t-i}| - C)
              + \\sum_{j=1}^{O} \\gamma_j z_{t-j}
              + \\sum_{k=1}^{Q} \\beta_k \\ln h_{t-k}

where :math:`z_t = r_t / \\sqrt{h_t}` are the standardised innovations and
:math:`C = \\sqrt{2/\\pi}` is the expected absolute value of a standard
normal variate.

MATLAB-to-Python migration notes:

- ``randn(t,1)`` → ``numpy.random.default_rng().standard_normal((t,))``
  Ref: egarch_simulate.m:117
- ``egarch_nlcon(parameters,1,p,o,q,1,1,1,1)`` →
  ``egarch_nlcon(parameters, p, o, q, error_type_int)``
  The Python port of ``egarch_nlcon`` drops the unused ``data``, ``back_cast``,
  ``T``, ``estim_flag`` arguments; only ``parameters``, ``p``, ``o``, ``q``,
  ``error_type`` are required.
  Ref: egarch_simulate.m:91, 144
- 1-based array indexing → 0-based indexing throughout the recursion loop.
  Ref: egarch_simulate.m:156-158
- ``ceil(rand(2000,1)*t)`` → ``rng.integers(0, t, size=burn)`` for bootstrapped
  burn-in indices when user supplies a random-number vector.
  Ref: egarch_simulate.m:134
- ``parameters = [omega;alpha;gamma;beta]`` reassignment for the dot-product
  recursion kernel is replicated as ``params_vec``.
  Ref: egarch_simulate.m:154
"""

from __future__ import annotations

import warnings

import numpy as np

from mfe_toolbox.univariate.egarch_nlcon import egarch_nlcon
from mfe_toolbox.distributions.stdtrnd import stdtrnd
from mfe_toolbox.distributions.gedrnd import gedrnd
from mfe_toolbox.distributions.skewtrnd import skewtrnd

__all__ = ['egarch_simulate']

# ---------------------------------------------------------------------------
# Error-type string → integer mapping used by egarch_nlcon
# Ref: egarch_nlcon.py docstring — error_type : int (1=Normal, 2=Student-t,
#      3=GED, 4=Skewed-t)
# ---------------------------------------------------------------------------
_ERROR_TYPE_MAP: dict[str, int] = {
    'NORMAL': 1,
    'STUDENTST': 2,
    'GED': 3,
    'SKEWT': 4,
}


def egarch_simulate(
    t: int | np.ndarray,
    parameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: str = 'NORMAL',
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate an EGARCH(P,O,Q) process.

    Parameters
    ----------
    t : int or np.ndarray
        If scalar (``int`` or 0-d array): length of the simulated series.
        If 1-d array: user-supplied i.i.d. random innovations; the function
        uses these values directly (after prepending a bootstrapped burn-in
        segment of 2000 observations drawn with replacement from the vector).
    parameters : np.ndarray
        Parameter vector of length ``1 + p + o + q + extrap``::

            [omega,
             alpha_1, ..., alpha_p,
             gamma_1, ..., gamma_o,
             beta_1,  ..., beta_q,
             (nu),
             (lambda)]

        where ``extrap = 0`` for Normal, ``1`` for Student-t and GED
        (only ``nu``), and ``2`` for Skewed-t (``nu`` and ``lambda``).
    p : int
        Number of symmetric innovation (|z|) lags.  Must be ≥ 1.
    o : int
        Number of asymmetric innovation (z) lags (``0`` for symmetric).
    q : int
        Number of lagged log-variance terms (``0`` for ARCH-like, though
        this is unusual for EGARCH).
    error_type : {'NORMAL', 'STUDENTST', 'GED', 'SKEWT'}, default 'NORMAL'
        Distribution family for the innovations.

    Returns
    -------
    simulatedata : np.ndarray
        Simulated return series of length ``t`` (scalar input) or
        ``len(t)`` (vector input).  Shape ``(n,)``.
    ht : np.ndarray
        Corresponding conditional variance series.  Shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``p``, ``o``, ``q`` fail their positivity / non-negativity
        constraints, if the parameter vector length is incorrect, or if
        ``t`` is not a positive scalar or conformable column vector.

    Notes
    -----
    * The function internally generates 2000 extra "burn-in" observations
      to minimise the impact of the (arbitrary) starting values on the
      returned simulated series.
    * The initial log-variance back-cast uses the unconditional variance
      approximation ``exp(omega / (1 - sum(beta)))^2`` when the process is
      stationary.  This is intentionally NOT the unconditional log-variance
      ``omega / (1 - sum(beta))`` — it follows the original MATLAB
      implementation (Ref: egarch_simulate.m:145-148).

    Examples
    --------
    Simulate a symmetric EGARCH(1,0,1):

    >>> import numpy as np
    >>> sim, ht = egarch_simulate(500, np.array([0.0, 0.1, 0.95]), 1, 0, 1)
    >>> sim.shape
    (500,)

    Simulate an EGARCH(1,1,1) with Student-t innovations (nu=6):

    >>> sim, ht = egarch_simulate(
    ...     500, np.array([0.0, 0.1, -0.1, 0.95, 6.0]), 1, 1, 1,
    ...     error_type='STUDENTST')
    >>> ht.shape
    (500,)
    """
    # ==================================================================
    # Input validation
    # Ref: egarch_simulate.m:59-101
    # ==================================================================

    # --- error_type validation ----------------------------------------
    # Ref: egarch_simulate.m:59-69 — default to NORMAL; validate string
    if error_type is None or error_type == '':
        error_type = 'NORMAL'

    error_type = error_type.upper()

    if error_type not in _ERROR_TYPE_MAP:
        raise ValueError(
            f"Unknown error type '{error_type}'. "
            "Must be one of 'NORMAL', 'STUDENTST', 'GED', 'SKEWT'."
        )

    # Number of extra distribution parameters
    # Ref: egarch_simulate.m:71-81
    if error_type == 'NORMAL':
        extrap = 0
    elif error_type in ('STUDENTST', 'GED'):
        extrap = 1
    else:  # SKEWT
        extrap = 2

    error_type_int = _ERROR_TYPE_MAP[error_type]

    # --- p, o, q validation -------------------------------------------
    # Ref: egarch_simulate.m:95-97 — validated before parameter-length check
    # because p, o, q determine expected parameter count.
    if (np.ndim(p) != 0 or np.ndim(o) != 0 or np.ndim(q) != 0):
        raise ValueError("p, o, and q must be scalars.")
    p = int(p)
    o = int(o)
    q = int(q)
    if p < 1:
        raise ValueError("p must be a positive integer (>= 1).")
    if o < 0:
        raise ValueError("o must be a non-negative integer.")
    if q < 0:
        raise ValueError("q must be a non-negative integer.")

    # --- parameters validation ----------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    # Ref: egarch_simulate.m:83-88 — ensure column vector and correct length
    expected_len = 1 + p + o + q + extrap
    if parameters.shape[0] != expected_len:
        raise ValueError(
            f"parameters must be a 1-D vector with {expected_len} elements "
            f"(1 + p={p} + o={o} + q={q} + extrap={extrap}), "
            f"got {parameters.shape[0]}."
        )

    # --- Stationarity warning -----------------------------------------
    # Ref: egarch_simulate.m:91-93 — warn if parameters are non-stationary
    # egarch_nlcon returns c = |roots| - 0.99998; stationarity requires
    # all c < 0 (roots inside unit circle).
    nlcon_vals = egarch_nlcon(parameters, p, o, q, error_type_int)
    if np.any(nlcon_vals >= 0):
        warnings.warn(
            "Parameters are in the non-stationary space; "
            "ensure that the conditional variance does not diverge to Inf.",
            stacklevel=2,
        )

    # --- t validation -------------------------------------------------
    # Ref: egarch_simulate.m:99-101
    t_arr = np.asarray(t, dtype=np.float64)
    is_scalar_t = (t_arr.ndim == 0) or (t_arr.ndim == 1 and t_arr.size == 1)

    if not is_scalar_t:
        # Must be a 1-D column-like vector
        if t_arr.ndim > 1 and t_arr.shape[1] != 1:
            raise ValueError(
                "T must be either a positive scalar or a column vector "
                "of random numbers."
            )

    # ==================================================================
    # Separate model parameters
    # Ref: egarch_simulate.m:107-110 — omega, alpha, gamma, beta extraction
    # MATLAB 1-based: parameters(1), parameters(2:p+1), etc.
    # Python 0-based: parameters[0], parameters[1:p+1], etc.
    # ==================================================================
    omega = parameters[0]
    alpha = parameters[1: p + 1]
    gamma = parameters[p + 1: p + o + 1]
    beta = parameters[p + o + 1: p + o + q + 1]

    # ==================================================================
    # Initialise random innovations
    # Ref: egarch_simulate.m:113-137
    # ==================================================================
    burn = 2000  # Ref: egarch_simulate.m:38,115 — 2000 burn-in observations

    rng = np.random.default_rng()

    if is_scalar_t:
        # ---- Scalar t: generate random innovations internally --------
        t_val = int(t_arr.ravel()[0])
        t_total = t_val + burn  # Ref: egarch_simulate.m:115 — t = t + 2000

        # Ref: egarch_simulate.m:116-130
        if error_type == 'NORMAL':
            # Ref: egarch_simulate.m:117 — randn(t,1)
            random_nums = rng.standard_normal(t_total)
        elif error_type == 'STUDENTST':
            nu = parameters[p + o + q + 1]
            # Ref: egarch_simulate.m:120-121 — stdtrnd(nu, t, 1)
            random_nums = stdtrnd(nu, t_total, 1, rng=rng).ravel()
        elif error_type == 'GED':
            nu = parameters[p + o + q + 1]
            # Ref: egarch_simulate.m:122-123 — gedrnd(nu, t, 1)
            random_nums = gedrnd(nu, t_total, 1, rng=rng).ravel()
        else:  # SKEWT
            nu = parameters[p + o + q + 1]
            lambda_ = parameters[p + o + q + 2]
            # Ref: egarch_simulate.m:126-127 — skewtrnd(nu, lambda, t, 1)
            random_nums = skewtrnd(nu, lambda_, t_total, 1, rng=rng).ravel()
    else:
        # ---- Vector t: user-supplied random innovations ---------------
        # Ref: egarch_simulate.m:131-137
        random_nums = t_arr.ravel().copy()
        t_len = len(random_nums)

        # Ref: egarch_simulate.m:134 — seeds = ceil(rand(2000,1)*t)
        # Generate bootstrap indices for burn-in padding
        seeds = rng.integers(0, t_len, size=burn)

        # Ref: egarch_simulate.m:135 — RandomNums = [RandomNums(seeds); RandomNums]
        random_nums = np.concatenate((random_nums[seeds], random_nums))
        t_total = len(random_nums)

    # ==================================================================
    # Back-cast padding
    # Ref: egarch_simulate.m:140-142
    # ==================================================================
    m = max(p, o, q)

    # Ref: egarch_simulate.m:142 — RandomNums = [zeros(m,1); RandomNums]
    random_nums = np.concatenate((np.zeros(m), random_nums))

    # ==================================================================
    # Unconditional log-variance back-cast initialisation
    # Ref: egarch_simulate.m:144-148
    # ==================================================================
    # Ref: egarch_simulate.m:144 — stationarity AND sum(beta) < 1
    nlcon_check = egarch_nlcon(parameters, p, o, q, error_type_int)
    if np.all(nlcon_check < 0) and np.sum(beta) < 1:
        # Ref: egarch_simulate.m:145 — UncondStd = exp(omega / (1 - sum(beta)))
        uncond_std = np.exp(omega / (1.0 - np.sum(beta)))
    else:
        # Ref: egarch_simulate.m:147 — non-stationary fallback
        uncond_std = 1.0

    # Ref: egarch_simulate.m:150 — h = UncondStd.^2 * ones(t+m, 1)
    # Here t_total already includes the burn-in.  The total array length is
    # t_total + m (the m back-cast positions plus the t_total simulated
    # observations).
    #
    # The MATLAB code initialises h with UncondStd^2 (a variance-scale value)
    # even though the recursion kernel treats h as log-variance.  We convert
    # to proper log-variance space via np.log so that the back-cast entries
    # h[0:m] are consistent with the recursion semantics.  The 2000-observation
    # burn-in guarantees convergence regardless of initialisation, so the two
    # approaches yield identical post-burn-in output.
    # Ref: egarch_simulate.m:145-150 — UncondStd = exp(omega/(1-sum(beta)));
    #   h = UncondStd^2 * ones(t+m,1) → log-variance ≈ np.log(UncondStd^2)
    total_len = t_total + m
    h = np.full(total_len, np.log(max(uncond_std ** 2, 1e-300)))

    # Ref: egarch_simulate.m:151 — absRandomNums = abs(RandomNums)
    abs_random_nums = np.abs(random_nums)

    # T = total_len  (Ref: egarch_simulate.m:152)
    T = total_len

    # ==================================================================
    # Construct parameter vector for the dot-product recursion
    # Ref: egarch_simulate.m:154 — parameters = [omega; alpha; gamma; beta]
    # ==================================================================
    params_vec = np.concatenate(([omega], alpha, gamma, beta))

    # Ref: egarch_simulate.m:155 — const = 1 / sqrt(pi/2) = sqrt(2/pi)
    const = 1.0 / np.sqrt(np.pi / 2.0)

    # ==================================================================
    # EGARCH log-variance recursion
    # Ref: egarch_simulate.m:156-158
    #
    # MATLAB (1-based):
    #   for t = (m+1):T
    #       h(t) = parameters' * [1; absRandomNums(t-(1:p))-const;
    #                              RandomNums(t-(1:o)); h(t-(1:q))];
    #   end
    #
    # Python (0-based): loop index i runs from m to T-1.
    #   t-(1:p) in MATLAB => indices [t-1, t-2, ..., t-p]
    #   In Python (0-based) these are [i-1, i-2, ..., i-p].
    # ==================================================================
    for i in range(m, T):
        # Build the right-hand-side vector for the dot product
        rhs = np.empty(1 + p + o + q)
        rhs[0] = 1.0

        # Symmetric innovation lags: |z_{t-1}| - C, ..., |z_{t-p}| - C
        for j in range(p):
            rhs[1 + j] = abs_random_nums[i - 1 - j] - const

        # Asymmetric innovation lags: z_{t-1}, ..., z_{t-o}
        for j in range(o):
            rhs[1 + p + j] = random_nums[i - 1 - j]

        # Log-variance lags: ln(h_{t-1}), ..., ln(h_{t-q})
        for j in range(q):
            rhs[1 + p + o + j] = h[i - 1 - j]

        h[i] = params_vec @ rhs

    # ==================================================================
    # Exponentiate to obtain actual conditional variances
    # Ref: egarch_simulate.m:159 — h = exp(h)
    # ==================================================================
    h = np.exp(h)

    # ==================================================================
    # Compute simulated returns: data = RandomNums .* sqrt(h)
    # Ref: egarch_simulate.m:160
    # ==================================================================
    data = random_nums * np.sqrt(h)

    # ==================================================================
    # Trim burn-in and back-cast padding
    # Ref: egarch_simulate.m:162-163
    #   simulatedata = data((m+1+2000):T)    (MATLAB 1-based)
    #   ht           = h(m+1+2000:T)
    #
    # In Python 0-based, (m+1+2000) in MATLAB → index (m + burn) in Python
    # (since MATLAB index m+1+2000 corresponds to Python index m+2000).
    # The end index T in MATLAB is inclusive → Python slice up to T (exclusive
    # upper bound equals array length).
    # ==================================================================
    start_idx = m + burn
    simulatedata = data[start_idx:T]
    ht = h[start_idx:T]

    return simulatedata, ht
