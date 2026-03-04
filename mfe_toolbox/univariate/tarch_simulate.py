"""
Simulate TARCH/GJR-GARCH(P,O,Q) time series with threshold asymmetry.

Migrated from univariate/tarch_simulate.m (MFE Toolbox Version 4.0).

The TARCH (Threshold ARCH) model, also known as the GJR-GARCH model,
captures asymmetric volatility clustering where negative returns
('bad news') have a different impact on future volatility than positive
returns ('good news').

The conditional variance h(t) of a TARCH(P,O,Q) process is modeled as:

    g(h(t)) = omega
            + alpha(1)*f(r_{t-1}) + ... + alpha(p)*f(r_{t-p})
            + gamma(1)*I(t-1)*f(r_{t-1}) + ... + gamma(o)*I(t-o)*f(r_{t-o})
            + beta(1)*g(h(t-1)) + ... + beta(q)*g(h(t-q))

where:
    f(x) = abs(x),  g(x) = sqrt(x)   if tarch_type='AVGARCH' (type 1)
    f(x) = x^2,     g(x) = x         if tarch_type='GARCH'   (type 2)

I(t) = 1 if r_t < 0 (negative return indicator), 0 otherwise.

This module generates 2000 extra observations beyond the requested length
to minimize starting bias, then trims the burn-in period before returning
the simulated series.

Author: Kevin Sheppard (original MATLAB)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005

MATLAB-to-Python migration notes:
- randn(t,1) → numpy.random.default_rng().standard_normal(t_total)
  Ref: tarch_simulate.m:118
- MATLAB 1-indexed loop 'for t=(m+1):T' → Python 0-indexed 'for i in range(m, total_len)'
  Ref: tarch_simulate.m:157,164
- MATLAB ceil(rand(2000,1)*t) for bootstrap seeds → rng.integers(0, t_len, size=2000)
  Ref: tarch_simulate.m:135 — MATLAB 1-indexed random ints; Python 0-indexed
- MATLAB warning('UCSD_GARCH:nonstationary',...) → warnings.warn(...)
  Ref: tarch_simulate.m:94
- MATLAB error(...) → raise ValueError(...)
  Ref: tarch_simulate.m throughout
- MATLAB tarch_type numeric 1/2 → Python string 'AVGARCH'/'GARCH' mapped to int
  Ref: tarch_simulate.m:81-83

See Also
--------
tarch : TARCH/GJR-GARCH estimation driver.
egarch_simulate : EGARCH simulation.
aparch_simulate : APARCH simulation.
"""

import warnings

import numpy

from mfe_toolbox.distributions.stdtrnd import stdtrnd
from mfe_toolbox.distributions.gedrnd import gedrnd
from mfe_toolbox.distributions.skewtrnd import skewtrnd

__all__ = ['tarch_simulate']


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BURN_IN = 2000  # Number of burn-in observations to discard


def tarch_simulate(
    t: int | numpy.ndarray,
    parameters: numpy.ndarray,
    p: int,
    o: int,
    q: int,
    tarch_type: str = 'GARCH',
    error_type: str = 'NORMAL',
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Simulate a TARCH/GJR-GARCH(P,O,Q) time series.

    Parameters
    ----------
    t : int or numpy.ndarray
        If scalar (int): number of time steps to simulate.  The function
        internally adds 2000 burn-in observations.
        If 1-D array: user-supplied random innovations.  A bootstrapped
        burn-in of 2000 observations is prepended automatically.
    parameters : numpy.ndarray
        Model parameter vector of length ``1 + p + o + q + extrap`` where
        *extrap* depends on ``error_type`` (0 for NORMAL, 1 for STUDENTST
        or GED, 2 for SKEWT)::

            [omega, alpha(1), ..., alpha(p),
             gamma(1), ..., gamma(o),
             beta(1), ..., beta(q),
             [nu, [lambda]]]

        - ``omega`` : positive constant (intercept)
        - ``alpha`` : symmetric innovation coefficients
        - ``gamma`` : asymmetric (threshold) innovation coefficients
        - ``beta``  : lagged conditional variance coefficients
        - ``nu``    : shape parameter (STUDENTST, GED, SKEWT only)
        - ``lambda``: skewness parameter (SKEWT only)
    p : int
        Number of symmetric innovation lags.  Must be >= 1.
    o : int
        Number of asymmetric (threshold) innovation lags.  Must be >= 0.
    q : int
        Number of lagged conditional variance terms.  Must be >= 0.
    tarch_type : str, optional
        Variance process type:

        - ``'GARCH'``   (default) — model evolves in squared values
        - ``'AVGARCH'`` — model evolves in absolute values
    error_type : str, optional
        Error distribution for innovations:

        - ``'NORMAL'``    (default) — Gaussian innovations
        - ``'STUDENTST'`` — standardized Student-t innovations
        - ``'GED'``       — Generalized Error Distribution innovations
        - ``'SKEWT'``     — Hansen's skewed-t innovations

    Returns
    -------
    simulatedata : numpy.ndarray
        Simulated return series of length ``t`` (burn-in removed).
    ht : numpy.ndarray
        Conditional variance series of length ``t`` (burn-in removed).

    Raises
    ------
    ValueError
        If any input fails validation (invalid error_type, tarch_type,
        parameter length mismatch, or invalid p/o/q).

    Warns
    -----
    UserWarning
        If the parameters imply a non-stationary process, i.e.
        ``sum(alpha) + 0.5 * sum(gamma) + sum(beta) >= 1``.

    Notes
    -----
    The function generates 2000 extra observations beyond the requested
    length to minimize any starting-value bias.  These observations are
    discarded before returning results.

    For the AVGARCH variant (``tarch_type='AVGARCH'``), the recursion
    operates on the conditional standard deviation internally and converts
    to variance in the final output so that ``ht`` is always the
    conditional variance regardless of model type.

    References
    ----------
    .. [1] Glosten, L.R., Jagannathan, R. and Runkle, D.E. (1993).
           "On the Relation between the Expected Value and the Volatility
           of the Nominal Excess Return on Stocks," *Journal of Finance*,
           48(5), 1779-1801.
    .. [2] Zakoian, J.-M. (1994). "Threshold Heteroskedastic Models,"
           *Journal of Economic Dynamics and Control*, 18(5), 931-955.

    Examples
    --------
    Simulate a TARCH(1,1,1) series with 1000 observations:

    >>> import numpy as np
    >>> params = np.array([0.01, 0.05, 0.04, 0.90])
    >>> data, ht = tarch_simulate(1000, params, 1, 1, 1)
    >>> data.shape
    (1000,)
    >>> ht.shape
    (1000,)
    """
    # ==================================================================
    # Phase 1: Input Validation
    # Ref: tarch_simulate.m:54-103
    # ==================================================================

    # --- Validate error_type and determine extra parameter count -------
    # Ref: tarch_simulate.m:65-79
    if error_type is None or (isinstance(error_type, str) and error_type.strip() == ''):
        error_type = 'NORMAL'
    if not isinstance(error_type, str):
        raise ValueError(
            "ERROR_TYPE must be a string: 'NORMAL', 'STUDENTST', 'GED', or 'SKEWT'."
        )
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
            "Unknown ERROR_TYPE. Must be one of 'NORMAL', 'STUDENTST', "
            "'GED', or 'SKEWT'."
        )

    # --- Map tarch_type string to internal integer ---------------------
    # Ref: tarch_simulate.m:81-83  —  MATLAB uses numeric 1 or 2
    # Python spec: 'AVGARCH' → 1, 'GARCH' → 2
    if isinstance(tarch_type, str):
        tarch_type_str = tarch_type.upper()
        if tarch_type_str == 'AVGARCH':
            tarch_type_int = 1
        elif tarch_type_str == 'GARCH':
            tarch_type_int = 2
        else:
            raise ValueError(
                "TARCH_TYPE must be 'AVGARCH' or 'GARCH'."
            )
    elif tarch_type in (1, 2):
        # Also accept numeric for direct MATLAB parity
        tarch_type_int = int(tarch_type)
    else:
        raise ValueError("TARCH_TYPE must be either 1, 2, 'AVGARCH', or 'GARCH'.")

    # --- Validate p, o, q scalar constraints ---------------------------
    # Ref: tarch_simulate.m:97-99
    if not (numpy.isscalar(p) and numpy.isscalar(o) and numpy.isscalar(q)):
        raise ValueError(
            "P, O and Q must be scalars with P positive and O and Q non-negative."
        )
    p = int(p)
    o = int(o)
    q = int(q)
    if p < 1 or o < 0 or q < 0:
        raise ValueError(
            "P, O and Q must be scalars with P positive and O and Q non-negative."
        )

    # --- Ensure parameters is a 1-D float64 array ---------------------
    # Ref: tarch_simulate.m:85-91
    parameters = numpy.asarray(parameters, dtype=numpy.float64)
    if parameters.ndim == 2:
        # Ref: tarch_simulate.m:85-87 — if row vector, transpose/flatten
        parameters = parameters.flatten()
    if parameters.ndim != 1:
        raise ValueError(
            "PARAMETERS must be a 1D vector with the correct number of "
            "parameters."
        )

    # --- Validate parameter vector length ------------------------------
    # Ref: tarch_simulate.m:89-91
    expected_len = 1 + p + o + q + extrap
    if parameters.shape[0] != expected_len:
        raise ValueError(
            "PARAMETERS must be a column vector with the correct number of "
            f"parameters.  Expected {expected_len}, got {parameters.shape[0]}."
        )

    # --- Extract model parameters --------------------------------------
    # Ref: tarch_simulate.m:108-112 (0-indexed slicing)
    omega = parameters[0]
    alpha = parameters[1:p + 1]                 # p elements
    gamma_params = parameters[p + 1:p + o + 1]  # o elements
    beta = parameters[p + o + 1:p + o + q + 1]  # q elements

    # --- Stationarity check --------------------------------------------
    # Ref: tarch_simulate.m:93-95
    # persistence = sum(alpha) + 0.5*sum(gamma) + sum(beta)
    persistence = (
        numpy.sum(alpha) + 0.5 * numpy.sum(gamma_params) + numpy.sum(beta)
    )
    if persistence >= 1.0:
        warnings.warn(
            "PARAMETERS are in the non-stationary space, be sure to check "
            "that H is not inf.",
            stacklevel=2,
        )

    # --- Validate t input shape ----------------------------------------
    # Ref: tarch_simulate.m:101-103
    t_input = numpy.asarray(t)
    if t_input.ndim == 2 and t_input.shape[1] != 1:
        raise ValueError(
            "T must be either a positive scalar or a vector of random numbers."
        )

    # ==================================================================
    # Phase 2: Random Number Generation
    # Ref: tarch_simulate.m:108-138
    # ==================================================================
    rng = numpy.random.default_rng()

    if numpy.isscalar(t) or (isinstance(t_input, numpy.ndarray) and t_input.ndim == 0):
        # t is a scalar — generate random numbers internally
        t_val = int(t)
        # Ref: tarch_simulate.m:116 — add 2000 burn-in observations
        t_total = t_val + _BURN_IN

        if error_type_upper == 'NORMAL':
            # Ref: tarch_simulate.m:118 — RandomNums=randn(t,1)
            random_nums = rng.standard_normal(t_total)
        elif error_type_upper == 'STUDENTST':
            # Ref: tarch_simulate.m:120-121
            nu = parameters[1 + p + o + q]
            random_nums = stdtrnd(nu, t_total, 1, rng=rng).ravel()
        elif error_type_upper == 'GED':
            # Ref: tarch_simulate.m:123-124
            nu = parameters[1 + p + o + q]
            random_nums = gedrnd(nu, t_total, 1, rng=rng).ravel()
        elif error_type_upper == 'SKEWT':
            # Ref: tarch_simulate.m:126-128
            nu = parameters[1 + p + o + q]
            lambda_ = parameters[2 + p + o + q]
            random_nums = skewtrnd(nu, lambda_, t_total, 1, rng=rng).ravel()
        else:
            # Should never reach here due to earlier validation
            raise ValueError("Unknown error type.")
    else:
        # t is a user-supplied random number vector
        # Ref: tarch_simulate.m:132-138
        random_nums = numpy.asarray(t, dtype=numpy.float64).flatten()
        t_len = len(random_nums)
        # Ref: tarch_simulate.m:135 — seeds=ceil(rand(2000,1)*t)
        # MATLAB: ceil(rand*n) gives 1..n; Python: integers(0, n) gives 0..n-1
        seeds = rng.integers(0, t_len, size=_BURN_IN)
        # Ref: tarch_simulate.m:136 — RandomNums=[RandomNums(seeds);RandomNums]
        random_nums = numpy.concatenate([random_nums[seeds], random_nums])
        t_total = len(random_nums)

    # ==================================================================
    # Phase 3: Simulation Recursion
    # Ref: tarch_simulate.m:140-169
    # ==================================================================

    # Maximum lag length for back-cast initialization
    # Ref: tarch_simulate.m:140 — m = max([p,o,q])
    m = max(p, o, q)

    # Prepend m zeros to random_nums for back-cast padding
    # Ref: tarch_simulate.m:142 — RandomNums=[zeros(m,1);RandomNums]
    random_nums = numpy.concatenate([numpy.zeros(m), random_nums])

    # Compute unconditional standard deviation for initialization
    # Ref: tarch_simulate.m:144-148
    stationarity_denom = (
        1.0 - numpy.sum(alpha) - numpy.sum(beta) - 0.5 * numpy.sum(gamma_params)
    )
    if stationarity_denom > 0:
        uncond_std = numpy.sqrt(omega / stationarity_denom)
    else:
        # Non-stationary case: use unit standard deviation
        uncond_std = 1.0

    # Total array length including back-cast padding
    total_len = t_total + m

    # Initialize arrays with unconditional values
    # Ref: tarch_simulate.m:150-153
    h = numpy.ones(total_len) * (uncond_std ** 2)     # conditional variance
    data = numpy.ones(total_len) * uncond_std          # simulated returns
    idata = numpy.zeros(total_len)                     # negative return indicator

    # --- TARCH recursion loop ------------------------------------------
    if tarch_type_int == 1:
        # AVGARCH mode: model evolves in absolute values
        # Ref: tarch_simulate.m:156-162
        # In this mode, h[i] represents the conditional standard deviation
        # during the recursion. After the loop, h is squared to obtain
        # the conditional variance for output.
        for i in range(m, total_len):
            # Ref: tarch_simulate.m:158 — inner product construction
            # h(t) = params' * [1; abs(data(t-(1:p)));
            #         Idata(t-(1:o)).*abs(data(t-(1:o))); h(t-(1:q))]
            # Python 0-indexed: data[i-1-j] for j in range(p)
            h_val = omega
            for j in range(p):
                # Ref: tarch_simulate.m:158 — alpha * |data(t-(j+1))|
                h_val += alpha[j] * numpy.abs(data[i - 1 - j])
            for j in range(o):
                # Ref: tarch_simulate.m:158 — gamma * I(t-(j+1)) * |data(t-(j+1))|
                h_val += gamma_params[j] * idata[i - 1 - j] * numpy.abs(
                    data[i - 1 - j]
                )
            for j in range(q):
                # Ref: tarch_simulate.m:158 — beta * h(t-(j+1))
                h_val += beta[j] * h[i - 1 - j]
            h[i] = h_val
            # Ref: tarch_simulate.m:159 — data(t)=RandomNums(t)*h(t)
            # h is the conditional standard deviation in AVGARCH mode
            data[i] = random_nums[i] * h[i]
            # Ref: tarch_simulate.m:160 — Idata(t)=data(t)<0
            idata[i] = 1.0 if data[i] < 0.0 else 0.0

        # Ref: tarch_simulate.m:162 — h=h.^2
        # Convert conditional standard deviation to variance for output
        h = h ** 2

    else:
        # GARCH mode (default): model evolves in squared values
        # Ref: tarch_simulate.m:163-169
        # In this mode, h[i] directly represents the conditional variance.
        for i in range(m, total_len):
            # Ref: tarch_simulate.m:165 — inner product construction
            # h(t) = params' * [1; data(t-(1:p)).^2;
            #         Idata(t-(1:o)).*data(t-(1:o)).^2; h(t-(1:q))]
            h_val = omega
            for j in range(p):
                # Ref: tarch_simulate.m:165 — alpha * data(t-(j+1))^2
                h_val += alpha[j] * data[i - 1 - j] ** 2
            for j in range(o):
                # Ref: tarch_simulate.m:165 — gamma * I(t-(j+1)) * data(t-(j+1))^2
                h_val += gamma_params[j] * idata[i - 1 - j] * data[i - 1 - j] ** 2
            for j in range(q):
                # Ref: tarch_simulate.m:165 — beta * h(t-(j+1))
                h_val += beta[j] * h[i - 1 - j]
            h[i] = h_val
            # Ref: tarch_simulate.m:166 — data(t)=RandomNums(t)*sqrt(h(t))
            data[i] = random_nums[i] * numpy.sqrt(h[i])
            # Ref: tarch_simulate.m:167 — Idata(t)=data(t)<0
            idata[i] = 1.0 if data[i] < 0.0 else 0.0

    # ==================================================================
    # Phase 4: Output — trim burn-in
    # Ref: tarch_simulate.m:170-171
    # MATLAB (1-indexed): simulatedata=data((m+1+2000):T); ht=h(m+1+2000:T);
    # Python (0-indexed): data[m+2000:], h[m+2000:]
    # ==================================================================
    simulatedata = data[m + _BURN_IN:]
    ht = h[m + _BURN_IN:]

    return simulatedata, ht
