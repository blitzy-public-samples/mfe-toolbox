"""
EGARCH log-variance recursion core.

Implements the EGARCH(P,O,Q) conditional variance recursion with Numba JIT
acceleration, replacing both the MATLAB fallback (univariate/egarch_core.m)
and the C MEX kernel (mex_source/egarch_core.c).

The conditional variance h(t) of an EGARCH(P,O,Q) process is modeled as:

    ln(h(t)) = omega
             + alpha(1)*(|e_{t-1}| - C) + ... + alpha(p)*(|e_{t-p}| - C)
             + gamma(1)*e_{t-1} + ... + gamma(o)*e_{t-o}
             + beta(1)*ln(h(t-1)) + ... + beta(q)*ln(h(t-q))

where C = sqrt(2/pi) is approximately 0.797884560802865, the expected value
of |Z| for standard normal Z, and e_t = data_t / sqrt(h_t) are standardized
residuals.

Sources:
    - univariate/egarch_core.m (MATLAB fallback, 90 lines)
    - mex_source/egarch_core.c (C MEX kernel, 120 lines — authoritative reference)

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005
Python migration by Blitzy
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def egarch_core(data, parameters, back_cast, upper, p, o, q, m, T):
    """
    Compute conditional variance for an EGARCH(P,O,Q) process.

    This function implements the EGARCH log-variance recursion using Numba JIT
    compilation for performance equivalent to the original C MEX kernel. The
    recursion computes log-variance as a linear function of past standardized
    residuals (symmetric and asymmetric components) and past log-variances,
    then exponentiates to obtain actual conditional variance.

    Parameters
    ----------
    data : numpy.ndarray
        A 1-D array of mean-zero data of length T.
    parameters : numpy.ndarray
        Parameter vector of length 1 + p + o + q, ordered as:
        [omega, alpha(1), ..., alpha(p), gamma(1), ..., gamma(o),
         beta(1), ..., beta(q)].
        - omega: constant in the log-variance equation
        - alpha: coefficients on |standardized residuals| - E[|Z|]
        - gamma: coefficients on standardized residuals (asymmetry)
        - beta: coefficients on lagged log-variance (persistence)
    back_cast : float
        Value used for initializing the log-variance recursion for
        the first m periods. Typically log(unconditional variance).
    upper : float
        Upper bound for conditional variance; values exceeding this
        bound are capped to prevent variance explosion.
    p : int
        Number of symmetric innovation terms (ARCH terms). Must be >= 1.
    o : int
        Number of asymmetric innovation terms. Set to 0 for symmetric
        EGARCH (no leverage effect).
    q : int
        Number of lagged log-variance terms (GARCH terms). Set to 0
        for a pure ARCH specification.
    m : int
        Number of back-cast initialization periods. Equals max(p, o, q).
    T : int
        Total length of the data array.

    Returns
    -------
    numpy.ndarray
        Conditional variance array of shape (T,). Contains actual
        variances (not log-variances), matching the MATLAB return
        convention where ht = exp(logHt).

    Notes
    -----
    The implementation follows the C MEX kernel (mex_source/egarch_core.c)
    as the authoritative reference for bound enforcement logic, which
    differs slightly from the MATLAB fallback (univariate/egarch_core.m)
    in the upper bound handling.

    Parameter indexing:
        parameters[0]           = omega
        parameters[1:p+1]       = alpha (symmetric ARCH coefficients)
        parameters[p+1:p+o+1]   = gamma (asymmetric coefficients)
        parameters[p+o+1:p+o+q+1] = beta (GARCH coefficients)
    """
    # Ref: egarch_core.c:16-20 — Allocate working arrays
    # ht stores actual conditional variance (exp of log-variance)
    ht = np.zeros(T)
    # logHt stores log conditional variance (the quantity being modeled)
    logHt = np.zeros(T)
    # stdData stores standardized residuals: data[t] / sqrt(ht[t])
    stdData = np.zeros(T)
    # absStdData stores |standardized residuals| - E[|Z|] for ARCH terms
    absStdData = np.zeros(T)

    # Ref: egarch_core.c:22 — Exponentiate back_cast for actual variance
    exp_back_cast = np.exp(back_cast)

    # Ref: egarch_core.c:23, egarch_core.m:50 — Expected value of |Z| for
    # standard normal Z: sqrt(2/pi) = 0.797884560802865. Hardcoded constant
    # for exact reproducibility within Numba nopython mode.
    subconst = 0.797884560802865

    # Ref: egarch_core.c:33-34 — Lower bounds to prevent variance underflow.
    # eLB is the lower bound for actual variance (exp(back_cast) / 10000).
    # hLB is the lower bound for log-variance (back_cast - log(10000)).
    eLB = exp_back_cast / 10000.0
    hLB = back_cast - np.log(10000.0)

    # Ref: egarch_core.c:24-32 — Initialization of back-cast values.
    # The first m periods use the back_cast value for log-variance, which
    # represents the unconditional log-variance estimate. Standardized
    # residuals are computed using the back-cast variance to provide
    # lagged values for the recursion.
    for j in range(m):
        logHt[j] = back_cast
        ht[j] = exp_back_cast
        vol = np.sqrt(ht[j])
        stdData[j] = data[j] / vol
        absStdData[j] = np.abs(stdData[j]) - subconst

    # Ref: egarch_core.c:35-69 — Main recursion loop.
    # For each time period from m to T-1, compute the log-variance as a
    # linear function of omega, past absolute standardized residuals
    # (symmetric ARCH effect), past standardized residuals (asymmetric
    # leverage effect), and past log-variances (GARCH persistence).
    for i in range(m, T):
        # Ref: egarch_core.c:36 — Omega (constant term in log-variance)
        logHt[i] = parameters[0]

        # Ref: egarch_core.c:37-39 — ARCH (symmetric |z|) terms.
        # alpha(j) * (|e_{t-1-j}| - sqrt(2/pi))
        # C code uses 0-based indexing: absStdData[i-1-j], which directly
        # maps to Python 0-based indexing.
        for j in range(p):
            logHt[i] += parameters[j + 1] * absStdData[i - 1 - j]

        # Ref: egarch_core.c:40-42 — Asymmetric z terms (leverage effect).
        # gamma(j) * e_{t-1-j}
        # These capture the asymmetric response to positive vs. negative
        # shocks. When gamma < 0, negative shocks increase volatility more
        # than positive shocks of equal magnitude.
        for j in range(o):
            logHt[i] += parameters[j + p + 1] * stdData[i - 1 - j]

        # Ref: egarch_core.c:43-45 — GARCH (log(h)) persistence terms.
        # beta(j) * ln(h_{t-1-j})
        # These capture the persistence of volatility clustering.
        for j in range(q):
            logHt[i] += parameters[j + p + o + 1] * logHt[i - 1 - j]

        # Ref: egarch_core.c:47 — Exponentiate log-variance to get actual
        # conditional variance. This is the key EGARCH feature: modeling
        # log(h_t) ensures h_t > 0 without requiring parameter constraints.
        ht[i] = np.exp(logHt[i])

        # Ref: egarch_core.c:48-52 — Lower bound enforcement.
        # Prevents variance from becoming too small (numerical underflow).
        # If actual variance falls below eLB, reset to the lower bound.
        if ht[i] < eLB:
            ht[i] = eLB
            logHt[i] = hLB

        # Ref: egarch_core.c:54-65 — Upper bound enforcement.
        # Prevents variance explosion during optimization. Follows C MEX
        # logic exactly, which differs from the MATLAB .m version.
        # C MEX is authoritative per migration specification.
        if ht[i] > upper:
            if np.isinf(ht[i]):
                # Ref: egarch_core.c:56-58 — Infinity: hard cap at upper
                ht[i] = upper
            else:
                # Ref: egarch_core.c:60-62 — Finite overflow: soft cap
                # using log-variance offset. This allows the variance to
                # exceed the upper bound by the log-variance amount, giving
                # the optimizer a gradient signal.
                ht[i] = upper + logHt[i]
            # Ref: egarch_core.c:64 — Reset log-variance to log(upper).
            # This prevents the log-variance feedback loop from diverging
            # in subsequent GARCH lag terms.
            logHt[i] = np.log(upper)

        # Ref: egarch_core.c:66-68 — Update standardized residuals.
        # These values feed into the next iteration's ARCH and asymmetric
        # terms via the lagged absStdData and stdData arrays.
        vol = np.sqrt(ht[i])
        stdData[i] = data[i] / vol
        absStdData[i] = np.abs(stdData[i]) - subconst

    # Ref: egarch_core.m:90 — Return actual conditional variance array.
    # MATLAB returns ht=eht (actual variance, not log-variance).
    return ht
