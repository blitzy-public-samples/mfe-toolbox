"""
ARMAX Filter Core Recursion — Inner-loop forward recursion for MA/MAX/ARMA/ARMAX estimation.

Migrated from timeseries/armaxfilter_core.m (Kevin Sheppard, Revision 3, 1/1/2007).
This is the M-file version of the inner loop used when the MEX file is not available.
The recursion computes ARMAX residuals with numerical clamping for stability and
returns both the error vector and sum-of-squared errors.

Notes
-----
This module provides a pure-Python (NumPy) implementation of the core ARMAX recursion.
For the Numba JIT-accelerated variant used in production, see
``mfe_toolbox.timeseries.armaxerrors``.

References
----------
- Source: timeseries/armaxfilter_core.m (MFE Toolbox Version 4.0, 28-Oct-2009)
- Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
"""

import numpy as np


def armaxfilter_core(
    regressand: np.ndarray,
    regressors: np.ndarray,
    arx_parameters: np.ndarray,
    ma_parameters: np.ndarray,
    q: np.ndarray,
    K: int,
    tau: int,
    maxq: int,
    maxe: float,
) -> tuple[np.ndarray, float]:
    """
    Compute the forward recursion for MA/MAX/ARMA/ARMAX error filtering.

    This function implements the inner-loop forward recursion for ARMAX estimation,
    computing the error vector and sum-of-squared errors. It is the Python translation
    of the M-file version of the ARMAX filter core from the MFE Toolbox.

    Parameters
    ----------
    regressand : numpy.ndarray
        T-length vector of dependent variable values (response variable).
    regressors : numpy.ndarray
        T x K matrix of regressor values (AR and exogenous variables).
        Each row corresponds to a time observation, each column to a regressor.
    arx_parameters : numpy.ndarray
        K-length vector of AR and exogenous regressor coefficients.
    ma_parameters : numpy.ndarray
        nq-length vector of MA (moving average) coefficients.
    q : numpy.ndarray
        nq-length vector of MA lag indices. Each element specifies how many
        periods back the corresponding MA coefficient applies (e.g., q=[1,2]
        means errors at lags 1 and 2).
    K : int
        Number of regressors (columns in regressors matrix). This equals
        the length of arx_parameters.
    tau : int
        Total number of observations (length of regressand/errors vectors).
    maxq : int
        Maximum lag index used for burn-in. The recursion starts at index
        maxq (0-indexed), leaving earlier entries as zero.
    maxe : float
        Maximum absolute error threshold for clamping. Errors exceeding
        100000 * maxe in absolute value are clamped to maxe * sign(error)
        for numerical stability.

    Returns
    -------
    errors : numpy.ndarray
        T-length vector of computed residuals. Elements before index maxq
        are zero (burn-in period).
    E : float
        Sum of squared errors from index maxq to tau-1 (inclusive),
        computed as ``errors[maxq:tau] @ errors[maxq:tau]``.

    Notes
    -----
    The recursion at each time step t (for t = maxq, ..., tau-1) is:

        errors[t] = regressand[t]
                    - sum_{i=0}^{K-1} arx_parameters[i] * regressors[t, i]
                    - sum_{i=0}^{nq-1} ma_parameters[i] * errors[t - q[i]]

    After computing each error, a clamping check is applied for numerical
    stability:

        if |errors[t]| > 100000 * maxe:
            errors[t] = maxe * sign(errors[t])

    The sum-of-squared errors E is computed only over the non-burn-in portion:

        E = dot(errors[maxq:tau], errors[maxq:tau])

    MATLAB to Python indexing notes:
    - Ref: armaxfilter_core.m:24 — MATLAB loop ``(1+maxq):tau`` uses 1-based
      indexing; Python ``range(maxq, tau)`` uses 0-based indexing, covering
      the same elements.
    - Ref: armaxfilter_core.m:36 — MATLAB ``errors(maxq+1:tau)`` (1-based)
      becomes ``errors[maxq:tau]`` (0-based), slicing the same elements.

    Examples
    --------
    >>> import numpy as np
    >>> T = 100
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal(T)
    >>> X = rng.standard_normal((T, 2))
    >>> arx_params = np.array([0.5, -0.3])
    >>> ma_params = np.array([0.2])
    >>> q_lags = np.array([1.0])
    >>> errors, E = armaxfilter_core(y, X, arx_params, ma_params, q_lags, 2, T, 1, 10.0)
    >>> errors.shape
    (100,)
    """
    # Validate inputs for robustness
    if tau <= 0:
        raise ValueError("tau must be a positive integer representing the number of observations.")
    if maxq < 0:
        raise ValueError("maxq must be a non-negative integer.")
    if maxq > tau:
        raise ValueError("maxq must not exceed tau.")
    if K < 0:
        raise ValueError("K must be a non-negative integer.")
    if maxe <= 0:
        raise ValueError("maxe must be a positive scalar for clamping threshold.")

    # Ref: armaxfilter_core.m:22 — zeros(tau,1) → np.zeros(tau) (1D vector)
    errors = np.zeros(tau)

    # Ref: armaxfilter_core.m:23 — size(q,1) → len(q) for number of MA lags
    nq = len(q)

    # Ref: armaxfilter_core.m:24-35 — Main recursion loop
    # MATLAB: for t=(1+maxq):tau   (1-based, inclusive on both ends)
    # Python: for t in range(maxq, tau)  (0-based, exclusive upper bound)
    # These iterate over the same elements because MATLAB's (1+maxq) in 1-based
    # corresponds to Python's maxq in 0-based indexing.
    for t in range(maxq, tau):
        # Ref: armaxfilter_core.m:25 — errors(t) = regressand(t)
        errors[t] = regressand[t]

        # Ref: armaxfilter_core.m:26-28 — AR/exogenous subtraction loop
        # MATLAB: for i=1:K → Python: for i in range(K) (same elements, 0-based)
        for i in range(K):
            # Ref: armaxfilter_core.m:27
            errors[t] -= arx_parameters[i] * regressors[t, i]

        # Ref: armaxfilter_core.m:29-31 — MA error recursion loop
        # MATLAB: for i=1:nq → Python: for i in range(nq) (same elements, 0-based)
        for i in range(nq):
            # Ref: armaxfilter_core.m:30 — errors(t-q(i))
            # q stores lag distances (e.g., 1, 2, 3), so t-q[i] gives the index
            # of the lagged error. The lag arithmetic is identical regardless of
            # 0-based vs 1-based since q encodes relative offsets.
            # Cast q[i] to int since MATLAB stores all numbers as doubles.
            errors[t] -= ma_parameters[i] * errors[t - int(q[i])]

        # Ref: armaxfilter_core.m:32-34 — Numerical stability clamping
        # If the absolute error exceeds 100000 * maxe, clamp to maxe * sign(error).
        # This prevents numerical overflow during optimization iterations.
        if abs(errors[t]) > 100000.0 * maxe:
            # Ref: armaxfilter_core.m:33 — errors(t) = maxe * sign(errors(t))
            errors[t] = maxe * np.sign(errors[t])

    # Ref: armaxfilter_core.m:36 — Sum of squared errors
    # MATLAB: errors(maxq+1:tau)' * errors(maxq+1:tau)  (1-based inclusive slice)
    # Python: np.dot(errors[maxq:tau], errors[maxq:tau])  (0-based exclusive-end slice)
    # Both select exactly the same (tau - maxq) elements.
    E = np.dot(errors[maxq:tau], errors[maxq:tau])

    return errors, E
