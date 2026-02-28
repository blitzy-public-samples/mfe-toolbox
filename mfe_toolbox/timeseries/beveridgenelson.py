"""
Beveridge-Nelson decomposition for a trending (I(1)) time series.

Uses the exact BN decomposition as presented in:

    Paul Newbold, "Precise and efficient computation of the Beveridge-Nelson
    decomposition of economic time series", Journal of Monetary Economics,
    Volume 26, Issue 3, December 1990, Pages 453-457.

This decomposition is based on a demeaned ARMA model for the short-run
dynamics. If constant == 1, this function will demean the data using the
model-implied long-run mean (the constant parameter divided by one minus
the sum of the AR coefficients).

Migrated from: timeseries/beveridgenelson.m (Kevin Sheppard, Revision 1, 11/1/2009)

See Also
--------
mfe_toolbox.timeseries.armaxfilter : ARMAX estimation driver for obtaining parameters
"""

import numpy as np


def _compute_arma_errors(parameters, p, q, y, m):
    """
    Compute ARMA residuals for use in the BN decomposition.

    This is a local implementation of the ARMA error computation matching the
    interface used by beveridgenelson.m line 119, equivalent to calling
    armaxerrors(parameters, p, q, 0, y, [], m, ones(size(y))).

    Translated from: timeseries/armaxerrors.m lines 30-53

    Parameters
    ----------
    parameters : numpy.ndarray
        1D array of ARMA parameters: [AR_1, ..., AR_np, MA_1, ..., MA_nq].
        No constant parameter is included (constant=0 in the call).
    p : numpy.ndarray
        Integer array of AR lag indices (e.g., [1] for AR(1), [1, 3] for lags 1 and 3).
        Values represent lag offsets, not 0-based array indices.
    q : numpy.ndarray
        Integer array of MA lag indices (e.g., [1, 2] for MA(2)).
        Values represent lag offsets, not 0-based array indices.
    y : numpy.ndarray
        1D data array (possibly zero-padded at the front).
    m : int
        Number of presample observations to skip. Errors for indices 0..m-1 are zero.

    Returns
    -------
    numpy.ndarray
        1D array of errors, same length as y. First m elements are zero.
    """
    np_ = len(p)
    nq = len(q)
    T = len(y)
    errors = np.zeros(T, dtype=np.float64)

    # Ref: armaxerrors.m:37-52 — Main recursion loop (0-based Python equivalent)
    # MATLAB loop: for t=m+1:T → Python: for t in range(m, T)
    for t in range(m, T):
        errors[t] = y[t]
        # AR terms: Ref armaxerrors.m:42-44
        for i in range(np_):
            lag = int(p[i])
            errors[t] -= parameters[i] * y[t - lag]
        # MA terms: Ref armaxerrors.m:48-50
        for i in range(nq):
            lag = int(q[i])
            errors[t] -= parameters[np_ + i] * errors[t - lag]

    # Ref: armaxerrors.m:53 — sigma normalization (sigma=ones here, so no-op)
    return errors


def beveridgenelson(y, parameters, constant, p=None, q=None):
    """
    Beveridge-Nelson decomposition for a trending time series.

    Decomposes an I(1) time series into a stochastic trend and a stationary
    cyclic component using an ARMA model fitted to the first differences.
    The decomposition satisfies: trend + cyclic = y.

    Parameters
    ----------
    y : numpy.ndarray
        T-length vector of trending data (assumed I(1)).
    parameters : numpy.ndarray
        1D array of ARMA parameters estimated on diff(y).
        Length must equal constant + len(p) + len(q).
        Order: [constant_value (if constant=1), AR_params, MA_params].
    constant : int
        1 if the fitted model included a constant, 0 otherwise.
    p : numpy.ndarray or array_like, optional
        Non-negative integer array of AR lag indices.
        For example, ``np.array([1])`` for AR(1), ``np.array([1, 3])`` for
        lags at positions 1 and 3. Default is an empty array (no AR terms).
    q : numpy.ndarray or array_like, optional
        Non-negative integer array of MA lag indices.
        For example, ``np.array([1, 2, 3, 4])`` for MA(4).
        Default is an empty array (no MA terms).

    Returns
    -------
    trend : numpy.ndarray
        T-length vector containing the stochastic trend component.
        The first max(p)+1 observations are set equal to y.
    cyclic : numpy.ndarray
        T-length vector containing the stationary cyclic component.
        The first max(p)+1 observations are zero.

    Raises
    ------
    ValueError
        If inputs fail validation (wrong shapes, invalid constant, parameter
        count mismatch, or non-stationary AR dynamics).

    Notes
    -----
    The key property of the decomposition is: ``trend + cyclic = y`` for all
    observations. The first ``max(p) + 1`` trend values equal the corresponding
    ``y`` values, with cyclic values of zero.

    The algorithm follows Newbold (1990):

    1. Extract the constant and compute the long-run mean.
    2. Compute demeaned first differences: delta_y = diff(y) - long_run_mean.
    3. Build a companion matrix from the AR parameters and verify stationarity.
    4. Compute ARMA residuals from the demeaned first differences.
    5. For each time step, compute the cumulative forecast of future delta_y
       values using the finite MA forecasts and the long-run AR contribution
       via the companion matrix formula.

    Examples
    --------
    BN decomposition for log GDP using an AR(1) for short-run dynamics:

    >>> import numpy as np
    >>> from mfe_toolbox.timeseries.beveridgenelson import beveridgenelson
    >>> # Assume parameters were estimated via armaxfilter on diff(lnGDP)
    >>> # trend, cyclic = beveridgenelson(lnGDP, parameters, 1, np.array([1]))

    References
    ----------
    Paul Newbold, "Precise and efficient computation of the Beveridge-Nelson
    decomposition of economic time series", Journal of Monetary Economics,
    Volume 26, Issue 3, December 1990, Pages 453-457.
    """
    # ===================================================================
    # Input validation
    # Ref: beveridgenelson.m:48-82
    # ===================================================================

    # Handle default p and q: Ref beveridgenelson.m:49-55 (nargin handling)
    if p is None:
        p = np.array([], dtype=np.int64)
    if q is None:
        q = np.array([], dtype=np.int64)

    # Ensure numpy arrays with proper dtypes
    y = np.asarray(y, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.int64).ravel()
    q = np.asarray(q, dtype=np.int64).ravel()

    # Ref: beveridgenelson.m:59-64 — Ensure y is a column vector
    # MATLAB: if ndims(y) > 2 || min(size(y)) ~= 1 → error
    if y.ndim > 2 or (y.ndim == 2 and min(y.shape) != 1):
        raise ValueError('Y must be a T by 1 vector.')
    if y.ndim == 2:
        # Ref: beveridgenelson.m:59-61 — Transpose row vectors to columns
        if y.shape[1] > y.shape[0]:
            y = y.T
        y = y.ravel()
    # After this point, y is always 1D

    # Ref: beveridgenelson.m:65-67
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be 0 or 1.')

    # Ref: beveridgenelson.m:68-79
    np_ = len(p)  # number of AR terms
    nq = len(q)   # number of MA terms

    if np_ > 0:
        maxp = int(np.max(p))
    else:
        maxp = 0

    if nq > 0:
        maxq = int(np.max(q))
    else:
        maxq = 0

    # Ref: beveridgenelson.m:80-82
    if len(parameters) != (constant + np_ + nq):
        raise ValueError(
            'PARAMETERS must have CONSTANT + length(P) + length(Q) parameters.'
        )

    # ===================================================================
    # Extract constant and compute long-run mean
    # Ref: beveridgenelson.m:84-92
    # ===================================================================
    long_run_mean = 0.0
    parameters_arma = parameters.copy()

    if constant:
        # Ref: beveridgenelson.m:85 — Extract constant value from first parameter
        const_val = parameters_arma[0]
        # Ref: beveridgenelson.m:86 — Remove constant from parameter vector
        parameters_arma = parameters_arma[1:]
        if np_ > 0:
            # Ref: beveridgenelson.m:88 — Long-run mean = constant / (1 - sum(AR params))
            # Note: division by zero is safe here — if ar_sum == 1, the stationarity
            # check below will raise ValueError before long_run_mean is used.
            ar_sum = np.sum(parameters_arma[0:np_])
            with np.errstate(divide='ignore', invalid='ignore'):
                long_run_mean = const_val / (1.0 - ar_sum)
        else:
            # Ref: beveridgenelson.m:90 — Pure MA with constant
            long_run_mean = const_val

    # Ref: beveridgenelson.m:93 — Compute demeaned first differences
    delta_y = np.diff(y) - long_run_mean

    # ===================================================================
    # Build companion matrix and check stationarity
    # Ref: beveridgenelson.m:94-106
    # ===================================================================
    if maxp > 0:
        # Ref: beveridgenelson.m:95 — Initialize companion matrix
        A = np.zeros((maxp, maxp), dtype=np.float64)
        # Ref: beveridgenelson.m:96 — Place AR parameters in first row
        # MATLAB: A(1,p) = parameters(1:np) — p values are 1-based lag indices
        for i in range(np_):
            # Ref: beveridgenelson.m:96 — MATLAB 1-indexed p(i) → Python 0-indexed p[i]-1
            A[0, int(p[i]) - 1] = parameters_arma[i]
        # Ref: beveridgenelson.m:97-99 — Sub-diagonal entries = 1 (companion form)
        for i in range(1, maxp):
            A[i, i - 1] = 1.0

        # Ref: beveridgenelson.m:100-103 — Stationarity check via eigenvalues
        eigenvalues = np.abs(np.linalg.eig(A)[0])
        if np.max(eigenvalues) >= 1.0:
            raise ValueError(
                'The model for the short-run dynamics is not stationary. '
                'Stationarity is required for the BN decomposition to be well defined.'
            )
    else:
        # Ref: beveridgenelson.m:105 — No AR component
        A = np.zeros((1, 1), dtype=np.float64)  # placeholder, not used in computation

    # ===================================================================
    # Compute ARMA errors
    # Ref: beveridgenelson.m:114-120
    # ===================================================================
    T = len(delta_y)

    # Ref: beveridgenelson.m:116 — Presample length
    m = max(maxp, maxq)
    # Ref: beveridgenelson.m:117 — Zero-padding for MA initialization
    zeros_to_pad = max(maxq - maxp, 0)

    # Ref: beveridgenelson.m:118 — Augment delta_y with leading zeros
    if zeros_to_pad > 0:
        delta_y_augmented = np.concatenate([np.zeros(zeros_to_pad), delta_y])
    else:
        delta_y_augmented = delta_y.copy()

    # Ref: beveridgenelson.m:119 — Compute ARMA errors
    # Original: armaxerrors(parameters, p, q, 0, deltaYAugmentedForMA, [], m, ones(...))
    errors_aug = _compute_arma_errors(parameters_arma, p, q, delta_y_augmented, m)

    # Ref: beveridgenelson.m:120 — Remove zero-padding from errors
    # MATLAB: errors(zerosToPad+1:length(errors)) → Python: errors_aug[zeros_to_pad:]
    errors_arr = errors_aug[zeros_to_pad:]

    # ===================================================================
    # Initialize trend and cyclic components
    # Ref: beveridgenelson.m:122-131
    # ===================================================================
    N = T + 1  # Same length as y (T = len(diff(y)), so N = len(y))

    # Ref: beveridgenelson.m:123 — Initialize trend to zeros
    trend = np.zeros(N, dtype=np.float64)
    # Ref: beveridgenelson.m:124 — First maxp+1 trend values = y values
    # MATLAB: trend(1:maxp+1) = y(1:maxp+1) → Python: trend[0:maxp+1] = y[0:maxp+1]
    trend[0:maxp + 1] = y[0:maxp + 1]

    # Ref: beveridgenelson.m:125 — Initialize cyclic to zeros
    cyclic = np.zeros(N, dtype=np.float64)

    # Ref: beveridgenelson.m:126-127 — Forecast working arrays
    delta_y_forecast = np.zeros(T + m, dtype=np.float64)
    errors_forecast = np.zeros(T + m, dtype=np.float64)

    # Ref: beveridgenelson.m:129-131 — Long-run scale factor
    # Only computed when AR component is present
    if maxp > 0:
        # Ref: beveridgenelson.m:129 — Selection vector e = [1, 0, ..., 0]
        e = np.zeros(maxp, dtype=np.float64)
        e[0] = 1.0
        # Ref: beveridgenelson.m:130 — (I - A)^(-1) * A
        inv_I_minus_A = np.linalg.inv(np.eye(maxp) - A)
        long_run_matrix = inv_I_minus_A @ A
        # Ref: beveridgenelson.m:131 — e * longRunScale (row vector extraction)
        long_run_scale = e @ long_run_matrix
    else:
        long_run_scale = None

    # ===================================================================
    # Main BN decomposition loop
    # Ref: beveridgenelson.m:132-156
    # ===================================================================
    # MATLAB: for t = maxp+1:T → Python: for t in range(maxp, T)
    # In Python 0-based, t goes from maxp to T-1
    for t in range(maxp, T):
        # Ref: beveridgenelson.m:134-135 — Reset forecast arrays
        delta_y_forecast[:] = 0.0
        errors_forecast[:] = 0.0

        # Ref: beveridgenelson.m:136-137 — Fill observed history up to time t
        # MATLAB: deltaYForecast(1:t) = deltaY(1:t) → first t_matlab = t+1 elements
        # Python: delta_y_forecast[0:t+1] = delta_y[0:t+1]
        delta_y_forecast[0:t + 1] = delta_y[0:t + 1]
        errors_forecast[0:t + 1] = errors_arr[0:t + 1]

        # Ref: beveridgenelson.m:138-147 — Compute maxq-step-ahead forecasts
        # MATLAB: for h = 1:maxq
        for h in range(1, maxq + 1):
            idx = t + h  # 0-based index in forecast array
            # Ref: beveridgenelson.m:139-141 — AR contribution to forecast
            for j in range(np_):
                lag = int(p[j])
                delta_y_forecast[idx] += parameters_arma[j] * delta_y_forecast[idx - lag]
            # Ref: beveridgenelson.m:142-146 — MA contribution to forecast
            for j in range(nq):
                lag = int(q[j])
                # Ref: beveridgenelson.m:143 — Guard against negative indices
                # MATLAB: if (t+h-q(j))>0 → Python: if idx-lag >= 0
                if idx - lag >= 0:
                    delta_y_forecast[idx] += (
                        parameters_arma[np_ + j] * errors_forecast[idx - lag]
                    )

        # Ref: beveridgenelson.m:149 — Sum of finite-horizon forecasts
        # MATLAB: cumForecast = sum(deltaYForecast(t+1:t+maxq))
        # Python: delta_y_forecast[t+1:t+maxq+1] (inclusive end adjustment)
        if maxq > 0:
            cum_forecast = np.sum(delta_y_forecast[t + 1:t + maxq + 1])
        else:
            cum_forecast = 0.0

        # Ref: beveridgenelson.m:150-153 — Long-run AR contribution
        if np_ > 0 and maxp > 0:
            # Ref: beveridgenelson.m:151 — Extract last maxp forecasted values
            # MATLAB: deltaYForecast(t+maxq-maxp+1:t+maxq) → maxp elements
            # Python: delta_y_forecast[t+maxq-maxp+1:t+maxq+1]
            delta_y_hat = delta_y_forecast[t + maxq - maxp + 1:t + maxq + 1]
            # Ref: beveridgenelson.m:152 — Add long-run cumulative contribution
            cum_forecast = cum_forecast + np.dot(long_run_scale, delta_y_hat)

        # Ref: beveridgenelson.m:154 — Trend = y + cumulative forecast
        # MATLAB: trend(t+1) = y(t+1) + cumForecast → Python: trend[t+1] = y[t+1] + ...
        trend[t + 1] = y[t + 1] + cum_forecast
        # Ref: beveridgenelson.m:155 — Cyclic = negative of cumulative forecast
        cyclic[t + 1] = -cum_forecast

    return trend, cyclic
