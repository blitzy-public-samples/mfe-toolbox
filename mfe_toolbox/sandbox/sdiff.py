"""
Seasonal differencing operator.

Migrated from sandbox/sdiff.m (MFE Toolbox, Version 4.0).
This module implements seasonal differencing by constructing the product polynomial
prod(1 - B^{S[i]})^{d[i]} and applying the resulting filter to the input series.

The filter is constructed by iteratively convolving seasonal lag operators
(1 - B^{S[i]}) raised to the specified differencing order d[i], then extracting
the nonzero lag positions and corresponding polynomial coefficients.
"""

import numpy as np


def sdiff(
    x: np.ndarray, d: np.ndarray, S: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Seasonal differencing operator.

    Constructs the polynomial product prod(1 - B^{S[i]})^{d[i]} and applies the
    resulting filter to the input series.  The polynomial encodes the combined
    seasonal differencing across all specified seasonal components.

    Parameters
    ----------
    x : np.ndarray
        Input time series vector (1-D).  If empty or None, only the lag and
        scale information is returned (with an empty differenced series).
    d : np.ndarray
        Vector of differencing orders, one per seasonal component.  Each
        element must be a non-negative integer specifying how many times the
        corresponding seasonal operator (1 - B^{S[i]}) is applied.
    S : np.ndarray
        Vector of seasonal periods, one per seasonal component.  Each element
        must be a positive integer specifying the seasonal lag length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        y : np.ndarray
            Differenced series.  Empty array if *x* is empty, too short for
            the filter, or if no differencing lags are produced.
        lags : np.ndarray
            Lag indices of the nonzero polynomial coefficients (excluding the
            constant term at lag 0).  These are 0-indexed integer positions
            that correspond directly to the lag values.
        scales : np.ndarray
            Corresponding polynomial coefficients at the lag positions.

    Notes
    -----
    Faithfully migrated from ``sandbox/sdiff.m``.  The MATLAB source applies
    the filter starting from the second nonzero lag (``j=2:length(lags)`` in
    MATLAB, ``range(1, len(lags))`` in Python), which preserves the original
    implementation behaviour.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.sandbox.sdiff import sdiff
    >>> x = np.arange(1.0, 11.0)
    >>> y, lags, scales = sdiff(x, np.array([2]), np.array([3]))
    >>> lags
    array([3, 6])
    >>> scales
    array([-2.,  1.])
    """
    # Ensure d and S are at least 1-D arrays for consistent iteration.
    # Ref: sdiff.m:1 — MATLAB handles scalars and vectors interchangeably.
    d = np.atleast_1d(np.asarray(d, dtype=np.float64))
    S = np.atleast_1d(np.asarray(S, dtype=np.float64))

    # Ref: sdiff.m:3 — Initialize polynomial as the constant 1.
    poly: np.ndarray = np.array([1.0])

    # ------------------------------------------------------------------
    # Polynomial construction: prod(1 - B^{S[i]})^{d[i]}
    # Ref: sdiff.m:4-8 — Nested loops convolve with (1 - B^{S[i]}) kernel.
    # ------------------------------------------------------------------
    for i in range(len(d)):  # Ref: sdiff.m:4 — MATLAB 1:length(d) -> Python range(len(d))
        for _j in range(int(d[i])):  # Ref: sdiff.m:5 — MATLAB 1:d(i) -> Python range(int(d[i]))
            # Ref: sdiff.m:6 — MATLAB [1 zeros(1,S(i)-1) -1] has length S(i)+1.
            # This constructs the seasonal lag operator (1 - B^{S[i]}).
            kernel_len = int(S[i]) + 1
            seasonal_kernel = np.zeros(kernel_len)
            seasonal_kernel[0] = 1.0   # Coefficient at lag 0
            seasonal_kernel[-1] = -1.0  # Coefficient at lag S[i]
            poly = np.convolve(poly, seasonal_kernel)

    # ------------------------------------------------------------------
    # Extract nonzero lag positions and their polynomial coefficients.
    # ------------------------------------------------------------------
    # Ref: sdiff.m:9 — MATLAB find(poly~=0) returns 1-indexed positions.
    # np.nonzero returns 0-indexed positions that directly equal lag values.
    nonzero_idx = np.nonzero(poly)[0]

    # Ref: sdiff.m:10-11 — Remove the constant term (the first nonzero entry,
    # which is always at position/lag 0 with coefficient 1).
    nonzero_idx = nonzero_idx[1:]

    # Ref: sdiff.m:12 — Get polynomial coefficients at the nonzero lag positions.
    # In MATLAB, scales = poly(lags) uses 1-indexed lags BEFORE the subtract-1
    # step.  In Python, nonzero_idx are 0-indexed array positions which are
    # identical to the actual lag values, so poly[nonzero_idx] is correct.
    scales: np.ndarray = poly[nonzero_idx].copy()

    # Ref: sdiff.m:13 — In Python the 0-indexed positions from np.nonzero
    # already represent the lag values (MATLAB subtracts 1 to convert from
    # 1-indexed positions to lag values; that subtraction is not needed here).
    lags: np.ndarray = nonzero_idx.copy()

    # ------------------------------------------------------------------
    # Apply the differencing filter to the input series.
    # ------------------------------------------------------------------
    # Ref: sdiff.m:14 — Initialize output as an empty array.
    y: np.ndarray = np.array([])

    # Ref: sdiff.m:16-25 — Filter application (only if x is non-empty and
    # there are differencing lags to apply).
    if x is not None and len(x) > 0 and len(lags) > 0:
        # Ref: sdiff.m:17 — Maximum lag determines the required burn-in period.
        ml = int(np.max(lags))
        # Ref: sdiff.m:18 — Total number of observations.
        T = len(x)

        # Ref: sdiff.m:19 — Sufficient data length check.
        if (ml + 1) <= T:
            # Ref: sdiff.m:20 — MATLAB x(ml+1:T) is 1-indexed.
            # Python equivalent: x[ml:T] (0-indexed slice).
            # This represents the constant-term contribution (coefficient 1)
            # at lag 0, i.e. y[t] starts as x[t].
            y = x[ml:T].copy()

            # Ref: sdiff.m:21-23 — MATLAB loop j=2:length(lags).
            # Python equivalent: range(1, len(lags)).
            # Adds weighted lagged copies of x for each remaining nonzero
            # polynomial term.
            for j in range(1, len(lags)):
                # Ref: sdiff.m:22 — MATLAB x(ml+1-lags(j) : T-lags(j))
                # Python equivalent: x[ml - lags[j] : T - lags[j]]
                lag_j = int(lags[j])
                y = y + scales[j] * x[ml - lag_j: T - lag_j]

    return y, lags, scales
