"""
Hodrick-Prescott filtering of multiple time series.

Migrated from timeseries/hp_filter.m (Version 4.0, Kevin Sheppard).
Uses scipy.sparse for the pentadiagonal Gamma matrix as required by AAP Section 0.5.1.

The HP filter decomposes a time series y into a trend component and a cyclical
component:  y = trend + cyclic.  The trend is obtained by solving:

    min_{trend} sum((y - trend)^2) + lambda * sum((trend(t+1) - 2*trend(t) + trend(t-1))^2)

This leads to a banded linear system  Gamma * trend = y  where Gamma is a
T x T symmetric positive-definite pentadiagonal matrix.

Parameters
----------
lambda_param : float
    Smoothing parameter.  Common values:
    - 1600  for quarterly data (Hodrick & Prescott, 1997)
    - 14400 for monthly data
    - 6.25  for annual data (Ravn & Uhlig, 2002)

References
----------
Hodrick, R.J. and Prescott, E.C. (1997). "Postwar U.S. Business Cycles:
An Empirical Investigation." Journal of Money, Credit, and Banking, 29(1), 1-16.
"""

import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def hp_filter(y, lambda_param):
    """
    Hodrick-Prescott filtering of multiple time series.

    Parameters
    ----------
    y : array_like
        A T-by-K matrix of data to be filtered. Each column is filtered
        independently.  T must be >= 1.
    lambda_param : float
        Positive scalar smoothing parameter of the HP filter.  1600 is the
        recommended value for quarterly data; 14400 for monthly data.

    Returns
    -------
    trend : numpy.ndarray
        A T-by-K matrix containing the filtered trend component.
    cyclic : numpy.ndarray
        A T-by-K matrix containing the filtered cyclical component,
        computed as ``cyclic = y - trend``.

    Raises
    ------
    ValueError
        If *y* has more than 2 dimensions, has zero rows, or if
        *lambda_param* is not a non-negative scalar.

    Warns
    -----
    UserWarning
        If *lambda_param* exceeds 1e10, the filter forces a pure linear
        trend and may not be accurate.

    Notes
    -----
    The cyclical component is simply the original data minus the trend:
    ``cyclic = y - trend``.

    For small T the filter degenerates:
    - T = 1 or T = 2 : the trend equals y (identity, no smoothing possible).
    - T = 3 : a dense 3×3 system is solved.
    - T >= 4 : a sparse pentadiagonal system is solved via
      :func:`scipy.sparse.linalg.spsolve`.

    When *lambda_param* exceeds 1e10 the filter is replaced by a simple
    OLS linear trend regression to avoid numerical instability.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.timeseries.hp_filter import hp_filter
    >>> y = np.array([100.0, 102.0, 101.5, 104.0, 103.5, 106.0])
    >>> trend, cyclic = hp_filter(y, 1600)
    >>> np.allclose(y, trend + cyclic)
    True

    See Also
    --------
    mfe_toolbox.timeseries.bkfilter : Baxter-King band-pass filter.
    mfe_toolbox.timeseries.beveridgenelson : Beveridge-Nelson decomposition.
    """

    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------
    # Ref: hp_filter.m:37 — MATLAB accepts row/column vectors and matrices
    y = np.asarray(y, dtype=np.float64)
    squeeze_output = False
    if y.ndim == 1:
        # Treat 1-D input as a column vector (T, 1) for uniform processing
        y = np.atleast_2d(y).T
        squeeze_output = True

    # ------------------------------------------------------------------
    # Input Checking — Ref: hp_filter.m:37-46
    # ------------------------------------------------------------------
    if y.ndim > 2 or y.shape[0] < 1:
        raise ValueError('Y must be a T by K matrix with T>=4')

    if not np.isscalar(lambda_param) or lambda_param < 0:
        raise ValueError('LAMBDA must be a positive scalar.')

    # Ref: hp_filter.m:43-45 — warn for extremely large lambda
    lambda_upper_bound = 1e10
    if lambda_param > lambda_upper_bound:
        warnings.warn(
            'HP_FILTER may not be accurate for very large values of LAMBDA. '
            f'Values above {lambda_upper_bound:.0f} force a pure linear trend.',
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Core computation — Ref: hp_filter.m:51-87
    # ------------------------------------------------------------------
    T = y.shape[0]  # Ref: hp_filter.m:51

    if lambda_param < lambda_upper_bound:
        # ----------------------------------------------------------
        # Handle small-T cases — Ref: hp_filter.m:54-66
        # ----------------------------------------------------------
        if T <= 2:
            # Ref: hp_filter.m:55-56 — Gamma = eye(T), trend = y
            trend = y.copy()
            cyclic = np.zeros_like(y)
        elif T == 3:
            # Ref: hp_filter.m:57-66 — Dense 3×3 Gamma matrix
            lam = float(lambda_param)
            gamma = np.zeros((3, 3), dtype=np.float64)
            # Ref: hp_filter.m:58-60 — Row 1 (MATLAB 1-indexed → Python 0-indexed)
            gamma[0, 0] = 1.0 + lam
            gamma[0, 1] = -2.0 * lam
            gamma[0, 2] = lam
            # Ref: hp_filter.m:61-63 — Row 2
            gamma[1, 0] = -2.0 * lam
            gamma[1, 1] = 1.0 + 4.0 * lam
            gamma[1, 2] = -2.0 * lam
            # Ref: hp_filter.m:64-66 — Row 3 mirrors Row 1
            gamma[2, 2] = gamma[0, 0]
            gamma[2, 1] = gamma[0, 1]
            gamma[2, 0] = gamma[0, 2]
            # Ref: hp_filter.m:81 — Gamma\y  →  numpy.linalg.solve
            trend = np.linalg.solve(gamma, y)
            cyclic = y - trend
        else:
            # ----------------------------------------------------------
            # T >= 4 — Sparse pentadiagonal system
            # Ref: hp_filter.m:67-79
            # ----------------------------------------------------------
            lam = float(lambda_param)

            # Ref: hp_filter.m:68 — spalloc(T,T,nnz) → lil_matrix for row-slice writes
            gamma = scipy.sparse.lil_matrix((T, T), dtype=np.float64)

            # Ref: hp_filter.m:69 — interior row weights
            weights = np.array([lam, -4.0 * lam, 1.0 + 6.0 * lam, -4.0 * lam, lam],
                               dtype=np.float64)

            # Ref: hp_filter.m:71-73 — for i=3:T-2  (MATLAB 1-based inclusive)
            # Python 0-based: rows 2 .. T-3 (inclusive), i.e. range(2, T-2)
            for i in range(2, T - 2):
                # Ref: hp_filter.m:72 — Gamma(i, i-2:i+2) = weights
                # MATLAB i-2:i+2 is inclusive on both ends → Python [i-2 : i+3)
                gamma[i, i - 2:i + 3] = weights

            # Ref: hp_filter.m:74-76 — first row  (MATLAB row 1 → Python row 0)
            row_one_weights = np.array([1.0 + lam, -2.0 * lam, lam], dtype=np.float64)
            gamma[0, 0:3] = row_one_weights  # Ref: hp_filter.m:76 — Gamma(1,1:3)

            # Ref: hp_filter.m:75,77 — second row (MATLAB row 2 → Python row 1)
            row_two_weights = np.array([-2.0 * lam, 1.0 + 5.0 * lam, -4.0 * lam, lam],
                                       dtype=np.float64)
            gamma[1, 0:4] = row_two_weights  # Ref: hp_filter.m:77 — Gamma(2,1:4)

            # Ref: hp_filter.m:78 — Gamma(T, T-2:T) = fliplr(rowOneWeights)
            # MATLAB row T → Python row T-1; MATLAB T-2:T → Python T-3:T
            gamma[T - 1, T - 3:T] = row_one_weights[::-1]

            # Ref: hp_filter.m:79 — Gamma(T-1, T-3:T) = fliplr(rowTwoWeights)
            # MATLAB row T-1 → Python row T-2; MATLAB T-3:T → Python T-4:T
            gamma[T - 2, T - 4:T] = row_two_weights[::-1]

            # Ref: hp_filter.m:81 — trend = Gamma\y
            # Convert to CSC for efficient direct solve
            gamma_csc = gamma.tocsc()

            K = y.shape[1]
            trend = np.empty_like(y)
            if K == 1:
                # spsolve handles 1-D rhs efficiently
                trend[:, 0] = scipy.sparse.linalg.spsolve(gamma_csc, y[:, 0])
            else:
                # Solve each column independently to match MATLAB's backslash
                for k in range(K):
                    trend[:, k] = scipy.sparse.linalg.spsolve(gamma_csc, y[:, k])

            cyclic = y - trend
    else:
        # ----------------------------------------------------------
        # Large lambda fallback — pure linear trend via OLS
        # Ref: hp_filter.m:83-86
        # ----------------------------------------------------------
        # Ref: hp_filter.m:84 — x = [ones(T,1) (1:T)']
        x = np.column_stack([np.ones(T, dtype=np.float64),
                             np.arange(1, T + 1, dtype=np.float64)])
        # Ref: hp_filter.m:85 — trend = x * (x \ y)
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        trend = x @ beta
        cyclic = y - trend

    # ------------------------------------------------------------------
    # If the caller passed a 1-D array, return 1-D arrays to match shape
    # ------------------------------------------------------------------
    if squeeze_output:
        trend = trend.ravel()
        cyclic = cyclic.ravel()

    return trend, cyclic
