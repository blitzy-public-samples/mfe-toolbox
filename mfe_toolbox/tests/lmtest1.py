"""
LM test for serial correlation with optional heteroskedasticity robustness.

Migrated from tests/lmtest1.m — Kevin Sheppard, MFE Toolbox v4.0.
Copyright: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3, Date: 1/1/2007

Provides the :func:`lmtest1` function which computes Lagrange Multiplier
statistics for testing the null hypothesis of no serial correlation up to
*q* lags.  Under H0, each statistic is asymptotically chi-squared
distributed with degrees of freedom equal to the corresponding lag order.

The variance estimator is computed under the alternative hypothesis (to
increase the power of the test), making this an LR-class test.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from mfe_toolbox.utility.newlagmatrix import newlagmatrix

__all__ = ['lmtest1']


def lmtest1(
    data: np.ndarray,
    q: int,
    robust: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    LM tests for the presence of serial correlation in q lags.

    Computes *q* LM statistics testing for serial correlation at each lag
    order from 1 through *q*.  Under the null hypothesis of no serial
    correlation, each statistic is asymptotically chi-squared distributed
    with degrees of freedom equal to its lag order.

    Parameters
    ----------
    data : array_like
        A T-element array of data.  Must be a 1-D vector or a single-column
        2-D array.  Converted internally to float64.
    q : int
        The maximum number of lags to test.  Must be a positive integer with
        ``q <= T``.  Statistics and p-values are returned for each lag from
        1 to *q*.
    robust : bool, optional
        If ``True`` (default), use heteroskedasticity-robust standard errors
        in the variance estimator.  If ``False``, assume homoskedasticity.

    Returns
    -------
    lm : numpy.ndarray
        A ``(q,)`` array of LM test statistics, one per lag order.
    pval : numpy.ndarray
        A ``(q,)`` array of chi-squared p-values corresponding to each
        LM statistic.

    Raises
    ------
    ValueError
        If inputs fail validation: *data* must be 1-D or column vector,
        *q* must be a positive integer not exceeding the sample size, and
        *robust* must be boolean-equivalent (``True``/``False``/0/1).

    Notes
    -----
    The variance estimator is computed under the alternative (to increase
    the power of the test).  As a result, this test is an LR-class test.
    It is otherwise identical to the usual LM test for serial correlation.

    The data vector is demeaned at each iteration of the loop (cumulative
    demeaning), exactly matching the original MATLAB implementation
    (Ref: lmtest1.m:67).

    See Also
    --------
    mfe_toolbox.tests.ljungbox : Ljung-Box portmanteau test for serial
        correlation.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal(200)
    >>> lm_stats, p_values = lmtest1(data, 5)
    >>> lm_stats.shape
    (5,)
    >>> p_values.shape
    (5,)
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: lmtest1.m:38-60
    # ------------------------------------------------------------------
    # Ref: lmtest1.m:38-39 — nargin check (Python handles via signature)

    # Ref: lmtest1.m:47-51 — Convert and ensure column vector
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2 and data.shape[1] == 1:
        # Ref: lmtest1.m:48 — if size(data,1)~=T, data=data'
        data = data.ravel()
    elif data.ndim == 2 and data.shape[0] == 1:
        # Ref: lmtest1.m:48 — row vector case, transpose to column
        data = data.ravel()
    elif data.ndim > 2 or (data.ndim == 2 and min(data.shape) > 1):
        # Ref: lmtest1.m:50-51 — DATA must be a column vector
        raise ValueError('DATA must be a column vector')
    else:
        data = data.ravel()

    T = len(data)

    # Ref: lmtest1.m:53-54 — Q must be a positive integer
    if not isinstance(q, (int, np.integer)) or q <= 0:
        raise ValueError('Q must be a positive integer')

    # Ref: lmtest1.m:44-45 — T must be >= q
    if T < q:
        raise ValueError('At least Q observations required')

    # Ref: lmtest1.m:56-59 — ROBUST must be 0 or 1
    if not isinstance(robust, (bool, int, np.integer)) or robust not in (
        0,
        1,
        True,
        False,
    ):
        raise ValueError('ROBUST must be either 0 or 1')
    robust = bool(robust)

    # ------------------------------------------------------------------
    # Computation
    # Ref: lmtest1.m:65-81
    # ------------------------------------------------------------------

    # Ref: lmtest1.m:65 — lm = zeros(q, 1)
    lm = np.zeros(q, dtype=np.float64)

    for Q_idx in range(1, q + 1):
        # Ref: lmtest1.m:67 — data = data - mean(data)
        # CRITICAL: data is demeaned cumulatively at each iteration,
        # faithfully reproducing the MATLAB source behavior.
        data = data - np.mean(data)

        # Ref: lmtest1.m:68 — [y, x] = newlagmatrix(data, Q, 0)
        # Third argument 0 means no constant column in the lag matrix.
        # Returns y_trimmed: (T-Q, 1), x: (T-Q, Q)
        y, x = newlagmatrix(data, Q_idx, 0)

        # Ref: lmtest1.m:69 — e = y (residuals under H0 = data itself)
        # y_trimmed is (T-Q, 1); ravel to 1-D for vector operations
        e = y.ravel()

        # Ref: lmtest1.m:70 — s = x(:,1:Q) .* repmat(e, 1, Q)
        # Since x already has exactly Q_idx columns (no constant),
        # x[:, :Q_idx] is the full x matrix.  Broadcasting: e[:, newaxis]
        # replicates the residuals across Q_idx columns.
        s = x[:, :Q_idx] * e[:, np.newaxis]

        # Ref: lmtest1.m:71 — sbar = mean(s)
        # MATLAB mean(s) on a matrix returns column means (1 x Q).
        # NumPy axis=0 gives shape (Q_idx,).
        sbar = np.mean(s, axis=0)

        # Ref: lmtest1.m:72-79 — Variance estimator (robust vs non-robust)
        if robust:
            # Ref: lmtest1.m:74 — s = bsxfun(@minus, s, sbar)
            # Demean score vectors (broadcasting handles row-wise subtraction)
            s = s - sbar

            # Ref: lmtest1.m:75 — S = s'*s / T
            # HAC-like robust variance estimator (Q x Q)
            S = s.T @ s / T
        else:
            # Ref: lmtest1.m:77 — e = e - mean(e)
            e = e - np.mean(e)

            # Ref: lmtest1.m:78 — S = (e'*e / T) * (x'*x) / T
            # Non-robust (homoskedastic) variance estimator.
            # (e @ e) is scalar dot product; (x.T @ x) is (Q x Q) matrix.
            S = (e @ e / T) * (x.T @ x) / T

        # Ref: lmtest1.m:80 — lm(Q) = T * sbar * S^(-1) * sbar'
        # Quadratic form: 1xQ * QxQ * Qx1 → scalar.
        # Using np.linalg.solve(S, sbar) instead of S^(-1)*sbar' for
        # improved numerical stability.
        lm[Q_idx - 1] = float(T * sbar @ np.linalg.solve(S, sbar))

    # Ref: lmtest1.m:82 — pval = 1 - chi2cdf(lm, (1:q)')
    # Each LM statistic at lag Q is chi2(Q) distributed under H0.
    pval = 1.0 - chi2.cdf(lm, np.arange(1, q + 1))

    return lm, pval
