"""
Ljung-Box test for serial correlation.

Computes the Ljung-Box Q statistic for each lag from 1 to LAGS to test
the null hypothesis of no serial correlation up to q lags.  Under H0
of no serial correlation and assuming homoskedasticity, the Ljung-Box
test statistic is asymptotically distributed chi-squared(q).

Migrated from tests/ljungbox.m — MFE Toolbox v4.0
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 1/1/2007
"""

import numpy as np
from scipy.stats import chi2

from mfe_toolbox.timeseries.sacf import sacf


def ljungbox(data, lags):
    """
    Ljung-Box test for the presence of serial correlation up to q lags.

    Returns LAGS Ljung-Box Q statistics, one for each lag between 1 and
    LAGS.  Under the null of no serial correlation and assuming
    homoskedasticity, the Ljung-Box test statistic is asymptotically
    distributed chi-squared(q).

    Parameters
    ----------
    data : array_like
        A T-element array of data.  Must be convertible to a 1-D float64
        array.
    lags : int
        The maximum number of lags to compute the LB statistic.  Must be a
        positive integer strictly less than the length of data.  The
        statistic and p-value will be returned for all sets of lags up to
        and including LAGS.

    Returns
    -------
    q : numpy.ndarray
        A (lags,) array of Ljung-Box Q statistics, one per lag.
    pval : numpy.ndarray
        A (lags,) array of p-values from the chi-squared distribution.

    Raises
    ------
    ValueError
        If lags is not a positive integer, or if T <= lags, or if data
        cannot be converted to a 1-D vector.

    Notes
    -----
    This test statistic is common but often inappropriate since it assumes
    homoskedasticity.  For a heteroskedasticity-consistent serial
    correlation test, see :func:`mfe_toolbox.tests.lmtest1.lmtest1`.

    The Q statistic at lag L is defined as:

    .. math::

        Q(L) = T(T+2) \\sum_{k=1}^{L} \\frac{\\hat{\\rho}_k^2}{T - k}

    where :math:`\\hat{\\rho}_k` is the sample autocorrelation at lag k,
    and T is the sample size.

    See Also
    --------
    mfe_toolbox.tests.lmtest1 : LM test for ARCH effects (heteroskedasticity-robust).
    mfe_toolbox.timeseries.sacf : Sample autocorrelation function.
    mfe_toolbox.timeseries.spacf : Sample partial autocorrelation function.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal(500)
    >>> q, pval = ljungbox(data, 10)
    >>> q.shape
    (10,)
    >>> pval.shape
    (10,)

    References
    ----------
    Ljung, G. M. and Box, G. E. P. (1978). "On a Measure of Lack of Fit
    in Time Series Models." Biometrika, 65(2), 297-303.
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: ljungbox.m:34-49 — Input checking block
    # ------------------------------------------------------------------

    # Ref: ljungbox.m:37,41-45 — Convert to column vector
    # In MATLAB: if size(data,1)~=T, data=data'; if size(data,2)~=1 error
    # In Python: flatten to 1-D float64 array
    data = np.asarray(data, dtype=np.float64).ravel()
    T = len(data)

    # Ref: ljungbox.m:47-49 — Validate lags is a positive integer scalar
    # NOTE: The original MATLAB code has a known bug using && (AND) instead
    # of || (OR) in the condition ~isscalar(lags) && lags>0 && floor(lags)==lags.
    # The Python translation corrects this to properly validate that lags is
    # a positive integer scalar.
    if not isinstance(lags, (int, np.integer)) or lags <= 0:
        raise ValueError('LAGS must be a positive integer')

    # Ref: ljungbox.m:38-39 — T<=lags check
    if T <= lags:
        raise ValueError('At least LAGS observations required')

    # ------------------------------------------------------------------
    # Computation
    # Ref: ljungbox.m:54-61 — Main computation block
    # ------------------------------------------------------------------

    # Ref: ljungbox.m:54 — Compute sample autocorrelations using sacf
    # sacf(data, lags, 0, 0) → robust=False, graph=False
    # sacf returns (ac, acstd, fig); we need only ac
    ac, _, _ = sacf(data, lags, 0, 0)

    # Ref: ljungbox.m:56-60 — Compute Q statistics for each lag
    # MATLAB:
    #   q = zeros(lags,1);
    #   T = length(data);
    #   for L=1:lags
    #       q(L) = T*(T+2)*sum(ac(1:L).^2 ./ (T-(1:L)'));
    #   end
    q = np.zeros(lags)
    for L in range(1, lags + 1):
        # Ref: ljungbox.m:59 — MATLAB 1-indexed: ac(1:L) → Python 0-indexed: ac[0:L]
        # (T-(1:L)') → T - np.arange(1, L+1) where arange produces [1, 2, ..., L]
        ac_sq = ac[:L] ** 2
        divisors = T - np.arange(1, L + 1, dtype=np.float64)
        q[L - 1] = T * (T + 2) * np.sum(ac_sq / divisors)

    # Ref: ljungbox.m:61 — chi2cdf(q, (1:lags)') → scipy.stats.chi2.cdf(q, df)
    # Degrees of freedom for each lag: df = [1, 2, ..., lags]
    pval = 1.0 - chi2.cdf(q, np.arange(1, lags + 1))

    return q, pval
