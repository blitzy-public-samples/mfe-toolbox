"""
Jarque-Bera test for normality.

Computes the Jarque-Bera test for normality using the skewness and kurtosis
to determine if a distribution is normal.

Migrated from tests/jarquebera.m — Kevin Sheppard, MFE Toolbox v4.0.
Revision: 3    Date: 2/1/2008
"""

import numpy as np
from scipy.stats import chi2


def jarquebera(
    data: np.ndarray, K: int = 2, alpha: float = 0.05
) -> tuple[float, float, bool]:
    """
    Computes the Jarque-Bera test for normality using skewness and kurtosis.

    Parameters
    ----------
    data : array_like
        A set of data to be tested for deviations from normality.
    K : int, optional
        The number of dependent variables used in constructing the errors.
        For example, if there were 4 regressors (4 mean parameters + 1
        variance parameter), K should be 5. Default is 2.
    alpha : float, optional
        The level of the test for the null of normality. Default is 0.05.

    Returns
    -------
    statistic : float
        The Jarque-Bera test statistic.
    pval : float
        The p-value of the null hypothesis of normality.
    H : bool
        Hypothesis indicator: False for fail to reject the null of normality,
        True otherwise.

    Raises
    ------
    ValueError
        If ``data`` is empty, ``alpha`` is not a scalar in [0, 1], or ``K``
        is not a non-negative integer scalar.

    Notes
    -----
    The data can be mean 0 or not. In either case the sample mean is
    subtracted and the data are standardized by the sample standard deviation
    (using the unbiased estimator with ddof=1, matching MATLAB's ``std``)
    before computing the statistic.

    The Jarque-Bera statistic is computed as:

    .. math::

        JB = (T - K) \\left( \\frac{\\hat{\\mu}_3^2}{6}
              + \\frac{(\\hat{\\mu}_4 - 3)^2}{24} \\right)

    where :math:`\\hat{\\mu}_3` and :math:`\\hat{\\mu}_4` are the sample
    skewness and kurtosis of the standardized data, and *T* is the sample
    size.

    Under the null hypothesis of normality, the test statistic is
    asymptotically distributed as :math:`\\chi^2(2)`.

    Examples
    --------
    Jarque-Bera test on normal data:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> x = rng.standard_normal(100)
    >>> statistic, pval, H = jarquebera(x)

    Jarque-Bera test on regression residuals with 4 regressors
    (4 mean parameters + 1 variance):

    >>> statistic, pval, H = jarquebera(x, K=5)
    """
    # ------------------------------------------------------------------ #
    # Input Validation
    # ------------------------------------------------------------------ #

    # Ref: jarquebera.m:53-56 — MATLAB checks vector shape via size();
    # Python equivalent converts to 1-D float64 array via ravel().
    data = np.asarray(data, dtype=np.float64).ravel()
    T: int = len(data)
    if T == 0:
        raise ValueError("DATA must be a non-empty vector")

    # Ref: jarquebera.m:58-59 — MATLAB error message reads
    # "ALPHA must be a scalar value between 1 and 0"
    if not np.isscalar(alpha) or alpha > 1.0 or alpha < 0.0:
        raise ValueError(
            "ALPHA must be a scalar value between 0 and 1"
        )

    # Ref: jarquebera.m:63-64 — K must be a non-negative integer scalar.
    # isinstance check covers both Python int and numpy integer types.
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError(
            "K must be an integer scalar value greater than or equal to 0."
        )

    # ------------------------------------------------------------------ #
    # Computation
    # ------------------------------------------------------------------ #

    # Ref: jarquebera.m:69 — demean the data
    z = data - np.mean(data)

    # Ref: jarquebera.m:70 — standardize using sample std with ddof=1
    # to match MATLAB's default std() which uses (N-1) divisor.
    std_z = np.std(z, ddof=1)
    if std_z == 0.0:
        # All observations identical; skewness and kurtosis are undefined.
        # Return statistic = 0, pval = 1 (cannot reject normality).
        return 0.0, 1.0, False
    z = z / std_z

    # Ref: jarquebera.m:71 — Jarque-Bera test statistic
    # skew_term = (mean(z^3))^2 / 6   (squared sample skewness / 6)
    # kurt_term = (mean(z^4) - 3)^2 / 24  (squared excess kurtosis / 24)
    skew_term: float = np.mean(z ** 3) ** 2 / 6.0
    kurt_term: float = (np.mean(z ** 4) - 3.0) ** 2 / 24.0
    statistic: float = float((T - K) * (skew_term + kurt_term))

    # Ref: jarquebera.m:72 — p-value from chi-squared CDF with 2 df.
    # Replaces MATLAB chi2cdf / duplication/chi2cdf.m with scipy.stats.chi2.
    pval: float = float(1.0 - chi2.cdf(statistic, 2))

    # Ref: jarquebera.m:73 — hypothesis decision at significance level alpha
    H: bool = bool(pval < alpha)

    return statistic, pval, H
