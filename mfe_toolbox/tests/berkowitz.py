"""
Berkowitz distributional forecast test.

Transforms data to normal space via the Probability Integral Transform (PIT)
followed by the inverse normal CDF (``norm.ppf``), then tests for departures
from N(0,1) using a likelihood ratio statistic.  Two modes are supported:

- **'CS'** (cross-sectional) — Tests whether the transformed data have mean 0
  and variance 1 using a chi-squared(2) LR test.
- **'TS'** (time-series, default) — Additionally tests for first-order
  autocorrelation via an AR(1) regression, yielding a chi-squared(3) LR test.

Migrated from ``tests/berkowitz.m`` — Author: Kevin Sheppard
(kevin.sheppard@economics.ox.ac.uk, Revision 3, Date: 3/1/2007)

See Also
--------
mfe_toolbox.tests.kolmogorov : Kolmogorov-Smirnov distributional test.
"""

import numpy as np
from scipy.stats import chi2, norm
from typing import Callable, Optional

from mfe_toolbox.utility.newlagmatrix import newlagmatrix
from mfe_toolbox.distributions.normloglik import normloglik

__all__ = ['berkowitz']


def berkowitz(
    x: np.ndarray,
    test_type: str = 'TS',
    alpha: float = 0.05,
    dist: Optional[Callable] = None,
    *args,
) -> tuple[float, float, bool]:
    """
    Berkowitz distributional forecast test via likelihood ratio.

    Applies the Probability Integral Transform (PIT) to the input data using
    an optional CDF function, transforms the resulting uniform variates to
    standard normal quantiles via ``norm.ppf``, and then tests whether the
    transformed data are consistent with N(0, 1) using a likelihood ratio
    statistic.

    Parameters
    ----------
    x : numpy.ndarray
        A 1-D array of random variables to be tested.  If ``dist`` is
        ``None``, the values must satisfy ``0 < x < 1`` (already PIT'd).
    test_type : str, optional
        ``'TS'`` (default) for time-series data (includes AR(1)
        autocorrelation check, chi-squared with 3 d.f.) or ``'CS'`` for
        cross-sectional data (no autocorrelation check, chi-squared with
        2 d.f.).
    alpha : float, optional
        Significance level for the hypothesis test (default ``0.05``).
        Must satisfy ``0 < alpha < 1``.
    dist : callable or None, optional
        A CDF function applied to ``x`` before the inverse-normal transform.
        Called as ``dist(x, *args)``.  If ``None``, ``x`` is assumed to
        already be uniform on ``(0, 1)`` (i.e. already transformed via PIT).
    *args
        Additional positional arguments passed through to ``dist``.

    Returns
    -------
    stat : float
        The Berkowitz likelihood ratio test statistic.
    pval : float
        Asymptotic p-value (from chi-squared distribution).
    H : bool
        ``True`` if the null hypothesis (correct distribution) is rejected
        at the specified ``alpha`` level; ``False`` otherwise.

    Raises
    ------
    ValueError
        If no input is provided, if ``x`` is not a 1-D vector, if ``dist``
        is ``None`` and any element of ``x`` is outside ``(0, 1)``, if
        ``dist`` is provided but not callable, if ``alpha`` is not a scalar
        in ``(0, 1)``, or if ``test_type`` is not ``'TS'`` or ``'CS'``.

    Examples
    --------
    Test uniform data from a time-series model:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> u = rng.uniform(0.01, 0.99, size=200)
    >>> stat, pval, H = berkowitz(u)

    Test standard-normal data using ``norm.cdf`` as the CDF:

    >>> from scipy.stats import norm
    >>> z = rng.standard_normal(200)
    >>> stat, pval, H = berkowitz(z, 'TS', 0.05, norm.cdf)

    Test normal(mu=1, sigma=2) data from a cross-sectional model:

    >>> z2 = rng.normal(1.0, 2.0, size=200)
    >>> stat, pval, H = berkowitz(z2, 'CS', 0.05, norm.cdf, 1.0, 2.0)

    Notes
    -----
    Migrated from ``tests/berkowitz.m`` (MFE Toolbox v4.0, Kevin Sheppard).

    The MATLAB ``feval(dist, x, varargin{:})`` pattern is replaced by the
    Python callable ``dist(x, *args)``.  The MATLAB ``norminv`` from the
    ``duplication/`` directory is replaced by ``scipy.stats.norm.ppf``,
    and ``chi2cdf`` is replaced by ``scipy.stats.chi2.cdf``.

    The likelihood ratio test statistic is computed as:

        LR = -2 * (LL_restricted - LL_unrestricted)

    Under the null hypothesis, LR ~ chi2(df) where df = 2 (CS) or df = 3 (TS).
    """
    # ------------------------------------------------------------------
    # PARAMETER VALIDATION
    # Ref: berkowitz.m:46-93
    # ------------------------------------------------------------------

    # Ref: berkowitz.m:64-65 — nargin==0 → error
    if x is None:
        raise ValueError('1 or more inputs required.')

    # Ref: berkowitz.m:67-68 — x must be a vector (min(size(x))~=1)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError('X must be a 1-D vector of data.')
    if x.ndim > 2 or (x.ndim == 2 and min(x.shape) != 1):
        raise ValueError('X must be a 1-D vector of data.')
    # Ref: berkowitz.m:70-72 — Ensure column vector; Python uses 1-D ravel
    x = x.ravel()

    # Ref: berkowitz.m:73-74 — Empty type defaults to 'TS'
    if test_type is None or test_type == '':
        test_type = 'TS'

    # Ref: berkowitz.m:76-84 — Validate dist / x range
    if dist is None:
        # Ref: berkowitz.m:77-78 — data must satisfy 0 < x < 1
        if np.any(x >= 1.0) or np.any(x <= 0.0):
            raise ValueError(
                'If dist is not provided, x must satisfy 0 < x < 1.'
            )
    else:
        # Ref: berkowitz.m:81-83 — dist must be callable (char in MATLAB)
        if not callable(dist):
            raise ValueError('dist must be a callable CDF function.')

    # Ref: berkowitz.m:85-86 — alpha must be scalar in (0, 1)
    if not np.isscalar(alpha) or alpha <= 0 or alpha >= 1:
        raise ValueError(
            'alpha must be a scalar between 0 and 1 (exclusive).'
        )

    # Ref: berkowitz.m:88-89 — type must be 'TS' or 'CS'
    if test_type not in ('TS', 'CS'):
        raise ValueError("test_type must be either 'TS' or 'CS'.")

    # ------------------------------------------------------------------
    # CDF TRANSFORMATION
    # Ref: berkowitz.m:95-107
    # ------------------------------------------------------------------
    if dist is not None:
        # Ref: berkowitz.m:96-104 — Apply CDF via dist callable
        # MATLAB: feval(dist, x, varargin{:}) → Python: dist(x, *args)
        try:
            cdfvals = dist(x, *args)
        except Exception:
            raise ValueError(
                'There was an error calling the dist function with the '
                'provided arguments.'
            )
        cdfvals = np.asarray(cdfvals, dtype=np.float64).ravel()
    else:
        # Ref: berkowitz.m:106 — cdfvals = x (already uniform)
        cdfvals = x.copy()

    # Ref: berkowitz.m:108 — workingdata = norminv(cdfvals)
    # MATLAB norminv (duplication/) → scipy.stats.norm.ppf
    workingdata = norm.ppf(cdfvals)

    # Reshape to column vector (T, 1) for normloglik compatibility
    # normloglik requires shape (T, 1) per its validation
    workingdata = workingdata.reshape(-1, 1)

    # ------------------------------------------------------------------
    # LIKELIHOOD RATIO TEST
    # Ref: berkowitz.m:110-129
    # ------------------------------------------------------------------
    if test_type == 'CS':
        # ------- Cross-Sectional Branch (2 degrees of freedom) -------

        # Ref: berkowitz.m:111 — mu = mean(workingdata)
        mu = float(np.mean(workingdata))

        # Ref: berkowitz.m:112 — e = workingdata - mu
        e = workingdata - mu

        # Ref: berkowitz.m:113 — sigma2 = e'*e / length(e)
        # MLE variance estimate (using n, not n-1)
        e_flat = e.ravel()
        sigma2 = float(np.dot(e_flat, e_flat) / e.shape[0])

        # Ref: berkowitz.m:114 — restrictedLL = normloglik(workingdata, 0, 1)
        # Restricted model: N(0, 1)
        restricted_ll, _ = normloglik(workingdata, 0, 1)

        # Ref: berkowitz.m:115 — unrestrictedLL = normloglik(e, 0, sigma2)
        # Unrestricted model: N(mu_hat, sigma2_hat), residuals already demeaned
        unrestricted_ll, _ = normloglik(e, 0, sigma2)

        # Ref: berkowitz.m:116 — stat = -2*(restrictedLL - unrestrictedLL)
        stat = -2.0 * (restricted_ll - unrestricted_ll)

        # Ref: berkowitz.m:117 — pval = 1 - chi2cdf(stat, 2)
        # MATLAB chi2cdf (duplication/) → scipy.stats.chi2.cdf
        pval = float(1.0 - chi2.cdf(stat, 2))

        # Ref: berkowitz.m:118 — H = pval < alpha
        H = bool(pval < alpha)

    else:
        # ------- Time-Series Branch (3 degrees of freedom) -------

        # Ref: berkowitz.m:120 — [y, lags] = newlagmatrix(workingdata, 1, 1)
        # Constructs AR(1) regression: y = workingdata[1:], lags = [1, workingdata[:-1]]
        y, lags = newlagmatrix(workingdata, 1, 1)

        # Ref: berkowitz.m:121 — p = lags\y (MATLAB left-divide = OLS)
        # Python equivalent: np.linalg.lstsq for overdetermined system
        p = np.linalg.lstsq(lags, y, rcond=None)[0]

        # Ref: berkowitz.m:122 — e = y - lags*p
        e = y - lags @ p

        # Ref: berkowitz.m:123 — sigma2 = e'*e / length(e)
        # MLE variance estimate of AR(1) residuals
        e_flat = e.ravel()
        sigma2 = float(np.dot(e_flat, e_flat) / e.shape[0])

        # Ref: berkowitz.m:124 — restrictedLL = normloglik(y, 0, 1)
        # Restricted model: N(0, 1) with no autocorrelation
        restricted_ll, _ = normloglik(y, 0, 1)

        # Ref: berkowitz.m:125 — unrestrictedLL = normloglik(e, 0, sigma2)
        # Unrestricted model: AR(1) with estimated mean, persistence, and variance
        unrestricted_ll, _ = normloglik(e, 0, sigma2)

        # Ref: berkowitz.m:126 — stat = -2*(restrictedLL - unrestrictedLL)
        stat = -2.0 * (restricted_ll - unrestricted_ll)

        # Ref: berkowitz.m:127 — pval = 1 - chi2cdf(stat, 3)
        pval = float(1.0 - chi2.cdf(stat, 3))

        # Ref: berkowitz.m:128 — H = pval < alpha
        H = bool(pval < alpha)

    # Ref: berkowitz.m:1 — function [stat, pval, H] = berkowitz(...)
    # Return order matches MATLAB: (stat, pval, H)
    return stat, pval, H
