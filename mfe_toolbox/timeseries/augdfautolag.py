"""
Augmented Dickey-Fuller test with automatic lag selection.

Migrated from ``timeseries/augdfautolag.m`` — Automates lag selection for
the ADF unit root test by iterating over candidate lag lengths (0 to
*maxlags*), computing AIC or BIC for each, selecting the lag that minimizes
the chosen information criterion, and then calling :func:`augdf` with that
optimal lag.

The ADF regression for information criterion evaluation is estimated in
levels form:

    y_t = ρ·y_{t-1} + Σ_{j=1}^{i} φ_j·Δy_{t-j} + [deterministic] + ε_t

for each candidate lag length *i* in ``{0, 1, ..., maxlags}``.  The
residual variance is ``s² = e'e / tau`` (MLE divisor) and the number of
estimated parameters ``K = dim(X)`` is used to form:

    AIC = log(s²) + 2·K / tau
    BIC = log(s²) + K·log(tau) / tau

The lag minimizing the selected criterion is passed to :func:`augdf` for
full ADF inference.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3.0.1    Date: 1/1/2007
"""

import numpy as np

from mfe_toolbox.timeseries.augdf import augdf
from mfe_toolbox.utility.newlagmatrix import newlagmatrix

__all__ = ['augdfautolag']


def augdfautolag(y, p, maxlags, ic='AIC'):
    """
    Dickey-Fuller and Augmented Dickey-Fuller with automatic lag selection.

    Selects the optimal number of augmenting lags for the ADF test by
    minimizing the Akaike Information Criterion (AIC) or Bayesian Information
    Criterion (BIC) over a grid of candidate lag lengths from 0 to *maxlags*.

    Parameters
    ----------
    y : array_like
        A T-element vector of time series data.  Must be 1-D or a column
        vector (T × 1).
    p : int
        Order of the polynomial to include in the ADF regression:

        - 0 : No deterministic terms
        - 1 : Constant
        - 2 : Time Trend
        - 3 : Constant, DGP assumed to have a time trend
    maxlags : int
        The maximum number of lags to include in the ADF test.  Must be a
        positive integer.
    ic : {'AIC', 'BIC'}, optional
        Information criterion to use for lag selection.  Default is ``'AIC'``.

    Returns
    -------
    adfstat : float
        Dickey-Fuller test statistic computed at the optimal lag.
    pval : float
        Probability the series contains a unit root (one-sided p-value).
    critval : numpy.ndarray
        A 6-element array of critical values at the
        ``[0.01, 0.05, 0.10, 0.90, 0.95, 0.99]`` significance levels.
    resid : numpy.ndarray
        Residuals from the ADF regression at the optimal lag.
    lags : int
        The selected number of augmenting lags (minimizer of the IC).
    ics : numpy.ndarray
        A ``(maxlags + 1)``-element array containing the information
        criterion value at each candidate lag (index 0 → lag 0, index 1 →
        lag 1, ..., index *maxlags* → lag *maxlags*).

    Raises
    ------
    ValueError
        If *y* is not a vector, *p* is not in ``{0, 1, 2, 3}``, *maxlags*
        is not a positive integer, the data length is insufficient, or
        *ic* is not ``'AIC'`` or ``'BIC'``.

    See Also
    --------
    augdf : ADF test with a fixed number of lags.

    Notes
    -----
    Migrated from ``timeseries/augdfautolag.m`` (Kevin Sheppard,
    Revision 3.0.1, Date: 1/1/2007).

    The information criterion evaluation uses the levels form of the ADF
    regression (y_t on y_{t-1} plus lagged differences), which is
    algebraically equivalent to the differenced form for the purpose of
    residual variance estimation and lag selection.

    MATLAB-to-Python index translations applied throughout:

    - MATLAB ``y(maxlags+2:T)`` → Python ``y[maxlags+1:T]``
    - MATLAB ``y(maxlags+1:T-1)`` → Python ``y[maxlags:T-1]``
    - MATLAB ``ydifflags(:,1:i)`` → Python ``ydifflags[:, :i]``
    - MATLAB ``[~,lags]=min(ICs); lags=lags-1`` → Python ``lags = int(np.argmin(ics))``
    - MATLAB ``X\\Y`` → ``np.linalg.lstsq(X, Y, rcond=None)[0]``

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = np.cumsum(rng.standard_normal(200))  # unit root process
    >>> adfstat, pval, critval, resid, lags, ics = augdfautolag(y, 1, 12)
    """
    # ------------------------------------------------------------------
    # Input Validation
    # Ref: augdfautolag.m:39-69
    # ------------------------------------------------------------------

    # Ref: augdfautolag.m:42 — T=length(y)
    y = np.asarray(y, dtype=np.float64)

    # Ref: augdfautolag.m:46-48 — Ensure y is a column vector;
    # transpose row vectors to columns
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim == 2:
        # Ref: augdfautolag.m:46 — if size(y,1)~=T, y=y'
        # MATLAB length(y) returns max dimension; transpose if rows != max
        t_max = max(y.shape)
        if y.shape[0] != t_max:
            y = y.T
    else:
        raise ValueError('Y must be a column vector')

    # Ref: augdfautolag.m:49-50 — if size(y,2)~=1, error(...)
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError('Y must be a column vector')

    # Ref: augdfautolag.m:42 — T=length(y)
    T_orig = y.shape[0]

    # Ref: augdfautolag.m:52-54 — Validate maxlags: must be a positive integer
    # NOTE: MATLAB source has a logic bug (uses && instead of ||); we implement
    # the intended semantics from the error message: positive integer scalar.
    try:
        maxlags_int = int(maxlags)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('LAGS must be a positive integer')
    if maxlags_int != maxlags or maxlags_int <= 0:
        raise ValueError('LAGS must be a positive integer')
    maxlags = maxlags_int

    # Ref: augdfautolag.m:43-44 — if T<=(maxlags+1), error(...)
    if T_orig <= (maxlags + 1):
        raise ValueError('Length of data must be larger than LAGS')

    # Ref: augdfautolag.m:55-60 — Validate p: scalar integer in {0,1,2,3}
    if not isinstance(p, (int, np.integer)):
        try:
            p_int = int(p)
            if p_int != p:
                raise ValueError('P must be a scalar integer in {0, 1, 2, 3}')
            p = p_int
        except (TypeError, ValueError, OverflowError):
            raise ValueError('P must be a scalar integer in {0, 1, 2, 3}')
    if p not in (0, 1, 2, 3):
        raise ValueError('P must be a scalar integer in {0, 1, 2, 3}')

    # Ref: augdfautolag.m:61-69 — Validate IC: string, 'AIC' or 'BIC'
    if not isinstance(ic, str):
        raise ValueError("IC must be a string, either 'AIC' or 'BIC'")
    ic_upper = ic.upper()
    if ic_upper not in ('AIC', 'BIC'):
        raise ValueError("IC must be a string, either 'AIC' or 'BIC'")

    # ------------------------------------------------------------------
    # Setup common to all deterministic specifications
    # Ref: augdfautolag.m:78-83
    # ------------------------------------------------------------------

    # Ref: augdfautolag.m:79 — ydiff=diff(y)
    ydiff = np.diff(y, axis=0)  # (T_orig - 1, 1)

    # Ref: augdfautolag.m:80 — [ydiffcurr, ydifflags]=newlagmatrix(ydiff,maxlags)
    # Only ydifflags is used (ydiffcurr discarded, marked #ok<ASGLU> in MATLAB)
    _, ydifflags = newlagmatrix(ydiff, maxlags)
    # ydifflags shape: (T_orig - 1 - maxlags, maxlags) when maxlags > 0
    #                  (T_orig - 1, 0)                  when maxlags == 0

    # Ref: augdfautolag.m:81 — T=length(y) (original length, reassigned)
    T = T_orig

    # Ref: augdfautolag.m:82 — Y=y(maxlags+2:T)
    # MATLAB 1-based index maxlags+2 → Python 0-based index maxlags+1
    Y = y[maxlags + 1:T, 0]  # 1-D array, length tau

    # Ref: augdfautolag.m:83 — tau=length(Y)
    tau = len(Y)

    # Lagged level of y used as regressor for unit-root coefficient
    # Ref: augdfautolag.m:89,98,109,118,130,139 — y(maxlags+1:T-1)
    # MATLAB 1-based → Python 0-based: y[maxlags:T-1]
    y_lag = y[maxlags:T - 1, 0]  # 1-D array, length tau

    # Pre-allocate information criterion component arrays
    s2 = np.zeros(maxlags + 1)
    K = np.zeros(maxlags + 1, dtype=np.int64)

    # ------------------------------------------------------------------
    # Branch on deterministic structure
    # Ref: augdfautolag.m:85-146
    # ------------------------------------------------------------------
    if p == 0:
        # ==============================================================
        # Case 1: No deterministic terms
        # Ref: augdfautolag.m:86-104
        # ==============================================================

        # Ref: augdfautolag.m:88-94 — i=0 (zero lags of differences)
        # X = y(maxlags+1:T-1)  [MATLAB] → y_lag reshaped to column
        X = y_lag.reshape(-1, 1)
        # Ref: augdfautolag.m:90 — rho = X\Y  (OLS via left-divide)
        rho = np.linalg.lstsq(X, Y, rcond=None)[0]
        # Ref: augdfautolag.m:92 — e = Y - X*rho
        e = Y - X @ rho
        # Ref: augdfautolag.m:93 — s2(1) = e'*e / tau  (MLE variance)
        s2[0] = float(e @ e) / tau
        # Ref: augdfautolag.m:94 — K(1) = size(X, 2)
        K[0] = X.shape[1]

        # Ref: augdfautolag.m:97-104 — loop i=1:maxlags
        for i in range(1, maxlags + 1):
            # Ref: augdfautolag.m:98 — X = [y(maxlags+1:T-1) ydifflags(:,1:i)]
            # MATLAB ydifflags(:,1:i) → Python ydifflags[:, :i]
            X = np.column_stack([y_lag, ydifflags[:, :i]])
            # Ref: augdfautolag.m:99 — rho = X\Y
            rho = np.linalg.lstsq(X, Y, rcond=None)[0]
            # Ref: augdfautolag.m:101 — e = Y - X*rho
            e = Y - X @ rho
            # Ref: augdfautolag.m:102 — s2(i+1) = e'*e / tau
            s2[i] = float(e @ e) / tau
            # Ref: augdfautolag.m:103 — K(i+1) = size(X, 2)
            K[i] = X.shape[1]

    elif p in (1, 3):
        # ==============================================================
        # Case 2: Constant included (p=1 or p=3)
        # Ref: augdfautolag.m:106-124
        # ==============================================================

        # Ref: augdfautolag.m:108-114 — i=0
        # X = [ones(size(Y)) y(maxlags+1:T-1)]
        ones_col = np.ones(tau)
        X = np.column_stack([ones_col, y_lag])
        # Ref: augdfautolag.m:110 — rho = X\Y
        rho = np.linalg.lstsq(X, Y, rcond=None)[0]
        # Ref: augdfautolag.m:112 — e = Y - X*rho
        e = Y - X @ rho
        # Ref: augdfautolag.m:113 — s2(1) = e'*e / tau
        s2[0] = float(e @ e) / tau
        # Ref: augdfautolag.m:114 — K(1) = size(X, 2)
        K[0] = X.shape[1]

        # Ref: augdfautolag.m:117-124 — loop i=1:maxlags
        for i in range(1, maxlags + 1):
            # Ref: augdfautolag.m:118 — X = [ones(size(Y)) y(maxlags+1:T-1) ydifflags(:,1:i)]
            X = np.column_stack([ones_col, y_lag, ydifflags[:, :i]])
            # Ref: augdfautolag.m:119 — rho = X\Y
            rho = np.linalg.lstsq(X, Y, rcond=None)[0]
            # Ref: augdfautolag.m:121 — e = Y - X*rho
            e = Y - X @ rho
            # Ref: augdfautolag.m:122 — s2(i+1) = e'*e / tau
            s2[i] = float(e @ e) / tau
            # Ref: augdfautolag.m:123 — K(i+1) = size(X, 2)
            K[i] = X.shape[1]

    elif p == 2:
        # ==============================================================
        # Case 4: Constant + time trend
        # Ref: augdfautolag.m:127-145
        # ==============================================================

        # Ref: augdfautolag.m:130 — (1:tau)'
        # MATLAB 1-based time trend preserved for numerical parity
        ones_col = np.ones(tau)
        trend = np.arange(1, tau + 1, dtype=np.float64)

        # Ref: augdfautolag.m:129-135 — i=0
        # X = [ones(size(Y)) y(maxlags+1:T-1) (1:tau)']
        X = np.column_stack([ones_col, y_lag, trend])
        # Ref: augdfautolag.m:131 — rho = X\Y
        rho = np.linalg.lstsq(X, Y, rcond=None)[0]
        # Ref: augdfautolag.m:133 — e = Y - X*rho
        e = Y - X @ rho
        # Ref: augdfautolag.m:134 — s2(1) = e'*e / tau
        s2[0] = float(e @ e) / tau
        # Ref: augdfautolag.m:135 — K(1) = size(X, 2)
        K[0] = X.shape[1]

        # Ref: augdfautolag.m:138-145 — loop i=1:maxlags
        for i in range(1, maxlags + 1):
            # Ref: augdfautolag.m:139 — X = [ones(size(Y)) y(maxlags+1:T-1) (1:tau)' ydifflags(:,1:i)]
            X = np.column_stack([ones_col, y_lag, trend, ydifflags[:, :i]])
            # Ref: augdfautolag.m:140 — rho = X\Y
            rho = np.linalg.lstsq(X, Y, rcond=None)[0]
            # Ref: augdfautolag.m:142 — e = Y - X*rho
            e = Y - X @ rho
            # Ref: augdfautolag.m:143 — s2(i+1) = e'*e / tau
            s2[i] = float(e @ e) / tau
            # Ref: augdfautolag.m:144 — K(i+1) = size(X, 2)
            K[i] = X.shape[1]

    # ------------------------------------------------------------------
    # Compute information criterion
    # Ref: augdfautolag.m:149-153
    # ------------------------------------------------------------------
    K_float = K.astype(np.float64)
    if ic_upper == 'AIC':
        # Ref: augdfautolag.m:150 — ICs = log(s2) + 2*K/tau
        ics = np.log(s2) + 2.0 * K_float / tau
    else:
        # Ref: augdfautolag.m:152 — ICs = log(s2) + K*log(tau)/tau
        ics = np.log(s2) + K_float * np.log(float(tau)) / tau

    # ------------------------------------------------------------------
    # Select optimal lag and run ADF test
    # Ref: augdfautolag.m:154-156
    # ------------------------------------------------------------------
    # Ref: augdfautolag.m:154 — [~,lags]=min(ICs)
    # Ref: augdfautolag.m:155 — lags=lags-1
    # MATLAB min returns 1-based index, then subtracts 1.
    # Python np.argmin returns 0-based index which directly equals the lag count.
    lags = int(np.argmin(ics))

    # Ref: augdfautolag.m:156 — [adfstat,pval,critval,resid]=augdf(y,p,lags)
    adfstat, pval, critval, resid = augdf(y, p, lags)

    return adfstat, pval, critval, resid, lags, ics
