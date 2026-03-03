"""
Augmented Dickey-Fuller unit root test.

Migrated from ``timeseries/augdf.m`` — Implements the Dickey-Fuller and
Augmented Dickey-Fuller test for the presence of a unit root in a
univariate time series. Supports four deterministic structure options
(p=0: none, p=1: constant, p=2: time trend, p=3: constant with DGP trend).

The test regression is:

    Δy_t = ρ·y_{t-1} + Σ_{j=1}^{lags} φ_j·Δy_{t-j} + deterministic + ε_t

The null hypothesis H₀: ρ=0 (unit root) is tested via a t-type statistic
whose distribution depends on the deterministic specification.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3.0.1    Date: 1/1/2007
"""

import numpy as np
from scipy.stats import norm

from mfe_toolbox.timeseries.augdfcv import augdfcv
from mfe_toolbox.utility.newlagmatrix import newlagmatrix

__all__ = ['augdf']


def augdf(y, p, lags):
    """
    Dickey-Fuller and Augmented Dickey-Fuller unit root testing.

    Estimates an ADF regression and returns the test statistic, p-value,
    critical values, and regression residuals.  The deterministic
    specification is selected via the *p* parameter.

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
    lags : int
        Number of lags of the differenced series to include in the ADF
        test regression.  Use ``lags=0`` for a standard (non-augmented)
        Dickey-Fuller test.

    Returns
    -------
    adfstat : float
        Dickey-Fuller test statistic.
    pval : float
        Probability the series contains a unit root (one-sided p-value).
    critval : numpy.ndarray
        A 6-element array of critical values at the
        ``[0.01, 0.05, 0.10, 0.90, 0.95, 0.99]`` significance levels
        from the Dickey-Fuller distribution (or standard normal when
        ``p=3``).
    resid : numpy.ndarray
        Residuals (adjusted for lags) from the ADF regression.

    Raises
    ------
    ValueError
        If *y* is not a vector, *p* is not in ``{0, 1, 2, 3}``, *lags*
        is not a non-negative integer, or the data length is too short
        for the requested number of lags.

    See Also
    --------
    augdfautolag : Automatic lag selection for the ADF test.

    Notes
    -----
    Migrated from ``timeseries/augdf.m`` (Kevin Sheppard, Revision 3.0.1,
    Date: 1/1/2007).

    The four deterministic structures correspond to the following ADF
    regressions (with ``q = lags`` augmenting lags):

    - ``p=0``: Δy_t = ρ·y_{t-1} + Σ φ_j·Δy_{t-j} + ε_t
    - ``p=1``: Δy_t = α + ρ·y_{t-1} + Σ φ_j·Δy_{t-j} + ε_t
    - ``p=2``: y_t  = α + ρ·y_{t-1} + β·t + Σ φ_j·Δy_{t-j} + ε_t
    - ``p=3``: y_t  = α + ρ·y_{t-1} + Σ φ_j·Δy_{t-j} + ε_t

    For ``p=0``, ``p=1``, and ``p=2`` the test statistic follows a
    non-standard Dickey-Fuller distribution; critical values and p-values
    are interpolated from Monte Carlo tables via :func:`augdfcv`.
    For ``p=3`` the test statistic is asymptotically standard normal.

    MATLAB-to-Python index translation:
    - MATLAB ``y(lags+1:T-1)`` → Python ``y[lags:T-1]``
    - MATLAB ``y(lags+2:T)``   → Python ``y[lags+1:T]``
    - MATLAB ``X\\y``          → ``np.linalg.lstsq(X, y, rcond=None)[0]``
    """
    # ------------------------------------------------------------------
    # Input Validation
    # Ref: augdf.m:37-58
    # ------------------------------------------------------------------
    # Ref: augdf.m:44-49 — Ensure y is a column vector / 1-D array
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        # Ref: augdf.m:44-46 — MATLAB transposes row vectors to columns;
        # Python reshapes 1-D to (T, 1) for consistency
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[1] != 1:
        # Ref: augdf.m:47-48 — error('Y must be a column vector')
        raise ValueError('Y must be a vector (1-D array or column vector).')

    # Ref: augdf.m:40 — T=length(y)
    T_orig = y.shape[0]

    # Ref: augdf.m:50-52 — Validate lags: must be a non-negative integer
    try:
        lags_int = int(lags)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('LAGS must be a non-negative integer.')
    if lags_int < 0:
        raise ValueError('LAGS must be a non-negative integer.')
    lags = lags_int

    # Ref: augdf.m:41-43 — Length check: T > lags + 1
    if T_orig <= (lags + 1):
        raise ValueError(
            'Length of data must be larger than LAGS + 1. '
            f'Got T={T_orig}, lags={lags}.'
        )

    # Ref: augdf.m:53-58 — Validate p: must be in {0, 1, 2, 3}
    if p not in (0, 1, 2, 3):
        raise ValueError('P must be a scalar integer in {0, 1, 2, 3}.')

    # ------------------------------------------------------------------
    # Setup common to all deterministic specifications
    # Ref: augdf.m:66-72
    # ------------------------------------------------------------------
    # Ref: augdf.m:68 — ydiff=diff(y)
    ydiff = np.diff(y, axis=0)  # (T_orig - 1, 1)

    # Ref: augdf.m:69 — [ydiffcurr, ydifflags]=newlagmatrix(ydiff,lags)
    # ydiffcurr: trimmed current differences (tau, 1)
    # ydifflags: lagged differences matrix  (tau, lags)  or (tau, 0)
    ydiffcurr_2d, ydifflags = newlagmatrix(ydiff, lags)
    # Flatten to 1-D for OLS via lstsq
    ydiffcurr = ydiffcurr_2d.ravel()

    # Ref: augdf.m:70 — T=length(y)  (original length re-assigned)
    T = T_orig

    # Ref: augdf.m:71 — Y=y(lags+2:T)   [MATLAB 1-based]
    # Python 0-based: y[lags+1 : T]
    Y = y[lags + 1:T, 0]  # 1-D, length tau

    # Ref: augdf.m:72 — tau=length(Y)
    tau = len(Y)

    # Ref: augdf.m:77,94,112,131 — y(lags+1:T-1)  [MATLAB 1-based]
    # Python 0-based: y[lags : T-1]
    # Lagged level of y used as regressor for unit root coefficient
    y_lag = y[lags:T - 1, 0]  # 1-D, length tau

    # ------------------------------------------------------------------
    # Switch on deterministic structure
    # Ref: augdf.m:74-147
    # ------------------------------------------------------------------
    if p == 0:
        # ==============================================================
        # Case 1: No deterministic terms
        # Ref: augdf.m:75-91
        # ==============================================================

        # Ref: augdf.m:77 — X=[y(lags+1:T-1) ydifflags]
        if lags > 0:
            X = np.column_stack([y_lag, ydifflags])
        else:
            # Ref: augdf.m:77 — When lags=0, ydifflags is empty;
            # X contains only the lagged level
            X = y_lag.reshape(-1, 1)

        # Ref: augdf.m:78 — rho = X\ydiffcurr  (OLS via left-divide)
        rho = np.linalg.lstsq(X, ydiffcurr, rcond=None)[0]

        # Ref: augdf.m:80 — e = ydiffcurr - X*rho
        e = ydiffcurr - X @ rho

        # Ref: augdf.m:83 — s2 = e'*e/(tau-size(X,2))
        k = X.shape[1]
        s2 = float(e @ e / (tau - k))

        # Ref: augdf.m:84 — Uinv=inv(diag([T T^(0.5)*ones(1,lags)]))
        # For a diagonal matrix, inv(diag(d)) = diag(1/d)
        diag_vals = np.hstack([np.array([float(T)]),
                               T**0.5 * np.ones(lags)])
        Uinv = np.diag(1.0 / diag_vals)

        # Ref: augdf.m:85 — sel=[1 zeros(1,lags)]
        sel = np.zeros(1 + lags)
        sel[0] = 1.0

        # Ref: augdf.m:86 — sigp=s2*sel*(Uinv*(X'*X)*Uinv)^(-1)*sel'
        A = Uinv @ (X.T @ X) @ Uinv
        A_inv = np.linalg.inv(A)
        sigp = s2 * float(sel @ A_inv @ sel)

        # Ref: augdf.m:89 — adfstat = tau*rho(1)/sqrt(sigp)
        # MATLAB 1-indexed rho(1) → Python 0-indexed rho[0]
        adfstat = float(tau * rho[0] / np.sqrt(sigp))

        # Ref: augdf.m:91 — [pval,critval]=augdfcv(adfstat,p,tau)
        pval, critval = augdfcv(adfstat, p, tau)

    elif p == 1:
        # ==============================================================
        # Case 2: Constant only
        # Ref: augdf.m:92-108
        # ==============================================================

        # Ref: augdf.m:94 — X=[ones(size(Y)) y(lags+1:T-1) ydifflags]
        if lags > 0:
            X = np.column_stack([np.ones(tau), y_lag, ydifflags])
        else:
            X = np.column_stack([np.ones(tau), y_lag])

        # Ref: augdf.m:95 — rho = X\ydiffcurr
        rho = np.linalg.lstsq(X, ydiffcurr, rcond=None)[0]

        # Ref: augdf.m:97 — e = ydiffcurr - X*rho
        e = ydiffcurr - X @ rho

        # Ref: augdf.m:100 — s2 = e'*e/(tau-size(X,2))
        k = X.shape[1]
        s2 = float(e @ e / (tau - k))

        # Ref: augdf.m:101 — Uinv=inv(diag([T^(0.5) T T^(0.5)*ones(1,lags)]))
        diag_vals = np.hstack([np.array([T**0.5, float(T)]),
                               T**0.5 * np.ones(lags)])
        Uinv = np.diag(1.0 / diag_vals)

        # Ref: augdf.m:102 — sel=[0 1 zeros(1,lags)]
        sel = np.zeros(2 + lags)
        sel[1] = 1.0

        # Ref: augdf.m:103 — sigp=s2*sel*(Uinv*(X'*X)*Uinv)^(-1)*sel'
        A = Uinv @ (X.T @ X) @ Uinv
        A_inv = np.linalg.inv(A)
        sigp = s2 * float(sel @ A_inv @ sel)

        # Ref: augdf.m:106 — adfstat = tau*rho(2)/sqrt(sigp)
        # MATLAB 1-indexed rho(2) → Python 0-indexed rho[1]
        adfstat = float(tau * rho[1] / np.sqrt(sigp))

        # Ref: augdf.m:108 — [pval,critval]=augdfcv(adfstat,p,tau)
        pval, critval = augdfcv(adfstat, p, tau)

    elif p == 2:
        # ==============================================================
        # Case 4: Constant + time trend
        # Ref: augdf.m:110-127
        # Note: MATLAB comments label this "Case 4" despite being p=2
        # ==============================================================

        # Ref: augdf.m:112 — X=[ones(size(Y)) y(lags+1:T-1) (1:tau)' ydifflags]
        # Ref: augdf.m:112 — MATLAB (1:tau)' → Python np.arange(1, tau+1)
        trend = np.arange(1, tau + 1, dtype=np.float64)
        if lags > 0:
            X = np.column_stack([np.ones(tau), y_lag, trend, ydifflags])
        else:
            X = np.column_stack([np.ones(tau), y_lag, trend])

        # Ref: augdf.m:113 — rho = X\Y  (NOTE: regresses on levels Y, not differences)
        rho = np.linalg.lstsq(X, Y, rcond=None)[0]

        # Ref: augdf.m:115 — e = Y - X*rho
        e = Y - X @ rho

        # Ref: augdf.m:118 — s2 = e'*e/(tau-size(X,2))
        k = X.shape[1]
        s2 = float(e @ e / (tau - k))

        # Ref: augdf.m:119 — X(:,2)=X(:,2)-rho(1)*(1:tau)'
        # MATLAB 1-indexed column 2 → Python 0-indexed column 1
        # This de-trending adjustment is needed for the covariance computation
        X[:, 1] = X[:, 1] - rho[0] * trend

        # Ref: augdf.m:120 — Uinv=inv(diag([tau^(0.5) tau tau^(1.5) tau^(0.5)*ones(1,lags)]))
        diag_vals = np.hstack([np.array([tau**0.5, float(tau), tau**1.5]),
                               tau**0.5 * np.ones(lags)])
        Uinv = np.diag(1.0 / diag_vals)

        # Ref: augdf.m:121 — sel=[0 1 0 zeros(1,lags)]
        sel = np.zeros(3 + lags)
        sel[1] = 1.0

        # Ref: augdf.m:122 — sigp=s2*sel*(Uinv*(X'*X)*Uinv)^(-1)*sel'
        A = Uinv @ (X.T @ X) @ Uinv
        A_inv = np.linalg.inv(A)
        sigp = s2 * float(sel @ A_inv @ sel)

        # Ref: augdf.m:125 — adfstat = tau*(rho(2)-1)/sqrt(sigp)
        # MATLAB 1-indexed rho(2) → Python 0-indexed rho[1]
        adfstat = float(tau * (rho[1] - 1.0) / np.sqrt(sigp))

        # Ref: augdf.m:127 — [pval,critval]=augdfcv(adfstat,p,tau)
        pval, critval = augdfcv(adfstat, p, tau)

    elif p == 3:
        # ==============================================================
        # Case 3: Constant with DGP assumed to have a time trend
        # Ref: augdf.m:129-146
        # Note: MATLAB comments label this "Case 3" for p=3
        # ==============================================================

        # Ref: augdf.m:131 — X=[ones(size(Y)) y(lags+1:T-1) ydifflags]
        if lags > 0:
            X = np.column_stack([np.ones(tau), y_lag, ydifflags])
        else:
            X = np.column_stack([np.ones(tau), y_lag])

        # Ref: augdf.m:132 — rho = X\Y  (NOTE: regresses on levels Y, not differences)
        rho = np.linalg.lstsq(X, Y, rcond=None)[0]

        # Ref: augdf.m:134 — e = Y - X*rho
        e = Y - X @ rho

        # Ref: augdf.m:137 — s2 = e'*e/(tau-size(X,2))
        k = X.shape[1]
        s2 = float(e @ e / (tau - k))

        # Ref: augdf.m:138 — Uinv=inv(diag([tau^(0.5) tau^(1.5) tau^(0.5)*ones(1,lags)]))
        diag_vals = np.hstack([np.array([tau**0.5, tau**1.5]),
                               tau**0.5 * np.ones(lags)])
        Uinv = np.diag(1.0 / diag_vals)

        # Ref: augdf.m:139 — sel=[0 1 zeros(1,lags)]
        sel = np.zeros(2 + lags)
        sel[1] = 1.0

        # Ref: augdf.m:140 — sigp=s2*sel*(Uinv*(X'*X)*Uinv)^(-1)*sel'
        A = Uinv @ (X.T @ X) @ Uinv
        A_inv = np.linalg.inv(A)
        sigp = s2 * float(sel @ A_inv @ sel)

        # Ref: augdf.m:143 — adfstat = tau^(1.5)*(rho(2)-1)/sqrt(sigp)
        # MATLAB 1-indexed rho(2) → Python 0-indexed rho[1]
        adfstat = float(tau**1.5 * (rho[1] - 1.0) / np.sqrt(sigp))

        # Ref: augdf.m:145 — critval=norminv([.01 .05 .1 .9 .95 .99]')
        # Per AAP §0.6.2 duplication elimination: norminv(p) → norm.ppf(p)
        critval = norm.ppf(np.array([0.01, 0.05, 0.10, 0.90, 0.95, 0.99]))

        # Ref: augdf.m:146 — pval = normcdf(adfstat)
        # Per AAP §0.6.2 duplication elimination: normcdf(x) → norm.cdf(x)
        pval = float(norm.cdf(adfstat))

    # Ref: augdf.m:148 — resid=e
    resid = e

    return adfstat, pval, critval, resid
