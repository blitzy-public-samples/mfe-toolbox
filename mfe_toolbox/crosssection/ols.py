"""
OLS regression with White heteroskedasticity-robust standard errors.

Migrated from crosssection/ols.m (127 lines) — MFE Toolbox Version 4.0.
Provides OLS estimation with both homoskedastic and White heteroskedasticity-
robust variance-covariance matrices, t-statistics, R-squared, and adjusted
R-squared.

Author: Kevin Sheppard
Original MATLAB Revision: 3, Date: 9/1/2005
Python migration: faithful 1:1 translation with numpy.linalg.lstsq
"""

import numpy as np


def ols(y, x, c=1):
    """
    Linear regression with homoskedastic and White heteroskedasticity-robust
    standard errors.

    Estimates the model ``Y = X * B + epsilon`` where ``Var(epsilon) = S2``.

    Parameters
    ----------
    y : numpy.ndarray
        N x 1 vector of dependent data. If provided as a row vector, it is
        automatically transposed.
    x : numpy.ndarray
        N x K matrix of independent data. Can have 0 columns if ``c=1``
        (intercept-only model). A 1-D array is treated as a single column.
    c : int, optional
        1 (default) to include a constant as the first regressor, 0 to
        exclude a constant.

    Returns
    -------
    b : numpy.ndarray
        (K+c) x 1 parameter vector. If a constant is included (``c=1``),
        it appears as the first element.
    tstat : numpy.ndarray
        (K+c) x 1 t-statistics computed using White heteroskedasticity-
        robust standard errors.
    s2 : float
        Estimated residual variance of the regression
        (``epsilon' * epsilon / (N - K - c)``).
    vcv : numpy.ndarray
        (K+c) x (K+c) parameter covariance matrix assuming
        homoskedasticity.
    vcvwhite : numpy.ndarray
        (K+c) x (K+c) White heteroskedasticity-robust parameter
        covariance matrix (sandwich estimator).
    R2 : float
        R-squared of the regression. Centered when ``c=1``, uncentered
        when ``c=0``.
    Rbar : float
        Adjusted R-squared. Centered when ``c=1``, uncentered when
        ``c=0``.
    yhat : numpy.ndarray
        N x 1 fitted values of the dependent variable.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, ``x`` is rank-deficient,
        ``c`` is not 0 or 1, or the model contains neither a constant
        nor regressors.

    See Also
    --------
    mfe_toolbox.timeseries.olsnw : OLS with Newey-West HAC standard errors.

    Examples
    --------
    Estimate a regression with a constant:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal((100, 1))
    >>> x = rng.standard_normal((100, 3))
    >>> b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, x)
    >>> b.shape
    (4, 1)

    Estimate a regression without a constant:

    >>> b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, x, 0)
    >>> b.shape
    (3, 1)
    """
    # ====================================================================
    # Input Checking — Ref: ols.m:41-80
    # ====================================================================

    # Convert inputs to float64 numpy arrays for numerical consistency
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    # Ensure y is 2-D column vector
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Ref: ols.m:48-49 — Auto-transpose if row vector
    # MATLAB: if size(y,1)<size(y,2), y=y';
    if y.shape[0] < y.shape[1]:
        y = y.T

    # Ref: ols.m:51-52 — Column check
    if y.shape[1] != 1:
        raise ValueError('Y must be a column vector')

    # Ref: ols.m:54
    N = y.shape[0]

    # Ensure x is 2-D; treat 1-D array as single-column regressor
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        if x.size == 0:
            # Empty array — intercept-only model (if c=1)
            x = np.empty((N, 0), dtype=np.float64)
        else:
            x = x.reshape(-1, 1)

    # Ref: ols.m:56-66 — X dimension and rank checks (if x not empty)
    if x.size > 0:
        # Ref: ols.m:57-58
        if x.shape[0] != N:
            raise ValueError('X must have the same number of rows as Y.')
        # Ref: ols.m:60-61 — Preserve original MATLAB error text (including typo)
        if x.shape[1] > N:
            raise ValueError(
                'The number of columns of X must be grater than or equal to T'
            )
        # Ref: ols.m:63-64
        if np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValueError('X is rank deficient')

    # Ref: ols.m:71-72 — c validation
    if c not in (0, 1):
        raise ValueError('C must be either 0 or 1.')

    # Ref: ols.m:75-77 — Empty model check
    if c == 0 and x.shape[1] == 0:
        raise ValueError(
            'The model must include a constant or at least one X'
        )

    # ====================================================================
    # Post-validation and constant prepending — Ref: ols.m:82-94
    # ====================================================================

    # Ref: ols.m:83
    K = x.shape[1]

    # Ref: ols.m:85-86 — Full-rank check
    if K > 0 and np.linalg.matrix_rank(x) != K:
        raise ValueError('X is must be of full rank.')

    # Ref: ols.m:88-93 — Prepend constant column if c=1
    if c == 1:
        # Ref: ols.m:89 — x=[ones(N,1) x]
        x = np.column_stack([np.ones((N, 1), dtype=np.float64), x])
        # Ref: ols.m:90-91 — Check that adding a constant did not create
        # rank deficiency (indicates x already contained a constant column)
        if np.linalg.matrix_rank(x) != K + 1:
            raise ValueError(
                'X appears to contains a constant column.  '
                'Use C to add a constant.'
            )
        # Ref: ols.m:93
        K = K + 1

    # ====================================================================
    # OLS Estimation — Ref: ols.m:96-117
    # ====================================================================

    # Ref: ols.m:97 — Coefficient estimation via left-divide (x\y)
    # AAP mandates numpy.linalg.lstsq for OLS regression
    b = np.linalg.lstsq(x, y, rcond=None)[0]  # (K, 1) column vector

    # Ref: ols.m:103 — Fitted values
    yhat = x @ b  # (N, 1)

    # Ref: ols.m:105 — Residuals
    epsilon = y - yhat  # (N, 1)

    # Ref: ols.m:107 — Estimated residual variance (scalar)
    # MATLAB: s2=epsilon'*epsilon/(N-K)
    # epsilon is (N,1) so epsilon.T @ epsilon is (1,1); extract scalar via .item()
    s2 = (epsilon.T @ epsilon).item() / (N - K)

    # Ref: ols.m:109 — Homoskedastic parameter covariance
    # MATLAB: vcv=s2*(x'*x)^(-1)
    XpX_inv = np.linalg.inv(x.T @ x)
    vcv = s2 * XpX_inv  # (K, K)

    # Ref: ols.m:111-115 — White heteroskedasticity-robust VCV (sandwich)
    # MATLAB: scores = x.*repmat(epsilon,1,K)
    # Python broadcasting: (N, K) * (N, 1) → (N, K)
    scores = x * epsilon  # (N, K) element-wise scaled columns

    # Ref: ols.m:112 — XeeX = scores'*scores
    XeeX = scores.T @ scores  # (K, K)

    # Ref: ols.m:114 — XpXi = (x'*x)^(-1)
    XpXi = np.linalg.inv(x.T @ x)  # (K, K)

    # Ref: ols.m:115 — White's sandwich VCV
    vcvwhite = XpXi @ XeeX @ XpXi  # (K, K)

    # Ref: ols.m:117 — T-statistics using White robust standard errors
    # MATLAB: tstat=b./sqrt(diag(vcvwhite))
    # np.diag returns (K,) shape; reshape to (K, 1) for element-wise division
    tstat = b / np.sqrt(np.diag(vcvwhite)).reshape(-1, 1)  # (K, 1)

    # ====================================================================
    # R-squared and Adjusted R-squared — Ref: ols.m:119-127
    # ====================================================================

    # Sum of squared errors — extract scalar from (1,1) matrix via .item()
    sse = (epsilon.T @ epsilon).item()

    if c == 1:
        # Ref: ols.m:120-123 — Centered R² (constant included)
        # MATLAB: ytilde=y-mean(y)
        ytilde = y - np.mean(y)
        sst = (ytilde.T @ ytilde).item()
        # Ref: ols.m:122
        R2 = 1.0 - sse / sst
        # Ref: ols.m:123
        Rbar = 1.0 - (sse / sst) * (N - 1) / (N - K)
    else:
        # Ref: ols.m:124-126 — Uncentered R² (no constant)
        sst = (y.T @ y).item()
        # Ref: ols.m:125
        R2 = 1.0 - sse / sst
        # Ref: ols.m:126
        Rbar = 1.0 - (sse / sst) * (N - 1) / (N - K)

    return b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat
