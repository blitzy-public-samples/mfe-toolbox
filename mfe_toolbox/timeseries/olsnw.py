"""
OLS regression with Newey-West HAC standard errors.

This module provides the :func:`olsnw` function which estimates a linear
regression model by ordinary least squares and computes heteroskedasticity
and autocorrelation consistent (HAC) standard errors using the Newey-West
(Bartlett kernel) covariance estimator.

When the number of Newey-West lags is set to zero, the estimator reduces
to White's heteroskedasticity-consistent variance-covariance.

Migrated from: timeseries/olsnw.m (132 lines)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3  Date: 9/1/2005
"""

import numpy as np

from mfe_toolbox.utility.covnw import covnw


def olsnw(
    y: np.ndarray,
    x: np.ndarray,
    c: int = 1,
    nwlags: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float, np.ndarray]:
    """
    Linear regression estimation with Newey-West HAC standard errors.

    Estimates the model ``Y = X * B + epsilon`` where ``Var(epsilon) = S2``
    and reports Newey-West (Bartlett kernel) heteroskedasticity and
    autocorrelation consistent standard errors for inference.

    Parameters
    ----------
    y : array_like
        T-element vector of dependent variable data.  If passed as a row
        vector (1 × T), it is automatically transposed.
    x : array_like
        T × K matrix of independent variable data.  May be empty (0 columns)
        when ``c=1`` for an intercept-only model.
    c : {0, 1}, optional
        Whether to include a constant (intercept) term.  ``1`` (default)
        prepends a column of ones to *x*; ``0`` excludes the constant.
    nwlags : int or None, optional
        Number of lags for the Newey-West covariance estimator.  If ``None``
        (default), the bandwidth is automatically set to
        ``floor(T ** (1/3))``.  If ``0``, White's heteroskedasticity-
        consistent variance-covariance is computed (no autocorrelation
        correction).

    Returns
    -------
    b : numpy.ndarray
        K (+1 if ``c=1``) vector of estimated regression coefficients.  When
        a constant is included it is the **first** element of *b*.
    tstat : numpy.ndarray
        K (+1) vector of t-statistics computed using Newey-West HAC standard
        errors.
    s2 : float
        Newey-West HAC estimate of the error variance.
    vcvnw : numpy.ndarray
        K (+1) × K (+1) Newey-West variance-covariance matrix of the
        estimated coefficients.
    r2 : float
        R-squared of the regression.  Centered when ``c=1``, uncentered
        otherwise.
    rbar : float
        Adjusted R-squared.  Centered when ``c=1``.
    yhat : numpy.ndarray
        T-element vector of fitted values.

    Raises
    ------
    ValueError
        If *y* is not a vector, *x* has incompatible dimensions, *x* is rank
        deficient, *c* is not in ``{0, 1}``, or the model has no regressors
        and no constant.

    Notes
    -----
    The Newey-West sandwich estimator for the coefficient variance-covariance
    matrix is computed as:

    .. math::

        \\hat{V}_{NW} = (X^\\top X / T)^{-1}
                        \\; \\hat{S}_{NW} \\;
                        (X^\\top X / T)^{-1} / T

    where :math:`\\hat{S}_{NW}` is the Bartlett-kernel HAC covariance of the
    score matrix :math:`X \\odot \\varepsilon`, and :math:`\\varepsilon` are
    the OLS residuals.

    When ``c=1`` the R-squared uses centered sums of squares
    :math:`(y - \\bar{y})^\\top (y - \\bar{y})`; when ``c=0`` the uncentered
    total sum of squares :math:`y^\\top y` is used.

    References
    ----------
    Newey, W. K. and West, K. D. (1987).  "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    *Econometrica*, 55(3), 703-708.

    See Also
    --------
    mfe_toolbox.utility.covnw : Newey-West long-run covariance estimator.
    mfe_toolbox.crosssection.ols : OLS with White/HAC standard errors.

    Examples
    --------
    Regression with automatic bandwidth selection:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal(100)
    >>> x = rng.standard_normal((100, 2))
    >>> b, tstat, s2, vcvnw, r2, rbar, yhat = olsnw(y, x)

    Regression without a constant:

    >>> b, tstat, s2, vcvnw, r2, rbar, yhat = olsnw(y, x, c=0)

    Regression with a specific lag length of 10:

    >>> b, tstat, s2, vcvnw, r2, rbar, yhat = olsnw(y, x, c=1, nwlags=10)

    Regression with White heteroskedasticity-consistent standard errors:

    >>> b, tstat, s2, vcvnw, r2, rbar, yhat = olsnw(y, x, c=1, nwlags=0)
    """
    # ------------------------------------------------------------------
    # Input Validation  (Ref: olsnw.m:49-96)
    # ------------------------------------------------------------------

    # Convert inputs to float64 numpy arrays
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    # Ref: olsnw.m:61-63 — If y is a row vector, transpose to column
    if y.ndim == 1:
        # Already a 1-D vector; treat as column
        pass
    elif y.ndim == 2:
        # Ref: olsnw.m:61 — if size(y,1)<size(y,2), y=y'
        if y.shape[0] < y.shape[1]:
            y = y.T
        # Ref: olsnw.m:64-65 — Y must be a column vector
        if y.shape[1] != 1:
            raise ValueError("Y must be a column vector")
        y = y.ravel()
    else:
        raise ValueError("Y must be a column vector")

    # Ref: olsnw.m:67 — T = size(y,1)
    T: int = y.shape[0]

    # Ensure x is 2-D
    if x.ndim == 1:
        # Ref: single regressor passed as 1-D → reshape to column
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("X must be a 2-D array")

    # Ref: olsnw.m:69-79 — X validation (row count, column count, rank)
    if x.size > 0:
        # Ref: olsnw.m:70 — size(x,1)~=T
        if x.shape[0] != T:
            raise ValueError("X must have the same number of rows as Y.")
        # Ref: olsnw.m:73-74 — columns must not exceed T
        if x.shape[1] > T:
            raise ValueError(
                "The number of columns of X must be less than or equal to T"
            )
        # Ref: olsnw.m:76-77 — rank(x)<size(x,2)
        if np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValueError("X is rank deficient")

    # Ref: olsnw.m:81-86 — C must be 0 or 1
    if c is None:
        c = 1
    if c not in (0, 1):
        raise ValueError("C must be either 0 or 1.")

    # Ref: olsnw.m:88-90 — Must have constant or at least one regressor
    if c == 0 and (x.size == 0 or x.shape[1] == 0):
        raise ValueError(
            "The model must include a constant or at least one X"
        )

    # Ref: olsnw.m:91-93 — Default NW bandwidth: floor(T^(1/3))
    if nwlags is None:
        nwlags = int(np.floor(T ** (1.0 / 3.0)))

    # ------------------------------------------------------------------
    # Regressor matrix assembly  (Ref: olsnw.m:97-105)
    # ------------------------------------------------------------------

    # Ref: olsnw.m:98 — K=size(x,2)
    K: int = x.shape[1] if x.size > 0 else 0

    # Ref: olsnw.m:100-102 — Prepend column of ones if c=1
    if c == 1:
        if K > 0:
            # Ref: olsnw.m:101 — x=[ones(T,1) x]
            x = np.column_stack([np.ones(T), x])
        else:
            # Intercept-only model when x is empty
            x = np.ones((T, 1))
        K = K + 1

    # ------------------------------------------------------------------
    # OLS estimation  (Ref: olsnw.m:108-119)
    # ------------------------------------------------------------------

    # Ref: olsnw.m:109 — b=x\y  (MATLAB backslash → lstsq)
    b: np.ndarray = np.linalg.lstsq(x, y, rcond=None)[0]

    # Ref: olsnw.m:111 — yhat=x*b
    yhat: np.ndarray = x @ b

    # Ref: olsnw.m:113 — epsilon=y-yhat
    epsilon: np.ndarray = y - yhat

    # ------------------------------------------------------------------
    # HAC covariance estimation  (Ref: olsnw.m:115-119)
    # ------------------------------------------------------------------

    # Ref: olsnw.m:115 — s2=covnw(epsilon,nwlags,0)
    # covnw requires a 2-D array; epsilon is (T,) so reshape to (T, 1).
    # The third MATLAB argument 0 maps to demean=False in Python.
    s2_mat: np.ndarray = covnw(
        epsilon.reshape(-1, 1), nw_lags=nwlags, demean=False
    )
    # s2_mat is a (1, 1) array — extract as a Python float
    s2: float = float(s2_mat.item())

    # Ref: olsnw.m:117 — scores = x.*repmat(epsilon,1,K)
    # Element-wise multiplication using broadcasting: (T, K) * (T, 1)
    scores: np.ndarray = x * epsilon[:, np.newaxis]

    # Ref: olsnw.m:118 — XpXi=(x'*x/T)^(-1)
    XpXi: np.ndarray = np.linalg.inv(x.T @ x / T)

    # Ref: olsnw.m:119 — vcvnw = XpXi * covnw(scores,nwlags,0) * XpXi /T
    vcvnw: np.ndarray = (
        XpXi @ covnw(scores, nw_lags=nwlags, demean=False) @ XpXi / T
    )

    # ------------------------------------------------------------------
    # t-statistics  (Ref: olsnw.m:122)
    # ------------------------------------------------------------------

    # Ref: olsnw.m:122 — tstat=b./sqrt(diag(vcvnw))
    tstat: np.ndarray = b / np.sqrt(np.diag(vcvnw))

    # ------------------------------------------------------------------
    # R-squared and adjusted R-squared  (Ref: olsnw.m:124-131)
    # ------------------------------------------------------------------

    # Sum of squared residuals (common to both centered and uncentered)
    ss_res: float = float(np.sum(epsilon * epsilon))

    if c == 1:
        # Ref: olsnw.m:126-128 — Centered R² when constant is included
        ytilde: np.ndarray = y - np.mean(y)
        ss_tot: float = float(np.sum(ytilde * ytilde))
    else:
        # Ref: olsnw.m:130-131 — Uncentered R² when no constant
        ss_tot = float(np.sum(y * y))

    r2: float = 1.0 - ss_res / ss_tot
    # Ref: olsnw.m:128 or 131 — Adjusted R²
    # K already includes the constant when c=1, so T-K is the correct dof
    rbar: float = 1.0 - (ss_res / ss_tot) * (T - 1) / (T - K)

    return b, tstat, s2, vcvnw, r2, rbar, yhat
