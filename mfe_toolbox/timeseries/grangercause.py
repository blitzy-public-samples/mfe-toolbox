"""
Granger Causality Testing.

Migrated from ``timeseries/grangercause.m`` — Author: Kevin Sheppard
(Revision 3.0, Date: 1/1/2007)

Implements Granger causality testing with a variance-covariance matrix
estimated under four error structure assumptions:

1. Conditionally Homoskedastic and Uncorrelated
2. Conditionally Homoskedastic but Correlated
3. Heteroskedastic but Conditionally Uncorrelated
4. Heteroskedastic and Correlated

Three inference methods are supported:

1. Likelihood Ratio (LR)
2. Lagrange Multiplier (LM) test
3. Wald test

The test is based on a VAR specification:

    y(t)' = CONST + P(1) * y(t-1) + P(2) * y(t-2) + ... + P(m) * y(t-m)

where P(j) are K x K parameter matrices and CONST is a K x 1 parameter
vector (if ``constant == 1``).

Notes
-----
The MATLAB source ``grangercause.m`` includes a nested helper function
``vectorarscorecov`` (lines 334-358) for computing the VAR score covariance
matrix.  This is translated as the private ``_vectorar_score_cov`` helper
in this module.

The MATLAB source uses a mean-padding trick to avoid losing initial
observations: each variable is prepended with ``m`` copies of its sample
mean before constructing the lag matrix.  This results in the design
matrix ``X`` having ``T`` (not ``T - m``) rows, which differs from the
general-purpose :func:`~mfe_toolbox.timeseries.vectorar.vectorar` function.

References
----------
Sheppard, K. (2009). MFE Toolbox Version 4.0, grangercause.m.
Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press.
Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
    Springer.
"""

import numpy as np
from scipy.stats import chi2

from mfe_toolbox.utility.newlagmatrix import newlagmatrix
from mfe_toolbox.timeseries.vectorar import vectorar  # noqa: F401 — cross-ref

__all__ = ['grangercause']


def _vectorar_score_cov(errors, X, het, uncorr, K, T, Np):
    """
    Compute the VAR score covariance matrix under specified error assumptions.

    This is a faithful translation of the nested MATLAB function
    ``vectorarscorecov`` from ``grangercause.m`` (lines 334-358).

    Parameters
    ----------
    errors : numpy.ndarray
        ``(T, K)`` matrix of VAR residuals.
    X : numpy.ndarray
        ``(T, Np)`` design matrix (regressors per equation).
    het : int
        Heteroskedasticity flag: 0 = homoskedastic, 1 = heteroskedastic.
    uncorr : int
        Uncorrelated flag: 0 = correlated, 1 = uncorrelated.
    K : int
        Number of variables in the system.
    T : int
        Number of observations.
    Np : int
        Number of regressors per equation (``p * K + constant``).

    Returns
    -------
    S : numpy.ndarray
        ``(K * Np, K * Np)`` score covariance matrix.

    Notes
    -----
    The four covariance modes correspond to the following estimators:

    - ``het=0, uncorr=1``: ``kron(diag(diag(s2)), X'X/T)``
    - ``het=0, uncorr=0``: ``kron(s2, X'X/T)``
    - ``het=1, uncorr=1``: Block-diagonal heteroskedastic outer product
    - ``het=1, uncorr=0``: Full heteroskedastic outer product

    Ref: grangercause.m:334-358
    """
    # Ref: grangercause.m:335 — s2=errors'*errors/T
    s2 = errors.T @ errors / T
    # Ref: grangercause.m:336 — XpX=X'*X/T
    XpX = X.T @ X / T

    if not het and uncorr:
        # Ref: grangercause.m:338 — S=kron(diag(diag(s2)),XpX)
        # Homoskedastic, uncorrelated: Kronecker of diagonal variances with XpX
        S = np.kron(np.diag(np.diag(s2)), XpX)
    elif not het and not uncorr:
        # Ref: grangercause.m:340 — S=kron(s2,XpX)
        # Homoskedastic, correlated: Kronecker of full covariance with XpX
        S = np.kron(s2, XpX)
    elif het and uncorr:
        # Ref: grangercause.m:342-351
        # Heteroskedastic, uncorrelated: block-diagonal score covariance
        # Ref: grangercause.m:342 — X2=repmat(X,1,K)
        X2 = np.tile(X, (1, K))  # (T, Np*K)
        # Ref: grangercause.m:343 — e2=reshape(repmat(errors,Np,1),T,K*Np)
        # Each column of errors is repeated Np times consecutively
        e2 = np.repeat(errors, Np, axis=1)  # (T, K*Np)
        # Ref: grangercause.m:344 — s=X2.*e2
        s = X2 * e2
        # Ref: grangercause.m:345 — s=s-repmat(mean(s),T,1)
        # Demean each column; NumPy broadcasting handles the replication
        s = s - np.mean(s, axis=0)
        # Ref: grangercause.m:346 — S=zeros(Np*K)
        S = np.zeros((Np * K, Np * K))
        for i in range(K):
            # Ref: grangercause.m:348 — sel=(i-1)*Np+1:i*Np
            # Python 0-indexed slice
            sel = slice(i * Np, (i + 1) * Np)
            # Ref: grangercause.m:349 — temp = s(:,sel)
            temp = s[:, sel]
            # Ref: grangercause.m:350 — S(sel,sel)=temp'*temp/T
            S[sel, sel] = temp.T @ temp / T
    else:
        # het and not uncorr
        # Ref: grangercause.m:352-357
        # Heteroskedastic, correlated: full score covariance
        # Ref: grangercause.m:353 — X2=repmat(X,1,K)
        X2 = np.tile(X, (1, K))
        # Ref: grangercause.m:354 — e2=reshape(repmat(errors,Np,1),T,K*Np)
        e2 = np.repeat(errors, Np, axis=1)
        # Ref: grangercause.m:355 — s=X2.*e2
        s = X2 * e2
        # Ref: grangercause.m:356 — s=s-repmat(mean(s),T,1)
        s = s - np.mean(s, axis=0)
        # Ref: grangercause.m:357 — S=s'*s/T
        S = s.T @ s / T

    return S


def grangercause(y, constant, lags, het=1, uncorr=0, inference=1):
    """
    Granger causality testing with multiple VCV estimators and inference methods.

    Tests whether each variable in a multivariate system Granger-causes
    each other variable, using a VAR framework.  Returns both pairwise
    test statistics (K x K matrix) and joint test statistics (K x 1 vector).

    Parameters
    ----------
    y : numpy.ndarray
        ``(T, K)`` matrix of data, where ``T`` is the number of observations
        and ``K`` is the number of variables.
    constant : int
        Include a constant in the VAR:

        - ``1`` — include constant
        - ``0`` — exclude constant
    lags : array_like
        Non-negative integer vector of VAR lag orders to include.
        Must contain positive unique integers.  Automatically sorted.
    het : int, optional
        Error covariance estimator type:

        - ``0`` — Homoskedastic
        - ``1`` — Heteroskedastic **[default]**
    uncorr : int, optional
        Assumed error correlation structure:

        - ``0`` — Correlated errors **[default]**
        - ``1`` — Uncorrelated errors
    inference : int, optional
        Inference method:

        - ``1`` — Likelihood ratio **[default]**
        - ``2`` — LM test
        - ``3`` — Wald test

    Returns
    -------
    stat : numpy.ndarray
        ``(K, K)`` matrix of Granger causality test statistics.
        ``stat[i, j]`` corresponds to a test that ``y[:, i]`` is not
        caused by ``y[:, j]``.
    pval : numpy.ndarray
        ``(K, K)`` matrix of p-values corresponding to *stat*.
    stat_all : numpy.ndarray
        ``(K, 1)`` vector of joint Granger causality test statistics.
        ``stat_all[i]`` corresponds to a test that ``y[:, i]`` is not
        caused by any ``y[:, j]``, ``j != i``.
    pval_all : numpy.ndarray
        ``(K, 1)`` vector of p-values corresponding to *stat_all*.

    Raises
    ------
    ValueError
        If any input fails validation:

        - ``y`` is not a 2-D array
        - ``constant`` is not 0 or 1
        - ``lags`` contains non-positive, non-integer, or duplicate values
        - ``het`` is not 0 or 1
        - ``uncorr`` is not 0 or 1
        - ``inference`` is not 1, 2, or 3

    Examples
    --------
    Conduct GC testing in a VAR(1) with a constant:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal((200, 3))
    >>> stat, pval, stat_all, pval_all = grangercause(y, 1, [1])

    VAR(3) with no constant:

    >>> stat, pval, stat_all, pval_all = grangercause(y, 0, [1, 2, 3])

    VAR with lags 1 and 3, constant, heteroskedastic, correlated, LR test:

    >>> stat, pval, stat_all, pval_all = grangercause(y, 1, [1, 3])

    Wald test with homoskedastic, uncorrelated errors:

    >>> stat, pval, stat_all, pval_all = grangercause(
    ...     y, 1, [1], het=0, uncorr=1, inference=3)

    See Also
    --------
    vectorar : General-purpose VAR estimation.
    mfe_toolbox.timeseries.vectorarvcv : VAR variance-covariance estimation.

    Notes
    -----
    Migrated from ``timeseries/grangercause.m`` (Kevin Sheppard, Revision 3.0,
    Date: 1/1/2007).

    The implementation preserves the MATLAB source's mean-padding approach
    for initial conditions, which differs from the general
    :func:`~mfe_toolbox.timeseries.vectorar.vectorar` function that trims
    the first ``m`` observations.  This ensures that the design matrix ``X``
    has the full ``T`` rows, matching the original numerical output.

    The test statistic includes a small-sample correction factor:

    - Pairwise: ``T - p * K**2 + p``
    - Joint:    ``T - p * K**2 + (K - 1) * p``

    P-values are computed from the chi-squared distribution with degrees
    of freedom ``p`` (pairwise) or ``(K - 1) * p`` (joint), where ``p``
    is ``len(lags)``.

    References
    ----------
    .. [1] Sheppard, K. (2009). MFE Toolbox Version 4.0, grangercause.m.
    .. [2] Hamilton, J.D. (1994). *Time Series Analysis*. Ch. 11.
    .. [3] Lütkepohl, H. (2005). *New Introduction to Multiple Time
       Series Analysis*. Ch. 3.
    """
    # ==================================================================
    # Input Validation
    # Ref: grangercause.m:63-127
    # ==================================================================

    # Ref: grangercause.m:77-79 — Check Y
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError('Y must be T by K')

    # Ref: grangercause.m:81-86 — Check constant
    if not np.isscalar(constant):
        raise ValueError('CONSTANT must be either 0 or 1')
    constant = int(constant)
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be either 0 or 1')

    # Ref: grangercause.m:88-103 — Check lags
    lags_arr = np.asarray(lags, dtype=np.float64).ravel()
    if lags_arr.ndim != 1:
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )
    # Ref: grangercause.m:91-93 — Ensure row vector orientation (handled
    # by ravel above)
    # Ref: grangercause.m:94-96 — All lags must be positive
    if not np.all(lags_arr > 0):
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )
    # Ref: grangercause.m:97-99 — All lags must be integers
    if not np.all(np.floor(lags_arr) == lags_arr):
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )
    # Ref: grangercause.m:100-102 — All lags must be unique
    if len(lags_arr) != len(np.unique(lags_arr)):
        raise ValueError('LAGS must be a vector of unique elements')
    # Ref: grangercause.m:103 — Sort lags
    lags_arr = np.sort(lags_arr).astype(np.int64)

    # Ref: grangercause.m:105-110 — Check het
    if not np.isscalar(het):
        raise ValueError('HET must be either 0 or 1')
    het = int(het)
    if het not in (0, 1):
        raise ValueError('HET must be either 0 or 1')

    # Ref: grangercause.m:112-117 — Check uncorr
    if not np.isscalar(uncorr):
        raise ValueError('UNCORR must be either 0 or 1')
    uncorr = int(uncorr)
    if uncorr not in (0, 1):
        raise ValueError('UNCORR must be either 0 or 1')

    # Ref: grangercause.m:119-124 — Check inference
    if not np.isscalar(inference):
        raise ValueError('INFERENCE must be either 1, 2, or 3')
    inference = int(inference)
    if inference not in (1, 2, 3):
        raise ValueError('INFERENCE must be either 1, 2, or 3')

    # ==================================================================
    # Core Setup
    # Ref: grangercause.m:145-173
    # ==================================================================

    T = y.shape[0]          # Ref: grangercause.m:145 — T=size(y,1)
    K = y.shape[1]          # Ref: grangercause.m:146 — K=size(y,2)
    p = len(lags_arr)       # Ref: grangercause.m:147 — P=length(lags)
    m = int(np.max(lags_arr))  # Ref: grangercause.m:148 — m=max(lags)

    # ------------------------------------------------------------------
    # Build lag matrices for each variable with mean-padding
    # Ref: grangercause.m:149-154
    # MATLAB prepends m copies of the variable mean to avoid trimming
    # the first m observations, keeping X with T rows.
    # ------------------------------------------------------------------
    ylags = []
    for k in range(K):
        # Ref: grangercause.m:151-152 — ytemp = [ones(m,1)*mean(y(:,k)); y(:,k)]
        ymean_k = np.mean(y[:, k])
        ytemp = np.concatenate([np.ones(m) * ymean_k, y[:, k]])
        # Ref: grangercause.m:153 — [nothing,ylags{k}]=newlagmatrix(ytemp,m,0)
        # newlagmatrix returns (y_trimmed, x); only x (the lag matrix) is needed.
        # ytemp has length (m + T), so x has (m + T - m) = T rows and m columns.
        _, yl = newlagmatrix(ytemp, m, 0)
        ylags.append(yl)

    # ------------------------------------------------------------------
    # Build design matrix X
    # Ref: grangercause.m:156-168
    # Total columns: constant + p * K
    # Column ordering: [constant | var1_lag1 var2_lag1 ... varK_lag1 |
    #                   var1_lag2 ... | ... | var1_lagP ... varK_lagP]
    # ------------------------------------------------------------------
    n_cols = constant + p * K  # Ref: grangercause.m:156 (auto-extended by MATLAB)
    X = np.zeros((T, n_cols))
    col_idx = 0

    if constant:
        # Ref: grangercause.m:159 — X(:,1)=ones(T,1)
        X[:, 0] = 1.0
        col_idx = 1

    # Ref: grangercause.m:162-168 — Fill lagged variable columns
    for i in range(p):
        for k in range(K):
            # Ref: grangercause.m:165 — X(:,index)=ylags{k}(:,lags(i))
            # MATLAB uses 1-indexed column lags(i); Python 0-indexed: lags_arr[i]-1
            X[:, col_idx] = ylags[k][:, lags_arr[i] - 1]
            col_idx += 1

    # ------------------------------------------------------------------
    # OLS Estimation (unrestricted model)
    # Ref: grangercause.m:171 — paramvec = X\y
    # ------------------------------------------------------------------
    paramvec, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # paramvec shape: (n_cols, K) — each column has parameters for one equation

    # Ref: grangercause.m:172 — errors = y-X*paramvec
    errors = y - X @ paramvec

    # Ref: grangercause.m:173 — s2=errors'*errors/T
    s2 = errors.T @ errors / T

    # ==================================================================
    # Restricted Models — Pairwise (drop variable j from equation i)
    # Ref: grangercause.m:174-191
    # e[i][j] = restricted residuals for equation i when all lags of
    #           variable j are excluded from equation i.
    # ==================================================================
    e = [[None] * K for _ in range(K)]

    for i in range(K):
        for j in range(K):
            # Ref: grangercause.m:179 — allcols = 1:size(X,2)
            all_cols = np.arange(n_cols)  # 0-indexed

            # Ref: grangercause.m:181-185 — Compute columns to drop
            # (variable j's lags in the X matrix)
            if constant:
                # Ref: grangercause.m:182 — drop = (j:K:size(X,2)-1)+constant
                # MATLAB 1-indexed: j_matlab+constant : K : n_cols
                # Python 0-indexed: j + constant : K : n_cols
                # This selects columns corresponding to variable j's lags
                drop = np.arange(j + constant, n_cols, K)
            else:
                # Ref: grangercause.m:184 — drop = (j:K:size(X,2))
                # MATLAB 1-indexed: j_matlab : K : n_cols
                # Python 0-indexed: j : K : n_cols
                drop = np.arange(j, n_cols, K)

            # Ref: grangercause.m:186 — remain = setdiff(allcols, drop)
            remain = np.setdiff1d(all_cols, drop)

            # Ref: grangercause.m:187 — beta = tempX(:,remain)\y(:,i)
            beta, _, _, _ = np.linalg.lstsq(X[:, remain], y[:, i], rcond=None)

            # Ref: grangercause.m:189 — e{i,j} = y(:,i)-tempX(:,remain)*beta
            e[i][j] = y[:, i] - X[:, remain] @ beta

    # ==================================================================
    # Restricted Models — Joint (keep only variable i's own lags + constant)
    # Ref: grangercause.m:192-204
    # eall[i] = restricted residuals for equation i when ALL other
    #           variables' lags are excluded.
    # ==================================================================
    eall = [None] * K

    for i in range(K):
        if constant:
            # Ref: grangercause.m:198 — remain = [constant (i:K:(size(X,2)-1))+constant]
            # MATLAB 1-indexed: [1, (i_matlab:K:(n_cols-1))+1]
            # Python 0-indexed: [0] union arange(i+constant, n_cols, K)
            remain_i_lags = np.arange(i + constant, n_cols, K)
            remain = np.concatenate([[0], remain_i_lags]).astype(np.int64)
        else:
            # Ref: grangercause.m:200 — remain = i:K:size(X,2)-1
            # MATLAB 1-indexed, operator precedence: i_matlab : K : (n_cols - 1)
            # Python 0-indexed: i : K : (n_cols - 1)
            # NOTE: The MATLAB original uses size(X,2)-1 in the no-constant
            # case for the joint test, which may clip the last lag for
            # variable K when K > 1.  This is faithfully preserved.
            remain = np.arange(i, n_cols - 1, K).astype(np.int64)

        # Ref: grangercause.m:202 — beta = tempX(:,remain)\y(:,i)
        beta, _, _, _ = np.linalg.lstsq(X[:, remain], y[:, i], rcond=None)

        # Ref: grangercause.m:203 — eall{i} = y(:,i)-tempX(:,remain)*beta
        eall[i] = y[:, i] - X[:, remain] @ beta

    # ==================================================================
    # Number of parameters per equation
    # Ref: grangercause.m:206 — Np = p*K+constant
    # ==================================================================
    Np = p * K + constant

    # ==================================================================
    # Compute Test Statistics
    # Ref: grangercause.m:207-327
    # ==================================================================
    stat = np.zeros((K, K))
    stat_all = np.zeros((K, 1))

    # Small-sample correction factors
    # Ref: grangercause.m:216,223
    corr_pair = T - p * K ** 2 + p
    corr_joint = T - p * K ** 2 + (K - 1) * p

    if inference == 1 and het == 0:
        # ==============================================================
        # Homoskedastic Likelihood Ratio
        # Ref: grangercause.m:210-224
        # ==============================================================
        log_det_s2 = np.log(np.linalg.det(s2))

        # --- Pairwise tests ---
        for i in range(K):
            for j in range(K):
                # Ref: grangercause.m:213-216
                # Replace column i of unrestricted errors with restricted
                e2 = errors.copy()
                e2[:, i] = e[i][j]
                sR = e2.T @ e2 / T
                stat[i, j] = corr_pair * (
                    np.log(np.linalg.det(sR)) - log_det_s2
                )

        # --- Joint tests ---
        for i in range(K):
            # Ref: grangercause.m:219-224
            e2 = errors.copy()
            e2[:, i] = eall[i]
            sR = e2.T @ e2 / T
            stat_all[i, 0] = corr_joint * (
                np.log(np.linalg.det(sR)) - log_det_s2
            )

    elif inference == 1 and het == 1:
        # ==============================================================
        # Heteroskedastic Robust Likelihood Ratio
        # Ref: grangercause.m:227-251
        # Key: score covariance S is estimated from unrestricted errors
        #      (errors under the alternative hypothesis).
        # ==============================================================

        # --- Pairwise tests ---
        for i in range(K):
            for j in range(K):
                # Ref: grangercause.m:230-238
                e2 = errors.copy()
                e2[:, i] = e[i][j]
                # Ref: grangercause.m:232 — X2=repmat(X,1,K)
                X2 = np.tile(X, (1, K))
                # Ref: grangercause.m:233 — e2=reshape(repmat(e2,Np,1),T,K*Np)
                e2_expanded = np.repeat(e2, Np, axis=1)
                # Ref: grangercause.m:234 — s=X2.*e2
                s = X2 * e2_expanded
                # Ref: grangercause.m:235 — sbar=mean(s)
                sbar = np.mean(s, axis=0)
                # Ref: grangercause.m:237 — S estimated from unrestricted errors
                S = _vectorar_score_cov(errors, X, het, uncorr, K, T, Np)
                S_inv = np.linalg.inv(S)
                # Ref: grangercause.m:238
                stat[i, j] = corr_pair * (sbar @ S_inv @ sbar)

        # --- Joint tests ---
        for i in range(K):
            # Ref: grangercause.m:241-250
            e2 = errors.copy()
            e2[:, i] = eall[i]
            X2 = np.tile(X, (1, K))
            e2_expanded = np.repeat(e2, Np, axis=1)
            s = X2 * e2_expanded
            sbar = np.mean(s, axis=0)
            S = _vectorar_score_cov(errors, X, het, uncorr, K, T, Np)
            S_inv = np.linalg.inv(S)
            # Ref: grangercause.m:250
            stat_all[i, 0] = corr_joint * (sbar @ S_inv @ sbar)

    elif inference == 2:
        # ==============================================================
        # LM Test (all VCV modes)
        # Ref: grangercause.m:254-280
        # Key: score covariance S is estimated from restricted errors
        #      (errors under the null hypothesis).
        # ==============================================================

        # --- Pairwise tests ---
        for i in range(K):
            for j in range(K):
                # Ref: grangercause.m:257-266
                # Compute scores using restricted residuals
                e2_for_score = errors.copy()
                e2_for_score[:, i] = e[i][j]
                X2 = np.tile(X, (1, K))
                e2_expanded = np.repeat(e2_for_score, Np, axis=1)
                s = X2 * e2_expanded
                sbar = np.mean(s, axis=0)
                # Ref: grangercause.m:263-265 — S from restricted errors
                e2_for_cov = errors.copy()
                e2_for_cov[:, i] = e[i][j]
                S = _vectorar_score_cov(e2_for_cov, X, het, uncorr, K, T, Np)
                S_inv = np.linalg.inv(S)
                # Ref: grangercause.m:266
                stat[i, j] = corr_pair * (sbar @ S_inv @ sbar)

        # --- Joint tests ---
        for i in range(K):
            # Ref: grangercause.m:269-279
            e2_for_score = errors.copy()
            e2_for_score[:, i] = eall[i]
            X2 = np.tile(X, (1, K))
            e2_expanded = np.repeat(e2_for_score, Np, axis=1)
            s = X2 * e2_expanded
            sbar = np.mean(s, axis=0)
            e2_for_cov = errors.copy()
            e2_for_cov[:, i] = eall[i]
            S = _vectorar_score_cov(e2_for_cov, X, het, uncorr, K, T, Np)
            S_inv = np.linalg.inv(S)
            # Ref: grangercause.m:279
            stat_all[i, 0] = corr_joint * (sbar @ S_inv @ sbar)

    elif inference == 3:
        # ==============================================================
        # Wald Test
        # Ref: grangercause.m:282-326
        # ==============================================================
        # Ref: grangercause.m:283 — XpXi = ((X'*X)/T)^(-1)
        XpXi = np.linalg.inv(X.T @ X / T)
        # Ref: grangercause.m:284 — Ainv = kron(eye(K), XpXi)
        Ainv = np.kron(np.eye(K), XpXi)
        # Ref: grangercause.m:285 — B = vectorarscorecov(errors,X,het,uncorr,K,T,Np)
        B = _vectorar_score_cov(errors, X, het, uncorr, K, T, Np)
        # Ref: grangercause.m:286 — V = Ainv*B*Ainv
        V = Ainv @ B @ Ainv

        # --- Pairwise Wald tests ---
        # Ref: grangercause.m:291-308
        for i in range(K):
            for j in range(K):
                # Ref: grangercause.m:293 — temp=zeros(size(paramvec))'
                # paramvec is (Np, K), so zeros(size(paramvec))' is (K, Np)
                temp = np.zeros((K, Np))

                # Ref: grangercause.m:295-299 — Select parameter columns
                # corresponding to variable j's lags
                if constant:
                    # Ref: grangercause.m:296 — pl = (j:K:size(X,2)-1)+constant
                    # MATLAB 1-indexed: (j_m:K:(n_cols-1))+1 where j_m = j+1
                    # Python 0-indexed: (j+constant) : K : n_cols
                    pl_cols = np.arange(j + constant, n_cols, K)
                else:
                    # Ref: grangercause.m:298 — pl = (j:K:size(X,2))
                    # MATLAB 1-indexed: j_m:K:n_cols where j_m = j+1
                    # Python 0-indexed: j : K : n_cols
                    pl_cols = np.arange(j, n_cols, K)

                # Ref: grangercause.m:301 — temp(i,pl)=1
                temp[i, pl_cols] = 1.0

                # Ref: grangercause.m:302-303 — temp=temp'; temp=temp(:)
                # Transpose (K,Np) → (Np,K), then flatten column-major
                temp_flat = temp.T.ravel(order='F')

                # Ref: grangercause.m:304 — pl = find(temp)
                # MATLAB find returns 1-indexed; Python where returns 0-indexed
                pl = np.where(temp_flat > 0)[0]

                # Ref: grangercause.m:305-306 — p = paramvec; p = p(:)
                # Flatten paramvec (Np,K) column-major to (Np*K,)
                pvec = paramvec.ravel(order='F')

                # Ref: grangercause.m:307
                # stat(i,j) = (T-P*K^2+P) * p(pl)' * V(pl,pl)^(-1) * p(pl)
                V_sub = V[np.ix_(pl, pl)]
                p_sub = pvec[pl]
                stat[i, j] = corr_pair * (
                    p_sub @ np.linalg.inv(V_sub) @ p_sub
                )

        # --- Joint Wald tests ---
        # Ref: grangercause.m:310-326
        for i in range(K):
            # Ref: grangercause.m:311 — temp=zeros(size(paramvec))'
            temp = np.zeros((K, Np))

            # Ref: grangercause.m:313-317 — Determine "remain" columns
            # (own lags + constant), then drop = complement
            if constant:
                # Ref: grangercause.m:314
                # remain = [constant (i:K:(size(X,2)-1))+constant]
                # MATLAB 1-indexed: [1, (i_m:K:(n_cols-1))+1]
                # Python 0-indexed: [0] union arange(i+constant, n_cols, K)
                remain_cols = np.concatenate(
                    [[0], np.arange(i + constant, n_cols, K)]
                ).astype(np.int64)
            else:
                # Ref: grangercause.m:316 — remain = (i:K:size(X,2)-1)
                # MATLAB 1-indexed: i_m:K:(n_cols-1)
                # Python 0-indexed: i:K:(n_cols-1)
                # NOTE: faithfully preserves MATLAB's size(X,2)-1
                remain_cols = np.arange(i, n_cols - 1, K).astype(np.int64)

            # Ref: grangercause.m:318 — pl = setdiff(1:size(X,2), remain)
            pl_cols = np.setdiff1d(np.arange(n_cols), remain_cols)

            # Ref: grangercause.m:319 — temp(i,pl)=1
            temp[i, pl_cols] = 1.0

            # Ref: grangercause.m:320-322 — temp=temp'; temp=temp(:); pl=find(temp)
            temp_flat = temp.T.ravel(order='F')
            pl = np.where(temp_flat > 0)[0]

            # Ref: grangercause.m:323-325
            pvec = paramvec.ravel(order='F')
            V_sub = V[np.ix_(pl, pl)]
            p_sub = pvec[pl]
            stat_all[i, 0] = corr_joint * (
                p_sub @ np.linalg.inv(V_sub) @ p_sub
            )

    # ==================================================================
    # P-values from chi-squared distribution
    # Ref: grangercause.m:329-330
    # chi2cdf (from duplication/) → scipy.stats.chi2.cdf
    # ==================================================================
    # Ref: grangercause.m:329 — pval=1-chi2cdf(stat,length(lags))
    pval = 1.0 - chi2.cdf(stat, p)
    # Ref: grangercause.m:330 — pvalAll = 1-chi2cdf(statAll,(K-1)*length(lags))
    pval_all = 1.0 - chi2.cdf(stat_all, (K - 1) * p)

    return stat, pval, stat_all, pval_all
