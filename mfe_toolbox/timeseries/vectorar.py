"""
Vector Autoregression (VAR) estimation.

Migrated from ``timeseries/vectorar.m`` — Author: Kevin Sheppard
(Revision 3.1, Date: 3/1/2014)

Estimates a Vector Autoregression (VAR) model and produces the parameter
variance-covariance matrix under four error structure assumptions:

1. Conditionally Homoskedastic and Uncorrelated
2. Conditionally Homoskedastic but Correlated
3. Heteroskedastic but Conditionally Uncorrelated
4. Heteroskedastic and Correlated

The model specification is:

    y(t)' = CONST + P(1) * y(t-1) + P(2) * y(t-2) + ... + P(m) * y(t-m)

where P(j) are K x K parameter matrices and CONST is a K x 1 vector.

Notes
-----
The VAR is estimated equation-by-equation using OLS, which is efficient
under the assumption that every equation has the same set of regressors
(the common case for VAR models). The VCV matrix is then computed under
the selected error structure assumption via :func:`vectorarvcv`.

References
----------
Sheppard, K. (2009). MFE Toolbox Version 4.0.
Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.
Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
    Springer.
"""

import numpy as np
from scipy.stats import norm

from mfe_toolbox.utility.newlagmatrix import newlagmatrix
from mfe_toolbox.timeseries.vectorarvcv import vectorarvcv

__all__ = ['vectorar']


def vectorar(
    y: np.ndarray,
    constant: int,
    lags: np.ndarray,
    het: int = 1,
    uncorr: int = 0,
) -> tuple:
    """
    Estimate a Vector Autoregression (VAR) model.

    Fits a VAR model by OLS and computes standard errors, t-statistics,
    and p-values under one of four VCV estimation modes selected by
    the ``het`` and ``uncorr`` flags.

    Parameters
    ----------
    y : np.ndarray
        A ``(T, K)`` matrix of multivariate data, where ``T`` is the
        number of observations and ``K`` is the number of variables.
        Every column must have non-zero variance (i.e., no constant
        columns).
    constant : int
        Include a constant in the regression:

        - ``1`` — include a K x 1 intercept vector
        - ``0`` — no intercept
    lags : np.ndarray or array_like
        Non-negative integer array of lag orders to include.  Can be
        non-contiguous (e.g., ``[1, 3]`` includes lags 1 and 3 only).
        If ``lags`` contains both ``0`` and positive values, the ``0``
        is silently removed.
    het : int, optional
        Covariance estimator type:

        - ``0`` — Homoskedastic
        - ``1`` — Heteroskedastic **[default]**
    uncorr : int, optional
        Assumed error covariance structure:

        - ``0`` — Correlated errors **[default]**
        - ``1`` — Uncorrelated errors

    Returns
    -------
    tuple
        An 11-element tuple containing:

        - **parameters** (*list*) — Length-``m`` list (``m = max(lags)``)
          of ``(K, K)`` parameter matrices.  Only positions corresponding
          to requested lag values are populated; other positions are
          ``None``.
        - **stderr** (*list*) — Same structure as *parameters*, containing
          standard errors.
        - **tstat** (*list*) — Same structure, containing t-statistics.
        - **pval** (*list*) — Same structure, containing two-sided
          p-values computed from the standard normal distribution.
        - **const** (*np.ndarray or None*) — ``(K,)`` array of intercept
          estimates, or ``None`` if ``constant=0``.
        - **conststd** (*np.ndarray or None*) — ``(K,)`` array of
          intercept standard errors, or ``None`` if ``constant=0``.
        - **r2** (*np.ndarray*) — ``(K,)`` array of R-squared values per
          equation.
        - **errors** (*np.ndarray*) — ``(T - m, K)`` matrix of residuals.
        - **s2** (*np.ndarray*) — ``(K, K)`` error covariance matrix
          (divided by original ``T``).
        - **paramvec** (*np.ndarray*) — ``(K * Np,)`` stacked parameter
          vector in column-major (Fortran) order, where
          ``Np = K * p + constant`` and ``p`` is the number of positive
          lag positions.
        - **vcv** (*np.ndarray*) — ``(K * Np, K * Np)`` parameter
          variance-covariance matrix.

    Raises
    ------
    ValueError
        If any input fails validation (see Notes for details).

    Notes
    -----
    Input validation mirrors the original MATLAB implementation:

    - ``y`` must be a 2-D array with no constant columns.
    - ``constant`` must be exactly ``0`` or ``1``.
    - ``lags`` must be a vector of unique non-negative integers.
    - If ``lags`` equals ``[0]``, a constant must be included.
    - ``het`` and ``uncorr`` must each be ``0`` or ``1``.

    The parameter vector ordering in *paramvec* follows the convention::

        [CONST(1) P1(1,1) P1(1,2) ... P1(1,K) P2(1,1) ... P2(1,K) ...
         CONST(2) P1(2,1) P1(2,2) ... P1(2,K) P2(2,1) ... P2(2,K) ...
         ...
         CONST(K) P1(K,1) P1(K,2) ... P1(K,K) P2(K,1) ... P2(K,K) ...]

    where ``P_j(i, k)`` is the coefficient of variable ``k`` at lag ``j``
    in equation ``i``.

    The error covariance ``s2`` is divided by the original number of
    observations ``T`` (not the effective sample size ``T - m``), matching
    the original MATLAB implementation.

    Examples
    --------
    Fit a VAR(1) with a constant:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal((200, 3))
    >>> params, se, t, pv, c, cs, r2, e, s2, pvec, vcv = vectorar(
    ...     y, 1, np.array([1]))

    Fit a VAR with lags 1 and 3, no constant:

    >>> params, *_ = vectorar(y, 0, np.array([1, 3]))

    See Also
    --------
    vectorarvcv : VCV estimation for VAR parameters.
    mfe_toolbox.timeseries.impulseresponse : VAR impulse response functions.
    mfe_toolbox.timeseries.grangercause : Granger causality tests using VARs.

    References
    ----------
    .. [1] Sheppard, K. (2009). MFE Toolbox Version 4.0, vectorar.m.
    .. [2] Lütkepohl, H. (2005). *New Introduction to Multiple Time
       Series Analysis*. Springer.
    """
    # ==================================================================
    # Input Validation
    # Ref: vectorar.m:70-138
    # ==================================================================

    # ------------------------------------------------------------------
    # Validate y
    # Ref: vectorar.m:83-88
    # ------------------------------------------------------------------
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        # Ref: vectorar.m:84 — error('Y must be T by K')
        raise ValueError('Y must be T by K')

    # Ref: vectorar.m:86-88 — All columns must have non-zero variance
    if np.any(np.var(y, axis=0, ddof=0) == 0):
        raise ValueError(
            'At least one component of Y is constant. All components of Y '
            'are required to be non-constant (i.e. have some variation).'
        )

    # ------------------------------------------------------------------
    # Validate constant
    # Ref: vectorar.m:89-95
    # ------------------------------------------------------------------
    if not np.isscalar(constant):
        raise ValueError('CONSTANT must be either 0 or 1')
    constant = int(constant)
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be either 0 or 1')

    # ------------------------------------------------------------------
    # Validate and process lags
    # Ref: vectorar.m:97-121
    # ------------------------------------------------------------------
    lags = np.asarray(lags, dtype=np.float64).ravel()

    # Ref: vectorar.m:97-98 — lags must be a vector
    if lags.ndim != 1:
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )

    # Ref: vectorar.m:103-104 — all elements non-negative
    if not np.all(lags >= 0):
        raise ValueError(
            'LAGS must be a vector of nonnegative integers containing '
            'lags to include'
        )

    # Ref: vectorar.m:106-108 — all elements are integers
    if not np.all(np.floor(lags) == lags):
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )

    # Ref: vectorar.m:109-111 — all elements unique
    if len(lags) != len(np.unique(lags)):
        raise ValueError('LAGS must be a vector of unique elements')

    # Ref: vectorar.m:113-114 — Remove 0 if positive lags also present
    if np.any(lags == 0) and np.any(lags > 0):
        lags = lags[lags != 0]

    # Ref: vectorar.m:116 — Sort and deduplicate
    lags = np.unique(lags)

    # Ref: vectorar.m:117-121 — If lags is [0], constant must be included
    if lags.size == 1 and lags[0] == 0:
        if not constant:
            raise ValueError('If LAGS=0, a constant must be included')

    # Convert to integer array for indexing
    lags = lags.astype(np.int64)

    # ------------------------------------------------------------------
    # Validate het
    # Ref: vectorar.m:123-128
    # ------------------------------------------------------------------
    if not np.isscalar(het):
        raise ValueError('HET must be either 0 or 1')
    het = int(het)
    if het not in (0, 1):
        raise ValueError('HET must be either 0 or 1')

    # ------------------------------------------------------------------
    # Validate uncorr
    # Ref: vectorar.m:130-135
    # ------------------------------------------------------------------
    if not np.isscalar(uncorr):
        raise ValueError('UNCORR must be either 0 or 1')
    uncorr = int(uncorr)
    if uncorr not in (0, 1):
        raise ValueError('UNCORR must be either 0 or 1')

    # ==================================================================
    # Core Estimation
    # Ref: vectorar.m:140-174
    # ==================================================================

    # Ref: vectorar.m:141-142 — Dimensions
    T: int = y.shape[0]
    K: int = y.shape[1]

    # Ref: vectorar.m:143-146 — Maximum lag order
    m: int = int(np.max(lags)) if lags.size > 0 else 0

    # Count positive lag positions
    # Ref: vectorar.m:148 — K*sum(lags>0)+constant
    positive_lags = lags[lags > 0]
    p: int = positive_lags.size  # number of distinct positive lag positions

    # Number of regressors per equation
    # Ref: vectorar.m:148
    ncols: int = K * p + constant

    # Build regressor matrix X and dependent variable Y
    # Ref: vectorar.m:148-149
    X = np.zeros((T - m, ncols))
    # Ref: vectorar.m:149 — Y = y(m+1:T,:)
    # MATLAB 1-indexed m+1:T corresponds to Python 0-indexed m:T
    Y = y[m:, :]

    # Ref: vectorar.m:150 — Column insertion index
    index: int = 0

    # Ref: vectorar.m:152-155 — Constant column
    if constant:
        # Ref: vectorar.m:153 — X(:,1) = ones(T-m,1)
        X[:, 0] = 1.0
        index = 1

    # Ref: vectorar.m:156-167 — Lagged variable columns
    if p > 0:
        # Ref: vectorar.m:157-159 — Build lag matrices for each variable
        ylags = []
        for k in range(K):
            # Ref: vectorar.m:158 — [~, ylags{k}] = newlagmatrix(y(:,k), m, 0)
            # In Python, newlagmatrix expects (y, lags, include_constant)
            _, yl = newlagmatrix(y[:, k], m, 0)
            ylags.append(yl)

        # Ref: vectorar.m:160-166 — Fill X with lagged columns
        # Ordering: for each lag position, all K variables, then next lag
        for i in range(p):
            for k in range(K):
                # Ref: vectorar.m:163 — X(:,index) = ylags{k}(:, lags(i))
                # MATLAB 1-indexed column → Python 0-indexed
                X[:, index] = ylags[k][:, positive_lags[i] - 1]
                index += 1

    # ------------------------------------------------------------------
    # OLS Estimation
    # Ref: vectorar.m:170 — paramvec = X\Y
    # ------------------------------------------------------------------
    # np.linalg.lstsq provides the least-squares solution matching MATLAB's
    # backslash operator for overdetermined systems.
    paramvec, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    # paramvec shape: (ncols, K) — each column has params for one equation

    # Ref: vectorar.m:171 — errors = Y - X * paramvec
    errors = Y - X @ paramvec

    # Ref: vectorar.m:172 — s2 = errors' * errors / T
    # NOTE: T is the ORIGINAL number of observations (size(y,1)), not T-m.
    # This matches the original MATLAB implementation exactly.
    s2 = errors.T @ errors / T

    # ------------------------------------------------------------------
    # VCV Estimation
    # Ref: vectorar.m:174 — vcv = vectorarvcv(X, errors, het, uncorr)
    # ------------------------------------------------------------------
    vcv = vectorarvcv(X, errors, het, uncorr)

    # ==================================================================
    # Reshape Parameters into Structured Output
    # Ref: vectorar.m:177-204
    # ==================================================================

    # Ref: vectorar.m:180 — Number of parameters per equation
    Np: int = paramvec.size // K

    # Ref: vectorar.m:181 — tempParam = paramvec'
    # paramvec is (Np, K), transpose to (K, Np) so each row is one equation
    tempParam = paramvec.T.copy()

    # Ref: vectorar.m:182 — tempStd = reshape(sqrt(diag(vcv)), Np, K)'
    # The diagonal of vcv contains variances ordered as:
    #   [var(p1_eq1), ..., var(pNp_eq1), var(p1_eq2), ..., var(pNp_eq2), ...]
    # MATLAB reshape(vec, Np, K) fills column-major → (Np, K)
    # Transpose → (K, Np)
    # Equivalent in NumPy: reshape with C-order to (K, Np) directly
    diag_vcv = np.sqrt(np.diag(vcv))
    tempStd = diag_vcv.reshape(K, Np).copy()

    # ------------------------------------------------------------------
    # Extract constant coefficients and standard errors
    # Ref: vectorar.m:183-191
    # ------------------------------------------------------------------
    if constant:
        # Ref: vectorar.m:184 — const = tempParam(:,1)
        const_out = tempParam[:, 0].copy()
        # Ref: vectorar.m:185 — conststd = tempStd(:,1)
        conststd_out = tempStd[:, 0].copy()
        # Ref: vectorar.m:186-187 — Remove constant column from temp arrays
        tempParam = tempParam[:, 1:]
        tempStd = tempStd[:, 1:]
    else:
        # Ref: vectorar.m:189-190 — const=[]; conststd=[]
        const_out = None
        conststd_out = None

    # ------------------------------------------------------------------
    # Build parameter, stderr, tstat, pval lists indexed by lag position
    # Ref: vectorar.m:193-204
    # ------------------------------------------------------------------
    # Ref: vectorar.m:193-196 — Cell arrays of size m
    # In Python, list of length m where each entry is None or (K, K) array.
    # Index j represents lag j+1 (0-based indexing).
    parameters = [None] * m
    stderr = [None] * m
    tstat_list = [None] * m
    pval_list = [None] * m

    if p > 0:
        # Ref: vectorar.m:198-204 — Fill for each requested lag
        for i in range(p):
            lag_val = positive_lags[i]
            # Ref: vectorar.m:199 — parameters{lags(i)} = tempParam(:,(i-1)*K+1:i*K)
            # MATLAB 1-indexed columns (i-1)*K+1 to i*K
            # Python 0-indexed: i*K to (i+1)*K
            param_block = tempParam[:, i * K:(i + 1) * K]
            std_block = tempStd[:, i * K:(i + 1) * K]

            # Ref: vectorar.m:199-200
            # lag_val is 1-based lag number, Python list is 0-based
            parameters[lag_val - 1] = param_block

            # Ref: vectorar.m:200
            stderr[lag_val - 1] = std_block

            # Ref: vectorar.m:201 — tstat{lags(i)} = parameters{lags(i)} ./ stderr{lags(i)}
            tstat_block = param_block / std_block
            tstat_list[lag_val - 1] = tstat_block

            # Ref: vectorar.m:202 — pval{lags(i)} = 2 - 2 * normcdf(abs(tstat{lags(i)}))
            # Two-sided p-value from the standard normal distribution
            pval_block = 2.0 - 2.0 * norm.cdf(np.abs(tstat_block))
            pval_list[lag_val - 1] = pval_block

    # ==================================================================
    # R-squared
    # Ref: vectorar.m:206-212
    # ==================================================================
    if constant:
        # Ref: vectorar.m:207 — ytil = y - repmat(mean(y), T, 1)
        # NOTE: uses the FULL y (T rows), not the trimmed Y (T-m rows).
        # sum(errors.^2) uses T-m observations; sum(ytil.^2) uses T observations.
        # This matches the original MATLAB implementation.
        ytil = y - np.mean(y, axis=0)
        # Ref: vectorar.m:208 — r2 = 1 - sum(errors.^2) ./ sum(ytil.^2)
        r2 = 1.0 - np.sum(errors ** 2, axis=0) / np.sum(ytil ** 2, axis=0)
    else:
        # Ref: vectorar.m:210 — r2 = 1 - sum(errors.^2) ./ sum(y.^2)
        r2 = 1.0 - np.sum(errors ** 2, axis=0) / np.sum(y ** 2, axis=0)

    # Ref: vectorar.m:212 — r2 = r2'
    # MATLAB transposes 1×K to K×1 column vector.
    # In Python this is already a (K,) 1-D array; no transpose needed.

    # ==================================================================
    # Flatten parameter vector (column-major)
    # Ref: vectorar.m:213 — paramvec = paramvec(:)
    # ==================================================================
    # MATLAB (:) flattens column-major. NumPy ravel/flatten with order='F'
    # produces the same ordering: all params for equation 1, then equation 2, etc.
    paramvec_flat = paramvec.ravel(order='F')

    return (
        parameters,
        stderr,
        tstat_list,
        pval_list,
        const_out,
        conststd_out,
        r2,
        errors,
        s2,
        paramvec_flat,
        vcv,
    )
