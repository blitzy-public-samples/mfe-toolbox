"""
Heterogeneous Autoregression (HAR) model parameter estimation.

Migrated from timeseries/heterogeneousar.m (Kevin Sheppard, Revision 1,
7/13/2009).

Estimates a Heterogeneous Autoregressive (HAR) model using OLS with robust
(White or Newey-West) standard errors.  Supports STANDARD (overlapping)
and MODIFIED (non-overlapping) lag parametrizations.

The HAR model of Corsi (2009) regresses a dependent variable on averages
of its own lagged values over different horizons (e.g. daily, weekly,
monthly).  This formulation parsimoniously captures long-memory-like
behaviour commonly observed in realised-volatility series.

Notes
-----
MATLAB ``fminunc``/``fmincon`` optimisers are **not** used here — the HAR
model is estimated via OLS (closed-form), so ``scipy.optimize`` is not
needed.  Robust standard errors are computed manually via the sandwich
formula using :func:`mfe_toolbox.utility.covnw.covnw`.

See Also
--------
mfe_toolbox.timeseries.armaxfilter : ARMAX model estimation driver.
mfe_toolbox.univariate.tarch : TARCH / GJR-GARCH model.
"""

import numpy as np
import warnings

from mfe_toolbox.utility.newlagmatrix import newlagmatrix
from mfe_toolbox.utility.covnw import covnw
from mfe_toolbox.timeseries.aicsbic import aicsbic
from mfe_toolbox.timeseries.armaroots import armaroots

__all__ = ['heterogeneousar']


def heterogeneousar(y, constant, p, nw=0, spec='STANDARD'):
    """
    Heterogeneous Autoregression parameter estimation.

    Parameters
    ----------
    y : array_like
        A T-element column of data.
    constant : int
        Scalar variable: 1 to include a constant, 0 to exclude.
    p : array_like
        A column vector or a matrix.

        * **Vector format** — each element is an endpoint for an averaging
          window that starts at lag 1.  For example ``[1, 5, 22]`` produces
          regressors from the 1st lag, the average of lags 1–5, and the
          average of lags 1–22.  Must be a 1-D array or column vector.
        * **Matrix format** — a ``(num_entries, 2)`` array where each row
          ``[start, end]`` specifies a custom lag-averaging window.  The
          vector ``[1, 5, 22]`` is equivalent to ``[[1,1],[1,5],[1,22]]``.
          Allows non-contiguous windows, e.g. ``[[1,1],[5,5],[1,22]]``.
    nw : int, optional
        Number of Newey-West lags for the long-run variance of the scores
        in *VCVrobust*.  Default is 0 (White heteroskedasticity-consistent).
    spec : {'STANDARD', 'MODIFIED'}, optional
        Parametrization mode.  ``'STANDARD'`` (default) uses overlapping
        lag-averaging windows.  ``'MODIFIED'`` transforms to non-overlapping
        windows via row-echelon elimination of the indicator matrix.

    Returns
    -------
    parameters : numpy.ndarray
        A ``(numX,)`` vector ``[constant, har(1), …, har(numP)]`` (constant
        included only when *constant* = 1).
    errors : numpy.ndarray
        A ``(T,)`` residual vector with leading zeros for the
        ``max(max(P))`` observations lost to lagging.
    SEregression : float
        Standard error of the regression.
    diagnostics : dict
        Diagnostic information:

        * ``'P'``  — lag specification matrix
        * ``'C'``  — constant indicator
        * ``'T'``  — total number of observations
        * ``'adjT'``  — effective sample size after lag removal
        * ``'spec'``  — parametrization string
        * ``'AIC'``  — Akaike Information Criterion
        * ``'SBIC'``  — Bayesian (Schwartz) Information Criterion
        * ``'ARparameterization'``  — implied AR coefficient vector
        * ``'arroots'``  — roots of the AR characteristic polynomial
        * ``'absarroots'``  — absolute values of *arroots*
    VCVrobust : numpy.ndarray
        ``(numX, numX)`` robust variance-covariance matrix (White if
        *nw* = 0, Newey-West if *nw* > 0).
    VCV : numpy.ndarray
        ``(numX, numX)`` non-robust VCV (inverse-Hessian based).

    Raises
    ------
    ValueError
        If any input fails validation.

    Examples
    --------
    Standard HAR with 1, 5, and 22 day lags:

    >>> params, e, se, diag, vcvr, vcv = heterogeneousar(y, 1, np.array([1, 5, 22]))

    Matrix notation equivalent:

    >>> params, e, se, diag, vcvr, vcv = heterogeneousar(
    ...     y, 1, np.array([[1,1],[1,5],[1,22]]))

    Modified (non-overlapping) parametrization:

    >>> params, e, se, diag, vcvr, vcv = heterogeneousar(
    ...     y, 1, np.array([1, 5, 22]), spec='MODIFIED')

    With Newey-West standard errors (bandwidth = ceil(T^(1/3))):

    >>> import math
    >>> nw_lags = math.ceil(len(y) ** (1.0 / 3.0))
    >>> params, e, se, diag, vcvr, vcv = heterogeneousar(
    ...     y, 1, np.array([1, 5, 22]), nw=nw_lags)

    Notes
    -----
    Migrated from ``timeseries/heterogeneousar.m`` (Kevin Sheppard,
    Revision 1, 7/13/2009).

    The robust VCV is computed via the sandwich formula
    ``Ainv @ B @ Ainv / T`` where ``B = covnw(scores, nw, demean=False)``.
    The function :func:`robustvcv` is **not** called — the sandwich is
    assembled inline, exactly matching the MATLAB source.
    """

    # ==================================================================
    # Input Checking  (Ref: heterogeneousar.m:77-148)
    # ==================================================================

    # ------------------------------------------------------------------
    # y validation  (Ref: heterogeneousar.m:90-95)
    # ------------------------------------------------------------------
    y = np.asarray(y, dtype=np.float64)
    # Ref: heterogeneousar.m:91 — size(y,2)>1 rejects multi-column arrays
    if y.ndim >= 2 and y.shape[1] > 1:
        raise ValueError('y series must be a column vector.')
    y = y.ravel()  # flatten (T,1) → (T,)
    # Ref: heterogeneousar.m:91 — length(y)==1 rejects scalars
    if y.size == 1:
        raise ValueError('y series must be a column vector.')
    # Ref: heterogeneousar.m:93-94 — isempty(y)
    if y.size == 0:
        raise ValueError('y is empty.')

    # ------------------------------------------------------------------
    # p validation  (Ref: heterogeneousar.m:100-114)
    # ------------------------------------------------------------------
    p = np.asarray(p, dtype=np.float64)
    if p.size > 0:
        # Ref: heterogeneousar.m:102 — matrix format: size(p,2)==2
        if p.ndim == 2 and p.shape[0] >= 1 and p.shape[1] == 2:
            pass  # Matrix format — use as-is
        # Ref: heterogeneousar.m:104 — vector format: min(size(p))==1
        elif p.ndim == 1 or (p.ndim == 2 and min(p.shape) == 1):
            p_vec = p.ravel()
            # Ref: heterogeneousar.m:105 — p=[ones(size(p)) p]
            p = np.column_stack([np.ones(len(p_vec), dtype=np.float64),
                                 p_vec])
        else:
            raise ValueError(
                'P must be either a column vector or a # entries by 2 '
                'matrix.')
    else:
        # Empty p — reshape to (0, 2) for consistent downstream handling
        p = p.reshape(0, 2) if p.ndim < 2 else p

    # Ref: heterogeneousar.m:111 — unique(p,'rows')
    if p.shape[0] > 0:
        p = np.unique(p, axis=0)

    # Ref: heterogeneousar.m:112-113 — positive integer check
    if p.size > 0:
        if np.min(p) < 1 or np.any(p != np.floor(p)):
            raise ValueError('P must contain only positive integers.')

    # Convert to integer for indexing operations
    p = p.astype(np.int32)

    # ------------------------------------------------------------------
    # constant validation  (Ref: heterogeneousar.m:119-125)
    # ------------------------------------------------------------------
    # Ref: heterogeneousar.m:120-121
    if not constant and p.shape[0] == 0:
        raise ValueError('At least one of CONSTANT or P must be nonempty.')
    # Ref: heterogeneousar.m:123-124 — ismember(constant,[0 1])
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be 0 or 1.')

    # ------------------------------------------------------------------
    # nw validation  (Ref: heterogeneousar.m:130-135)
    # ------------------------------------------------------------------
    if nw is None:
        nw = 0
    # Ref: heterogeneousar.m:133 — length(nw)>1 || floor(nw)~=nw || nw<0
    nw_arr = np.asarray(nw, dtype=np.float64).ravel()
    if (nw_arr.size > 1
            or nw_arr[0] != np.floor(nw_arr[0])
            or nw_arr[0] < 0):
        raise ValueError('NW must be a non-negative integer.')
    nw = int(nw_arr[0])

    # ------------------------------------------------------------------
    # spec validation  (Ref: heterogeneousar.m:140-146)
    # ------------------------------------------------------------------
    if spec is None or (isinstance(spec, str) and spec == ''):
        spec = 'STANDARD'
    # Ref: heterogeneousar.m:143 — upper(spec)
    spec = str(spec).upper()
    # Ref: heterogeneousar.m:144 — ismember(spec,{'STANDARD','MODIFIED'})
    if spec not in ('STANDARD', 'MODIFIED'):
        raise ValueError("SPEC must be either 'STANDARD' or 'MODIFIED'.")

    # ==================================================================
    # MODIFIED Parametrization Transform  (Ref: heterogeneousar.m:152-202)
    # ==================================================================
    numP = p.shape[0]
    # Ref: heterogeneousar.m:154 — maxP = max(max(p))
    maxP = int(np.max(p)) if p.size > 0 else 0

    if spec == 'MODIFIED':
        # Ref: heterogeneousar.m:156 — int32(zeros(numP,maxP))
        # Use float64 for safe arithmetic during Gaussian elimination
        ind = np.zeros((numP, maxP), dtype=np.float64)
        for i in range(numP):
            # Ref: heterogeneousar.m:158 — ind(i, p(i,1):p(i,2)) = 1
            # MATLAB 1-indexed inclusive range → Python 0-indexed:
            # start = p[i,0]-1, exclusive end = p[i,1]
            ind[i, p[i, 0] - 1:p[i, 1]] = 1.0

        # ----- Row echelon form via Gaussian elimination -----
        # Ref: heterogeneousar.m:160-183
        used = []
        for _pivot_iter in range(numP):
            # Ref: heterogeneousar.m:163 — setdiff(1:numP, used)
            # Python 0-indexed equivalent
            not_used = np.setdiff1d(np.arange(numP),
                                    np.array(used, dtype=np.int64))
            rows = ind[not_used, :]

            # Ref: heterogeneousar.m:165-169 — find pivot column
            # max(max(abs(rows))) for matrix; max(abs(rows)) for vector
            abs_rows = np.abs(rows)
            col_max = np.max(abs_rows, axis=0)
            position = int(np.argmax(col_max))
            entry = float(col_max[position])

            if entry > 0:
                # Ref: heterogeneousar.m:171 — row with max abs at pivot col
                row_selected = int(np.argmax(np.abs(rows[:, position])))
                original_position = int(not_used[row_selected])
                used.append(original_position)

                row = rows[row_selected, :].copy()
                # Ref: heterogeneousar.m:175-176 — normalise by leading elt
                leading_element = row[position]
                row = row / leading_element

                # Ref: heterogeneousar.m:178-181 — eliminate from other rows
                for j in range(numP):
                    if j != original_position:
                        ind[j, :] = ind[j, :] - ind[j, position] * row

        # Ref: heterogeneousar.m:185-190 — normalise each row by 1st nonzero
        for i in range(numP):
            nonzero_idx = np.flatnonzero(ind[i])
            if len(nonzero_idx) > 0:
                element = ind[i, nonzero_idx[0]]
                ind[i, :] = ind[i, :] / element

        # Ref: heterogeneousar.m:193-200 — validity check
        max_col_sum = float(np.max(np.sum(np.abs(ind), axis=0)))
        newp = np.zeros_like(p, dtype=np.int32)

        if max_col_sum == 1.0:
            for i in range(numP):
                nz = np.flatnonzero(ind[i])
                # Ref: heterogeneousar.m:195 — find returns 1-indexed;
                # flatnonzero returns 0-indexed → add 1
                newp[i, 0] = nz[0] + 1
                newp[i, 1] = nz[-1] + 1
            p = newp
        else:
            # Ref: heterogeneousar.m:199
            warnings.warn(
                'Input P is not compatible with the MODIFIED '
                'parameterization. Using STANDARD parameterization')

        # Ref: heterogeneousar.m:201 — sortrows(p)
        # np.lexsort sorts by last key first, so reverse column order
        sort_idx = np.lexsort((p[:, 1], p[:, 0]))
        p = p[sort_idx]

    # ==================================================================
    # Core HAR Estimation  (Ref: heterogeneousar.m:206-226)
    # ==================================================================

    # Ref: heterogeneousar.m:206 — [Y, X] = newlagmatrix(y, maxP, 0)
    Y, X = newlagmatrix(y, maxP, 0)
    # newlagmatrix returns (T-maxP, 1) shaped Y; flatten for vector ops
    Y = Y.ravel()
    # Ref: heterogeneousar.m:207 — T = length(Y)
    T = len(Y)

    # Ref: heterogeneousar.m:208-211 — construct HAR regressors
    newX = np.zeros((T, numP))
    for i in range(numP):
        # Ref: heterogeneousar.m:210 — mean(X(:,p(i,1):p(i,2)),2)
        # MATLAB 1-indexed inclusive → Python 0-indexed: start=p[i,0]-1,
        # exclusive end=p[i,1]
        newX[:, i] = np.mean(X[:, p[i, 0] - 1:p[i, 1]], axis=1)

    # Ref: heterogeneousar.m:212-214 — prepend constant column
    if constant:
        newX = np.column_stack([np.ones(T), newX])

    numX = newX.shape[1]

    # Ref: heterogeneousar.m:216 — parameters = newX\Y (OLS)
    parameters = np.linalg.lstsq(newX, Y, rcond=None)[0]

    # Ref: heterogeneousar.m:217 — errors = Y - newX*parameters
    errors = Y - newX @ parameters

    # Ref: heterogeneousar.m:218 — SEregression = sqrt(e'e / (T-numP))
    # Note: denominator is (T - numP), not (T - numX), matching MATLAB
    SEregression = float(np.sqrt(np.dot(errors, errors) / (T - numP)))

    # Ref: heterogeneousar.m:219 — A = newX'*newX / T
    A = newX.T @ newX / T

    # Ref: heterogeneousar.m:220 — Ainv = A\eye(size(A))
    Ainv = np.linalg.solve(A, np.eye(numX))

    # Ref: heterogeneousar.m:221 — s = newX .* repmat(errors,1,numX)
    # Broadcasting replacement: errors[:, np.newaxis] is (T, 1)
    s = newX * errors[:, np.newaxis]

    # Ref: heterogeneousar.m:222 — B = covnw(s, nw, 0)
    # Third arg 0 = demean=False (scores are already zero-mean residuals)
    B = covnw(s, nw, 0)

    # Ref: heterogeneousar.m:223 — VCVrobust = Ainv*B*Ainv / T
    VCVrobust = Ainv @ B @ Ainv / T

    # Ref: heterogeneousar.m:224 — VCV = SEregression^2 * Ainv / T
    VCV = SEregression ** 2 * Ainv / T

    # Ref: heterogeneousar.m:226 — pad errors with leading zeros
    T_full = len(y)
    errors = np.concatenate([np.zeros(T_full - T), errors])

    # ==================================================================
    # Diagnostics  (Ref: heterogeneousar.m:230-253)
    # ==================================================================
    # In Python we always compute diagnostics (no nargout gating)
    diagnostics = {}
    # Ref: heterogeneousar.m:231-235
    diagnostics['P'] = p
    diagnostics['C'] = constant
    diagnostics['T'] = T_full
    diagnostics['adjT'] = T
    diagnostics['spec'] = spec

    # Ref: heterogeneousar.m:236 — [aic, sbic] = aicsbic(errors, ...)
    aic, sbic = aicsbic(errors, constant, numX - constant, 0)
    diagnostics['AIC'] = aic
    diagnostics['SBIC'] = sbic

    # Ref: heterogeneousar.m:240-248 — AR parameterisation
    if constant:
        # Ref: heterogeneousar.m:241 — MATLAB 1-indexed parameters(2:numX)
        # Python 0-indexed parameters[1:numX]
        noConstParam = parameters[1:numX]
    else:
        noConstParam = parameters.copy()

    coefficients = np.zeros(maxP)
    for i in range(numP):
        # Ref: heterogeneousar.m:247 — scatter each HAR weight into the
        # individual AR coefficient positions.
        # MATLAB 1-indexed p(i,1):p(i,2) → Python 0-indexed p[i,0]-1:p[i,1]
        # Range length = p[i,1] - p[i,0] + 1
        lag_range = p[i, 1] - p[i, 0] + 1
        coefficients[p[i, 0] - 1:p[i, 1]] += (
            noConstParam[i] / lag_range
        )
    diagnostics['ARparameterization'] = coefficients

    # Ref: heterogeneousar.m:250 — armaroots(coefficients, 0, 1:maxP, 0)
    arroots_val, absarroots_val = armaroots(
        coefficients, 0, np.arange(1, maxP + 1), 0
    )
    diagnostics['arroots'] = arroots_val
    diagnostics['absarroots'] = absarroots_val

    return parameters, errors, SEregression, diagnostics, VCVrobust, VCV
