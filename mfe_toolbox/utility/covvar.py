"""
Long-run covariance estimation using VAR-based method of Den Haan and Levin (1996).

Migrated from utility/covvar.m — Author: Kevin Sheppard
(Revision 3, Date: 5/1/2007)

Estimates the long-run covariance matrix of a T-by-K data matrix using a
VAR-based approach with information criterion-based lag selection. Five
estimation methods are supported: fixed lags, SIC-selected, AIC-selected,
global SIC search, and global AIC search.

References
----------
Den Haan, W.J. and A. Levin (1996), "Inferences from Parametric and
Non-Parametric Covariance Matrix Estimation Procedures," NBER Technical
Working Paper 195.
"""

import numpy as np

from mfe_toolbox.utility.newlagmatrix import newlagmatrix

__all__ = ['covvar']


def _cov_var_dec2bin(maxlag):
    """
    Generate all non-empty subsets of lag indices {1, ..., maxlag} plus an
    empty set, for exhaustive global search (methods 4 and 5).

    Each integer *p* from 1 to 2^maxlag - 1 is decomposed into its binary
    representation, where bit *i* (0-indexed) maps to lag *i + 1*.  The
    resulting boolean selector picks the corresponding lag indices from the
    default set ``[1, 2, ..., maxlag]``.

    Ref: covvar.m:169-184 — internal helper ``cov_VAR_dec2bin``

    Parameters
    ----------
    maxlag : int
        Maximum lag length.  Must be non-negative.

    Returns
    -------
    indices : list of list of int
        List of ``2^maxlag`` entries.  The first ``2^maxlag - 1`` entries are
        all non-empty subsets of ``{1, ..., maxlag}`` (ordered by the integer
        encoding of the binary selector).  The last entry is an empty list
        (no lags).
    """
    # Ref: covvar.m:172 — default=1:maxlag
    default = list(range(1, maxlag + 1))
    indices = []

    # Ref: covvar.m:173 — for p=1:(2^maxlag-1)
    for p in range(1, 2 ** maxlag):
        rem = p
        # Ref: covvar.m:175 — selector=false(1,maxlag)
        selector = [False] * maxlag
        # Ref: covvar.m:176-180 — binary decomposition of p into lag flags
        for i in range(maxlag - 1, -1, -1):
            if rem // (2 ** i):
                # Ref: covvar.m:178 — selector(i+1)=true
                # MATLAB 1-based index i+1 → Python 0-based index i
                selector[i] = True
                rem = rem - 2 ** i
        # Ref: covvar.m:182 — indices{p}=default(selector)
        selected = [default[j] for j in range(maxlag) if selector[j]]
        indices.append(selected)

    # Ref: covvar.m:184 — indices{2^maxlag}=[] (empty set — no lags)
    indices.append([])
    return indices


def _build_column_indices(lag_indices, K):
    """
    Build 0-based column indices into the X regressor matrix for a given set
    of 1-based lag numbers.

    The X matrix has columns ordered as:
        [constant | lag1_v0 lag1_v1 ... lag1_vK-1 | lag2_v0 ... | ...]

    For each selected lag *j* (1-based) and variable *v* (0-based, 0..K-1),
    the 0-based column index in X (with constant at column 0) is::

        col = (j - 1) * K + v + 1

    Ref: covvar.m:109 — ``cols=repmat(indices{i}-1,K,1)*K+repmat((0:K-1)',1,P)+2``
    The MATLAB formula uses 1-based indexing (+2); Python uses 0-based (+1).

    Parameters
    ----------
    lag_indices : list of int
        1-based lag numbers to select (e.g., [1, 3]).
    K : int
        Number of variables (columns in the data matrix).

    Returns
    -------
    cols_flat : list of int
        Flattened 0-based column indices (column-major order matching MATLAB).
    """
    if not lag_indices:
        return []
    P = len(lag_indices)
    lag_arr = np.array(lag_indices, dtype=np.float64)
    # Ref: covvar.m:109 — repmat(indices{i}-1,K,1)*K + repmat((0:K-1)',1,P) + 2
    # Python 0-based: -1 from MATLAB 1-based result → change +2 to +1
    cols_matrix = (np.tile(lag_arr - 1, (K, 1)) * K
                   + np.tile(np.arange(K, dtype=np.float64).reshape(-1, 1),
                             (1, P))
                   + 1)
    # Ref: covvar.m:109 — cols(:)' flattens column-major (Fortran order)
    cols_flat = cols_matrix.ravel(order='F').astype(int).tolist()
    return cols_flat


def _build_yx_matrices(data, maxlag, K, T):
    """
    Build the Y (dependent) and X (lagged regressors + constant) matrices.

    For each of the K data columns, constructs the trimmed dependent vector
    and lag matrix using :func:`newlagmatrix`, then interleaves the lag
    columns so that all K variable-columns for lag *j* are contiguous.

    Ref: covvar.m:83-99

    Parameters
    ----------
    data : numpy.ndarray
        T by K data matrix.
    maxlag : int
        Maximum lag length.
    K : int
        Number of variables.
    T : int
        Number of observations.

    Returns
    -------
    Y : numpy.ndarray
        (T - maxlag) x K dependent variable matrix.
    X : numpy.ndarray
        (T - maxlag) x (1 + K * maxlag) regressor matrix (constant first).
    """
    T_eff = T - maxlag
    Y = np.zeros((T_eff, K))
    X_lags = np.zeros((T_eff, K * maxlag))
    indep_parts = [None] * K

    for i in range(K):
        # Ref: covvar.m:88 — [dep,indep{i}]=newlagmatrix(data(:,i),maxlag,0)
        dep, indep_i = newlagmatrix(data[:, i], maxlag, 0)
        indep_parts[i] = indep_i
        # Ref: covvar.m:89 — Y(:,i)=dep; dep is (T_eff, 1), flatten for col assign
        Y[:, i] = dep.ravel()

    # Ref: covvar.m:91-98 — interleave columns: for each lag j, all K variables
    # Ordering: lag1_v0, lag1_v1, ..., lag1_vK-1, lag2_v0, lag2_v1, ...
    index = 0
    for j in range(maxlag):
        for i in range(K):
            # Ref: covvar.m:94-95 — temp=indep{i}; X(:,index)=temp(:,j)
            # MATLAB j is 1-based, Python j is 0-based
            X_lags[:, index] = indep_parts[i][:, j].ravel()
            index += 1

    # Ref: covvar.m:99 — X=[ones(size(X,1),1) X]
    X = np.column_stack([np.ones((T_eff, 1)), X_lags])
    return Y, X


def covvar(data, maxlag=None, method=2):
    """
    Long-run covariance estimation using VAR-based method of Den Haan and
    Levin (1996).

    Estimates the long-run covariance matrix by fitting a VAR model to the
    data and using the implied spectral density at frequency zero.  Lag
    selection is performed via information criteria (SIC or AIC) with either
    sequential or exhaustive global search over all lag subsets.

    Parameters
    ----------
    data : numpy.ndarray
        T by K matrix of data.  If 1-D, it is reshaped to a column vector
        ``(T, 1)``.
    maxlag : int or None, optional
        Maximum lag length.  Must be a non-negative integer not exceeding
        ``floor(T / K)``.  Default (when ``None``):
        ``min(floor(T / K), floor(1.2 * T**(1/3)))``.
    method : int, optional
        Lag selection method:

        - 1 : Use lags 1 to *maxlag* (fixed — no selection)
        - 2 : (default) Select lags using SIC, sequential search
        - 3 : Select lags using AIC, sequential search
        - 4 : Select lags using SIC, global search over all 2^maxlag subsets
        - 5 : Select lags using AIC, global search over all 2^maxlag subsets

    Returns
    -------
    V : numpy.ndarray
        K x K long-run covariance matrix.
    lagsused : numpy.ndarray
        1-D integer array of 1-based lag indices used in the final estimate.
        Empty array (``shape=(0,)``) if no lags were selected, in which case
        ``V`` equals the population covariance estimator.

    Raises
    ------
    ValueError
        If *method* is not in ``{1, 2, 3, 4, 5}``, *maxlag* is not a
        non-negative integer or exceeds ``floor(T / K)``, or *data* is not
        a 1-D or 2-D numeric array.

    Notes
    -----
    Methods 4 and 5 enumerate all ``2^maxlag`` subsets of lags, so they are
    only practically viable for ``maxlag`` up to approximately 10–20 and
    small K.

    When the best model has no lags (empty lag set), the returned ``V`` is
    the usual sample covariance multiplied by ``(T - 1) / T`` to convert
    from the unbiased estimator (``/(T - 1)``) to the population-style
    estimator (``/T``).

    Migrated from ``utility/covvar.m`` (Kevin Sheppard, Revision 3,
    Date: 5/1/2007).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((200, 3))
    >>> V, lags = covvar(data)
    >>> V.shape
    (3, 3)
    """
    # ------------------------------------------------------------------
    # Input validation — Ref: covvar.m:37-67
    # ------------------------------------------------------------------
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        # Reshape 1-D input to column vector (T, 1) to match MATLAB convention
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        # Ref: covvar.m:56-58 — error('DATA must be a T by K matrix of data.')
        raise ValueError('DATA must be a T by K matrix of data.')

    T, K = data.shape  # Ref: covvar.m:37 — [T,K]=size(data)

    # Ref: covvar.m:38-46 — default maxlag computation
    if maxlag is None:
        # Ref: covvar.m:39 — maxlag=min(floor(T/K),floor(1.2*T^(1/3)))
        maxlag = int(min(np.floor(T / K), np.floor(1.2 * T ** (1.0 / 3.0))))
    else:
        # Ref: covvar.m:53 — floor(maxlag)~=maxlag check (non-integer detection)
        maxlag_float = float(maxlag)
        if np.floor(maxlag_float) != maxlag_float:
            raise ValueError(
                'MAXLAG must be a non-negative integer with '
                'MAXLAG <= floor(T/K).'
            )
        maxlag = int(maxlag_float)

    # Ref: covvar.m:50-52 — validate method
    if method not in (1, 2, 3, 4, 5):
        raise ValueError('METHOD must be a scalar between 1 and 5.')

    # Ref: covvar.m:53-55 — validate maxlag bounds
    max_allowed = int(np.floor(T / K))
    if maxlag < 0 or maxlag > max_allowed:
        raise ValueError(
            'MAXLAG must be a non-negative integer with '
            'MAXLAG <= floor(T/K).'
        )

    # Ref: covvar.m:59-64 — IC type selection
    # Methods 1, 2, 4 use SIC (ICtype=1); Methods 3, 5 use AIC (ICtype=2)
    if method in (1, 2, 4):
        ic_type = 1  # SIC (Schwarz/Bayesian Information Criterion)
    else:
        ic_type = 2  # AIC (Akaike Information Criterion)

    # ------------------------------------------------------------------
    # Set up the lag index combinations to evaluate
    # Ref: covvar.m:70-80
    # ------------------------------------------------------------------
    if method == 1:
        # Ref: covvar.m:71 — indices{1}=1:maxlag
        # Fixed lags: single candidate containing all lags 1..maxlag
        if maxlag > 0:
            indices = [list(range(1, maxlag + 1))]
        else:
            indices = [[]]
    elif method in (2, 3):
        # Ref: covvar.m:73-77 — progressive lag sets: {1}, {1,2}, ..., {1,...,maxlag}, {}
        indices = []
        for i in range(1, maxlag + 1):
            # Ref: covvar.m:75 — indices{i}=1:i
            indices.append(list(range(1, i + 1)))
        # Ref: covvar.m:77 — indices{maxlag+1}=[] (no lags)
        indices.append([])
    else:
        # Ref: covvar.m:79 — indices=cov_VAR_dec2bin(maxlag)
        # Global search: all 2^maxlag subsets + empty set
        indices = _cov_var_dec2bin(maxlag)

    # ------------------------------------------------------------------
    # Build Y (dependent) and X (lagged regressors) matrices
    # using the full maxlag to ensure comparable IC values
    # Ref: covvar.m:82-99
    # ------------------------------------------------------------------
    Y, X = _build_yx_matrices(data, maxlag, K, T)
    T2 = T - maxlag  # Ref: covvar.m:105 — T2=(T-maxlag)

    # ------------------------------------------------------------------
    # IC computation loop over all candidate lag combinations
    # Ref: covvar.m:103-122
    # ------------------------------------------------------------------
    N = len(indices)  # Ref: covvar.m:103 — N=length(indices)
    IC = np.zeros(N)  # Ref: covvar.m:104 — IC=zeros(N,1)

    for idx in range(N):
        lag_indices = indices[idx]
        P = len(lag_indices)  # Ref: covvar.m:107 — P=length(indices{i})

        # Build column indices for selected lags
        cols_flat = _build_column_indices(lag_indices, K)

        # Ref: covvar.m:113 — regressors=X(:,[1 cols(:)'])
        # Select constant column (0) plus selected lag columns
        col_select = [0] + cols_flat
        regressors = X[:, col_select]

        # Ref: covvar.m:114 — B=regressors\Y (OLS via least-squares)
        B = np.linalg.lstsq(regressors, Y, rcond=None)[0]

        # Ref: covvar.m:115 — e=Y-X(:,[1 cols(:)'])*B
        e = Y - regressors @ B

        # Ref: covvar.m:116 — covE=e'*e/T2
        covE = e.T @ e / T2

        # Ref: covvar.m:117-121 — compute information criterion
        log_det_covE = np.log(np.linalg.det(covE))
        if ic_type == 1:
            # Ref: covvar.m:118 — SIC: log(det(covE))+log(T2)/T2*(P*K^2+K)
            IC[idx] = log_det_covE + np.log(T2) / T2 * (P * K ** 2 + K)
        else:
            # Ref: covvar.m:120 — AIC: log(det(covE))+2/T2*(P*K^2+K)
            IC[idx] = log_det_covE + 2.0 / T2 * (P * K ** 2 + K)

    # ------------------------------------------------------------------
    # Select the best model (minimum IC)
    # Ref: covvar.m:125 — [temp,ICpos]=min(IC)
    # ------------------------------------------------------------------
    ICpos = int(np.argmin(IC))
    best_lags = indices[ICpos]

    # ------------------------------------------------------------------
    # Final estimation using maximum available data
    # Ref: covvar.m:128-165
    # ------------------------------------------------------------------
    if best_lags:
        # Ref: covvar.m:129 — maxlag=max(indices{ICpos})
        # Re-estimate with only the needed lags to maximize sample size
        final_maxlag = max(best_lags)

        # Rebuild Y and X with the final (potentially smaller) maxlag
        # Ref: covvar.m:130-144
        Y_final, X_final = _build_yx_matrices(data, final_maxlag, K, T)
        T2_final = T - final_maxlag  # Ref: covvar.m:145

        # Ref: covvar.m:147 — P=length(indices{ICpos})
        P = len(best_lags)

        # Build column selection for the best lag combination
        # Ref: covvar.m:148-152
        cols_flat = _build_column_indices(best_lags, K)
        col_select = [0] + cols_flat

        # Ref: covvar.m:153 — regressors=X(:,[1 cols(:)'])
        regressors_final = X_final[:, col_select]

        # Ref: covvar.m:154 — B=regressors\Y
        B = np.linalg.lstsq(regressors_final, Y_final, rcond=None)[0]

        # Ref: covvar.m:155 — e=Y-X(:,[1 cols(:)'])*B
        e = Y_final - regressors_final @ B

        # Ref: covvar.m:156 — covE=e'*e/T2
        covE = e.T @ e / T2_final

        # Ref: covvar.m:157 — B=B(2:P*K+1,:)
        # Remove constant row from coefficients
        # MATLAB B(2:P*K+1,:) → Python B[1:P*K+1, :]
        B_coefs = B[1:P * K + 1, :]

        # Ref: covvar.m:158 — B=reshape(B',[K,K,P])
        # MATLAB reshape is column-major → use order='F' in Python
        # B_coefs is P*K x K; transpose to K x P*K then reshape to K x K x P
        B_reshaped = np.reshape(B_coefs.T, (K, K, P), order='F')

        # Ref: covvar.m:159 — A=(eye(K)-sum(B,3))^(-1)
        # Sum coefficient matrices across the lag dimension (axis=2 in Python)
        B_sum = np.sum(B_reshaped, axis=2)
        A = np.linalg.inv(np.eye(K) - B_sum)

        # Ref: covvar.m:160 — V=A*covE*A'
        V = A @ covE @ A.T
    else:
        # Ref: covvar.m:162-163 — empty lag selection → standard covariance
        # MATLAB cov(Y) uses 1/(N-1) normalization; multiply by (T-1)/T to
        # convert to population-style 1/T normalization.
        # Note: T here is the original observation count, matching MATLAB.
        V = np.cov(Y, rowvar=False) * ((T - 1) / T)
        # Ensure V is always a 2-D K x K matrix (np.cov returns 0-D scalar
        # for single-variable input; MATLAB treats scalars as 1x1 matrices)
        V = np.atleast_2d(V)

    # Ref: covvar.m:165 — lagsused=indices{ICpos}
    if best_lags:
        lagsused = np.array(best_lags, dtype=np.int64)
    else:
        lagsused = np.array([], dtype=np.int64)

    return V, lagsused
