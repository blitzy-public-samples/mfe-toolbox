"""
HEAVY (High-frEquency bAsed VolatilitY) model estimation driver.

Estimates the HEAVY volatility model of Shephard and Sheppard, which jointly
models returns and realized measures using a bivariate (or K-variate)
conditional variance structure with cross-series spillover dynamics.

The variance dynamics follow:

    h(t,:)' = O + A(:,:,1)*f(data(t-1,:))' + ... + A(:,:,maxP)*f(data(t-maxP,:))'
                + B(:,:,1)*h(t-1,:)' + ... + B(:,:,maxQ)*h(t-maxQ,:)'

where h(t) is the K-dimensional conditional variance vector, O is the intercept
vector, A_j are K x K innovation coefficient matrices, B_j are K x K smoothing
coefficient matrices, and f(.) squares returns while passing realized measures
unchanged.

Migrated from: univariate/heavy.m (Version 4.0, 183 lines)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 4/2/2012

Migration notes:
    - fmincon replaced by scipy.optimize.minimize(method='SLSQP')
    - bsxfun replaced by numpy broadcasting
    - optimset replaced by scipy options dict
    - MATLAB 1-based indexing converted to Python 0-based throughout
"""

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.univariate.heavy_likelihood import heavy_likelihood
from mfe_toolbox.univariate.heavy_parameter_transform import heavy_parameter_transform
from mfe_toolbox.utility.robustvcv import robustvcv


def heavy(
    data: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    cons: str | None = None,
    starting_vals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the HEAVY volatility model of Shephard and Sheppard.

    Also estimates general volatility spillover models with 2 or more
    dimensions.

    Parameters
    ----------
    data : np.ndarray
        A T x K matrix of input data.  Data can be either returns or
        realized-measure type data.  Returns are detected by examining a
        series for negative values, and are squared for estimation of the
        model.
    p : np.ndarray
        A K x K matrix containing the lag length of model innovations.
        Position ``(i, j)`` indicates the number of lags of series *j*
        in the model for series *i*.
    q : np.ndarray
        A K x K matrix containing the lag length of conditional variances.
        Position ``(i, j)`` indicates the number of lags of series *j*
        in the model for series *i*.
    cons : str or None, optional
        String indicating the type of constraint to use on parameters:

        * ``'None'`` (default) — Only constrain the intercepts to be
          positive; other parameters may be negative.
        * ``'Positive'`` — Restrict all parameters to be non-negative.
    starting_vals : np.ndarray or None, optional
        A 1-D array of starting values with ``K + sum(sum(p)) + sum(sum(q))``
        elements.  See Notes for parameter ordering.  If ``None``, naive
        starting values are computed automatically.
    options : dict or None, optional
        Optimization options passed to :func:`scipy.optimize.minimize`.
        If ``None``, default options are used:
        ``{'maxiter': 1000, 'disp': True, 'ftol': 1e-8}``.

    Returns
    -------
    parameters : np.ndarray
        A 1-D array with ``sum(sum(P)) + sum(sum(Q)) + K`` estimated
        parameters.  See Notes for ordering.
    ll : float
        The log-likelihood value at the optimum.
    ht : np.ndarray
        A T x K matrix of conditional variances.
    VCV : np.ndarray
        A ``num_params x num_params`` robust parameter variance-covariance
        matrix (A^(-1)*B*A^(-1)/T).
    scores : np.ndarray
        A T x ``num_params`` matrix of individual scores.

    Raises
    ------
    ValueError
        If inputs fail validation (wrong shapes, invalid constraints, or
        incompatible starting values).

    Notes
    -----
    Dynamics are given by:

    .. math::

        h(t,:)' = O + A(:,:,1) f(data(t-1,:))' + \\ldots
                     + A(:,:,\\max P) f(data(t-\\max P,:))'
                     + B(:,:,1) h(t-1,:)' + \\ldots
                     + B(:,:,\\max Q) h(t-\\max Q,:)'

    PARAMETERS are ordered::

        [O' A(1,1,1:p[0,0]) A(1,2,1:p[0,1]) ... A(1,K,1:p[0,K-1])
             A(2,1,1:p[1,0]) ... A(K,K,1:p[K-1,K-1])
             B(1,1,1:q[0,0]) ... B(K,K,1:q[K-1,K-1])]

    See Also
    --------
    mfe_toolbox.univariate.heavy_simulate : HEAVY model simulation.
    mfe_toolbox.univariate.tarch : TARCH/GJR-GARCH model.
    mfe_toolbox.univariate.egarch : EGARCH model.

    Examples
    --------
    >>> import numpy as np
    >>> p_mat = np.array([[0, 1], [0, 1]])
    >>> q_mat = np.eye(2, dtype=int)
    >>> # data = np.column_stack([returns, realized_measure])
    >>> # parameters, ll, ht, VCV, scores = heavy(data, p_mat, q_mat, 'None')
    """
    # ==================================================================
    # Phase 1: Input Validation
    # Ref: heavy.m:55-116
    # ==================================================================

    # Ensure numpy arrays — keep p, q as float for validation before int cast
    data = np.asarray(data, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Ref: heavy.m:55-69 — Handle default arguments
    if cons is None:
        cons = 'None'

    # Ref: heavy.m:71-73 — Transpose data if more columns than rows
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] > data.shape[0]:
        data = data.T

    # Ref: heavy.m:74 — Get dimensions
    T, K = data.shape

    # Ref: heavy.m:76-78 — Validate P matrix (before int conversion)
    if (np.any(np.array(p.shape) != K)
            or np.any(np.floor(p) != p)
            or np.any(p < 0)):
        raise ValueError('P must be a K by K matrix of non-negative integers.')

    # Ref: heavy.m:80-82 — Validate Q matrix (before int conversion)
    if (np.any(np.array(q.shape) != K)
            or np.any(np.floor(q) != q)
            or np.any(q < 0)):
        raise ValueError('Q must be a K by K matrix of non-negative integers.')

    # Convert validated p, q to integer arrays
    p = p.astype(np.int64)
    q = q.astype(np.int64)

    # Ref: heavy.m:84-94 — Parse constraint mode
    cons_lower = cons.lower()
    if cons_lower == 'none':
        cons_mode = 0
    elif cons_lower == 'positive':
        cons_mode = 1
    else:
        raise ValueError('Unknown values for CONS')

    # Ref: heavy.m:96 — Total parameter count
    parameter_count = K + int(np.sum(p)) + int(np.sum(q))

    # Ref: heavy.m:97-103 — Validate starting values if provided
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        if len(starting_vals) != parameter_count:
            raise ValueError('Incorrect number of parameters in STARTINGVALS.')
        # Ref: heavy.m:101-102 — First K values (intercepts) must be positive
        # MATLAB: if startingVals(1:K)<=0 → triggers only when ALL are <= 0
        # (MATLAB if-on-array checks all()), but intent is all-positive check.
        # Use np.all to match MATLAB's if-array semantics exactly.
        if np.all(starting_vals[:K] <= 0):
            raise ValueError(
                'Initial values for O in STARTINGVALS must be strictly positive.'
            )

    # Ref: heavy.m:111-116 — Set default optimization options
    # MATLAB uses optimset('fmincon') with 'interior-point' algorithm;
    # Python uses scipy.optimize.minimize(method='SLSQP')
    if options is None:
        options = {'maxiter': 1000, 'disp': True, 'ftol': 1e-8}

    # ==================================================================
    # Phase 2: Data Transformation
    # Ref: heavy.m:119-125
    # ==================================================================

    # Ref: heavy.m:120 — Returns detected by negative values; realized
    # measures are non-negative
    is_return = np.any(data < 0, axis=0)

    # Ref: heavy.m:121-123 — Square returns; pass realized measures unchanged
    data2 = data ** 2
    data2[:, ~is_return] = data[:, ~is_return]

    # Ref: heavy.m:123-124 — Rescale data for numerical stability during
    # optimization
    scale = np.mean(data2, axis=0)  # (K,) array
    # Ref: heavy.m:124 — bsxfun replaced by numpy broadcasting
    data2 = data2 / scale

    # ==================================================================
    # Phase 3: Starting Values
    # Ref: heavy.m:128-141
    # ==================================================================

    if starting_vals is None:
        # Ref: heavy.m:131 — Naive strategy: intercepts = 5% of mean
        sv_parts = [np.mean(data2, axis=0) * 0.05]

        # Ref: heavy.m:132-135 — A parameters: 0.1 distributed across lags
        for i in range(K):
            # Ref: heavy.m:133 — As = sum(p(i,:)); 0-indexed
            a_count = int(np.sum(p[i, :]))
            if a_count > 0:
                sv_parts.append(0.1 * np.ones(a_count) / a_count)

        # Ref: heavy.m:136-139 — B parameters: 0.8 distributed across lags
        # Ref: heavy.m:137 — MATLAB source uses sum(p(i,:)) for B starting
        # values (apparent bug preserved for parity)
        for i in range(K):
            b_count = int(np.sum(p[i, :]))  # BUG: should be q[i, :]
            if b_count > 0:
                sv_parts.append(0.8 * np.ones(b_count) / b_count)

        # Ref: heavy.m:140 — Concatenate and transpose to column vector
        starting_vals = np.concatenate(sv_parts)

    # ==================================================================
    # Phase 4: Optimization Setup
    # Ref: heavy.m:142-157
    # ==================================================================

    # Ref: heavy.m:145-146 — Volatility bounds for barrier transformations
    vol_lb = np.min(data2, axis=0) / 10000.0  # (K,) array
    vol_ub = np.max(data2, axis=0) * 100000.0  # (K,) array

    # Ref: heavy.m:147-149 — Exponentially weighted back-cast initialization
    w_len = int(np.ceil(T ** 0.5)) + 1
    w = 0.06 * 0.94 ** np.arange(w_len)
    w = w / np.sum(w)
    # Ref: heavy.m:149 — bsxfun replaced by broadcasting;
    # w[:, np.newaxis] is (w_len,1), data2[:w_len,:] is (w_len,K)
    back_cast = np.sum(w[:, np.newaxis] * data2[:w_len, :], axis=0)  # (K,)

    # Ref: heavy.m:151-156 — Bounds: intercepts >= 0; other params in
    # [-1,1] or [0,1]
    LB = np.zeros(parameter_count)
    UB = np.full(parameter_count, np.inf)
    UB[K:] = 1.0  # Non-intercept parameters bounded at 1
    if cons_mode == 0:
        LB[K:] = -1.0  # Allow negative non-intercept params in 'None' mode

    # Create scipy bounds from LB/UB arrays
    bounds = list(zip(LB.tolist(), UB.tolist()))

    # Ref: heavy.m:157 — fmincon replaced by scipy.optimize.minimize(
    # method='SLSQP')
    # heavy_likelihood returns (ll, lls, h); minimize needs scalar objective
    def _objective(params, *args):
        """Scalar objective wrapper for scipy.optimize.minimize."""
        ll_val, _, _ = heavy_likelihood(params, *args)
        return ll_val

    result = minimize(
        _objective,
        starting_vals,
        method='SLSQP',
        bounds=bounds,
        args=(data2.T, p, q, back_cast, vol_lb, vol_ub),
        options=options,
    )
    parameters = result.x.copy()

    # ==================================================================
    # Phase 5: Parameter Rescaling
    # Ref: heavy.m:159-178
    # ==================================================================

    # Ref: heavy.m:161 — Rescale data back to original scale
    data2 = data2 * scale  # broadcasting: (T,K) * (K,)

    # Ref: heavy.m:162-163 — Recompute volatility bounds on rescaled data
    vol_lb = np.min(data2, axis=0) / 10000.0
    vol_ub = np.max(data2, axis=0) * 100000.0

    # Ref: heavy.m:164 — Recompute back_cast on rescaled data
    back_cast = np.sum(w[:, np.newaxis] * data2[:w_len, :], axis=0)

    # Ref: heavy.m:166 — Only O and A used for rescaling; B unchanged
    O, A, _B = heavy_parameter_transform(parameters, p, q, K)

    # Ref: heavy.m:167 — Rescale intercepts by data scale
    O = O * scale

    # Ref: heavy.m:168-170 — Rescale innovation params; 0-based indexing
    max_p = int(np.max(p)) if np.any(p > 0) else 0
    for i in range(K):
        if max_p > 0:
            # A[:, i, :] has shape (K, max_p)
            # scale / scale[i] has shape (K,)
            # Need (K, 1) for proper broadcasting with (K, max_p)
            A[:, i, :] = A[:, i, :] * (scale / scale[i])[:, np.newaxis]

    # Ref: heavy.m:172-178 — Rebuild parameter vector after rescaling
    # Concatenate rescaled O and A elements, then append original B portion
    temp_parts = list(O.ravel())
    for i in range(K):
        for j in range(K):
            p_ij = int(p[i, j])
            if p_ij > 0:
                # Ref: heavy.m:175 — squeeze(A(i,j,1:p(i,j)));
                # np.squeeze mirrors MATLAB squeeze(); Python slice is
                # already 1D but squeeze ensures scalar-safe extraction
                temp_parts.extend(np.squeeze(A[i, j, :p_ij]).ravel().tolist())
    temp_arr = np.array(temp_parts, dtype=np.float64)

    # Ref: heavy.m:178 — Append original B portion (not rescaled)
    parameters = np.concatenate([temp_arr, parameters[len(temp_arr):]])

    # ==================================================================
    # Phase 6: Inference
    # Ref: heavy.m:180-183
    # ==================================================================

    # Ref: heavy.m:182 — Final log-likelihood and conditional variances
    ll, _, ht = heavy_likelihood(
        parameters, data2.T, p, q, back_cast, vol_lb, vol_ub
    )

    # Ref: heavy.m:183 — Robust sandwich VCV with nw=0 (no Newey-West lags)
    # robustvcv expects fun returning (scalar, vector); heavy_likelihood
    # returns (scalar, vector, matrix), so we wrap it.
    def _hl_for_vcv(params, *extra_args):
        """Wrapper returning only (ll, lls) for robustvcv compatibility."""
        ll_val, lls_val, _ = heavy_likelihood(params, *extra_args)
        return ll_val, lls_val

    VCV, _, _, scores, _, _ = robustvcv(
        _hl_for_vcv, parameters, 0,
        data2.T, p, q, back_cast, vol_lb, vol_ub
    )

    return parameters, ll, ht, VCV, scores
