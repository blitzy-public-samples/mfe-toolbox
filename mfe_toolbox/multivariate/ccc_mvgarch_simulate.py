"""
CCC-MVGARCH Simulation — Constant Conditional Correlation Multivariate GARCH

Simulates TARCH(p,o,q)-CCC MVGARCH processes where each marginal variance
follows a TARCH model and the conditional correlation matrix is constant
over time. Uses a 2000-sample burn-in to minimize start-up bias and generates
pseudo-realized covariance matrices from intra-daily returns.

Migrated from multivariate/ccc_mvgarch_simulate.m
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 4    Date: 10/28/2009
"""

import warnings

import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import sqrtm as _scipy_sqrtm


def _corr_ivech(stacked_data: np.ndarray) -> np.ndarray:
    """Transform a K*(K-1)/2 vector of off-diagonal correlations into a K x K
    symmetric correlation matrix with ones on the diagonal.

    This is a self-contained reimplementation of ``utility/corr_ivech.m``
    included here to keep the simulation module free of internal package
    dependencies.

    Parameters
    ----------
    stacked_data : ndarray
        K*(K-1)/2 vector of lower-triangular off-diagonal correlation elements.

    Returns
    -------
    ndarray
        K x K symmetric correlation matrix with ones on the diagonal.

    Raises
    ------
    ValueError
        If the length of *stacked_data* is not a valid K*(K-1)/2.
    """
    stacked_data = np.asarray(stacked_data, dtype=np.float64).ravel()
    k2 = len(stacked_data)
    # Ref: corr_ivech.m:34 — Solve k*(k-1)/2 = k2 for k via quadratic formula
    k = int((-1.0 + np.sqrt(1.0 + 8.0 * k2)) / 2.0) + 1
    if k * (k - 1) // 2 != k2:
        raise ValueError(
            'The length of stacked_data is not conformable to a correlation '
            'matrix inverse-vech operation.'
        )
    # Ref: corr_ivech.m:48-51 — Fill strict lower triangle, mirror, add identity
    matrix_data = np.zeros((k, k), dtype=np.float64)
    idx = np.tril_indices(k, -1)
    matrix_data[idx] = stacked_data
    matrix_data = matrix_data + matrix_data.T + np.eye(k, dtype=np.float64)
    return matrix_data


def _tarch_simulate(random_nums: np.ndarray,
                    parameters: np.ndarray,
                    p: int, o: int, q: int,
                    error_type: str = 'NORMAL',
                    tarch_type: int = 2) -> tuple:
    """TARCH(p,o,q) time-series simulation with pre-generated random numbers.

    This is a self-contained reimplementation of the core logic from
    ``univariate/tarch_simulate.m`` for the specific case where a vector of
    pre-generated random numbers is provided (as called by
    ``ccc_mvgarch_simulate``).

    Parameters
    ----------
    random_nums : ndarray
        T-length vector of pre-generated standard-normal innovations.
    parameters : ndarray
        [omega, alpha(1:p), gamma(1:o), beta(1:q)] TARCH parameter vector.
    p : int
        Number of symmetric innovation lags (>= 1).
    o : int
        Number of asymmetric innovation lags (>= 0).
    q : int
        Number of conditional-variance lags (>= 0).
    error_type : str, optional
        Error distribution type.  Only ``'NORMAL'`` is used in the CCC
        context.  Default ``'NORMAL'``.
    tarch_type : int, optional
        1 for absolute-value model, 2 for squared model.  Default 2.

    Returns
    -------
    sim_data : ndarray
        Simulated data of the same length as *random_nums*.
    sim_ht : ndarray
        Conditional variances of the same length as *random_nums*.
    """
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    random_nums = np.asarray(random_nums, dtype=np.float64).ravel()

    # Ref: tarch_simulate.m:108-112 — Separate omega, alpha, gamma, beta
    omega = parameters[0]
    alpha = parameters[1:1 + p]
    gamma_params = parameters[1 + p:1 + p + o]
    beta = parameters[1 + p + o:1 + p + o + q]

    # Ref: tarch_simulate.m:93-95 — Check stationarity
    persistence = np.sum(alpha) + np.sum(beta) + 0.5 * np.sum(gamma_params)
    if persistence >= 1.0:
        warnings.warn(
            'Parameters are in the non-stationary space; conditional '
            'variance may diverge.',
            stacklevel=2,
        )

    # Ref: tarch_simulate.m:132-138 — When a vector is provided, prepend a
    # random burn-in of 2000 observations sampled from the input vector.
    original_t = len(random_nums)
    rng = np.random.default_rng()
    seeds = rng.integers(0, original_t, size=2000)
    random_nums_ext = np.concatenate([random_nums[seeds], random_nums])
    t_aug = len(random_nums_ext)  # original_t + 2000

    # Ref: tarch_simulate.m:140-142 — m = max(p,o,q); prepend zeros
    m_max = max(p, o, q) if max(p, o, q) > 0 else 1
    random_nums_padded = np.concatenate([np.zeros(m_max), random_nums_ext])

    # Ref: tarch_simulate.m:144-148 — Unconditional standard deviation
    if (1.0 - persistence) > 0:
        uncond_std = np.sqrt(omega / (1.0 - persistence))
    else:
        uncond_std = 1.0

    total_len = t_aug + m_max
    h = np.full(total_len, uncond_std ** 2, dtype=np.float64)
    data = np.full(total_len, uncond_std, dtype=np.float64)
    indicator = np.zeros(total_len, dtype=np.float64)

    # Ref: tarch_simulate.m:155 — Combined parameter vector for dot product
    params_vec = np.concatenate([[omega], alpha, gamma_params, beta])

    # Ref: tarch_simulate.m:156-169 — Main TARCH recursion
    if tarch_type == 1:
        # Absolute-value model
        for t_idx in range(m_max, total_len):
            regressors = np.empty(1 + p + o + q, dtype=np.float64)
            regressors[0] = 1.0
            for j in range(p):
                # Ref: tarch_simulate.m:158 — abs(data(t-(1:p)))
                regressors[1 + j] = abs(data[t_idx - (j + 1)])
            for j in range(o):
                # Ref: tarch_simulate.m:158 — Idata.*abs(data)
                regressors[1 + p + j] = (
                    indicator[t_idx - (j + 1)]
                    * abs(data[t_idx - (j + 1)])
                )
            for j in range(q):
                regressors[1 + p + o + j] = h[t_idx - (j + 1)]
            h[t_idx] = np.dot(params_vec, regressors)
            data[t_idx] = random_nums_padded[t_idx] * h[t_idx]
            indicator[t_idx] = 1.0 if data[t_idx] < 0 else 0.0
        # Ref: tarch_simulate.m:163 — Square h for variance in abs model
        h = h ** 2
    else:
        # Squared model (tarch_type == 2, default)
        for t_idx in range(m_max, total_len):
            regressors = np.empty(1 + p + o + q, dtype=np.float64)
            regressors[0] = 1.0
            for j in range(p):
                # Ref: tarch_simulate.m:165 — data(t-(1:p)).^2
                regressors[1 + j] = data[t_idx - (j + 1)] ** 2
            for j in range(o):
                # Ref: tarch_simulate.m:165 — Idata.*data.^2
                regressors[1 + p + j] = (
                    indicator[t_idx - (j + 1)]
                    * data[t_idx - (j + 1)] ** 2
                )
            for j in range(q):
                regressors[1 + p + o + j] = h[t_idx - (j + 1)]
            h[t_idx] = np.dot(params_vec, regressors)
            # Ref: tarch_simulate.m:166 — data(t)=RandomNums(t)*sqrt(h(t))
            data[t_idx] = random_nums_padded[t_idx] * np.sqrt(max(h[t_idx], 0.0))
            indicator[t_idx] = 1.0 if data[t_idx] < 0 else 0.0

    # Ref: tarch_simulate.m:170-171 — Truncate m_max back-cast + 2000 burn-in
    # MATLAB: simulatedata = data((m+1+2000):T) → Python 0-indexed: data[m_max+2000:]
    sim_data = data[m_max + 2000:]
    sim_ht = h[m_max + 2000:]

    return sim_data, sim_ht


def ccc_mvgarch_simulate(t: int, k: int, parameters, p, o, q, m: int = 72):
    """Simulate a TARCH(p,o,q)-CCC Multivariate GARCH process.

    Generates simulated data, conditional covariance matrices, and pseudo-
    realized covariance matrices for a Constant Conditional Correlation
    Multivariate GARCH model with TARCH(p,o,q) marginal variance dynamics.

    Migrated from ``multivariate/ccc_mvgarch_simulate.m``.

    Parameters
    ----------
    t : int
        Length of the time series to be simulated (post burn-in).
    k : int
        Cross-sectional dimension (number of series, must be >= 2).
    parameters : array_like
        Full parameter vector of the form::

            [tarch(1)' tarch(2)' ... tarch(k)' corr_vech(R)]

        where each ``tarch(i)`` block has ``1 + p(i) + o(i) + q(i)``
        elements ``[omega(i), alpha(i,1:p(i)), gamma(i,1:o(i)),
        beta(i,1:q(i))]`` and ``corr_vech(R)`` is the ``K*(K-1)/2``
        lower-triangular off-diagonal elements of the constant correlation
        matrix *R*.
    p : int or array_like
        Symmetric innovation order(s).  Positive integer applied to all
        series, or a *K*-vector of individual orders.
    o : int or array_like
        Asymmetric innovation order(s).  Non-negative integer(s).
    q : int or array_like
        Conditional-variance lag order(s).  Non-negative integer(s).
    m : int, optional
        Number of intra-daily returns used to construct pseudo-realized
        covariances.  Default is 72.

    Returns
    -------
    simulatedata : ndarray, shape (t, k)
        Simulated daily return series.
    ht : ndarray, shape (k, k, t)
        Conditional covariance matrices ``H(t) = Sigma(t) R Sigma(t)``.
    pseudorc : ndarray, shape (k, k, t)
        Pseudo-realized covariance matrices constructed from scaled
        intra-daily returns.

    Raises
    ------
    ValueError
        If inputs fail validation (wrong sizes, non-PD correlation, etc.).

    Notes
    -----
    The CCC model defines the conditional covariance as

    .. math::
        H_t = \\Sigma_t \\, R \\, \\Sigma_t

    where :math:`\\Sigma_t = \\text{diag}(\\sqrt{h_{1,t}}, \\ldots,
    \\sqrt{h_{k,t}})` contains per-series TARCH conditional volatilities
    and *R* is the constant correlation matrix.

    Pseudo-realized covariances are simulated by drawing *m* intra-daily
    returns from :math:`N(0, 1/m)`, scaling them by the daily volatility,
    and computing their outer-product sum.  See Patton and Sheppard (2009).

    A burn-in of 2000 observations is applied; the first 2000 time steps
    are discarded from all outputs.

    References
    ----------
    Bollerslev, T. (1990). Modelling the coherence in short-run nominal
    exchange rates: A multivariate generalized ARCH model. *Review of
    Economics and Statistics*, 72(3), 498-505.
    """
    # ==================================================================
    # Input validation
    # Ref: ccc_mvgarch_simulate.m:62-131
    # ==================================================================

    # --- Validate t ------------------------------------------------
    if not np.isscalar(t) or int(t) != t or t < 1:
        raise ValueError('T must be a positive integer.')
    t = int(t)

    # --- Validate k ------------------------------------------------
    if not np.isscalar(k) or int(k) != k or k < 2:
        raise ValueError(
            'K must be a positive integer greater than or equal to 2.'
        )
    k = int(k)

    # --- Validate and expand p (symmetric innovation order) --------
    # Ref: ccc_mvgarch_simulate.m:80-89
    p_arr = np.asarray(p, dtype=np.int64).ravel()
    if p_arr.size == 1:
        if p_arr[0] < 1 or int(p_arr[0]) != p_arr[0]:
            raise ValueError('P must be a positive integer if scalar.')
        p_arr = np.full(k, int(p_arr[0]), dtype=np.int64)
    else:
        if len(p_arr) != k:
            raise ValueError(
                'P must be a scalar or a K-element vector.'
            )
        if np.any(p_arr < 1):
            raise ValueError(
                'All elements of P must be positive integers.'
            )
    # Ensure native ints
    p_arr = p_arr.astype(np.int64)

    # --- Validate and expand o (asymmetric innovation order) -------
    # Ref: ccc_mvgarch_simulate.m:91-100
    o_arr = np.asarray(o, dtype=np.int64).ravel()
    if o_arr.size == 1:
        if o_arr[0] < 0 or int(o_arr[0]) != o_arr[0]:
            raise ValueError('O must be a non-negative integer if scalar.')
        o_arr = np.full(k, int(o_arr[0]), dtype=np.int64)
    else:
        if len(o_arr) != k:
            raise ValueError(
                'O must be a scalar or a K-element vector.'
            )
        if np.any(o_arr < 0):
            raise ValueError(
                'All elements of O must be non-negative integers.'
            )
    o_arr = o_arr.astype(np.int64)

    # --- Validate and expand q (conditional variance lag order) ----
    # Ref: ccc_mvgarch_simulate.m:102-111
    q_arr = np.asarray(q, dtype=np.int64).ravel()
    if q_arr.size == 1:
        if q_arr[0] < 0 or int(q_arr[0]) != q_arr[0]:
            raise ValueError('Q must be a non-negative integer if scalar.')
        q_arr = np.full(k, int(q_arr[0]), dtype=np.int64)
    else:
        if len(q_arr) != k:
            raise ValueError(
                'Q must be a scalar or a K-element vector.'
            )
        if np.any(q_arr < 0):
            raise ValueError(
                'All elements of Q must be non-negative integers.'
            )
    q_arr = q_arr.astype(np.int64)

    # --- Validate m ------------------------------------------------
    # Ref: ccc_mvgarch_simulate.m:113-118
    if m is None:
        m = 72
    if not np.isscalar(m) or int(m) != m or m < 1:
        raise ValueError('M must be a positive integer.')
    m = int(m)

    # --- Validate parameter vector length --------------------------
    # Ref: ccc_mvgarch_simulate.m:120-131
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    n_corr = k * (k - 1) // 2
    parameter_count = (
        k
        + int(np.sum(p_arr))
        + int(np.sum(o_arr))
        + int(np.sum(q_arr))
        + n_corr
    )
    if len(parameters) != parameter_count:
        raise ValueError(
            'PARAMETERS must have K + sum(P) + sum(O) + sum(Q) + K*(K-1)/2 '
            f'elements. Expected {parameter_count}, got {len(parameters)}.'
        )

    # --- Extract and validate correlation matrix R -----------------
    # Ref: ccc_mvgarch_simulate.m:128-131
    R = _corr_ivech(parameters[-n_corr:])
    eig_vals = eigvalsh(R)
    if np.min(eig_vals) <= 0.0:
        raise ValueError(
            'The correlation matrix R implied by PARAMETERS must be '
            'positive definite.'
        )
    if not np.allclose(R, R.T, atol=1e-12):
        raise ValueError(
            'The correlation matrix R must be symmetric.'
        )

    # ==================================================================
    # Simulation
    # Ref: ccc_mvgarch_simulate.m:140-176
    # ==================================================================

    # --- Burn-in ---------------------------------------------------
    # Ref: ccc_mvgarch_simulate.m:141-143
    burnin = 2000
    t_total = t + burnin

    # --- Generate correlated intra-daily normals -------------------
    # Ref: ccc_mvgarch_simulate.m:148-149
    rng = np.random.default_rng()
    intra_random_nums = (
        rng.standard_normal((m * t_total, k)) * np.sqrt(1.0 / m)
    )

    # Correlate using R^(0.5) — the symmetric matrix square root
    # Ref: ccc_mvgarch_simulate.m:149 — intraRandomNums*R^(0.5)
    R_sqrt = np.real(_scipy_sqrtm(R))
    intra_random_nums = intra_random_nums @ R_sqrt

    # --- Aggregate intra-daily returns to daily --------------------
    # Ref: ccc_mvgarch_simulate.m:150-151
    cum_random = np.cumsum(intra_random_nums, axis=0)
    # MATLAB m:m:m*t (1-indexed) → Python indices m-1, 2m-1, …, m*t_total-1
    indices = np.arange(m - 1, m * t_total, m)
    sampled = cum_random[indices]  # (t_total, k)
    # Ref: ccc_mvgarch_simulate.m:151 — diff([zeros(1,k); sampled])
    random_nums = np.diff(
        np.vstack([np.zeros((1, k), dtype=np.float64), sampled]),
        axis=0,
    )  # (t_total, k)

    # --- Per-series TARCH simulation -------------------------------
    # Ref: ccc_mvgarch_simulate.m:153-161
    data = np.zeros((t_total, k), dtype=np.float64)
    ht_mat = np.zeros((t_total, k), dtype=np.float64)

    param_start = 0
    for i in range(k):
        # Ref: ccc_mvgarch_simulate.m:157-158
        n_params_i = 1 + int(p_arr[i]) + int(o_arr[i]) + int(q_arr[i])
        tarch_params = parameters[param_start:param_start + n_params_i]

        # Ref: ccc_mvgarch_simulate.m:159 — tarch_simulate called with the
        # pre-generated random numbers vector, NORMAL errors, tarch_type=2
        sim_data_i, sim_ht_i = _tarch_simulate(
            random_nums[:, i],
            tarch_params,
            int(p_arr[i]),
            int(o_arr[i]),
            int(q_arr[i]),
            error_type='NORMAL',
            tarch_type=2,
        )
        data[:, i] = sim_data_i
        ht_mat[:, i] = sim_ht_i
        param_start += n_params_i

    # --- Construct conditional covariance and pseudo-RC matrices ---
    # Ref: ccc_mvgarch_simulate.m:164-171
    ht = np.zeros((k, k, t_total), dtype=np.float64)
    pseudorc = np.zeros((k, k, t_total), dtype=np.float64)

    for time_idx in range(t_total):
        # Ref: ccc_mvgarch_simulate.m:167 — vol = sqrt(htMat(i,:))
        vol = np.sqrt(np.maximum(ht_mat[time_idx, :], 0.0))
        # Ref: ccc_mvgarch_simulate.m:168 — ht(:,:,i) = R .* (vol'*vol)
        vol_outer = np.outer(vol, vol)
        ht[:, :, time_idx] = R * vol_outer

        # Ref: ccc_mvgarch_simulate.m:169-170 — pseudo realized covariance
        # Intra-daily returns scaled by daily volatility
        intra_slice = intra_random_nums[time_idx * m:(time_idx + 1) * m, :]
        r_scaled = intra_slice * vol[np.newaxis, :]
        pseudorc[:, :, time_idx] = r_scaled.T @ r_scaled

    # --- Truncate burn-in ------------------------------------------
    # Ref: ccc_mvgarch_simulate.m:174-176
    simulatedata = data[burnin:, :]
    ht = ht[:, :, burnin:]
    pseudorc = pseudorc[:, :, burnin:]

    return simulatedata, ht, pseudorc
