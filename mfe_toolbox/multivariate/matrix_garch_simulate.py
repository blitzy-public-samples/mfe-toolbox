"""
Matrix GARCH multivariate simulation.

Migrated from multivariate/matrix_garch_simulate.m (199 lines).
Simulates symmetric and asymmetric MATRIX multivariate GARCH models
including conditional covariances and pseudo-realized covariances.

The conditional covariance H(t) follows the Matrix GARCH recursion:

    H(t) = CC' + sum_j AA'(j) .* r_{t-j}'*r_{t-j}
               + sum_j GG'(j) .* n_{t-j}'*n_{t-j}
               + sum_j BB'(j) .* H(t-j)

where .* denotes element-wise (Hadamard) multiplication,
r_t are return vectors, and n_t = r_t .* (r_t < 0).

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: Python migration from MATLAB
"""

import warnings

import numpy as np
from scipy.linalg import sqrtm

from mfe_toolbox.utility.vec2chol import vec2chol


def matrix_garch_simulate(t, k, parameters, p=1, o=0, q=1, m=72):
    """
    Simulate from a symmetric or asymmetric Matrix GARCH model.

    Simulates multivariate return data from a Matrix GARCH model where the
    conditional covariance H(t) follows:

        H(t) = CC' + AA'(1) .* r_{t-1}'*r_{t-1} + ...
             + GG'(1) .* n_{t-1}'*n_{t-1} + ...
             + BB'(1) .* H(t-1) + ...

    where .* denotes Hadamard (element-wise) product, r_t are simulated return
    vectors, and n_t = r_t .* (r_t < 0) captures the asymmetric component.

    Parameters
    ----------
    t : int
        Length of the time series to simulate (number of observations after
        removing the burn-in period). Must be a positive integer.
    k : int
        Cross-sectional dimension (number of assets). Must be >= 2.
    parameters : array_like
        Either a 3-D array of shape (K, K, 1+P+O+Q) where:
            parameters[:,:,0] = CC' (intercept)
            parameters[:,:,1:P+1] = AA'(j) symmetric innovation matrices
            parameters[:,:,P+1:P+O+1] = GG'(j) asymmetric matrices
            parameters[:,:,P+O+1:] = BB'(j) lag covariance matrices
        OR a 1-D vector of length K(K+1)/2 * (1+P+O+Q) containing stacked
        half-vec representations of lower-triangular Cholesky factors.
    p : int, optional
        Number of symmetric innovation lags. Must be >= 1. Default is 1.
    o : int, optional
        Number of asymmetric (leverage) lags. Must be >= 0. Default is 0.
    q : int, optional
        Number of conditional covariance lags. Must be >= 0. Default is 1.
    m : int, optional
        Number of intraday returns to simulate for pseudo-Realized
        Covariance. Default is 72.

    Returns
    -------
    simulate_data : numpy.ndarray
        T x K matrix of simulated return data.
    ht : numpy.ndarray
        K x K x T array of simulated conditional covariances.
    pseudo_rc : numpy.ndarray
        K x K x T array of pseudo-Realized Covariances constructed from
        m intraday returns per period.

    Raises
    ------
    ValueError
        If any input parameter is invalid (non-positive T, K < 2,
        non-integer orders, incompatible parameter dimensions).

    Notes
    -----
    A burn-in period of 2000 observations is used to minimize start-up bias.

    Pseudo Realized Covariances are simulated by generating m intraday
    returns from N(0, 1/m) and computing the realized covariance. If m=1,
    then pseudo_rc is the outer product of the simulated data.

    Matrix square root via ``scipy.linalg.sqrtm`` replaces MATLAB's
    ``Ht^(0.5)`` matrix power operation.

    References
    ----------
    Ref: matrix_garch_simulate.m — Kevin Sheppard (2011)

    See Also
    --------
    matrix_garch : Matrix GARCH estimation driver.
    matrix_garch_likelihood : Matrix GARCH log-likelihood.
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: matrix_garch_simulate.m:54-92
    # ------------------------------------------------------------------
    # Ref: matrix_garch_simulate.m:64-65 — T must be a positive integer
    if not isinstance(t, (int, np.integer)) or t < 1:
        raise ValueError('T must be a positive integer.')

    # Ref: matrix_garch_simulate.m:68-69 — K must be >= 2
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError(
            'K must be a positive integer greater than or equal to 2.'
        )

    # Ref: matrix_garch_simulate.m:72-73 — P must be >= 1
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError('P must be a positive integer if scalar.')

    # Ref: matrix_garch_simulate.m:78-79 — O must be >= 0
    if not isinstance(o, (int, np.integer)) or o < 0:
        raise ValueError('O must be a non-negative integer if scalar.')

    # Ref: matrix_garch_simulate.m:83-84 — Q must be >= 0
    if not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError('Q must be a non-negative integer if scalar.')

    # Ref: matrix_garch_simulate.m:90-92 — M must be a positive integer
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError('M must be a positive integer.')

    # Convert parameters to numpy float64 array
    parameters = np.asarray(parameters, dtype=np.float64)

    # Number of unique elements in lower triangular K x K matrix
    # Ref: matrix_garch_simulate.m:94
    k2 = k * (k + 1) // 2

    # ------------------------------------------------------------------
    # Parameter format handling
    # Ref: matrix_garch_simulate.m:95-118
    # ------------------------------------------------------------------
    if parameters.ndim <= 2:
        # Vector form: length must be K(K+1)/2 * (1+P+O+Q)
        parameters_flat = parameters.ravel()
        parameter_count = k2 * (1 + p + o + q)

        # Ref: matrix_garch_simulate.m:100-101
        if len(parameters_flat) != parameter_count:
            raise ValueError(
                'PARAMETERS must be K(K+1)/2*(1+P+O+Q) when using a vector.'
            )

        # Allocate 3-D parameter matrix array
        # Ref: matrix_garch_simulate.m:103
        param_matrices = np.zeros((k, k, 1 + p + o + q))
        index = 0

        # Ref: matrix_garch_simulate.m:105 — MATLAB loop is 1:(1+p+o+1),
        # i.e. (2+p+o) iterations. This faithfully preserves the original
        # MATLAB behavior. Note: when q != 1, the loop bound does not
        # include all B matrices (the original code uses 1+p+o+1 rather
        # than 1+p+o+q). For the default case q=1, these are equivalent.
        num_to_unpack = 2 + p + o
        for i in range(num_to_unpack):
            # Ref: matrix_garch_simulate.m:106 — vec2chol converts K(K+1)/2
            # vector to K x K lower-triangular Cholesky factor
            temp = vec2chol(parameters_flat[index:index + k2])
            # Ref: matrix_garch_simulate.m:107 — symmetrize: (temp*temp' +
            # temp*temp')/2 is a redundant symmetrization of temp @ temp.T;
            # preserved for numerical parity with MATLAB
            sym = temp @ temp.T
            param_matrices[:, :, i] = (sym + sym.T) / 2.0
            index += k2
    elif parameters.ndim == 3:
        # 3-D matrix form: must be K x K x (1+P+O+Q)
        # Ref: matrix_garch_simulate.m:112-113
        expected_shape = (k, k, 1 + p + o + q)
        if parameters.shape != expected_shape:
            raise ValueError(
                'PARAMETERS must be K by K by (1+P+O+Q) when using a '
                '3-D matrix.'
            )
        param_matrices = parameters.copy()
    else:
        raise ValueError(
            'The size of PARAMETERS is not compatible with this function.'
        )

    # ------------------------------------------------------------------
    # Stationarity check
    # Ref: matrix_garch_simulate.m:120-136
    # ------------------------------------------------------------------
    num_slices = param_matrices.shape[2]
    sum_parameters = np.zeros((k, k))

    # Ref: matrix_garch_simulate.m:122-129 — sum parameter matrices with
    # weights: A matrices (indices 1..p) get w=1, G matrices
    # (indices p+1..p+o) get w=0.5, B matrices (indices > p+o) get w=1.
    for i in range(1, num_slices):
        # Ref: matrix_garch_simulate.m:123-126 — MATLAB 1-based condition:
        #   if i<=(p+1) || i>(1+p+o) then w=1, else w=0.5
        # Python 0-based equivalent: if i<=p or i>p+o then w=1, else w=0.5
        if i <= p or i > p + o:
            w = 1.0
        else:
            w = 0.5
        sum_parameters += w * param_matrices[:, :, i]

    # Ref: matrix_garch_simulate.m:130 — ensure symmetry
    sum_parameters = (sum_parameters + sum_parameters.T) / 2.0

    # Ref: matrix_garch_simulate.m:131-136
    if np.max(np.diag(sum_parameters)) > 1.0:
        warnings.warn(
            'The parameters do not correspond to a stationary solution. '
            'Check for overflow.',
            stacklevel=2
        )
        # Ref: matrix_garch_simulate.m:133 — nonstationary fallback
        uncond = param_matrices[:, :, 0] / 0.005
    else:
        # Ref: matrix_garch_simulate.m:135 — element-wise division by
        # (ones(k) - sumParameters) to get unconditional covariance
        uncond = param_matrices[:, :, 0] / (
            np.ones((k, k)) - sum_parameters
        )

    # ------------------------------------------------------------------
    # Simulation setup
    # Ref: matrix_garch_simulate.m:141-149
    # ------------------------------------------------------------------
    burnin = 2000  # Ref: matrix_garch_simulate.m:141
    total_t = t + burnin  # Ref: matrix_garch_simulate.m:143

    # Draw intraday normal random numbers scaled by sqrt(1/m)
    # Ref: matrix_garch_simulate.m:146
    rng = np.random.default_rng()
    intra_random_nums = (
        rng.standard_normal((m * total_t, k)) * np.sqrt(1.0 / m)
    )

    # Aggregate intraday returns into daily returns via cumsum + subsample
    # Ref: matrix_garch_simulate.m:147-148
    cumulative = np.cumsum(intra_random_nums, axis=0)
    # Ref: matrix_garch_simulate.m:148 — MATLAB randomNums(m:m:m*t,:)
    # selects every m-th row (1-based: m, 2m, ...).
    # Python 0-based: indices m-1, 2m-1, etc.
    subsampled = cumulative[m - 1::m, :]
    # Ref: matrix_garch_simulate.m:148 — diff([zeros(1,k); subsampled])
    random_nums = np.diff(
        np.vstack([np.zeros((1, k)), subsampled]), axis=0
    )

    # ------------------------------------------------------------------
    # Main recursion
    # Ref: matrix_garch_simulate.m:152-183
    # ------------------------------------------------------------------
    back_cast = uncond.copy()       # Ref: matrix_garch_simulate.m:152
    back_cast_asym = uncond.copy()  # Ref: matrix_garch_simulate.m:153

    ht_full = np.zeros((k, k, total_t))     # Ref: matrix_garch_simulate.m:155
    simulated_data = np.zeros((total_t, k))  # Ref: matrix_garch_simulate.m:156

    for t_idx in range(total_t):
        # Ref: matrix_garch_simulate.m:158 — start with intercept CC'
        ht_full[:, :, t_idx] = param_matrices[:, :, 0].copy()

        # Ref: matrix_garch_simulate.m:159-166 — symmetric innovation terms
        # Hadamard product: AA'(j) .* (r_{t-j} * r_{t-j}')
        for j in range(1, p + 1):
            if t_idx - j < 0:
                # Ref: matrix_garch_simulate.m:161 — pre-sample uses backcast
                ht_full[:, :, t_idx] += (
                    param_matrices[:, :, j] * back_cast
                )
            else:
                # Ref: matrix_garch_simulate.m:163-164 — outer product r'*r
                r = simulated_data[t_idx - j, :]
                ht_full[:, :, t_idx] += (
                    param_matrices[:, :, j] * np.outer(r, r)
                )

        # Ref: matrix_garch_simulate.m:167-174 — asymmetric innovation terms
        # Hadamard product: GG'(j) .* (n_{t-j} * n_{t-j}')
        for j in range(1, o + 1):
            if t_idx - j < 0:
                # Ref: matrix_garch_simulate.m:169 — pre-sample asymmetric
                ht_full[:, :, t_idx] += (
                    param_matrices[:, :, p + j] * back_cast_asym
                )
            else:
                # Ref: matrix_garch_simulate.m:171-172 — n = r .* (r < 0)
                r_prev = simulated_data[t_idx - j, :]
                n = r_prev * (r_prev < 0)
                ht_full[:, :, t_idx] += (
                    param_matrices[:, :, p + j] * np.outer(n, n)
                )

        # Ref: matrix_garch_simulate.m:175-181 — lagged covariance terms
        # Hadamard product: BB'(j) .* H(t-j)
        for j in range(1, q + 1):
            if t_idx - j < 0:
                # Ref: matrix_garch_simulate.m:177 — pre-sample lag
                ht_full[:, :, t_idx] += (
                    param_matrices[:, :, p + o + j] * back_cast
                )
            else:
                # Ref: matrix_garch_simulate.m:179
                ht_full[:, :, t_idx] += (
                    param_matrices[:, :, p + o + j] * ht_full[:, :, t_idx - j]
                )

        # Ref: matrix_garch_simulate.m:182 — simData(t,:) = rNums(t,:)*Ht^(.5)
        # Use scipy.linalg.sqrtm for matrix square root; take real part to
        # discard negligible imaginary components from numerical artifacts
        ht_sqrt = np.real(sqrtm(ht_full[:, :, t_idx]))
        simulated_data[t_idx, :] = random_nums[t_idx, :] @ ht_sqrt

    # ------------------------------------------------------------------
    # Pseudo-Realized Covariance
    # Ref: matrix_garch_simulate.m:190-194
    # ------------------------------------------------------------------
    pseudo_rc_full = np.zeros((k, k, total_t))
    for t_idx in range(total_t):
        # Ref: matrix_garch_simulate.m:192 — intraday returns scaled by H^(0.5)
        ht_sqrt = np.real(sqrtm(ht_full[:, :, t_idx]))
        # Ref: matrix_garch_simulate.m:192 — MATLAB (t-1)*m+1:t*m is 1-based;
        # Python 0-based equivalent: t_idx*m : (t_idx+1)*m
        r_intra = intra_random_nums[t_idx * m:(t_idx + 1) * m, :] @ ht_sqrt
        # Ref: matrix_garch_simulate.m:193 — realized covariance = r' * r
        pseudo_rc_full[:, :, t_idx] = r_intra.T @ r_intra

    # ------------------------------------------------------------------
    # Trim burn-in
    # Ref: matrix_garch_simulate.m:197-199
    # ------------------------------------------------------------------
    simulate_data = simulated_data[burnin:total_t, :]
    pseudo_rc = pseudo_rc_full[:, :, burnin:total_t]
    ht = ht_full[:, :, burnin:total_t]

    return simulate_data, ht, pseudo_rc
