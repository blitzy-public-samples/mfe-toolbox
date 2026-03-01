"""
Scalar VT-VECH multivariate GARCH simulation.

Migrated from multivariate/scalar_vt_vech_simulate.m (154 lines).
Generates simulated returns, conditional covariances, and pseudo-realized
covariances from a Scalar Variance-Targeting VECH model with symmetric
and asymmetric components.

The conditional covariance H(t) follows the scalar VT-VECH recursion:
    H(t) = C + sum_j alpha(j)*r_{t-j}'*r_{t-j}
              + sum_j gamma(j)*n_{t-j}'*n_{t-j}
              + sum_j beta(j)*H_{t-j}
where n_{t} = r_{t} * (r_{t} < 0) is the asymmetric (leverage) component
and C is the intercept covariance matrix scaled by the variance targeting
constraint: C = (1 - sum(alpha) - 0.5*sum(gamma) - sum(beta)) * intercept.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: Python migration from MATLAB
"""

import warnings

import numpy as np


def scalar_vt_vech_simulate(t, parameters, intercept, p=1, o=0, q=1,
                            burn_in=500, m=72):
    """
    Simulate from a Symmetric and Asymmetric Scalar VT-VECH(P,O,Q) model.

    Simulates multivariate return data from a scalar variance-targeting VECH
    model. The conditional covariance H(t) follows:

        H(t) = C + alpha(1)*r_{t-1}'*r_{t-1} + ... + alpha(p)*r_{t-p}'*r_{t-p}
             + gamma(1)*n_{t-1}'*n_{t-1} + ... + gamma(o)*n_{t-o}'*n_{t-o}
             + beta(1)*H_{t-1} + ... + beta(q)*H_{t-q}

    where C is the variance-targeting intercept, r_t are return vectors, and
    n_t = r_t .* (r_t < 0) captures the asymmetric (leverage) component.

    Parameters
    ----------
    t : int
        Length of the time series to simulate (number of observations after
        removing the burn-in period). Must be a positive integer.
    parameters : array_like
        A (p+o+q,) parameter vector ordered as:
        [alpha(1), ..., alpha(p), gamma(1), ..., gamma(o), beta(1), ..., beta(q)].
        All elements must be non-negative.
    intercept : array_like
        K x K positive-definite intercept covariance matrix. This is the
        variance-targeting matrix C used in the model recursion.
    p : int, optional
        Number of symmetric innovation lags. Must be >= 1. Default is 1.
    o : int, optional
        Number of asymmetric (leverage) lags. Must be >= 0. Default is 0.
    q : int, optional
        Number of conditional covariance lags. Must be >= 0. Default is 1.
    burn_in : int, optional
        Number of initial observations to discard for stationarity.
        Default is 500. Ref: scalar_vt_vech_simulate.m uses hardcoded
        burnin=2000; Python schema specifies 500 as default.
    m : int, optional
        Number of intradaily returns per period for pseudo-realized covariance
        computation. If m=1, pseudo_rc is just the outer product of the
        simulated data. Default is 72.

    Returns
    -------
    data : numpy.ndarray
        T x K matrix of simulated returns.
    Ht : numpy.ndarray
        K x K x T array of simulated conditional covariance matrices.
    pseudo_rc : numpy.ndarray
        K x K x T array of pseudo-realized covariance matrices computed from
        m intradaily simulated returns.

    Raises
    ------
    ValueError
        If any input parameter fails validation (non-positive t, non-PD
        intercept, negative parameters, invalid p/o/q/m, etc.).

    Notes
    -----
    Pseudo Realized Covariances are simulated by generating m intradaily
    returns from N(0, 1/m) and computing the Realized Covariance of these.
    If m=1, then pseudo_rc is just the outer product of the simulated data.

    The function generates ``burn_in`` extra observations at the beginning
    of the simulation to minimize start-up bias, which are discarded before
    returning results.

    The Cholesky decomposition is used in place of MATLAB's matrix square root
    (A^(0.5)) for generating covariance-scaled innovations. Both approaches
    produce statistically equivalent multivariate Gaussian draws with the
    correct conditional covariance structure.

    See Also
    --------
    scalar_vt_vech : Scalar VT-VECH estimation driver.
    scalar_vt_vech_likelihood : Scalar VT-VECH log-likelihood evaluation.

    References
    ----------
    Patton, A. J. and Sheppard, K. (2009). Evaluating Volatility and
    Correlation Forecasts. Handbook of Financial Time Series. Springer.

    Examples
    --------
    >>> import numpy as np
    >>> intercept = np.array([[0.04, 0.01], [0.01, 0.09]])
    >>> params = np.array([0.05, 0.90])  # alpha=0.05, beta=0.90
    >>> data, Ht, prc = scalar_vt_vech_simulate(500, params, intercept)
    >>> data.shape
    (500, 2)
    >>> Ht.shape
    (2, 2, 500)
    """
    # --- Input validation ---
    # Ref: scalar_vt_vech_simulate.m:49-56 — parameter conversion
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    intercept = np.asarray(intercept, dtype=np.float64)

    # Validate t: must be a positive scalar integer
    # Ref: scalar_vt_vech_simulate.m:59-61
    if not np.isscalar(t) or int(t) != t or t <= 0:
        raise ValueError('T must be a positive scalar integer value.')
    t = int(t)

    # Validate p: must be a positive integer (>= 1)
    # Ref: scalar_vt_vech_simulate.m:64-66
    if not np.isscalar(p) or int(p) != p or p < 1:
        raise ValueError('P must be a positive scalar integer.')
    p = int(p)

    # Validate o: must be a non-negative integer
    # Ref: scalar_vt_vech_simulate.m:68-70
    if not np.isscalar(o) or int(o) != o or o < 0:
        raise ValueError('O must be a non-negative scalar integer.')
    o = int(o)

    # Validate q: must be a non-negative integer
    # Ref: scalar_vt_vech_simulate.m:72-74
    if not np.isscalar(q) or int(q) != q or q < 0:
        raise ValueError('Q must be a non-negative scalar integer.')
    q = int(q)

    # Validate intercept: must be square and positive definite
    # Ref: scalar_vt_vech_simulate.m:77-79
    if intercept.ndim != 2 or intercept.shape[0] != intercept.shape[1]:
        raise ValueError('Intercept must be a K x K square matrix.')
    k = intercept.shape[0]
    # Ref: scalar_vt_vech_simulate.m:79 — min(eig(c))<0
    eig_min = np.min(np.linalg.eigvalsh(intercept))
    if eig_min <= 0:
        raise ValueError(
            'Intercept must be a positive definite covariance matrix.'
        )

    # Validate parameters: all non-negative and correct length
    # Ref: scalar_vt_vech_simulate.m:83-84
    if len(parameters) < (p + o + q):
        raise ValueError(
            'PARAMETERS must have at least p+o+q elements.'
        )
    if np.any(parameters < 0):
        raise ValueError(
            'All PARAMETERS must be non-negative.'
        )

    # Validate m: must be a positive integer
    # Ref: scalar_vt_vech_simulate.m:92-94
    if not np.isscalar(m) or int(m) != m or m < 1:
        raise ValueError('M must be a positive integer.')
    m = int(m)

    # Validate burn_in: must be a non-negative integer
    if not np.isscalar(burn_in) or int(burn_in) != burn_in or burn_in < 0:
        raise ValueError('burn_in must be a non-negative integer.')
    burn_in = int(burn_in)

    # --- Extract parameter groups ---
    # Ref: scalar_vt_vech_simulate.m:125-127
    alpha = parameters[:p]
    gamma = parameters[p:p + o]
    beta = parameters[p + o:p + o + q]

    # --- Stationarity check ---
    # Ref: scalar_vt_vech_simulate.m:86-91
    # Persistence = sum(alpha) + 0.5*sum(gamma) + sum(beta)
    persistence = np.sum(alpha) + 0.5 * np.sum(gamma) + np.sum(beta)
    if persistence >= 1.0:
        # Ref: scalar_vt_vech_simulate.m:87 — warning('MFEToolbox:Stationarity', ...)
        warnings.warn(
            'Parameters are not compatible with a covariance stationary '
            'process. Please check Ht for problems.',
            stacklevel=2,
        )
        # Ref: scalar_vt_vech_simulate.m:89 — use c/(1-0.99) when non-stationary
        initial_value = intercept / (1.0 - 0.99)
    else:
        # Ref: scalar_vt_vech_simulate.m:91 — unconditional covariance
        initial_value = intercept / (1.0 - persistence)

    # --- Simulation setup ---
    # Ref: scalar_vt_vech_simulate.m:99-105
    # Increase t by burn-in amount
    t_total = t + burn_in

    # Draw normal random numbers for intradaily returns
    # Ref: scalar_vt_vech_simulate.m:108 — randn(m*t, k) * sqrt(1/m)
    rng = np.random.default_rng()
    intra_random_nums = rng.standard_normal((m * t_total, k)) * np.sqrt(
        1.0 / m
    )

    # Compute daily returns from intradaily via cumulative sum and differencing
    # Ref: scalar_vt_vech_simulate.m:109-110
    # MATLAB: randomNums = cumsum(intraRandomNums);
    #         randomNums = diff([zeros(1,k); randomNums(m:m:m*t,:)]);
    random_nums_cumsum = np.cumsum(intra_random_nums, axis=0)
    # Ref: scalar_vt_vech_simulate.m:110 — randomNums(m:m:m*t,:)
    # MATLAB 1-indexed m:m:m*t selects rows m, 2m, 3m, ... = 0-indexed m-1, 2m-1, ...
    sampled = random_nums_cumsum[m - 1::m, :]  # shape: (t_total, k)
    # Ref: scalar_vt_vech_simulate.m:110 — diff([zeros(1,k); ...])
    random_nums = np.diff(
        np.vstack([np.zeros((1, k)), sampled]), axis=0
    )

    # Determine back cast length: max lag for pre-sample initialization
    # Ref: scalar_vt_vech_simulate.m:113 — n = max(max(p,o),q)
    n = max(p, o, q)

    # Initialize random numbers in back cast range to unconditional scale
    # Ref: scalar_vt_vech_simulate.m:116 — randomNums(1:n,:)*initialValue^(0.5)
    # MATLAB: initialValue^(0.5) is the symmetric matrix square root.
    # Python: use Cholesky decomposition for efficiency (statistically equivalent).
    # np.linalg.cholesky returns lower-triangular L such that L @ L.T = A.
    # For right-multiplication: z @ L.T produces N(0, L @ L.T) = N(0, A).
    chol_init = np.linalg.cholesky(initial_value)
    random_nums[:n, :] = random_nums[:n, :] @ chol_init.T

    # Initialize conditional covariance array
    # Ref: scalar_vt_vech_simulate.m:119 — ht = repmat(initialValue, [1 1 t])
    ht = np.tile(initial_value[:, :, np.newaxis], (1, 1, t_total))

    # Initialize pseudo-RC array
    # Ref: scalar_vt_vech_simulate.m:122 — pseudorc = repmat(initialValue, [1 1 t])
    pseudorc = np.tile(initial_value[:, :, np.newaxis], (1, 1, t_total))

    # Set data equal to random nums (will be overwritten in recursion for i >= n)
    # Ref: scalar_vt_vech_simulate.m:130
    data = random_nums.copy()

    # Initialize asymmetric component: eta = data * sqrt(0.5) for back-cast period
    # Ref: scalar_vt_vech_simulate.m:131 — eta = data.*sqrt(0.5)
    # This provides a rough initialization for the asymmetric back-cast:
    # since E[data .* (data<0)] ~ data * 0.5 in variance terms,
    # scaling by sqrt(0.5) approximates the asymmetric contribution.
    eta = data * np.sqrt(0.5)

    # --- VT-VECH recursion ---
    # Ref: scalar_vt_vech_simulate.m:133-150
    for i in range(n, t_total):
        # Ref: scalar_vt_vech_simulate.m:134 — ht(:,:,i) = c
        ht[:, :, i] = intercept.copy()

        # Symmetric innovation terms: alpha(j) * r_{t-j}' * r_{t-j}
        # Ref: scalar_vt_vech_simulate.m:135-137
        # MATLAB: for j=1:p, ht(:,:,i) = ht(:,:,i) + alpha(j)*(data(i-j,:)'*data(i-j,:))
        # Python: 0-indexed j from 0 to p-1, lag index = i - (j+1)
        for j in range(p):
            lag_idx = i - (j + 1)
            ht[:, :, i] += alpha[j] * np.outer(
                data[lag_idx, :], data[lag_idx, :]
            )

        # Asymmetric (leverage) terms: gamma(j) * n_{t-j}' * n_{t-j}
        # Ref: scalar_vt_vech_simulate.m:138-140
        for j in range(o):
            lag_idx = i - (j + 1)
            ht[:, :, i] += gamma[j] * np.outer(
                eta[lag_idx, :], eta[lag_idx, :]
            )

        # Conditional covariance lag terms: beta(j) * H_{t-j}
        # Ref: scalar_vt_vech_simulate.m:141-143
        for j in range(q):
            lag_idx = i - (j + 1)
            ht[:, :, i] += beta[j] * ht[:, :, lag_idx]

        # Symmetrize to correct for any floating-point asymmetry
        # Ref: scalar_vt_vech_simulate.m:144 — ht(:,:,i) = (ht(:,:,i)+ht(:,:,i)')/2
        ht[:, :, i] = (ht[:, :, i] + ht[:, :, i].T) / 2.0

        # Compute Cholesky factor for covariance-scaled sampling
        # Ref: scalar_vt_vech_simulate.m:145 — ht12 = ht(:,:,i)^(0.5)
        # MATLAB uses matrix square root; Python uses Cholesky (equivalent for PD)
        ht12 = np.linalg.cholesky(ht[:, :, i])  # lower triangular L

        # Generate pseudo-realized covariance from intradaily returns
        # Ref: scalar_vt_vech_simulate.m:146
        # MATLAB: r = intraRandomNums((i-1)*m+1:i*m,:)*ht12  (1-indexed i)
        # Python: intra_random_nums[i*m:(i+1)*m, :] @ ht12.T  (0-indexed i)
        # Since ht12 is lower-triangular L, right-multiply by L.T to match
        # MATLAB's right-multiply by symmetric matrix square root.
        r = intra_random_nums[i * m:(i + 1) * m, :] @ ht12.T

        # Pseudo-realized covariance: RC(t) = r' * r
        # Ref: scalar_vt_vech_simulate.m:147 — pseudorc(:,:,i) = r'*r
        pseudorc[:, :, i] = r.T @ r

        # Scale data by conditional covariance Cholesky factor
        # Ref: scalar_vt_vech_simulate.m:148 — data(i,:)=data(i,:)*ht12
        data[i, :] = data[i, :] @ ht12.T

        # Compute asymmetric indicator: n_t = r_t * (r_t < 0)
        # Ref: scalar_vt_vech_simulate.m:149 — eta(i,:)=data(i,:).*(data(i,:)<0)
        eta[i, :] = data[i, :] * (data[i, :] < 0)

    # --- Truncate burn-in period ---
    # Ref: scalar_vt_vech_simulate.m:152-154
    simulated_data = data[burn_in:, :]
    ht_out = ht[:, :, burn_in:]
    pseudorc_out = pseudorc[:, :, burn_in:]

    return simulated_data, ht_out, pseudorc_out
