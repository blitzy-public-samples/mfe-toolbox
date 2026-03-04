"""
Simulation of symmetric and asymmetric BEKK(p, o, q) multivariate volatility models.

Generates simulated returns and conditional covariance matrices from a BEKK
model given a parameter specification.  Supports scalar, diagonal, and full
BEKK parameterisations.

The dynamics of a BEKK model are:

    H_t = C C' + sum_{i=1}^{p} A_i' eps_{t-i} eps_{t-i}' A_i
                + sum_{j=1}^{o} G_j' eta_{t-j} eta_{t-j}' G_j
                + sum_{l=1}^{q} B_l' H_{t-l} B_l

where eta_t = eps_t * 1(eps_t < 0)  (element-wise negative part).

Notes
-----
Migrated from ``multivariate/bekk_simulate.m`` (MATLAB MFE Toolbox v4.0,
Kevin Sheppard).

Translation decisions
---------------------
- ``type`` renamed to ``type_model`` to avoid shadowing the Python built-in.
- MATLAB ``randn(2*T, k)`` → ``numpy.random.default_rng().standard_normal(…)``.
- MATLAB ``Ht(:,:,i)^(0.5)`` (Schur-based matrix power) →
  ``scipy.linalg.sqrtm(Ht[:,:,i])`` for numerical parity.
- MATLAB ``repmat(eye(k), [1 1 2*T])`` →
  ``np.tile(np.eye(k).reshape(k, k, 1), (1, 1, 2*T))``.
- MATLAB 1-based loop indices adjusted to 0-based throughout.
- MATLAB ``C(:)`` column-major vectorisation → ``C.ravel(order='F')``.
- MATLAB ``kron(A, A)`` → ``np.kron(A, A)`` (identical operation).
- MATLAB ``eigs(m)`` → ``np.linalg.eigvals(m)`` with ``np.abs`` for
  spectral-radius computation.
- MATLAB ``ceil(rand(T,1)*T)`` → ``rng.integers(0, total_t, size=total_t)``
  (0-based sampling with replacement for burn-in prepend).

See Also
--------
bekk : BEKK model estimation driver.
bekk_parameter_transform : Unpack flat parameter vector into BEKK matrices.
bekk_likelihood : BEKK log-likelihood computation.
"""

import warnings

import numpy as np
from scipy.linalg import sqrtm

from mfe_toolbox.multivariate.bekk_parameter_transform import bekk_parameter_transform


def bekk_simulate(t, k, parameters, p=1, o=0, q=1, type_model='Scalar'):
    """
    Simulate a BEKK(p, o, q) multivariate GARCH process.

    Parameters
    ----------
    t : int or numpy.ndarray
        If scalar, the number of observations to simulate.  An internal
        burn-in of *t* additional observations is generated and discarded.
        If a T × K array, the rows are treated as pre-generated standardised
        innovations; T is inferred from the row count, and a burn-in of T
        rows sampled with replacement is prepended automatically.
    k : int
        Number of assets / series (determines the K × K covariance dimension).
    parameters : numpy.ndarray
        1-D parameter vector whose layout depends on *type_model*:

        * The first ``K*(K+1)/2`` elements encode the lower-triangular
          Cholesky half-vec of the covariance intercept ``C * C'``.
        * ``'Scalar'``:  ``p + o + q`` additional scalar entries.
        * ``'Diagonal'``: ``(p + o + q) * K`` additional diagonal entries.
        * ``'Full'``:     ``(p + o + q) * K²`` additional full-matrix entries.

    p : int, optional
        Number of symmetric innovation lags (default 1, must be ≥ 1).
    o : int, optional
        Number of asymmetric innovation lags (default 0, must be ≥ 0).
    q : int, optional
        Number of conditional covariance lags (default 1, must be ≥ 0).
    type_model : {'Scalar', 'Diagonal', 'Full'}, optional
        BEKK parameterisation type (default ``'Scalar'``).  Case-insensitive.

    Returns
    -------
    data : numpy.ndarray
        T × K array of simulated returns.
    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices.

    Raises
    ------
    ValueError
        If *type_model* is not one of the three recognised strings, if *p* is
        not a positive integer, if *o* or *q* is negative, if *parameters*
        does not have the expected number of elements, or if a pre-generated
        innovation matrix *t* does not have *k* columns.

    Warns
    -----
    UserWarning
        If the spectral radius of the companion stationarity matrix exceeds 1,
        indicating that the supplied parameters do not correspond to the
        stationary region.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.bekk_simulate import bekk_simulate
    >>> # 2-asset scalar BEKK(1,1,1)
    >>> CCp = np.array([[1.0, 0.5], [0.5, 4.0]])
    >>> L = np.linalg.cholesky(CCp)               # lower-triangular factor
    >>> from mfe_toolbox.utility.chol2vec import chol2vec
    >>> params = np.concatenate([chol2vec(L), np.sqrt([0.05, 0.10, 0.88])])
    >>> data, Ht = bekk_simulate(500, 2, params, p=1, o=1, q=1,
    ...                          type_model='Scalar')
    >>> data.shape
    (500, 2)
    >>> Ht.shape
    (2, 2, 500)
    """
    # ------------------------------------------------------------------
    # Phase 1: Coerce parameter vector to 1-D float64
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Phase 2: Input validation — innovation matrix or scalar T
    # Ref: bekk_simulate.m:56-65
    # ------------------------------------------------------------------
    t_arr = np.asarray(t)
    if t_arr.ndim == 0:
        # Scalar: generate standard-normal innovations with a burn-in prefix
        # Ref: bekk_simulate.m:57 — e = randn(2*T, k)
        total_t = int(t_arr)
        if total_t <= 0:
            raise ValueError('T must be a positive integer when provided as a scalar.')
        e = rng.standard_normal((2 * total_t, k))
    else:
        # Matrix: use supplied innovations; prepend burn-in by resampling rows
        # Ref: bekk_simulate.m:59-64
        e = np.asarray(t, dtype=np.float64)
        if e.ndim != 2 or e.shape[1] != k:
            # Ref: bekk_simulate.m:61
            raise ValueError(
                'T must have K columns when providing simulated random numbers.'
            )
        total_t = e.shape[0]
        # Ref: bekk_simulate.m:64 — e = [e(ceil(rand(T,1)*T),:); e]
        # MATLAB ceil(rand(T,1)*T) → integers 1..T; Python 0-based: 0..T-1
        resample_idx = rng.integers(0, total_t, size=total_t)
        e = np.vstack([e[resample_idx, :], e])

    # ------------------------------------------------------------------
    # Phase 3: Convert type_model string → integer code
    # Ref: bekk_simulate.m:67-75
    # ------------------------------------------------------------------
    type_model_lower = str(type_model).strip().lower()
    if type_model_lower == 'scalar':
        type_code = 1
    elif type_model_lower == 'diagonal':
        type_code = 2
    elif type_model_lower == 'full':
        type_code = 3
    else:
        raise ValueError(
            "type_model must be 'Scalar', 'Diagonal' or 'Full'; "
            f"received {type_model!r}."
        )

    # ------------------------------------------------------------------
    # Phase 4: Validate lag orders p, o, q
    # Ref: bekk_simulate.m:77-85
    # ------------------------------------------------------------------
    p = int(p)
    o = int(o)
    q = int(q)
    if p <= 0:
        # Ref: bekk_simulate.m:78
        raise ValueError('P must be a positive scalar integer.')
    if o < 0:
        # Ref: bekk_simulate.m:81
        raise ValueError('O must be a non-negative scalar integer.')
    if q < 0:
        # Ref: bekk_simulate.m:84
        raise ValueError('Q must be a non-negative scalar integer.')

    # ------------------------------------------------------------------
    # Phase 5: Validate parameter vector length
    # Ref: bekk_simulate.m:87-99
    # ------------------------------------------------------------------
    k2 = k * (k + 1) // 2  # Ref: bekk_simulate.m:87 — k*(k+1)/2

    if type_code == 1:
        # Ref: bekk_simulate.m:90 — Scalar: one param per lag
        dyn_count = p + o + q
    elif type_code == 2:
        # Ref: bekk_simulate.m:92 — Diagonal: k params per lag
        dyn_count = (p + o + q) * k
    else:
        # Ref: bekk_simulate.m:94 — Full: k² params per lag
        dyn_count = (p + o + q) * k * k

    expected_count = dyn_count + k2  # Ref: bekk_simulate.m:96
    if len(parameters) != expected_count:
        # Ref: bekk_simulate.m:98
        raise ValueError(
            f'PARAMETERS does not have the expected number of elements. '
            f'Expected {expected_count}, received {len(parameters)}.'
        )

    # ------------------------------------------------------------------
    # Phase 6: Unpack parameter vector into BEKK matrices
    # Ref: bekk_simulate.m:100
    # ------------------------------------------------------------------
    C, A, G, B = bekk_parameter_transform(parameters, p, o, q, k, type_code)

    # ------------------------------------------------------------------
    # Phase 7: Build stationarity companion matrix
    #   m = kron(A_1,A_1) + … + kron(A_p,A_p)
    #     + 0.5*(kron(G_1,G_1) + … + kron(G_o,G_o))
    #     + kron(B_1,B_1) + … + kron(B_q,B_q)
    # Ref: bekk_simulate.m:101-110
    # ------------------------------------------------------------------
    k_sq = k * k
    m = np.zeros((k_sq, k_sq))  # Ref: bekk_simulate.m:101 — zeros(k*k)

    for i in range(p):
        # Ref: bekk_simulate.m:103
        m += np.kron(A[:, :, i], A[:, :, i])
    for i in range(o):
        # Ref: bekk_simulate.m:106 — 0.5 factor for asymmetric terms
        m += 0.5 * np.kron(G[:, :, i], G[:, :, i])
    for i in range(q):
        # Ref: bekk_simulate.m:109
        m += np.kron(B[:, :, i], B[:, :, i])

    # ------------------------------------------------------------------
    # Phase 8: Stationarity check and backcast computation
    # Ref: bekk_simulate.m:112-118
    # ------------------------------------------------------------------
    eigs = np.linalg.eigvals(m)
    # Ref: bekk_simulate.m:112 — max(eigs(m)) > 1
    # Use absolute values for robustness with potentially complex eigenvalues
    spectral_radius = float(np.max(np.abs(eigs)))

    if spectral_radius > 1.0:
        # Ref: bekk_simulate.m:113 — backCast = C / .001
        back_cast = C / 0.001
        # Ref: bekk_simulate.m:114
        warnings.warn(
            'The parameters do not correspond to the stationary region.',
            UserWarning,
            stacklevel=2,
        )
    else:
        # Ref: bekk_simulate.m:116 — backCast = ((eye(k*k)-m)\eye(k*k))*C(:)
        # Solve (I - m) @ x = C_vec  for the stationary unconditional covariance.
        # Column-major vectorisation matches MATLAB C(:) / kron convention.
        identity_kk = np.eye(k_sq)
        c_vec = C.ravel(order='F')  # Ref: bekk_simulate.m:116 — C(:)
        backcast_vec = np.linalg.solve(identity_kk - m, c_vec)
        # Ref: bekk_simulate.m:117 — reshape(backCast, k, k)
        back_cast = backcast_vec.reshape((k, k), order='F')

    # ------------------------------------------------------------------
    # Phase 9: Initialise arrays for the main simulation loop
    # Ref: bekk_simulate.m:122-123
    # ------------------------------------------------------------------
    total_length = 2 * total_t

    # Ref: bekk_simulate.m:122 — Ht = repmat(eye(k), [1 1 2*T])
    Ht = np.tile(np.eye(k).reshape(k, k, 1), (1, 1, total_length))

    # Ref: bekk_simulate.m:123 — eta = zeros(size(e))
    eta = np.zeros_like(e)

    # ------------------------------------------------------------------
    # Phase 10: Main BEKK simulation loop
    # Ref: bekk_simulate.m:124-150
    # ------------------------------------------------------------------
    for i in range(total_length):
        # Ref: bekk_simulate.m:125 — Ht(:,:,i) = C
        Ht[:, :, i] = C.copy()

        # --- Symmetric innovation contribution (A matrices) ---
        # Ref: bekk_simulate.m:126-131
        for j in range(p):
            # MATLAB 1-based j → Python 0-based: lag index = i - (j+1)
            lag_idx = i - j - 1
            if lag_idx < 0:
                # Ref: bekk_simulate.m:128 — pre-sample: use back_cast
                Ht[:, :, i] += A[:, :, j].T @ back_cast @ A[:, :, j]
            else:
                # Ref: bekk_simulate.m:130 — A'*(e'*e)*A
                outer_ee = np.outer(e[lag_idx, :], e[lag_idx, :])
                Ht[:, :, i] += A[:, :, j].T @ outer_ee @ A[:, :, j]

        # --- Asymmetric innovation contribution (G matrices) ---
        # Ref: bekk_simulate.m:133-138
        for j in range(o):
            lag_idx = i - j - 1
            if lag_idx < 0:
                # Ref: bekk_simulate.m:135 — pre-sample: use back_cast
                Ht[:, :, i] += G[:, :, j].T @ back_cast @ G[:, :, j]
            else:
                # Ref: bekk_simulate.m:137 — G'*(eta'*eta)*G
                outer_eta = np.outer(eta[lag_idx, :], eta[lag_idx, :])
                Ht[:, :, i] += G[:, :, j].T @ outer_eta @ G[:, :, j]

        # --- Smoothing / lagged covariance contribution (B matrices) ---
        # Ref: bekk_simulate.m:140-145
        for j in range(q):
            lag_idx = i - j - 1
            if lag_idx < 0:
                # Ref: bekk_simulate.m:142 — pre-sample: use back_cast
                Ht[:, :, i] += B[:, :, j].T @ back_cast @ B[:, :, j]
            else:
                # Ref: bekk_simulate.m:144 — B'*H(t-lag)*B
                Ht[:, :, i] += B[:, :, j].T @ Ht[:, :, lag_idx] @ B[:, :, j]

        # --- Matrix square root and innovation scaling ---
        # Ref: bekk_simulate.m:147 — Ht12 = Ht(:,:,i)^(0.5)
        # MATLAB uses Schur-based matrix power; scipy.linalg.sqrtm replicates
        # this for numerical parity (NOT Cholesky, which gives a triangular
        # factor rather than the symmetric principal square root).
        Ht12 = sqrtm(Ht[:, :, i])
        # sqrtm may introduce tiny imaginary artefacts for near-singular Ht
        if np.iscomplexobj(Ht12):
            Ht12 = np.real(Ht12)

        # Ref: bekk_simulate.m:148 — e(i,:) = e(i,:) * Ht12
        # Post-multiply row vector by matrix square root to induce covariance
        e[i, :] = e[i, :] @ Ht12

        # Ref: bekk_simulate.m:149 — eta(i,:) = e(i,:) .* (e(i,:) < 0)
        # Asymmetric innovation: retain only negative elements, zero the rest
        eta[i, :] = e[i, :] * (e[i, :] < 0)

    # ------------------------------------------------------------------
    # Phase 11: Trim burn-in and return
    # Ref: bekk_simulate.m:152-153
    # MATLAB e(T+1:2*T,:) is 1-based → Python e[total_t:2*total_t,:]
    # ------------------------------------------------------------------
    data = e[total_t:2 * total_t, :]
    Ht = Ht[:, :, total_t:2 * total_t]

    return data, Ht
