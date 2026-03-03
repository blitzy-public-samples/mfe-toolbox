"""
Simulation of HEAVY (High-frEquency-bAsed VolatilitY) volatility model.

Generates both return and realized measure series for a K-dimensional
volatility spillover model.  The HEAVY model of Shephard and Sheppard
captures cross-series volatility dynamics where realized measures (such as
realized variance computed from high-frequency data) can drive the conditional
variance of returns, and vice versa.

Dynamics are given by::

    h(t,:)' = O + A(:,:,1)*f(data(t-1,:))' + ... + A(:,:,maxP)*f(data(t-maxP,:))'
                + B(:,:,1)*h(t-1,:)' + ... + B(:,:,maxQ)*h(t-maxQ,:)'

where *f* depends on the type of each series:

* For return series (``m[i] == 1``): ``f(data) = data**2``
* For realized-measure series (``m[i] > 1``): ``f(data) = data``

Migrated from ``univariate/heavy_simulate.m`` (Version 4.0, Revision 1, 4/2/2012).
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import chi2, norm

from mfe_toolbox.univariate.heavy_parameter_transform import heavy_parameter_transform


def heavy_simulate(
    t: int | np.ndarray,
    k: int,
    parameters: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    m: np.ndarray | None = None,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate data from a HEAVY volatility model.

    Simulates ``T`` observations of a K-dimensional HEAVY process producing
    both return-like (mean-zero, variance ``h``) and realized-measure-like
    (mean ``h``) series depending on the ``m`` vector.

    Parameters
    ----------
    t : int or np.ndarray
        Either a scalar integer giving the number of periods to simulate, or a
        ``(T, K)`` array of pre-generated random variables.  When a matrix is
        provided, the ``k`` parameter is ignored and ``T``, ``K`` are inferred
        from the array shape.
    k : int
        Number of series to simulate.  Ignored when ``t`` is a matrix.
    parameters : np.ndarray
        A 1-D array with ``K + sum(sum(p)) + sum(sum(q))`` elements.
        Ordering: ``[O' A(0,0,0:p[0,0]) ... A(K-1,K-1,0:p[K-1,K-1])
        B(0,0,0:q[0,0]) ... B(K-1,K-1,0:q[K-1,K-1])]``.
    p : np.ndarray
        A ``(K, K)`` integer matrix where element ``(i, j)`` gives the number
        of lags of series *j* innovations in the equation for series *i*.
    q : np.ndarray
        A ``(K, K)`` integer matrix where element ``(i, j)`` gives the number
        of lags of series *j* conditional variance in the equation for series *i*.
    m : np.ndarray or None, optional
        A length-*K* vector of positive integers.  ``m[i] == 1`` means series
        *i* is return-like (mean zero, variance ``h``); ``m[i] > 1`` means
        series *i* is realized-measure-like with ``m[i]`` underlying
        observations (mean ``h``).  Default is ``numpy.ones(K, dtype=int)``.
    r : np.ndarray or None, optional
        A ``(K, K)`` positive-definite symmetric correlation matrix for the
        Gaussian copula used when generating random variables internally.
        Only used when ``t`` is scalar.  Default is ``numpy.eye(K)``.

    Returns
    -------
    data : np.ndarray
        A ``(T, K)`` array of simulated data.  For return series (``m == 1``),
        data has mean zero and variance ``h``.  For realized-measure series
        (``m > 1``), the mean of data is ``h``.
    ht : np.ndarray
        A ``(T, K)`` array of conditional variances.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, parameter count is wrong,
        ``p`` or ``q`` contain non-integer or negative values, ``m`` contains
        non-positive values, or ``r`` is not positive definite symmetric.

    Notes
    -----
    A burn-in period is used to reduce sensitivity to initial conditions:

    * ``T <= 500``: burn-in = ``T`` (total simulation length ``2*T``)
    * ``T > 500``: burn-in = ``500`` (total simulation length ``T + 500``)

    When ``t`` is a matrix of pre-generated random variables, no burn-in is
    applied and the provided data is used directly.

    See Also
    --------
    mfe_toolbox.univariate.heavy : HEAVY model estimation driver.
    mfe_toolbox.univariate.heavy_parameter_transform : Parameter vector unpacking.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.heavy_simulate import heavy_simulate
    >>> params = np.array([0.15, 0.05, 0.2, 0.4, 0.7, 0.55])
    >>> p_mat = np.array([[0, 1], [0, 1]])
    >>> q_mat = np.eye(2, dtype=int)
    >>> m_vec = np.array([1, 78])
    >>> data, ht = heavy_simulate(1000, 2, params, p_mat, q_mat, m_vec)
    >>> data.shape
    (1000, 2)
    """
    # ------------------------------------------------------------------
    # Phase 1: Input handling — determine scalar vs. matrix mode
    # Ref: heavy_simulate.m:56-78 — nargin-based dispatch
    # ------------------------------------------------------------------
    e_provided = False
    signs: np.ndarray | None = None  # only populated when generating random vars

    if not np.isscalar(t):
        # Ref: heavy_simulate.m:60-63 — T is a matrix of pre-generated random vars
        t_array = np.asarray(t, dtype=np.float64)
        if t_array.ndim != 2:
            raise ValueError(
                "When t is not scalar, it must be a 2-D array of shape (T, K)."
            )
        # Ref: heavy_simulate.m:62 — [T,K] = size(e); extract dimensions
        T: int = t_array.shape[0]
        K: int = t_array.shape[1]
        # Transpose from T×K (external format) to K×T (internal processing format)
        # Ref: heavy_simulate.m:61 — e=T; stored as-is, but simulation loop uses
        # e(:,t) which selects a K×1 column vector, so we transpose here
        e = t_array.T.copy()
        e_provided = True
        # Ref: heavy_simulate.m:63 — R = eye(K) when T is a matrix
        r = np.eye(K, dtype=np.float64)
    else:
        T = int(t)
        K = int(k)
        e = None  # will be generated below

    # ------------------------------------------------------------------
    # Phase 1b: Apply defaults for optional arguments
    # Ref: heavy_simulate.m:68-78 — switch nargin
    # ------------------------------------------------------------------
    if m is None:
        m = np.ones(K, dtype=np.intp)
    else:
        m = np.asarray(m, dtype=np.intp).ravel()

    if r is None:
        r = np.eye(K, dtype=np.float64)
    else:
        r = np.asarray(r, dtype=np.float64)

    # ------------------------------------------------------------------
    # Phase 1c: Input validation
    # Ref: heavy_simulate.m:80-98
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.intp)
    q = np.asarray(q, dtype=np.intp)

    # Validate p — K×K matrix of non-negative integers
    # Ref: heavy_simulate.m:80
    if p.shape != (K, K):
        raise ValueError("P must be a K by K matrix of non-negative integers.")
    if np.any(p < 0):
        raise ValueError("P must be a K by K matrix of non-negative integers.")

    # Validate q — K×K matrix of non-negative integers
    # Ref: heavy_simulate.m:83
    if q.shape != (K, K):
        raise ValueError("Q must be a K by K matrix of non-negative integers.")
    if np.any(q < 0):
        raise ValueError("Q must be a K by K matrix of non-negative integers.")

    # Validate parameter count
    # Ref: heavy_simulate.m:87-89
    count: int = K + int(np.sum(p)) + int(np.sum(q))
    if parameters.size != count:
        raise ValueError(
            f"PARAMETERS has the wrong number of inputs. "
            f"Expected {count} (K + sum(sum(P)) + sum(sum(Q))), got {parameters.size}."
        )

    # Validate m — length K with positive integers
    # Ref: heavy_simulate.m:92-94
    if m.size < K or np.any(m < 1):
        raise ValueError("M must be a K by 1 vector of positive integers.")

    # Validate R — K×K positive definite symmetric matrix
    # Ref: heavy_simulate.m:96-98
    if r.shape != (K, K):
        raise ValueError("R must be a K by K positive definite matrix")
    if not np.allclose(r, r.T, atol=1e-12):
        raise ValueError("R must be a K by K positive definite matrix")
    r_eigvals = np.linalg.eigvalsh(r)
    if np.min(r_eigvals) <= 0.0:
        raise ValueError("R must be a K by K positive definite matrix")

    # ------------------------------------------------------------------
    # Phase 2: Burn-in period calculation
    # Ref: heavy_simulate.m:104-108
    # ------------------------------------------------------------------
    if e_provided:
        # When pre-generated random variables are supplied, use all T periods
        # directly with no additional burn-in padding.
        tau: int = T
    elif T <= 500:
        tau = 2 * T
    else:
        tau = T + 500

    # ------------------------------------------------------------------
    # Phase 3: Random variable generation
    # Ref: heavy_simulate.m:110-123
    # ------------------------------------------------------------------
    if not e_provided:
        rng = np.random.default_rng()

        # Compute matrix square root of correlation matrix R
        # Ref: heavy_simulate.m:111 — u = R^(0.5)*randn(K,tau)
        # scipy.linalg.sqrtm replaces MATLAB R^(0.5) matrix power
        r_sqrt = np.real(sqrtm(r))

        # Generate correlated standard normals: K × tau
        raw_normals = rng.standard_normal((K, tau))
        e = r_sqrt @ raw_normals

        # Transform to uniform marginals using normal CDF
        # Ref: heavy_simulate.m:113 — u = normcdf(e)
        # scipy.stats.norm.cdf replaces MATLAB normcdf (from duplication/)
        u = norm.cdf(e)

        # Allocate sign storage for return series
        # Ref: heavy_simulate.m:114 — signs = zeros(K,tau)
        signs = np.zeros((K, tau), dtype=np.float64)

        # Ref: heavy_simulate.m:115-122 — per-series transformation
        for i in range(K):
            if m[i] != 1:
                # Realized measure series: chi-squared transform
                # Ref: heavy_simulate.m:117 — e(i,:) = chi2inv(u(i,:),m(i))/m(i)
                # scipy.stats.chi2.ppf replaces MATLAB chi2inv
                e[i, :] = chi2.ppf(u[i, :], int(m[i])) / float(m[i])
            else:
                # Return series: store signs, then square the innovations
                # Ref: heavy_simulate.m:119-120
                signs[i, :] = 2.0 * (e[i, :] > 0.0) - 1.0
                e[i, :] = e[i, :] ** 2

    # ------------------------------------------------------------------
    # Phase 4: Parameter transformation
    # Ref: heavy_simulate.m:125
    # ------------------------------------------------------------------
    O, A, B = heavy_parameter_transform(parameters, p, q, K)

    # ------------------------------------------------------------------
    # Phase 5: Stationarity check and unconditional variance
    # Ref: heavy_simulate.m:126-146
    # ------------------------------------------------------------------
    # Ref: heavy_simulate.m:128-129 — pMax, qMax
    pMax: int = int(np.max(p)) if p.size > 0 else 0
    qMax: int = int(np.max(q)) if q.size > 0 else 0
    # Ref: heavy_simulate.m:127 — pq = max([p(:)',q(:)'])
    pq: int = max(pMax, qMax)

    if pq == 0:
        # Degenerate case: no dynamics, constant variance
        uncond = O.copy()
    elif pq == 1:
        # Ref: heavy_simulate.m:131-132 — companion = A+B (squeezed to 2D)
        # A is (K,K,1) and B is (K,K,1); MATLAB auto-squeezes trailing dims
        companion = A[:, :, 0] + B[:, :, 0]
        eigenvalues = np.linalg.eig(companion)[0]

        if np.max(np.abs(eigenvalues)) < 1.0:
            # Stationary: compute unconditional variance
            # Ref: heavy_simulate.m:142
            # uncond = ((eye(K)-sum(A,3)-sum(B,3))\eye(K))*O
            # Equivalent: solve( I - sumA - sumB, O )
            sum_A = np.sum(A, axis=2)
            sum_B = np.sum(B, axis=2)
            uncond = np.linalg.solve(np.eye(K) - sum_A - sum_B, O)
        else:
            # Non-stationary: use inflated intercept
            # Ref: heavy_simulate.m:144-145
            uncond = O / 0.001
            warnings.warn(
                "The HEAVY process is non-stationary",
                stacklevel=2,
            )
    else:
        # Ref: heavy_simulate.m:134-138 — build companion matrix for pq > 1
        companion = np.zeros((K * pq, K * pq), dtype=np.float64)

        # Ref: heavy_simulate.m:135 — companion(K+1:K*pq,1:K*(pq-1)) = eye(K*(pq-1))
        # MATLAB 1-based [K+1, K*pq] × [1, K*(pq-1)] → Python 0-based [K, K*pq) × [0, K*(pq-1))
        if pq > 1:
            companion[K : K * pq, 0 : K * (pq - 1)] = np.eye(K * (pq - 1))

        # Ref: heavy_simulate.m:136 — companion(1:K,1:K*pMax) = reshape(A,[K K*pMax])
        # MATLAB column-major reshape ≡ Fortran-order reshape in NumPy
        if pMax > 0:
            companion[0:K, 0 : K * pMax] = A.reshape((K, K * pMax), order="F")

        # Ref: heavy_simulate.m:137 — companion(1:K,1:K*qMax) += reshape(B,[K K*qMax])
        if qMax > 0:
            companion[0:K, 0 : K * qMax] += B.reshape((K, K * qMax), order="F")

        eigenvalues = np.linalg.eig(companion)[0]

        if np.max(np.abs(eigenvalues)) < 1.0:
            # Ref: heavy_simulate.m:142
            sum_A = np.sum(A, axis=2)
            sum_B = np.sum(B, axis=2)
            uncond = np.linalg.solve(np.eye(K) - sum_A - sum_B, O)
        else:
            # Ref: heavy_simulate.m:144-145
            uncond = O / 0.001
            warnings.warn(
                "The HEAVY process is non-stationary",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Phase 6: Main simulation loop
    # Ref: heavy_simulate.m:147-168
    # ------------------------------------------------------------------
    data = np.zeros((K, tau), dtype=np.float64)
    ht = np.ones((K, tau), dtype=np.float64)
    backCast = uncond.copy()

    # Ref: heavy_simulate.m:151-168 — MATLAB loop t=1:tau (1-based)
    # Python loop t_idx=0..tau-1 (0-based)
    for t_idx in range(tau):
        # Ref: heavy_simulate.m:152 — ht(:,t) = O
        ht[:, t_idx] = O.copy()

        # Innovation lag contributions
        # Ref: heavy_simulate.m:153-158 — for j=1:pMax
        for j in range(pMax):
            # Ref: heavy_simulate.m:154 — if (t-j)<=0 → Python: t_idx <= j
            # MATLAB 1-based: t=1, j=1 → (t-j)=0 <=0 → use backCast
            # Python 0-based: t_idx=0, j=0 → t_idx<=j → 0<=0 → use backCast
            if t_idx <= j:
                ht[:, t_idx] += A[:, :, j] @ backCast
            else:
                # Ref: heavy_simulate.m:157 — data(:,t-j) → Python: data[:, t_idx-j-1]
                # MATLAB: data(:,t-j) where t,j 1-based → Python: data[:, (t_idx+1)-(j+1)-1]
                ht[:, t_idx] += A[:, :, j] @ data[:, t_idx - j - 1]

        # Conditional variance lag contributions
        # Ref: heavy_simulate.m:160-166 — for j=1:qMax
        for j in range(qMax):
            if t_idx <= j:
                ht[:, t_idx] += B[:, :, j] @ backCast
            else:
                # Ref: heavy_simulate.m:165 — ht(:,t-j) → Python: ht[:, t_idx-j-1]
                ht[:, t_idx] += B[:, :, j] @ ht[:, t_idx - j - 1]

        # Ref: heavy_simulate.m:167 — data(:,t) = e(:,t).*ht(:,t)
        data[:, t_idx] = e[:, t_idx] * ht[:, t_idx]

    # ------------------------------------------------------------------
    # Phase 7: Post-processing
    # Ref: heavy_simulate.m:170-175
    # ------------------------------------------------------------------
    # Apply sign correction and square-root transform for return series
    # Ref: heavy_simulate.m:170 — data(m==1,:) = signs(m==1,:).*sqrt(data(m==1,:))
    # signs is only populated when random variables were generated internally
    if signs is not None:
        m_mask = m == 1
        if np.any(m_mask):
            data[m_mask, :] = signs[m_mask, :] * np.sqrt(
                np.abs(data[m_mask, :])
            )

    # Transpose from K×tau (internal) to tau×K (external) layout
    # Ref: heavy_simulate.m:171-172 — ht=ht'; data=data';
    ht = ht.T
    data = data.T

    # Trim burn-in period to return only the last T observations
    # Ref: heavy_simulate.m:174-175 — data = data(tau-T+1:tau,:)
    # MATLAB 1-based (tau-T+1:tau) → Python 0-based [tau-T : tau]
    data = data[tau - T : tau, :]
    ht = ht[tau - T : tau, :]

    return data, ht
