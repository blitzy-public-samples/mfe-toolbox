"""
Log-likelihood evaluation for the Scalar Variance-Targeting VECH (VT-VECH)
multivariate GARCH model.

Evaluates the Gaussian (or composite) negative log-likelihood for a Scalar
VT-VECH(P,O,Q) process with variance targeting. The conditional covariance
dynamics follow:

    H(t) = (1 - sum(alpha) - sum(beta)) * C  -  sum(gamma) * Casym
           + sum_j alpha(j) * r(t-j)'r(t-j)
           + sum_j gamma(j) * n(t-j)'n(t-j)
           + sum_j beta(j)  * H(t-j)

where:
    C     = unconditional covariance (variance-targeting intercept)
    Casym = unconditional expectation of the asymmetric outer product
    r(t)  = return vector at time t
    n(t)  = r(t) .* (r(t) < 0)  (negative-only returns for asymmetry)

Three operating modes are supported:

1. **Estimation mode** (``estim_flag=True``): Parameters are in unconstrained
   optimizer space and are transformed to the constrained GARCH space via
   :func:`scalar_vt_vech_itransform` before evaluation.
2. **Joint mode** (``is_joint=True``, ``estim_flag=False``): Parameters include
   vech(C) (and optionally vech(Casym) if o > 0) prepended to the dynamics
   vector [alpha, gamma, beta].  Used for joint Hessian/score computation.
3. **Standard evaluation** (both ``False``): Parameters are already in
   constrained space.

Notes
-----
Migrated from ``multivariate/scalar_vt_vech_likelihood.m`` (117 lines).

Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 4    Date: 10/28/2009

References
----------
Kevin Sheppard, MFE Toolbox Version 4.0 (2009).
See also: SCALAR_VT_VECH, SCALAR_VT_VECH_ITRANSFORM, COMPOSITE_LIKELIHOOD
"""

import numpy as np

from mfe_toolbox.multivariate.scalar_vt_vech_itransform import scalar_vt_vech_itransform
from mfe_toolbox.utility.ivech import ivech
from mfe_toolbox.distributions.composite_likelihood import composite_likelihood


def scalar_vt_vech_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    data_asym: np.ndarray,
    p: int,
    o: int,
    q: int,
    c: np.ndarray,
    c_asym: np.ndarray,
    kappa: float,
    back_cast: np.ndarray,
    back_cast_asym: np.ndarray,
    is_joint: bool,
    use_composite: int,
    estim_flag: bool = False,
) -> tuple:
    """
    Evaluate the negative log-likelihood of a Scalar VT-VECH(P,O,Q) model.

    Parameters
    ----------
    parameters : np.ndarray
        Parameter vector.  Its interpretation depends on the operating mode:

        - **Estimation mode** (``estim_flag=True``): Unconstrained parameters
          of length ``p + o + q``, mapped to constrained space by
          :func:`scalar_vt_vech_itransform`.
        - **Joint mode** (``is_joint=True``, ``estim_flag=False``):
          ``[vech(C); (vech(Casym) if o > 0); alpha; gamma; beta]``.
        - **Standard mode** (both ``False``):
          ``[alpha(1), ..., alpha(p), gamma(1), ..., gamma(o),
          beta(1), ..., beta(q)]``.

    data : np.ndarray, shape (K, K, T)
        K × K × T array of (symmetric) covariance innovations.  Typically
        outer products of return vectors: ``data[:, :, t] = r_t @ r_t.T``.
    data_asym : np.ndarray, shape (K, K, T)
        K × K × T array of asymmetric covariance innovations.  Typically
        ``data_asym[:, :, t] = n_t @ n_t.T`` where ``n_t = r_t * (r_t < 0)``.
    p : int
        Positive integer — number of symmetric innovation lags.
    o : int
        Non-negative integer — number of asymmetric innovation lags.
    q : int
        Non-negative integer — number of lagged conditional covariance terms.
    c : np.ndarray, shape (K, K)
        Unconditional covariance matrix (variance-targeting intercept).
        Overridden if ``is_joint`` is True (extracted from ``parameters``).
    c_asym : np.ndarray, shape (K, K)
        Unconditional expectation of the asymmetric covariance matrix.
        Overridden if ``is_joint`` is True and ``o > 0``.
    kappa : float
        Positive scalar — asymmetry scaling factor from the eigenvalue
        structure of the unconditional covariance.  Passed through to
        :func:`scalar_vt_vech_itransform` when ``estim_flag`` is True.
    back_cast : np.ndarray, shape (K, K)
        Back-cast value for initialising the GARCH recursion when
        lagged values are unavailable (i.e. for early time steps).
    back_cast_asym : np.ndarray, shape (K, K)
        Back-cast value for the asymmetric recursion terms.
    is_joint : bool
        If True, ``parameters`` includes the intercept (vech(C) and
        optionally vech(Casym)) prepended to the dynamics vector.
    use_composite : int
        Likelihood type selector:

        - ``0``: Standard K-dimensional multivariate normal log-likelihood.
        - ``1``: Diagonal composite likelihood (adjacent pairs only).
        - ``2``: Full composite likelihood (all unique pairs).

    estim_flag : bool, optional
        If True, ``parameters`` are in unconstrained optimizer space and
        will be transformed via :func:`scalar_vt_vech_itransform`.
        Default is False.

    Returns
    -------
    ll : float
        Negative total log-likelihood (summed over all time steps).
    lls : np.ndarray, shape (T,)
        Per-period negative log-likelihoods.
    Ht : np.ndarray, shape (K, K, T)
        3-D array of conditional covariance matrices.

    Notes
    -----
    - **Indexing**: MATLAB uses 1-based indexing; this Python port uses
      0-based indexing throughout.  Lag ``j`` (0-based) corresponds to
      MATLAB's lag ``j + 1``.
    - **Numerical stability**: The standard likelihood decomposes Ht into
      a correlation matrix R and standard-deviation vector Q, then uses
      ``np.linalg.slogdet`` for the log-determinant and ``np.linalg.solve``
      for the inverse-quadratic form.  This avoids direct calls to ``det``
      and ``inv``, matching the MATLAB trick for ill-conditioned covariance
      matrices (Ref: scalar_vt_vech_likelihood.m:106-108).
    - **Composite indices**: For ``use_composite == 1``, adjacent pairs
      ``(i, i+1)`` are used.  For ``use_composite == 2``, all strictly
      lower-triangular pairs are used.  All indices are **0-based** as
      required by the Python ``composite_likelihood`` function.

    See Also
    --------
    mfe_toolbox.multivariate.scalar_vt_vech : Driver function.
    mfe_toolbox.multivariate.scalar_vt_vech_itransform : Inverse transform.
    mfe_toolbox.distributions.composite_likelihood : Composite likelihood.

    Examples
    --------
    >>> import numpy as np
    >>> k, T = 2, 100
    >>> data = np.random.randn(k, k, T)
    >>> data_asym = np.zeros((k, k, T))
    >>> c = np.eye(k) * 0.01
    >>> params = np.array([0.05, 0.90])
    >>> ll, lls, Ht = scalar_vt_vech_likelihood(
    ...     params, data, data_asym, 1, 0, 1,
    ...     c, np.zeros((k, k)), 2.0,
    ...     c.copy(), np.zeros((k, k)),
    ...     False, 0, False
    ... )
    """
    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_likelihood.m:41 — Extract dimensions
    # data is K × K × T (3-D array of outer products)
    # ------------------------------------------------------------------
    k = data.shape[0]
    T = data.shape[2]

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_likelihood.m:43-57 — Parameter handling
    # Three modes: estimation (itransform), joint (extract C/Casym), standard
    # ------------------------------------------------------------------
    # Ensure 1-D float64 copy to avoid mutating caller's data
    parameters = np.asarray(parameters, dtype=np.float64).ravel().copy()

    base = 0  # Ref: scalar_vt_vech_likelihood.m:43

    if estim_flag:
        # Ref: scalar_vt_vech_likelihood.m:44-45 — Transform unconstrained
        # parameters to constrained GARCH space via itransform
        parameters = scalar_vt_vech_itransform(parameters, p, o, q, kappa)
    elif is_joint:
        # Ref: scalar_vt_vech_likelihood.m:47-56 — Joint mode: intercept
        # is embedded in the parameter vector as vech(C), optionally vech(Casym)
        k2 = k * (k + 1) // 2
        base = k2
        # Ref: scalar_vt_vech_likelihood.m:50-51 — Reconstruct C from vech
        c = ivech(parameters[:k2])
        if o > 0:
            # Ref: scalar_vt_vech_likelihood.m:52-55 — Reconstruct Casym
            c_asym = ivech(parameters[k2:2 * k2])
            base = base + k2

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_likelihood.m:58-60 — Extract alpha, gamma, beta
    # MATLAB: alpha = parameters(base+(1:p))  (1-indexed)
    # Python: alpha = parameters[base:base+p]  (0-indexed)
    # ------------------------------------------------------------------
    alpha = parameters[base:base + p]
    gamma = parameters[base + p:base + p + o]
    beta = parameters[base + p + o:base + p + o + q]

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_likelihood.m:62-65 — Compute the constant
    # const = (1 - sum(alpha) - sum(beta)) * C  (variance targeting)
    # If o > 0: const = const - sum(gamma) * Casym
    # ------------------------------------------------------------------
    const = (1.0 - np.sum(alpha) - np.sum(beta)) * c
    if o > 0:
        # Ref: scalar_vt_vech_likelihood.m:64
        const = const - np.sum(gamma) * c_asym

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_likelihood.m:68 — Initialize Ht with backcast
    # MATLAB: Ht = repmat(backCast, [1 1 T])
    # ------------------------------------------------------------------
    Ht = np.zeros((k, k, T))
    for t_init in range(T):
        Ht[:, :, t_init] = back_cast

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_likelihood.m:71 — Initialize log-likelihoods
    # ------------------------------------------------------------------
    lls = np.zeros(T)

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_likelihood.m:74-81 — Prepare likelihood constants
    # and composite index pairs (0-based for Python composite_likelihood)
    # ------------------------------------------------------------------
    if use_composite == 0:
        # Ref: scalar_vt_vech_likelihood.m:75 — Standard MV normal constant
        likconst = k * np.log(2.0 * np.pi)
    elif use_composite == 1:
        # Ref: scalar_vt_vech_likelihood.m:77 — Diagonal composite:
        # MATLAB 1-based: [(1:k-1)' (2:k)'] → Python 0-based adjacent pairs
        indices = np.column_stack([
            np.arange(k - 1), np.arange(1, k)
        ]).astype(np.int64)
    elif use_composite == 2:
        # Ref: scalar_vt_vech_likelihood.m:79-80 — Full composite:
        # MATLAB: [i,j] = meshgrid(1:k); indices = [i(~triu(true(k))) j(~triu(true(k)))]
        # Python: all strictly lower-triangular pairs (0-based)
        # np.eye(k, dtype=bool) identifies the diagonal; np.tril minus diagonal
        # yields strictly lower-triangular mask matching MATLAB's ~triu(true(k))
        eye_k = np.eye(k, dtype=bool)
        lower_tri_mask = np.tril(np.ones((k, k), dtype=bool)) & ~eye_k
        i_grid, j_grid = np.meshgrid(np.arange(k), np.arange(k))
        indices = np.column_stack([
            i_grid[lower_tri_mask], j_grid[lower_tri_mask]
        ]).astype(np.int64)

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_likelihood.m:83-117 — Main GARCH recursion and
    # log-likelihood evaluation
    # ------------------------------------------------------------------
    for t in range(T):
        # Ref: scalar_vt_vech_likelihood.m:84 — Start from the constant
        Ht[:, :, t] = const.copy()

        # Ref: scalar_vt_vech_likelihood.m:85-91 — Alpha (symmetric) terms
        # MATLAB loop j=1:p with lag j; Python j=0..p-1 with lag j+1
        for j in range(p):
            lag = j + 1  # Ref: MATLAB j is 1-based, representing lag order
            if t - lag < 0:
                # Ref: scalar_vt_vech_likelihood.m:87 — Use backcast for
                # unavailable lags (t - lag < 0 ↔ MATLAB (t-j) < 1)
                Ht[:, :, t] += alpha[j] * back_cast
            else:
                # Ref: scalar_vt_vech_likelihood.m:89
                Ht[:, :, t] += alpha[j] * data[:, :, t - lag]

        # Ref: scalar_vt_vech_likelihood.m:92-98 — Gamma (asymmetric) terms
        for j in range(o):
            lag = j + 1
            if t - lag < 0:
                # Ref: scalar_vt_vech_likelihood.m:94
                Ht[:, :, t] += gamma[j] * back_cast_asym
            else:
                # Ref: scalar_vt_vech_likelihood.m:96
                Ht[:, :, t] += gamma[j] * data_asym[:, :, t - lag]

        # Ref: scalar_vt_vech_likelihood.m:99-105 — Beta (lagged Ht) terms
        for j in range(q):
            lag = j + 1
            if t - lag < 0:
                # Ref: scalar_vt_vech_likelihood.m:101
                Ht[:, :, t] += beta[j] * back_cast
            else:
                # Ref: scalar_vt_vech_likelihood.m:103
                Ht[:, :, t] += beta[j] * Ht[:, :, t - lag]

        # ---------------------------------------------------------------
        # Log-likelihood evaluation for time t
        # ---------------------------------------------------------------
        if use_composite == 0:
            # Ref: scalar_vt_vech_likelihood.m:106-113
            # Decompose Ht into standard deviations Q and correlation R
            # for better numerical stability with ill-conditioned matrices.
            #
            # MATLAB original (replaced for stability):
            #   likconst + log(det(Ht)) + data'*Ht^{-1}*data
            #
            # Numerically stable form:
            #   likconst + 2*sum(log(Q)) + log(det(R)) + trace(R^{-1}*stdresid)

            # Ref: scalar_vt_vech_likelihood.m:110 — Q = sqrt(diag(Ht(:,:,t)))
            Q = np.sqrt(np.diag(Ht[:, :, t]))

            # Ref: scalar_vt_vech_likelihood.m:111 — R = Ht ./ (Q*Q')
            # Compute outer product Q*Q' using matrix multiply with .T
            Q_col = Q.reshape(-1, 1)
            QQ = Q_col @ Q_col.T  # K×K outer product; uses np.ndarray.T

            R = Ht[:, :, t] / QQ

            # Ref: scalar_vt_vech_likelihood.m:112 — stdresid = data ./ (Q*Q')
            stdresid = data[:, :, t] / QQ

            # Ref: scalar_vt_vech_likelihood.m:113
            # Use np.linalg.slogdet for numerically stable log-determinant
            # instead of np.log(np.linalg.det(R))
            sign, logdetR = np.linalg.slogdet(R)

            # Use np.linalg.solve(R, stdresid) instead of inv(R) @ stdresid
            # trace(R^{-1} * stdresid) = trace(solve(R, stdresid))
            solved = np.linalg.solve(R, stdresid)

            lls[t] = 0.5 * (
                likconst
                + 2.0 * np.sum(np.log(Q))
                + logdetR
                + np.trace(solved)
            )
        elif use_composite > 0:
            # Ref: scalar_vt_vech_likelihood.m:114-115 — Composite likelihood
            # composite_likelihood expects 0-based index pairs (handled above)
            lls[t] = composite_likelihood(Ht[:, :, t], data[:, :, t], indices)

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_likelihood.m:118 — Sum individual likelihoods
    # ------------------------------------------------------------------
    ll = np.sum(lls)

    return ll, lls, Ht
