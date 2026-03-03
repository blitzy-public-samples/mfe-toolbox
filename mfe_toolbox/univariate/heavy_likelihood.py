"""
HEAVY model log-likelihood function.

Computes the recursive conditional variance and Gaussian log-likelihood for
the multivariate HEAVY (High-frEquency-bAsed VolatilitY) model of Shephard
and Sheppard.  The variance dynamics follow:

    h(t) = O + sum_{j=1}^{pMax} A_j * data(t-j) + sum_{j=1}^{qMax} B_j * h(t-j)

where h(t) is the K-dimensional conditional variance vector at time t,
O is the intercept vector, and A_j / B_j are K x K coefficient matrices
for innovation and smoothing lags respectively.

Volatility bounds are enforced through smooth barrier transformations to
keep the optimiser in a feasible region without hard constraints.

Migrated from univariate/heavy_likelihood.m (Version 4.0, 28-Oct-2009).
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk).
Revision: 1    Date: 4/2/2012
"""

import numpy as np

from mfe_toolbox.univariate.heavy_parameter_transform import heavy_parameter_transform


def heavy_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    back_cast: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute HEAVY model log-likelihood with recursive conditional variances.

    Evaluates the Gaussian log-likelihood for the multivariate HEAVY model
    given a parameter vector, data matrix, and model specification arrays.
    The conditional variance for each series is computed recursively using
    matrix-valued lag polynomials on past innovations and past variances.

    Parameters
    ----------
    parameters : np.ndarray
        A 1-D array with ``K + sum(sum(p)) + sum(sum(q))`` elements containing
        all HEAVY model parameters in canonical ordering:
        ``[O(1),...,O(K), A(1,1,1:p[0,0]),...,A(K,K,1:p[K-1,K-1]),
        B(1,1,1:q[0,0]),...,B(K,K,1:q[K-1,K-1])]``.
    data : np.ndarray
        A K x T matrix of non-negative data (typically squared returns and/or
        realized measures).  Columns represent time observations; the caller
        is responsible for transposing from the T x K user convention before
        calling this function.
    p : np.ndarray
        A K x K integer matrix where element ``(i, j)`` gives the number of
        lags of series *j* innovations in the variance equation for series *i*.
    q : np.ndarray
        A K x K integer matrix where element ``(i, j)`` gives the number of
        lags of series *j* conditional variance in the equation for series *i*.
    back_cast : np.ndarray
        A (K,) vector of values used for back-casting when lagged values are
        not yet available (initial conditions for the recursion).
    lb : np.ndarray
        A (K,) vector of volatility lower bounds.  When a conditional variance
        falls below the corresponding lower bound, a smooth barrier
        transformation maps it back into the feasible region.
    ub : np.ndarray
        A (K,) vector of volatility upper bounds.  When a conditional variance
        exceeds the corresponding upper bound, a logarithmic barrier
        transformation dampens the value.

    Returns
    -------
    ll : float
        Total log-likelihood (sum of per-observation log-likelihoods).
        Set to the penalty value ``1e7`` if the computed value is NaN,
        infinite, or complex.
    lls : np.ndarray
        A (T,) vector of individual observation log-likelihoods.
    h : np.ndarray
        A T x K matrix of conditional variances.  Internally computed as
        K x T and transposed on output for user convenience.

    Notes
    -----
    This function computes the *negative* log-likelihood (without the
    leading minus sign) — i.e., ``lls[t] = 0.5 * (K*log(2*pi) +
    sum(log(h[:,t])) + sum(data[:,t] / h[:,t]))``.  The optimiser
    *minimises* ``ll = sum(lls)`` to find maximum-likelihood estimates.

    Volatility bound transformations (smooth barriers):

    * **Lower bound** (``h < lb``):
      ``h_new = lb / (1 - (h - lb))``
      This maps values below ``lb`` to ``(0, lb)`` smoothly.

    * **Upper bound** (``h > ub``):
      ``h_new = ub + log(h - ub)``
      This maps values above ``ub`` via a logarithmic dampening.

    See Also
    --------
    mfe_toolbox.univariate.heavy : HEAVY model estimation driver.
    mfe_toolbox.univariate.heavy_parameter_transform : Parameter extraction.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.heavy_likelihood import heavy_likelihood
    >>> rng = np.random.default_rng(42)
    >>> K, T = 2, 100
    >>> data = rng.random((K, T)) * 0.01  # Simulated squared returns
    >>> params = np.array([0.001, 0.002, 0.3, 0.4, 0.5, 0.6])
    >>> p_mat = np.array([[1, 0], [0, 1]])
    >>> q_mat = np.array([[1, 0], [0, 1]])
    >>> bc = np.mean(data, axis=1)
    >>> lb_vec = np.array([1e-12, 1e-12])
    >>> ub_vec = np.array([1e4, 1e4])
    >>> ll, lls, h = heavy_likelihood(params, data, p_mat, q_mat, bc, lb_vec, ub_vec)
    >>> isinstance(ll, float) and lls.shape == (T,) and h.shape == (T, K)
    True
    """
    # ------------------------------------------------------------------
    # Phase 1: Setup — extract dimensions and pre-allocate arrays
    # Ref: heavy_likelihood.m:33-37
    # ------------------------------------------------------------------
    # Ref: heavy_likelihood.m:33 — pMax = max(max(p))
    p_max: int = int(np.max(p))
    # Ref: heavy_likelihood.m:34 — qMax = max(max(q))
    q_max: int = int(np.max(q))

    # Ref: heavy_likelihood.m:35 — [K,T] = size(data); data is K×T
    K: int = data.shape[0]
    T: int = data.shape[1]

    # Ref: heavy_likelihood.m:36 — lls = zeros(T,1)
    lls: np.ndarray = np.zeros(T, dtype=np.float64)
    # Ref: heavy_likelihood.m:37 — h = zeros(K,T)
    h: np.ndarray = np.zeros((K, T), dtype=np.float64)

    # ------------------------------------------------------------------
    # Phase 2: Parameter transformation
    # Ref: heavy_likelihood.m:39-40
    # ------------------------------------------------------------------
    # Ref: heavy_likelihood.m:39 — [O,A,B] = heavy_parameter_transform(parameters,p,q,K)
    O, A, B = heavy_parameter_transform(parameters, p, q, K)

    # Ref: heavy_likelihood.m:40 — MATLAB realmin() → np.finfo(float).tiny
    # Clamp negative intercepts to the smallest positive normalised float
    # to ensure positivity during the variance recursion.
    O[O < 0] = np.finfo(float).tiny

    # Ref: heavy_likelihood.m:42 — likConst = K*log(2*pi)
    lik_const: float = float(K) * np.log(2.0 * np.pi)

    # Ensure back_cast is a contiguous 1-D float64 array of length K
    back_cast_vec: np.ndarray = np.asarray(back_cast, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Phase 3: Recursive conditional variance computation
    # Ref: heavy_likelihood.m:43-67
    # MATLAB loop t=1:T (1-based) → Python t=0:T-1 (0-based)
    # MATLAB (t-j)>0 guard (1-based) → Python t-j >= 0 (0-based)
    # ------------------------------------------------------------------
    for t in range(T):
        # Ref: heavy_likelihood.m:44 — h(:,t) = O
        # Initialise current column with intercept vector
        h[:, t] = O

        # Ref: heavy_likelihood.m:45-51 — Innovation terms with back-cast fallback; 0-indexed
        # MATLAB: for j=1:pMax → Python: for j in range(1, p_max+1)
        # A array 3rd dimension: MATLAB A(:,:,j) → Python A[:,:,j-1]
        for j in range(1, p_max + 1):
            if t - j >= 0:
                # Ref: heavy_likelihood.m:47 — h(:,t) = h(:,t) + A(:,:,j)*data(:,t-j)
                # MATLAB 1-based index t-j maps to Python 0-based t-j since Python t starts at 0
                h[:, t] += A[:, :, j - 1] @ data[:, t - j]
            else:
                # Ref: heavy_likelihood.m:49 — h(:,t) = h(:,t) + A(:,:,j)*backCast'
                # backCast' in MATLAB transposes 1×K row to K×1 column;
                # in Python, 1-D (K,) array works directly with @ operator
                h[:, t] += A[:, :, j - 1] @ back_cast_vec

        # Ref: heavy_likelihood.m:52-58 — Smoothing terms with back-cast fallback; 0-indexed
        # MATLAB: for j=1:qMax → Python: for j in range(1, q_max+1)
        # B array 3rd dimension: MATLAB B(:,:,j) → Python B[:,:,j-1]
        for j in range(1, q_max + 1):
            if t - j >= 0:
                # Ref: heavy_likelihood.m:54 — h(:,t) = h(:,t) + B(:,:,j)*h(:,t-j)
                h[:, t] += B[:, :, j - 1] @ h[:, t - j]
            else:
                # Ref: heavy_likelihood.m:56 — h(:,t) = h(:,t) + B(:,:,j)*backCast'
                h[:, t] += B[:, :, j - 1] @ back_cast_vec

        # Ref: heavy_likelihood.m:59-65 — Volatility bounds clamping
        # MATLAB for j=1:K → Python for k in range(K); 0-indexed
        for k in range(K):
            if h[k, t] < lb[k]:
                # Ref: heavy_likelihood.m:61 — lb(j) * 1./(1-(h(j,t)-lb(j)))
                # Smooth lower barrier: maps h below lb into (0, lb)
                h[k, t] = lb[k] / (1.0 - (h[k, t] - lb[k]))
            elif h[k, t] > ub[k]:
                # Ref: heavy_likelihood.m:63 — ub(j) + log(h(j,t)-ub(j))
                # Logarithmic upper barrier: dampens h above ub
                h[k, t] = ub[k] + np.log(h[k, t] - ub[k])

        # Ref: heavy_likelihood.m:66 — lls(t) = 0.5*(likConst + sum(log(h(:,t))) + sum(data(:,t)./h(:,t)))
        # Per-observation Gaussian negative-log-likelihood (without sign flip)
        lls[t] = 0.5 * (
            lik_const
            + np.sum(np.log(h[:, t]))
            + np.sum(data[:, t] / h[:, t])
        )

    # ------------------------------------------------------------------
    # Phase 4: Aggregate and apply degenerate-value guard
    # Ref: heavy_likelihood.m:68-75
    # ------------------------------------------------------------------
    # Ref: heavy_likelihood.m:68 — ll = sum(lls)
    ll_raw = np.sum(lls)

    # Ref: heavy_likelihood.m:70-72 — Guard against NaN / Inf / complex
    # In Python, np.log of a negative number returns NaN (not complex as in
    # MATLAB), so np.isnan is the primary guard.  The np.isreal check is
    # retained for exact MATLAB parity in case upstream changes introduce
    # complex arithmetic.
    if np.isnan(ll_raw) or np.isinf(ll_raw) or not np.isreal(ll_raw):
        ll: float = 1e7
    else:
        ll = float(ll_raw)

    # Ref: heavy_likelihood.m:74-75 — h = h' (transpose K×T → T×K for output)
    # In MATLAB this was conditional on nargout>2; in Python all three
    # return values are always produced.
    h = h.T

    return ll, lls, h
