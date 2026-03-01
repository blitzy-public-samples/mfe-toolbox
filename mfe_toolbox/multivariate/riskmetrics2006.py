"""
RiskMetrics 2006 multi-frequency EWMA covariance estimator.

Computes the RiskMetrics 2006 covariance, which is a weighted average of
EWMA covariances at multiple frequencies (half-lives). This is the
multi-frequency extension of the original RiskMetrics 1994 methodology.

The conditional variance H(t) of a RM2006 covariance model is:

    H(t) = sum_i w(i) * Htilde(t, i)

where Htilde(t, i) is an EWMA covariance with half-life tau_i, and

    w(i) = 1 - log(tau_i) / log(tau0)

where the weights have been normalized so that they sum to 1. All EWMAs
are initialized using a backward EWMA with the same decay.

Migrated from: multivariate/riskmetrics2006.m
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3, Date: 03/10/2011
"""

import numpy as np


def riskmetrics2006(data, tau0=None, tau1=None, kmax=None, rho=None):
    """
    Computes the RiskMetrics 2006 covariance, which is a weighted average of
    EWMA covariances at multiple frequencies.

    Parameters
    ----------
    data : numpy.ndarray
        A T-by-K matrix of zero mean residuals, or a K-by-K-by-T array of
        covariance estimators (e.g., realized covariance).
    tau0 : float or None, optional
        Half-life of the slowest EWMA. Default is 1560.
    tau1 : float or None, optional
        Half-life of the fastest EWMA. Default is 4.
    kmax : int or None, optional
        Number of EWMA components to use. Default is 14.
    rho : float or None, optional
        Decay factor to use in half-lives. The half-lives used are
        TAU1, TAU1*RHO, TAU1*RHO^2, ... Default is sqrt(2).

    Returns
    -------
    Ht : numpy.ndarray
        A (K, K, T) array of conditional covariance matrices.
    weights : numpy.ndarray
        A (T, T) matrix containing the final weights used for computing all
        conditional covariances. weights[i, j] contains the weight of the
        outer-product at time j in the covariance forecast at period i+1.

    Raises
    ------
    ValueError
        If any parameter constraints are violated, or if data has an
        unsupported number of dimensions.

    Notes
    -----
    The model decomposes the covariance into K_max EWMA components with
    geometrically spaced half-lives:

        tau_k = TAU1 * RHO^(k-1),  k = 1, ..., KMAX

    Each component is weighted by:

        w_k = 1 - log(tau_k) / log(TAU0)

    normalized so that sum(w_k) = 1.

    Each EWMA component uses backward EWMA backcasting to initialize
    the recursion, ensuring stable estimation from the first observation.

    Examples
    --------
    RiskMetrics 2006 methodology using the suggested reference values:

    >>> import numpy as np
    >>> data = np.random.randn(500, 3)
    >>> tau0 = 1560
    >>> tau1 = 4
    >>> taumax = 512
    >>> rho_val = np.sqrt(2)
    >>> kmax = round(np.log(taumax / tau1) / np.log(rho_val))  # 14
    >>> Ht, wts = riskmetrics2006(data, tau0, tau1, kmax, rho_val)

    RiskMetrics 1994 methodology using .94 as a special case of RM2006:

    >>> tau0 = 1560         # Does not matter
    >>> rho_val = 1         # Does not matter
    >>> tau1 = -1 / np.log(0.94)
    >>> kmax = 1
    >>> Ht, wts = riskmetrics2006(data, tau0, tau1, kmax, rho_val)
    """
    # ---------------------------------------------------------------
    # Ref: riskmetrics2006.m:60-92 — Handle default parameters
    # MATLAB uses nargin switch and isempty checks; Python uses None defaults
    # ---------------------------------------------------------------
    if tau0 is None:
        tau0 = 1560
    if tau1 is None:
        tau1 = 4
    if kmax is None:
        kmax = 14
    if rho is None:
        rho = np.sqrt(2)

    # ---------------------------------------------------------------
    # Ref: riskmetrics2006.m:94-108 — Input validation
    # Preserve exact MATLAB validation order and error messages
    # ---------------------------------------------------------------
    # Ref: riskmetrics2006.m:94-96
    if tau1 * rho ** (kmax - 1) > tau0:
        raise ValueError('The inputs must satisfy: TAU1*RHO^(KMAX-1)<TAU0')
    # Ref: riskmetrics2006.m:97-98
    if tau1 < 0:
        raise ValueError('TAU1 must be positive')
    # Ref: riskmetrics2006.m:100-101
    if tau0 < 0:
        raise ValueError('TAU0 must be positive')
    # Ref: riskmetrics2006.m:103-105
    if kmax < 1 or np.floor(kmax) != kmax:
        raise ValueError('KMAX must be an integer (weakly) larger than 1.')
    # Ref: riskmetrics2006.m:106-108
    if rho < 0:
        raise ValueError('RHO must be positive')

    # Ensure kmax is a Python int for range() and indexing
    kmax = int(kmax)

    # ---------------------------------------------------------------
    # Ref: riskmetrics2006.m:110-120 — Data preprocessing
    # ---------------------------------------------------------------
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 2:
        # Interpret as T×K matrix of zero-mean returns; compute outer products
        T, K = data.shape
        temp = np.zeros((K, K, T))
        for t in range(T):
            # Ref: riskmetrics2006.m:114 — data(t,:)'*data(t,:) outer product
            # MATLAB 1-indexed t=1:T → Python 0-indexed t in range(T)
            temp[:, :, t] = np.outer(data[t, :], data[t, :])
        data = temp
    elif data.ndim == 3:
        # Interpret as K×K×T covariance array
        K = data.shape[0]
        T = data.shape[2]
    else:
        raise ValueError('DATA must be a 2-D (T x K) or 3-D (K x K x T) array.')

    # ---------------------------------------------------------------
    # Ref: riskmetrics2006.m:124-128 — Compute half-lives and weights
    # ---------------------------------------------------------------
    # Ref: riskmetrics2006.m:124 — MATLAB: tauks = tau1*rho.^((1:kmax)-1)
    # (1:kmax)-1 produces [0, 1, ..., kmax-1] → np.arange(kmax) is equivalent
    tauks = tau1 * rho ** np.arange(kmax)

    # Ref: riskmetrics2006.m:125 — Unnormalized component weights
    w = 1.0 - np.log(tauks) / np.log(tau0)

    # Ref: riskmetrics2006.m:126 — Normalize weights to sum to 1
    w = w / np.sum(w)

    # Ref: riskmetrics2006.m:127-128 — Initialize output and temporary arrays
    Ht = np.zeros((K, K, T))
    Httilde = np.zeros((K, K, T))

    # ---------------------------------------------------------------
    # Ref: riskmetrics2006.m:131-148 — Multi-frequency EWMA recursion
    # MATLAB: for k=1:kmax (1-indexed) → Python: for k in range(kmax) (0-indexed)
    # ---------------------------------------------------------------
    for k in range(kmax):
        tauk = tauks[k]
        mu = np.exp(-1.0 / tauk)

        # -----------------------------------------------------------
        # Ref: riskmetrics2006.m:135 — Backcasting endpoint computation
        # MATLAB: endPoint = max(min(floor(log(.01)/log(mu)),T),k)
        # CRITICAL: MATLAB k is 1-indexed in loop for k=1:kmax;
        # Python k is 0-indexed. MATLAB max(...,k) where k starts at 1
        # → Python max(..., k+1) to maintain identical minimum endpoint
        # -----------------------------------------------------------
        end_point = max(min(int(np.floor(np.log(0.01) / np.log(mu))), T), k + 1)

        # Ref: riskmetrics2006.m:136-137 — Backward EWMA weights for backcast
        # MATLAB: weights = (1-mu).*mu.^(0:endPoint-1)
        bw = (1.0 - mu) * mu ** np.arange(end_point)
        bw = bw / np.sum(bw)

        # Ref: riskmetrics2006.m:138-141 — Compute backcast as weighted sum
        # MATLAB: zeros(K) creates K×K zeros matrix
        back_cast = np.zeros((K, K))
        for i in range(end_point):
            # Ref: riskmetrics2006.m:140 — MATLAB i=1:endPoint (1-indexed)
            # → Python i in range(endPoint) (0-indexed); data indexing matches
            back_cast += bw[i] * data[:, :, i]

        # Ref: riskmetrics2006.m:142 — Httilde(:,:,1) = backCast
        # MATLAB index 1 → Python index 0
        Httilde[:, :, 0] = back_cast

        # Ref: riskmetrics2006.m:144-146 — EWMA recursion
        # MATLAB: for t=2:T → Python: for t in range(1, T)
        # H_tilde(t) = mu * H_tilde(t-1) + (1-mu) * data(t-1)
        for t in range(1, T):
            Httilde[:, :, t] = mu * Httilde[:, :, t - 1] + (1.0 - mu) * data[:, :, t - 1]

        # Ref: riskmetrics2006.m:147 — Accumulate weighted EWMA component
        Ht += w[k] * Httilde

    # ---------------------------------------------------------------
    # Ref: riskmetrics2006.m:151-166 — Compute weight matrix
    # In MATLAB this is conditional on nargout>1; in Python we always
    # return it as part of the (Ht, weights) tuple.
    # ---------------------------------------------------------------
    weights = np.zeros((T, T))

    for k in range(kmax):
        tauk = tauks[k]
        mu = np.exp(-1.0 / tauk)

        # Ref: riskmetrics2006.m:156 — Initialize weight matrix
        # MATLAB: weightMatrix = [mu.^(0:T-1)' zeros(T,T-1)]
        # Creates T×T with first column = [1, mu, mu^2, ..., mu^(T-1)]
        weight_matrix = np.zeros((T, T))
        weight_matrix[:, 0] = mu ** np.arange(T)

        # Ref: riskmetrics2006.m:157 — Set last-row, last-column element
        # MATLAB: weightMatrix(T,T) = (1-mu) → Python: [T-1, T-1]
        weight_matrix[T - 1, T - 1] = 1.0 - mu

        # Ref: riskmetrics2006.m:158-160 — Fill last row right to left
        # MATLAB: for j=1:(T-2); weightMatrix(T,T-j) = mu * weightMatrix(T,T-j+1)
        # Each position gets mu times the position to its right
        # Python: MATLAB column T-j (1-indexed) → Python column T-1-j (0-indexed)
        #         MATLAB column T-j+1 (1-indexed) → Python column T-j (0-indexed)
        for j in range(1, T - 1):
            weight_matrix[T - 1, T - 1 - j] = mu * weight_matrix[T - 1, T - j]

        # Ref: riskmetrics2006.m:161-163 — Copy last row pattern to other rows
        # MATLAB: for t=2:T-1; weightMatrix(t,2:t) = weightMatrix(T,T+(-t+2:0))
        # This copies the tail of the last row's EWMA weight pattern into
        # each earlier row, giving the correct EWMA weights for that timestep.
        # Python: MATLAB t (1-indexed) maps to t_py = t-1 (0-indexed)
        #   Left:  MATLAB (t, 2:t) → Python [t_py, 1:t_py+1]
        #   Right: MATLAB (T, T-t+2:T) → Python [T-1, T-t_py:T]
        #   (using t_py as the Python loop variable and t_mat = t_py + 1)
        for t in range(1, T - 1):
            weight_matrix[t, 1:t + 1] = weight_matrix[T - 1, T - t:T]

        # Ref: riskmetrics2006.m:164 — Accumulate weighted component matrix
        weights += w[k] * weight_matrix

    return Ht, weights
