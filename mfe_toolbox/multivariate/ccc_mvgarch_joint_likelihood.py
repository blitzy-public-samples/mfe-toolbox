"""
CCC-MVGARCH joint log-likelihood computation.

Computes the joint log-likelihood for the Constant Conditional Correlation
Multivariate GARCH model. This function is primarily used for computing
the Hessian matrix during inference, where both TARCH variance parameters
and correlation parameters are jointly evaluated.

Migrated from multivariate/ccc_mvgarch_joint_likelihood.m
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 4    Date: 10/28/2009
"""

import numpy as np


def _tarch_core(fdata, fIdata, parameters, back_cast, p, o, q, m, T, tarch_type):
    """
    Compute conditional variance for a TARCH(P,O,Q) process.

    Local implementation of the TARCH core recursion, equivalent to
    univariate/tarch_core.m and mex_source/tarch_core.c.  This is
    inlined here because the schema forbids importing from other
    project modules (depends_on_files is empty).

    Parameters
    ----------
    fdata : numpy.ndarray
        (T,) vector of transformed data (squared returns plus m-length
        backcast padding prepended).
    fIdata : numpy.ndarray
        (T,) vector of asymmetric component data (fdata * indicator(returns<0)
        with m-length backcast padding prepended).
    parameters : numpy.ndarray
        (1+P+O+Q,) parameter vector [omega, alpha(1:p), gamma(1:o), beta(1:q)].
    back_cast : float
        Scalar value used to initialise the first m entries of the
        conditional-variance vector.
    p : int
        Number of symmetric innovation lags.
    o : int
        Number of asymmetric innovation lags.
    q : int
        Number of lagged variance terms.
    m : int
        Number of backcast positions (max(p, o, q)).
    T : int
        Total length of fdata (including prepended backcasts).
    tarch_type : int
        1 for absolute-value model (result is squared at end),
        2 for squared model (no post-squaring).

    Returns
    -------
    ht : numpy.ndarray
        (T,) vector of conditional variances.

    Notes
    -----
    Ref: univariate/tarch_core.m — direct translation of the recursion loop.
    Ref: mex_source/tarch_core.c — C MEX gateway implementing the same loop.
    """
    ht = np.zeros(T, dtype=np.float64)
    # Ref: tarch_core.m:51-54 — initialise backcast entries
    for i in range(m):
        ht[i] = back_cast

    # Ref: tarch_core.m:56-67 — main recursion loop
    for i in range(m, T):
        ht[i] = parameters[0]  # omega
        # Ref: tarch_core.m:59 — symmetric ARCH terms
        for j in range(1, p + 1):
            ht[i] += parameters[j] * fdata[i - j]
        # Ref: tarch_core.m:62 — asymmetric terms
        for j in range(1, o + 1):
            ht[i] += parameters[p + j] * fIdata[i - j]
        # Ref: tarch_core.m:65 — GARCH terms
        for j in range(1, q + 1):
            ht[i] += parameters[p + o + j] * ht[i - j]

    # Ref: tarch_core.m:71-73 — post-square for absolute-value model
    if tarch_type == 1:
        ht = ht ** 2

    return ht


def _corr_ivech(stacked_data):
    """
    Reconstruct a symmetric correlation matrix from its lower-triangular
    off-diagonal elements (inverse vech of correlation).

    Local implementation equivalent to utility/corr_ivech.m.

    Parameters
    ----------
    stacked_data : numpy.ndarray
        K*(K-1)/2 vector of lower-triangular correlation elements
        (excluding the unit diagonal).

    Returns
    -------
    matrix_data : numpy.ndarray
        (K, K) symmetric correlation matrix with ones on the diagonal.

    Notes
    -----
    Ref: utility/corr_ivech.m — complete translation.
    The stacking convention follows MATLAB's column-major lower-triangle
    ordering: elements are filled column-by-column in the strict lower
    triangle, then mirrored across the diagonal, and the identity is added.
    """
    stacked_data = np.asarray(stacked_data, dtype=np.float64).ravel()
    k2 = len(stacked_data)

    # Ref: corr_ivech.m:39 — solve K*(K-1)/2 = k2 for K
    k = int((-1.0 + np.sqrt(1.0 + 8.0 * k2)) / 2.0) + 1

    if k * (k - 1) // 2 != k2:
        raise ValueError(
            "The number of elements in stacked_data is not conformable "
            "to the inverse correlation vech operation."
        )

    # Ref: corr_ivech.m:48-51 — fill lower triangle, mirror, add identity
    matrix_data = np.zeros((k, k), dtype=np.float64)
    idx = 0
    # Ref: corr_ivech.m uses column-major fill of lower triangle
    for col in range(k):
        for row in range(col + 1, k):
            matrix_data[row, col] = stacked_data[idx]
            idx += 1
    matrix_data = matrix_data + matrix_data.T + np.eye(k, dtype=np.float64)

    return matrix_data


def ccc_mvgarch_joint_likelihood(parameters, data, vol_data, p, o, q):
    """
    Compute the joint log-likelihood for the CCC-MVGARCH model.

    This function computes the joint log-likelihood for the Constant
    Conditional Correlation Multivariate GARCH model.  It is primarily
    used for computing the Hessian matrix for inference purposes.  The
    full parameter vector (TARCH parameters for every series **and**
    the off-diagonal correlation parameters) is passed in, so the
    likelihood is jointly differentiable with respect to all of them.

    Migrated from multivariate/ccc_mvgarch_joint_likelihood.m

    Parameters
    ----------
    parameters : numpy.ndarray
        Full parameter vector of the form::

            [tarch(1)', tarch(2)', ..., tarch(k)', corr_vech(R)]

        where each ``tarch(i)`` consists of
        ``[omega(i), alpha(i,1:p(i)), gamma(i,1:o(i)), beta(i,1:q(i))]``
        and ``corr_vech(R)`` is the K*(K-1)/2 lower-triangular off-diagonal
        elements of the constant correlation matrix.
    data : numpy.ndarray
        (K, K, T) 3-D array of positive semi-definite matrices (e.g. outer
        products of standardised residuals or realised covariance matrices).
    vol_data : numpy.ndarray
        (T, K) matrix of data used for estimating the individual TARCH
        volatility models (typically the raw return series).
    p : numpy.ndarray or array_like
        (K,) vector of symmetric innovation lag orders for each series.
    o : numpy.ndarray or array_like
        (K,) vector of asymmetric innovation lag orders for each series.
    q : numpy.ndarray or array_like
        (K,) vector of lagged-variance orders for each series.

    Returns
    -------
    ll : float
        Negative of the total log-likelihood value (negated for
        minimisation conventions).
    lls : numpy.ndarray
        (T,) vector of per-observation negative log-likelihoods.
    ht : numpy.ndarray
        (K, K, T) array of conditional covariance matrices.  Constructed
        as ``ht[:,:,t] = R * outer(sqrt(h_t), sqrt(h_t))``, where ``h_t``
        is the vector of individual conditional variances at time *t*.

    Notes
    -----
    The joint log-likelihood at each observation *t* is

    .. math::

        l_t = \\tfrac{1}{2}\\bigl(K\\ln(2\\pi)
              + \\sum_i \\ln h_{it}
              + \\ln|R|
              + \\operatorname{tr}(R^{-1} S_t)\\bigr)

    where :math:`S_t = \\text{data}_t \\oslash (\\sqrt{h_t} \\sqrt{h_t}^\\top)`
    (element-wise division), *R* is the constant correlation matrix, and
    :math:`h_{it}` are the individual conditional variances from TARCH models.

    The function reconstructs per-series TARCH conditional variances from the
    combined parameter vector, then uses the constant correlation matrix *R*
    to compute the multivariate normal log-likelihood.

    Ref: multivariate/ccc_mvgarch_joint_likelihood.m — complete translation.

    See Also
    --------
    ccc_mvgarch : Main CCC-MVGARCH estimation driver.
    ccc_mvgarch_likelihood : Correlation-only likelihood for score computation.
    """
    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    data = np.asarray(data, dtype=np.float64)
    vol_data = np.asarray(vol_data, dtype=np.float64)
    p = np.asarray(p, dtype=np.int64).ravel()
    o = np.asarray(o, dtype=np.int64).ravel()
    q = np.asarray(q, dtype=np.int64).ravel()

    # Ref: ccc_mvgarch_joint_likelihood.m:32 — extract dimensions
    k = data.shape[0]
    t = data.shape[2]

    # ------------------------------------------------------------------
    # Backcast setup
    # Ref: ccc_mvgarch_joint_likelihood.m:35-37
    # ------------------------------------------------------------------
    # Ref: ccc_mvgarch_joint_likelihood.m:35 — backCastLength = max(floor(t^(1/2)),1)
    # Clamp so that back_cast_length+1 <= t, preventing out-of-bounds
    # on vol_data[:back_cast_length+1, i] when t is very small.
    back_cast_length = max(int(np.floor(np.sqrt(t))), 1)
    if back_cast_length + 1 > t:
        back_cast_length = max(t - 1, 0)
    # Ref: ccc_mvgarch_joint_likelihood.m:36-37 — exponentially decaying weights
    n_bc = back_cast_length + 1
    back_cast_weights = 0.05 * (0.9 ** np.arange(n_bc))
    back_cast_weights = back_cast_weights / np.sum(back_cast_weights)

    # ------------------------------------------------------------------
    # Per-series TARCH variance recursion
    # Ref: ccc_mvgarch_joint_likelihood.m:39-57
    # ------------------------------------------------------------------
    ht_mat = np.zeros((t, k), dtype=np.float64)
    parameter_count = 0  # 0-indexed offset (MATLAB uses 1-indexed)

    for i in range(k):
        epsilon = vol_data[:, i]
        m = max(int(p[i]), int(o[i]), int(q[i]))

        # Ref: ccc_mvgarch_joint_likelihood.m:43 — squared data with backcast
        mean_sq = np.mean(epsilon ** 2)
        fdata = np.concatenate([mean_sq * np.ones(m, dtype=np.float64),
                                epsilon ** 2])

        # Ref: ccc_mvgarch_joint_likelihood.m:45 — asymmetric component
        f_i_data = np.concatenate([
            0.5 * mean_sq * np.ones(m, dtype=np.float64),
            (epsilon ** 2) * (epsilon < 0).astype(np.float64)
        ])

        # Ref: ccc_mvgarch_joint_likelihood.m:46-49 — weighted backcast
        back_cast_val = np.dot(
            back_cast_weights,
            vol_data[:n_bc, i] ** 2
        )
        if back_cast_val == 0.0:
            back_cast_val = mean_sq

        total_len = len(fdata)
        num_params = 1 + int(p[i]) + int(o[i]) + int(q[i])
        tarch_params = parameters[parameter_count:parameter_count + num_params]

        # Ref: ccc_mvgarch_joint_likelihood.m:54 — tarch_core with type=2
        variance = _tarch_core(
            fdata, f_i_data, tarch_params,
            back_cast_val,
            int(p[i]), int(o[i]), int(q[i]),
            m, total_len, 2
        )

        # Ref: ccc_mvgarch_joint_likelihood.m:55 — trim backcast padding
        # MATLAB: htMat(:,i) = variance(m+1:T+m) → Python: variance[m:]
        ht_mat[:, i] = variance[m:total_len]
        parameter_count += num_params

    # ------------------------------------------------------------------
    # Reconstruct constant correlation matrix R
    # Ref: ccc_mvgarch_joint_likelihood.m:59
    # ------------------------------------------------------------------
    num_corr_params = k * (k - 1) // 2
    r_matrix = _corr_ivech(parameters[parameter_count:parameter_count + num_corr_params])

    # ------------------------------------------------------------------
    # Compute R^{-1} and log|R| once (constant across t)
    # Ref: ccc_mvgarch_joint_likelihood.m:61-63
    # ------------------------------------------------------------------
    r_inv = np.linalg.inv(r_matrix)
    log_det_r = np.log(np.linalg.det(r_matrix))
    ll_const = k * np.log(2.0 * np.pi)

    # ------------------------------------------------------------------
    # Per-timestep log-likelihood
    # Ref: ccc_mvgarch_joint_likelihood.m:64-68
    # ------------------------------------------------------------------
    lls = np.zeros(t, dtype=np.float64)
    for i in range(t):
        # Ref: ccc_mvgarch_joint_likelihood.m:66 — standardised residuals
        # stdResid = data(:,:,i) ./ sqrt(htMat(i,:)' * htMat(i,:))
        h_sqrt = np.sqrt(ht_mat[i, :])
        h_outer = np.outer(h_sqrt, h_sqrt)
        std_resid = data[:, :, i] / h_outer

        # Ref: ccc_mvgarch_joint_likelihood.m:67
        # lls(i) = 0.5*(llconst + sum(log(htMat(i,:))) + LogDetR
        #               + trace(Rinv * stdResid))
        lls[i] = 0.5 * (
            ll_const
            + np.sum(np.log(ht_mat[i, :]))
            + log_det_r
            + np.trace(r_inv @ std_resid)
        )

    # Ref: ccc_mvgarch_joint_likelihood.m:69 — total (negated)
    ll = float(np.sum(lls))

    # ------------------------------------------------------------------
    # Construct conditional covariance array ht
    # Ref: ccc_mvgarch_joint_likelihood.m:70-76
    # In MATLAB this block is gated on nargout>2; in Python we always
    # return all three outputs for API consistency.
    # ------------------------------------------------------------------
    ht = np.zeros((k, k, t), dtype=np.float64)
    for i in range(t):
        h12 = np.sqrt(ht_mat[i, :])
        # Ref: ccc_mvgarch_joint_likelihood.m:74
        # ht(:,:,i) = R .* (h12' * h12)
        ht[:, :, i] = r_matrix * np.outer(h12, h12)

    return ll, lls, ht
