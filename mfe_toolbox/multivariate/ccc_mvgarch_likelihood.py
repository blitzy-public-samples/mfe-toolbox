"""
CCC-MVGARCH Correlation Log-Likelihood

Computes the negated log-likelihood for Constant Conditional Correlation
(CCC) Multivariate GARCH inference. This function is used for computing
scores during robust variance-covariance matrix (VCV) estimation.

Migrated from: multivariate/ccc_mvgarch_likelihood.m
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 4, Date: 10/28/2009
"""

import numpy as np


def _corr_ivech(stacked_data: np.ndarray) -> np.ndarray:
    """
    Reconstruct a symmetric correlation matrix from its off-diagonal
    lower-triangular half-vectorization (stored in MATLAB column-major order).

    This is a local helper equivalent to utility/corr_ivech.m, implemented
    here because depends_on_files is empty — no internal imports are allowed.

    Parameters
    ----------
    stacked_data : ndarray
        K*(K-1)/2 vector of off-diagonal correlation parameters. The elements
        are ordered column-by-column in the lower triangle (MATLAB convention).

    Returns
    -------
    R : ndarray
        K x K symmetric correlation matrix with ones on the diagonal.

    Raises
    ------
    ValueError
        If the length of stacked_data is not conformable to a valid K.

    Notes
    -----
    MATLAB fills the lower triangle in column-major order (corr_ivech.m:48-51).
    In Python/NumPy, boolean indexing follows row-major order, so directly
    assigning to the lower triangle via ``R[mask] = stacked_data`` would produce
    incorrect results for K >= 4. Instead, we fill the upper triangle in
    row-major order, which is equivalent to MATLAB's lower-triangle column-major
    order after symmetrization via ``R + R.T + eye(K)``.

    Ref: corr_ivech.m — full 51-line source analyzed for this translation.
    """
    stacked_data = np.asarray(stacked_data, dtype=np.float64).ravel()
    k2 = len(stacked_data)

    # Ref: corr_ivech.m:39 — solve K from quadratic: K*(K-1)/2 = k2
    # K = (-1 + sqrt(1 + 8*k2)) / 2 + 1
    k = int((-1.0 + np.sqrt(1.0 + 8.0 * k2)) / 2.0) + 1

    if k * (k - 1) // 2 != k2:
        raise ValueError(
            f"The number of elements in stacked_data ({k2}) is not conformable "
            "to the inverse vech operation for a correlation matrix."
        )

    # Ref: corr_ivech.m:48-51
    # MATLAB fills lower triangle column-major; Python equivalent is
    # filling upper triangle row-major, then symmetrizing.
    R = np.zeros((k, k), dtype=np.float64)
    rows, cols = np.triu_indices(k, 1)
    R[rows, cols] = stacked_data
    # Ref: corr_ivech.m:51 — matrixData = matrixData + matrixData' + eye(K)
    R = R + R.T + np.eye(k, dtype=np.float64)

    return R


def ccc_mvgarch_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    ht_mat: np.ndarray,
) -> tuple:
    """
    Log-likelihood for CCC-MVGARCH inference.

    Computes the negated log-likelihood for a Constant Conditional Correlation
    (CCC) multivariate GARCH model. This function is primarily used for
    computing scores during robust VCV estimation and not for optimization.

    Parameters
    ----------
    parameters : ndarray
        K*(K-1)/2 vector of constant correlation parameters. The full
        correlation matrix is reconstructed as ``R = corr_ivech(parameters)``.
        Elements are the off-diagonal entries of the lower triangle of the
        correlation matrix, stored in column-major (MATLAB) order.
    data : ndarray
        K x K x T three-dimensional array of positive semi-definite matrices.
        Typically the outer products of the observation vectors
        (epsilon_t @ epsilon_t.T).
    ht_mat : ndarray
        T x K matrix of fitted conditional variances from the first-stage
        univariate GARCH estimation.

    Returns
    -------
    ll : float
        Negated sum of log-likelihoods (i.e., minus one times the total
        log-likelihood). Suitable as a minimization objective.
    lls : ndarray
        T-element array of per-observation negated log-likelihoods.

    Notes
    -----
    The CCC negated log-likelihood for observation t is:

        lls_t = 0.5 * [K*log(2*pi) + sum_i(log(h_{it})) + log|R|
                        + tr(R^{-1} * D_t^{-1} * Sigma_t * D_t^{-1})]

    where:
        - K is the number of assets
        - h_{it} is the conditional variance for asset i at time t
        - R is the constant correlation matrix
        - D_t = diag(sqrt(h_{1t}), ..., sqrt(h_{Kt}))
        - Sigma_t = data[:, :, t] is the cross-product matrix at time t

    The standardized residual matrix is computed as:

        stdResid_t = Sigma_t / sqrt(h_t * h_t')

    where h_t * h_t' is the K x K outer product of conditional variances.

    The function returns the negated log-likelihood so it can be used directly
    as a minimization objective for optimization.

    References
    ----------
    Bollerslev, T. (1990). Modelling the coherence in short-run nominal
    exchange rates: A multivariate generalized ARCH model.
    Review of Economics and Statistics, 72, 498-505.

    Migrated from multivariate/ccc_mvgarch_likelihood.m
    """
    # Ensure inputs are numpy arrays with proper dtype for numerical stability
    parameters = np.asarray(parameters, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    ht_mat = np.asarray(ht_mat, dtype=np.float64)

    # Ref: ccc_mvgarch_likelihood.m:24 — extract dimensions from data array
    # [k, nothing, t] = size(data);  MATLAB uses 1-indexed dimensions
    k = data.shape[0]
    t = data.shape[2]

    # Ref: ccc_mvgarch_likelihood.m:26 — reconstruct constant correlation matrix
    # R = corr_ivech(parameters);
    R = _corr_ivech(parameters)

    # Ref: ccc_mvgarch_likelihood.m:28 — compute inverse of correlation matrix
    # Rinv = inv(R);
    R_inv = np.linalg.inv(R)

    # Ref: ccc_mvgarch_likelihood.m:29 — compute log-determinant of R
    # LogDetR = log(det(R));
    # Using slogdet for numerical stability; MATLAB uses log(det(R)) directly
    sign, log_det_r = np.linalg.slogdet(R)
    if sign <= 0:
        # Non-positive-definite R: return large penalty to steer optimizer away
        # This can happen during optimization with invalid parameter proposals
        lls = np.full(t, 1e7, dtype=np.float64)
        ll = float(np.sum(lls))
        return ll, lls

    # Ref: ccc_mvgarch_likelihood.m:30 — constant part of multivariate normal LL
    # llconst = k*log(2*pi);
    ll_const = k * np.log(2.0 * np.pi)

    # Ref: ccc_mvgarch_likelihood.m:31 — initialize per-observation log-likelihoods
    # lls = zeros(t,1);
    lls = np.zeros(t, dtype=np.float64)

    # Ref: ccc_mvgarch_likelihood.m:32-35 — main likelihood loop
    # for i=1:t
    #     stdResid = data(:,:,i)./sqrt(htMat(i,:)'*htMat(i,:));
    #     lls(i) = 0.5*(llconst+sum(log(htMat(i,:)))+LogDetR+trace(Rinv*stdResid));
    # end
    for i in range(t):
        # Ref: ccc_mvgarch_likelihood.m:33 — compute standardized residual matrix
        # htMat(i,:)' * htMat(i,:) produces a K×K outer product of conditional
        # variances; sqrt applies element-wise; data(:,:,i) ./ sqrt(...)
        # divides element-wise to produce the standardized residual covariance.
        ht_i = ht_mat[i, :]  # K-element vector of conditional variances at time i
        # Ref: ccc_mvgarch_likelihood.m:33 — outer product of conditional variances
        ht_outer = np.outer(ht_i, ht_i)  # K×K matrix: h_it * h_jt
        std_resid = data[:, :, i] / np.sqrt(ht_outer)  # Element-wise division

        # Ref: ccc_mvgarch_likelihood.m:34 — per-observation negated log-likelihood
        lls[i] = 0.5 * (
            ll_const
            + np.sum(np.log(ht_i))
            + log_det_r
            + np.trace(R_inv @ std_resid)
        )

    # Ref: ccc_mvgarch_likelihood.m:36 — total negated log-likelihood
    # ll = sum(lls);
    ll = float(np.sum(lls))

    return ll, lls
