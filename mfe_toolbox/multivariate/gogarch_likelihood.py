"""
GO-GARCH / O-GARCH joint log-likelihood evaluation.

Computes the multivariate log-likelihood for the Generalized Orthogonal GARCH
(GO-GARCH) and Orthogonal GARCH (O-GARCH) models by:

1. Optionally reconstructing eigenvectors/eigenvalues from vectorised covariance
   parameters (inference branch).
2. Constructing the orthonormal rotation matrix U (GO-GARCH) or using the
   identity (O-GARCH).
3. Whitening the observed outer-product data through the mixing matrix Z.
4. Running K independent univariate TARCH/GJR-GARCH recursions on the diagonal
   of the whitened covariance to obtain per-component conditional variances.
5. Accumulating the Gaussian log-likelihood and reconstructing the K×K×T
   conditional covariance path Ht.

Migrated from ``multivariate/gogarch_likelihood.m`` (88 lines).
Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 1, Date: 4/15/2012

Notes
-----
* Numerical parity target: ±1e-6 against MATLAB reference outputs.
* The function negates log-likelihood for minimization by scipy.optimize.
* Invalid log-likelihoods (NaN, Inf, complex) are clamped to 1e7 to protect
  optimizers from unbounded objective landscapes.
"""

import numpy as np

from mfe_toolbox.utility.ivech import ivech
from mfe_toolbox.utility.phi2u import phi2u
from mfe_toolbox.univariate.tarch_core_simple import tarch_core_simple


def gogarch_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    gjr_type: np.ndarray,
    P: np.ndarray,
    L: np.ndarray,
    is_ogarch: bool,
    is_inference: bool,
) -> tuple:
    """
    Log-likelihood for GO-GARCH and O-GARCH models.

    Evaluates the multivariate Gaussian log-likelihood conditional on
    K independent univariate TARCH/GJR-GARCH processes applied to the
    whitened (orthogonalised) components of the data.

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D parameter vector.  Layout depends on model type and mode:

        * **Inference mode** (``is_inference=True``):
          ``[ivech(S)_{K(K+1)/2}, phi_{K(K-1)/2}, vol_params]``
          where ivech(S) encodes the unconditional covariance.
        * **O-GARCH** (``is_ogarch=True``):
          ``[vol_params]`` — no rotation parameters.
        * **GO-GARCH** (``is_ogarch=False``, ``is_inference=False``):
          ``[phi_{K(K-1)/2}, vol_params]``

        ``vol_params`` concatenates ``p[i] + q[i]`` GARCH parameters for
        each of the K component series (omega is derived as
        ``1 - sum(alphas + betas)``).
    data : numpy.ndarray
        K × K × T array of outer-product data or realised covariance
        measures, where K is the number of assets and T is the number
        of time-periods.
    p : numpy.ndarray
        K-element vector of positive integers — number of symmetric
        innovation (ARCH) lags per component.
    q : numpy.ndarray
        K-element vector of non-negative integers — number of
        conditional-variance (GARCH) lags per component.
    gjr_type : numpy.ndarray
        K-element vector indicating variance recursion type per component:

        * 1 — TARCH / AVGARCH (model evolves in absolute values)
        * 2 — GJR-GARCH / standard GARCH (model evolves in squares)
    P : numpy.ndarray
        K × K matrix whose **rows** are the eigenvectors of the
        unconditional covariance matrix (i.e. ``eig(Sigma).T``).
        Overridden when ``is_inference=True``.
    L : numpy.ndarray
        K × K diagonal matrix of eigenvalues of the unconditional
        covariance matrix.  Overridden when ``is_inference=True``.
    is_ogarch : bool
        If ``True`` the model is O-GARCH (rotation matrix U = I_K).
        If ``False`` the model is GO-GARCH with a free rotation.
    is_inference : bool
        If ``True`` the first K(K+1)/2 elements of ``parameters``
        encode ``ivech(S)`` where S is the unconditional covariance;
        P and L are reconstructed via eigendecomposition of S.

    Returns
    -------
    ll : float
        Total log-likelihood (sum of per-observation contributions).
        Clamped to 1e7 if any numerical validity check fails
        (NaN / Inf / complex).
    lls : numpy.ndarray
        T-element 1-D array of per-observation log-likelihoods.
    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices.

    See Also
    --------
    mfe_toolbox.multivariate.gogarch : GO-GARCH driver function.
    mfe_toolbox.utility.ivech.ivech : Inverse half-vectorisation.
    mfe_toolbox.utility.phi2u.phi2u : Angles to orthogonal matrix.
    mfe_toolbox.univariate.tarch_core_simple.tarch_core_simple :
        Univariate TARCH recursion (Numba JIT).

    Notes
    -----
    Ref: ``gogarch_likelihood.m`` — Kevin Sheppard, University of Oxford.

    MATLAB uses 1-based indexing throughout; this Python version uses 0-based
    arrays.  All index translations are annotated with inline comments.

    The MATLAB version conditionally computes Ht only when ``nargout > 2``;
    the Python translation always computes and returns it.
    """
    # ------------------------------------------------------------------
    # Dimension extraction
    # Ref: gogarch_likelihood.m:30 — [k,~,T] = size(data);
    # data is K×K×T; MATLAB uses 1-based "size", Python uses .shape.
    # ------------------------------------------------------------------
    k = data.shape[0]
    T = data.shape[2]

    # ------------------------------------------------------------------
    # Ensure parameters is a 1-D float64 vector
    # Ref: gogarch_likelihood.m:31-33 — row vector enforcement
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    # Running offset into the parameter vector — tracks how many elements
    # have been consumed as we parse inference, rotation, and GARCH blocks.
    offset = 0

    # ------------------------------------------------------------------
    # Inference branch: reconstruct P, L from ivech(S)
    # Ref: gogarch_likelihood.m:36-41
    # ------------------------------------------------------------------
    if is_inference:
        n_vech = k * (k + 1) // 2
        # Ref: gogarch_likelihood.m:37 — S = ivech(parameters(1:k*(k+1)/2));
        S = ivech(parameters[:n_vech])

        # Ref: gogarch_likelihood.m:38 — [P,L] = eig(S);
        # np.linalg.eig returns (eigenvalues, eigenvectors_as_columns).
        eigvals, eigvecs = np.linalg.eig(S)

        # Ref: gogarch_likelihood.m:39 — P = P';
        # MATLAB eig returns column eigenvectors; we transpose so that
        # rows are eigenvectors, matching the MATLAB convention.
        P = eigvecs.T
        L = np.diag(np.real(eigvals))

        offset += n_vech

    # ------------------------------------------------------------------
    # Rotation matrix construction
    # Ref: gogarch_likelihood.m:43-49
    # ------------------------------------------------------------------
    if not is_ogarch:
        n_phi = k * (k - 1) // 2
        # Ref: gogarch_likelihood.m:44 — phi = parameters(offset + (1:k*(k-1)/2));
        phi = parameters[offset:offset + n_phi]
        # Ref: gogarch_likelihood.m:45 — U = phi2u(phi);
        U = phi2u(phi)
        offset += n_phi
    else:
        # Ref: gogarch_likelihood.m:48 — U = eye(k);
        U = np.eye(k)

    # ------------------------------------------------------------------
    # Whitening / standardisation transforms
    # Ref: gogarch_likelihood.m:50-55
    #
    # L is a diagonal eigenvalue matrix.  MATLAB L^(0.5) computes the
    # matrix square root which, for a diagonal matrix, is element-wise
    # sqrt of the diagonal.  We construct it explicitly:
    #   L_sqrt     = diag(sqrt(diag(L)))
    #   L_inv_sqrt = diag(1 / sqrt(diag(L)))
    # ------------------------------------------------------------------
    L_diag = np.real(np.diag(L))
    L_sqrt = np.diag(np.sqrt(np.abs(L_diag)))  # abs guards against tiny negatives
    # Guard against division by zero for very small eigenvalues
    L_diag_safe = np.where(np.abs(L_diag) < 1e-300, 1e-300, np.abs(L_diag))
    L_inv_sqrt = np.diag(1.0 / np.sqrt(L_diag_safe))

    # Ref: gogarch_likelihood.m:50 — Z = P*L^(0.5)*U;
    Z = P @ L_sqrt @ U
    # Ref: gogarch_likelihood.m:51 — Zinv = U'*L^(-0.5)*P';
    Zinv = U.T @ L_inv_sqrt @ P.T

    # Ref: gogarch_likelihood.m:52-55 — Standardise each time-slice
    std_data = np.zeros((k, k, T))
    for t in range(T):
        # Ref: gogarch_likelihood.m:54 — stdData(:,:,t) = Zinv*data(:,:,t)*Zinv';
        std_data[:, :, t] = Zinv @ data[:, :, t] @ Zinv.T

    # ------------------------------------------------------------------
    # Per-series univariate GARCH recursions
    # Ref: gogarch_likelihood.m:57-72
    # ------------------------------------------------------------------
    V = np.zeros((T, k))

    # Exponential decay weights for back-casting
    # Ref: gogarch_likelihood.m:58 — w = .06 * .94.^(0:sqrt(T));
    # MATLAB 0:sqrt(T) produces integers 0..floor(sqrt(T)), length = floor(sqrt(T)) + 1.
    w_len = int(np.sqrt(T)) + 1
    w = 0.06 * 0.94 ** np.arange(w_len)
    # Ref: gogarch_likelihood.m:59 — w = w/sum(w);
    w = w / np.sum(w)

    lik_data = np.zeros((T, k))

    for i in range(k):
        # Ref: gogarch_likelihood.m:62 — count = p(i) + q(i);
        p_i = int(p[i])
        q_i = int(q[i])
        count = p_i + q_i

        # Ref: gogarch_likelihood.m:63 — volParameters = parameters(offset + (1:count));
        vol_parameters = parameters[offset:offset + count].copy()
        # Ref: gogarch_likelihood.m:64 — volParameters = max(volParameters,0);
        vol_parameters = np.maximum(vol_parameters, 0.0)
        # Ref: gogarch_likelihood.m:65 — volParameters = [1-sum(volParameters) volParameters];
        # Prepend omega = 1 - sum(alpha + beta) so that parameters encode a
        # variance-targeting representation.
        omega = 1.0 - np.sum(vol_parameters)
        vol_parameters = np.concatenate(([omega], vol_parameters))

        offset += count

        # Ref: gogarch_likelihood.m:67 — likData(:,i) = squeeze(stdData(i,i,:));
        # Extract the diagonal variance series of the whitened data.
        lik_data[:, i] = std_data[i, i, :]

        # Ref: gogarch_likelihood.m:68 — volData = sqrt(likData(:,i));
        # These are standard-deviation-scale "returns" for the GARCH recursion.
        vol_data = np.sqrt(np.maximum(lik_data[:, i], 0.0))

        # Ref: gogarch_likelihood.m:69 — backCast = w*volData(1:length(w)).^2;
        # Weighted average of squared volatility for back-cast initialisation.
        actual_w_len = min(w_len, T)
        back_cast = float(np.dot(w[:actual_w_len], vol_data[:actual_w_len] ** 2))

        # -----------------------------------------------------------------
        # Call Python tarch_core_simple (Numba JIT)
        #
        # CRITICAL: The Python implementation has a DIFFERENT interface from
        # the MATLAB version.  MATLAB tarch_core_simple transforms data
        # internally; the Python version expects pre-transformed fdata and
        # fIdata, plus explicit m and T arguments.
        #
        # Ref: tarch_core_simple.m:45-50 — internal data transformation
        # Ref: gogarch_likelihood.m:70 — v = tarch_core_simple(volData,
        #   volParameters, backCast, 0, p(i), 0, q(i), gjrType(i));
        # -----------------------------------------------------------------
        tarch_type_i = int(gjr_type[i])
        if tarch_type_i == 1:
            # AVGARCH: model evolves in absolute values → f(x) = |x|
            fdata_i = np.abs(vol_data)
        else:
            # GJR-GARCH / standard GARCH: model evolves in squares → f(x) = x²
            fdata_i = vol_data ** 2

        # Asymmetric indicator-weighted data: fIdata = fdata * (data < 0)
        # Since vol_data = sqrt(max(lik_data, 0)) >= 0, the indicator
        # (vol_data < 0) is always False → fIdata is all zeros.
        # Ref: gogarch_likelihood.m:70 — backCastAsym=0, o=0 (no asymmetry)
        fIdata_i = np.zeros(T, dtype=np.float64)

        o_i = 0  # No asymmetric terms in GO-GARCH specification
        m_i = max(p_i, o_i, q_i)

        v = tarch_core_simple(
            fdata_i,         # pre-transformed data
            fIdata_i,        # asymmetric data (all zeros)
            vol_parameters,  # [omega, alpha_1..alpha_p, beta_1..beta_q]
            back_cast,       # exponential back-cast value
            p_i,             # ARCH order
            o_i,             # asymmetric order (0)
            q_i,             # GARCH order
            m_i,             # max lag
            T,               # number of observations
            tarch_type_i,    # recursion type
        )
        # Ref: gogarch_likelihood.m:71 — V(:,i) = v;
        V[:, i] = v

    # ------------------------------------------------------------------
    # Log-likelihood computation
    # Ref: gogarch_likelihood.m:74-77
    # ------------------------------------------------------------------

    # Ref: gogarch_likelihood.m:74 — likConst = k*log(2*pi);
    lik_const = k * np.log(2.0 * np.pi)

    # Ref: gogarch_likelihood.m:75 — logdetZZp = log(det(Z*Z'));
    ZZt = Z @ Z.T
    det_ZZt = np.linalg.det(ZZt)
    log_det_ZZp = np.log(np.abs(det_ZZt)) if det_ZZt != 0.0 else -np.inf

    # Ref: gogarch_likelihood.m:76 — lls = 0.5*(likConst + logdetZZp
    #   + sum(log(V),2) + sum(likData./V,2));
    # MATLAB sum(X, 2) sums along columns → Python np.sum(X, axis=1).
    lls = 0.5 * (
        lik_const
        + log_det_ZZp
        + np.sum(np.log(V), axis=1)
        + np.sum(lik_data / V, axis=1)
    )

    # Ref: gogarch_likelihood.m:77 — ll = sum(lls);
    ll = np.sum(lls)

    # ------------------------------------------------------------------
    # Validity check
    # Ref: gogarch_likelihood.m:79-81 — clamp invalid likelihoods to 1e7
    # In MATLAB, ~isreal(ll) || isnan(ll) || isinf(ll) triggers clamping.
    # In Python, complex values from eigendecomposition could propagate;
    # NaN/Inf arise from log(0), div-by-zero, or negative variances.
    # ------------------------------------------------------------------
    ll_real = np.real(ll)
    if (not np.isreal(ll)) or np.isnan(ll_real) or np.isinf(ll_real):
        ll = 1e7
    else:
        ll = float(ll_real)

    # ------------------------------------------------------------------
    # Conditional covariance reconstruction
    # Ref: gogarch_likelihood.m:83-88
    # MATLAB conditionally computes Ht when nargout > 2; Python always
    # returns it as part of the output tuple.
    # ------------------------------------------------------------------
    Ht = np.zeros((k, k, T))
    for t in range(T):
        # Ref: gogarch_likelihood.m:86 — Ht(:,:,t) = Z*diag(V(t,:))*Z';
        Ht[:, :, t] = Z @ np.diag(V[t, :]) @ Z.T

    return (ll, lls, Ht)
