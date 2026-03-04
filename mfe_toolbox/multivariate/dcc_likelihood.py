"""
DCC log-likelihood evaluation for scalar DCC(m,n) and ADCC(m,l,n) models.

Computes the quasi-maximum-likelihood (QMLE) or composite normal
log-likelihood for Dynamic Conditional Correlation models with
TARCH(p,o,q) or GJR-GARCH(p,o,q) univariate conditional variances.

The function supports three estimation stages:

- **Stage 1 (joint):** Univariate GARCH + correlation intercept + DCC dynamics
  are estimated jointly.
- **Stage 2 (two-stage):** Correlation intercept and DCC dynamics estimated
  conditionally on fixed univariate GARCH parameters. During inference
  (``is_inference=True``), all parameters are included.
- **Stage 3 (three-stage):** DCC dynamics only, conditioned on fixed
  correlation intercept and univariate GARCH. During inference all
  parameters are included plus the asymmetric intercept *N*.

Three composite-likelihood modes are available:

- ``composite=0``: Full QMLE using log-determinant and quadratic form.
- ``composite=1``: Diagonal composite — only adjacent series pairs.
- ``composite=2``: Full composite — all K(K-1)/2 lower-triangular pairs.

Migrated from ``multivariate/dcc_likelihood.m`` (177 lines).

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 4/13/2012

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration

See Also
--------
mfe_toolbox.multivariate.dcc : DCC model driver.
mfe_toolbox.multivariate.dcc_reconstruct_variance :
    Reconstruct TARCH variances from parameter vector.
mfe_toolbox.distributions.composite_likelihood :
    Bivariate pairwise composite normal log-likelihood.
"""

import numpy as np

from mfe_toolbox.distributions.composite_likelihood import composite_likelihood
from mfe_toolbox.multivariate.dcc_reconstruct_variance import dcc_reconstruct_variance
from mfe_toolbox.utility.corr_ivech import corr_ivech
from mfe_toolbox.utility.ivech import ivech
from mfe_toolbox.utility.z2r import z2r


def dcc_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    data_asym: np.ndarray,
    m: int,
    l: int,
    n: int,
    R: np.ndarray,
    N: np.ndarray,
    back_cast: np.ndarray,
    back_cast_asym: np.ndarray,
    stage: int,
    composite: int,
    is_joint: bool,
    is_inference: bool,
    g_scale: np.ndarray,
    univariate: list,
) -> tuple:
    """Evaluate DCC/ADCC log-likelihood at a given parameter vector.

    Computes the negative log-likelihood (for minimisation) of a scalar
    DCC(m, n) or asymmetric DCC (ADCC(m, l, n)) model with TARCH-type
    univariate conditional variances.

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D parameter vector whose layout depends on *stage*, *is_joint*,
        and *l*:

        - Stage 1 or ``is_joint=True``: ``[garch_params, R_vech, (N_vech), a, g, b]``
        - Stage 2 estimation: ``[R_vech, a, g, b]``
        - Stage 2 inference: ``[garch_params, R_vech, a, g, b]``
        - Stage 3 estimation: ``[a, g, b]``
        - Stage 3 inference: ``[garch_params, R_vech, N_vech, a, g, b]``

    data : numpy.ndarray
        K × K × T array of outer-product data matrices (e.g. return outer
        products ``r_t * r_t'``) or covariance estimators (e.g. realised
        covariance).
    data_asym : numpy.ndarray
        K × K × T array of asymmetric outer-product data (e.g.
        ``r_t^- * r_t^{-'}`` where ``r_t^- = min(r_t, 0)``).  Only
        required when ``l > 0`` or the univariate models have leverage.
    m : int
        Order of symmetric innovation terms in the DCC dynamics.
    l : int
        Order of asymmetric innovation terms in the ADCC dynamics.
        Set to 0 for standard (non-asymmetric) DCC.
    n : int
        Order of lagged correlation (Q) terms in the DCC dynamics.
    R : numpy.ndarray
        K × K unconditional correlation matrix of standardised data.
        Overwritten by the corresponding slice of *parameters* when
        ``stage <= 2`` or ``is_joint`` is ``True``.
    N : numpy.ndarray
        K × K matrix of mean asymmetric standardised data outer products.
        Overwritten by the corresponding slice of *parameters* when
        ``stage == 3`` and ``is_joint`` and ``l > 0``.
    back_cast : numpy.ndarray
        K × K back-cast matrix for initialising the symmetric Q(t)
        recursion when pre-sample periods are needed.
    back_cast_asym : numpy.ndarray
        K × K back-cast matrix for initialising the asymmetric Q(t)
        recursion when pre-sample periods are needed.
    stage : int
        Estimation stage, one of ``{1, 2, 3}``.
    composite : int
        Composite likelihood mode:

        - ``0`` — Full QMLE (log-determinant + quadratic form).
        - ``1`` — Diagonal composite (adjacent pair likelihoods).
        - ``2`` — Full composite (all K(K-1)/2 pair likelihoods).

    is_joint : bool
        If ``True``, *parameters* includes the univariate GARCH block
        and the correlation intercept block in addition to the DCC
        dynamics block.
    is_inference : bool
        If ``True``, parameters are on their natural scale (no
        transformation applied to R).  If ``False``, R is transformed
        from an unconstrained parameterisation via :func:`z2r`.
    g_scale : numpy.ndarray
        K-element vector used in stage-2 estimation to scale the
        intercept for the asymmetric term.  See :func:`dcc`.
    univariate : list[dict]
        List of *K* dictionaries containing first-stage univariate
        TARCH estimation results.  Each dict must contain keys
        ``'p'``, ``'o'``, ``'q'`` (and additional fields expected by
        :func:`dcc_reconstruct_variance`).

    Returns
    -------
    ll : float
        Total negative log-likelihood (sum of per-period values).
        Returns ``1e7`` if the value is NaN, Inf, or complex.
    lls : numpy.ndarray
        T-element 1-D array of per-period negative log-likelihoods.
    Rt : numpy.ndarray
        K × K × T array of conditional correlation matrices.

    Notes
    -----
    **Stage / mode matrix** (Ref: dcc_likelihood.m:44-49):

    =====  ===========  =====================================================
    Stage  Estimation   Parameter blocks in *parameters*
    =====  ===========  =====================================================
    1      joint        ``[garch, R_vech, a, g, b]``
    2      estimation   ``[R_vech, a, g, b]``
    2      inference    ``[garch, R_vech, a, g, b]``
    3      estimation   ``[a, g, b]``
    3      inference    ``[garch, R_vech, N_vech, a, g, b]``
    =====  ===========  =====================================================

    The DCC dynamics recursion for Q(t) is:

    .. math::

       Q_t = \\text{intercept} + \\sum_{i=1}^{m} a_i \\, z_{t-i} z_{t-i}'
             + \\sum_{i=1}^{l} g_i \\, n_{t-i} n_{t-i}'
             + \\sum_{i=1}^{n} b_i \\, Q_{t-i}

    The conditional correlation is obtained by rescaling:

    .. math::

       R_t = \\text{diag}(Q_t)^{-1/2} \\, Q_t \\, \\text{diag}(Q_t)^{-1/2}
    """
    # ------------------------------------------------------------------
    # Phase 1 — Extract dimensions
    # Ref: dcc_likelihood.m:51 — [k,~,T] = size(data)
    # ------------------------------------------------------------------
    k = data.shape[0]
    T = data.shape[2]
    offset = 0

    # ------------------------------------------------------------------
    # Phase 2 — Parse parameters
    # Ref: dcc_likelihood.m:52-86
    # ------------------------------------------------------------------

    # 2a. GARCH parameters (stage 1 or joint)
    # Ref: dcc_likelihood.m:54-65
    if stage == 1 or is_joint:
        count = 0
        for i in range(k):
            u = univariate[i]
            # Ref: dcc_likelihood.m:58 — count = count + u.p + u.o + u.q + 1
            count += u['p'] + u['o'] + u['q'] + 1
        garch_parameters = parameters[offset:offset + count]
        offset += count
        compute_vol = True
    else:
        compute_vol = False

    # 2b. R (correlation intercept) parameters (stage <= 2 or joint)
    # Ref: dcc_likelihood.m:66-70
    if stage <= 2 or is_joint:
        count = k * (k - 1) // 2
        R = parameters[offset:offset + count]
        offset += count

    # 2c. N (asymmetric intercept) parameters (stage 3, joint, l > 0)
    # Ref: dcc_likelihood.m:71-77
    if stage == 3 and is_joint and l > 0:
        count = k * (k + 1) // 2
        N = parameters[offset:offset + count]
        # Ref: dcc_likelihood.m:75 — ivech transforms vector to symmetric matrix
        N = ivech(N)
        offset += count

    # 2d. DCC dynamics parameters: a(1:m), g(1:l), b(1:n)
    # Ref: dcc_likelihood.m:78-80 — MATLAB 1-based (1:m) → Python 0-based [0:m]
    a = parameters[offset:offset + m]
    g = parameters[offset + m:offset + m + l]
    b = parameters[offset + m + l:offset + m + l + n]

    # Ref: dcc_likelihood.m:81-86 — handle empty parameter arrays
    if g.size == 0:
        g = 0.0
    if b.size == 0:
        b = 0.0

    # ------------------------------------------------------------------
    # Phase 3 — Compute conditional variances
    # Ref: dcc_likelihood.m:87-103
    # ------------------------------------------------------------------
    H = np.ones((T, k))
    if compute_vol:
        # Ref: dcc_likelihood.m:90 — reconstruct TARCH variances
        H = dcc_reconstruct_variance(garch_parameters, univariate)
        std_data = np.zeros((k, k, T))
        std_data_asym = np.zeros((k, k, T))
        for t in range(T):
            # Ref: dcc_likelihood.m:94 — h = sqrt(H(t,:))
            h = np.sqrt(H[t, :])
            # Ref: dcc_likelihood.m:95 — h'*h is the k×k outer product
            hh = np.outer(h, h)
            # Ref: dcc_likelihood.m:95-96 — element-wise division ./(h'*h)
            std_data[:, :, t] = data[:, :, t] / hh
            std_data_asym[:, :, t] = data_asym[:, :, t] / hh
        # Ref: dcc_likelihood.m:98 — sum(log(H), 2) → axis=1 in Python
        logdet_H = np.sum(np.log(H), axis=1)
    else:
        # Ref: dcc_likelihood.m:100-102 — data already standardised
        std_data = data
        std_data_asym = data_asym
        logdet_H = np.zeros(T)

    # ------------------------------------------------------------------
    # Phase 4 — Transform R and N
    # Ref: dcc_likelihood.m:105-113
    # ------------------------------------------------------------------
    if stage <= 2 or is_joint:
        if is_inference:
            # Ref: dcc_likelihood.m:109 — direct correlation ivech
            R = corr_ivech(R)
        else:
            # Ref: dcc_likelihood.m:111 — unconstrained → correlation via z2r
            R = z2r(R)

    # ------------------------------------------------------------------
    # Phase 5 — Compute intercept for Q(t) recursion
    # Ref: dcc_likelihood.m:115-125
    # ------------------------------------------------------------------
    if stage == 3:
        # Ref: dcc_likelihood.m:117 — R * scalar
        intercept = R * (1.0 - np.sum(a) - np.sum(b))
        if l > 0:
            # Ref: dcc_likelihood.m:119 — subtract asymmetric intercept
            intercept = intercept - N * np.sum(g)
    else:
        # Ref: dcc_likelihood.m:122 — g_scale is K-vector
        scale = (1.0 - np.sum(a) - np.sum(b)) - g_scale * np.sum(g)
        # Ref: dcc_likelihood.m:123 — element-wise sqrt
        scale = np.sqrt(scale)
        # Ref: dcc_likelihood.m:124 — R .* (scale*scale') outer product
        intercept = R * np.outer(scale, scale)

    # ------------------------------------------------------------------
    # Phase 6 — Composite likelihood index setup
    # Ref: dcc_likelihood.m:128-136
    # ------------------------------------------------------------------
    if composite == 0:
        # Ref: dcc_likelihood.m:130 — QMLE constant
        likconst = k * np.log(2.0 * np.pi)
    elif composite == 1:
        # Ref: dcc_likelihood.m:132 — diagonal adjacent pairs
        # MATLAB: [(1:k-1)' (2:k)'] → Python 0-based: [(0,1),(1,2),...,(k-2,k-1)]
        indices = np.column_stack([np.arange(0, k - 1), np.arange(1, k)])
    elif composite == 2:
        # Ref: dcc_likelihood.m:134-135 — all lower-triangular pairs
        # MATLAB meshgrid(1:k) with ~triu → Python meshgrid(arange(k)) with ~triu
        ii, jj = np.meshgrid(np.arange(k), np.arange(k))
        mask = ~np.triu(np.ones((k, k), dtype=bool))
        indices = np.column_stack([ii[mask], jj[mask]])

    # ------------------------------------------------------------------
    # Phase 7-9 — DCC Q(t) recursion, R(t) rescaling, per-period LL
    # Ref: dcc_likelihood.m:138-173
    # ------------------------------------------------------------------
    I_k = np.eye(k)
    Qt = np.zeros((k, k, T))
    Rt = np.zeros((k, k, T))
    lls = np.zeros(T)

    for t in range(T):
        # Ref: dcc_likelihood.m:143 — Qt(:,:,t) = intercept
        Qt[:, :, t] = intercept

        # --- Symmetric innovation terms ---
        # Ref: dcc_likelihood.m:144-150
        # MATLAB: for i=1:m, check (t-i)>0, use stdData(:,:,t-i)
        # Python: for i=0:m-1, lag=i+1, check t-lag>=0, use std_data[:,:,t-lag]
        for i in range(m):
            if t - i - 1 >= 0:
                Qt[:, :, t] += a[i] * std_data[:, :, t - i - 1]
            else:
                Qt[:, :, t] += a[i] * back_cast

        # --- Asymmetric innovation terms ---
        # Ref: dcc_likelihood.m:151-157
        for i in range(l):
            if t - i - 1 >= 0:
                Qt[:, :, t] += g[i] * std_data_asym[:, :, t - i - 1]
            else:
                Qt[:, :, t] += g[i] * back_cast_asym

        # --- Lagged Q terms ---
        # Ref: dcc_likelihood.m:158-163
        for i in range(n):
            if t - i - 1 >= 0:
                Qt[:, :, t] += b[i] * Qt[:, :, t - i - 1]
            else:
                Qt[:, :, t] += b[i] * back_cast

        # --- Correlation rescaling: R(t) = Q(t) / (q * q') ---
        # Ref: dcc_likelihood.m:165 — q = sqrt(diag(Qt(:,:,t)))
        q = np.sqrt(np.diag(Qt[:, :, t]))
        # Ref: dcc_likelihood.m:166 — Qt ./ (q*q') outer product normalisation
        Rt[:, :, t] = Qt[:, :, t] / np.outer(q, q)

        # --- Per-period negative log-likelihood ---
        if composite == 0:
            # Ref: dcc_likelihood.m:168 — Full QMLE path
            # Use slogdet for numerical stability instead of log(det(...))
            sign, logdet_Rt = np.linalg.slogdet(Rt[:, :, t])
            # Ref: dcc_likelihood.m:168 — (Rt\I)*stdData → solve(Rt, stdData)
            # np.linalg.solve(A, B) solves A @ X = B → X = A^{-1} @ B
            Rt_inv_std = np.linalg.solve(Rt[:, :, t], std_data[:, :, t])
            # Ref: dcc_likelihood.m:168 — sum(diag(...)) = trace(...)
            lls[t] = 0.5 * (
                likconst
                + logdet_H[t]
                + logdet_Rt
                + np.sum(np.diag(Rt_inv_std))
            )
        else:
            # Ref: dcc_likelihood.m:170 — composite path (diagonal or full)
            # S = (sqrt(H(t,:))' * sqrt(H(t,:))) .* Rt(:,:,t)
            h_sqrt = np.sqrt(H[t, :])
            S = np.outer(h_sqrt, h_sqrt) * Rt[:, :, t]
            # Ref: dcc_likelihood.m:171 — composite_likelihood uses 0-based indices
            lls[t] = composite_likelihood(S, data[:, :, t], indices)

    # ------------------------------------------------------------------
    # Phase 10 — Aggregate and apply safety check
    # Ref: dcc_likelihood.m:174-178
    # ------------------------------------------------------------------
    ll = float(np.sum(lls))

    # Ref: dcc_likelihood.m:176-178 — guard against invalid likelihood values
    if np.isnan(ll) or np.isinf(ll) or not np.isreal(ll):
        ll = 1e7

    return ll, lls, Rt
