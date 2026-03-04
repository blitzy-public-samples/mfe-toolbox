"""
RCC (Rotated Conditional Correlation) log-likelihood evaluation.

Evaluates the correlation-only (or joint) log-likelihood for the RCC model,
which applies RARCH-style dynamics in the *correlation* (not covariance) space.
The function supports three model types matching RARCH:

    * **Scalar** (``type_model=1``)  — Scalar identity matrices for A and B
    * **Common Persistence / CP** (``type_model=2``) — Diagonal A, shared
      persistence theta for B
    * **Diagonal** (``type_model=3``) — Fully diagonal A and B

The correlation dynamics are:

.. math::

    e_t = R^{-1/2} \\cdot \\mathrm{stdData}_t \\cdot R^{-1/2}

    Q_t = \\mathrm{intercept} + \\sum_{j=1}^{m} A_j \\, e_{t-j} \\, A_j
          + \\sum_{j=1}^{n} B_j \\, Q_{t-j} \\, B_j

    R_t = \\mathrm{diag}(\\tilde{R}_t)^{-1/2} \\, \\tilde{R}_t
          \\, \\mathrm{diag}(\\tilde{R}_t)^{-1/2}

where :math:`\\tilde{R}_t = R^{1/2} \\, Q_t \\, R^{1/2}` and the intercept
is clamped to ensure positive diagonal entries.

Migrated from ``multivariate/rcc_likelihood.m`` (107 lines) in the MFE
Toolbox (Version 4.0, Kevin Sheppard, University of Oxford).

Notes
-----
- Uses RARCH-style dynamics via :func:`rarch_parameter_transform`.
- Supports composite (bivariate pairwise) likelihood for large-dimensional
  problems via :func:`composite_likelihood`.
- When ``stage == 1`` or ``is_joint`` is True, per-series GARCH variances
  are reconstructed from the parameter vector via
  :func:`dcc_reconstruct_variance`.

References
----------
Kevin Sheppard, MFE Toolbox Version 4.0 (2009).

See Also
--------
mfe_toolbox.multivariate.rcc : RCC model driver.
mfe_toolbox.multivariate.rcc_constraint : RCC nonlinear constraints.
mfe_toolbox.multivariate.rarch_parameter_transform : RARCH parameter unpacking.
mfe_toolbox.multivariate.dcc_reconstruct_variance : Variance reconstruction.
mfe_toolbox.distributions.composite_likelihood : Composite likelihood helper.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from mfe_toolbox.utility.corr_ivech import corr_ivech
from mfe_toolbox.utility.z2r import z2r
from mfe_toolbox.multivariate.rarch_parameter_transform import rarch_parameter_transform
from mfe_toolbox.multivariate.dcc_reconstruct_variance import dcc_reconstruct_variance
from mfe_toolbox.distributions.composite_likelihood import composite_likelihood


def rcc_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    m: int,
    n: int,
    R: np.ndarray,
    back_cast: np.ndarray,
    stage: int,
    type_model: int,
    composite: int,
    is_joint: bool,
    is_inference: bool,
    r_scale: np.ndarray,
    univariate: list[dict],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the RCC (Rotated Conditional Correlation) log-likelihood.

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D parameter vector.  The layout depends on ``stage`` and
        ``is_joint``:

        * If ``stage == 1`` or ``is_joint`` is True, the first
          ``sum(1 + p_i + o_i + q_i)`` elements encode concatenated
          per-series GARCH parameters.
        * If ``stage <= 2`` or ``is_joint`` is True, the next
          ``K(K-1)/2`` elements encode the correlation intercept
          (either direct correlation elements or unconstrained z-values).
        * The remaining elements encode the RARCH dynamics parameters
          (A and B matrices via :func:`rarch_parameter_transform`).

    data : numpy.ndarray
        K × K × T array of outer-product data matrices. Typically
        ``data[:, :, t] = r_t @ r_t.T`` where ``r_t`` is the K-vector
        of returns at time *t*.
    m : int
        Number of innovation (A-matrix) lags in the RARCH recursion.
    n : int
        Number of persistence (B-matrix) lags in the RARCH recursion.
    R : numpy.ndarray
        K × K unconditional correlation matrix **or** K(K-1)/2 vector of
        correlation parameters (overwritten when ``stage <= 2`` or
        ``is_joint`` is True).
    back_cast : numpy.ndarray
        K × K back-cast matrix used to initialise the recursion for
        periods where lagged values are not yet available.
    stage : int
        Estimation stage indicator:

        * 1 — Joint estimation (GARCH + correlation intercept + dynamics)
        * 2 — Intercept and dynamics (GARCH parameters held fixed)
        * 3 — Dynamics only (intercept and GARCH held fixed)

    type_model : int
        RARCH model type:

        * 1 — Scalar
        * 2 — Common Persistence (CP)
        * 3 — Diagonal

    composite : int
        Composite likelihood mode:

        * 0 — Standard (full) multivariate normal likelihood
        * 1 — Composite with adjacent bivariate pairs
        * 2 — Composite with all bivariate pairs

    is_joint : bool
        If True, the parameter vector includes GARCH, intercept, and
        dynamics sub-vectors regardless of ``stage``.
    is_inference : bool
        If True, the correlation intercept parameters are interpreted as
        direct correlation elements (via :func:`corr_ivech`).  If False,
        they are interpreted as unconstrained z-values (via :func:`z2r`).
    r_scale : numpy.ndarray
        K-vector of correlation scaling factors.  The unconditional
        correlation is element-wise multiplied by
        ``sqrt(outer(r_scale, r_scale))`` before computing matrix
        square roots.
    univariate : list[dict]
        List of K per-series GARCH estimation dictionaries, each
        containing at minimum the keys ``'p'``, ``'o'``, ``'q'``, ``'m'``,
        ``'T'``, ``'fdata'``, ``'fIdata'``, ``'back_cast'``,
        ``'tarch_type'``.  Used by :func:`dcc_reconstruct_variance`
        when computing conditional variances (``stage == 1`` or
        ``is_joint``).

    Returns
    -------
    ll : float
        Total (negative) log-likelihood, i.e. ``sum(lls)``.
    lls : numpy.ndarray
        1-D array of length T containing per-period (negative)
        log-likelihood contributions.
    Rt : numpy.ndarray
        K × K × T array of time-varying conditional correlation matrices.

    Notes
    -----
    This is a faithful line-by-line migration of ``rcc_likelihood.m``.
    Every index adjustment and MATLAB→Python translation decision is
    documented with inline comments referencing the original MATLAB
    source line numbers.

    The MATLAB function signature is:

        ``[ll, lls, Rt] = rcc_likelihood(parameters, data, m, n, R,``
        ``backCast, stage, type, composite, isJoint, isInference,``
        ``rScale, univariate)``

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.rcc_likelihood import rcc_likelihood
    >>> # Minimal example with 2 assets and 10 periods
    >>> k, T = 2, 10
    >>> data = np.random.randn(k, k, T)
    >>> R = np.eye(k)
    >>> back_cast = np.eye(k) * 0.5
    >>> r_scale = np.ones(k)
    >>> params = np.array([0.3, 0.6])  # Scalar: 1 A param + 1 B param
    >>> ll, lls, Rt = rcc_likelihood(
    ...     params, data, 1, 1, R, back_cast,
    ...     stage=3, type_model=1, composite=0,
    ...     is_joint=False, is_inference=True,
    ...     r_scale=r_scale, univariate=[])
    """
    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    data = np.asarray(data, dtype=np.float64)
    back_cast = np.asarray(back_cast, dtype=np.float64)
    r_scale = np.asarray(r_scale, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Dimensions — Ref: rcc_likelihood.m:9
    # MATLAB: [k,~,T] = size(data);
    # ------------------------------------------------------------------
    k = data.shape[0]
    T = data.shape[2]
    offset = 0

    # ------------------------------------------------------------------
    # Phase 1: Parse GARCH parameters (stage==1 or isJoint)
    # Ref: rcc_likelihood.m:12-23
    # ------------------------------------------------------------------
    compute_vol = False
    garch_parameters = np.empty(0, dtype=np.float64)

    if stage == 1 or is_joint:
        # Ref: rcc_likelihood.m:13-17 — count total GARCH params across
        # all K series.  MATLAB 1-based loop for t=1:k → Python 0-based.
        count = 0
        for i in range(k):
            u = univariate[i]
            count += u['p'] + u['o'] + u['q'] + 1
        # Ref: rcc_likelihood.m:18 — garchParameters = parameters(1:count)
        garch_parameters = parameters[:count]
        offset += count
        compute_vol = True

    # ------------------------------------------------------------------
    # Phase 2: Parse correlation intercept (stage<=2 or isJoint)
    # Ref: rcc_likelihood.m:24-33
    # ------------------------------------------------------------------
    if stage <= 2 or is_joint:
        # Ref: rcc_likelihood.m:25 — count = k*(k-1)/2
        count = k * (k - 1) // 2
        # Ref: rcc_likelihood.m:26 — R = parameters(offset + (1:count))
        R = parameters[offset:offset + count]
        offset += count
        # Ref: rcc_likelihood.m:28-31 — transform to correlation matrix
        if is_inference:
            # Ref: rcc_likelihood.m:29 — R = corr_ivech(R)
            R = corr_ivech(R)
        else:
            # Ref: rcc_likelihood.m:31 — R = z2r(R)
            R = z2r(R)
    else:
        # Ensure R is a proper numpy array when passed through
        R = np.asarray(R, dtype=np.float64)

    # ------------------------------------------------------------------
    # Phase 3: Unpack dynamics parameters via rarch_parameter_transform
    # Ref: rcc_likelihood.m:35
    # MATLAB: [R,A,B] = rarch_parameter_transform(
    #             parameters(offset+(1:length(parameters)-offset)),
    #             m, n, k, R, type, false, false)
    # ------------------------------------------------------------------
    remaining_params = parameters[offset:]
    R, A, B = rarch_parameter_transform(
        remaining_params, m, n, k, R, type_model, False, False
    )

    # Ref: rcc_likelihood.m:37 — B(B<0) = 0;  (clamp negatives for CP)
    B = np.maximum(B, 0.0)

    # Ensure B is always 3-D for consistent indexing in the recursion.
    # For the CP model (type_model==2, q>0), rarch_parameter_transform
    # returns B as a 2-D (k, k) matrix.  We promote it to (k, k, 1).
    if B.ndim == 2:
        B = B[:, :, np.newaxis]

    # ------------------------------------------------------------------
    # Phase 4: Compute per-series conditional volatilities
    # Ref: rcc_likelihood.m:40-50
    # ------------------------------------------------------------------
    H = np.ones((T, k))
    if compute_vol:
        # Ref: rcc_likelihood.m:42 — H = dcc_reconstruct_variance(...)
        H = dcc_reconstruct_variance(garch_parameters, univariate)
        # Ref: rcc_likelihood.m:43 — stdData = zeros(k,k,T)
        std_data = np.zeros((k, k, T))
        for t in range(T):
            # Ref: rcc_likelihood.m:45 — h = sqrt(H(t,:))
            # MATLAB 1-based H(t,:) → Python 0-based H[t, :]
            h = np.sqrt(H[t, :])
            # Ref: rcc_likelihood.m:46 — stdData(:,:,t) = data(:,:,t)./(h'*h)
            # MATLAB h'*h is outer product (column * row) → np.outer(h, h)
            std_data[:, :, t] = data[:, :, t] / np.outer(h, h)
    else:
        # Ref: rcc_likelihood.m:49 — stdData = data
        std_data = data.copy()

    # ------------------------------------------------------------------
    # Phase 5: Initialise recursion arrays
    # Ref: rcc_likelihood.m:53-55
    # ------------------------------------------------------------------
    # Allocate T+1 slices for Qt because the loop writes to Qt[:,:,t+1]
    # at every iteration (MATLAB dynamically extends the array).
    Qt = np.zeros((k, k, T + 1))
    e = np.zeros((k, k, T))
    lls = np.zeros(T)

    # ------------------------------------------------------------------
    # Phase 6: Scale R and compute matrix square roots
    # Ref: rcc_likelihood.m:57-59
    # ------------------------------------------------------------------
    # Ref: rcc_likelihood.m:57 — R = R .* sqrt(rScale*rScale')
    R = R * np.sqrt(np.outer(r_scale, r_scale))

    # Ref: rcc_likelihood.m:58 — R12 = R^(0.5)
    # scipy.linalg.sqrtm computes the principal matrix square root.
    # .real handles negligible imaginary parts from numerical noise.
    R12 = scipy.linalg.sqrtm(R)
    if np.iscomplex(R12).any():
        R12 = R12.real
    R12 = np.asarray(R12, dtype=np.float64)

    # Ref: rcc_likelihood.m:59 — Rm12 = R^(-0.5) = inv(R^(0.5))
    Rm12 = np.linalg.inv(R12)

    # ------------------------------------------------------------------
    # Phase 7: Compute intercept and clamp diagonal
    # Ref: rcc_likelihood.m:60-63
    # ------------------------------------------------------------------
    # Ref: rcc_likelihood.m:60 — intercept = eye(k) - sum(A.^2,3) - sum(B.^2,3)
    intercept = np.eye(k) - np.sum(A ** 2, axis=2) - np.sum(B ** 2, axis=2)

    # Ref: rcc_likelihood.m:61-63 — clamp small diagonal entries
    d_int = np.diag(intercept).copy()
    d_int[d_int < 0.000001] = 0.000001
    intercept = np.diag(d_int)

    # ------------------------------------------------------------------
    # Phase 8: Set up composite likelihood indices / constant
    # Ref: rcc_likelihood.m:66-73
    # ------------------------------------------------------------------
    likconst = 0.0
    indices = np.empty((0, 2), dtype=np.int64)

    if composite == 0:
        # Ref: rcc_likelihood.m:67 — likconst = k*log(2*pi)
        likconst = k * np.log(2.0 * np.pi)
    elif composite == 1:
        # Ref: rcc_likelihood.m:69 — indices = [(1:k-1)' (2:k)']
        # MATLAB 1-based adjacent pairs → Python 0-based adjacent pairs
        indices = np.column_stack([
            np.arange(k - 1, dtype=np.int64),
            np.arange(1, k, dtype=np.int64),
        ])
    elif composite == 2:
        # Ref: rcc_likelihood.m:71-72 — [i,j] = meshgrid(1:k);
        #   indices = [i(~triu(true(k))) j(~triu(true(k)))];
        # MATLAB column-major lower-triangle extraction from meshgrid
        # produces pairs (i,j) with i < j (all upper-triangular pairs in
        # 0-based terms).  np.triu_indices(k, k=1) gives the same set
        # in row-major order.
        rows, cols = np.triu_indices(k, k=1)
        indices = np.column_stack([rows, cols]).astype(np.int64)

    # ------------------------------------------------------------------
    # Phase 9: Main recursion loop
    # Ref: rcc_likelihood.m:76-106
    # ------------------------------------------------------------------
    Rt = np.zeros((k, k, T))

    for t in range(T):
        # ----- Rotated innovations -----
        # Ref: rcc_likelihood.m:77 — e(:,:,t) = Rm12 * stdData(:,:,t) * Rm12
        # MATLAB 1-based t → Python 0-based t (same index into 0-based arrays)
        e[:, :, t] = Rm12 @ std_data[:, :, t] @ Rm12

        # ----- Pre-load next period intercept -----
        # Ref: rcc_likelihood.m:78 — Qt(:,:,t+1) = intercept
        # MATLAB dynamically extends Qt; Python pre-allocated T+1 slices.
        Qt[:, :, t + 1] = intercept

        # ----- AR (innovation) terms -----
        # Ref: rcc_likelihood.m:79-85
        # MATLAB: for j=1:m, if (t-j)>0 ...
        # Python: j in range(m) corresponds to MATLAB j = j_py+1.
        # Condition: MATLAB (t-j)>0 → Python (t_py+1)-(j_py+1) > 0 → t_py > j_py
        # Past index: MATLAB e(:,:,t-j) [1-based] → Python e[:,:, t-j-1] [0-based]
        for j in range(m):
            if t > j:
                # Ref: rcc_likelihood.m:81
                Qt[:, :, t] += A[:, :, j] @ e[:, :, t - j - 1] @ A[:, :, j]
            else:
                # Ref: rcc_likelihood.m:83
                Qt[:, :, t] += A[:, :, j] @ back_cast @ A[:, :, j]

        # ----- MA (persistence) terms -----
        # Ref: rcc_likelihood.m:86-91
        # Same index mapping as above, using Qt instead of e.
        for j in range(n):
            if t > j:
                # Ref: rcc_likelihood.m:88
                Qt[:, :, t] += B[:, :, j] @ Qt[:, :, t - j - 1] @ B[:, :, j]
            else:
                # Ref: rcc_likelihood.m:90
                Qt[:, :, t] += B[:, :, j] @ back_cast @ B[:, :, j]

        # ----- Correlation rescaling -----
        # Ref: rcc_likelihood.m:93 — R = R12*Qt(:,:,t)*R12
        R_t = R12 @ Qt[:, :, t] @ R12

        # Ref: rcc_likelihood.m:94 — rr = sqrt(diag(R)*diag(R)')
        d = np.diag(R_t)
        rr = np.sqrt(np.outer(d, d))

        # Ref: rcc_likelihood.m:95 — R = R ./ rr
        R_t = R_t / rr

        # Ref: rcc_likelihood.m:96 — Rt(:,:,t) = R
        Rt[:, :, t] = R_t

        # ----- Covariance matrix V -----
        # Ref: rcc_likelihood.m:97-100
        h = np.sqrt(H[t, :])
        # Ref: rcc_likelihood.m:98 — hh = h'*h (outer product)
        hh = np.outer(h, h)
        # Ref: rcc_likelihood.m:99 — V = R.*hh (element-wise multiply)
        V = R_t * hh
        # Ref: rcc_likelihood.m:100 — V = (V + V')/2 (force symmetry)
        V = (V + V.T) / 2.0

        # ----- Per-period log-likelihood -----
        if composite == 0:
            # Ref: rcc_likelihood.m:102
            # MATLAB: lls(t) = 0.5*(likconst + log(det(V))
            #                       + sum(diag(V^(-1)*data(:,:,t))))
            # Use np.linalg.slogdet for numerical stability instead of
            # log(det(V)) — Ref: rcc_likelihood.m:102 log(det(V))
            _, logdet_val = np.linalg.slogdet(V)
            # V^(-1) * data(:,:,t) via np.linalg.solve for stability
            inv_V_data = np.linalg.solve(V, data[:, :, t])
            lls[t] = 0.5 * (
                likconst + logdet_val + np.sum(np.diag(inv_V_data))
            )
        elif composite > 0:
            # Ref: rcc_likelihood.m:104
            # composite_likelihood expects 0-based indices (Python version)
            lls[t] = composite_likelihood(V, data[:, :, t], indices)

    # ------------------------------------------------------------------
    # Phase 10: Total log-likelihood
    # Ref: rcc_likelihood.m:107 — ll = sum(lls)
    # ------------------------------------------------------------------
    ll = float(np.sum(lls))

    return ll, lls, Rt
