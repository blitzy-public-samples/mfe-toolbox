"""
Log-likelihood evaluation for the RARCH(p,q) multivariate volatility model.

Computes the negative Gaussian log-likelihood for the Rotated ARCH model of
Noureldin, Shephard, and Sheppard, operating in a rotated covariance space.
The recursion for the time-varying conditional covariance in the rotated space
follows:

    G_t = Omega + sum_{j=1}^{p} A_j @ OP_{t-j} @ A_j + sum_{j=1}^{q} B_j @ G_{t-j} @ B_j

where:
    * Omega = I_k - sum(A_j^2) - sum(B_j^2)  (intercept)
    * OP_t = C^{-1/2} @ data_t @ C^{-1/2}     (rotated outer product)
    * C = unconditional covariance matrix
    * The conditional covariance in the original space is H_t = C^{1/2} @ G_t @ C^{1/2}

The model supports three parameterisation types:

    * **Scalar** (type_model=1) — A_j = a_j * I_k, B_j = b_j * I_k
    * **Common Persistence / CP** (type_model=2) — diagonal A, with B derived
      from a common persistence parameter theta
    * **Diagonal** (type_model=3) — fully diagonal A and B

Source Reference
----------------
Migrated from ``multivariate/rarch_likelihood.m`` (84 lines).
Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 1, Date: 3/27/2012

See Also
--------
mfe_toolbox.multivariate.rarch : RARCH model driver
mfe_toolbox.multivariate.rarch_parameter_transform : Parameter vector to matrix transform
mfe_toolbox.multivariate.rarch_constraint : RARCH nonlinear constraints
mfe_toolbox.multivariate.rarch_simulate : RARCH simulation
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

from mfe_toolbox.multivariate.rarch_parameter_transform import rarch_parameter_transform


def rarch_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    p: int,
    q: int,
    C: np.ndarray,
    back_cast: np.ndarray,
    type_model: int,
    is_joint: bool = False,
    compute_scores: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the negative log-likelihood for a RARCH(p,q) model.

    Computes the Gaussian negative log-likelihood in the rotated covariance
    space, where the unconditional covariance C is used for the rotation
    C^{1/2} / C^{-1/2}.  Returns the total negative log-likelihood (suitable
    for minimisation), per-observation contributions, and the full path of
    conditional covariance matrices in the original space.

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D vector of model parameters.  Layout depends on ``is_joint`` and
        ``type_model``; see :func:`rarch_parameter_transform` for details.
    data : numpy.ndarray
        K × K × T array of outer-product matrices for T observations of K
        assets.  Typically ``data[:, :, t] = r_t @ r_t.T`` where ``r_t`` is
        the K × 1 return vector at time *t*.
    p : int
        Positive integer — number of symmetric innovation lags (A matrices).
    q : int
        Non-negative integer — number of conditional covariance lags
        (B matrices).
    C : numpy.ndarray
        K × K unconditional covariance matrix of the data.  Overridden by
        the parameter vector when ``is_joint`` is True.
    back_cast : numpy.ndarray
        K × K matrix used for pre-sample values in the recursion.  Typically
        the sample-average outer product in the rotated space.
    type_model : int
        Model type indicator:

        * 1 — Scalar
        * 2 — Common Persistence (CP)
        * 3 — Diagonal
    is_joint : bool, optional
        If True the parameter vector contains both the covariance intercept
        and dynamics parameters.  Default is False.
    compute_scores : bool, optional
        Passed through to :func:`rarch_parameter_transform` as the Cholesky
        parameterisation flag (``is_c_chol``).  When True **and**
        ``is_joint`` is True, the intercept portion of the parameter vector
        is interpreted as a Cholesky factor of C.  Default is False.

        .. note::
           This parameter corresponds to ``isCChol`` in the original MATLAB
           source (``rarch_likelihood.m:1``).

    Returns
    -------
    ll : float
        Total negative log-likelihood evaluated at ``parameters``.  Set to
        1e7 when the result is NaN, infinite, or complex — a large penalty
        value that steers the optimiser away from invalid parameter regions.
    lls : numpy.ndarray
        T-element 1-D array of per-observation negative log-likelihood
        contributions.
    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices in the original
        (non-rotated) space: ``Ht[:, :, t] = C^{1/2} @ G_t @ C^{1/2}``.

    Notes
    -----
    The negative log-likelihood contribution for observation *t* under the
    multivariate Gaussian is:

    .. math::

        \\ell_t = \\tfrac{1}{2}\\bigl[
            k \\ln(2\\pi)
            + \\ln|H_t|
            + \\operatorname{tr}(H_t^{-1} \\, \\text{data}_t)
        \\bigr]

    The function clamps the diagonal of the intercept matrix to a minimum of
    1e-6 and negative elements of B to zero, matching the original MATLAB
    implementation.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.rarch_likelihood import rarch_likelihood
    >>> k, T = 2, 50
    >>> data = np.zeros((k, k, T))
    >>> for t in range(T):
    ...     r = np.random.randn(k, 1)
    ...     data[:, :, t] = r @ r.T
    >>> C = np.eye(k)
    >>> back_cast = np.eye(k)
    >>> params = np.array([0.2, 0.7])  # scalar a, b
    >>> ll, lls, Ht = rarch_likelihood(params, data, 1, 1, C, back_cast, 1)
    >>> ll > 0  # negative log-likelihood is positive
    True
    """
    # ------------------------------------------------------------------
    # Validate and extract dimensions
    # Ref: rarch_likelihood.m:35-36
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    data = np.asarray(data, dtype=np.float64)

    # Ref: rarch_likelihood.m:35 — T = size(data,3)
    T: int = data.shape[2]
    # Ref: rarch_likelihood.m:36 — k = size(data,2)
    k: int = data.shape[1]

    # ------------------------------------------------------------------
    # Transform parameter vector into C, A, B matrices
    # Ref: rarch_likelihood.m:38
    # The 8th argument maps MATLAB's isCChol to compute_scores
    # ------------------------------------------------------------------
    C_mat, A, B = rarch_parameter_transform(
        parameters, p, q, k, C, type_model, is_joint, compute_scores
    )

    # ------------------------------------------------------------------
    # Clamp negative B values to zero
    # Ref: rarch_likelihood.m:40 — B(B<0) = 0
    # This is necessary for the CP model where reparameterisation can
    # produce small negative values due to floating-point arithmetic.
    # ------------------------------------------------------------------
    B = np.maximum(B, 0.0)

    # ------------------------------------------------------------------
    # Normalise B to 3-D for uniform loop handling
    # For CP model (type_model=2, q>0), rarch_parameter_transform returns
    # B as a 2-D (k, k) matrix.  We reshape to (k, k, 1) so that the
    # recursion loop can index B[:, :, j] uniformly.
    # ------------------------------------------------------------------
    if B.ndim == 2:
        B = B[:, :, np.newaxis]

    # ------------------------------------------------------------------
    # Allocate working arrays
    # Ref: rarch_likelihood.m:42-44
    # Gt needs T+1 slices because the MATLAB code sets Gt(:,:,i+1) at
    # each iteration to prepare the intercept for the next step.
    # ------------------------------------------------------------------
    Gt = np.zeros((k, k, T + 1))
    e = np.zeros((k, k, T))
    lls = np.zeros(T)
    Ht = np.zeros((k, k, T))

    # ------------------------------------------------------------------
    # Compute matrix square root and its inverse for rotation
    # Ref: rarch_likelihood.m:45-46
    # MATLAB: C12 = C^(0.5); Cm12 = C^(-0.5);
    # Uses scipy.linalg.sqrtm for the principal matrix square root.
    # np.real() strips tiny imaginary parts that can arise from numerical
    # imprecision in sqrtm even for positive-definite input.
    # ------------------------------------------------------------------
    C12 = np.real(sqrtm(C_mat))
    # Ref: rarch_likelihood.m:46 — C^(-0.5) = inv(C^(0.5))
    Cm12 = np.linalg.inv(C12)

    # ------------------------------------------------------------------
    # Compute intercept matrix: Omega = I - sum(A_j^2) - sum(B_j^2)
    # Ref: rarch_likelihood.m:47 — intercept = eye(k) - sum(A.^2,3) - sum(B.^2,3)
    # A is (k, k, p), B is (k, k, q_eff) after normalisation
    # ------------------------------------------------------------------
    intercept = np.eye(k) - np.sum(A ** 2, axis=2) - np.sum(B ** 2, axis=2)

    # ------------------------------------------------------------------
    # Clamp small diagonal values of the intercept to maintain positive
    # definiteness of the implied unconditional covariance.
    # Ref: rarch_likelihood.m:48-50
    # MATLAB: dint = diag(intercept); dint(dint<.000001)=.000001;
    #         intercept = diag(dint);
    # ------------------------------------------------------------------
    dint = np.diag(intercept).copy()
    dint[dint < 1e-6] = 1e-6
    intercept = np.diag(dint)

    # ------------------------------------------------------------------
    # Log-likelihood normalisation constant
    # Ref: rarch_likelihood.m:51 — logLikConst = k*log(2*pi)
    # ------------------------------------------------------------------
    log_lik_const: float = float(k * np.log(2.0 * np.pi))

    # ------------------------------------------------------------------
    # Main recursion loop
    # Ref: rarch_likelihood.m:53-72
    #
    # For each observation t = 0, …, T-1 (Python 0-based):
    #   1. Rotate the outer product into the RARCH space
    #   2. Initialise next-step Gt to the intercept
    #   3. Accumulate A_j contributions (innovation terms)
    #   4. Accumulate B_j contributions (smoothing terms)
    #   5. Rotate back to original space and evaluate likelihood
    #
    # Index mapping (MATLAB i → Python t):
    #   MATLAB i = 1..T  ↔  Python t = 0..T-1
    #   MATLAB (i-j)<=0  ↔  Python (t+1-j)<=0  i.e. t < j
    #   MATLAB e(:,:,i-j) (1-based) ↔ Python e[:,:,t-j] (0-based, when t>=j)
    #   MATLAB Gt(:,:,i-j)          ↔ Python Gt[:,:,t-j]
    # ------------------------------------------------------------------
    for t in range(T):
        # Ref: rarch_likelihood.m:54 — e(:,:,i) = Cm12 * data(:,:,i) * Cm12
        e[:, :, t] = Cm12 @ data[:, :, t] @ Cm12

        # Ref: rarch_likelihood.m:55 — Gt(:,:,i+1) = intercept
        # Prepare intercept for the *next* time step
        Gt[:, :, t + 1] = intercept

        # --- Innovation terms (A lags) ---
        # Ref: rarch_likelihood.m:56-62
        for j in range(1, p + 1):
            Aj = A[:, :, j - 1]
            # Ref: rarch_likelihood.m:57 — if (i-j)<=0
            if t + 1 - j <= 0:
                # Pre-sample: use back_cast
                # Ref: rarch_likelihood.m:58
                Gt[:, :, t] += Aj @ back_cast @ Aj
            else:
                # In-sample: use lagged rotated outer product
                # Ref: rarch_likelihood.m:60 — e(:,:,i-j) 1-based → e[:,:,t-j] 0-based
                Gt[:, :, t] += Aj @ e[:, :, t - j] @ Aj

        # --- Smoothing terms (B lags) ---
        # Ref: rarch_likelihood.m:63-69
        for j in range(1, q + 1):
            Bj = B[:, :, j - 1]
            # Ref: rarch_likelihood.m:64 — if (i-j)<=0
            if t + 1 - j <= 0:
                # Pre-sample: use back_cast
                # Ref: rarch_likelihood.m:65
                Gt[:, :, t] += Bj @ back_cast @ Bj
            else:
                # In-sample: use lagged conditional covariance (rotated)
                # Ref: rarch_likelihood.m:67 — Gt(:,:,i-j) 1-based → Gt[:,:,t-j] 0-based
                Gt[:, :, t] += Bj @ Gt[:, :, t - j] @ Bj

        # --- Compute conditional covariance in original space ---
        # Ref: rarch_likelihood.m:70 — V = C12*Gt(:,:,i)*C12
        V = C12 @ Gt[:, :, t] @ C12
        Ht[:, :, t] = V

        # --- Per-observation negative log-likelihood ---
        # Ref: rarch_likelihood.m:71
        # MATLAB: lls(i) = 0.5*(logLikConst + log(det(V)) + sum(diag(V^(-1)*data(:,:,i))))
        # Using np.linalg.slogdet for numerical stability (avoids det overflow)
        sign, logdet_V = np.linalg.slogdet(V)
        # Using np.linalg.solve instead of inv for stability:
        # trace(V^{-1} @ data_t) = trace(solve(V, data_t))
        V_inv_data = np.linalg.solve(V, data[:, :, t])
        lls[t] = 0.5 * (log_lik_const + logdet_V + np.trace(V_inv_data))

    # ------------------------------------------------------------------
    # Total negative log-likelihood
    # Ref: rarch_likelihood.m:73 — ll = sum(lls)
    # ------------------------------------------------------------------
    ll: float = float(np.sum(lls))

    # ------------------------------------------------------------------
    # Penalty for invalid parameters
    # Ref: rarch_likelihood.m:75-77
    # if isnan(ll) || isinf(ll) || ~isreal(ll) → ll = 1e7
    # ------------------------------------------------------------------
    if np.isnan(ll) or np.isinf(ll) or not np.isreal(ll):
        ll = 1e7

    return ll, lls, Ht
