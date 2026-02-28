"""
AGARCH/NAGARCH forward parameter transformation.

Maps constrained AGARCH/NAGARCH model parameters to unconstrained real-line
space for use with scipy.optimize.minimize. This is the forward transform,
paired with agarch_itransform (the inverse transform).

The transform maps:
  - omega (>0) -> log(omega)  (ensures positivity on inverse)
  - alpha (>=0, stationarity) -> cascading logistic transform
  - gamma (bounded) -> logistic transform with model-specific bounds
  - beta (>=0, stationarity) -> cascading logistic transform
  - nu (distribution DOF) -> sqrt(nu-2.01) or logistic
  - lambda (skewness) -> logistic

Migrated from: univariate/agarch_transform.m (135 lines, 4252 bytes)
Original author: Kevin Sheppard, University of Oxford

References
----------
agarch_transform.m — MFE Toolbox Version 4.0
"""

import numpy as np


def agarch_transform(
    parameters: np.ndarray,
    p: int,
    q: int,
    model_type: int,
    error_type: int,
    transform_bounds: np.ndarray,
) -> tuple[np.ndarray, float | None, float | None]:
    """Transform constrained AGARCH/NAGARCH parameters to unconstrained space.

    Maps the constrained parameter vector used in AGARCH/NAGARCH models to an
    unconstrained real-line representation suitable for numerical optimizers.
    The forward transform is invertible — paired with ``agarch_itransform``.

    Parameters
    ----------
    parameters : np.ndarray
        Constrained parameter vector of length (2 + p + q + extra_dist_params).
        Layout: [omega, alpha(1)...alpha(p), gamma, beta(1)...beta(q),
        (nu), (lambda)].
        - omega > 0 (variance intercept)
        - alpha >= 0 (ARCH coefficients, stationarity constrained)
        - gamma (asymmetry parameter, model-specific bounds)
        - beta >= 0 (GARCH coefficients, stationarity constrained)
        - nu (distribution degrees of freedom, if error_type in {2,3,4})
        - lambda (distribution skewness, if error_type == 4)
    p : int
        Number of ARCH lags (alpha coefficients). Must be >= 1.
    q : int
        Number of GARCH lags (beta coefficients). Must be >= 0.
    model_type : int
        Model specification:
        1 = AGARCH (Asymmetric GARCH, Engle 1990):
            shock = (epsilon - gamma)^2
            gamma bounded by quantile-based transform_bounds
        2 = NAGARCH (Nonlinear Asymmetric GARCH, Engle & Ng 1993):
            shock = (epsilon - gamma * sqrt(h))^2
            gamma bounded by [-sqrt(10), sqrt(10)]
    error_type : int
        Error distribution type:
        1 = Normal (Gaussian)
        2 = Student's t (nu > 2.01)
        3 = GED (1 < nu < 50)
        4 = Skewed Student's t (nu > 2.01, -0.995 < lambda < 0.995)
    transform_bounds : np.ndarray
        Two-element array [lower_quantile, upper_quantile] used for
        gamma transform when model_type == 1 (AGARCH). Typically the
        0.01 and 0.99 quantiles of the residual epsilon series.

    Returns
    -------
    tuple[np.ndarray, float | None, float | None]
        (trans_parameters, nu, lam) where:
        - trans_parameters : np.ndarray — unconstrained parameter vector
          of length (2 + p + q), layout [tomega, talpha(1:p), tgamma, tbeta(1:q)]
        - nu : float or None — transformed distribution shape parameter
        - lam : float or None — transformed distribution skewness parameter

    Raises
    ------
    ValueError
        If parameters do not satisfy the stationarity/positivity constraints
        required for valid transformation.

    Notes
    -----
    The stationarity constraint enforced before transformation is:
        sum(alpha) * (1 + gamma^2) + sum(beta) < 0.999998

    For AGARCH (model_type=1), gamma is bounded by empirical quantiles of
    epsilon via transform_bounds. For NAGARCH (model_type=2), gamma is
    bounded by [-sqrt(10), sqrt(10)].

    All transforms use the logistic function log(x/(1-x)) to map bounded
    parameters to the unconstrained real line.

    MATLAB index adjustments are documented inline with "Ref:" comments.
    """

    # -------------------------------------------------------------------------
    # Phase 1: Distribution parameter transforms (MATLAB lines 49-68)
    # -------------------------------------------------------------------------
    # Ref: agarch_transform.m:49-50 — Initialize nu and lambda as empty
    nu: float | None = None
    lam: float | None = None  # 'lam' avoids Python reserved keyword 'lambda'

    if error_type == 2 or error_type == 4:
        # Ref: agarch_transform.m:54-56 — Student's t or Skewed-t nu transform
        # MATLAB index p+q+3 → Python index p+q+2 (0-based)
        nu = float(parameters[p + q + 2])
        nu = np.sqrt(nu - 2.01)
    elif error_type == 3:
        # Ref: agarch_transform.m:57-60 — GED nu logistic transform
        # MATLAB index p+q+3 → Python index p+q+2 (0-based)
        nu = float(parameters[p + q + 2])
        temp = (nu - 1.0) / 49.0
        nu = float(np.log(temp / (1.0 - temp)))

    if error_type == 4:
        # Ref: agarch_transform.m:64-67 — Skewed-t lambda logistic transform
        # MATLAB index p+q+4 → Python index p+q+3 (0-based)
        lam = float(parameters[p + q + 3])
        temp = (lam + 0.995) / 1.99
        lam = float(np.log(temp / (1.0 - temp)))

    # -------------------------------------------------------------------------
    # Phase 2: Parse core parameters (MATLAB lines 70-74)
    # -------------------------------------------------------------------------
    # Ref: agarch_transform.m:71 — MATLAB parameters(1) → Python parameters[0]
    omega = float(parameters[0])

    # Ref: agarch_transform.m:72 — MATLAB parameters(2:p+1) → Python [1:p+1]
    # .copy() to avoid mutating input array
    alpha = parameters[1:p + 1].copy().astype(np.float64)

    # Ref: agarch_transform.m:73 — MATLAB parameters(p+3:p+q+2) → Python [p+2:p+q+2]
    beta = parameters[p + 2:p + q + 2].copy().astype(np.float64)

    # Ref: agarch_transform.m:74 — MATLAB parameters(p+2) → Python parameters[p+1]
    gamma = float(parameters[p + 1])

    # -------------------------------------------------------------------------
    # Phase 3: Omega transform (MATLAB lines 76-77)
    # -------------------------------------------------------------------------
    # Ref: agarch_transform.m:77 — log transform ensures positivity on inverse
    tomega = float(np.log(omega))

    # -------------------------------------------------------------------------
    # Phase 4: Gamma transform (MATLAB lines 79-87)
    # -------------------------------------------------------------------------
    if model_type == 1:
        # Ref: agarch_transform.m:81-83 — AGARCH: quantile-bounded logistic
        # MATLAB transform_bounds(1) → Python transform_bounds[0]
        lower_quantile = float(transform_bounds[0])
        # MATLAB transform_bounds(2) → Python transform_bounds[1]
        upper_quantile = float(transform_bounds[1])
        tgamma = (gamma - lower_quantile) / (upper_quantile - lower_quantile)
    else:
        # Ref: agarch_transform.m:85 — NAGARCH: sqrt(10)-scaled logistic
        tgamma = (gamma + np.sqrt(10.0)) / (2.0 * np.sqrt(10.0))

    # Ref: agarch_transform.m:87 — Logistic transform for both model types
    tgamma = float(np.log(tgamma / (1.0 - tgamma)))

    # -------------------------------------------------------------------------
    # Phase 5: Stationarity check and clamping (MATLAB lines 92-103)
    # -------------------------------------------------------------------------
    # Ref: agarch_transform.m:93 — Upper stationarity bound
    UB = 0.999998

    # Ref: agarch_transform.m:96-98 — Constraint validation
    alpha_sum = float(np.sum(alpha))
    beta_sum = float(np.sum(beta))
    if (np.any(alpha < 0)
            or np.any(beta < 0)
            or (alpha_sum * (1.0 + gamma ** 2) + beta_sum) >= UB):
        raise ValueError(
            'These do not conform to the necessary set of restrictions '
            'to be transformed.'
        )

    # Ref: agarch_transform.m:102-103 — Clamp exact zeros to avoid log(0)
    alpha[alpha == 0] = 1e-8
    beta[beta == 0] = 1e-8

    # -------------------------------------------------------------------------
    # Phase 6: Alpha transform — cascading logistic (MATLAB lines 105-119)
    # -------------------------------------------------------------------------
    # Ref: agarch_transform.m:106 — Initialize scale
    scale = UB

    # Ref: agarch_transform.m:110 — MATLAB initializes talpha=alpha*(1+gamma^2)
    # but the loop on lines 111-119 overwrites each element using ORIGINAL alpha.
    # The enlargement on line 110 is effectively dead code.
    talpha = np.empty(p, dtype=np.float64)

    for i in range(p):
        # Ref: agarch_transform.m:114 — talpha(i)=alpha(i)./scale
        # Uses ORIGINAL alpha(i), not the enlarged talpha
        talpha[i] = alpha[i] / scale

        # Ref: agarch_transform.m:116 — Inverse logistic transform
        talpha[i] = np.log(talpha[i] / (1.0 - talpha[i]))

        # Ref: agarch_transform.m:118 — Update scale with ORIGINAL alpha
        scale = scale - alpha[i]

    # -------------------------------------------------------------------------
    # Phase 7: Beta transform — cascading logistic (MATLAB lines 121-131)
    # -------------------------------------------------------------------------
    if q > 0:
        tbeta = np.empty(q, dtype=np.float64)
        for i in range(q):
            # Ref: agarch_transform.m:126 — tbeta(i)=tbeta(i)./scale
            # Uses original beta value (tbeta was set to beta on line 122)
            tbeta[i] = beta[i] / scale

            # Ref: agarch_transform.m:128 — Inverse logistic transform
            tbeta[i] = np.log(tbeta[i] / (1.0 - tbeta[i]))

            # Ref: agarch_transform.m:130 — Update scale with original beta
            scale = scale - beta[i]
    else:
        # Edge case: q == 0 means no beta parameters
        tbeta = np.empty(0, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Phase 8: Concatenate output (MATLAB line 134)
    # -------------------------------------------------------------------------
    # Ref: agarch_transform.m:134 — trans_parameters=[tomega;talpha;tgamma;tbeta]
    trans_parameters = np.concatenate([
        np.array([tomega]),
        talpha,
        np.array([tgamma]),
        tbeta,
    ])

    return (trans_parameters, nu, lam)
