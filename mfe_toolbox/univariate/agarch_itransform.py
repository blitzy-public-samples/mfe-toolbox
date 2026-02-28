"""
AGARCH(P,Q) inverse parameter transformation.

Migrated from univariate/agarch_itransform.m (121 lines, 4062 bytes).
Maps parameters from the unconstrained real line back to constrained parameter space
appropriate for AGARCH/NAGARCH model estimation. This is the inverse of agarch_transform.

The transformation ensures:
  (1) omega > 0
  (2) alpha(i) >= 0 for i = 1,...,p
  (3) beta(i)  >= 0 for i = 1,...,q
  (4) -q(.01, epsilon) < gamma < q(.99, epsilon) for AGARCH
  (5) sum(alpha(i) + beta(k)) < 1 for AGARCH;
      sum(alpha(i)*(1+gamma^2) + beta(k)) < 1 for NAGARCH
  (6) nu > 2 for Student's t; nu > 1 for GED
  (7) -0.99 < lambda < 0.99 for Skewed-t

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/12/2009
"""

import numpy as np


def agarch_itransform(
    parameters: np.ndarray,
    p: int,
    q: int,
    model_type: int,
    error_type: int,
    transform_bounds: np.ndarray = None,
) -> tuple:
    """
    AGARCH(P,Q) inverse parameter transformation.

    Maps parameters from the unconstrained real line to constrained space for
    AGARCH/NAGARCH model estimation. This is the mathematical inverse of
    agarch_transform.

    Parameters
    ----------
    parameters : np.ndarray
        Column parameter vector in unconstrained space. Length depends on model
        configuration:
        - Base length: 2 + p + q  (omega, alpha(1..p), gamma, beta(1..q))
        - error_type 2 or 3: base + 1  (adds nu)
        - error_type 4: base + 2  (adds nu and lambda)
    p : int
        Positive scalar integer representing the number of symmetric
        innovations (ARCH terms).
    q : int
        Non-negative scalar integer representing the number of lags of
        conditional variance (GARCH terms). Use 0 for pure ARCH.
    model_type : int
        Type of variance process:
          1 - AGARCH  (Asymmetric GARCH)
          2 - NAGARCH (Nonlinear Asymmetric GARCH)
    error_type : int
        Distribution type for innovations:
          1 - Gaussian innovations
          2 - Student's t-distributed errors
          3 - Generalized Error Distribution (GED)
          4 - Skewed Student's t distribution
    transform_bounds : np.ndarray, optional
        2-element array containing the 0.01 and 0.99 quantiles of epsilon
        for the gamma parameter transformation. Required when model_type == 1
        (AGARCH). Ignored when model_type == 2 (NAGARCH).

    Returns
    -------
    tuple of (trans_parameters, nu, lambda_param)
        trans_parameters : np.ndarray
            Constrained parameter vector of length 2 + p + q:
            [omega, alpha(1), ..., alpha(p), gamma, beta(1), ..., beta(q)]
        nu : float or None
            Distribution shape parameter. For Student's t / Skewed-t, this is
            the degrees of freedom (> 2.01). For GED, this is the shape
            parameter (1.01 to 50.01). None if error_type == 1 (Gaussian).
        lambda_param : float or None
            Distribution asymmetry parameter for Skewed-t, ranging from
            -0.99 to 0.99. None if error_type != 4.

    Raises
    ------
    ValueError
        If model_type == 1 and transform_bounds is None or has incorrect shape.
    ValueError
        If p < 1, q < 0, model_type not in {1, 2}, or error_type not in {1, 2, 3, 4}.

    Notes
    -----
    The cascading logistic transformation ensures the stationarity constraint
    by allocating a diminishing budget. Each successive alpha/beta parameter
    is mapped to the interval [0, remaining_scale], where remaining_scale
    starts at 0.9998 and decreases as parameters are allocated.

    For NAGARCH (model_type == 2), the alpha parameters are divided by
    (1 + gamma^2) after the cascading transform, ensuring the NAGARCH-specific
    stationarity condition sum(alpha_i * (1 + gamma^2) + beta_j) < 1 is
    maintained while preserving the cascading budget calculation.

    See Also
    --------
    agarch_transform : Forward parameter transformation (constrained -> unconstrained)
    agarch : Main AGARCH/NAGARCH estimation driver
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if p < 1:
        raise ValueError(
            f"p must be a positive integer, got {p}."
        )
    if q < 0:
        raise ValueError(
            f"q must be a non-negative integer, got {q}."
        )
    if model_type not in (1, 2):
        raise ValueError(
            f"model_type must be 1 (AGARCH) or 2 (NAGARCH), got {model_type}."
        )
    if error_type not in (1, 2, 3, 4):
        raise ValueError(
            f"error_type must be 1, 2, 3, or 4, got {error_type}."
        )
    if model_type == 1 and transform_bounds is None:
        raise ValueError(
            "transform_bounds is required when model_type == 1 (AGARCH). "
            "Provide a 2-element array with [lower_quantile, upper_quantile]."
        )
    if model_type == 1 and transform_bounds is not None:
        transform_bounds = np.array(transform_bounds, dtype=np.float64).ravel()
        if transform_bounds.shape[0] < 2:
            raise ValueError(
                "transform_bounds must have at least 2 elements "
                "[lower_quantile, upper_quantile]."
            )

    # Determine expected parameter vector length
    # Ref: agarch_itransform.m — Base: omega(1) + alpha(p) + gamma(1) + beta(q)
    expected_len = 2 + p + q
    if error_type in (2, 3, 4):
        expected_len += 1  # nu parameter
    if error_type == 4:
        expected_len += 1  # lambda parameter
    parameters_input = np.array(parameters, dtype=np.float64).ravel()
    if parameters_input.shape[0] < expected_len:
        raise ValueError(
            f"parameters vector length {parameters_input.shape[0]} is shorter "
            f"than expected {expected_len} for p={p}, q={q}, "
            f"error_type={error_type}."
        )

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:50-51 — Initialize nu and lambda as empty
    # In MATLAB these are [], in Python we use None
    # ------------------------------------------------------------------
    nu = None
    lambda_param = None

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:54 — Upper constraint to prevent exp overflow
    # CRITICAL: Cap ALL unconstrained parameters at 100 before any
    # exponential transforms. exp(100) ~ 2.69e43, safely below float64 max.
    # ------------------------------------------------------------------
    parameters_work = np.copy(parameters_input)
    parameters_work = np.minimum(parameters_work, 100.0)

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:57-60 — Parse parameters from vector
    # MATLAB 1-based indexing translated to Python 0-based indexing
    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:57 — omega = parameters(1)
    omega = parameters_work[0]
    # Ref: agarch_itransform.m:58 — alpha = parameters(2:p+1)  [p elements]
    alpha = parameters_work[1:p + 1].copy()
    # Ref: agarch_itransform.m:59 — gamma = parameters(p+2)
    gamma = parameters_work[p + 1]
    # Ref: agarch_itransform.m:60 — beta = parameters(p+3:p+q+2)  [q elements]
    beta = parameters_work[p + 2:p + q + 2].copy()

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:63-75 — Handle the transformation of nu
    # ------------------------------------------------------------------
    if error_type == 2 or error_type == 4:
        # Ref: agarch_itransform.m:65-66 — Student-t or Skewed-t
        # nu = 2.01 + x^2 ensures nu > 2.01 (finite variance requirement)
        # This is the inverse of the forward transform: x = sqrt(nu - 2.01)
        nu_raw = parameters_work[p + q + 2]
        nu = 2.01 + nu_raw ** 2
    elif error_type == 3:
        # Ref: agarch_itransform.m:68-74 — GED shape parameter
        # Logistic transform maps (-inf, inf) -> (0, 1), then scaled to (1.01, 50.01)
        # This is the inverse of: x = log((nu-1)/49 / (1 - (nu-1)/49))
        nu_raw = parameters_work[p + q + 2]
        # Ref: agarch_itransform.m:70-72 — Additional overflow guard for nu
        # Redundant with line 54 cap, but preserved for exact MATLAB parity
        if nu_raw > 100.0:
            nu_raw = 100.0
        nu = np.exp(nu_raw) / (1.0 + np.exp(nu_raw))
        nu = 49.0 * nu + 1.01

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:77-81 — Lambda for Skewed-t distribution
    # Logistic maps (-inf, inf) -> (0, 1), then scaled to (-0.99, 0.99)
    # ------------------------------------------------------------------
    if error_type == 4:
        lambda_raw = parameters_work[p + q + 3]
        lambda_param = np.exp(lambda_raw) / (1.0 + np.exp(lambda_raw))
        lambda_param = 1.98 * lambda_param - 0.99

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:84 — Simple exponential transform of omega
    # omega_constrained = exp(omega_unconstrained) ensures omega > 0
    # This is the inverse of: omega_unconstrained = log(omega_constrained)
    # ------------------------------------------------------------------
    tomega = np.exp(omega)

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:86-93 — Gamma parameter transform via logistic
    # First map to (0, 1) via standard logistic, then scale to target range
    # ------------------------------------------------------------------
    tgamma = np.exp(gamma) / (1.0 + np.exp(gamma))
    if model_type == 1:
        # Ref: agarch_itransform.m:88-90 — AGARCH: map to quantile-bounded range
        # gamma is bounded by the empirical quantiles of epsilon
        lower_quantile = transform_bounds[0]
        upper_quantile = transform_bounds[1]
        tgamma = tgamma * (upper_quantile - lower_quantile) + lower_quantile
    else:
        # Ref: agarch_itransform.m:92 — NAGARCH: map to symmetric range via sqrt(10)
        # Maps (0, 1) -> (-sqrt(10)/2, sqrt(10)/2) ~ (-1.58, 1.58)
        tgamma = (tgamma - 0.5) * np.sqrt(10.0)

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:97-98 — Initialize transformed alpha and beta
    # ------------------------------------------------------------------
    talpha = alpha.copy()
    tbeta = beta.copy()

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:101-102 — Upper bound for stationarity constraint
    # UB = 0.9998 keeps sum(alpha + beta) strictly below 1
    # ------------------------------------------------------------------
    UB = 0.9998
    scale = UB

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:104-109 — Cascading logistic transform for alpha
    # Each alpha(i) is mapped from (-inf, inf) to (0, remaining_scale)
    # via logistic, then the budget is reduced by the allocated amount
    # ------------------------------------------------------------------
    for i in range(p):
        # Ref: agarch_itransform.m:106 — Alpha is between 0 and scale
        talpha[i] = (np.exp(talpha[i]) / (1.0 + np.exp(talpha[i]))) * scale
        # Ref: agarch_itransform.m:108 — Update remaining budget
        scale = scale - talpha[i]

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:110-112 — NAGARCH gamma^2 adjustment
    # For NAGARCH, divide alpha by (1 + gamma^2) after the cascading loop.
    # The scale variable is NOT updated here, so the beta loop uses the
    # pre-adjustment budget. This ensures:
    #   sum(alpha_adjusted * (1+gamma^2)) + sum(beta) < UB
    # because sum(alpha_adjusted * (1+gamma^2)) = sum(alpha_pre_adjust),
    # and scale = UB - sum(alpha_pre_adjust), so sum(beta) < scale.
    # ------------------------------------------------------------------
    if model_type == 2:
        talpha = talpha / (1.0 + tgamma ** 2)

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:113-118 — Cascading logistic transform for beta
    # Same cascading pattern as alpha, using the remaining scale budget
    # ------------------------------------------------------------------
    for i in range(q):
        # Ref: agarch_itransform.m:115 — Beta is between 0 and scale
        tbeta[i] = (np.exp(tbeta[i]) / (1.0 + np.exp(tbeta[i]))) * scale
        # Ref: agarch_itransform.m:117 — Update remaining budget
        scale = scale - tbeta[i]

    # ------------------------------------------------------------------
    # Ref: agarch_itransform.m:121 — Reassemble constrained parameter vector
    # Output: [omega, alpha(1..p), gamma, beta(1..q)]
    # ------------------------------------------------------------------
    trans_parameters = np.concatenate(
        (np.array([tomega]), talpha, np.array([tgamma]), tbeta)
    )

    return trans_parameters, nu, lambda_param
