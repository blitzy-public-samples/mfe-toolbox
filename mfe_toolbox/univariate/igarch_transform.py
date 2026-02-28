"""
IGARCH(P,Q) parameter transformation from constrained to unconstrained space.

Maps constrained IGARCH parameters to the real line for use with unconstrained
optimizers. IGARCH enforces the unit-root constraint sum(alpha) + sum(beta) = 1,
so the last beta is computed as a residual and is excluded from the free parameter
vector (only q-1 betas are free).

This module is the forward complement of igarch_itransform, which maps
unconstrained parameters back to the constrained space.

Migrated from: univariate/igarch_transform.m
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005
"""

import numpy as np


def igarch_transform(
    parameters: np.ndarray,
    p: int,
    q: int,
    error_type: int,
    constant: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map constrained IGARCH parameters to unconstrained space.

    IGARCH has the unit-root constraint: sum(alpha) + sum(beta) = 1, so only
    p + q - 1 GARCH parameters are free (the last beta is determined as a
    residual). This function applies element-wise transforms that map the
    constrained parameter domain to the full real line, enabling the use of
    unconstrained numerical optimizers (e.g. scipy.optimize.minimize with
    method='L-BFGS-B').

    Parameters
    ----------
    parameters : np.ndarray
        Column parameter vector in constrained space. Layout:
        ``[omega?, alpha_1, ..., alpha_p, beta_1, ..., beta_{q-1}, nu?, lambda?]``
        where ``omega`` is present only when ``constant == 1``, ``nu`` is present
        for ``error_type`` in {2, 3, 4}, and ``lambda`` is present only for
        ``error_type == 4``.
    p : int
        Positive integer, number of symmetric innovation (ARCH) terms.
    q : int
        Non-negative integer, number of lagged conditional variance (GARCH)
        terms. Use 0 for a pure ARCH specification.
    error_type : int
        Distribution assumption for innovations:
        1 — Gaussian,
        2 — Student-t,
        3 — Generalized Error Distribution (GED),
        4 — Skewed Student-t.
    constant : int
        1 if the model includes a variance intercept (omega), 0 otherwise.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(trans_parameters, nu, lambda_param)`` where:

        - ``trans_parameters`` : np.ndarray of shape ``(constant + p + q - 1,)``
          containing the unconstrained transformed GARCH parameters
          ``[log(omega)?, talpha_1, ..., talpha_p, tbeta_1, ..., tbeta_{q-1}]``.
        - ``nu`` : np.ndarray — transformed distribution kurtosis parameter.
          Empty array ``np.empty(0)`` when not applicable.
        - ``lambda_param`` : np.ndarray — transformed distribution asymmetry
          parameter. Empty array ``np.empty(0)`` when not applicable.

    Raises
    ------
    ValueError
        If the input parameters violate the required constraints:
        alpha_i >= 0, beta_j >= 0, and sum(alpha) + sum(beta) < upper bound.

    Notes
    -----
    Input parameter constraints that must hold:

    1. omega > 0  (if constant == 1)
    2. alpha_i >= 0  for i = 1, ..., p
    3. beta_j >= 0   for j = 1, ..., q
    4. sum(alpha) + sum(beta) = 1  (IGARCH unit-root constraint)
    5. nu > 2 for Student-t / Skewed-t; 1 < nu < 50 for GED
    6. -0.99 < lambda < 0.99  for Skewed-t

    Transformation details:

    - **omega** → ``log(omega)`` (positivity via exponential inverse)
    - **alpha** → cascading logistic: each alpha_i is divided by the remaining
      feasible scale, then mapped via ``log(x / (1 - x))``
    - **beta** (first q-1) → cascading logistic with the same scheme, continuing
      from the scale remaining after all alphas
    - **nu** (Student-t / Skewed-t) → ``sqrt(nu - 2.01)``
    - **nu** (GED) → logistic mapping ``(1, 50) → (-inf, inf)``
    - **lambda** (Skewed-t) → logistic mapping ``(-0.99, 0.99) → (-inf, inf)``

    See Also
    --------
    igarch_itransform : Inverse transformation (unconstrained → constrained).
    igarch : IGARCH model estimation driver.
    """
    # Ref: igarch_transform.m:51-52 — Initialize distribution parameters as empty
    nu: np.ndarray = np.empty(0)
    lambda_param: np.ndarray = np.empty(0)

    # ---------------------------------------------------------------
    # Handle nu: distribution shape parameter
    # Ref: igarch_transform.m:56-63
    # Student-t / Skewed-t: nu > 2.01 → unconstrained via sqrt(nu - 2.01)
    # GED: 1.01 < nu < 50 → unconstrained via logistic
    # ---------------------------------------------------------------
    if error_type == 2 or error_type == 4:
        # Ref: igarch_transform.m:57-58 — MATLAB 1-indexed p+q+constant
        nu_val = parameters[p + q + constant - 1]
        nu = np.array(np.sqrt(nu_val - 2.01))
    elif error_type == 3:
        # Ref: igarch_transform.m:60-62 — GED logistic transform
        nu_val = parameters[p + q + constant - 1]
        temp = (nu_val - 1.0) / 49.0
        nu = np.array(np.log(temp / (1.0 - temp)))

    # ---------------------------------------------------------------
    # Handle lambda: asymmetry parameter for Skewed-t
    # Ref: igarch_transform.m:66-69
    # -0.995 < lambda < 0.995 → unconstrained via logistic
    # ---------------------------------------------------------------
    if error_type == 4:
        # Ref: igarch_transform.m:67-69 — MATLAB 1-indexed p+q+constant+1
        lambda_val = parameters[p + q + constant]
        temp = (lambda_val + 0.995) / 1.99
        lambda_param = np.array(np.log(temp / (1.0 - temp)))

    # ---------------------------------------------------------------
    # Transform omega using natural logarithm (ensures positivity)
    # Ref: igarch_transform.m:74-79
    # ---------------------------------------------------------------
    if constant:
        omega = parameters[0]
        tomega = np.array([np.log(omega)])
    else:
        tomega = np.empty(0)

    # ---------------------------------------------------------------
    # Parse alpha and beta from the parameter vector
    # Ref: igarch_transform.m:82-83
    # MATLAB: alpha = parameters(constant+1 : p+constant)        → p elements
    # MATLAB: beta  = parameters(constant+p+1 : constant+p+q-1)  → q-1 elements
    # Python 0-indexed slicing is exclusive at the end
    # ---------------------------------------------------------------
    alpha = np.copy(parameters[constant: constant + p])
    beta = np.copy(parameters[constant + p: constant + p + q - 1])

    # ---------------------------------------------------------------
    # Upper bound keeps the sum safely below 1
    # Ref: igarch_transform.m:86
    # ---------------------------------------------------------------
    ub = 0.999998

    # ---------------------------------------------------------------
    # Validate that parameters satisfy the necessary constraints
    # Ref: igarch_transform.m:89-96
    # ---------------------------------------------------------------
    if beta.size == 0:
        sumbeta = 0.0
    else:
        sumbeta = float(np.sum(beta))

    if np.any(alpha < 0) or np.any(beta < 0) or (float(np.sum(alpha)) + sumbeta) >= ub:
        raise ValueError(
            "These do not conform to the necessary set of restrictions "
            "to be transformed."
        )

    # ---------------------------------------------------------------
    # Replace exact zeros with a small positive value to avoid log(0)
    # Ref: igarch_transform.m:100-101
    # ---------------------------------------------------------------
    alpha[alpha == 0] = 1e-8
    beta[beta == 0] = 1e-8

    # ---------------------------------------------------------------
    # Adjust upper bound slightly to accommodate the epsilon replacements
    # Ref: igarch_transform.m:104
    # ---------------------------------------------------------------
    ub = ub + 1e-8 * (p + q)

    # ---------------------------------------------------------------
    # Cascading logistic transform for alpha parameters
    # Ref: igarch_transform.m:107-117
    # Each alpha_i is scaled by the remaining feasible range, then
    # mapped to the real line via the logit function log(x/(1-x)).
    # The scale is reduced after each alpha to enforce the cascading
    # partition of the feasible simplex.
    # ---------------------------------------------------------------
    scale = ub
    talpha = np.copy(alpha)
    for i in range(p):
        # Ref: igarch_transform.m:112 — Scale alpha_i into (0, 1)
        talpha[i] = alpha[i] / scale
        # Ref: igarch_transform.m:114 — Inverse logistic (logit)
        talpha[i] = np.log(talpha[i] / (1.0 - talpha[i]))
        # Ref: igarch_transform.m:116 — Reduce remaining scale
        scale = scale - alpha[i]

    # ---------------------------------------------------------------
    # Cascading logistic transform for beta parameters (q-1 free betas)
    # Ref: igarch_transform.m:119-133
    # The last beta is implicitly 1 - sum(alpha) - sum(first q-1 betas)
    # and is NOT included as a free parameter.
    # ---------------------------------------------------------------
    if q > 1:
        tbeta = np.copy(beta)
        for i in range(q - 1):
            # Ref: igarch_transform.m:125 — Scale beta_i into (0, 1)
            tbeta[i] = beta[i] / scale
            # Ref: igarch_transform.m:127 — Inverse logistic (logit)
            tbeta[i] = np.log(tbeta[i] / (1.0 - tbeta[i]))
            # Ref: igarch_transform.m:129 — Reduce remaining scale
            scale = scale - beta[i]
    else:
        tbeta = np.empty(0)

    # ---------------------------------------------------------------
    # Assemble the transformed parameter vector
    # Ref: igarch_transform.m:136 — transParameters=[tomega;talpha;tbeta]
    # ---------------------------------------------------------------
    parts = []
    if tomega.size > 0:
        parts.append(tomega)
    if talpha.size > 0:
        parts.append(talpha)
    if tbeta.size > 0:
        parts.append(tbeta)

    if len(parts) > 0:
        trans_parameters = np.concatenate(parts)
    else:
        trans_parameters = np.empty(0)

    return trans_parameters, nu, lambda_param
