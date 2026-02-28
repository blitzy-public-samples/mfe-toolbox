"""
TARCH/GJR-GARCH parameter transformation — constrained to unconstrained space.

Maps constrained TARCH/GJR-GARCH parameters to unconstrained real-line values
suitable for numerical optimisation via scipy.optimize.minimize. Uses logarithmic
and cascading logistic transforms to handle positivity and stationarity constraints.

Paired with ``tarch_itransform`` for the inverse mapping (unconstrained → constrained).

Migrated from ``univariate/tarch_transform.m`` — MFE Toolbox by Kevin Sheppard.
"""

import numpy as np


def tarch_transform(
    parameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
    tarch_type: int,
) -> np.ndarray:
    """Transform constrained TARCH/GJR-GARCH parameters to unconstrained space.

    Maps the parameter vector from the constrained TARCH parameter space
    (omega > 0, alpha >= 0, gamma + alpha > 0, beta >= 0, stationarity) to
    unconstrained values suitable for ``scipy.optimize.minimize``.

    Parameters
    ----------
    parameters : np.ndarray
        Column parameter vector with layout::

            [omega, alpha(1), ..., alpha(p), gamma(1), ..., gamma(o),
             beta(1), ..., beta(q), (nu), (lambda)]

        where ``nu`` and ``lambda`` are present only for non-Gaussian
        error distributions.
    p : int
        Number of symmetric innovation lags (positive integer, >= 1).
    o : int
        Number of asymmetric (threshold) innovation lags (non-negative integer).
    q : int
        Number of lagged conditional variance terms (non-negative integer).
    error_type : int
        Innovation distribution type:

        * 1 — Gaussian (Normal)
        * 2 — Student's *t*
        * 3 — Generalised Error Distribution (GED)
        * 4 — Hansen's Skewed Student's *t*
    tarch_type : int
        Model type indicator:

        * 1 — AVGARCH (model evolves in absolute values / std-dev)
        * 2 — GJR-GARCH / TARCH (model evolves in variance)

        Accepted for API compatibility with the TARCH driver; the transform
        logic follows the MATLAB source identically regardless of type since
        the stationarity factor (0.5 for gamma) is the same in both cases.

    Returns
    -------
    np.ndarray
        Transformed parameter vector in unconstrained space with layout::

            [t_omega, t_alpha(1), ..., t_alpha(p), t_gamma(1), ..., t_gamma(o),
             t_beta(1), ..., t_beta(q), (t_nu), (t_lambda)]

    Raises
    ------
    ValueError
        If the input parameters do not satisfy the required inequality
        constraints for the transform to be well-defined.

    Notes
    -----
    The input parameters must satisfy:

    1. ``omega > 0``
    2. ``alpha(i) >= 0`` for all *i*
    3. ``gamma(j) + alpha(j) >= 0`` for *j* <= min(p, o)
    4. ``gamma(j) >= 0`` for *j* > p  (when o > p)
    5. ``beta(k) >= 0`` for all *k*
    6. ``sum(alpha) + 0.5 * sum(gamma) + sum(beta) < 1``  (stationarity)

    Transform details (constrained → unconstrained):

    * **omega** → ``log(omega)``
    * **alpha / beta** → cascading logistic: each element is divided by
      the remaining "budget" and then mapped via ``log(x / (1 − x))``.
    * **gamma** → shifted + scaled logistic accounting for the corresponding
      alpha and available scale.
    * **nu** (Student's *t*, Skewed *t*) → ``sqrt(nu − 2.01)``
    * **nu** (GED) → logistic mapping (1, 50) → ℝ
    * **lambda** (Skewed *t*) → logistic mapping (−0.995, 0.995) → ℝ

    References
    ----------
    Ref: ``tarch_transform.m`` — MATLAB MFE Toolbox, Kevin Sheppard,
    University of Oxford.
    """
    # ------------------------------------------------------------------ #
    #  Defensive copy — avoid mutating the caller's array                #
    # ------------------------------------------------------------------ #
    parameters = np.copy(np.asarray(parameters, dtype=np.float64)).ravel()

    # ------------------------------------------------------------------ #
    #  Distribution parameter transforms  (nu, lambda)                   #
    #  Ref: tarch_transform.m:53-75                                      #
    # ------------------------------------------------------------------ #
    nu_transformed = np.array([], dtype=np.float64)
    lambda_transformed = np.array([], dtype=np.float64)

    if error_type == 2 or error_type == 4:
        # Student's t or Skewed t: nu > 2.01  →  sqrt(nu − 2.01)
        # Ref: tarch_transform.m:58-61 — MATLAB parameters(p+o+q+2), Python 0-based
        nu = parameters[p + o + q + 1]
        nu_transformed = np.array([np.sqrt(nu - 2.01)], dtype=np.float64)
        parameters[p + o + q + 1] = nu_transformed[0]
    elif error_type == 3:
        # GED: 1 < nu < 50  →  logistic mapping to ℝ
        # Ref: tarch_transform.m:62-66
        nu = parameters[p + o + q + 1]
        temp = (nu - 1.0) / 49.0
        nu_transformed = np.array([np.log(temp / (1.0 - temp))], dtype=np.float64)
        parameters[p + o + q + 1] = nu_transformed[0]

    if error_type == 4:
        # Skewed t: −0.995 < lambda < 0.995  →  logistic mapping to ℝ
        # Ref: tarch_transform.m:70-75 — MATLAB parameters(p+o+q+3), Python 0-based
        lam = parameters[p + o + q + 2]
        temp = (lam + 0.995) / 1.99
        lambda_transformed = np.array([np.log(temp / (1.0 - temp))], dtype=np.float64)
        parameters[p + o + q + 2] = lambda_transformed[0]

    # ------------------------------------------------------------------ #
    #  Omega transform: log ensures positivity                           #
    #  Ref: tarch_transform.m:79-80                                      #
    # ------------------------------------------------------------------ #
    omega = parameters[0]  # Ref: MATLAB parameters(1)
    tomega = np.log(omega)

    # ------------------------------------------------------------------ #
    #  Parse alpha, gamma, beta from parameter vector                    #
    #  Ref: tarch_transform.m:83-85 — MATLAB 1-based → Python 0-based   #
    # ------------------------------------------------------------------ #
    alpha = parameters[1 : p + 1].copy()          # MATLAB parameters(2:p+1)
    gamma = parameters[p + 1 : p + o + 1].copy()  # MATLAB parameters(p+2:p+o+1)
    beta = parameters[p + o + 1 : p + o + q + 1].copy()  # MATLAB parameters(p+o+2:p+o+q+1)

    # ------------------------------------------------------------------ #
    #  Upper bound for stationarity                                      #
    #  Ref: tarch_transform.m:88                                         #
    # ------------------------------------------------------------------ #
    UB = 0.999998

    # ------------------------------------------------------------------ #
    #  Pad alpha / gamma for constraint checking                         #
    #  Ref: tarch_transform.m:91-92                                      #
    # ------------------------------------------------------------------ #
    gamma2 = np.concatenate([gamma, np.zeros(max(0, p - o))])
    alpha2 = np.concatenate([alpha, np.zeros(max(0, o - p))])

    # ------------------------------------------------------------------ #
    #  Validate constraints                                              #
    #  Ref: tarch_transform.m:94-101                                     #
    # ------------------------------------------------------------------ #
    stationarity_sum = np.sum(alpha) + 0.5 * np.sum(gamma) + np.sum(beta)

    if (
        np.any(alpha < 0)
        or np.any(beta < 0)
        or np.any(gamma[min(p, o) : o] < 0)
        or np.any((alpha2 + gamma2) < 0)
        or stationarity_sum >= UB
    ):
        raise ValueError(
            "These do not conform to the necessary set of restrictions "
            "to be transformed."
        )

    # ------------------------------------------------------------------ #
    #  Replace exact zeros with small positive values to avoid log(0)    #
    #  Ref: tarch_transform.m:105-106                                    #
    # ------------------------------------------------------------------ #
    alpha[alpha == 0] = 1e-8
    beta[beta == 0] = 1e-8

    # Recompute padded arrays after zero replacement
    # Ref: tarch_transform.m:108-109
    gamma2 = np.concatenate([gamma, np.zeros(max(0, p - o))])
    alpha2 = np.concatenate([alpha, np.zeros(max(0, o - p))])

    # ------------------------------------------------------------------ #
    #  Handle positions where alpha2 + gamma2 == 0                       #
    #  Ref: tarch_transform.m:111-118                                    #
    # ------------------------------------------------------------------ #
    pl = np.where((alpha2 + gamma2) == 0)[0]
    if p >= o:
        # Ref: tarch_transform.m:113 — bump alpha at problematic indices
        alpha[pl] = alpha[pl] + 1e-8
    else:
        # Ref: tarch_transform.m:116 — bump gamma at problematic indices
        gamma[pl] = gamma[pl] + 1e-8

    # ------------------------------------------------------------------ #
    #  Adjust upper bound to allow for zero-replacement slack            #
    #  Ref: tarch_transform.m:120                                        #
    # ------------------------------------------------------------------ #
    UB = UB + 1e-8 * (p + o + q)

    # ------------------------------------------------------------------ #
    #  Cascading logistic transform for alpha                            #
    #  Ref: tarch_transform.m:122-133                                    #
    # ------------------------------------------------------------------ #
    scale = UB
    talpha = np.copy(alpha)
    for i in range(p):
        # Ref: tarch_transform.m:127 — alpha(i) / scale
        talpha[i] = alpha[i] / scale
        # Ref: tarch_transform.m:129 — inverse logistic
        talpha[i] = np.log(talpha[i] / (1.0 - talpha[i]))
        # Ref: tarch_transform.m:131 — reduce remaining scale
        scale = scale - alpha[i]

    # ------------------------------------------------------------------ #
    #  Bounded logistic transform for gamma                              #
    #  Ref: tarch_transform.m:136-159                                    #
    # ------------------------------------------------------------------ #
    eps_val = np.finfo(float).eps
    tgamma = np.copy(gamma)
    for i in range(o):
        tgamma[i] = gamma[i]
        if i < p:
            # Ref: tarch_transform.m:141-142 — has corresponding alpha
            tgamma[i] = tgamma[i] + alpha[i]
            tgamma[i] = tgamma[i] / (2.0 * scale + alpha[i])
        else:
            # Ref: tarch_transform.m:144 — no corresponding alpha
            tgamma[i] = tgamma[i] / (2.0 * scale)

        # Clamp away from exact 0 or 1 to keep logistic well-defined
        # Ref: tarch_transform.m:149-153
        if tgamma[i] == 1.0:
            tgamma[i] = 1.0 - eps_val
        elif tgamma[i] == 0.0:
            tgamma[i] = eps_val

        # Ref: tarch_transform.m:155 — inverse logistic
        tgamma[i] = np.log(tgamma[i] / (1.0 - tgamma[i]))
        # Ref: tarch_transform.m:157 — reduce remaining scale by half-gamma
        scale = scale - gamma[i] / 2.0

    # ------------------------------------------------------------------ #
    #  Cascading logistic transform for beta                             #
    #  Ref: tarch_transform.m:162-171                                    #
    # ------------------------------------------------------------------ #
    tbeta = np.copy(beta)
    for i in range(q):
        # Ref: tarch_transform.m:165 — beta(i) / scale
        tbeta[i] = beta[i] / scale
        # Ref: tarch_transform.m:167 — inverse logistic
        tbeta[i] = np.log(tbeta[i] / (1.0 - tbeta[i]))
        # Ref: tarch_transform.m:169 — reduce remaining scale
        scale = scale - beta[i]

    # ------------------------------------------------------------------ #
    #  Assemble transformed parameter vector                             #
    #  Ref: tarch_transform.m:174                                        #
    # ------------------------------------------------------------------ #
    trans_parameters = np.concatenate(
        [np.array([tomega]), talpha, tgamma, tbeta]
    )

    # Append distribution parameters when applicable
    if error_type in (2, 3, 4):
        trans_parameters = np.concatenate([trans_parameters, nu_transformed])
    if error_type == 4:
        trans_parameters = np.concatenate([trans_parameters, lambda_transformed])

    return trans_parameters
