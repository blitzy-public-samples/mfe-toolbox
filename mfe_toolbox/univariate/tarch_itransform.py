"""
Inverse parameter transformation for TARCH/GJR-GARCH models.

Maps parameters from the unconstrained real line back to the constrained
parameter space appropriate for TARCH models.  Used after numerical
optimization (which operates in unconstrained space) to recover valid
GARCH parameters.

This module provides the exact inverse of tarch_transform, restoring:
  * omega  > 0
  * alpha >= 0
  * gamma + alpha > 0 (where alpha exists for the corresponding lag)
  * beta  >= 0
  * sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1  (stationarity)
  * Distribution shape/asymmetry parameters within valid bounds

Ref: univariate/tarch_itransform.m — Kevin Sheppard, University of Oxford
     Revision 3, Date 9/1/2005
"""

import numpy as np


def tarch_itransform(
    parameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
    tarch_type: int,
) -> np.ndarray:
    """
    TARCH(P,O,Q) inverse parameter transformation.  Maps parameters from
    the real line to a set of parameters appropriate for a TARCH model.

    This is the exact inverse of ``tarch_transform``.  The cascading
    logistic scheme guarantees that all transformed parameters satisfy
    the non-negativity and stationarity constraints required by the TARCH
    variance recursion.

    Parameters
    ----------
    parameters : np.ndarray
        Column parameter vector of unconstrained parameters produced by
        the numerical optimizer.  Length is ``1 + p + o + q`` plus any
        distribution parameters:

        * error_type 1 (Normal): no extra parameters
        * error_type 2 (Student's t): +1 (nu)
        * error_type 3 (GED): +1 (nu)
        * error_type 4 (Skewed t): +2 (nu, lambda)

    p : int
        Positive integer representing the number of symmetric innovation
        (ARCH) terms.
    o : int
        Non-negative integer representing the number of asymmetric
        innovation terms.  Set to 0 for symmetric processes.
    q : int
        Non-negative integer representing the number of lagged conditional
        variance (GARCH) terms.  Set to 0 for a pure ARCH model.
    error_type : int
        Distribution assumption for innovations:

        * 1 — Gaussian (Normal)
        * 2 — Student's t
        * 3 — Generalized Error Distribution (GED)
        * 4 — Skewed Student's t

    tarch_type : int
        TARCH model type specification.  Does not affect the inverse
        transform computation; included for API symmetry with the driver.

    Returns
    -------
    np.ndarray
        Flat 1-D array of constrained parameters.  Layout::

            [omega, alpha_1, ..., alpha_p,
             gamma_1, ..., gamma_o,
             beta_1, ..., beta_q,
             nu?,  lambda?]

        Distribution parameters ``nu`` and ``lambda`` are appended when
        the corresponding ``error_type`` requires them.

        Output parameters satisfy:

        * (1) omega > 0
        * (2) alpha(i) >= 0  for i = 1, ..., p
        * (3) gamma(i) + alpha(i) > 0  for i = 1, ..., min(p, o)
        * (4) beta(i) >= 0  for i = 1, ..., q
        * (5) sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1
        * (6) nu > 2 for Student's t; 1 < nu < 50 for GED
        * (7) -0.99 < lambda < 0.99 for Skewed t

    Raises
    ------
    ValueError
        If *parameters* is not an ``np.ndarray``, if lag orders are
        negative, if *error_type* is not in {1, 2, 3, 4}, or if the
        parameter vector is too short.

    Notes
    -----
    Transformation details (inverse of tarch_transform):

    * **omega**: ``exp(x)`` ensures strict positivity.
    * **alpha**: Cascading logistic ``sigmoid(x) * scale`` where *scale*
      starts at ``UB = 0.9998`` and is decremented after each alpha.
    * **gamma**: Logistic mapped to ``[-alpha_i, 2*scale]`` when the
      corresponding alpha exists, or ``[0, 2*scale]`` otherwise.  Scale
      is decremented by ``0.5 * gamma_i``.
    * **beta**: Same cascading logistic as alpha using the remaining
      scale budget.
    * **nu** (Student's t / Skewed t): ``2.01 + x**2`` ensures nu > 2.
    * **nu** (GED): logistic scaled to ``(1.01, 50.01)``.
    * **lambda** (Skewed t): logistic scaled to ``(-0.99, 0.99)``.

    Ref: tarch_itransform.m — Kevin Sheppard, University of Oxford

    See Also
    --------
    tarch_transform : Forward (constrained → unconstrained) transform.
    tarch : Main TARCH estimation driver.
    """
    # ------------------------------------------------------------------ #
    #  Input validation                                                   #
    # ------------------------------------------------------------------ #
    if not isinstance(parameters, np.ndarray):
        raise ValueError("parameters must be a numpy ndarray")
    if p < 0:
        raise ValueError("p must be a non-negative integer")
    if o < 0:
        raise ValueError("o must be a non-negative integer")
    if q < 0:
        raise ValueError("q must be a non-negative integer")
    if error_type not in (1, 2, 3, 4):
        raise ValueError("error_type must be 1, 2, 3, or 4")

    # Compute minimum expected length of the parameter vector
    num_params = 1 + p + o + q
    if error_type in (2, 3):
        num_params += 1
    elif error_type == 4:
        num_params += 2

    if parameters.size < num_params:
        raise ValueError(
            f"parameters has {parameters.size} elements but model with "
            f"p={p}, o={o}, q={q}, error_type={error_type} requires "
            f"at least {num_params}"
        )

    # ------------------------------------------------------------------ #
    #  Work on a copy; apply overflow protection                          #
    # ------------------------------------------------------------------ #
    # Ref: tarch_itransform.m:50 — parameters(parameters>100)=100
    # Upper clip prevents exp() overflow; lower values are safe for the
    # logistic (sigmoid approaches 0 without numerical issues).
    params = np.copy(parameters).astype(np.float64).ravel()
    params = np.clip(params, a_min=None, a_max=100.0)

    # ------------------------------------------------------------------ #
    #  Distribution parameter transforms (nu, lambda)                     #
    # ------------------------------------------------------------------ #
    # Ref: tarch_itransform.m:46-47
    nu = np.zeros(0)       # empty by default (Normal)
    lam = np.zeros(0)      # 'lam' avoids shadowing Python's 'lambda'

    if error_type == 2 or error_type == 4:
        # Student's t or Skewed t: quadratic transform ensures nu > 2.01
        # Ref: tarch_itransform.m:61-62 — nu = 2.01 + nu^2
        nu_raw = params[p + o + q + 1]
        nu_val = 2.01 + nu_raw ** 2
        nu = np.array([nu_val])
    elif error_type == 3:
        # GED: logistic scaled to (1.01, 50.01)
        # Ref: tarch_itransform.m:65-70
        nu_raw = params[p + o + q + 1]
        # Ref: tarch_itransform.m:66-68 — extra overflow guard for GED
        nu_raw = np.clip(nu_raw, a_min=None, a_max=100.0)
        nu_val = np.exp(nu_raw) / (1.0 + np.exp(nu_raw))
        nu_val = 49.0 * nu_val + 1.01
        nu = np.array([nu_val])

    if error_type == 4:
        # Skewed t: logistic mapped to (-0.99, 0.99)
        # Ref: tarch_itransform.m:73-77
        lam_raw = params[p + o + q + 2]
        lam_val = np.exp(lam_raw) / (1.0 + np.exp(lam_raw))
        lam_val = 1.98 * lam_val - 0.99
        lam = np.array([lam_val])

    # ------------------------------------------------------------------ #
    #  Parse core GARCH parameters                                        #
    # ------------------------------------------------------------------ #
    # Ref: tarch_itransform.m:53-56
    # MATLAB 1-based → Python 0-based index adjustment throughout
    omega = params[0]                                   # Ref: tarch_itransform.m:53 — parameters(1)
    alpha = np.copy(params[1: p + 1])                   # Ref: tarch_itransform.m:54 — parameters(2:p+1)
    gamma = np.copy(params[p + 1: p + o + 1])           # Ref: tarch_itransform.m:55 — parameters(p+2:p+o+1)
    beta  = np.copy(params[p + o + 1: p + o + q + 1])   # Ref: tarch_itransform.m:56 — parameters(p+o+2:p+o+q+1)

    # ------------------------------------------------------------------ #
    #  Transform omega: exp ensures omega > 0                             #
    # ------------------------------------------------------------------ #
    # Ref: tarch_itransform.m:80,83
    UB = 0.9998
    tomega = np.exp(omega)

    # ------------------------------------------------------------------ #
    #  Initialize transformed arrays                                      #
    # ------------------------------------------------------------------ #
    # Ref: tarch_itransform.m:86-88
    talpha = np.copy(alpha)
    tgamma = np.copy(gamma)
    tbeta  = np.copy(beta)

    # ------------------------------------------------------------------ #
    #  Cascading logistic transform — alpha                               #
    # ------------------------------------------------------------------ #
    # Ref: tarch_itransform.m:91-98
    # Each alpha is mapped via sigmoid(x)*scale into (0, scale), then
    # scale is decremented to enforce the stationarity budget.
    scale = UB
    for i in range(p):
        # Logistic sigmoid scaled to (0, scale)
        # Ref: tarch_itransform.m:95 — MATLAB 1-based i; Python 0-based
        talpha[i] = (np.exp(talpha[i]) / (1.0 + np.exp(talpha[i]))) * scale
        # Decrement remaining budget
        # Ref: tarch_itransform.m:97
        scale = scale - talpha[i]

    # ------------------------------------------------------------------ #
    #  Conditional logistic transform — gamma (asymmetry)                 #
    # ------------------------------------------------------------------ #
    # Ref: tarch_itransform.m:99-114
    # When the corresponding alpha exists (p > i in 0-based), gamma is
    # mapped to (-alpha_i, 2*scale).  Otherwise to (0, 2*scale).
    for i in range(o):
        # First map into (0, 1) via logistic
        # Ref: tarch_itransform.m:102
        tgamma[i] = np.exp(tgamma[i]) / (1.0 + np.exp(tgamma[i]))

        # Ref: tarch_itransform.m:103 — 'if p>=i' with MATLAB 1-based i
        # Python equivalent with 0-based i: p > i  (MATLAB i=k ↔ Python i=k-1,
        # condition p >= k  ↔  p > k-1  ↔  p > i)
        if p > i:
            # Alpha(i) exists — map gamma into [-alpha_i, 2*scale]
            # Stretch to [0, 2*scale + alpha_i], then shift down
            # Ref: tarch_itransform.m:105,107
            tgamma[i] = tgamma[i] * (2.0 * scale + talpha[i])
            tgamma[i] = tgamma[i] - talpha[i]
        else:
            # No corresponding alpha — map gamma into [0, 2*scale]
            # Ref: tarch_itransform.m:110
            tgamma[i] = tgamma[i] * (2.0 * scale)

        # Update scale: gamma contributes half its value to persistence
        # Ref: tarch_itransform.m:113
        scale = scale - 0.5 * tgamma[i]

    # ------------------------------------------------------------------ #
    #  Cascading logistic transform — beta                                #
    # ------------------------------------------------------------------ #
    # Ref: tarch_itransform.m:115-120
    for i in range(q):
        # Logistic sigmoid scaled to (0, scale)
        # Ref: tarch_itransform.m:117
        tbeta[i] = (np.exp(tbeta[i]) / (1.0 + np.exp(tbeta[i]))) * scale
        # Decrement remaining budget
        # Ref: tarch_itransform.m:119
        scale = scale - tbeta[i]

    # ------------------------------------------------------------------ #
    #  Assemble output vector                                             #
    # ------------------------------------------------------------------ #
    # Ref: tarch_itransform.m:122 — trans_parameters=[tomega;talpha;tgamma;tbeta]
    # Python: single flat array with distribution parameters appended
    core_params = np.concatenate([
        np.array([tomega]),
        talpha,
        tgamma,
        tbeta,
    ])

    # Append distribution parameters when applicable
    if nu.size > 0 and lam.size > 0:
        result = np.concatenate([core_params, nu, lam])
    elif nu.size > 0:
        result = np.concatenate([core_params, nu])
    else:
        result = core_params

    return result
