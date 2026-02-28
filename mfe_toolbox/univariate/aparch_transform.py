"""
APARCH(P,O,Q) parameter transformation for unconstrained optimization.

Maps constrained APARCH parameters to the real line (unconstrained space)
for use with scipy.optimize.minimize. This is the forward transform, paired
with aparch_itransform (inverse transform).

APARCH models include a delta power parameter and asymmetry gamma parameters.
The transformation uses cascading logistic transforms for alpha and beta,
simple logistic transforms for gamma and delta, and distribution-specific
transforms for nu and lambda.

Parameter Constraints (input):
    (1) omega > 0
    (2) alpha(i) >= 0 for i = 1, ..., p
    (3) -1 < gamma(i) < 1 for i = 1, ..., o
    (4) beta(i) >= 0 for i = 1, ..., q
    (5) sum(alpha) + sum(beta) < 1
    (6) 0.3 < delta < 4.0
    (7) nu > 2 for Student's T; 1 < nu < 49 for GED
    (8) -0.99 < lambda < 0.99 for Skewed T

Notes
-----
Migrated from univariate/aparch_transform.m (MFE Toolbox Version 4.0).
All MATLAB 1-based indices have been adjusted to Python 0-based indices.
Each index adjustment is documented with an inline reference comment.

Copyright: Kevin Sheppard, University of Oxford
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005
"""

import numpy as np


def aparch_transform(
    parameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
    delta_is_estimated: int = 1,
) -> tuple[np.ndarray, float | None, float | None]:
    """
    Transform constrained APARCH parameters to unconstrained real-line space.

    Maps the constrained parameter vector of an APARCH(P,O,Q) model to an
    unconstrained representation suitable for numerical optimization. This
    function is the forward transform; the inverse is ``aparch_itransform``.

    Parameters
    ----------
    parameters : np.ndarray
        Column parameter vector of constrained APARCH parameters. The vector
        structure is:
        [omega, alpha(1:p), gamma(1:o), beta(1:q), delta (if estimated),
         nu (if error_type in {2,3,4}), lambda_ (if error_type == 4)].
    p : int
        Positive integer representing the number of symmetric innovation
        (ARCH) terms.
    o : int
        Non-negative integer representing the number of asymmetric innovation
        terms. Use 0 for symmetric processes.
    q : int
        Non-negative integer representing the number of lagged conditional
        variance (GARCH) terms. Use 0 for pure ARCH models.
    error_type : int
        Distribution of innovations:
            1 - Gaussian (Normal)
            2 - Student's t
            3 - Generalized Error Distribution (GED)
            4 - Skewed Student's t
    delta_is_estimated : int, optional
        Flag indicating whether the delta power parameter is estimated (1)
        or fixed by the user (0). Default is 1.

    Returns
    -------
    trans_parameters : np.ndarray
        Transformed parameter vector on the real line. Structure:
        [tomega, talpha(1:p), tgamma(1:o), tbeta(1:q), tdelta (if estimated)].
        Distribution parameters (nu, lambda_) are returned separately.
    nu : float or None
        Transformed distribution shape parameter. None if error_type == 1.
        For Student's t (error_type 2 or 4): sqrt(nu - 2.01).
        For GED (error_type 3): logistic transform of (nu - 1) / 49.
    lambda_ : float or None
        Transformed distribution asymmetry parameter. None unless
        error_type == 4 (Skewed t), in which case it is the logistic
        transform of (lambda + 0.995) / 1.99.

    Raises
    ------
    ValueError
        If the input parameters do not satisfy the required constraints for
        transformation, or if complex values arise during transformation.

    See Also
    --------
    aparch_itransform : Inverse parameter transformation.
    aparch : APARCH model estimation driver.

    Examples
    --------
    >>> import numpy as np
    >>> params = np.array([0.01, 0.05, 0.1, 0.85, 2.0])
    >>> tp, nu, lam = aparch_transform(params, 1, 1, 1, 1, 1)
    """
    # -------------------------------------------------------------------------
    # CRITICAL: Copy input array — MATLAB passes by value; Python numpy arrays
    # are mutable references. We must not modify the caller's array.
    # -------------------------------------------------------------------------
    parameters = np.array(parameters, dtype=np.float64).copy()

    # Ensure parameters is 1-D
    parameters = np.atleast_1d(parameters).ravel()

    # =========================================================================
    # Phase 1: Distribution parameter transforms (nu, lambda_)
    # Ref: aparch_transform.m:54-74
    # =========================================================================
    nu: float | None = None
    lambda_: float | None = None

    # Handle nu transform for T-distribution (error_type == 2 or 4)
    # Ref: aparch_transform.m:58-61 — T-distribution nu uses sqrt transform
    # MATLAB index: p+o+q+2+deltaIsEstimated → Python: p+o+q+1+delta_is_estimated (0-based)
    if error_type == 2 or error_type == 4:
        nu_idx = p + o + q + 1 + delta_is_estimated
        nu = float(parameters[nu_idx])
        # Ref: aparch_transform.m:60 — sqrt(nu - 2.01) maps (2.01, inf) to (0, inf)
        nu = float(np.sqrt(nu - 2.01))
        parameters[nu_idx] = nu
    elif error_type == 3:
        # Ref: aparch_transform.m:63-66 — GED nu logistic transform
        # Maps (1, 50) to real line via logistic on (nu - 1) / 49
        nu_idx = p + o + q + 1 + delta_is_estimated
        nu = float(parameters[nu_idx])
        temp = (nu - 1.0) / 49.0
        nu = float(np.log(temp / (1.0 - temp)))
        parameters[nu_idx] = nu

    # Handle lambda transform for Skewed T (error_type == 4)
    # Ref: aparch_transform.m:70-74 — Skewed T lambda logistic transform
    # MATLAB index: p+o+q+3+deltaIsEstimated → Python: p+o+q+2+delta_is_estimated (0-based)
    if error_type == 4:
        lam_idx = p + o + q + 2 + delta_is_estimated
        lambda_ = float(parameters[lam_idx])
        # Ref: aparch_transform.m:72-73 — logistic on (lambda + 0.995) / 1.99
        temp = (lambda_ + 0.995) / 1.99
        lambda_ = float(np.log(temp / (1.0 - temp)))
        parameters[lam_idx] = lambda_

    # =========================================================================
    # Phase 2: Core parameter parsing and validation
    # Ref: aparch_transform.m:78-103
    # =========================================================================

    # Transform omega with log — maps (0, inf) to real line
    # Ref: aparch_transform.m:79-80 — MATLAB: parameters(1) → Python: parameters[0]
    omega = parameters[0]
    tomega = np.log(omega)

    # Parse alpha, gamma, beta with 0-based indexing
    # Ref: aparch_transform.m:83 — MATLAB: parameters(2:p+1) → Python: parameters[1:p+1]
    alpha = parameters[1:p + 1].copy()
    # Ref: aparch_transform.m:84 — MATLAB: parameters(p+2:p+o+1) → Python: parameters[p+1:p+o+1]
    gamma = parameters[p + 1:p + o + 1].copy()
    # Ref: aparch_transform.m:85 — MATLAB: parameters(p+o+2:p+o+q+1) → Python: parameters[p+o+1:p+o+q+1]
    beta = parameters[p + o + 1:p + o + q + 1].copy()

    # Parse delta if estimated
    # Ref: aparch_transform.m:86-88 — MATLAB: parameters(p+o+q+2) → Python: parameters[p+o+q+1]
    delta = 0.0  # placeholder — only used if delta_is_estimated is True
    if delta_is_estimated:
        delta = float(parameters[p + o + q + 1])

    # Ref: aparch_transform.m:90 — Upper bound to keep sum a bit away from 1
    UB = 0.999998

    # Validate constraints
    # Ref: aparch_transform.m:93-95
    alpha_neg = np.any(alpha < 0) if alpha.size > 0 else False
    beta_neg = np.any(beta < 0) if beta.size > 0 else False
    gamma_low = np.any(gamma < -1) if gamma.size > 0 else False
    gamma_high = np.any(gamma > 1) if gamma.size > 0 else False
    sum_ab = float(np.sum(alpha) + np.sum(beta))

    if alpha_neg or beta_neg or gamma_low or gamma_high or sum_ab >= UB:
        raise ValueError(
            'Parameters do not conform to the necessary set of restrictions '
            'to be transformed.'
        )

    # Replace exact zeros with small epsilon to avoid log(0) in logistic transform
    # Ref: aparch_transform.m:99-100
    alpha[alpha == 0] = 1e-8
    beta[beta == 0] = 1e-8

    # Adjust upper bound slightly upward to accommodate the epsilon replacements
    # Ref: aparch_transform.m:103
    UB = UB + 1e-8 * (p + o + q)

    # =========================================================================
    # Phase 3: Cascading logistic transforms
    # Ref: aparch_transform.m:105-141
    # =========================================================================

    # --- Alpha cascading logistic transform ---
    # Ref: aparch_transform.m:106-116
    # Each alpha is scaled by the remaining capacity, then logistic-transformed.
    scale = UB
    talpha = alpha.copy()
    for i in range(p):
        # Ref: aparch_transform.m:111 — Scale alpha(i) by remaining capacity
        talpha[i] = alpha[i] / scale
        # Ref: aparch_transform.m:113 — Inverse logistic (logit) transform
        talpha[i] = np.log(talpha[i] / (1.0 - talpha[i]))
        # Ref: aparch_transform.m:115 — Update remaining capacity
        scale = scale - alpha[i]

    # --- Beta cascading logistic transform ---
    # Ref: aparch_transform.m:118-128
    # Similar cascading logistic, continuing from where alpha left off.
    tbeta = beta.copy()
    for i in range(q):
        # Ref: aparch_transform.m:123 — Scale beta(i) by remaining capacity
        tbeta[i] = tbeta[i] / scale
        # Ref: aparch_transform.m:125 — Inverse logistic (logit) transform
        tbeta[i] = np.log(tbeta[i] / (1.0 - tbeta[i]))
        # Ref: aparch_transform.m:127 — Update remaining capacity
        scale = scale - beta[i]

    # --- Gamma logistic transform (vectorized, not cascading) ---
    # Ref: aparch_transform.m:132-133 — Maps gamma from (-1, 1) to real line
    if gamma.size > 0:
        tgamma = (gamma + 1.0) / 2.0
        tgamma = np.log(tgamma / (1.0 - tgamma))
    else:
        tgamma = np.array([])

    # --- Delta logistic transform (if estimated) ---
    # Ref: aparch_transform.m:136-141 — Maps delta from (0.3, 4.0) to real line
    if delta_is_estimated:
        tdelta_val = (delta - 0.3) / 3.7
        tdelta = np.log(tdelta_val / (1.0 - tdelta_val))
    else:
        tdelta = None  # Not estimated — omit from output

    # =========================================================================
    # Phase 4: Output assembly
    # Ref: aparch_transform.m:144-148
    # =========================================================================

    # Concatenate transformed core parameters: [tomega, talpha, tgamma, tbeta, tdelta]
    # Ref: aparch_transform.m:145
    parts: list[np.ndarray] = [np.atleast_1d(tomega), talpha]
    if tgamma.size > 0:
        parts.append(tgamma)
    else:
        # Handle o=0 case — append empty array for consistent structure
        parts.append(np.array([]))
    parts.append(tbeta)
    if delta_is_estimated and tdelta is not None:
        parts.append(np.atleast_1d(tdelta))

    trans_parameters = np.concatenate(parts)

    # Validate no complex values arose during transformation
    # Ref: aparch_transform.m:146-148 — MATLAB uses `keyboard` (debug breakpoint);
    # Python raises ValueError instead as per migration rules.
    if np.any(~np.isreal(trans_parameters)):
        raise ValueError(
            'Complex values detected in transformed parameters.'
        )

    return trans_parameters, nu, lambda_
