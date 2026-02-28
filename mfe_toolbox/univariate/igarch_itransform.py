"""
IGARCH(P,Q) inverse parameter transformation.

Maps parameters from the real line (unconstrained optimization space) to a set of
parameters appropriate for an IGARCH model (constrained parameter space).
Used during IGARCH estimation to convert optimizer output back to model parameters.

This is the exact inverse of igarch_transform: for any valid constrained parameter
vector x, igarch_itransform(igarch_transform(x)) == x to numerical precision.

The IGARCH unit-root constraint (sum(alpha) + sum(beta) = 1) is enforced by
parameterizing only q-1 free betas; the final beta is implicitly
1 - sum(alpha) - sum(beta_1:q-1).

Notes
-----
Migrated from univariate/igarch_itransform.m
Original author: Kevin Sheppard, University of Oxford
Revision: 1, Date: 7/12/2009

References
----------
Engle, R.F. and Bollerslev, T. (1986), "Modelling the Persistence of Conditional
Variances", Econometric Reviews, 5, 1-50.
"""

import numpy as np


def igarch_itransform(
    parameters: np.ndarray, p: int, q: int, error_type: int, constant: int
) -> tuple[np.ndarray, np.ndarray | float, np.ndarray | float]:
    """
    IGARCH(P,Q) inverse parameter transformation.

    Maps parameters from the real line (unconstrained) to constrained space
    appropriate for an IGARCH model. Used in the estimation of IGARCH.

    Parameters
    ----------
    parameters : np.ndarray
        Column parameter vector in unconstrained space. Length depends on model
        specification:
        - constant + p + (q-1) structural parameters
        - Plus 1 for error_type 2 or 3 (nu)
        - Plus 2 for error_type 4 (nu and lambda)
    p : int
        Positive scalar integer representing the number of symmetric
        innovations (ARCH terms).
    q : int
        Non-negative scalar integer representing the number of lags of
        conditional variance (0 for ARCH). For IGARCH, typically q >= 1.
    error_type : int
        Distribution type for the error term:
            1 - Gaussian Innovations
            2 - T-distributed errors
            3 - Generalized Error Distribution (GED)
            4 - Skewed T distribution
    constant : int
        1 if model includes a constant (omega), 0 otherwise.

    Returns
    -------
    trans_parameters : np.ndarray
        A (constant + p + max(q-1, 0)) length vector of constrained parameters
        ordered as [omega, alpha(1),...,alpha(p), beta(1),...,beta(q-1)].
        The final beta (beta_q) is NOT included; it is determined by the
        unit-root constraint: beta_q = 1 - sum(alpha) - sum(beta_1:q-1).
    nu : np.ndarray or float
        Distribution kurtosis parameter. Empty ndarray if not applicable
        (error_type == 1). For Student's t (error_type 2 or 4): nu > 2.01.
        For GED (error_type 3): 1.01 < nu < 50.01.
    lambda_ : np.ndarray or float
        Distribution asymmetry parameter. Empty ndarray if not applicable
        (error_type != 4). For Skewed T (error_type 4): -0.99 < lambda < 0.99.

    Notes
    -----
    Output parameters satisfy:
        (1) omega > 0
        (2) alpha(i) >= 0 for i = 1, 2, ..., p
        (3) beta(i) >= 0 for i = 1, 2, ..., q
        (4) sum(alpha) + sum(beta) = 1 (IGARCH unit-root constraint)
        (5) nu > 2 for Student's T and nu > 1 for GED
        (6) -0.99 < lambda < 0.99 for Skewed T

    The cascading logistic transform works as follows:
    - A running 'scale' starts at UB = 0.999998 (just below 1)
    - Each alpha is mapped via sigmoid to [0, scale], then scale is reduced
    - Each free beta is similarly mapped to [0, remaining_scale]
    - This ensures all parameters are non-negative and their sum < 1

    See Also
    --------
    igarch : IGARCH model estimation driver.
    igarch_transform : Forward parameter transformation (constrained -> unconstrained).

    Examples
    --------
    >>> import numpy as np
    >>> # Unconstrained parameters for IGARCH(1,1) with Gaussian errors
    >>> params_unc = np.array([-2.0, 0.5])  # [omega_unc, alpha_unc]
    >>> trans, nu, lam = igarch_itransform(params_unc, p=1, q=1, error_type=1, constant=1)
    >>> # trans contains [omega_constrained, alpha_constrained]
    >>> # beta is implied: beta = 1 - alpha_constrained
    """
    # Initialize nu and lambda as empty arrays (not applicable by default)
    # Ref: igarch_itransform.m:43-44
    nu: np.ndarray | float = np.empty(0)
    lambda_: np.ndarray | float = np.empty(0)

    # Upper constraint to make sure that there is no overflow
    # Values > 100 would cause exp() overflow, so cap them
    # Ref: igarch_itransform.m:47 — parameters(parameters>100)=100
    parameters = np.copy(parameters).astype(np.float64)
    parameters[parameters > 100] = 100.0

    # Parse the parameters from the unconstrained vector
    # Ref: igarch_itransform.m:50-54 — MATLAB 1-indexed to Python 0-indexed
    if constant:
        # Ref: igarch_itransform.m:51 — omega=parameters(1) -> parameters[0]
        omega = parameters[0]

    # Ref: igarch_itransform.m:53 — alpha=parameters(constant+1:p+constant)
    # In Python 0-based: parameters[constant : p+constant], length = p
    alpha = parameters[constant: p + constant].copy()

    # Ref: igarch_itransform.m:54 — beta=parameters(p+constant+1:p+q+constant-1)
    # In Python 0-based: parameters[p+constant : p+q+constant-1], length = q-1
    # Only q-1 betas are free; the last beta is set by unit-root constraint
    beta = parameters[p + constant: p + q + constant - 1].copy()

    # Handle the transformation of nu and lambda (distribution parameters)
    # Ref: igarch_itransform.m:57-75
    if error_type == 2 or error_type == 4:
        # Student's t (or skewed-t): nu = 2.01 + x^2
        # This maps any real x to nu > 2.01, ensuring valid degrees of freedom
        # Ref: igarch_itransform.m:59-60 — nu=parameters(p+q+constant); nu=2.01+nu^2
        # Python 0-based index: p + q + constant - 1
        nu_raw = parameters[p + q + constant - 1]
        nu = 2.01 + nu_raw ** 2
    elif error_type == 3:
        # GED: logistic transform maps to (1.01, 50.01)
        # Ref: igarch_itransform.m:62-68
        # Python 0-based index: p + q + constant - 1
        nu_raw = float(parameters[p + q + constant - 1])
        # Overflow protection for the logistic function
        if nu_raw > 100:
            nu_raw = 100.0
        nu = np.exp(nu_raw) / (1.0 + np.exp(nu_raw))
        nu = 49.0 * nu + 1.01

    # If skewed-t, use a logistic to map to (-0.99, 0.99)
    # Ref: igarch_itransform.m:71-74
    if error_type == 4:
        # Python 0-based index: p + q + constant
        lambda_raw = parameters[p + q + constant]
        lambda_ = np.exp(lambda_raw) / (1.0 + np.exp(lambda_raw))
        lambda_ = 1.98 * lambda_ - 0.99

    # Upper bound of transform — keeps sum of alpha + beta just below 1
    # Ref: igarch_itransform.m:78
    upper_bound = 0.999998

    # Simple transform of omega: constrained = exp(unconstrained) ensures omega > 0
    # Ref: igarch_itransform.m:81-85
    if constant:
        tomega = np.exp(omega)

    # Initialize the transformed parameters as copies for in-place modification
    # Ref: igarch_itransform.m:88-89
    talpha = alpha.copy()
    tbeta = beta.copy()

    # Set the initial scale for cascading logistic bounds
    # Ref: igarch_itransform.m:92
    scale = upper_bound

    # Cascading logistic transform for alpha parameters
    # Each alpha_i is mapped via sigmoid to [0, scale], then scale is reduced
    # This ensures all alphas are non-negative and sum < upper_bound
    # Ref: igarch_itransform.m:94-99
    for i in range(p):
        # Logistic (sigmoid) function maps real line to (0, 1), then multiply by scale
        # Ref: igarch_itransform.m:96 — talpha(i)=(exp(talpha(i))/(1+exp(talpha(i))))*scale
        talpha[i] = (np.exp(talpha[i]) / (1.0 + np.exp(talpha[i]))) * scale
        # Reduce remaining scale by the allocated portion
        # Ref: igarch_itransform.m:98 — scale=scale-talpha(i)
        scale = scale - talpha[i]

    # Cascading logistic transform for q-1 free beta parameters
    # The q-th beta is determined by the IGARCH unit-root constraint:
    #   beta_q = 1 - sum(alpha) - sum(beta_1:q-1)
    # Ref: igarch_itransform.m:100-109
    if q > 1:
        for i in range(q - 1):
            # Ref: igarch_itransform.m:103 — tbeta(i)=(exp(tbeta(i))/(1+exp(tbeta(i))))*scale
            tbeta[i] = (np.exp(tbeta[i]) / (1.0 + np.exp(tbeta[i]))) * scale
            # Ref: igarch_itransform.m:105 — scale=scale-tbeta(i)
            scale = scale - tbeta[i]
    else:
        # When q <= 1, there are no free betas
        # For q=1, the single beta is entirely determined by unit-root: beta = 1 - sum(alpha)
        # Ref: igarch_itransform.m:108 — tbeta = []
        tbeta = np.zeros(0)

    # Regroup the transformed parameters into a single vector
    # Ref: igarch_itransform.m:111 — transParameters=[tomega;talpha;tbeta]
    parts: list[np.ndarray] = []
    if constant:
        parts.append(np.atleast_1d(tomega))
    if talpha.size > 0:
        parts.append(talpha)
    if tbeta.size > 0:
        parts.append(tbeta)

    if parts:
        trans_parameters = np.concatenate(parts)
    else:
        trans_parameters = np.empty(0)

    return trans_parameters, nu, lambda_
