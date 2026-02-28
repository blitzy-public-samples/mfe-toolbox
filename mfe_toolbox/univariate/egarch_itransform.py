"""
EGARCH(P,O,Q) inverse parameter transformation.

Maps parameters from unconstrained real line back to the constrained EGARCH
parameter space. Used after numerical optimization to recover valid
distribution shape parameters.

EGARCH models operate on log-variance, so omega, alpha, gamma, and beta
parameters are relatively unconstrained and are NOT transformed. Only the
distribution shape parameters (nu, lambda) require inverse transformation.

Notes
-----
This is the exact inverse of egarch_transform. For a round-trip:
    egarch_itransform(egarch_transform(params, p, o, q, et), p, o, q, et)
must return the original params to within numerical precision (±1e-6).

Migrated from: univariate/egarch_itransform.m (MFE Toolbox, Kevin Sheppard)
"""

import numpy as np


def egarch_itransform(
    parameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
) -> np.ndarray:
    """
    Inverse transform EGARCH parameters from unconstrained to constrained space.

    Maps unconstrained optimizer output back to valid EGARCH parameter ranges.
    Only distribution parameters (nu, lambda) are transformed; the core EGARCH
    parameters (omega, alpha, gamma, beta) are passed through unchanged because
    EGARCH's log-variance specification imposes fewer constraints.

    Parameters
    ----------
    parameters : np.ndarray
        Column parameter vector of length ``1 + p + o + q`` (plus distribution
        parameters). Layout is
        ``[omega, alpha(1), ..., alpha(p), gamma(1), ..., gamma(o),
          beta(1), ..., beta(q), [nu, [lambda]]]``.
        Values are in unconstrained (optimizer) space.
    p : int
        Positive integer, number of symmetric innovation lags (ARCH terms).
    o : int
        Non-negative integer, number of asymmetric innovation lags.
        Use 0 for a symmetric EGARCH process.
    q : int
        Non-negative integer, number of conditional variance lags (GARCH terms).
        Use 0 for a pure ARCH-in-mean process.
    error_type : int
        Distribution assumption for innovations:
        - 1: Gaussian (normal) — no extra parameters
        - 2: Student's t — one extra parameter (nu)
        - 3: Generalized Error Distribution (GED) — one extra parameter (nu)
        - 4: Skewed Student's t — two extra parameters (nu, lambda)

    Returns
    -------
    np.ndarray
        Parameter vector with distribution shape parameters mapped to their
        constrained domain:

        - **error_type 2 or 4 (Student's t / Skewed t):**
          ``nu = 2.01 + nu_unconstrained ** 2``, ensuring ``nu > 2.01``
        - **error_type 3 (GED):**
          ``nu = 49 * sigmoid(clamp(nu_unconstrained, max=100)) + 1.01``,
          ensuring ``1.01 < nu < 50.01``
        - **error_type 4 (Skewed t), lambda:**
          ``lambda = 1.98 * sigmoid(lambda_unconstrained) - 0.99``,
          ensuring ``-0.99 < lambda < 0.99``

    See Also
    --------
    egarch_transform : Forward (constrained → unconstrained) transform.
    egarch : Main EGARCH estimation driver.

    Examples
    --------
    >>> import numpy as np
    >>> params = np.array([-0.1, 0.05, -0.03, 0.95, 0.0])
    >>> result = egarch_itransform(params, p=1, o=1, q=1, error_type=2)
    >>> result[3]  # beta passes through unchanged
    0.95
    >>> result[4]  # nu transformed: 2.01 + 0**2 = 2.01
    2.01
    """
    # Create a copy to avoid mutating the caller's array
    # Ref: egarch_itransform.m — MATLAB modifies in-place; Python uses copy for safety
    parameters = np.copy(parameters)

    # ----------------------------------------------------------------
    # Nu (degrees-of-freedom / shape) transform
    # ----------------------------------------------------------------
    # Index of nu in the parameter vector
    # Ref: egarch_itransform.m:39 — MATLAB 1-based index p+o+q+2 → Python 0-based p+o+q+1
    nu_idx = p + o + q + 1

    if error_type == 2 or error_type == 4:
        # Student's t or Skewed t: nu = 2.01 + nu_unconstrained^2
        # Ensures nu > 2.01 (required for finite variance of t-distribution)
        # Ref: egarch_itransform.m:40 — nu=2.01+nu^2
        nu = parameters[nu_idx]
        nu = 2.01 + nu ** 2
        parameters[nu_idx] = nu
    elif error_type == 3:
        # GED: logistic transform mapping (-inf, inf) → (1.01, ~50.01)
        # Ref: egarch_itransform.m:43-49 — clamp then logistic
        nu = parameters[nu_idx]
        # Clamp the unconstrained value to prevent overflow in exp()
        # Ref: egarch_itransform.m:44-46 — if nu>100; nu=100; end
        if nu > 100:
            nu = 100.0
        # Apply logistic (sigmoid) function: sigmoid(x) = exp(x) / (1 + exp(x))
        # Ref: egarch_itransform.m:47 — nu=exp(nu)/(1+exp(nu))
        nu = np.exp(nu) / (1.0 + np.exp(nu))
        # Scale from (0, 1) to (1.01, ~50.01)
        # Ref: egarch_itransform.m:48 — nu=49*nu+1.01
        nu = 49.0 * nu + 1.01
        parameters[nu_idx] = nu

    # ----------------------------------------------------------------
    # Lambda (skewness) transform — only for Skewed t
    # ----------------------------------------------------------------
    if error_type == 4:
        # Logistic transform mapping (-inf, inf) → (-0.99, 0.99)
        # Ref: egarch_itransform.m:53 — MATLAB 1-based index p+o+q+3 → Python 0-based p+o+q+2
        lam_idx = p + o + q + 2
        lam = parameters[lam_idx]
        # Apply logistic (sigmoid) function
        # Ref: egarch_itransform.m:54 — lambda=exp(lambda)/(1+exp(lambda))
        lam = np.exp(lam) / (1.0 + np.exp(lam))
        # Scale from (0, 1) to (-0.99, 0.99)
        # Ref: egarch_itransform.m:55 — lambda=1.98*lambda-.99
        lam = 1.98 * lam - 0.99
        parameters[lam_idx] = lam

    return parameters
