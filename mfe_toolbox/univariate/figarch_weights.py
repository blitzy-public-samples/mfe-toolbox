"""
FIGARCH ARCH(infinity) truncation weight computation.

Computes the ARCH(infinity) representation weights (lambda) for FIGARCH(P,d,Q)
models where P in {0, 1} and Q in {0, 1}. These weights define the long-memory
ARCH structure used in the FIGARCH conditional variance recursion.

The conditional variance h(t) of a FIGARCH(1,d,1) process is modeled as:

    h(t) = omega + [1 - beta*L - phi*L*(1-L)^d] * epsilon^2(t) + beta * h(t-1)

which is estimated using an ARCH(infinity) representation:

    h(t) = omega + sum_{i=1}^{truncLag} lambda(i) * epsilon^2(t-i)

where lambda(i) is a function of the fractional differencing parameter d,
the AR parameter phi, and the MA parameter beta.

Migrated from: univariate/figarch_weights.m (Version 4.0, Kevin Sheppard)

See Also
--------
figarch : FIGARCH model driver
figarch_likelihood : FIGARCH log-likelihood
figarch_parameter_check : FIGARCH parameter validation
figarch_starting_values : FIGARCH initialization
figarch_transform : FIGARCH parameter transformation
figarch_itransform : FIGARCH inverse parameter transformation
"""

# Copyright: Kevin Sheppard
# kevin.sheppard@economics.ox.ac.uk
# Revision: 1    Date: 7/13/2009
# Python migration: MATLAB-to-Python 3.12

import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def figarch_weights(parameters: np.ndarray, p: int, q: int, truncLag: int) -> np.ndarray:
    """
    Compute FIGARCH ARCH(infinity) truncation weights for P={0,1} and Q={0,1}.

    Recursively computes the lambda weights that define the long-memory ARCH
    representation of a FIGARCH(P,d,Q) model. The weights are used to express
    the conditional variance as a weighted sum of past squared innovations.

    Parameters
    ----------
    parameters : np.ndarray
        A (1 + P + Q) element array of parameters in the order [phi, d, beta]
        where phi is omitted if P=0 and beta is omitted if Q=0. The fractional
        differencing parameter d is always present.
    p : int
        0 or 1 indicating whether the autoregressive term (phi) is present
        in the model.
    q : int
        0 or 1 indicating whether the moving average term (beta) is present
        in the model.
    truncLag : int
        Number of weights to compute in the ARCH(infinity) representation.
        Controls the truncation length of the infinite-order representation.

    Returns
    -------
    np.ndarray
        A (truncLag,) array of ARCH(infinity) weights (lambda). These weights
        are used in the FIGARCH variance equation to compute conditional
        variance as a weighted sum of past squared residuals.

    Notes
    -----
    The parameter vector ordering depends on the model specification:

    - FIGARCH(0,d,0): parameters = [d]
    - FIGARCH(1,d,0): parameters = [phi, d]
    - FIGARCH(0,d,1): parameters = [d, beta]
    - FIGARCH(1,d,1): parameters = [phi, d, beta]

    The recursive computation follows the FIGARCH weight recursion:

    - delta[0] = d
    - delta[i] = (i - d) / (i + 1) * delta[i-1]   for i = 1, ..., truncLag-1
    - lambda[0] = phi - beta + d
    - lambda[i] = beta * lambda[i-1] + (delta[i] - phi * delta[i-1])

    The delta recursion computes the coefficients of the fractional differencing
    operator (1-L)^d, and the lambda weights combine these with the GARCH
    AR and MA parameters.

    References
    ----------
    Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
    Fractionally integrated generalized autoregressive conditional
    heteroskedasticity. Journal of Econometrics, 74(1), 3-30.
    """
    # --- Parse parameters based on model specification ---
    # Ref: figarch_weights.m:38-50 — parameter extraction depends on p, q flags
    # Parameter vector order: [phi (if p), d, beta (if q)]

    # Extract beta (MA parameter) if q > 0
    # Ref: figarch_weights.m:39 — MATLAB 1-based: parameters(2+p) → Python 0-based: parameters[1+p]
    if q:
        beta = parameters[1 + p]
    else:
        beta = 0.0

    # Extract phi (AR parameter) and d (fractional differencing parameter)
    # Ref: figarch_weights.m:44-49 — phi only present when p > 0; d position shifts
    if p:
        # Ref: figarch_weights.m:45-46 — MATLAB parameters(1), parameters(2)
        phi = parameters[0]
        d = parameters[1]
    else:
        phi = 0.0
        # Ref: figarch_weights.m:49 — when no phi, d is the first parameter
        d = parameters[0]

    # --- Recursive weight computation ---
    # Ref: figarch_weights.m:53-61
    # Allocate output arrays
    lambda_ = np.zeros(truncLag)
    delta = np.zeros(truncLag)

    # Initial values (index 0 in Python corresponds to index 1 in MATLAB)
    # Ref: figarch_weights.m:56 — lambda(1) = phi - beta + d
    lambda_[0] = phi - beta + d
    # Ref: figarch_weights.m:57 — delta(1) = d
    delta[0] = d

    # Recursive computation of remaining weights
    # Ref: figarch_weights.m:58 — MATLAB loop: for i=2:truncLag
    # Python loop: for i in range(1, truncLag) where Python i corresponds to MATLAB i+1
    #
    # Index translation verification:
    #   MATLAB i=2: delta(2) = (2-1-d)/2 * delta(1) = (1-d)/2 * delta(1)
    #   Python i=1: delta[1] = (1-d)/(1+1) * delta[0] = (1-d)/2 * delta[0] ✓
    #
    #   MATLAB i=3: delta(3) = (3-1-d)/3 * delta(2) = (2-d)/3 * delta(2)
    #   Python i=2: delta[2] = (2-d)/(2+1) * delta[1] = (2-d)/3 * delta[1] ✓
    for i in range(1, truncLag):
        # Ref: figarch_weights.m:59 — delta(i) = (i-1-d)/i * delta(i-1)
        # Python 0-based: delta[i] = (i-d)/(i+1) * delta[i-1]
        delta[i] = (i - d) / (i + 1) * delta[i - 1]
        # Ref: figarch_weights.m:60 — lambda(i) = beta*lambda(i-1) + (delta(i) - phi*delta(i-1))
        lambda_[i] = beta * lambda_[i - 1] + (delta[i] - phi * delta[i - 1])

    return lambda_
