"""
Parameter transformation for Scalar VT-VECH multivariate GARCH model.

Maps constrained parameters (alpha >= 0, gamma >= 0, beta >= 0, with persistence < 1)
to the real line (-inf, inf) for unconstrained optimization via sequential logit
(inverse logistic) transforms with a shrinking scale.

This module is the forward counterpart of scalar_vt_vech_itransform: the two functions
are mathematical inverses of each other within floating-point precision.

Migrated from: multivariate/scalar_vt_vech_transform.m
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005
"""

import numpy as np


def scalar_vt_vech_transform(
    parameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    kappa: float,
) -> np.ndarray:
    """
    Transform constrained Scalar VT-VECH parameters to the real line.

    Maps constrained parameters from a scalar MVGARCH process to an unconstrained
    space on (-inf, inf) using sequential logit (inverse logistic) transformations.
    Used during the estimation of SCALAR_VT_VECH to allow unconstrained optimization.

    Parameters
    ----------
    parameters : array_like
        Parameter vector of length ``p + o + q`` containing the constrained model
        parameters arranged as ``[alpha(1), ..., alpha(p), gamma(1), ..., gamma(o),
        beta(1), ..., beta(q)]``.  All elements must be non-negative and their
        weighted sum must satisfy the stationarity constraint.
    p : int
        Positive integer representing the number of symmetric innovation
        (ARCH / news) terms.
    o : int
        Non-negative integer representing the number of asymmetric innovation
        terms.
    q : int
        Non-negative integer representing the number of lagged conditional
        variance terms.
    kappa : float
        Positive scalar that adjusts the contribution of asymmetric (gamma) terms
        to the persistence constraint.  Typically ``kappa = E[z^2 * I(z<0)]``
        where ``z`` is the standardised residual.

    Returns
    -------
    tparameters : numpy.ndarray
        1-D array of length ``p + o + q`` containing the transformed parameters
        on the real line (-inf, inf).

    Raises
    ------
    ValueError
        If any element of alpha, gamma, or beta is negative, or if the weighted
        persistence ``sum(alpha) + sum(gamma) / kappa + sum(beta)`` exceeds the
        upper bound (0.999998).

    Notes
    -----
    The transformation proceeds sequentially through the parameter vector using a
    *shrinking-scale logit* approach:

    1. For each alpha_i: normalise by the remaining scale, apply logit, then
       subtract the original alpha_i from the scale.
    2. For each gamma_i: normalise by ``scale * kappa``, apply logit, then
       subtract ``gamma_i / kappa`` from the scale.
    3. For each beta_i: normalise by the remaining scale, apply logit, then
       subtract the original beta_i from the scale.

    The *original* (clamped but untransformed) parameter values are always used for
    updating the scale — never the transformed values.

    The upper bound ``UB = 0.999998`` is set slightly below 1.0 to ensure the logit
    transformation remains numerically stable.  Near-zero parameters are clamped to
    ``1e-8`` to avoid ``log(0)``, and the upper bound is correspondingly adjusted.

    This function is the mathematical inverse of
    :func:`scalar_vt_vech_itransform`.

    Examples
    --------
    >>> import numpy as np
    >>> params = np.array([0.05, 0.02, 0.90])
    >>> scalar_vt_vech_transform(params, p=1, o=0, q=1, kappa=2.0)
    array([...])  # unconstrained values on (-inf, inf)

    See Also
    --------
    scalar_vt_vech_itransform : Inverse transformation (real line → constrained).
    scalar_vt_vech : Main Scalar VT-VECH estimation driver.
    """
    # ------------------------------------------------------------------
    # Phase 1: Input validation and setup
    # Ref: scalar_vt_vech_transform.m:30-32
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()  # Ensure 1-D

    # Upper bound slightly below 1 to keep the logit numerically stable
    UB: float = 0.999998

    # ------------------------------------------------------------------
    # Phase 2: Extract alpha, gamma, beta and validate constraints
    # Ref: scalar_vt_vech_transform.m:36-42
    # ------------------------------------------------------------------
    alpha = parameters[:p].copy()
    gamma = parameters[p:p + o].copy()
    beta = parameters[p + o:p + o + q].copy()

    # Constraint check: all non-negative and persistence < UB
    if (
        np.any(alpha < 0)
        or np.any(beta < 0)
        or np.any(gamma < 0)
        or (np.sum(alpha) + np.sum(gamma) / kappa + np.sum(beta)) >= UB
    ):
        raise ValueError(
            "Parameters do not conform to the necessary set of "
            "restrictions to be transformed."
        )

    # ------------------------------------------------------------------
    # Phase 3: Clamp near-zero values to avoid log(0)
    # Ref: scalar_vt_vech_transform.m:44-49
    # ------------------------------------------------------------------
    alpha[alpha < 1e-8] = 1e-8
    gamma[gamma < 1e-8] = 1e-8
    beta[beta < 1e-8] = 1e-8

    # Adjust UB upward by a small amount to accommodate the clamping
    UB = UB + 1e-8 * (p + o + q)

    # ------------------------------------------------------------------
    # Phase 4: Build combined vectors and initialise transform output
    # Ref: scalar_vt_vech_transform.m:51-55
    # ------------------------------------------------------------------
    # parameters_combined holds the clamped (but untransformed) values and is
    # used exclusively for scale reduction.  tparameters starts as a copy and
    # is overwritten element-by-element with the logit-transformed values.
    parameters_combined = np.concatenate([alpha, gamma, beta])
    tparameters = parameters_combined.copy()

    # Remaining "budget" for the sequential logit
    scale: float = UB

    # ------------------------------------------------------------------
    # Phase 5: Sequential inverse-logistic (logit) for alpha terms
    # Ref: scalar_vt_vech_transform.m:56-63
    # MATLAB uses 1-indexed i=1:p; Python uses 0-indexed range(p)
    # ------------------------------------------------------------------
    for i in range(p):
        # Normalise the clamped value into (0, 1) by dividing by the
        # remaining scale
        tparameters[i] = tparameters[i] / scale
        # Apply the logit (inverse logistic) transform to map (0,1) → (-inf,inf)
        tparameters[i] = np.log(tparameters[i] / (1.0 - tparameters[i]))
        # Reduce the scale by the *original* (clamped, untransformed) value
        scale = scale - parameters_combined[i]

    # ------------------------------------------------------------------
    # Phase 6: Sequential inverse-logistic for gamma terms (with kappa)
    # Ref: scalar_vt_vech_transform.m:64-71
    # Gamma terms are divided by both scale AND kappa during normalisation,
    # and the scale is reduced by gamma_i / kappa (not gamma_i directly).
    # ------------------------------------------------------------------
    for i in range(p, p + o):
        # Normalise: divide by scale *and* kappa
        tparameters[i] = tparameters[i] / scale / kappa
        # Logit transform
        tparameters[i] = np.log(tparameters[i] / (1.0 - tparameters[i]))
        # Reduce scale by the kappa-adjusted original value
        # Ref: scalar_vt_vech_transform.m:70 — scale=scale-parameters(i)/kappa
        scale = scale - parameters_combined[i] / kappa

    # ------------------------------------------------------------------
    # Phase 7: Sequential inverse-logistic for beta terms
    # Ref: scalar_vt_vech_transform.m:73-80
    # ------------------------------------------------------------------
    for i in range(p + o, p + o + q):
        # Normalise by remaining scale
        tparameters[i] = tparameters[i] / scale
        # Logit transform
        tparameters[i] = np.log(tparameters[i] / (1.0 - tparameters[i]))
        # Reduce scale by the original (clamped, untransformed) value
        scale = scale - parameters_combined[i]

    return tparameters
