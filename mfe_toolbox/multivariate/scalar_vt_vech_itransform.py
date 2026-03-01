"""
Inverse parameter transformation for the Scalar VT-VECH multivariate GARCH model.

Maps parameters from the unconstrained real line back to the constrained
parameter space required for a valid Scalar VT-VECH process. This is the
inverse of :func:`scalar_vt_vech_transform`.

The output parameters satisfy:
    1. alpha(i) >= 0 for i = 1, ..., p   (symmetric innovation weights)
    2. gamma(j) >= 0 for j = 1, ..., o   (asymmetric innovation weights)
    3. beta(k)  >= 0 for k = 1, ..., q   (lagged covariance weights)
    4. sum(alpha) + sum(gamma)/kappa + sum(beta) < 1  (stationarity)

Source reference: multivariate/scalar_vt_vech_itransform.m
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005
"""

import numpy as np


def scalar_vt_vech_itransform(
    tparameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    kappa: float,
) -> np.ndarray:
    """
    Inverse parameter transformation for Scalar VT-VECH.

    Maps unconstrained transformed parameters (on the real line) back to the
    constrained parameter space using a sequential logistic sigmoid rescaling
    scheme. This is the inverse of ``scalar_vt_vech_transform``.

    Parameters
    ----------
    tparameters : np.ndarray
        Column vector of transformed (unconstrained) parameters on (-inf, inf).
        Length must be at least p + o + q.
    p : int
        Positive integer. Number of symmetric innovation lags (alpha parameters).
    o : int
        Non-negative integer. Number of asymmetric innovation lags (gamma parameters).
    q : int
        Non-negative integer. Number of lagged conditional covariance terms (beta parameters).
    kappa : float
        Positive scalar. Asymmetry scaling factor derived from the eigenvalue
        structure of the unconditional covariance matrix. Used to adjust the
        budget allocation for gamma (asymmetric) parameters.

    Returns
    -------
    np.ndarray
        1-D array of constrained parameters:
        ``[alpha(1), ..., alpha(p), gamma(1), ..., gamma(o), beta(1), ..., beta(q)]``
        satisfying all positivity and stationarity constraints.

    Notes
    -----
    The transformation proceeds in four steps:

    1. **Clamping**: Values exceeding 50 are clamped to prevent ``exp`` overflow.
    2. **Logistic sigmoid**: Each element is mapped from (-inf, inf) to (0, 1)
       via ``exp(t) / (1 + exp(t))``.
    3. **Sequential rescaling**: Parameters are rescaled sequentially so that
       their weighted sum stays below the upper bound UB = 0.999998.
       - Alpha parameters consume ``parameters[i] * scale`` of the remaining budget.
       - Gamma parameters consume ``parameters[i] * scale * kappa`` and reduce
         the scale by ``parameters[i] / kappa``.
       - Beta parameters consume ``parameters[i] * scale`` of the remaining budget.
    4. The output is a 1-D numpy array of the constrained parameters.

    This function is the exact inverse of :func:`scalar_vt_vech_transform`:
    ``transform(itransform(x)) ≈ x`` (within floating-point precision).

    References
    ----------
    Ref: scalar_vt_vech_itransform.m (Kevin Sheppard, MFE Toolbox v4.0)
    """
    # Ref: scalar_vt_vech_itransform.m:31 — Upper bound to keep parameters
    # a bit away from 1 for numerical stability
    UB = 0.999998

    # Copy input to avoid mutating caller's data
    # Ref: MATLAB does not mutate input arrays; Python numpy arrays are mutable
    tparameters = tparameters.copy()

    # Ensure 1-D array for consistent indexing
    tparameters = tparameters.ravel()

    # Ref: scalar_vt_vech_itransform.m:33 — Clamp extreme values to prevent
    # overflow in exp(). MATLAB: tparameters(tparameters>50)=50
    tparameters[tparameters > 50] = 50

    # Ref: scalar_vt_vech_itransform.m:35 — Logistic sigmoid: map from
    # (-inf, inf) to (0, 1). Uses manual exp(t)/(1+exp(t)) rather than
    # scipy.special.expit to match MATLAB exactly.
    # MATLAB: parameters = exp(tparameters)./(1+exp(tparameters))
    parameters = np.exp(tparameters) / (1.0 + np.exp(tparameters))

    # --- Sequential rescaling ---
    # Each parameter group (alpha, gamma, beta) is scaled so that the
    # weighted sum stays below UB, ensuring stationarity.

    # Ref: scalar_vt_vech_itransform.m:37-41 — Rescale alpha (symmetric
    # innovation) parameters. MATLAB 1-indexed loop for i=1:p becomes
    # Python 0-indexed range(p).
    scale = UB
    for i in range(p):
        # Ref: scalar_vt_vech_itransform.m:39 — parameters(i) = parameters(i)*scale
        parameters[i] = parameters[i] * scale
        # Ref: scalar_vt_vech_itransform.m:40 — scale = scale - parameters(i)
        scale = scale - parameters[i]

    # Ref: scalar_vt_vech_itransform.m:42-45 — Rescale gamma (asymmetric
    # innovation) parameters. Multiply by scale*kappa for the parameter
    # value, but subtract parameters[i]/kappa from the scale budget.
    # MATLAB 1-indexed loop for i=p+1:p+o becomes Python range(p, p+o).
    for i in range(p, p + o):
        # Ref: scalar_vt_vech_itransform.m:43 — parameters(i) = parameters(i)*scale*kappa
        parameters[i] = parameters[i] * scale * kappa
        # Ref: scalar_vt_vech_itransform.m:44 — scale = scale - parameters(i)/kappa
        scale = scale - parameters[i] / kappa

    # Ref: scalar_vt_vech_itransform.m:46-49 — Rescale beta (lag)
    # parameters. Same scale-and-subtract pattern as alpha.
    # MATLAB 1-indexed loop for i=p+o+1:p+o+q becomes Python range(p+o, p+o+q).
    for i in range(p + o, p + o + q):
        # Ref: scalar_vt_vech_itransform.m:47 — parameters(i) = parameters(i)*scale
        parameters[i] = parameters[i] * scale
        # Ref: scalar_vt_vech_itransform.m:48 — scale = scale - parameters(i)
        scale = scale - parameters[i]

    return parameters
