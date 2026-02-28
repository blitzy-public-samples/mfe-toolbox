"""
APARCH(P,O,Q) inverse parameter transformation.

Maps parameters from the unconstrained real line to the constrained parameter
space appropriate for an APARCH model. Used during APARCH estimation to convert
optimizer parameters (which live on the entire real line) back to valid,
constrained model parameters.

This module is the exact inverse of ``aparch_transform``: for each
constrained-to-unconstrained mapping in ``aparch_transform``, this module
applies the corresponding unconstrained-to-constrained mapping.

Migrated from: univariate/aparch_itransform.m
Original Author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
"""

import numpy as np


def aparch_itransform(
    parameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
    delta_is_estimated: int = 1,
) -> tuple[np.ndarray, float | None, float | None]:
    """
    APARCH(P,O,Q) inverse parameter transformation.

    Maps unconstrained real-line parameters to the constrained parameter space
    for an Asymmetric Power ARCH model.  Called internally during APARCH
    estimation by ``aparch_likelihood`` when ``estim_flag`` is ``True``.

    Parameters
    ----------
    parameters : np.ndarray
        1-D parameter vector in the unconstrained (optimizer) space.
        Length must be at least
        ``1 + p + o + q + delta_is_estimated``
        plus additional elements for distribution parameters depending
        on ``error_type``.
    p : int
        Positive integer — number of symmetric innovation (ARCH) terms.
    o : int
        Non-negative integer — number of asymmetric innovation terms.
        Set to 0 for a symmetric model.
    q : int
        Non-negative integer — number of lagged conditional-variance
        (GARCH) terms.  Set to 0 for an ARCH-only model.
    error_type : int
        Distribution type for the innovation process:

        * 1 — Gaussian (Normal) innovations
        * 2 — Student's *t* distributed errors
        * 3 — Generalized Error Distribution (GED)
        * 4 — Hansen's Skewed Student's *t* distribution
    delta_is_estimated : int, optional
        ``1`` if the power parameter delta is jointly estimated (default),
        ``0`` if a user-supplied value of delta is provided externally.

    Returns
    -------
    trans_parameters : np.ndarray
        Constrained parameter vector of length
        ``1 + p + o + q + delta_is_estimated``, ordered as::

            [omega, alpha(1), …, alpha(p),
             gamma(1), …, gamma(o),
             beta(1), …, beta(q),
             delta]          # included only when delta_is_estimated == 1

    nu : float or None
        Distribution shape (kurtosis) parameter.

        * ``None`` when ``error_type == 1`` (Gaussian).
        * ``> 2.01`` when ``error_type`` is 2 or 4 (Student's *t* / Skewed *t*).
        * In ``(1.01, 50.01)`` when ``error_type == 3`` (GED).
    lambda_param : float or None
        Distribution asymmetry parameter.

        * ``None`` unless ``error_type == 4`` (Skewed *t*).
        * In ``(-0.99, 0.99)`` when applicable.

    Notes
    -----
    The constrained parameter space satisfies:

    1. ``omega > 0``
    2. ``alpha(i) >= 0``  for ``i = 1, …, p``
    3. ``-1 < gamma(i) < 1``  for ``i = 1, …, o``
    4. ``beta(i) >= 0``  for ``i = 1, …, q``
    5. ``sum(alpha) + sum(beta) < 0.9998``
    6. ``0.3 < delta < 4.0``  (when jointly estimated)
    7. ``nu > 2.01`` for Student's *t*; ``1.01 < nu < 50.01`` for GED
    8. ``-0.99 < lambda < 0.99`` for Skewed *t*

    See Also
    --------
    aparch_transform : Forward (constrained → unconstrained) transform.
    aparch : Main APARCH estimation driver.
    """
    # ------------------------------------------------------------------
    # Initialise distribution parameters to None (MATLAB: nu=[], lambda=[])
    # ------------------------------------------------------------------
    nu: float | None = None
    lambda_param: float | None = None

    # Convert input to a writable float64 1-D array, making a copy so that
    # the caller's data is never mutated.
    # Ref: aparch_itransform.m — column vector expected
    parameters = np.array(np.atleast_1d(parameters), dtype=np.float64).copy()

    # ------------------------------------------------------------------
    # Overflow protection — cap all values at 100 to prevent exp() overflow
    # Ref: aparch_itransform.m:54 — parameters(parameters>100)=100
    # ------------------------------------------------------------------
    parameters = np.clip(parameters, None, 100.0)

    # Ensure delta_is_estimated is an integer for index arithmetic
    delta_is_estimated = int(delta_is_estimated)

    # ------------------------------------------------------------------
    # Parse parameter vector  (0-based indexing)
    # Ref: aparch_itransform.m:57-63 — MATLAB 1-based → Python 0-based
    # ------------------------------------------------------------------
    # omega is the first element
    # Ref: aparch_itransform.m:57 — omega=parameters(1)
    omega = parameters[0]

    # alpha parameters: indices 1 … p   (MATLAB: 2 … p+1)
    # Ref: aparch_itransform.m:58 — alpha=parameters(2:p+1)
    alpha = parameters[1 : p + 1].copy()

    # gamma (asymmetry) parameters: indices p+1 … p+o   (MATLAB: p+2 … p+o+1)
    # Ref: aparch_itransform.m:59 — gamma=parameters(p+2:p+o+1)
    gamma = parameters[p + 1 : p + o + 1].copy()

    # beta parameters: indices p+o+1 … p+o+q   (MATLAB: p+o+2 … p+o+q+1)
    # Ref: aparch_itransform.m:60 — beta=parameters(p+o+2:p+o+q+1)
    beta = parameters[p + o + 1 : p + o + q + 1].copy()

    # delta (power parameter) — only when jointly estimated
    if delta_is_estimated:
        # Ref: aparch_itransform.m:62 — delta=parameters(p+o+q+2)
        delta = parameters[p + o + q + 1]

    # ------------------------------------------------------------------
    # Distribution shape parameter  (nu)
    # Ref: aparch_itransform.m:66-78
    # ------------------------------------------------------------------
    if error_type == 2 or error_type == 4:
        # Student's t  /  Skewed t :  nu = 2.01 + x²  maps ℝ → (2.01, ∞)
        # Ref: aparch_itransform.m:68-69 — nu=2.01+nu^2
        nu_raw = float(parameters[p + o + q + 1 + delta_is_estimated])
        nu = 2.01 + nu_raw ** 2

    elif error_type == 3:
        # GED : logistic maps ℝ → (1.01, 50.01)
        # Ref: aparch_itransform.m:72-77
        nu_raw = float(parameters[p + o + q + 1 + delta_is_estimated])
        # Additional overflow guard (redundant after the global clip,
        # but preserved for faithfulness to the MATLAB source).
        # Ref: aparch_itransform.m:73-74 — if nu>100; nu=100; end
        if nu_raw > 100.0:
            nu_raw = 100.0
        sigmoid_nu = np.exp(nu_raw) / (1.0 + np.exp(nu_raw))
        nu = 49.0 * float(sigmoid_nu) + 1.01

    # ------------------------------------------------------------------
    # Distribution asymmetry parameter  (lambda)
    # Ref: aparch_itransform.m:80-84
    # ------------------------------------------------------------------
    if error_type == 4:
        # Skewed t : logistic maps ℝ → (-0.99, 0.99)
        # Ref: aparch_itransform.m:81-83
        lam_raw = float(parameters[p + o + q + 2 + delta_is_estimated])
        sigmoid_lam = np.exp(lam_raw) / (1.0 + np.exp(lam_raw))
        lambda_param = 1.98 * float(sigmoid_lam) - 0.99

    # ==================================================================
    # Transform model parameters to the constrained space
    # ==================================================================

    # Upper bound for the sequential alpha / beta logistic transforms.
    # Ensures sum(alpha) + sum(beta) < UB.
    # Ref: aparch_itransform.m:87 — UB=.9998
    upper_bound = 0.9998

    # ---- Omega : exp transform → omega > 0 ---------------------------
    # Ref: aparch_itransform.m:90 — tomega=exp(omega)
    tomega = np.exp(omega)

    # ---- Gamma : bounded logistic → (-0.9995, 0.9995) ⊂ (-1, 1) -----
    # Ref: aparch_itransform.m:95
    #   tgamma = 1.999*(exp(gamma)./(1+exp(gamma)))-.9995
    tgamma = 1.999 * (np.exp(gamma) / (1.0 + np.exp(gamma))) - 0.9995

    # ---- Delta : logistic → (0.3, 4.0) -------------------------------
    # Ref: aparch_itransform.m:96-100
    if delta_is_estimated:
        tdelta = np.atleast_1d(
            0.3 + 3.7 * np.exp(delta) / (1.0 + np.exp(delta))
        )
    else:
        tdelta = np.array([], dtype=np.float64)

    # ---- Alpha : sequential logistic with shrinking scale -------------
    # Each alpha(i) ∈ [0, remaining_scale) so that sum(alpha) < UB.
    # Ref: aparch_itransform.m:102-109
    scale = upper_bound
    talpha = alpha.copy()
    for i in range(p):
        # Ref: aparch_itransform.m:106
        #   talpha(i)=(exp(talpha(i))/(1+exp(talpha(i))))*scale
        sigmoid_val = np.exp(talpha[i]) / (1.0 + np.exp(talpha[i]))
        talpha[i] = sigmoid_val * scale
        # Reduce remaining headroom for subsequent parameters
        # Ref: aparch_itransform.m:108 — scale=scale-talpha(i)
        scale = scale - talpha[i]

    # ---- Beta : sequential logistic continuing from alpha's scale -----
    # Ref: aparch_itransform.m:110-115
    tbeta = beta.copy()
    for i in range(q):
        # Ref: aparch_itransform.m:112
        #   tbeta(i)=(exp(tbeta(i))/(1+exp(tbeta(i))))*scale
        sigmoid_val = np.exp(tbeta[i]) / (1.0 + np.exp(tbeta[i]))
        tbeta[i] = sigmoid_val * scale
        # Ref: aparch_itransform.m:114 — scale=scale-tbeta(i)
        scale = scale - tbeta[i]

    # ==================================================================
    # Assemble the constrained parameter vector
    # Ref: aparch_itransform.m:117
    #   trans_parameters=[tomega;talpha;tgamma;tbeta;tdelta]
    # ==================================================================
    trans_parameters = np.concatenate([
        np.atleast_1d(tomega),
        talpha,
        tgamma,
        tbeta,
        tdelta,
    ])

    # ------------------------------------------------------------------
    # Safety check — all parameters must be real-valued
    # Ref: aparch_itransform.m:118-120 — MATLAB 'keyboard' → ValueError
    # ------------------------------------------------------------------
    if np.any(~np.isreal(trans_parameters)):
        raise ValueError(
            "Inverse-transformed parameters contain complex values, "
            "indicating invalid unconstrained input parameters."
        )

    return trans_parameters, nu, lambda_param
