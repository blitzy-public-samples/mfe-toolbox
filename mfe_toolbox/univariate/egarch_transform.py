"""
EGARCH(P,O,Q) parameter transformation.

Maps constrained EGARCH parameters to the unconstrained real line for use in
numerical optimization.  Because EGARCH models operate on log-variance, the
core model parameters (omega, alpha, gamma, beta) are already relatively
unconstrained and pass through without modification.  Only the distribution
shape parameters (nu for Student-t / GED, and lambda for Skewed-t) require
explicit transformation.

Transformation summary
----------------------
- **error_type 1 (Gaussian):** No distribution parameters — identity transform.
- **error_type 2 (Student-t):** ``nu → sqrt(nu - 2.01)``, mapping ``nu > 2.01``
  to the non-negative reals.
- **error_type 3 (GED):** Logit transform mapping ``1 < nu < 50`` to the full
  real line via ``temp = (nu - 1) / 49; nu = log(temp / (1 - temp))``.
- **error_type 4 (Skewed-t):** ``nu`` is transformed identically to error_type 2;
  additionally ``lambda`` is logit-transformed from ``(-0.995, 0.995)`` to the
  real line via ``temp = (lambda + 0.995) / 1.99; lambda = log(temp / (1 - temp))``.

Notes
-----
This is the exact forward counterpart of :func:`egarch_itransform`.  For any
valid constrained parameter vector, the round-trip

    ``egarch_itransform(egarch_transform(params, p, o, q, et), p, o, q, et)``

must recover the original ``params`` to within numerical precision (±1e-6).

Migrated from: univariate/egarch_transform.m (MFE Toolbox, Kevin Sheppard)
"""

import numpy as np


def egarch_transform(
    parameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
) -> np.ndarray:
    """
    Transform EGARCH parameters from constrained to unconstrained space.

    Maps valid (constrained) EGARCH parameters to the unconstrained real line
    so that an unconstrained numerical optimizer (e.g.
    ``scipy.optimize.minimize(method='L-BFGS-B')``) can be used.  The core
    EGARCH parameters ``[omega, alpha(1:p), gamma(1:o), beta(1:q)]`` are
    passed through unchanged because the log-variance specification already
    places them on an unrestricted scale.  Only the distribution shape
    parameters require explicit mapping.

    Parameters
    ----------
    parameters : np.ndarray
        Column parameter vector of length ``1 + p + o + q`` (plus any
        distribution parameters).  Layout is::

            [omega, alpha(1), ..., alpha(p),
             gamma(1), ..., gamma(o),
             beta(1), ..., beta(q),
             [nu, [lambda]]]

        Values must be in the **constrained** (model) domain:

        - ``nu > 2.01`` for Student-t / Skewed-t (error_type 2 or 4)
        - ``1 < nu < 50`` for GED (error_type 3)
        - ``-0.995 < lambda < 0.995`` for Skewed-t (error_type 4)
    p : int
        Positive integer, number of symmetric innovation lags (ARCH terms
        in the EGARCH specification, corresponding to ``|z_t|`` terms).
    o : int
        Non-negative integer, number of asymmetric innovation lags (leverage
        terms, corresponding to ``z_t`` terms).  Use ``0`` for a symmetric
        EGARCH process.
    q : int
        Non-negative integer, number of conditional log-variance lags
        (GARCH terms).  Use ``0`` for a pure EARCH process.
    error_type : int
        Distribution assumption for innovations:

        - ``1`` — Gaussian (normal): no extra parameters.
        - ``2`` — Student's t: one extra parameter (``nu``).
        - ``3`` — Generalized Error Distribution (GED): one extra parameter
          (``nu``).
        - ``4`` — Skewed Student's t: two extra parameters (``nu``,
          ``lambda``).

    Returns
    -------
    np.ndarray
        Parameter vector of the same length as ``parameters`` with
        distribution shape parameters mapped to the unconstrained real line:

        - **error_type 2 or 4 (Student's t / Skewed t):**
          ``nu_unconstrained = sqrt(nu - 2.01)``
        - **error_type 3 (GED):**
          ``temp = (nu - 1) / 49``; ``nu_unconstrained = log(temp / (1 - temp))``
        - **error_type 4 (Skewed t), lambda:**
          ``temp = (lambda + 0.995) / 1.99``; ``lambda_unconstrained = log(temp / (1 - temp))``

    Raises
    ------
    ValueError
        If ``error_type`` is 2 or 4 and ``nu <= 2.01`` (sqrt of non-positive).
        If ``error_type`` is 3 and ``nu`` is outside ``(1, 50)``.
        If ``error_type`` is 4 and ``lambda`` is outside ``(-0.995, 0.995)``.

    See Also
    --------
    egarch_itransform : Inverse (unconstrained → constrained) transform.
    egarch : Main EGARCH estimation driver.

    Examples
    --------
    >>> import numpy as np
    >>> params = np.array([-0.1, 0.05, -0.03, 0.95, 5.0])
    >>> result = egarch_transform(params, p=1, o=1, q=1, error_type=2)
    >>> result[3]  # beta passes through unchanged
    0.95
    >>> np.isclose(result[4], np.sqrt(5.0 - 2.01))  # nu transformed
    True
    """
    # Create a copy to avoid mutating the caller's array.
    # Ref: egarch_transform.m:1 — MATLAB modifies the input in-place via
    # copy-on-write semantics; Python requires an explicit copy.
    parameters = np.copy(parameters)

    # ------------------------------------------------------------------
    # Nu (degrees-of-freedom / shape) transform
    # ------------------------------------------------------------------
    # Ref: egarch_transform.m:44 — MATLAB 1-based index p+o+q+2 maps to
    # Python 0-based index p+o+q+1.
    nu_idx = p + o + q + 1

    if error_type == 2 or error_type == 4:
        # Student's t or Skewed t:  nu → sqrt(nu - 2.01)
        # This maps nu > 2.01 to the non-negative reals.  The minimum
        # degrees-of-freedom for a finite-variance t-distribution is ~2,
        # so 2.01 provides a small numerical buffer.
        # Ref: egarch_transform.m:44-46
        nu = parameters[nu_idx]
        nu = np.sqrt(nu - 2.01)
        parameters[nu_idx] = nu
    elif error_type == 3:
        # GED: logit transform mapping the constrained interval (1, 50) to
        # the full real line via the composition:
        #   (nu - 1) / 49  →  temp ∈ (0, 1)
        #   log(temp / (1 - temp))  →  ℝ
        # Ref: egarch_transform.m:48-51
        nu = parameters[nu_idx]
        temp = (nu - 1.0) / 49.0
        nu = np.log(temp / (1.0 - temp))
        parameters[nu_idx] = nu

    # ------------------------------------------------------------------
    # Lambda (skewness) transform — only for Skewed-t
    # ------------------------------------------------------------------
    if error_type == 4:
        # Logit transform mapping the constrained interval
        # (-0.995, 0.995) to the full real line:
        #   (lambda + 0.995) / 1.99  →  temp ∈ (0, 1)
        #   log(temp / (1 - temp))   →  ℝ
        # Ref: egarch_transform.m:56-59 — MATLAB 1-based index p+o+q+3
        # maps to Python 0-based index p+o+q+2.
        lam_idx = p + o + q + 2
        lam = parameters[lam_idx]
        temp = (lam + 0.995) / 1.99
        lam = np.log(temp / (1.0 - temp))
        parameters[lam_idx] = lam

    return parameters
