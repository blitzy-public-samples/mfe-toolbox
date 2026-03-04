"""
Non-linear constraints for RARCH(p,q) multivariate volatility model estimation.

Provides the stationarity constraint function required by ``scipy.optimize.minimize``
during RARCH model estimation.  The constraint ensures that the persistence of the
variance dynamics is bounded below unity, which is a necessary condition for
covariance stationarity of the RARCH process.

Three model parameterisations are handled:

    * **Scalar** (type_model=1) — Single scalar constraint:
      ``sum(a_i^2, i=1..p) + sum(b_j^2, j=1..q) < 0.99998``
    * **Common Persistence / CP** (type_model=2) — Per-element constraint:
      ``diag(sum(A_i^2, i=1..p)) < theta`` where ``theta`` is the squared
      common-persistence parameter (last element of the parameter vector).
    * **Diagonal** (type_model=3) — K-element vector constraint:
      ``diag(sum(A_i^2, i=1..p) + sum(B_j^2, j=1..q)) < 0.99998``

All constraints follow the ``scipy.optimize`` convention ``c >= 0`` for
inequality constraints (feasible when non-negative).  Values are therefore
**negated** relative to the MATLAB source (where ``fmincon`` uses ``c <= 0``).
Equality constraints are always empty.

Source Reference
----------------
Migrated from ``multivariate/rarch_constraint.m`` (35 lines).
Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 1, Date: 3/27/2012

See Also
--------
mfe_toolbox.multivariate.rarch : RARCH model driver
mfe_toolbox.multivariate.rarch_likelihood : RARCH log-likelihood
mfe_toolbox.multivariate.rarch_parameter_transform : RARCH parameter unpacking
mfe_toolbox.multivariate.rarch_simulate : RARCH simulation
"""

from __future__ import annotations

import numpy as np

from mfe_toolbox.multivariate.rarch_parameter_transform import rarch_parameter_transform


def rarch_constraint(
    parameters: np.ndarray,
    p: int,
    q: int,
    k: int,
    type_model: int,
    C: np.ndarray | None = None,
    is_joint: bool = False,
    is_c_chol: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute nonlinear inequality constraints for RARCH(p,q) model estimation.

    The function unpacks the flat parameter vector into (C, A, B) coefficient
    matrices via :func:`rarch_parameter_transform`, then computes the
    stationarity inequality constraint appropriate for the chosen model type.

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D vector of model parameters.  Layout depends on ``is_joint`` and
        ``type_model`` — see :func:`rarch_parameter_transform` for details.
    p : int
        Number of symmetric innovation lags (A matrices).  Must be >= 0.
    q : int
        Number of conditional covariance lags (B matrices).  Must be >= 0.
    k : int
        Number of assets (dimension of the covariance matrices).  Must be >= 1.
    type_model : int
        Model type indicator:

        * 1 — Scalar RARCH
        * 2 — Common Persistence (CP) RARCH
        * 3 — Diagonal RARCH

    C : numpy.ndarray or None, optional
        K x K unconditional covariance matrix.  When ``is_joint`` is False,
        this matrix is passed through to ``rarch_parameter_transform``.  When
        ``is_joint`` is True, C is reconstructed from the parameter vector and
        this argument is ignored.  Defaults to ``np.eye(k)`` if not provided.
    is_joint : bool, optional
        If True, the parameter vector includes the covariance intercept
        parameters in addition to the dynamics parameters.  Default is False.
    is_c_chol : bool, optional
        If True **and** ``is_joint`` is True, the intercept parameters are
        interpreted as the Cholesky factor of C.  Default is False.

    Returns
    -------
    c : numpy.ndarray
        Inequality constraint vector (scipy convention: ``c >= 0`` = feasible).
        All values are **negated** relative to the MATLAB source.

        * Scalar (type_model=1): 1-element array.
        * CP (type_model=2): k-element array.
        * Diagonal (type_model=3): k-element array.

    ceq : numpy.ndarray
        Empty array — RARCH has no equality constraints.

    Raises
    ------
    ValueError
        If ``type_model`` is not in {1, 2, 3}, or if ``p``, ``q``, or ``k``
        have invalid values.

    Notes
    -----
    The 0.99998 margin (instead of 1.0) in the Scalar and Diagonal constraints
    provides a small numerical buffer to ensure strict stationarity during
    optimisation, preventing the optimizer from reaching the exact boundary.

    For the CP model (type_model=2), the constraint ensures that the
    per-element squared A coefficients (summed across lags) do not exceed
    the common persistence parameter ``theta = parameters[-1]**2``.  This
    is equivalent to requiring that the reparameterised B diagonal entries
    remain non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.rarch_constraint import rarch_constraint
    >>> # Scalar RARCH(1,1) with k=3 assets
    >>> params = np.array([0.3, 0.6])  # a, b parameters
    >>> c, ceq = rarch_constraint(params, p=1, q=1, k=3, type_model=1)
    >>> c  # 0.99998 - sum(a^2) - sum(b^2) should be >= 0 for feasibility
    array([0.54998])
    >>> ceq.size == 0
    True
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    if type_model not in (1, 2, 3):
        raise ValueError(
            f"type_model must be 1 (Scalar), 2 (CP), or 3 (Diagonal), "
            f"got {type_model}"
        )
    if p < 0:
        raise ValueError(f"p must be non-negative, got {p}")
    if q < 0:
        raise ValueError(f"q must be non-negative, got {q}")
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")

    # ------------------------------------------------------------------
    # Ref: rarch_constraint.m:22 — ceq is always empty (no equality constraints)
    # ------------------------------------------------------------------
    ceq = np.array([])

    # Default C to identity if not provided.
    # When is_joint=False, the actual value of C does not affect A and B
    # extraction inside rarch_parameter_transform, so identity is a safe default.
    if C is None:
        C = np.eye(k)

    # ------------------------------------------------------------------
    # Ref: rarch_constraint.m:24 — unpack parameter vector into matrices
    # MATLAB: [~,A,B] = rarch_parameter_transform(parameters,p,q,k,C,type,isJoint,isCChol)
    # ------------------------------------------------------------------
    _, A, B = rarch_parameter_transform(
        parameters, p, q, k, C, type_model, is_joint, is_c_chol
    )

    # ------------------------------------------------------------------
    # Compute inequality constraints by model type
    # ------------------------------------------------------------------
    # A has shape (k, k, p).  np.sum(A**2, axis=2) collapses the p-lag
    # dimension, yielding a (k, k) matrix.  np.diag() then extracts the k
    # diagonal entries.
    #
    # For Scalar and Diagonal models, B has shape (k, k, q) — same logic.
    # For CP model (type_model=2 with q>0), B is returned as 2-D (k, k) by
    # rarch_parameter_transform (after reparameterisation), so axis=2 sum
    # is NOT applicable.  However, the CP branch below does not use B at all,
    # so no special handling is needed.
    # ------------------------------------------------------------------

    if type_model == 1:
        # ------------------------------------------------------------------
        # Scalar RARCH — single constraint element
        # Ref: rarch_constraint.m:25-28
        # MATLAB: constraint = diag(sum(A.^2,3) + sum(B.^2,3) - 0.99998)
        #         c = constraint(1)
        # For Scalar model, A(:,:,i) = a_i * I_k and B(:,:,j) = b_j * I_k,
        # so all diagonal entries of the constraint are identical.  We take
        # only the first.
        # ------------------------------------------------------------------
        A_sq_sum = np.sum(A ** 2, axis=2)  # (k, k)
        B_sq_sum = np.sum(B ** 2, axis=2)  # (k, k)
        constraint = np.diag(A_sq_sum + B_sq_sum - 0.99998)  # k-vector
        # Ref: rarch_constraint.m:28 — MATLAB uses constraint(1) (1-indexed)
        # Python 0-indexed equivalent: constraint[0]
        # Negate for scipy convention: MATLAB c <= 0 → scipy c >= 0
        c = -np.array([constraint[0]])

    elif type_model == 2:
        # ------------------------------------------------------------------
        # Common Persistence (CP) RARCH — per-element A^2 constraint
        # Ref: rarch_constraint.m:30-31
        # MATLAB: theta = parameters(length(parameters)).^2
        #         c = diag(sum(A.^2,3)) - theta
        # Each diagonal element of the summed A^2 matrix must be less than
        # theta (the squared common-persistence parameter).  This ensures
        # the reparameterised B diagonal entries remain non-negative.
        # ------------------------------------------------------------------
        # Ref: rarch_constraint.m:30 — parameters(end) maps to parameters[-1]
        theta = parameters[-1] ** 2
        A_sq_sum = np.sum(A ** 2, axis=2)  # (k, k)
        # Negate for scipy convention: MATLAB c <= 0 → scipy c >= 0
        c = -(np.diag(A_sq_sum) - theta)  # k-vector; each entry >= 0 when feasible

    elif type_model == 3:
        # ------------------------------------------------------------------
        # Diagonal RARCH — full k-element constraint vector
        # Ref: rarch_constraint.m:25,33
        # MATLAB: constraint = diag(sum(A.^2,3) + sum(B.^2,3) - 0.99998)
        #         c = constraint
        # Each asset's persistence must be bounded below 0.99998.
        # ------------------------------------------------------------------
        A_sq_sum = np.sum(A ** 2, axis=2)  # (k, k)
        B_sq_sum = np.sum(B ** 2, axis=2)  # (k, k)
        constraint = np.diag(A_sq_sum + B_sq_sum - 0.99998)  # k-vector
        # Negate for scipy convention: MATLAB c <= 0 → scipy c >= 0
        c = -constraint

    return c, ceq
