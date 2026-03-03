"""
Non-linear constraint function for RCC (Rotated Conditional Correlation)
multivariate volatility model estimation.

Computes stationarity inequality constraints on the A and B coefficient
matrices of the RARCH dynamics embedded in the RCC model.  The constraints
ensure that the sum of squared diagonal coefficients remains strictly below
unity (threshold 0.99998), which is a necessary condition for covariance
stationarity.

Three RARCH model types are supported:

    * **Scalar** (type_model=1) — single constraint on the common persistence
    * **Common Persistence / CP** (type_model=2) — per-asset constraint
      relative to the shared theta parameter
    * **Diagonal** (type_model=3) — per-asset independent constraints

Source Reference
----------------
Migrated from ``multivariate/rcc_constraint.m`` (40 lines).
Related reference: ``multivariate/rarch_constraint.m`` (identical constraint
structure without stage-based parameter slicing).

Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 3/27/2012

See Also
--------
mfe_toolbox.multivariate.rcc : RCC model driver
mfe_toolbox.multivariate.rcc_likelihood : RCC log-likelihood
mfe_toolbox.multivariate.rarch_parameter_transform : parameter unpacking
mfe_toolbox.multivariate.rarch_constraint : RARCH constraint (shared logic)
"""

from __future__ import annotations

import numpy as np

from mfe_toolbox.multivariate.rarch_parameter_transform import (
    rarch_parameter_transform,
)


def rcc_constraint(
    parameters: np.ndarray,
    m: int,
    n: int,
    k: int,
    type_model: int,
    stage: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute non-linear inequality constraints for RCC model estimation.

    This function is intended to be used as the constraint callback within
    ``scipy.optimize.minimize`` when estimating an RCC model.  It ensures
    that the RARCH dynamics embedded in the RCC correlation process satisfy
    the covariance stationarity condition.

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D optimization parameter vector.  When ``stage != 3``, the first
        ``k*(k-1)/2`` elements encode correlation rotation parameters and are
        stripped before computing dynamics constraints.
    m : int
        RARCH order p — number of symmetric innovation lags (A matrices).
    n : int
        RARCH order q — number of conditional covariance lags (B matrices).
    k : int
        Number of assets (dimension of covariance matrices).  Must be >= 2
        for multivariate models.
    type_model : int
        Model type indicator:

        * 1 — Scalar
        * 2 — Common Persistence (CP)
        * 3 — Diagonal

    stage : int, optional
        Estimation stage indicator (default 3).

        * ``stage == 3``: all parameters belong to the dynamics — use the
          full parameter vector.
        * ``stage != 3``: the first ``k*(k-1)/2`` elements are correlation
          rotation parameters that must be skipped.

    Returns
    -------
    c : numpy.ndarray
        Vector of inequality constraints in **scipy convention** (c >= 0
        implies feasibility).  The values are the *negation* of the MATLAB
        ``fmincon`` convention (where c <= 0 implies feasibility).
    ceq : numpy.ndarray
        Empty array — no equality constraints are imposed.

    Notes
    -----
    **Sign Convention (CRITICAL):**

    * MATLAB ``fmincon`` requires ``c(x) <= 0`` for inequality constraints.
    * ``scipy.optimize.minimize`` with ``method='SLSQP'`` and constraint
      type ``'ineq'`` requires ``fun(x) >= 0``.
    * Therefore all constraint values produced by the original MATLAB logic
      are **negated** before return so that the stationarity condition
      ``sum(A^2) + sum(B^2) <= 0.99998`` becomes
      ``0.99998 - sum(A^2) - sum(B^2) >= 0``.

    References
    ----------
    .. [1] Noureldin, Shephard, Sheppard (2012). "Multivariate Rotated ARCH
       models." *Journal of Econometrics*.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.rcc_constraint import rcc_constraint
    >>> params = np.array([0.2, 0.8])   # 1 A-param, 1 B-param (Scalar, m=1, n=1)
    >>> c, ceq = rcc_constraint(params, m=1, n=1, k=2, type_model=1)
    >>> c.shape
    (1,)
    >>> ceq.shape
    (0,)
    """
    # Ensure parameters is a contiguous 1-D float64 array
    # Ref: rcc_constraint.m:1 — parameters is a column vector in MATLAB
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    # Early validation of type_model to provide a clear error message
    # rather than letting downstream rarch_parameter_transform fail cryptically
    if type_model not in (1, 2, 3):
        raise ValueError(
            f"Unknown model type: {type_model}. "
            "Must be 1 (Scalar), 2 (CP), or 3 (Diagonal)."
        )

    # ------------------------------------------------------------------
    # Equality constraints — always empty for RCC stationarity
    # Ref: rcc_constraint.m:22 — ceq = []
    # ------------------------------------------------------------------
    ceq: np.ndarray = np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # Stage-based parameter slicing
    # Ref: rcc_constraint.m:24-28
    # When stage==3, the full vector contains only dynamics parameters.
    # Otherwise, the first k*(k-1)/2 elements are correlation rotation
    # parameters that must be stripped before computing stationarity
    # constraints on the RARCH dynamics.
    # ------------------------------------------------------------------
    if stage == 3:
        temp_parameters = parameters.copy()
    else:
        # MATLAB 1-based: parameters(k*(k-1)/2 : length(parameters))
        # Python 0-based: parameters[k*(k-1)//2 - 1 :]
        # Ref: rcc_constraint.m:27 — tempParameters = parameters(k*(k-1)/2:length(parameters))
        start_idx: int = k * (k - 1) // 2 - 1
        temp_parameters = parameters[start_idx:]

    # ------------------------------------------------------------------
    # Unpack parameter vector into coefficient matrices via
    # rarch_parameter_transform.
    # Ref: rcc_constraint.m:29
    #   [~,A,B] = rarch_parameter_transform(tempParameters,m,n,k,[],type,false,false)
    # Discard C (first return); pass None for unused unconditional
    # covariance; is_joint=False, is_c_chol=False.
    # ------------------------------------------------------------------
    _, A, B = rarch_parameter_transform(
        temp_parameters, m, n, k, None, type_model, False, False
    )

    # ------------------------------------------------------------------
    # Stationarity constraint computation
    # Ref: rcc_constraint.m:30
    #   constraint = diag(sum(A.^2,3) + sum(B.^2,3) - .99998)
    #
    # A is always a 3-D array (k, k, m).
    # B is a 3-D array (k, k, n) for Scalar/Diagonal types, but
    # rarch_parameter_transform returns B as a 2-D array (k, k) for the
    # CP model (type_model=2) when n > 0 due to its theta reparameterisation.
    # In MATLAB, sum(X.^2, 3) on a 2-D matrix returns X.^2 (identity op).
    # We replicate this by checking B.ndim before summing.
    # ------------------------------------------------------------------
    sum_A2: np.ndarray = np.sum(A ** 2, axis=2)

    # Ref: rcc_constraint.m:30 — sum(B.^2, 3)
    # Handle B dimensionality: 3-D → sum over lags; 2-D → element-wise square
    if B.ndim == 3:
        sum_B2: np.ndarray = np.sum(B ** 2, axis=2)
    else:
        # CP model (type_model=2 with n>0): B is already a 2-D (k, k) matrix
        # MATLAB sum(B.^2, 3) on 2-D matrix is a no-op → equivalent to B.^2
        sum_B2 = B ** 2

    constraint: np.ndarray = np.diag(sum_A2 + sum_B2 - 0.99998)

    # ------------------------------------------------------------------
    # Type-specific constraint selection with scipy sign negation
    # Ref: rcc_constraint.m:31-39
    #
    # CRITICAL scipy sign convention:
    #   MATLAB fmincon: c(x) <= 0 (feasible when non-positive)
    #   scipy 'ineq':  fun(x) >= 0 (feasible when non-negative)
    # All values are therefore NEGATED relative to the MATLAB source.
    # ------------------------------------------------------------------
    if type_model == 1:
        # Scalar model — single constraint on common persistence
        # Ref: rcc_constraint.m:32-33 — c = constraint(1)
        # MATLAB constraint(1) is 1-based → Python constraint[0]
        # Use slice [0:1] to maintain 1-D array output shape
        c: np.ndarray = -constraint[0:1]

    elif type_model == 2:
        # Common Persistence (CP) model — per-asset constraint relative to
        # the shared theta parameter (last element of full parameters vector)
        # Ref: rcc_constraint.m:34-35
        #   theta = parameters(length(parameters)).^2
        #   c = diag(sum(A.^2,3)) - theta
        # Note: uses original 'parameters', not 'temp_parameters'
        theta: float = float(parameters[-1] ** 2)
        c = -(np.diag(np.sum(A ** 2, axis=2)) - theta)

    else:
        # Diagonal model (type_model == 3) — full vector of per-asset constraints
        # Ref: rcc_constraint.m:36-38 — c = constraint
        c = -constraint

    return c, ceq
