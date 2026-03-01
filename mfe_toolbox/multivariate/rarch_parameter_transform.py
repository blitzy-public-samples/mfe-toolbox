"""
Parameter transformation for RARCH(p,q) multivariate volatility model.

Unpacks a flat parameter vector into the structured matrices (C, A, B) used
by the RARCH variance recursion, constraint checking, simulation, and likelihood
evaluation.  Handles three RARCH model types:

    * **Scalar** (type_model=1) — A(i) = a(i)*I_k, B(j) = b(j)*I_k
    * **Common Persistence / CP** (type_model=2) — diagonal A matrices with
      shared persistence parameter theta, B derived from theta - sum(A^2)
    * **Diagonal** (type_model=3) — fully diagonal A and B matrices

When ``is_joint`` is True, the first K(K+1)/2 elements of the parameter
vector encode the unconditional covariance intercept C, either as the
Cholesky factor (``is_c_chol=True``) or as vech(C) (``is_c_chol=False``).

Source Reference
----------------
Migrated from ``multivariate/rarch_parameter_transform.m`` (88 lines).
Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3

See Also
--------
mfe_toolbox.multivariate.rarch : RARCH model driver
mfe_toolbox.multivariate.rarch_likelihood : RARCH log-likelihood
mfe_toolbox.multivariate.rarch_constraint : RARCH nonlinear constraints
mfe_toolbox.multivariate.rarch_simulate : RARCH simulation
"""

from __future__ import annotations

import numpy as np

# Conditional imports used only when is_joint is True
from mfe_toolbox.utility.vec2chol import vec2chol
from mfe_toolbox.utility.ivech import ivech


def rarch_parameter_transform(
    parameters: np.ndarray,
    p: int,
    q: int,
    k: int,
    C: np.ndarray | None,
    type_model: int,
    is_joint: bool,
    is_c_chol: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform a flat RARCH parameter vector into structured coefficient matrices.

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D vector of model parameters.  The layout depends on ``is_joint``
        and ``type_model``:

        * If ``is_joint`` is True the first K(K+1)/2 elements encode the
          unconditional covariance intercept C (either Cholesky or vech form).
        * The next ``p * num_a`` elements encode the innovation parameter
          matrices A(1)…A(p), where ``num_a`` is 1 for Scalar, K for
          CP / Diagonal.
        * The next ``q * num_b`` elements encode the smoothing parameter
          matrices B(1)…B(q), where ``num_b`` is 1 for Scalar / CP, K for
          Diagonal.

    p : int
        Positive integer — number of symmetric innovation lags (A matrices).
    q : int
        Non-negative integer — number of conditional covariance lags
        (B matrices).  For CP models 0 ≤ q ≤ 1.
    k : int
        Number of assets (dimension of covariance matrices).
    C : numpy.ndarray or None
        K × K unconditional covariance matrix.  Ignored when
        ``is_joint`` is True (C is extracted from ``parameters``).
    type_model : int
        Model type indicator:

        * 1 — Scalar
        * 2 — Common Persistence (CP)
        * 3 — Diagonal

    is_joint : bool
        If True, the parameter vector contains both the covariance intercept
        parameters and the dynamics parameters.  If False, only dynamics
        parameters are present and ``C`` is used as-is.
    is_c_chol : bool
        If True **and** ``is_joint`` is True, the intercept parameters are
        interpreted as the Cholesky factor of C (via ``vec2chol``), and
        C = L @ L.T is computed with a PSD enforcement step.  If False
        they are interpreted via ``ivech``.

    Returns
    -------
    C : numpy.ndarray
        K × K unconditional covariance matrix (either passed through or
        reconstructed from the parameter vector).
    A : numpy.ndarray
        K × K × P array of symmetric innovation coefficient matrices.
    B : numpy.ndarray
        K × K × Q array of smoothing coefficient matrices **or**, for the
        CP model (``type_model == 2`` and ``q > 0``), a K × K diagonal
        matrix obtained from the common-persistence reparameterisation.

    Notes
    -----
    When ``type_model == 2`` (CP) and ``q > 0``, B is reparameterised as::

        theta = B[0, 0, 0] ** 2
        B_diag = theta - diag(sum(A**2, axis=2))
        B_diag = max(B_diag, 0)       # clamp negatives
        B = diag(sqrt(B_diag))

    This means B is returned as a 2-D array (K, K) rather than a 3-D array
    (K, K, Q) in this special case.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.rarch_parameter_transform import (
    ...     rarch_parameter_transform,
    ... )
    >>> params = np.array([0.1, 0.9])
    >>> C = np.eye(2)
    >>> C_out, A, B = rarch_parameter_transform(params, 1, 1, 2, C, 1, False, False)
    >>> A.shape
    (2, 2, 1)
    >>> B.shape
    (2, 2, 1)
    """
    # Ensure parameters is a 1-D float64 array for safe slicing
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Phase 1: Determine per-matrix parameter counts based on model type
    # Ref: rarch_parameter_transform.m:31-38
    # ------------------------------------------------------------------
    # Scalar default: one scalar parameter shared across all k diagonal entries
    num_a: int = 1
    num_b: int = 1
    # Common Persistence (type>=2) or Diagonal (type==3): k diagonal params for A
    if type_model >= 2:
        num_a = k
    # Diagonal (type==3): k diagonal params for B as well
    if type_model == 3:
        num_b = k

    # ------------------------------------------------------------------
    # Phase 2: Extract unconditional covariance C (joint estimation only)
    # Ref: rarch_parameter_transform.m:40-59
    # ------------------------------------------------------------------
    parameter_count: int = 0

    if is_joint:
        # Number of lower-triangle elements of a k×k symmetric matrix
        k2 = k * (k + 1) // 2
        c_params = parameters[:k2]

        if is_c_chol:
            # Ref: rarch_parameter_transform.m:44-54
            # Interpret c_params as the half-vec of a lower-triangular
            # Cholesky factor L, then form C = L @ L.T.
            L = vec2chol(c_params)
            C = L @ L.T

            # --- PSD enforcement ---
            # Ref: rarch_parameter_transform.m:48-53
            eigenvalues, V = np.linalg.eigh(C)
            max_eig = np.max(eigenvalues)
            eps_val = np.finfo(float).eps

            if np.min(eigenvalues) < 2.0 * eps_val * max_eig:
                # Clamp tiny / negative eigenvalues to a small positive floor
                eigenvalues[eigenvalues / max_eig < eps_val] = (
                    2.0 * max_eig * eps_val
                )
                C = V @ np.diag(eigenvalues) @ V.T
                # Ref: rarch_parameter_transform.m:53 — force exact symmetry
                C = (C + C) / 2.0
        else:
            # Ref: rarch_parameter_transform.m:56
            # Interpret c_params as vech(C), reconstruct symmetric matrix
            C = ivech(c_params)

        parameter_count += k2

    # ------------------------------------------------------------------
    # Phase 3: Build A matrices (k × k × p)
    # Ref: rarch_parameter_transform.m:60-69
    # ------------------------------------------------------------------
    A = np.zeros((k, k, p))
    for i in range(p):
        # Ref: rarch_parameter_transform.m:62 — extract num_a params
        # MATLAB 1-indexed: parameters(parameterCount+1:parameterCount+numA)
        # Python 0-indexed: parameters[parameter_count:parameter_count+num_a]
        temp = parameters[parameter_count: parameter_count + num_a]
        parameter_count += num_a

        if num_a == 1:
            # Ref: rarch_parameter_transform.m:64 — scalar * I_k
            A[:, :, i] = temp[0] * np.eye(k)
        else:
            # Ref: rarch_parameter_transform.m:66 — diagonal matrix
            A[:, :, i] = np.diag(temp)

    # ------------------------------------------------------------------
    # Phase 4: Build B matrices (k × k × q)
    # Ref: rarch_parameter_transform.m:70-79
    # ------------------------------------------------------------------
    B = np.zeros((k, k, q))
    for i in range(q):
        # Ref: rarch_parameter_transform.m:72 — extract num_b params
        temp = parameters[parameter_count: parameter_count + num_b]
        parameter_count += num_b

        if num_b == 1:
            # Ref: rarch_parameter_transform.m:74 — scalar * I_k
            B[:, :, i] = temp[0] * np.eye(k)
        else:
            # Ref: rarch_parameter_transform.m:76 — diagonal matrix
            B[:, :, i] = np.diag(temp)

    # ------------------------------------------------------------------
    # Phase 5: Common Persistence (type_model == 2) reparameterisation
    # Ref: rarch_parameter_transform.m:82-87
    # ------------------------------------------------------------------
    if type_model == 2 and q > 0:
        # Ref: rarch_parameter_transform.m:83
        # theta = B(1,1,1)^2  — the single B-parameter squared
        theta = B[0, 0, 0] ** 2

        # Ref: rarch_parameter_transform.m:84
        # B = theta - diag(sum(A.^2, 3))
        # np.sum(A**2, axis=2) sums A^2 across the third dimension (p lags)
        # np.diag(...) extracts the diagonal of the resulting k×k matrix
        B_diag = theta - np.diag(np.sum(A ** 2, axis=2))

        # Ref: rarch_parameter_transform.m:85 — clamp negatives to zero
        B_diag = np.maximum(B_diag, 0.0)

        # Ref: rarch_parameter_transform.m:86 — return sqrt diagonal matrix
        # NOTE: B changes from 3-D (k, k, q) to 2-D (k, k) for CP model
        B = np.diag(np.sqrt(B_diag))

    # Ensure C is a proper numpy array (handles the pass-through case)
    if C is not None:
        C = np.asarray(C, dtype=np.float64)
    else:
        # Fallback: if C was not provided and is_joint was False,
        # return a zero matrix to avoid downstream errors.
        # In practice, callers should always provide C when is_joint=False.
        C = np.zeros((k, k))

    return C, A, B
