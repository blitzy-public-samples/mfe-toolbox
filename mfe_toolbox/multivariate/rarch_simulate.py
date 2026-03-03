"""
Simulation of RARCH(p,q) multivariate volatility model.

Generates simulated data and conditional covariance matrices from a Rotated
ARCH (RARCH) model as described by Noureldin, Shephard, and Sheppard.  The
dynamics evolve in the rotated (unconditionally standardised) space and are
then back-rotated via the unconditional covariance matrix square root.

The RARCH recursion in rotated space is:

    G(:,:,t) = (I_K - sum(A^2) - sum(B^2))
             + A(:,:,1) * OP(:,:,t-1) * A(:,:,1) + ...
             + B(:,:,1) * G(:,:,t-1) * B(:,:,1) + ...

where OP(:,:,t) = e_t * e_t' (outer product of standardised innovations).

Three model types are supported:

* **Scalar** — A(i) = a(i)*I_K, B(j) = b(j)*I_K
* **Common Persistence (CP)** — diagonal A with shared persistence theta,
  B derived from theta^2 - sum(A^2)
* **Diagonal** — fully diagonal A and B matrices

Source Reference
----------------
Migrated from ``multivariate/rarch_simulate.m`` (138 lines).
Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 1, Date: 3/27/2012

See Also
--------
mfe_toolbox.multivariate.rarch : RARCH model driver
mfe_toolbox.multivariate.rarch_likelihood : RARCH log-likelihood
mfe_toolbox.multivariate.rarch_parameter_transform : RARCH parameter unpacking
mfe_toolbox.multivariate.rarch_constraint : RARCH nonlinear constraints
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import sqrtm

from mfe_toolbox.multivariate.rarch_parameter_transform import rarch_parameter_transform


def rarch_simulate(
    t: int | np.ndarray,
    C: np.ndarray,
    parameters: np.ndarray,
    p: int = 1,
    q: int = 1,
    type_model: str = "Scalar",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a RARCH(p,q) multivariate volatility process.

    The dynamics of a RARCH model are identical to those of a BEKK, except
    that the model evolves in the rotated (unconditionally standardised) space.
    Innovations are drawn in the rotated space, scaled by the conditional
    covariance matrix square root, and then back-rotated via *C*\\ :sup:`1/2`.

    Parameters
    ----------
    t : int or numpy.ndarray
        If a scalar integer, it specifies the number of observations *T* to
        simulate.  Standard-normal innovations of shape ``(2*T, K)`` are
        generated internally (the first *T* observations are a burn-in and
        are discarded).

        If a 2-D array of shape ``(T, K)``, it is used as a matrix of
        pre-simulated random variates.  An additional *T* burn-in rows are
        prepended by randomly resampling rows from the provided array.

    C : numpy.ndarray
        K × K unconditional covariance matrix of the data.

    parameters : numpy.ndarray
        1-D vector of model parameters governing the dynamics.  The layout
        depends on ``type_model``:

        * ``'Scalar'``:
          ``[a(1), ..., a(p), b(1), ..., b(q)]``  — all scalars.
        * ``'CP'`` (Common Persistence):
          ``[diag(A1)', ..., diag(Ap)', theta]`` — theta is the common
          persistence scalar.
        * ``'Diagonal'``:
          ``[diag(A1)', ..., diag(Ap)', diag(B1)', ..., diag(Bq)']``

    p : int, optional
        Positive integer — number of symmetric innovation lags (A matrices).
        Default is 1.

    q : int, optional
        Non-negative integer — number of conditional covariance lags
        (B matrices).  For the ``'CP'`` model, ``0 <= q <= 1``.
        Default is 1.

    type_model : str, optional
        Model type, one of ``'Scalar'`` (default), ``'CP'``
        (Common Persistence), or ``'Diagonal'``.

    Returns
    -------
    data : numpy.ndarray
        T × K matrix of simulated data in the original (non-rotated) space.

    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices in the original
        space, where ``Ht[:, :, i]`` is the covariance at time *i*.

    Raises
    ------
    ValueError
        If ``t`` is a matrix with a column count not equal to *K*;
        if ``type_model`` is not one of the three recognised types;
        if ``q > 1`` for the ``'CP'`` model;
        if ``parameters`` has an unexpected length;
        or if the CP persistence constraint ``theta^2 >= sum(A^2)`` is
        violated.

    Warns
    -----
    UserWarning
        When the supplied parameters lie outside the stationary region, i.e.
        ``max(diag(sum(A^2) + sum(B^2))) >= 1``.

    Examples
    --------
    Scalar with A^2 = 0.05 and B^2 = 0.93:

    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.rarch_simulate import rarch_simulate
    >>> data, Ht = rarch_simulate(
    ...     1000, np.eye(2) + 1, np.sqrt(np.array([0.05, 0.93])),
    ...     1, 1, 'Scalar',
    ... )
    >>> data.shape
    (1000, 2)
    >>> Ht.shape
    (2, 2, 1000)

    Diagonal:

    >>> data, Ht = rarch_simulate(
    ...     1000, np.eye(2) + 1,
    ...     np.sqrt(np.array([0.05, 0.07, 0.93, 0.88])),
    ...     1, 1, 'Diagonal',
    ... )

    Common Persistence (note: uses theta, not B directly):

    >>> data, Ht = rarch_simulate(
    ...     1000, np.eye(2) + 1,
    ...     np.sqrt(np.array([0.05, 0.07, 0.99])),
    ...     1, 1, 'CP',
    ... )
    """
    # ------------------------------------------------------------------
    # Input coercion
    # Ref: rarch_simulate.m:60
    # ------------------------------------------------------------------
    C = np.asarray(C, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    k: int = C.shape[0]

    # ------------------------------------------------------------------
    # Handle t: scalar (number of observations) or T×K random variate matrix
    # Ref: rarch_simulate.m:61-70
    # ------------------------------------------------------------------
    if np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 0):
        # Ref: rarch_simulate.m:62 — e = randn(2*T, k)
        T: int = int(t)
        rng = np.random.default_rng()
        e = rng.standard_normal((2 * T, k))
    else:
        # Ref: rarch_simulate.m:64-70
        e = np.asarray(t, dtype=np.float64)
        if e.shape[1] != k:
            raise ValueError(
                "T must have K columns when providing simulated random numbers."
            )
        T = e.shape[0]
        rng = np.random.default_rng()
        # Ref: rarch_simulate.m:69 — e = [e(ceil(rand(T,1)*T),:); e]
        # MATLAB ceil(rand(T,1)*T) produces random 1-indexed integers in [1, T].
        # Python equivalent: integers in [0, T) with 0-based indexing.
        burn_in_indices = rng.integers(0, T, size=T)
        e = np.vstack([e[burn_in_indices, :], e])

    # ------------------------------------------------------------------
    # Convert type_model string to integer code
    # Ref: rarch_simulate.m:72-80
    # ------------------------------------------------------------------
    if isinstance(type_model, str):
        type_lower = type_model.strip().lower()
        if type_lower == "scalar":
            type_int: int = 1
        elif type_lower == "cp":
            type_int = 2
        elif type_lower == "diagonal":
            type_int = 3
        else:
            raise ValueError("TYPE must be 'Scalar', 'CP' or 'Diagonal'.")
    else:
        # Allow integer type codes to be passed directly (internal use)
        type_int = int(type_model)

    # ------------------------------------------------------------------
    # Validate Q for CP model
    # Ref: rarch_simulate.m:82-84
    # ------------------------------------------------------------------
    if type_int == 2 and q > 1:
        raise ValueError("Q must be either 0 or 1 for the 'CP' model.")

    # ------------------------------------------------------------------
    # Validate parameter vector length
    # Ref: rarch_simulate.m:86-96
    # ------------------------------------------------------------------
    if type_int == 1:
        count = p + q
    elif type_int == 2:
        count = p * k + q
    elif type_int == 3:
        count = (p + q) * k
    else:
        raise ValueError("type_model must be 1 (Scalar), 2 (CP), or 3 (Diagonal).")

    if len(parameters) != count:
        raise ValueError(
            "PARAMETERS does not have the expected number of elements."
        )

    # ------------------------------------------------------------------
    # Unpack flat parameter vector into (C, A, B) matrices
    # Ref: rarch_simulate.m:97
    # is_joint=False → C passes through unchanged
    # is_c_chol=False → irrelevant when is_joint=False
    # ------------------------------------------------------------------
    C, A, B = rarch_parameter_transform(
        parameters, p, q, k, C, type_int, False, False
    )

    # ------------------------------------------------------------------
    # Stationarity check
    # Ref: rarch_simulate.m:98-100
    # max(diag(sum(A.^2,3) + sum(B.^2,3))) >= 1 → warning
    # ------------------------------------------------------------------
    # Ref: rarch_simulate.m:98 — sum(A.^2, 3) sums element-wise squares
    # across the third dimension (lag axis = axis 2 in Python).
    sum_A2 = np.sum(A ** 2, axis=2)  # (k, k)

    # B is 3-D (k, k, q) for Scalar/Diagonal or 2-D (k, k) for CP with q>0
    if B.ndim == 3:
        sum_B2 = np.sum(B ** 2, axis=2)  # (k, k)
    else:
        # Ref: CP model — B is already (k, k)
        sum_B2 = B ** 2  # (k, k)

    persistence_diag = np.diag(sum_A2 + sum_B2)
    if np.max(persistence_diag) >= 1.0:
        warnings.warn(
            "The parameters do not correspond to the stationary region.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # CP model: verify B entries are non-negative
    # Ref: rarch_simulate.m:101-103
    # ------------------------------------------------------------------
    if type_int == 2 and q == 1 and np.min(B) < 0:
        raise ValueError(
            "When using 'CP', the common persistence parameter Theta must "
            "satisfy Theta^2>=sum(A.^2,3)"
        )

    # ------------------------------------------------------------------
    # Simulation in rotated space
    # Ref: rarch_simulate.m:107-128
    # ------------------------------------------------------------------
    total_T: int = 2 * T

    # Ref: rarch_simulate.m:107 — Gt = repmat(eye(k), [1 1 2*T])
    Gt = np.zeros((k, k, total_T))
    for idx in range(total_T):
        Gt[:, :, idx] = np.eye(k)

    # Ref: rarch_simulate.m:108 — intercept = eye(k) - sum(A.^2,3) - sum(B.^2,3)
    intercept = np.eye(k) - sum_A2 - sum_B2

    # Ref: rarch_simulate.m:109 — backCast = eye(k)
    backCast = np.eye(k)

    # Main recursion loop
    # Ref: rarch_simulate.m:110-128
    for i in range(total_T):
        # Ref: rarch_simulate.m:111
        Gt[:, :, i] = intercept.copy()

        # Innovation terms (A lags)
        # Ref: rarch_simulate.m:112-118
        for j in range(p):
            # MATLAB: j=1:p, lag index = i-j (1-indexed)
            # Python: j=0:p-1, lag index = i-(j+1) (0-indexed)
            lag = i - (j + 1)
            if lag < 0:
                # Ref: rarch_simulate.m:114 — use backCast for pre-sample lags
                Gt[:, :, i] += A[:, :, j] @ backCast @ A[:, :, j]
            else:
                # Ref: rarch_simulate.m:116 — e(i-j,:)'*e(i-j,:) is outer product
                op = np.outer(e[lag, :], e[lag, :])
                Gt[:, :, i] += A[:, :, j] @ op @ A[:, :, j]

        # Smoothing terms (B lags)
        # Ref: rarch_simulate.m:119-125
        for j in range(q):
            # Select the j-th B matrix
            # Ref: B is (k,k,q) for Scalar/Diagonal or (k,k) for CP
            if B.ndim == 3:
                B_j = B[:, :, j]
            else:
                # CP model: B is a single (k, k) matrix
                B_j = B

            lag = i - (j + 1)
            if lag < 0:
                # Ref: rarch_simulate.m:121 — use backCast for pre-sample lags
                Gt[:, :, i] += B_j @ backCast @ B_j
            else:
                # Ref: rarch_simulate.m:123
                Gt[:, :, i] += B_j @ Gt[:, :, lag] @ B_j

        # Ref: rarch_simulate.m:126 — Gt12 = Gt(:,:,i)^(0.5)
        # MATLAB mpower uses Schur-based principal matrix square root.
        # scipy.linalg.sqrtm provides the same computation; np.real strips
        # any residual imaginary components from near-PSD matrices.
        Gt12 = np.real(sqrtm(Gt[:, :, i]))

        # Ref: rarch_simulate.m:127 — e(i,:) = e(i,:)*Gt12
        # MATLAB row-vector right-multiplication: (1×K) @ (K×K) = (1×K)
        e[i, :] = e[i, :] @ Gt12

    # ------------------------------------------------------------------
    # Back-rotation to original space
    # Ref: rarch_simulate.m:130-136
    # ------------------------------------------------------------------
    data = np.zeros((total_T, k))
    Ht = np.zeros((k, k, total_T))

    # Ref: rarch_simulate.m:132 — C12 = C^(0.5)
    # Principal matrix square root of the unconditional covariance
    C12 = np.real(sqrtm(C))

    for i in range(total_T):
        # Ref: rarch_simulate.m:134 — data(i,:) = e(i,:)*C12
        data[i, :] = e[i, :] @ C12
        # Ref: rarch_simulate.m:135 — Ht(:,:,i) = C12*Gt(:,:,i)*C12
        Ht[:, :, i] = C12 @ Gt[:, :, i] @ C12

    # ------------------------------------------------------------------
    # Remove burn-in (first T observations)
    # Ref: rarch_simulate.m:138-139
    # MATLAB: data(T+1:2*T,:) → Python: data[T:2*T, :]
    # ------------------------------------------------------------------
    data = data[T: 2 * T, :]
    Ht = Ht[:, :, T: 2 * T]

    return data, Ht
