"""
Parameter transformation for BEKK(p, o, q) multivariate volatility model.

Unpacks a flat parameter vector into the constituent C (intercept),
A (symmetric innovation), G (asymmetric innovation), and B (smoothing)
matrix arrays used by BEKK estimation and simulation routines.

The transformation supports three parameterisation types:
    * Type 1 — **Scalar**: each dynamic matrix is a scalar multiple of I(K).
    * Type 2 — **Diagonal**: each dynamic matrix is diagonal (K free parameters).
    * Type 3 — **Full**: each dynamic matrix has K² free parameters.

The intercept C is always reconstructed from a K(K+1)/2 half-vec via
Cholesky factorisation (C = L @ L.T) and then projected onto the
positive-semi-definite cone by flooring eigenvalues near zero.

Notes
-----
Migrated from ``multivariate/bekk_parameter_transform.m`` (MATLAB MFE
Toolbox v4.0, Kevin Sheppard).

Translation decisions:
    - ``type`` renamed to ``type_`` to avoid shadowing the Python built-in.
    - MATLAB ``eig(C)`` → ``np.linalg.eigh(C)`` for symmetric matrices
      (returns eigenvalues sorted ascending, so min/max are at index 0 / -1).
    - MATLAB ``reshape(tempP, k, k)`` → ``tempP.reshape((k, k), order='F')``
      because MATLAB fills column-major.
    - MATLAB 1-based ``parameters(offset+(1:numParams))`` → Python 0-based
      ``parameters[offset:offset + num_params]``.
    - PSD symmetry enforcement ``C = (C + C) / 2`` in MATLAB is
      equivalent to ``(C + C.T) / 2`` since C is symmetric at that point;
      the transpose form is used here for clarity and robustness.

See Also
--------
bekk : BEKK model estimation driver.
bekk_simulate : BEKK simulation.
bekk_likelihood : BEKK log-likelihood computation.
"""

import numpy as np

from mfe_toolbox.utility.vec2chol import vec2chol


def bekk_parameter_transform(parameters, p, o, q, k, type_):
    """
    Transform a flat BEKK parameter vector into structured matrices.

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D array of model parameters.  The first K(K+1)/2 elements encode the
        lower-triangular Cholesky factor of the covariance intercept C.  The
        remaining elements encode (p + o + q) dynamic matrices whose size
        depends on *type_* (1 element for scalar, K for diagonal, K² for full).
    p : int
        Number of symmetric innovation lags (positive integer ≥ 1).
    o : int
        Number of asymmetric innovation lags (non-negative integer).
    q : int
        Number of conditional covariance lags (non-negative integer).
    k : int
        Number of assets (positive integer, determines matrix dimension).
    type_ : {1, 2, 3}
        Parameterisation type.

        * ``1`` — Scalar BEKK: each A/G/B matrix is ``a_j * I(K)``.
        * ``2`` — Diagonal BEKK: each A/G/B matrix is ``diag(a_j)``.
        * ``3`` — Full BEKK: each A/G/B matrix is an unrestricted K×K matrix.

    Returns
    -------
    C : numpy.ndarray
        K × K positive-semi-definite covariance intercept matrix.
    A : numpy.ndarray
        K × K × P array of symmetric innovation parameter matrices.
        If ``p == 0`` the third dimension has size 0.
    G : numpy.ndarray
        K × K × O array of asymmetric innovation parameter matrices.
        If ``o == 0`` the third dimension has size 0.
    B : numpy.ndarray
        K × K × Q array of smoothing parameter matrices.
        If ``q == 0`` the third dimension has size 0.

    Raises
    ------
    ValueError
        If *type_* is not in ``{1, 2, 3}``.
    ValueError
        If *parameters* does not contain enough elements for the requested
        parameterisation.

    Examples
    --------
    >>> import numpy as np
    >>> # 2-asset scalar BEKK(1,0,1): k2=3 intercept + 1 A + 1 B = 5 params
    >>> params = np.array([1.0, 0.0, 1.0, 0.5, 0.3])
    >>> C, A, G, B = bekk_parameter_transform(params, 1, 0, 1, 2, 1)
    >>> C.shape
    (2, 2)
    >>> A.shape
    (2, 2, 1)
    >>> G.shape
    (2, 2, 0)
    >>> B.shape
    (2, 2, 1)
    """
    # Ensure parameters is a contiguous 1-D float64 array
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Phase 1: Determine the number of free parameters per dynamic matrix
    # Ref: bekk_parameter_transform.m:28-34
    # ------------------------------------------------------------------
    if type_ == 1:
        # Scalar: one parameter multiplied by I(K)
        num_params = 1
    elif type_ == 2:
        # Diagonal: K parameters placed on the diagonal
        num_params = k
    elif type_ == 3:
        # Full: K² parameters reshaped into a K×K matrix
        num_params = k * k
    else:
        raise ValueError(
            f"type_ must be 1 (scalar), 2 (diagonal), or 3 (full); "
            f"received {type_!r}."
        )

    # ------------------------------------------------------------------
    # Phase 2: Extract and reconstruct the intercept matrix C
    # Ref: bekk_parameter_transform.m:36-47
    # ------------------------------------------------------------------
    # Number of elements in the lower-triangular Cholesky factor
    k2 = k * (k + 1) // 2

    # Total required length
    m = p + o + q
    required_length = k2 + m * num_params
    if len(parameters) < required_length:
        raise ValueError(
            f"parameters must have at least {required_length} elements for "
            f"k={k}, p={p}, o={o}, q={q}, type_={type_}; "
            f"received {len(parameters)}."
        )

    # Ref: bekk_parameter_transform.m:37-39
    # Extract the half-vec, reconstruct L via vec2chol, then C = L @ L.T
    c_vec = parameters[:k2]
    L = vec2chol(c_vec)          # K×K lower-triangular Cholesky factor
    C = L @ L.T                  # K×K covariance intercept

    # Ref: bekk_parameter_transform.m:41-47
    # PSD correction: eigenvalue floor to avoid numerical non-PSD artefacts
    eigenvalues, V = np.linalg.eigh(C)  # eigh for symmetric; sorted ascending
    eps_val = np.finfo(float).eps

    # Ref: bekk_parameter_transform.m:43 — if min(D) < 2*eps*max(D)
    max_eig = np.max(eigenvalues)
    min_eig = np.min(eigenvalues)

    if min_eig < 2.0 * eps_val * max_eig:
        # Ref: bekk_parameter_transform.m:44 — floor near-zero eigenvalues
        eigenvalues[eigenvalues / max_eig < eps_val] = 2.0 * max_eig * eps_val
        # Ref: bekk_parameter_transform.m:45 — reconstruct C
        C = V @ np.diag(eigenvalues) @ V.T
        # Ref: bekk_parameter_transform.m:46 — enforce symmetry
        # MATLAB writes C=(C+C)/2 which is a no-op; the intent is symmetry
        # enforcement, and since C is symmetric by construction the result
        # is identical to (C + C.T) / 2.
        C = (C + C.T) / 2.0

    # ------------------------------------------------------------------
    # Phase 3: Extract A, G, B dynamic matrices
    # Ref: bekk_parameter_transform.m:49-64
    # ------------------------------------------------------------------
    offset = k2
    temp = np.zeros((k, k, m))

    for j in range(m):
        # Ref: bekk_parameter_transform.m:52 — 0-indexed slice
        temp_p = parameters[offset:offset + num_params]
        offset += num_params

        if type_ == 1:
            # Ref: bekk_parameter_transform.m:55 — scalar * I(K)
            temp[:, :, j] = temp_p[0] * np.eye(k)
        elif type_ == 2:
            # Ref: bekk_parameter_transform.m:57 — diagonal matrix
            temp[:, :, j] = np.diag(temp_p)
        else:
            # Ref: bekk_parameter_transform.m:59 — reshape(tempP, k, k)
            # MATLAB reshape fills column-major; replicate with order='F'
            temp[:, :, j] = temp_p.reshape((k, k), order='F')

    # Ref: bekk_parameter_transform.m:62-64 — slice A, G, B from temp
    A = temp[:, :, :p]              # First p slices (symmetric innovation)
    G = temp[:, :, p:p + o]         # Next o slices (asymmetric innovation)
    B = temp[:, :, p + o:p + o + q] # Last q slices (smoothing)

    return C, A, G, B
