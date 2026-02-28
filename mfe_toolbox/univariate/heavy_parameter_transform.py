"""
HEAVY model parameter vector transformation.

Extracts the intercept vector O, innovation coefficient array A, and smoothing
coefficient array B from a flat parameter vector used during optimisation of the
HEAVY (High-frEquency-bAsed VolatilitY) model of Shephard and Sheppard.

The dynamics of the HEAVY model are:

    h(t,:)' = O + A(:,:,1)*f(data(t-1,:))' + ... + A(:,:,maxP)*f(data(t-maxP,:))'
                + B(:,:,1)*h(t-1,:)' + ... + B(:,:,maxQ)*h(t-maxQ,:)'

Parameters are ordered in the flat vector as:

    [O' A(1,1,1:p[0,0]) A(1,2,1:p[0,1]) ... A(1,K,1:p[0,K-1])
         A(2,1,1:p[1,0]) ... A(K,K,1:p[K-1,K-1])
         B(1,1,1:q[0,0]) ... B(K,K,1:q[K-1,K-1])]

Migrated from univariate/heavy_parameter_transform.m (Version 4.0, 28-Oct-2009).
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk).
"""

import numpy as np


def heavy_parameter_transform(
    parameters: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a flat parameter vector into HEAVY model coefficient matrices.

    Unpacks the concatenated parameter vector produced by the HEAVY model
    optimiser into the intercept vector O, the innovation coefficient array A
    and the smoothing coefficient array B.

    Parameters
    ----------
    parameters : np.ndarray
        A 1-D array with ``K + sum(sum(p)) + sum(sum(q))`` elements containing
        all model parameters in the canonical ordering described above.
    p : np.ndarray
        A K×K integer matrix where element ``(i, j)`` gives the number of lags
        of series *j* innovations in the equation for series *i*.
    q : np.ndarray
        A K×K integer matrix where element ``(i, j)`` gives the number of lags
        of series *j* conditional variance in the equation for series *i*.
    K : int
        The number of series (dimension of the system).

    Returns
    -------
    O : np.ndarray
        A (K,) intercept vector.
    A : np.ndarray
        A (K, K, max_p) array of innovation parameters, where
        ``max_p = max(max(p))``.  If ``max_p == 0`` the shape is ``(K, K, 0)``.
    B : np.ndarray
        A (K, K, max_q) array of smoothing parameters, where
        ``max_q = max(max(q))``.  If ``max_q == 0`` the shape is ``(K, K, 0)``.

    Raises
    ------
    ValueError
        If *parameters* does not have the expected number of elements, or if
        *p*, *q* are not conformable K×K arrays, or if *K* is not a positive
        integer.

    Notes
    -----
    This function performs pure parameter extraction with no numerical
    computation beyond array slicing.  Numerical parity with the MATLAB
    reference implementation is exact (zero floating-point difference).

    See Also
    --------
    mfe_toolbox.univariate.heavy : HEAVY model driver.
    mfe_toolbox.univariate.heavy_likelihood : HEAVY log-likelihood.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.heavy_parameter_transform import (
    ...     heavy_parameter_transform,
    ... )
    >>> params = np.array([0.01, 0.02, 0.3, 0.4, 0.5, 0.6])
    >>> p_mat = np.array([[1, 0], [0, 1]])
    >>> q_mat = np.array([[1, 0], [0, 1]])
    >>> O, A, B = heavy_parameter_transform(params, p_mat, q_mat, 2)
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError(
            f"K must be a positive integer, got {K!r}."
        )

    # Ensure p and q are integer numpy arrays of shape (K, K)
    p = np.asarray(p, dtype=np.intp)
    q = np.asarray(q, dtype=np.intp)

    if p.shape != (K, K):
        raise ValueError(
            f"p must be a ({K}, {K}) array, got shape {p.shape}."
        )
    if q.shape != (K, K):
        raise ValueError(
            f"q must be a ({K}, {K}) array, got shape {q.shape}."
        )

    # Verify parameter vector length
    expected_length = K + int(np.sum(p)) + int(np.sum(q))
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    if parameters.size != expected_length:
        raise ValueError(
            f"parameters must have {expected_length} elements "
            f"(K + sum(sum(p)) + sum(sum(q))), got {parameters.size}."
        )

    # ------------------------------------------------------------------
    # Phase 1: Extract intercepts
    # Ref: heavy_parameter_transform.m:39-41 — Extract K intercepts
    # ------------------------------------------------------------------
    O = np.zeros(K, dtype=np.float64)
    O[:] = parameters[0:K]
    # Ref: heavy_parameter_transform.m:41 — offset initialised after K intercepts
    offset: int = K

    # ------------------------------------------------------------------
    # Phase 2: Extract A (innovation) matrix
    # Ref: heavy_parameter_transform.m:42 — A = zeros(K,K,max(max(p)))
    # ------------------------------------------------------------------
    max_p: int = int(np.max(p)) if p.size > 0 else 0
    # Handle edge case where all lags are zero
    if max_p > 0:
        A = np.zeros((K, K, max_p), dtype=np.float64)
    else:
        A = np.zeros((K, K, 0), dtype=np.float64)

    # Ref: heavy_parameter_transform.m:44-49 — 0-based indexing
    # MATLAB loops: for i=1:K, for j=1:K → Python: for i in range(K), for j in range(K)
    for i in range(K):
        for j in range(K):
            p_ij: int = int(p[i, j])
            if p_ij > 0:
                # Ref: heavy_parameter_transform.m:46 — A(i,j,1:p(i,j)) = parameters(offset+(1:p(i,j)))
                # MATLAB 1-based offset+(1:p(i,j)) → Python offset:offset+p_ij (0-based)
                A[i, j, 0:p_ij] = parameters[offset:offset + p_ij]
            offset += p_ij

    # ------------------------------------------------------------------
    # Phase 3: Extract B (smoothing) matrix
    # Ref: heavy_parameter_transform.m:43 — B = zeros(K,K,max(max(q)))
    # ------------------------------------------------------------------
    max_q: int = int(np.max(q)) if q.size > 0 else 0
    if max_q > 0:
        B = np.zeros((K, K, max_q), dtype=np.float64)
    else:
        B = np.zeros((K, K, 0), dtype=np.float64)

    # Ref: heavy_parameter_transform.m:50-55 — 0-based indexing
    for i in range(K):
        for j in range(K):
            q_ij: int = int(q[i, j])
            if q_ij > 0:
                # Ref: heavy_parameter_transform.m:52 — B(i,j,1:q(i,j)) = parameters(offset+(1:q(i,j)))
                # MATLAB 1-based offset+(1:q(i,j)) → Python offset:offset+q_ij (0-based)
                B[i, j, 0:q_ij] = parameters[offset:offset + q_ij]
            offset += q_ij

    return O, A, B
