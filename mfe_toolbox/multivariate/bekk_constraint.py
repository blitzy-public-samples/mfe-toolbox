"""
Non-linear constraint for BEKK(p, o, q) multivariate volatility model estimation.

Computes the stationarity inequality constraint in scipy convention
(``c >= 0`` = feasible) that ensures the BEKK model's conditional covariance
process is covariance-stationary.  Constraint values are **negated** relative
to the MATLAB source (where ``fmincon`` uses ``c <= 0``).  The constraint form
depends on the parameterisation type:

    * Type 1 -- **Scalar**: sum of squared scalar parameters < 1.
    * Type 2 -- **Diagonal**: diagonal persistence < 1 for each asset.
    * Type 3 -- **Full**: spectral radius of Kronecker persistence matrix < 1.

No equality constraints are imposed (``ceq`` is always an empty array).

Notes
-----
Migrated from ``multivariate/bekk_constraint.m`` (MATLAB MFE Toolbox v4.0,
Kevin Sheppard, Revision 1, 2012-03-27).

Translation decisions
    - MATLAB ``A(1,1,:).^2`` -> ``A[0, 0, :] ** 2`` (0-indexed).
    - MATLAB ``sum(..., 3)`` -> ``np.sum(..., axis=2)`` (MATLAB dim 3 = axis 2).
    - MATLAB ``diag(sum(...,3))`` -> ``np.diag(np.sum(..., axis=2))``.
    - MATLAB ``zeros(k*k)`` creates (k^2 x k^2) -> ``np.zeros((k*k, k*k))``.
    - MATLAB ``kron(A(:,:,i), A(:,:,i))`` -> ``np.kron(A[:,:,i], A[:,:,i])``.
    - MATLAB ``abs(eig(m))`` -> ``np.abs(np.linalg.eigvals(m))``.
    - MATLAB ``type`` parameter renamed to ``type_model`` to avoid shadowing
      the Python built-in ``type``.
    - Constraint convention: MATLAB ``fmincon`` uses ``c <= 0`` for inequality
      constraints, while scipy ``{'type': 'ineq'}`` requires ``f(x) >= 0`` for
      feasibility.  All constraint values are therefore **negated** relative to
      the MATLAB source to match the scipy convention, consistent with
      ``rcc_constraint.py``.

See Also
--------
bekk : BEKK model estimation driver.
bekk_likelihood : BEKK log-likelihood computation.
bekk_parameter_transform : Parameter vector unpacking.
"""

import numpy as np

from mfe_toolbox.multivariate.bekk_parameter_transform import bekk_parameter_transform


def bekk_constraint(parameters, data, data_asym, p, o, q, back_cast,
                    back_cast_asym, type_model):
    """
    Compute non-linear inequality constraints for BEKK(p, o, q) estimation.

    The constraint ensures covariance stationarity of the BEKK process by
    bounding the persistence of the A (symmetric innovation), G (asymmetric
    innovation), and B (smoothing) coefficient matrices.

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D parameter vector for the BEKK model.  Passed through to
        :func:`bekk_parameter_transform` for unpacking into C, A, G, B
        matrices.
    data : numpy.ndarray
        T x K array of (de-meaned) return data.  Only ``data.shape[1]`` (the
        number of assets K) is used to determine matrix dimensions.
    data_asym : numpy.ndarray
        T x K array of asymmetric return data (e.g. negative-return
        indicators).  Not used directly; included for interface consistency
        with :func:`bekk_likelihood`.
    p : int
        Number of symmetric innovation lags (A matrices).  Must be >= 0.
    o : int
        Number of asymmetric innovation lags (G matrices).  Must be >= 0.
    q : int
        Number of conditional covariance lags (B matrices).  Must be >= 0.
    back_cast : numpy.ndarray
        K x K back-cast (initial) covariance matrix.  Not used directly;
        included for interface consistency with :func:`bekk_likelihood`.
    back_cast_asym : numpy.ndarray
        K x K asymmetric back-cast matrix.  Not used directly; included for
        interface consistency.
    type_model : {1, 2, 3}
        BEKK parameterisation type:

        * ``1`` -- Scalar BEKK.
        * ``2`` -- Diagonal BEKK.
        * ``3`` -- Full BEKK.

    Returns
    -------
    c : numpy.ndarray
        Inequality constraint vector (scipy convention: ``c >= 0`` = feasible).
        All values are **negated** relative to the MATLAB source.
        Shape depends on *type_model*:

        * Scalar  (1): shape ``(1,)``  -- single stationarity bound.
        * Diagonal (2): shape ``(K,)`` -- per-asset stationarity bound.
        * Full    (3): shape ``(K**2,)`` -- eigenvalue stationarity bounds.
    ceq : numpy.ndarray
        Empty 1-D float64 array (no equality constraints).

    Raises
    ------
    ValueError
        If *type_model* is not in ``{1, 2, 3}``.

    Notes
    -----
    The constraints encode the spectral-radius condition for covariance
    stationarity of the BEKK process (in scipy convention, ``c >= 0`` =
    feasible):

    - **Scalar**:
      ``1 - sum_j(a_j^2) - sum_j(b_j^2) - 0.5 * sum_j(g_j^2) >= 0``
    - **Diagonal**:
      ``1 - diag(sum_j A_j^2 + sum_j B_j^2 + 0.5 * sum_j G_j^2) >= 0``
      element-wise for each of the K assets.
    - **Full**:
      ``0.99998 - |lambda_i(M)| >= 0`` for every eigenvalue lambda_i of the
      K^2 x K^2 persistence matrix
      ``M = sum_j kron(A_j, A_j) + 0.5 sum_j kron(G_j, G_j) + sum_j kron(B_j, B_j)``.

    References
    ----------
    .. [1] Engle, R. F. and Kroner, K. F. (1995).  Multivariate Simultaneous
       Generalized ARCH.  *Econometric Theory*, 11(1), 122--150.

    Examples
    --------
    >>> import numpy as np
    >>> # 2-asset scalar BEKK(1,0,1): 3 intercept + 1 A + 1 B = 5 params
    >>> params = np.array([1.0, 0.0, 1.0, 0.3, 0.4])
    >>> data = np.random.default_rng(42).standard_normal((100, 2))
    >>> c, ceq = bekk_constraint(params, data, data, 1, 0, 1,
    ...                          np.eye(2), np.eye(2), 1)
    >>> c.shape
    (1,)
    >>> len(ceq) == 0
    True
    """
    # ------------------------------------------------------------------
    # Ref: bekk_constraint.m:24 -- no equality constraints
    # ------------------------------------------------------------------
    ceq = np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # Ref: bekk_constraint.m:25 -- number of assets from the data matrix
    # ------------------------------------------------------------------
    k = data.shape[1]

    # ------------------------------------------------------------------
    # Ref: bekk_constraint.m:26 -- unpack parameter vector into matrices
    # MATLAB: [~, A, G, B] = bekk_parameter_transform(parameters, p, o, q, k, type)
    # ------------------------------------------------------------------
    _, A, G, B = bekk_parameter_transform(parameters, p, o, q, k, type_model)

    # ------------------------------------------------------------------
    # Ref: bekk_constraint.m:28-45 -- compute constraint by model type
    # ------------------------------------------------------------------
    if type_model == 1:
        # Ref: bekk_constraint.m:29-30 -- Scalar BEKK stationarity constraint
        # MATLAB: c = sum(A(1,1,:).^2,3) + sum(B(1,1,:).^2,3)
        #             + 0.5*sum(G(1,1,:).^2,3) - 1;
        #
        # For scalar type each A[:,:,i] = a_i * I(K), so A[0,0,:] extracts
        # the scalar multipliers for all p lags.  Same logic for B and G.
        # np.sum of an empty (shape (0,)) array correctly returns 0.0 when
        # o == 0 (no asymmetric lags), so no special-casing is needed.
        c_scalar = (np.sum(A[0, 0, :] ** 2)
                    + np.sum(B[0, 0, :] ** 2)
                    + 0.5 * np.sum(G[0, 0, :] ** 2)
                    - 1.0)
        # Negate for scipy convention: MATLAB c <= 0 → scipy c >= 0
        c = -np.atleast_1d(np.asarray(c_scalar, dtype=np.float64))

    elif type_model == 2:
        # Ref: bekk_constraint.m:31-32 -- Diagonal BEKK stationarity constraint
        # MATLAB: c = diag(sum(A.^2,3) + sum(B.^2,3) + 0.5*sum(G.^2,3) - 1);
        #
        # A is K x K x P.  Element-wise squaring followed by sum over axis 2
        # (MATLAB dim 3) yields a K x K matrix.  For diagonal-type models the
        # off-diagonal entries of every slice are zero, so the meaningful
        # constraint information is on the main diagonal.
        persistence_matrix = (np.sum(A ** 2, axis=2)
                              + np.sum(B ** 2, axis=2)
                              + 0.5 * np.sum(G ** 2, axis=2)
                              - 1.0)
        # Negate for scipy convention: MATLAB c <= 0 → scipy c >= 0
        c = -np.diag(persistence_matrix)

    elif type_model == 3:
        # Ref: bekk_constraint.m:33-44 -- Full BEKK stationarity constraint
        # Build the K^2 x K^2 Kronecker persistence matrix M and check that
        # all eigenvalue magnitudes are below the 0.99998 threshold.

        # Ref: bekk_constraint.m:34 -- MATLAB zeros(k*k) creates (k^2 x k^2)
        k_sq = k * k
        m = np.zeros((k_sq, k_sq))

        # Ref: bekk_constraint.m:35-37 -- accumulate kron(A_i, A_i) for i = 1..p
        for i in range(p):
            m = m + np.kron(A[:, :, i], A[:, :, i])

        # Ref: bekk_constraint.m:38-40 -- accumulate 0.5 * kron(G_i, G_i) for i = 1..o
        for i in range(o):
            m = m + 0.5 * np.kron(G[:, :, i], G[:, :, i])

        # Ref: bekk_constraint.m:41-43 -- accumulate kron(B_i, B_i) for i = 1..q
        for i in range(q):
            m = m + np.kron(B[:, :, i], B[:, :, i])

        # Ref: bekk_constraint.m:44 -- eigenvalue magnitude constraint
        # MATLAB: c = abs(eig(m)) - .99998
        # Negate for scipy convention: MATLAB c <= 0 → scipy c >= 0
        c = -(np.abs(np.linalg.eigvals(m)) - 0.99998)

    else:
        raise ValueError(
            f"type_model must be 1 (scalar), 2 (diagonal), or 3 (full); "
            f"received {type_model!r}."
        )

    return c, ceq
