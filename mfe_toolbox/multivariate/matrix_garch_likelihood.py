"""
Matrix GARCH log-likelihood computation.

Computes the negative log-likelihood for Matrix GARCH(P,O,Q) models,
supporting both standard multivariate normal and composite likelihood
evaluation.

The Matrix GARCH model specifies conditional covariance dynamics as:

    H(t) = CC' + sum_{i=1}^{P} A(i)A(i)' .* data(t-i)
                + sum_{j=1}^{O} G(j)G(j)' .* data_asym(t-j)
                + sum_{l=1}^{Q} B(l)B(l)' .* H(t-l)

where ``.*`` denotes the Hadamard (element-wise) product, and C, A(i), G(j),
B(l) are K x K lower triangular parameter matrices packed via ``vec2chol``.
The formation ``LL'`` guarantees that each parameter matrix is positive
semi-definite.

Notes
-----
Migrated from ``multivariate/matrix_garch_likelihood.m`` (MATLAB MFE Toolbox
v4.0, Revision 4, 10/28/2009).

Author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk

Key migration changes from MATLAB to Python:

- ``vec2chol`` reconstructs K x K lower-triangular Cholesky factors from
  the vech-packed parameter vector (replaces MATLAB column-major triangle
  fill).
- ``numpy.linalg.slogdet`` replaces ``log(det())`` for numerically stable
  log-determinant computation.
- ``numpy.linalg.solve`` replaces explicit matrix inversion ``R^(-1)`` in the
  trace computation for better conditioning.
- An optional ``composite`` argument adds composite likelihood support
  (diagonal or full pairwise) following the pattern established in
  ``scalar_vt_vech_likelihood.m``.
- MATLAB 1-based loop indexing is converted to Python 0-based indexing
  throughout the recursion.

References
----------
Kevin Sheppard, MFE Toolbox Version 4.0 (2009).
    See also: MATRIX_GARCH, SCALAR_VT_VECH_LIKELIHOOD
"""

import numpy as np

from mfe_toolbox.utility.vec2chol import vec2chol
from mfe_toolbox.distributions.composite_likelihood import composite_likelihood


def matrix_garch_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    data_asym: np.ndarray,
    p: int,
    o: int,
    q: int,
    back_cast: np.ndarray,
    back_cast_asym: np.ndarray,
    intercept: np.ndarray = None,
    composite: int = 0,
):
    """
    Log-likelihood for Matrix GARCH(P,O,Q) estimation.

    Computes the negative log-likelihood for a Matrix GARCH model with
    conditional covariance dynamics::

        H(t) = CC' + A(1)A(1)' .* r_{t-1}r_{t-1}' + ... + A(P)A(P)' .* r_{t-P}r_{t-P}'
                   + G(1)G(1)' .* n_{t-1}n_{t-1}' + ... + G(O)G(O)' .* n_{t-O}n_{t-O}'
                   + B(1)B(1)' .* H(t-1) + ... + B(Q)B(Q)' .* H(t-Q)

    where each parameter matrix is reconstructed from a vech-packed lower
    triangular Cholesky factor using ``vec2chol``, and ``.*`` denotes the
    Hadamard (element-wise) product.

    Parameters
    ----------
    parameters : numpy.ndarray
        Vector of vech-packed GARCH process parameters.  When ``intercept``
        is ``None``, the layout is::

            [vech(C)' vech(A(1))' ... vech(A(P))' vech(G(1))' ...
             vech(G(O))' vech(B(1))' ... vech(B(Q))']'

        with total length ``K*(K+1)/2 * (1 + P + O + Q)``.

        When ``intercept`` is provided, the intercept block ``vech(C)`` is
        omitted and the vector has length ``K*(K+1)/2 * (P + O + Q)``.
    data : numpy.ndarray, shape (K, K, T)
        K x K x T array of covariance innovations (e.g., outer products of
        residuals ``r_t @ r_t.T`` or realized covariance matrices).
    data_asym : numpy.ndarray, shape (K, K, T) or None
        K x K x T array of asymmetric covariance innovations.  Required
        when ``o > 0``; may be ``None`` when ``o == 0``.
    p : int
        Positive scalar integer representing the number of lags of the
        innovation process (ARCH order).
    o : int
        Non-negative scalar integer representing the number of asymmetric
        innovation lags.
    q : int
        Non-negative scalar integer representing the number of conditional
        covariance lags (GARCH order).
    back_cast : numpy.ndarray, shape (K, K)
        Back-cast value for initializing the recursion in pre-sample periods
        for innovation and GARCH terms.
    back_cast_asym : numpy.ndarray, shape (K, K)
        Back-cast value for initializing the recursion in pre-sample periods
        for asymmetric terms.
    intercept : numpy.ndarray or None, optional
        If provided, a K x K pre-computed intercept matrix ``CC'``.  When
        ``None`` (default), the intercept Cholesky factor is unpacked from
        the first ``K*(K+1)/2`` elements of ``parameters``.
    composite : int, optional
        Composite likelihood mode:

        - ``0`` — Standard multivariate normal log-likelihood (default).
          Uses a correlation decomposition for improved numerical stability.
        - ``1`` — Diagonal composite likelihood using adjacent pairs.
        - ``2`` — Full composite likelihood using all lower-triangular pairs.

    Returns
    -------
    ll : float
        Negative log-likelihood value (scalar, for minimization).
    lls : numpy.ndarray, shape (T,)
        Time series of per-observation negative log-likelihoods.
    Ht : numpy.ndarray, shape (K, K, T)
        Time series of K x K conditional covariance matrices.

    Raises
    ------
    ValueError
        If ``o > 0`` and ``data_asym`` is ``None``.
    ValueError
        If ``composite`` is not 0, 1, or 2.

    Notes
    -----
    The standard log-likelihood at each observation uses a correlation
    decomposition (Ref: matrix_garch_likelihood.m:77-80) for numerical
    stability:

    .. math::

        -\\log L_t = \\tfrac{1}{2}\\bigl[
            k \\log(2\\pi) + 2\\sum_i \\log Q_i
            + \\log|R_t| + \\mathrm{tr}(R_t^{-1} S_t)
        \\bigr]

    where :math:`Q_i = \\sqrt{H_{t,ii}}` are conditional standard deviations,
    :math:`R_t = H_t \\oslash (QQ^\\top)` is the conditional correlation
    matrix, and :math:`S_t = \\text{data}_t \\oslash (QQ^\\top)` is the
    standardized data matrix.

    When a ``composite`` mode is selected, the per-observation likelihood is
    computed via :func:`~mfe_toolbox.distributions.composite_likelihood.composite_likelihood`
    using bivariate pairwise evaluation, following the pattern from
    ``scalar_vt_vech_likelihood.m`` lines 74-116.

    See Also
    --------
    mfe_toolbox.multivariate.matrix_garch : Matrix GARCH model driver.
    mfe_toolbox.utility.vec2chol : Vector to lower-triangular Cholesky
        reconstruction.
    mfe_toolbox.distributions.composite_likelihood : Composite normal
        log-likelihood.
    """
    # ------------------------------------------------------------------
    # Extract dimensions
    # Ref: matrix_garch_likelihood.m:30 — [k,nothing,T]=size(data)
    # ------------------------------------------------------------------
    k = data.shape[0]
    T = data.shape[2]
    # Ref: matrix_garch_likelihood.m:31 — k2 = k*(k+1)/2
    k2 = k * (k + 1) // 2

    # ------------------------------------------------------------------
    # Input coercion and validation
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    if o > 0 and data_asym is None:
        raise ValueError(
            "data_asym must be provided (non-None) when o > 0."
        )

    if composite not in (0, 1, 2):
        raise ValueError(
            f"composite must be 0, 1, or 2, got {composite}."
        )

    # ------------------------------------------------------------------
    # Unpack parameter matrices via vec2chol
    # Ref: matrix_garch_likelihood.m:33-39
    #
    # Each vech block of length k2 is reconstructed into a K x K lower
    # triangular factor L via vec2chol, then the positive semi-definite
    # parameter matrix is formed as (LL' + LL')/2 = LL' (symmetrized
    # against floating-point round-off).
    #
    # Layout in parameter_matrices (0-based):
    #   [:,:,0]         = intercept CC'
    #   [:,:,1..p]      = A(1)A(1)' ... A(P)A(P)'
    #   [:,:,p+1..p+o]  = G(1)G(1)' ... G(O)G(O)'
    #   [:,:,p+o+1..p+o+q] = B(1)B(1)' ... B(Q)B(Q)'
    # ------------------------------------------------------------------
    total_matrices = 1 + p + o + q
    parameter_matrices = np.zeros((k, k, total_matrices))
    index = 0

    if intercept is not None:
        # Intercept matrix provided externally — skip unpacking C from params.
        # This extension supports joint/two-stage estimation workflows where
        # the intercept is pre-computed (e.g., from scalar_vt_vech diagnostics).
        parameter_matrices[:, :, 0] = np.asarray(intercept, dtype=np.float64)
        start_matrix = 1
    else:
        # Intercept included in the parameter vector (original MATLAB behavior).
        start_matrix = 0

    for i in range(start_matrix, total_matrices):
        # Ref: matrix_garch_likelihood.m:36 — temp = vec2chol(parameters(...))
        temp = vec2chol(parameters[index:index + k2])
        # Ref: matrix_garch_likelihood.m:37 — parameterMatrices(:,:,i) = (temp*temp'+temp*temp')/2
        # LL' is inherently symmetric; averaging enforces exact symmetry
        # against floating-point round-off.
        mat = temp @ temp.T
        parameter_matrices[:, :, i] = (mat + mat.T) / 2.0
        index += k2

    # ------------------------------------------------------------------
    # Initialize conditional covariance array
    # Ref: matrix_garch_likelihood.m:41 — Ht=repmat(backCast,[1 1 T])
    # ------------------------------------------------------------------
    Ht = np.tile(back_cast[:, :, np.newaxis], (1, 1, T))

    # ------------------------------------------------------------------
    # Initialize per-observation log-likelihood series
    # Ref: matrix_garch_likelihood.m:44 — lls=zeros(T,1)
    # ------------------------------------------------------------------
    lls = np.zeros(T)

    # ------------------------------------------------------------------
    # Set up likelihood computation mode
    # Ref: matrix_garch_likelihood.m:47 — likconst=k*log(2*pi)
    #
    # For composite modes, pre-compute 0-based index pairs following the
    # pattern from scalar_vt_vech_likelihood.m:74-81. Index pairs are
    # 0-based because the Python composite_likelihood function expects
    # 0-based indices (unlike the MATLAB version which uses 1-based).
    # ------------------------------------------------------------------
    indices = None  # Only used in composite modes
    likconst = 0.0  # Only used in standard mode

    if composite == 0:
        # Standard multivariate normal likelihood constant
        # Ref: matrix_garch_likelihood.m:47
        likconst = k * np.log(2.0 * np.pi)
    elif composite == 1:
        # Diagonal composite: adjacent pairs (0,1), (1,2), ..., (k-2,k-1)
        # Ref: scalar_vt_vech_likelihood.m:77 — [(1:k-1)' (2:k)'] (1-based)
        # Converted to 0-based indexing for Python composite_likelihood.
        indices = np.column_stack([
            np.arange(k - 1, dtype=np.int64),
            np.arange(1, k, dtype=np.int64),
        ])
    elif composite == 2:
        # Full composite: all strictly lower-triangular pairs
        # Ref: scalar_vt_vech_likelihood.m:79-80 — meshgrid + ~triu mask
        # Converted to 0-based indexing for Python composite_likelihood.
        i_grid, j_grid = np.meshgrid(np.arange(k), np.arange(k))
        mask = ~np.triu(np.ones((k, k), dtype=bool))
        indices = np.column_stack([
            i_grid[mask].astype(np.int64),
            j_grid[mask].astype(np.int64),
        ])

    # ------------------------------------------------------------------
    # Main GARCH recursion
    # Ref: matrix_garch_likelihood.m:50-81
    #
    # MATLAB uses 1-based indexing: t = 1..T, (t-j)<1 triggers backcast.
    # Python uses 0-based indexing: t = 0..T-1, (t-j)<0 triggers backcast.
    # ------------------------------------------------------------------
    for t in range(T):
        # Intercept term: H(t) = CC'
        # Ref: matrix_garch_likelihood.m:51 — Ht(:,:,t)=parameterMatrices(:,:,1)
        Ht[:, :, t] = parameter_matrices[:, :, 0].copy()

        # ----------------------------------------------------------
        # Innovation (ARCH) terms: + A(j)A(j)' .* data(t-j)
        # Ref: matrix_garch_likelihood.m:52-58
        # j iterates from 1..p; parameter_matrices[:,:,j] is A(j)A(j)'
        # ----------------------------------------------------------
        for j in range(1, p + 1):
            if (t - j) < 0:
                # Pre-sample: use backcast
                # Ref: matrix_garch_likelihood.m:54
                Ht[:, :, t] += parameter_matrices[:, :, j] * back_cast
            else:
                # Ref: matrix_garch_likelihood.m:56 — Hadamard product (.*)
                Ht[:, :, t] += parameter_matrices[:, :, j] * data[:, :, t - j]

        # ----------------------------------------------------------
        # Asymmetric terms: + G(j)G(j)' .* data_asym(t-j)
        # Ref: matrix_garch_likelihood.m:59-65
        # j iterates from 1..o; parameter_matrices[:,:,p+j] is G(j)G(j)'
        # ----------------------------------------------------------
        for j in range(1, o + 1):
            if (t - j) < 0:
                # Ref: matrix_garch_likelihood.m:61
                Ht[:, :, t] += parameter_matrices[:, :, p + j] * back_cast_asym
            else:
                # Ref: matrix_garch_likelihood.m:63
                Ht[:, :, t] += parameter_matrices[:, :, p + j] * data_asym[:, :, t - j]

        # ----------------------------------------------------------
        # GARCH terms: + B(j)B(j)' .* H(t-j)
        # Ref: matrix_garch_likelihood.m:66-72
        # j iterates from 1..q; parameter_matrices[:,:,p+o+j] is B(j)B(j)'
        # ----------------------------------------------------------
        for j in range(1, q + 1):
            if (t - j) < 0:
                # Ref: matrix_garch_likelihood.m:68 — use backcast for H pre-sample
                Ht[:, :, t] += parameter_matrices[:, :, p + o + j] * back_cast
            else:
                # Ref: matrix_garch_likelihood.m:70
                Ht[:, :, t] += parameter_matrices[:, :, p + o + j] * Ht[:, :, t - j]

        # ----------------------------------------------------------
        # Compute per-observation log-likelihood
        # ----------------------------------------------------------
        if composite == 0:
            # Standard multivariate normal log-likelihood using the
            # correlation decomposition for numerical stability.
            # Ref: matrix_garch_likelihood.m:77-80
            #
            # Decompose H(t) = diag(Q) @ R @ diag(Q):
            #   Q_i = sqrt(H_{t,ii})        (conditional standard deviations)
            #   R   = H_t ./ (Q * Q')       (conditional correlation matrix)
            #   S   = data_t ./ (Q * Q')    (standardized outer-product data)
            #
            # Then: log|H_t| = 2*sum(log(Q)) + log|R|
            #        tr(H_t^{-1} data_t) = tr(R^{-1} S)

            # Ref: matrix_garch_likelihood.m:77 — Q=sqrt(diag(Ht(:,:,t)))
            Q_vec = np.sqrt(np.diag(Ht[:, :, t]))
            # Ref: matrix_garch_likelihood.m:78 — R=Ht(:,:,t)./(Q*Q')
            QQ = np.outer(Q_vec, Q_vec)
            R = Ht[:, :, t] / QQ
            # Ref: matrix_garch_likelihood.m:79 — stdresid=data(:,:,t)./(Q*Q')
            stdresid = data[:, :, t] / QQ

            # Ref: matrix_garch_likelihood.m:80
            # lls(t)=0.5*(likconst+2*sum(log(Q))+log(det(R))+trace(R^(-1)*stdresid))
            #
            # Using np.linalg.slogdet for numerical stability instead of
            # log(det(R)), and np.linalg.solve instead of explicit R^(-1).
            sign_R, logdet_R = np.linalg.slogdet(R)
            lls[t] = 0.5 * (
                likconst
                + 2.0 * np.sum(np.log(Q_vec))
                + logdet_R
                + np.trace(np.linalg.solve(R, stdresid))
            )
        else:
            # Composite likelihood mode (diagonal or full pairwise).
            # Ref: scalar_vt_vech_likelihood.m:115
            # composite_likelihood(Ht(:,:,t), data(:,:,t), indices)
            lls[t] = composite_likelihood(Ht[:, :, t], data[:, :, t], indices)

    # ------------------------------------------------------------------
    # Aggregate total log-likelihood
    # Ref: matrix_garch_likelihood.m:82 — ll = sum(lls)
    # ------------------------------------------------------------------
    ll = float(np.sum(lls))

    # ------------------------------------------------------------------
    # Numerical safeguard — penalty for degenerate parameter regions
    # Ref: matrix_garch_likelihood.m:85-86
    # if isnan(ll) || isinf(ll) || ll>1e7
    #     ll = 1e7;
    # end
    # ------------------------------------------------------------------
    if np.isnan(ll) or np.isinf(ll) or ll > 1e7:
        ll = 1e7

    return ll, lls, Ht
