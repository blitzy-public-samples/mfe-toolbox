"""
Orthogonal/Factor GARCH (O-GARCH) multivariate volatility model.

Estimates a multivariate GARCH model using Orthogonal (Factor) GARCH,
reducing volatility modelling to univariate GARCH problems via Principal
Component Analysis.  See Carroll 2000 ("An Introduction to O-GARCH").

The conditional covariance is:

    H_t = W' * F_t * W + Omega

where F_t is a diagonal matrix of conditional factor variances, W is a
NUMFACTORS x K matrix of factor loadings, and Omega is a diagonal matrix
of time-invariant idiosyncratic variances.  When NUMFACTORS == K,
Omega = 0.

Migrated from: multivariate/o_mvgarch.m — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 10/28/2009

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import numpy as np

from mfe_toolbox.univariate.tarch import tarch
from mfe_toolbox.crosssection.pca import pca

__all__ = ['o_mvgarch']


def o_mvgarch(
    data: np.ndarray,
    numfactors: int,
    p: 'int | np.ndarray' = 1,
    o: 'int | np.ndarray' = 0,
    q: 'int | np.ndarray' = 1,
    starting_vals: 'np.ndarray | None' = None,
    options: 'dict | None' = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Orthogonal/Factor GARCH multivariate volatility model.

    Uses Principal Component Analysis to decompose a K-asset dataset into
    orthogonal factors, fits a univariate TARCH(P,O,Q) model to each of
    the top ``numfactors`` principal components, and reconstructs the
    full K x K conditional covariance matrix at each time step.

    Parameters
    ----------
    data : numpy.ndarray
        T x K matrix of zero-mean residuals, where T is the number of
        observations and K is the number of assets.  Requires T > K > 1.
    numfactors : int
        Number of principal components to include in the model.  Must
        satisfy 1 <= numfactors <= K.
    p : int or array_like, optional
        Positive integer for the number of symmetric innovation (ARCH)
        lags.  A scalar value is broadcast to all factors; a vector of
        length ``numfactors`` specifies per-factor orders.  Default is 1.
    o : int or array_like, optional
        Non-negative integer for the number of asymmetric (threshold)
        lags.  A scalar is broadcast; a vector of length ``numfactors``
        specifies per-factor orders.  Default is 0.
    q : int or array_like, optional
        Non-negative integer for the number of conditional variance
        (GARCH) lags.  A scalar is broadcast; a vector of length
        ``numfactors`` specifies per-factor orders.  Default is 1.
    starting_vals : numpy.ndarray or None, optional
        Starting parameter vector for all factor TARCH models.  Length
        must be at least K + sum(P) + sum(O) + sum(Q).  If ``None``,
        starting values are determined automatically by the TARCH
        estimator.  Default is None.
    options : dict or None, optional
        Options dictionary passed to ``scipy.optimize.minimize`` via
        the TARCH estimator.  If ``None``, sensible defaults
        (``{'disp': False}``) are used.  Default is None.

    Returns
    -------
    parameters : numpy.ndarray
        1-D parameter vector concatenating per-factor TARCH parameters:

        ``[tarch(1)', tarch(2)', ..., tarch(numfactors)']``

        where each ``tarch(i)`` is
        ``[omega(i), alpha(i,1)...alpha(i,p(i)),
        gamma(i,1)...gamma(i,o(i)), beta(i,1)...beta(i,q(i))]``.

        Total length is ``numfactors + sum(p) + sum(o) + sum(q)``.
    ht : numpy.ndarray
        K x K x T array of conditional covariance matrices.
    w : numpy.ndarray
        K x K matrix of component weights from PCA (rows ordered by
        decreasing explained variance).
    pc : numpy.ndarray
        T x K matrix of principal components from PCA (columns ordered
        by decreasing explained variance).

    Raises
    ------
    ValueError
        If any input fails validation:

        - ``data`` is not 2-D with T > K > 1
        - ``numfactors`` is outside [1, K]
        - ``p`` contains non-positive or non-integer values
        - ``o`` or ``q`` contain negative or non-integer values
        - ``starting_vals`` is too short

    Notes
    -----
    The model decomposes the K-asset data matrix into ``numfactors``
    orthogonal factors via PCA (using the outer-product matrix), fits a
    univariate TARCH(P,O,Q) model (with normal errors, squared-GARCH
    specification) to each factor, then reconstructs the conditional
    covariance as:

    .. math::
        H_t = W^T \\, \\text{diag}(h_{1,t}, \\ldots, h_{F,t}) \\, W + \\Omega

    If ``numfactors < K``, the idiosyncratic variance matrix
    :math:`\\Omega` is set to ``diag(mean(e_t^2))``, where
    :math:`e_t = \\text{data} - \\text{pcs} \\times \\text{weights}`.
    If ``numfactors == K``, :math:`\\Omega = 0`.

    Ref: o_mvgarch.m — MATLAB MFE Toolbox, Kevin Sheppard,
    University of Oxford.

    See Also
    --------
    mfe_toolbox.crosssection.pca.pca : Principal Component Analysis.
    mfe_toolbox.univariate.tarch.tarch : Univariate TARCH/GJR-GARCH.
    mfe_toolbox.multivariate.ccc_mvgarch.ccc_mvgarch : CCC-MVGARCH.
    mfe_toolbox.multivariate.gogarch.gogarch : GO-GARCH.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((200, 3))
    >>> params, ht, w, pc = o_mvgarch(data, 2, p=1, o=0, q=1)
    >>> params.shape  # 2 factors * (1 + 1 + 0 + 1) = 6
    (6,)
    >>> ht.shape
    (3, 3, 200)
    """
    # ====================================================================
    # Input Argument Checking — Ref: o_mvgarch.m lines 54-131
    # ====================================================================

    # Ref: o_mvgarch.m:66 — [t,k] = size(data)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError('DATA must be a T by K matrix, T>K>1')

    t, k = data.shape

    # Ref: o_mvgarch.m:68-69 — min(t,k)<2 || t<k
    if min(t, k) < 2 or t < k:
        raise ValueError('DATA must be a T by K matrix, T>K>1')

    # Validate numfactors — must be between 1 and K (inclusive)
    numfactors = int(numfactors)
    if numfactors < 1 or numfactors > k:
        raise ValueError(
            'NUMFACTORS must be a positive integer between 1 and K.'
        )

    # ----------------------------------------------------------------
    # p validation — Ref: o_mvgarch.m lines 75-84
    # ----------------------------------------------------------------
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    if p_arr.size == 1:
        # Ref: o_mvgarch.m:76-77 — scalar: must be positive integer
        if p_arr[0] < 1 or np.floor(p_arr[0]) != p_arr[0]:
            raise ValueError('P must be a positive integer if scalar.')
        # Ref: o_mvgarch.m:79 — broadcast to numfactors-length vector
        p_arr = np.ones(numfactors, dtype=np.float64) * p_arr[0]
    else:
        # Ref: o_mvgarch.m:81-83 — vector: length must match numfactors,
        # all elements positive integers
        if (
            p_arr.size != numfactors
            or np.any(p_arr < 1)
            or np.any(np.floor(p_arr) != p_arr)
        ):
            raise ValueError(
                'P must contain K positive integer elements if a vector.'
            )
    p_vec = p_arr.astype(np.intp)

    # ----------------------------------------------------------------
    # o validation — Ref: o_mvgarch.m lines 87-96
    # ----------------------------------------------------------------
    o_arr = np.asarray(o, dtype=np.float64).ravel()
    if o_arr.size == 1:
        # Ref: o_mvgarch.m:88-89 — scalar: must be non-negative integer
        if o_arr[0] < 0 or np.floor(o_arr[0]) != o_arr[0]:
            raise ValueError('O must be a non-negative integer if scalar.')
        # Ref: o_mvgarch.m:91 — broadcast
        o_arr = np.ones(numfactors, dtype=np.float64) * o_arr[0]
    else:
        # Ref: o_mvgarch.m:93-95
        if (
            o_arr.size != numfactors
            or np.any(o_arr < 0)
            or np.any(np.floor(o_arr) != o_arr)
        ):
            raise ValueError(
                'O must contain K non-negative integer elements if a vector.'
            )
    o_vec = o_arr.astype(np.intp)

    # ----------------------------------------------------------------
    # q validation — Ref: o_mvgarch.m lines 99-108
    # ----------------------------------------------------------------
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    if q_arr.size == 1:
        # Ref: o_mvgarch.m:100-101 — scalar: must be non-negative integer
        if q_arr[0] < 0 or np.floor(q_arr[0]) != q_arr[0]:
            raise ValueError('Q must be a non-negative integer if scalar.')
        # Ref: o_mvgarch.m:103 — broadcast
        q_arr = np.ones(numfactors, dtype=np.float64) * q_arr[0]
    else:
        # Ref: o_mvgarch.m:105-107
        if (
            q_arr.size != numfactors
            or np.any(q_arr < 0)
            or np.any(np.floor(q_arr) != q_arr)
        ):
            raise ValueError(
                'Q must contain K non-negative integer elements if a vector.'
            )
    q_vec = q_arr.astype(np.intp)

    # ----------------------------------------------------------------
    # starting_vals validation — Ref: o_mvgarch.m lines 110-117
    # ----------------------------------------------------------------
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        # Ref: o_mvgarch.m:114 — length(startingVals)<(k+sum(p)+sum(o)+sum(q))
        # Note: MATLAB uses k (data columns), not numfactors, in this check.
        min_length = k + int(np.sum(p_vec)) + int(np.sum(o_vec)) + int(np.sum(q_vec))
        if len(starting_vals) < min_length:
            raise ValueError(
                'STARTINGVALS should be a K+sum(P)+sum(O)+sum(Q) by 1 vector'
            )

    # ----------------------------------------------------------------
    # options default — Ref: o_mvgarch.m lines 120-124
    # Replaces MATLAB optimset('fminunc') with Display='off', LargeScale='off'
    # using a minimal scipy.optimize.minimize options dict.
    # ----------------------------------------------------------------
    if options is None:
        # Ref: o_mvgarch.m:121-123
        options = {'disp': False}
    elif not isinstance(options, dict):
        raise ValueError('OPTIONS is not a valid options structure')

    # ====================================================================
    # PCA Decomposition — Ref: o_mvgarch.m line 134
    # ====================================================================
    # Ref: o_mvgarch.m:134 — [w, pc] = pca(data,'outer')
    # pca() returns 5 values; only weights and princomp are needed here.
    w, pc, _eigenvals, _explvar, _cumR2 = pca(data, 'outer')

    # Ref: o_mvgarch.m:138 — weights = w(1:numfactors,:)
    # Extract top numfactors rows (highest-variance components)
    weights = w[:numfactors, :]  # numfactors x K

    # Ref: o_mvgarch.m:139 — pcs = pc(:,1:numfactors)
    # Extract corresponding principal component columns
    pcs = pc[:, :numfactors]  # T x numfactors

    # ====================================================================
    # Per-Factor TARCH Estimation — Ref: o_mvgarch.m lines 141-156
    # ====================================================================
    # Ref: o_mvgarch.m:141 — htMat = zeros(t,numfactors)
    ht_mat = np.zeros((t, numfactors), dtype=np.float64)

    # Cell-array equivalents for per-factor results
    tarch_parameters_list: list[np.ndarray] = [
        np.empty(0) for _ in range(numfactors)
    ]

    for i in range(numfactors):
        # Ref: o_mvgarch.m:148 — tarchStartingVals = []
        tarch_starting_vals: np.ndarray | None = None

        if starting_vals is not None:
            # Ref: o_mvgarch.m:150 — parameterEnd = i + sum(p(1:i)) + sum(o(1:i)) + sum(q(1:i))
            # MATLAB is 1-based: MATLAB i maps to Python i+1 for cumulative sums
            parameter_end = (
                (i + 1)
                + int(np.sum(p_vec[: i + 1]))
                + int(np.sum(o_vec[: i + 1]))
                + int(np.sum(q_vec[: i + 1]))
            )
            # Ref: o_mvgarch.m:151 — parameterStart = parameterEnd - 1 - p(i) - o(i) - q(i) + 1
            parameter_start = (
                parameter_end - 1 - int(p_vec[i]) - int(o_vec[i]) - int(q_vec[i]) + 1
            )
            # Ref: o_mvgarch.m:152 — tarchStartingVals = startingVals(parameterStart:parameterEnd)
            # Convert MATLAB 1-based inclusive range to Python 0-based exclusive slice
            tarch_starting_vals = starting_vals[parameter_start - 1 : parameter_end]

        # Ref: o_mvgarch.m:154 —
        #   tarch(pcs(:,i), p(i), o(i), q(i), [], 2, tarchStartingVals, options)
        #
        # MATLAB arg order: (data, p, o, q, error_type, tarch_type, startingvals, options)
        # MATLAB passes:  error_type=[]  (default → NORMAL)
        #                 tarch_type=2   (squared GARCH)
        #
        # Python arg order: (epsilon, p, o, q, tarch_type, error_type, startingvals, options)
        # Python equivalents: tarch_type=2 ('GARCH' squared), error_type='NORMAL'
        params_i, _ll_i, ht_i, _vcv_robust_i, _vcv_i, _scores_i, _diag_i = tarch(
            pcs[:, i],
            int(p_vec[i]),
            int(o_vec[i]),
            int(q_vec[i]),
            tarch_type=2,
            error_type='NORMAL',
            startingvals=tarch_starting_vals,
            options=options,
        )

        tarch_parameters_list[i] = params_i
        # Ref: o_mvgarch.m:155 — htMat(:,i) = tarchHt{i}
        ht_mat[:, i] = ht_i.ravel()

    # ====================================================================
    # Idiosyncratic Variance — Ref: o_mvgarch.m lines 158-163
    # ====================================================================
    if numfactors < k:
        # Ref: o_mvgarch.m:159 — errors = data - pcs * weights
        errors = data - pcs @ weights
        # Ref: o_mvgarch.m:160 — omega = diag(mean(errors.^2))
        omega = np.diag(np.mean(errors ** 2, axis=0))
    else:
        # Ref: o_mvgarch.m:162 — omega = zeros(k)
        omega = np.zeros((k, k), dtype=np.float64)

    # ====================================================================
    # H_t Reconstruction — Ref: o_mvgarch.m lines 165-169
    # ====================================================================
    # Ref: o_mvgarch.m:165 — ht = zeros(k,k,t)
    ht = np.zeros((k, k, t), dtype=np.float64)

    for i in range(t):
        # Ref: o_mvgarch.m:168 — ht(:,:,i) = weights' * diag(htMat(i,:)) * weights + omega
        # weights is numfactors x K; weights.T is K x numfactors
        # diag(ht_mat[i,:]) is numfactors x numfactors diagonal matrix
        # Result: (K x numfactors) @ (numfactors x numfactors) @ (numfactors x K) + (K x K) = K x K
        ht[:, :, i] = weights.T @ np.diag(ht_mat[i, :]) @ weights + omega

    # ====================================================================
    # Parameter Assembly — Ref: o_mvgarch.m lines 171-176
    # ====================================================================
    # Ref: o_mvgarch.m:171 — parameters=zeros(numfactors+sum(p+o+q),1)
    total_params = numfactors + int(np.sum(p_vec + o_vec + q_vec))
    parameters = np.zeros(total_params, dtype=np.float64)

    # Ref: o_mvgarch.m:172 — count = 1 (MATLAB 1-based)
    count = 0  # Python 0-based

    for i in range(numfactors):
        # Each TARCH factor has 1 (omega) + p(i) + o(i) + q(i) parameters
        n_params_i = 1 + int(p_vec[i]) + int(o_vec[i]) + int(q_vec[i])
        # Ref: o_mvgarch.m:174 — parameters(count:count+p(i)+o(i)+q(i))=tarchParameters{i}
        # MATLAB uses inclusive indexing: count to count+p(i)+o(i)+q(i) = n_params_i elements
        parameters[count : count + n_params_i] = tarch_parameters_list[i][:n_params_i]
        # Ref: o_mvgarch.m:175 — count = count + 1 + p(i) + o(i) + q(i)
        count += n_params_i

    return parameters, ht, w, pc
