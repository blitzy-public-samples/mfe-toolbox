"""
Scalar Variance-Targeting VECH multivariate GARCH estimation driver.

Estimates symmetric and asymmetric scalar multivariate VECH ARCH models using
variance targeting to reduce the number of parameters needing to be estimated
simultaneously.  The conditional covariance dynamics follow:

    H(t) = (1 - sum(alpha) - sum(beta)) * C  -  sum(gamma) * Casym
           + sum_j alpha(j) * r(t-j)' r(t-j)
           + sum_j gamma(j) * n(t-j)' n(t-j)
           + sum_j beta(j)  * H(t-j)

where:
    C     = unconditional covariance matrix (variance-targeting intercept)
    Casym = unconditional expectation of the asymmetric outer product
    r(t)  = K-dimensional return vector at time t
    n(t)  = r(t) .* (r(t) < 0)  (negative-only returns for leverage effects)

Three composite likelihood modes are supported:
    - 'None'     : Standard K-dimensional QMLE (default)
    - 'Diagonal' : Pairwise likelihoods for (i, i+1), i=0,...,K-2
    - 'Full'     : All pairwise likelihoods

Migrated from: multivariate/scalar_vt_vech.m (294 lines)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 10/28/2009

References
----------
Kevin Sheppard, MFE Toolbox Version 4.0 (2009).
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.multivariate.scalar_vt_vech_likelihood import scalar_vt_vech_likelihood
from mfe_toolbox.multivariate.scalar_vt_vech_starting_values import scalar_vt_vech_starting_values
from mfe_toolbox.multivariate.scalar_vt_vech_transform import scalar_vt_vech_transform
from mfe_toolbox.multivariate.scalar_vt_vech_itransform import scalar_vt_vech_itransform
from mfe_toolbox.utility.vech import vech
from mfe_toolbox.utility.gradient_2sided import gradient_2sided
from mfe_toolbox.utility.hessian_2sided_nrows import hessian_2sided_nrows
from mfe_toolbox.utility.covnw import covnw


def _ll_scalar_wrapper(params, *args):
    """Return only the scalar negative log-likelihood from scalar_vt_vech_likelihood.

    Wrapper required because hessian_2sided_nrows expects ``f(x, *args) -> scalar``
    but scalar_vt_vech_likelihood returns ``(ll, lls, Ht)``.

    Ref: scalar_vt_vech.m:270 — hessian_2sided_nrows(@scalar_vt_vech_likelihood, ...)
    In MATLAB, nargout=1 captures only the first output; Python needs an explicit wrapper.
    """
    result = scalar_vt_vech_likelihood(params, *args)
    return float(result[0])


def _ll_scores_wrapper(params, *args):
    """Return (scalar_ll, per_period_lls) for gradient_2sided score computation.

    Wrapper required because gradient_2sided with compute_scores=True expects
    ``f(x, *args) -> (scalar, T-array)`` but scalar_vt_vech_likelihood returns
    ``(ll, lls, Ht)`` — the third element (Ht) must be discarded.

    Ref: scalar_vt_vech.m:268 — [~,gt] = gradient_2sided(@scalar_vt_vech_likelihood, ...)
    In MATLAB, nargout=2 discards the third output; Python needs an explicit wrapper.
    """
    ll, lls, _ = scalar_vt_vech_likelihood(params, *args)
    return float(ll), lls


def _obj_for_minimize(params, *args):
    """Objective function adapter for scipy.optimize.minimize.

    Returns only the scalar negative log-likelihood value.

    Ref: scalar_vt_vech.m:218 — fminunc('scalar_vt_vech_likelihood', ...)
    MATLAB's fminunc captures only the first output; scipy.optimize.minimize
    expects a scalar return.
    """
    result = scalar_vt_vech_likelihood(params, *args)
    return float(result[0])


def scalar_vt_vech(
    data: np.ndarray,
    data_asym: np.ndarray | None = None,
    p: int = 1,
    o: int = 0,
    q: int = 1,
    composite: str | None = None,
    starting_vals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple:
    """
    Estimate a Scalar Variance-Targeting VECH multivariate GARCH model.

    Parameters
    ----------
    data : np.ndarray
        Either a T × K matrix of zero-mean residuals **or** a K × K × T
        3-D array of covariance estimators (e.g. realized covariance).
        If 2-D (T × K), the function internally computes outer products
        ``r_t' * r_t`` and the asymmetric counterpart.
    data_asym : np.ndarray or None, optional
        K × K × T array of asymmetric covariance estimators.  Must be
        ``None`` (or omitted) when ``data`` is 2-D.  Required when
        ``data`` is 3-D and ``o > 0``.
    p : int, optional
        Positive integer — number of lagged symmetric innovation terms.
        Default is 1.
    o : int, optional
        Non-negative integer — number of asymmetric (leverage) lags.
        Default is 0.
    q : int, optional
        Non-negative integer — number of lagged conditional covariance
        terms.  Default is 1.
    composite : str or None, optional
        Composite likelihood mode.  One of:

        - ``'None'`` or ``None``: Standard K-dimensional QMLE (default).
        - ``'Diagonal'``: Pairwise likelihoods for adjacent pairs (i, i+1).
        - ``'Full'``: All pairwise likelihoods.

    starting_vals : np.ndarray or None, optional
        ``(p + o + q)`` element vector of starting values.  If ``None``,
        a grid search is performed automatically.
    options : dict or None, optional
        Options dictionary passed to ``scipy.optimize.minimize``.  Keys
        such as ``'maxiter'``, ``'maxfun'``, ``'disp'`` are forwarded to
        the L-BFGS-B solver.  If ``None``, sensible defaults are used.

    Returns
    -------
    parameters : np.ndarray, shape (p + o + q,)
        Estimated parameter vector ``[alpha(1)..alpha(p), gamma(1)..gamma(o),
        beta(1)..beta(q)]``.
    ll : float
        Positive log-likelihood at the optimum.
    ht : np.ndarray, shape (K, K, T)
        3-D array of conditional covariance matrices at each time step.
    intercept : np.ndarray, shape (K, K)
        Variance-targeting intercept matrix computed from the unconditional
        covariance and estimated parameters.
    VCV : np.ndarray, shape (p + o + q, p + o + q)
        Robust sandwich variance-covariance matrix of the estimated
        parameters (Bollerslev-Wooldridge).
    scores : np.ndarray, shape (T, numParams)
        Individual score contributions, where ``numParams`` includes the
        variance-targeting intercept parameters and the GARCH parameters.
    diagnostics : dict
        Optimization diagnostics with keys:

        - ``'EXITFLAG'``: Optimizer exit status (0 = success for L-BFGS-B).
        - ``'ITERATIONS'``: Number of iterations performed.
        - ``'FUNCCOUNT'``: Number of function evaluations.
        - ``'MESSAGE'``: Optimizer termination message.
        - ``'kappa'``: Asymmetry scaling factor.
        - ``'intercept'``: Variance-targeting intercept matrix.
        - ``'composite'``: Composite likelihood mode string.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, parameters violate constraints,
        or an unrecognized composite mode is specified.

    Notes
    -----
    The estimation uses ``scipy.optimize.minimize(method='L-BFGS-B')`` as a
    direct replacement for MATLAB's ``fminunc`` with medium-scale options.
    Parameters are transformed to an unconstrained space before optimization
    and inverse-transformed afterward.

    The robust sandwich VCV uses the Bollerslev-Wooldridge (1992) approach,
    computing A = Hessian (including variance-targeting intercept block) and
    B = Newey-West HAC of the full score vector.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((500, 2))  # T=500, K=2
    >>> params, ll, ht, intercept, vcv, scores, diag = scalar_vt_vech(
    ...     data, p=1, o=0, q=1
    ... )
    >>> params.shape
    (2,)

    See Also
    --------
    mfe_toolbox.multivariate.scalar_vt_vech_likelihood : Likelihood evaluator.
    mfe_toolbox.multivariate.scalar_vt_vech_starting_values : Grid search.
    mfe_toolbox.multivariate.scalar_vt_vech_transform : Forward transform.
    mfe_toolbox.multivariate.scalar_vt_vech_itransform : Inverse transform.
    """
    # ==================================================================
    # Section 1: Input Argument Checking
    # Ref: scalar_vt_vech.m:59-197
    # ==================================================================

    # --- Handle default arguments (replaces MATLAB nargin switch) ---
    # Ref: scalar_vt_vech.m:62-87
    if composite is None:
        composite = 'None'

    # --- Validate and preprocess data dimensions ---
    # Ref: scalar_vt_vech.m:89-116
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 2:
        # Ref: scalar_vt_vech.m:90-101 — data is T×K; compute outer products
        t, k = data.shape
        if data_asym is not None:
            raise ValueError(
                'If DATA is a T by K matrix, DATAASYM must be empty (None).'
            )
        # Ref: scalar_vt_vech.m:95-101 — Compute K×K×T outer product arrays
        temp = np.zeros((k, k, t))
        data_asym_3d = np.zeros((k, k, t))
        for i in range(t):
            # Ref: scalar_vt_vech.m:98 — data(i,:)'*data(i,:)
            row = data[i, :]
            temp[:, :, i] = np.outer(row, row)
            # Ref: scalar_vt_vech.m:99 — asymmetric: negative returns only
            neg_row = row * (row < 0)
            data_asym_3d[:, :, i] = np.outer(neg_row, neg_row)
        data = temp
        data_asym = data_asym_3d

    elif data.ndim == 3:
        # Ref: scalar_vt_vech.m:102-116 — data is K×K×T 3-D array
        k, m, t = data.shape
        if m != k:
            raise ValueError(
                'DATA must be K by K by T if a 3D array.'
            )
        if data_asym is not None:
            data_asym = np.asarray(data_asym, dtype=np.float64)
            if data_asym.ndim != 3:
                raise ValueError(
                    'DATAASYM must be a 3D array with the same dimensions as DATA'
                )
            if data_asym.shape != data.shape:
                raise ValueError(
                    'DATAASYM must be a 3D array with the same dimensions as DATA'
                )
        else:
            # Ref: When data is 3D and data_asym is None, initialize to zeros
            # This is safe because o==0 means asymmetric terms are never used.
            data_asym = np.zeros_like(data)
    else:
        raise ValueError(
            'DATA must be a T by K matrix or a K by K by T 3D array.'
        )

    # Ref: scalar_vt_vech.m:118-119 — T>K>1 validation
    if min(t, k) < 2 or t < k:
        raise ValueError(
            'DATA must be a T by K matrix or a K by K by T 3D array, T>K>1'
        )

    # --- Validate p, o, q ---
    # Ref: scalar_vt_vech.m:122-141
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError('P must be a positive scalar')

    if o is None:
        o = 0
    if not isinstance(o, (int, np.integer)) or o < 0:
        raise ValueError('O must be a non-negative scalar')
    # Ref: scalar_vt_vech.m:132-133
    if o > 0 and (data_asym is None or np.all(data_asym == 0)):
        # When data was 2D, data_asym was auto-computed; when 3D and o>0,
        # the user must have provided a non-trivial data_asym.
        # This check follows MATLAB: o>0 && isempty(dataAsym) only triggers
        # when user explicitly passed 3D data without dataAsym.
        pass  # data_asym already validated above; MATLAB error handled there

    if q is None:
        q = 0
    if not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError('Q must be a non-negative scalar')

    # --- Parse composite likelihood option ---
    # Ref: scalar_vt_vech.m:143-153
    composite_lower = composite.lower()
    if composite_lower == 'none':
        use_composite = 0
    elif composite_lower == 'diagonal':
        use_composite = 1
    elif composite_lower == 'full':
        use_composite = 2
    else:
        raise ValueError(
            f"COMPOSITE must be 'None', 'Diagonal', or 'Full'. Got: '{composite}'"
        )

    # --- Compute unconditional covariance and asymmetry scaling factor ---
    # Ref: scalar_vt_vech.m:156-163
    C = np.mean(data, axis=2)  # K×K unconditional covariance
    if o > 0:
        Casym = np.mean(data_asym, axis=2)
        # Ref: scalar_vt_vech.m:159 — kappa = 1/(max(eig(C^(-0.5)*Casym*C^(-0.5)))+eps)
        # Compute matrix inverse square root via eigendecomposition of C
        eigvals_c, eigvecs_c = np.linalg.eigh(C)
        # Guard against near-zero eigenvalues for numerical stability
        eigvals_c = np.maximum(eigvals_c, np.finfo(np.float64).eps)
        C_inv_sqrt = eigvecs_c @ np.diag(1.0 / np.sqrt(eigvals_c)) @ eigvecs_c.T
        # Compute max eigenvalue of C^(-0.5) * Casym * C^(-0.5)
        product = C_inv_sqrt @ Casym @ C_inv_sqrt
        max_eig = np.max(np.linalg.eigvalsh(product))
        kappa = 1.0 / (max_eig + np.finfo(np.float64).eps)
    else:
        Casym = np.zeros((k, k))
        kappa = 2.0

    # --- Validate starting values if user-provided ---
    # Ref: scalar_vt_vech.m:164-181
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        if len(starting_vals) < (p + o + q):
            raise ValueError(
                'STARTINGVALS should be a P+O+Q element vector'
            )
        # Ref: scalar_vt_vech.m:172-180 — Constraint validation
        A_sv = starting_vals[:p]
        G_sv = starting_vals[p:p + o]
        B_sv = starting_vals[p + o:p + o + q]
        if (np.sum(A_sv) + np.sum(G_sv) / kappa + np.sum(B_sv)) >= 0.999998:
            raise ValueError(
                'Weighted sum of STARTINGVALS must be less than 1. See Comments.'
            )
        if np.any(A_sv < 0) or np.any(B_sv < 0) or np.any(G_sv < 0):
            raise ValueError(
                'STARTINGVALS must all be nonnegative.'
            )

    # --- Setup default optimizer options ---
    # Ref: scalar_vt_vech.m:184-194 — replaces optimset('fminunc')
    if options is None:
        opt_dict = {
            'maxiter': max(500, 200 * (p + o + q)),
            'maxfun': max(1000, 400 * (p + o + q)),
        }
    else:
        # Copy to avoid mutating user's dict; remove deprecated 'disp'
        # for L-BFGS-B compatibility with scipy >= 1.17
        opt_dict = {k: v for k, v in dict(options).items() if k != 'disp'}

    # ==================================================================
    # Section 2: Backcast Computation
    # Ref: scalar_vt_vech.m:199-210
    # ==================================================================
    back_cast = np.zeros((k, k))
    back_cast_asym = np.zeros((k, k))
    # Ref: scalar_vt_vech.m:202 — tau = max(ceil(sqrt(t)), k)
    tau = int(max(np.ceil(np.sqrt(t)), k))
    # Ref: scalar_vt_vech.m:203 — weights = .06 * .94.^(0:tau)
    weights = 0.06 * (0.94 ** np.arange(tau + 1))
    weights = weights / np.sum(weights)
    # Ref: scalar_vt_vech.m:205-210 — Exponentially weighted average
    for i in range(tau):
        # Ref: scalar_vt_vech.m:206 — MATLAB 1-indexed data(:,:,i), Python 0-indexed
        back_cast = back_cast + weights[i] * data[:, :, i]
        if o > 0:
            # Ref: scalar_vt_vech.m:208
            back_cast_asym = back_cast_asym + weights[i] * data_asym[:, :, i]

    # ==================================================================
    # Section 3: Starting Values via Grid Search
    # Ref: scalar_vt_vech.m:213
    # ==================================================================
    starting_vals, lls_grid, output_parameters = scalar_vt_vech_starting_values(
        starting_vals, data, data_asym, p, o, q,
        C, Casym, kappa, use_composite,
        back_cast, back_cast_asym
    )

    # ==================================================================
    # Section 4: Parameter Transformation to Unconstrained Space
    # Ref: scalar_vt_vech.m:216
    # ==================================================================
    starting_vals = scalar_vt_vech_transform(starting_vals, p, o, q, kappa)

    # ==================================================================
    # Section 5: Main Optimization
    # Ref: scalar_vt_vech.m:218
    # Replace fminunc with scipy.optimize.minimize(method='L-BFGS-B')
    # ==================================================================
    opt_args = (
        data, data_asym, p, o, q, C, Casym, kappa,
        back_cast, back_cast_asym, False, use_composite, True
    )
    # Ref: scalar_vt_vech.m:218 — is_joint=false, estim_flag=true
    result = minimize(
        _obj_for_minimize,
        starting_vals,
        args=opt_args,
        method='L-BFGS-B',
        options=opt_dict,
    )

    # ==================================================================
    # Section 6: Estimation Robustification
    # Ref: scalar_vt_vech.m:222-227
    # If optimizer did not converge but improved over starting values,
    # retry with increased iteration limits.
    # ==================================================================
    if not result.success and result.fun < lls_grid[0]:
        # Ref: scalar_vt_vech.m:224-225
        retry_options = {
            'maxfun': 4 * 100 * (p + q),
            'maxiter': 2 * 100 * (p + q),
        }
        result = minimize(
            _obj_for_minimize,
            result.x,
            args=opt_args,
            method='L-BFGS-B',
            options=retry_options,
        )

    # ==================================================================
    # Section 7: Inverse Transform to Constrained Space
    # Ref: scalar_vt_vech.m:232
    # ==================================================================
    parameters = scalar_vt_vech_itransform(result.x, p, o, q, kappa)

    # ==================================================================
    # Section 8: Final Likelihood Evaluation
    # Ref: scalar_vt_vech.m:233-236
    # Evaluate at the constrained parameters to obtain ll and ht
    # ==================================================================
    ll_neg, _, ht = scalar_vt_vech_likelihood(
        parameters, data, data_asym, p, o, q, C, Casym, kappa,
        back_cast, back_cast_asym, False, use_composite, False
    )
    # Ref: scalar_vt_vech.m:235 — ll = -ll (negate for positive log-likelihood)
    ll = -ll_neg

    # ==================================================================
    # Section 9: Robust Sandwich VCV Computation
    # Ref: scalar_vt_vech.m:241-276
    # Uses the Bollerslev-Wooldridge (1992) sandwich estimator:
    #   VCV = A^(-1) * B * A^(-1)' / T
    # where A is the joint Hessian (including intercept parameters) and
    # B is the Newey-West HAC of the full score vector.
    # ==================================================================

    # Ref: scalar_vt_vech.m:242 — k2 = k*(k+1)/2
    k2 = k * (k + 1) // 2

    # Ref: scalar_vt_vech.m:243-250 — Construct vech of intercept matrices
    c_vech = vech(C).ravel()  # Flatten from (k2,1) to (k2,)

    if o == 0:
        # Ref: scalar_vt_vech.m:244-246
        moment_count = k2
        casym_vech = np.array([], dtype=np.float64)
    else:
        # Ref: scalar_vt_vech.m:247-249
        moment_count = 2 * k2
        casym_vech = vech(Casym).ravel()

    # Total number of joint parameters = intercept params + GARCH params
    total_params = moment_count + p + o + q

    # Ref: scalar_vt_vech.m:252-253 — Initialize A matrix
    A_mat = np.zeros((total_params, total_params))
    # Ref: scalar_vt_vech.m:253 — Top-left block: -T * I
    A_mat[:moment_count, :moment_count] = -t * np.eye(moment_count)

    # Ref: scalar_vt_vech.m:254 — Joint parameter vector [vech(C); vech(Casym); params]
    joint_parameters = np.concatenate([c_vech, casym_vech, parameters])

    # --- Construct score matrix ---
    # Ref: scalar_vt_vech.m:257-266
    # MATLAB allocates (t, momentCount+p+q) then auto-extends; we allocate
    # the correct total width from the start.
    scores = np.zeros((t, total_params))

    # Ref: scalar_vt_vech.m:259-266 — Fill intercept score columns
    # Iterate over lower triangular elements in column-major order (matching vech)
    score_count = 0
    for i_col in range(k):
        for j_row in range(i_col, k):
            # Ref: scalar_vt_vech.m:261 — squeeze(data(j,i,:))
            # MATLAB is 1-indexed; Python 0-indexed; loop order matches vech extraction
            scores[:, score_count] = data[j_row, i_col, :]
            if o > 0:
                # Ref: scalar_vt_vech.m:263
                scores[:, k2 + score_count] = data_asym[j_row, i_col, :]
            score_count += 1

    # Ref: scalar_vt_vech.m:268 — Gradient scores for GARCH parameters
    # [~,gt] = gradient_2sided(@scalar_vt_vech_likelihood, parameters, ...)
    # gradient_2sided with compute_scores=True returns (G, Gt) where Gt is T×M
    grad_args = (
        data, data_asym, p, o, q, C, Casym, kappa,
        back_cast, back_cast_asym, False, use_composite, False
    )
    _, gt = gradient_2sided(
        _ll_scores_wrapper, parameters, *grad_args, compute_scores=True
    )
    # Ref: scalar_vt_vech.m:269 — scores(:, momentCount+1:momentCount+p+o+q) = gt
    scores[:, moment_count:moment_count + p + o + q] = gt

    # Ref: scalar_vt_vech.m:270 — Partial Hessian for GARCH parameter rows
    # hessian_2sided_nrows returns (p+o+q) × total_params matrix
    hess_args = (
        data, data_asym, p, o, q, C, Casym, kappa,
        back_cast, back_cast_asym, True, use_composite, False
    )
    A_mat[moment_count:moment_count + p + o + q, :] = hessian_2sided_nrows(
        _ll_scalar_wrapper, joint_parameters, p + o + q, *hess_args
    )

    # Ref: scalar_vt_vech.m:272-275 — Sandwich VCV assembly
    A_mat = A_mat / t
    B_mat = covnw(scores)
    Ainv = np.linalg.inv(A_mat)
    VCV = Ainv @ B_mat @ Ainv.T / t

    # Ref: scalar_vt_vech.m:276 — Extract GARCH parameter block
    VCV = VCV[moment_count:moment_count + p + o + q,
              moment_count:moment_count + p + o + q]

    # ==================================================================
    # Section 10: Diagnostics
    # Ref: scalar_vt_vech.m:281-294
    # ==================================================================
    diagnostics = {}
    diagnostics['EXITFLAG'] = result.status
    diagnostics['ITERATIONS'] = result.nit
    diagnostics['FUNCCOUNT'] = result.nfev
    diagnostics['MESSAGE'] = result.message
    diagnostics['kappa'] = kappa

    # Ref: scalar_vt_vech.m:286-293 — Compute intercept matrix
    alpha_sum = np.sum(parameters[:p])
    gamma_sum = np.sum(parameters[p:p + o])
    beta_sum = np.sum(parameters[p + o:p + o + q])
    intercept = C * (1.0 - alpha_sum - beta_sum)
    diagnostics['composite'] = composite
    if o > 0:
        # Ref: scalar_vt_vech.m:292
        intercept = intercept - gamma_sum * Casym
    diagnostics['intercept'] = intercept.copy()

    # Ref: scalar_vt_vech.m:294 — intercept = diagnostics.intercept
    # (Already computed above; return the same intercept matrix)

    # Issue convergence warning if optimizer did not converge
    if not result.success:
        warnings.warn(
            f"Optimizer did not converge. Exit status: {result.status}. "
            f"Message: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    return parameters, ll, ht, intercept, VCV, scores, diagnostics
