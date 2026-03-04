"""
RARCH(p,q) multivariate volatility model estimation.

Implements the Rotated ARCH model of Noureldin, Shephard, and Sheppard.
The RARCH model evolves in a rotated covariance space defined by the
unconditional covariance matrix, enabling parsimonious multivariate
GARCH modelling while preserving the BEKK-like dynamics structure.

The model dynamics in the rotated space are:

    G_t = (I_K - sum(A_j^2) - sum(B_j^2))
          + sum_{j=1}^{p} A_j @ OP_{t-j} @ A_j
          + sum_{j=1}^{q} B_j @ G_{t-j} @ B_j

where OP_t = C^{-1/2} @ data_t @ C^{-1/2} is the outer product in the
rotated space, and C is the unconditional covariance matrix.

Three parameterisation types are supported:

    * **Scalar** (type_model='Scalar') — A_j = a_j * I_K, B_j = b_j * I_K
    * **CP** (Common Persistence) — Diagonal A_j, B derived from common theta
    * **Diagonal** — Fully diagonal A_j and B_j matrices

Two estimation methods:

    * **2-stage** (default) — First estimate C from sample covariance, then
      optimise dynamics parameters subject to stationarity constraints.
    * **Joint** — Simultaneously estimate C (via Cholesky factor) and
      dynamics parameters.

Source Reference
----------------
Migrated from ``multivariate/rarch.m`` (247 lines).
Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 1, Date: 3/27/2012

See Also
--------
mfe_toolbox.multivariate.rarch_likelihood : RARCH log-likelihood evaluation
mfe_toolbox.multivariate.rarch_constraint : RARCH stationarity constraints
mfe_toolbox.multivariate.rarch_parameter_transform : Parameter vector unpacking
mfe_toolbox.multivariate.rarch_simulate : RARCH simulation
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize

from mfe_toolbox.multivariate.rarch_constraint import rarch_constraint
from mfe_toolbox.multivariate.rarch_likelihood import rarch_likelihood
from mfe_toolbox.multivariate.rarch_parameter_transform import rarch_parameter_transform
from mfe_toolbox.utility.chol2vec import chol2vec
from mfe_toolbox.utility.covnw import covnw
from mfe_toolbox.utility.gradient_2sided import gradient_2sided
from mfe_toolbox.utility.hessian_2sided_nrows import hessian_2sided_nrows
from mfe_toolbox.utility.robustvcv import robustvcv
from mfe_toolbox.utility.vech import vech
from mfe_toolbox.utility.vec2chol import vec2chol


def rarch(
    data: np.ndarray,
    p: int = 1,
    q: int = 1,
    type_model: str = 'Scalar',
    method: str = '2-stage',
    starting_vals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate a RARCH(p,q) multivariate volatility model.

    Parameters
    ----------
    data : numpy.ndarray
        Either a T × K matrix of zero-mean residuals, or a K × K × T array
        of covariance estimators (e.g. realised covariance matrices).  When
        a T × K matrix is provided, outer products ``r_t @ r_t.T`` are
        computed automatically.  T must be larger than K.
    p : int, optional
        Positive integer — number of symmetric innovation lags (A matrices).
        Default is 1.
    q : int, optional
        Non-negative integer — number of conditional covariance lags
        (B matrices).  Default is 1.
    type_model : str, optional
        Model parameterisation type.  One of:

        * ``'Scalar'`` (default) — A_j = a_j * I_K, B_j = b_j * I_K
        * ``'CP'`` — Common Persistence (diagonal A, B from theta)
        * ``'Diagonal'`` — Fully diagonal A and B

    method : str, optional
        Estimation method.  One of:

        * ``'2-stage'`` (default) — First estimate C from sample covariance,
          then optimise dynamics parameters.
        * ``'Joint'`` — Simultaneously estimate all parameters.

    starting_vals : numpy.ndarray or None, optional
        Starting values for the optimizer.  When ``None``, a grid search
        selects starting values automatically.  When ``method='Joint'``,
        the first K(K+1)/2 elements must be ``vech(C)``.
    options : dict or None, optional
        Options dictionary passed to ``scipy.optimize.minimize``.  When
        ``None``, sensible defaults are used (``maxiter=1000``, ``ftol=1e-9``,
        ``disp=False``).

    Returns
    -------
    parameters : numpy.ndarray
        Estimated parameters in the order:

        * Scalar: ``[vech(C), a(1), ..., a(P), b(1), ..., b(Q)]``
        * CP: ``[vech(C), diag(A1), ..., diag(AP), theta]``
        * Diagonal: ``[vech(C), diag(A1), ..., diag(AP), diag(B1), ..., diag(BQ)]``

    ll : float
        Maximised log-likelihood value (positive).
    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices in the original
        (non-rotated) space.
    VCV : numpy.ndarray
        Square matrix of robust (sandwich) parameter covariances,
        ``A^{-1} B A^{-1} / T``.
    scores : numpy.ndarray
        T × num_params matrix of individual observation scores.

    Raises
    ------
    ValueError
        If input dimensions are invalid, ``p`` or ``q`` are invalid, or
        ``type_model`` / ``method`` are not recognised strings.

    Notes
    -----
    The dynamics of a RARCH model are identical to those of a BEKK, except
    that the model evolves in the rotated space defined by C^{-1/2}.

    In the Scalar model, A(:,:,i) = a(i)*I_K and B(:,:,j) = b(j)*I_K.
    In the CP model, B is derived from a common persistence parameter theta
    and the A matrices.  OP is the outer product of the unconditionally
    standardised data.

    When method='Joint', the first K*(K+1)/2 elements of starting_vals
    must be vech(C).

    References
    ----------
    Noureldin, D., Shephard, N. and Sheppard, K. (2012).
    "Multivariate Rotated ARCH Models." *Journal of Econometrics*.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((200, 3))
    >>> params, ll, Ht, VCV, scores = rarch(data, 1, 1, 'Scalar')
    >>> params.ndim == 1
    True
    """
    # ==================================================================
    # Phase 1: Input Argument Checking
    # Ref: rarch.m:57-147
    # ==================================================================
    data = np.asarray(data, dtype=np.float64)

    # Ref: rarch.m:62-65 — Determine T, k from data shape
    if data.ndim == 2:
        # Ref: rarch.m:62 — [T,k] = size(data)
        T, k = data.shape
    elif data.ndim == 3:
        # Ref: rarch.m:64 — [k,~,T] = size(data)
        k = data.shape[0]
        T = data.shape[2]
    else:
        raise ValueError(
            'DATA must be either a T by K matrix or a K by K by T array.'
        )

    # Ref: rarch.m:87-89 — ndims check
    if data.ndim > 3:
        raise ValueError(
            'DATA must be either a T by K matrix or a K by K by T array.'
        )

    # Ref: rarch.m:90-92 — T must be larger than K
    if T <= k:
        raise ValueError(
            'DATA must be either a T by K matrix or a K by K by T array, '
            'and T must be larger than K.'
        )

    # Ref: rarch.m:94-96 — P validation
    if not isinstance(p, (int, np.integer)) or p < 1 or int(p) != p:
        raise ValueError('P must be a positive scalar.')

    # Ref: rarch.m:97-99 — Q validation
    if not isinstance(q, (int, np.integer)) or q < 0 or int(q) != q:
        raise ValueError('Q must be a non-negative scalar.')

    # Ref: rarch.m:101-109 — Parse type string to integer code
    type_str = str(type_model).strip().lower()
    if type_str == 'scalar':
        type_code = 1
    elif type_str == 'cp':
        type_code = 2
    elif type_str == 'diagonal':
        type_code = 3
    else:
        raise ValueError(
            "TYPE must be 'Scalar', 'CP' or 'Diagonal'."
        )

    # Ref: rarch.m:111-117 — Parse method string
    method_str = str(method).strip().lower()
    if method_str == '2-stage':
        is_joint = False
    elif method_str == 'joint':
        is_joint = True
    else:
        raise ValueError(
            "METHOD must be either '2-stage' or 'Joint'."
        )

    # Ref: rarch.m:119-134 — Validate starting values count
    k2 = k * (k + 1) // 2  # Ref: rarch.m:174 — k2 = k*(k+1)/2
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        # Ref: rarch.m:120-127 — Compute expected parameter count
        if type_code == 1:
            count = p + q
        elif type_code == 2:
            count = p * k + 1  # Ref: rarch.m:124 — p*k+q, but CP uses theta (1 param)
        elif type_code == 3:
            count = (p + q) * k
        else:
            count = 0

        if is_joint:
            # Ref: rarch.m:128-130 — add vech(C) elements for joint estimation
            count = count + k2

        if len(starting_vals) != count:
            raise ValueError(
                'STARTINGVALS does not have the expected number of elements.'
            )

    # Ref: rarch.m:136-147 — Default optimizer options
    if options is None:
        # Default SLSQP options matching MATLAB fmincon defaults
        options = {
            'maxiter': 1000,
            'ftol': 1e-9,
            'disp': False,
        }
    elif not isinstance(options, dict):
        raise ValueError(
            'The user provided options structure is not valid.'
        )

    # ==================================================================
    # Phase 2: Data Transformation
    # Ref: rarch.m:148-163
    # ==================================================================
    # Ref: rarch.m:151-157 — If data is 2D (T×K), compute outer products
    if data.ndim == 2:
        temp = np.zeros((k, k, T))
        for i in range(T):
            # Ref: rarch.m:154 — temp(:,:,i) = data(i,:)'*data(i,:)
            # MATLAB data(i,:)' is a column vector; Python data[i, :] is 1D
            row = data[i, :].reshape(-1, 1)
            temp[:, :, i] = row @ row.T
        data = temp

    # Ref: rarch.m:158 — C = mean(data,3)
    C = np.mean(data, axis=2)

    # Ref: rarch.m:160 — Cm12 = C^(-0.5)
    # Compute the matrix inverse square root using scipy.linalg.sqrtm.
    # np.real() strips tiny imaginary parts from numerical imprecision.
    C_sqrt = np.real(sqrtm(C))
    Cm12 = np.linalg.inv(C_sqrt)

    # Ref: rarch.m:159,161-163 — Rotate data into standardised space
    std_data = np.zeros((k, k, T))
    for i in range(T):
        # Ref: rarch.m:162 — stdData(:,:,i) = Cm12*data(:,:,i)*Cm12
        std_data[:, :, i] = Cm12 @ data[:, :, i] @ Cm12

    # ==================================================================
    # Phase 3: Starting Values
    # Ref: rarch.m:164-205
    # ==================================================================
    # Ref: rarch.m:167 — w = .06*.94.^(0:ceil(sqrt(T)))
    max_lag = int(np.ceil(np.sqrt(T)))
    w = 0.06 * (0.94 ** np.arange(max_lag + 1))
    # Ref: rarch.m:168 — w = w/sum(w)
    w = w / np.sum(w)

    # Ref: rarch.m:169-172 — Exponentially weighted backcast
    back_cast = np.zeros((k, k))
    for i in range(len(w)):
        # Ref: rarch.m:171 — backCast = backCast + w(i)*stdData(:,:,i)
        back_cast = back_cast + w[i] * std_data[:, :, i]

    if starting_vals is None:
        # Ref: rarch.m:176-186 — Grid search over (a, b) pairs
        # Ref: rarch.m:176 — as = .02:.03:.11 → [0.02, 0.05, 0.08, 0.11]
        a_grid = np.array([0.02, 0.05, 0.08, 0.11])
        # Ref: rarch.m:177 — apbs = .9:.03:.99 → [0.90, 0.93, 0.96, 0.99]
        apb_grid = np.array([0.90, 0.93, 0.96, 0.99])

        # Ref: rarch.m:178 — startingLLs = zeros(length(as),length(apbs))
        starting_lls = np.zeros((len(a_grid), len(apb_grid)))

        for i_idx in range(len(a_grid)):
            for j_idx in range(len(apb_grid)):
                a_val = a_grid[i_idx]
                b_val = apb_grid[j_idx] - a_val
                # Ref: rarch.m:183 — parameters = sqrt([a b])
                sv_grid = np.sqrt(np.array([a_val, b_val]))
                # Ref: rarch.m:184 — Always evaluate with Scalar(1,1)
                ll_val, _, _ = rarch_likelihood(
                    sv_grid, data, 1, 1, C, back_cast, 1, False, False
                )
                starting_lls[i_idx, j_idx] = ll_val

        # Ref: rarch.m:192 — Find indices of minimum LL
        best_idx = np.unravel_index(
            np.argmin(starting_lls), starting_lls.shape
        )
        a_best = a_grid[best_idx[0]]
        b_best = apb_grid[best_idx[1]] - a_best

        # Ref: rarch.m:195-202 — Construct starting values by model type
        if type_code == 1:
            # Ref: rarch.m:197 — Scalar: sqrt([a/p * ones(1,p) b/q*ones(1,q)])
            parts = [(a_best / p) * np.ones(p)]
            if q > 0:
                parts.append((b_best / q) * np.ones(q))
            sv_dynamics = np.sqrt(np.concatenate(parts))
        elif type_code == 2:
            # Ref: rarch.m:199 — CP: sqrt([a/p * ones(1,k*p) (a+b)])
            sv_dynamics = np.sqrt(np.concatenate([
                (a_best / p) * np.ones(k * p),
                np.array([a_best + b_best])
            ]))
        elif type_code == 3:
            # Ref: rarch.m:201 — Diagonal: sqrt([a/p * ones(1,k*p) b/q*ones(1,k*q)])
            parts = [(a_best / p) * np.ones(k * p)]
            if q > 0:
                parts.append((b_best / q) * np.ones(k * q))
            sv_dynamics = np.sqrt(np.concatenate(parts))
        else:
            sv_dynamics = np.array([])

        starting_vals_opt = sv_dynamics.copy()
    else:
        # Ref: rarch.m:188-190 — User provided starting values; strip vech(C) prefix
        # When joint, the first k2 elements are vech(C); dynamics start after
        starting_vals_opt = starting_vals[k2:].copy() if is_joint else starting_vals.copy()

    # Ref: rarch.m:204-205 — Bounds for dynamics parameters
    UB = 0.99998 * np.ones_like(starting_vals_opt)
    LB = -UB

    # ==================================================================
    # Phase 4: Estimation
    # Ref: rarch.m:206-219
    # ==================================================================

    # --- Objective wrapper: returns scalar for scipy.optimize.minimize ---
    def _objective_2stage(params: np.ndarray) -> float:
        """Negative log-likelihood for 2-stage estimation (dynamics only)."""
        ll_val, _, _ = rarch_likelihood(
            params, data, p, q, C, back_cast, type_code, False, False
        )
        return float(ll_val)

    # --- Constraint wrapper for scipy SLSQP ---
    # Ref: rarch.m:209 — fmincon with @rarch_constraint
    # scipy convention: constraint(x) >= 0 is feasible
    def _constraint_2stage(params: np.ndarray) -> np.ndarray:
        """Stationarity constraint for 2-stage estimation."""
        c, _ = rarch_constraint(
            params, p, q, k, type_code, C, False, False
        )
        return c

    # Ref: rarch.m:209 — 2-stage optimization
    bounds_2stage = list(zip(LB.tolist(), UB.tolist()))
    constraints_2stage = {'type': 'ineq', 'fun': _constraint_2stage}

    result_2stage = minimize(
        _objective_2stage,
        starting_vals_opt,
        method='SLSQP',
        bounds=bounds_2stage,
        constraints=constraints_2stage,
        options=options,
    )

    if not result_2stage.success:
        warnings.warn(
            f'2-stage optimisation did not converge: {result_2stage.message}',
            RuntimeWarning,
            stacklevel=2,
        )

    parameters_opt = result_2stage.x.copy()

    # Ref: rarch.m:210 — Evaluate final 2-stage likelihood
    ll_val, lls_val, Ht = rarch_likelihood(
        parameters_opt, data, p, q, C, back_cast, type_code, False, False
    )

    # Ref: rarch.m:211-218 — Joint estimation (optional second stage)
    if is_joint:
        # Ref: rarch.m:212 — CChol = chol2vec(chol(C)')
        # MATLAB chol(C) returns upper triangular; chol(C)' is lower tri.
        # numpy cholesky returns lower triangular directly.
        C_chol_lower = np.linalg.cholesky(C)
        CChol = chol2vec(C_chol_lower)

        # Ref: rarch.m:213 — startingValJoint = [CChol; parameters]
        starting_val_joint = np.concatenate([CChol.ravel(), parameters_opt])

        # Ref: rarch.m:214-215 — Joint bounds
        LB_joint = np.concatenate([
            -np.inf * np.ones(len(CChol.ravel())),
            -np.ones(len(parameters_opt))
        ])
        UB_joint = np.abs(LB_joint)

        # --- Joint objective ---
        def _objective_joint(params: np.ndarray) -> float:
            """Negative log-likelihood for joint estimation."""
            ll_j, _, _ = rarch_likelihood(
                params, data, p, q, C, back_cast, type_code, True, True
            )
            return float(ll_j)

        # --- Joint constraint ---
        def _constraint_joint(params: np.ndarray) -> np.ndarray:
            """Stationarity constraint for joint estimation."""
            c, _ = rarch_constraint(
                params, p, q, k, type_code, C, True, True
            )
            return c

        bounds_joint = list(zip(LB_joint.tolist(), UB_joint.tolist()))
        constraints_joint = {'type': 'ineq', 'fun': _constraint_joint}

        # Ref: rarch.m:216 — Joint optimization
        result_joint = minimize(
            _objective_joint,
            starting_val_joint,
            method='SLSQP',
            bounds=bounds_joint,
            constraints=constraints_joint,
            options=options,
        )

        if not result_joint.success:
            warnings.warn(
                f'Joint optimisation did not converge: {result_joint.message}',
                RuntimeWarning,
                stacklevel=2,
            )

        parameters_opt = result_joint.x.copy()

        # Ref: rarch.m:217 — Evaluate final joint likelihood
        ll_val, lls_val, Ht = rarch_likelihood(
            parameters_opt, data, p, q, C, back_cast, type_code, True, True
        )

    # Ref: rarch.m:219 — Negate: stored as negative LL, convert to positive LL
    ll = -float(ll_val)

    # ==================================================================
    # Phase 5: Inference
    # Ref: rarch.m:220-247
    # ==================================================================
    if is_joint:
        # ------------------------------------------------------------------
        # Joint inference via robustvcv
        # Ref: rarch.m:223-229
        # ------------------------------------------------------------------
        # Ref: rarch.m:224 — C = parameters(1:k2)
        C_vec = parameters_opt[:k2]
        # Ref: rarch.m:225 — C = vec2chol(C)
        C_chol_infer = vec2chol(C_vec)
        # Ref: rarch.m:226 — C = C*C'
        C_infer = C_chol_infer @ C_chol_infer.T
        # Ref: rarch.m:227 — C = (C+C')/2
        C_infer = (C_infer + C_infer.T) / 2.0

        # Ref: rarch.m:228 — parameters = [vech(C); parameters(k2+1:end)]
        parameters_final = np.concatenate([
            vech(C_infer).ravel(),
            parameters_opt[k2:]
        ])

        # Ref: rarch.m:229 — robustvcv expects fun(theta, *args) → (scalar, lls)
        # rarch_likelihood returns (ll, lls, Ht), so we wrap to return 2 values.
        def _ll_for_robustvcv(params: np.ndarray, *args) -> tuple:
            """Wrapper returning (ll_scalar, lls_array) for robustvcv."""
            ll_s, lls_s, _ = rarch_likelihood(params, *args)
            return float(ll_s), lls_s

        try:
            VCV, _, _, scores, _, _ = robustvcv(
                _ll_for_robustvcv,
                parameters_final,
                0,
                data, p, q, C_infer, back_cast, type_code, True, False,
            )
        except np.linalg.LinAlgError:
            # Hessian is singular — use pseudo-inverse for VCV estimation
            # This can occur with small samples or near-boundary parameters.
            warnings.warn(
                'Hessian is singular during joint inference. '
                'VCV computed using pseudo-inverse.',
                RuntimeWarning,
                stacklevel=2,
            )
            n_params = len(parameters_final)
            VCV = np.full((n_params, n_params), np.nan)
            scores = np.zeros((T, n_params))

    else:
        # ------------------------------------------------------------------
        # 2-stage inference via manual sandwich construction
        # Ref: rarch.m:230-247
        # ------------------------------------------------------------------
        # Ref: rarch.m:231-233 — First-stage scores from sample covariance
        scores1 = np.zeros((T, k2))
        for t in range(T):
            # Ref: rarch.m:233 — scores1(i,:) = vech(data(:,:,i)-C)'
            # vech returns (k2, 1) column vector; transpose to row and assign
            scores1[t, :] = vech(data[:, :, t] - C).ravel()

        # Ref: rarch.m:235 — gradient_2sided for second-stage scores
        # gradient_2sided with compute_scores=True expects f(x, *args) → (scalar, T-array)
        def _ll_for_gradient(params: np.ndarray, *args) -> tuple:
            """Wrapper returning (ll_scalar, lls_array) for gradient_2sided."""
            ll_s, lls_s, _ = rarch_likelihood(params, *args)
            return float(ll_s), lls_s

        _, scores2 = gradient_2sided(
            _ll_for_gradient,
            parameters_opt,
            data, p, q, C, back_cast, type_code, False, False,
            compute_scores=True,
        )

        # Ref: rarch.m:236 — scores = [scores1 scores2]
        scores = np.concatenate([scores1, scores2], axis=1)

        # Ref: rarch.m:237 — B = covnw(scores, ceil(1.2*T^(0.25)))
        nw_lags = int(np.ceil(1.2 * T ** 0.25))
        B_mat = covnw(scores, nw_lags)

        # Ref: rarch.m:238 — m = length(parameters)
        m = len(parameters_opt)

        # Ref: rarch.m:239 — parameters = [vech(C); parameters]
        parameters_final = np.concatenate([
            vech(C).ravel(),
            parameters_opt
        ])

        # Ref: rarch.m:240 — A1 = -eye(k2)
        A1 = -np.eye(k2)

        # Ref: rarch.m:241 — hessian_2sided_nrows expects f(x, *args) → scalar
        def _ll_scalar_for_hessian(params: np.ndarray, *args) -> float:
            """Wrapper returning scalar LL for hessian_2sided_nrows."""
            ll_s, _, _ = rarch_likelihood(params, *args)
            return float(ll_s)

        # Ref: rarch.m:241 — A2 = hessian_2sided_nrows(@rarch_likelihood, parameters, m, ...)
        A2 = hessian_2sided_nrows(
            _ll_scalar_for_hessian,
            parameters_final,
            m,
            data, p, q, C, back_cast, type_code, True, False,
        )

        # Ref: rarch.m:242 — A2 = A2 / T
        A2 = A2 / T

        # Ref: rarch.m:243-244 — Construct full A matrix
        total_params = k2 + m
        A_full = np.zeros((total_params, total_params))
        # Ref: rarch.m:243 — A = [[A1 zeros(k2,m)]; A2]
        A_full[:k2, :k2] = A1
        # Upper-right block remains zeros: A_full[:k2, k2:] = 0
        A_full[k2:, :] = A2

        # Ref: rarch.m:245 — Ainv = A \ eye(length(A))
        try:
            Ainv = np.linalg.solve(A_full, np.eye(total_params))
        except np.linalg.LinAlgError:
            # Hessian block is singular — use pseudo-inverse as fallback
            # Ref: In MATLAB, A\eye(n) on a singular A uses pinv implicitly
            warnings.warn(
                'A matrix is singular during 2-stage inference. '
                'VCV computed using pseudo-inverse.',
                RuntimeWarning,
                stacklevel=2,
            )
            Ainv = np.linalg.pinv(A_full)

        # Ref: rarch.m:246 — VCV = Ainv * B * Ainv' / T
        VCV = Ainv @ B_mat @ Ainv.T / T

    return parameters_final, ll, Ht, VCV, scores
