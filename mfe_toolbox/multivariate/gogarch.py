"""
OGARCH(p,q) and GOGARCH(p,q) multivariate volatility model estimation.

Implements the Generalized Orthogonal GARCH (GO-GARCH) and Orthogonal GARCH
(O-GARCH) models for multivariate conditional covariance estimation.  Both
models decompose the conditional covariance via eigendecomposition and optional
rotation of the principal components:

* **O-GARCH**: Uses identity rotation (U = I_K), fitting independent GARCH
  processes on PCA-whitened components.
* **GO-GARCH**: Adds a free orthonormal rotation matrix U parametrised by
  K(K-1)/2 Givens rotation angles (phi), jointly optimising rotation and
  volatility parameters.

The estimation pipeline follows the MATLAB reference:

1. Input validation and default argument handling.
2. Eigendecomposition of the mean covariance for whitening.
3. Starting value generation via per-series TARCH fits.
4. Per-series OGARCH optimisation with unit-sum constraint.
5. (GO-GARCH only) Joint rotation + volatility optimisation.
6. Sandwich VCV inference using Newey-West HAC scores.

Migrated from ``multivariate/gogarch.m`` (257 lines).
Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 1, Date: 4/15/2012

Notes
-----
* Numerical parity target: ±1e-6 against MATLAB reference outputs.
* ``scipy.optimize.minimize(method='SLSQP')`` replaces MATLAB ``fmincon``
  with SQP algorithm for all constrained optimisations.
* Parameter augmentation: the returned ``parameters`` vector includes
  ``vech(S)`` prepended to the model parameters for VCV compatibility, matching
  the original MATLAB return convention (Ref: gogarch.m:253).

See Also
--------
mfe_toolbox.multivariate.gogarch_likelihood : GO-GARCH/O-GARCH log-likelihood.
mfe_toolbox.multivariate.ogarch_likelihood : Per-series OGARCH log-likelihood.
mfe_toolbox.univariate.tarch : TARCH/GJR-GARCH estimation for starting values.
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.multivariate.gogarch_likelihood import gogarch_likelihood
from mfe_toolbox.multivariate.ogarch_likelihood import ogarch_likelihood
from mfe_toolbox.univariate.tarch import tarch
from mfe_toolbox.utility.gradient_2sided import gradient_2sided
from mfe_toolbox.utility.covnw import covnw
from mfe_toolbox.utility.hessian_2sided_nrows import hessian_2sided_nrows
from mfe_toolbox.utility.vech import vech

__all__ = ['gogarch']


def gogarch(
    data: np.ndarray,
    p: int | np.ndarray = 1,
    q: int | np.ndarray = 1,
    gjr_type: int | np.ndarray | None = None,
    type_model: str | None = None,
    starting_vals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    OGARCH(p,q) and GOGARCH(p,q) multivariate volatility model estimation.

    Parameters
    ----------
    data : numpy.ndarray
        Either a T × K matrix of zero-mean residuals or a K × K × T array of
        covariance estimators (e.g. realised covariance).  If 2-D, outer
        products are computed internally.
    p : int or numpy.ndarray, optional
        Positive scalar integer or K-element vector of symmetric innovation
        lag orders (ARCH terms) per component.  Default is 1.
    q : int or numpy.ndarray, optional
        Non-negative scalar integer or K-element vector of conditional
        variance lag orders (GARCH terms) per component.  Default is 1.
    gjr_type : int, numpy.ndarray or None, optional
        Variance recursion type per component:

        * 1 — TARCH / AVGARCH (absolute values)
        * 2 — GJR-GARCH / standard GARCH (squares, default)

        Can be scalar (broadcast to all K series) or K-element vector.
    type_model : str or None, optional
        One of ``'gogarch'`` (default) or ``'ogarch'``.  Case-insensitive.
    starting_vals : numpy.ndarray or None, optional
        Starting values vector.  For GOGARCH:
        ``[phi(1)…phi(K(K-1)/2), vol(1)…vol(K)]``.  For OGARCH:
        ``[vol(1)…vol(K)]`` where ``vol(i)`` has ``p(i) + q(i)`` elements.
    options : dict or None, optional
        Options dict forwarded to ``scipy.optimize.minimize``.  If ``None``,
        defaults matching MATLAB ``optimset('fmincon')`` with SQP are used.

    Returns
    -------
    parameters : numpy.ndarray
        Estimated parameter vector prepended with ``vech(S)``
        (Ref: gogarch.m:253).  Layout: ``[vech(S), model_params]`` where
        ``model_params`` is ``[phi, vol]`` for GOGARCH or ``[vol]`` for OGARCH.
    ll : float
        Log-likelihood at the optimum (positive, i.e. negated from the
        minimiser's objective).
    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices.
    VCV : numpy.ndarray
        Square matrix of robust parameter covariances
        (``A^{-1} B A^{-1} / T``).
    scores : numpy.ndarray
        T × v matrix of individual scores for inference, where
        ``v = K(K+1)/2 + len(model_params)``.

    Raises
    ------
    ValueError
        For invalid inputs: wrong data shape, invalid model type, incorrect
        starting value length or range.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((500, 3))
    >>> params, ll, Ht, VCV, scores = gogarch(data, 1, 1, type_model='ogarch')
    """
    # ==================================================================
    # Phase 1: Input Validation
    # Ref: gogarch.m:52-138
    # ==================================================================

    # --- Convert data to numpy array ---
    data = np.asarray(data, dtype=np.float64)

    # --- Handle 3D (K×K×T) vs 2D (T×K) data ---
    # Ref: gogarch.m:73-82
    if data.ndim == 3:
        k = data.shape[0]
        T = data.shape[2]
    elif data.ndim == 2:
        T, k = data.shape
        # Ref: gogarch.m:77-81 — Compute outer products for 2D input
        temp = np.zeros((k, k, T), dtype=np.float64)
        for t in range(T):
            # Ref: gogarch.m:79 — temp(:,:,t) = data(t,:)'*data(t,:)
            row = data[t, :].reshape(-1, 1)
            temp[:, :, t] = row @ row.T
        data = temp
    else:
        raise ValueError(
            "DATA must be a T by K matrix or a K by K by T array."
        )

    # --- Broadcast scalar p to K-element vector ---
    # Ref: gogarch.m:84-86
    p = np.asarray(p, dtype=np.int64)
    if p.ndim == 0:
        p = np.ones(k, dtype=np.int64) * int(p)
    p = p.ravel().astype(np.int64)

    # --- Broadcast scalar q to K-element vector ---
    # Ref: gogarch.m:87-89
    q = np.asarray(q, dtype=np.int64)
    if q.ndim == 0:
        q = np.ones(k, dtype=np.int64) * int(q)
    q = q.ravel().astype(np.int64)

    # --- Default and broadcast gjr_type ---
    # Ref: gogarch.m:91-96
    if gjr_type is None:
        gjr_type = 2
    gjr_type = np.asarray(gjr_type, dtype=np.int64)
    if gjr_type.ndim == 0:
        gjr_type = np.ones(k, dtype=np.int64) * int(gjr_type)
    gjr_type = gjr_type.ravel().astype(np.int64)

    # --- Validate and set model type ---
    # Ref: gogarch.m:98-109
    if type_model is None:
        type_model = 'gogarch'
    type_model = type_model.lower()
    if type_model not in ('gogarch', 'ogarch'):
        raise ValueError("TYPE must be either 'GoGARCH' or 'OGARCH'.")
    is_gogarch = (type_model == 'gogarch')

    # --- Validate starting values if provided ---
    # Ref: gogarch.m:112-125
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        expected_count = int(np.sum(p) + np.sum(q))
        if is_gogarch:
            expected_count += k * (k - 1) // 2
        if len(starting_vals) != expected_count:
            raise ValueError(
                "STARTINGVALS does not have the correct number of parameters. "
                f"Expected {expected_count}, got {len(starting_vals)}."
            )
        if is_gogarch:
            n_phi = k * (k - 1) // 2
            # Ref: gogarch.m:121-123 — phi angles must be in [0, pi]
            phi_vals = starting_vals[:n_phi]
            if np.any(phi_vals > np.pi) or np.any(phi_vals < 0):
                raise ValueError(
                    "STARTINGVALS 1 to K*(K-1)/2 must be between 0 and "
                    f"{np.pi:.6f}."
                )

    # --- Default optimisation options ---
    # Ref: gogarch.m:127-138 — Replace optimset('fmincon') with scipy options
    # MATLAB: options.Algorithm = 'sqp', options.Display = 'iter'
    if options is None:
        options = {
            'maxiter': 1000,
            'ftol': 1e-10,
            'disp': False,
        }

    # ==================================================================
    # Phase 2: Preliminary Estimation
    # Ref: gogarch.m:141-175
    # ==================================================================

    # --- Compute mean covariance ---
    # Ref: gogarch.m:142 — S = mean(data,3)
    S = np.mean(data, axis=2)

    # --- Eigendecomposition of mean covariance ---
    # Ref: gogarch.m:143 — [P,L] = eig(S)
    # CRITICAL: MATLAB eig() returns [eigenvectors_columns, eigenvalues_diag].
    # numpy eigh() returns (eigenvalues_array, eigenvectors_columns) for
    # symmetric matrices.  Note the different return order.
    eigenvalues, eigvecs = np.linalg.eigh(S)

    # Ref: gogarch.m:144 — P = P' (transpose so rows are eigenvectors)
    P_mat = eigvecs.T

    # Ref: gogarch.m:143 — L is a diagonal eigenvalue matrix
    L_mat = np.diag(eigenvalues)

    # --- Compute whitening matrix Zinv ---
    # Ref: gogarch.m:157 — Zinv = L^(-0.5)*P'
    # L^(-0.5) is diagonal with 1/sqrt(eigenvalue) on diagonal.
    # P' after P=P' is actually eigvecs (the original column-eigenvector matrix).
    # Guard against very small or negative eigenvalues from numerical noise.
    eig_safe = np.maximum(np.abs(eigenvalues), 1e-300)
    L_inv_sqrt = np.diag(1.0 / np.sqrt(eig_safe))
    # Ref: Zinv = L^(-0.5) * P' where P has rows as eigenvectors
    # P_mat.T restores the original column-eigenvector matrix
    Zinv = L_inv_sqrt @ P_mat.T

    # --- Compute whitened (standardised) data ---
    # Ref: gogarch.m:158-161
    std_data = np.zeros((k, k, T), dtype=np.float64)
    for t in range(T):
        # Ref: gogarch.m:160 — stdData(:,:,t) = Zinv*data(:,:,t)*Zinv'
        std_data[:, :, t] = Zinv @ data[:, :, t] @ Zinv.T

    # --- Starting value generation ---
    # Ref: gogarch.m:148-175
    if starting_vals is None:
        # Ref: gogarch.m:149-154 — Starting optimisation options for tarch
        starting_options = {
            'ftol': 1e-5,
            'gtol': 1e-5,
            'disp': False,
            'maxiter': 400 * (int(np.max(p)) + int(np.max(q))),
        }

        vol_params_list = []
        V = np.zeros((T, k), dtype=np.float64)

        for i in range(k):
            # Ref: gogarch.m:165-167
            p_i = int(p[i])
            q_i = int(q[i])

            # Ref: gogarch.m:165 — volData = sqrt(squeeze(stdData(i,i,:)))
            vol_data = np.sqrt(np.maximum(std_data[i, i, :], 0.0))

            # Ref: gogarch.m:166 — tarch(volData,p(i),0,q(i),[],gjrType(i),[],startingOptions)
            # MATLAB tarch signature: tarch(eps, p, o, q, error_type, tarch_type, sv, opts)
            # Python tarch signature: tarch(eps, p, o, q, tarch_type, error_type, sv, opts)
            # Note: parameter order differs — using keyword args for clarity.
            # The gjrType(i) maps to tarch_type (1=AVGARCH, 2=GARCH).
            try:
                temp_params, _, ht_i, *_ = tarch(
                    vol_data,
                    p_i,
                    0,
                    q_i,
                    tarch_type=int(gjr_type[i]),
                    error_type='NORMAL',
                    startingvals=None,
                    options=starting_options,
                )
                V[:, i] = ht_i
                # Ref: gogarch.m:167 — volParams{i} = temp(2:(1+p(i)+q(i)))
                # MATLAB 1-indexed: temp(2) to temp(1+p_i+q_i)
                # Python 0-indexed: temp[1] to temp[1+p_i+q_i-1] = temp[1:1+p_i+q_i]
                vol_params_list.append(temp_params[1:1 + p_i + q_i])
            except Exception:
                # Fallback: equal-weight parameters summing to 0.9
                fallback = np.ones(p_i + q_i, dtype=np.float64) * (0.9 / (p_i + q_i))
                vol_params_list.append(fallback)
                warnings.warn(
                    f"TARCH starting value fit failed for series {i}; "
                    "using fallback values.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Ref: gogarch.m:169-174 — Assemble starting values
        if is_gogarch:
            # Ref: gogarch.m:170 — startingVals = zeros(1,k*(k-1)/2)+.0001
            starting_vals = np.zeros(k * (k - 1) // 2, dtype=np.float64) + 0.0001
        else:
            starting_vals = np.array([], dtype=np.float64)

        # Ref: gogarch.m:172-174 — Append vol params for each series
        for i in range(k):
            starting_vals = np.concatenate([starting_vals, vol_params_list[i]])

    # ==================================================================
    # Phase 3: OGARCH Per-Series Optimisation
    # Ref: gogarch.m:178-209
    # ==================================================================

    # --- Exponential decay weights for backcast ---
    # Ref: gogarch.m:179 — w = .06*.94.^(0:sqrt(T))
    # MATLAB 0:sqrt(T) produces integers 0 to floor(sqrt(T))
    w_len = int(np.sqrt(T)) + 1
    w = 0.06 * 0.94 ** np.arange(w_len, dtype=np.float64)
    # Ref: gogarch.m:180 — w = w/sum(w)
    w = w / np.sum(w)

    # --- Determine parameter offset ---
    # Ref: gogarch.m:181-185
    if is_gogarch:
        offset = k * (k - 1) // 2
    else:
        offset = 0

    parameters = starting_vals.copy()

    # --- Per-series OGARCH options ---
    # Ref: gogarch.m:187-193
    ogarch_options = {
        'maxiter': 1000,
        'ftol': 1e-10,
        'disp': False,
    }

    # --- Per-series volatility optimisation loop ---
    # Ref: gogarch.m:194-209
    for i in range(k):
        p_i = int(p[i])
        q_i = int(q[i])

        # Ref: gogarch.m:206 — volData = sqrt(squeeze(stdData(i,i,:)))
        # NOTE: In the MATLAB code, volData is used for backCast BEFORE it's
        # updated on line 206. For the first iteration, volData comes from
        # the last iteration of the starting values loop. We replicate this
        # by computing volData here (beginning of loop body) for current
        # series, matching the MATLAB line 206 placement.
        vol_data = np.sqrt(np.maximum(std_data[i, i, :], 0.0))

        # Ref: gogarch.m:195-199 — Compute backcast
        # MATLAB uses `if gjrType==1` which for a vector checks ALL elements.
        # Python equivalent: np.all(gjr_type == 1)
        actual_w_len = min(w_len, T)
        if np.all(gjr_type == 1):
            # Ref: gogarch.m:196 — backCast = w*abs(volData(1:length(w)))
            back_cast = float(np.dot(w[:actual_w_len], np.abs(vol_data[:actual_w_len])))
        else:
            # Ref: gogarch.m:198 — backCast = w*(volData(1:length(w))).^2
            back_cast = float(np.dot(w[:actual_w_len], vol_data[:actual_w_len] ** 2))

        count = p_i + q_i

        # Ref: gogarch.m:201-204 — Bounds and linear constraint
        bounds_i = [(0.0, 1.0)] * count
        # Ref: gogarch.m:203-204 — A*x <= b  →  sum(x) <= 1
        constraints_i = {
            'type': 'ineq',
            'fun': lambda x: 1.0 - np.sum(x),
        }

        # Ref: gogarch.m:205 — volStart = parameters(offset + (1:count))
        # MATLAB 1-indexed: offset + 1 to offset + count
        # Python 0-indexed: offset to offset + count
        vol_start = parameters[offset:offset + count].copy()

        # Ref: gogarch.m:207 — fmincon(@ogarch_likelihood,volStart,A,b,...,volData,p(i),q(i),gjrType(i),backCast)
        try:
            result_i = minimize(
                lambda x, d, pi, qi, gi, bc: ogarch_likelihood(x, d, pi, qi, gi, bc)[0],
                vol_start,
                args=(vol_data, p_i, q_i, int(gjr_type[i]), back_cast),
                method='SLSQP',
                bounds=bounds_i,
                constraints=constraints_i,
                options=ogarch_options,
            )
            parameters[offset:offset + count] = result_i.x
        except Exception as exc:
            warnings.warn(
                f"OGARCH per-series optimisation failed for series {i}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

        offset += count

    # ==================================================================
    # Phase 4: GOGARCH Joint Optimisation
    # Ref: gogarch.m:212-233
    # ==================================================================

    if is_gogarch:
        # Ref: gogarch.m:214 — v = length(startingVals)
        v_params = len(parameters)

        # Ref: gogarch.m:215-218 — Bounds setup
        bounds_all = []
        n_phi = k * (k - 1) // 2
        for j in range(n_phi):
            # Ref: gogarch.m:217-218 — phi bounds [1e-6, 0.99998*pi]
            bounds_all.append((1e-6, 0.99998 * np.pi))
        for j in range(n_phi, v_params):
            # Ref: gogarch.m:215-216 — vol bounds [0, 1]
            bounds_all.append((0.0, 1.0))

        # Ref: gogarch.m:219-227 — Per-series linear constraints: sum(vol_i) <= 0.99998
        constraints_all = []
        c_offset = n_phi
        for i in range(k):
            p_i = int(p[i])
            q_i = int(q[i])
            count_i = p_i + q_i
            idx_start = c_offset
            idx_end = c_offset + count_i
            # Capture loop variables by default argument binding
            # Ref: gogarch.m:224-226 — A(i, offset+(1:count)) = ones(1,count); b=0.99998
            constraints_all.append({
                'type': 'ineq',
                'fun': lambda x, s=idx_start, e=idx_end: 0.99998 - np.sum(x[s:e]),
            })
            c_offset += count_i

        # Ref: gogarch.m:228 — fmincon(@gogarch_likelihood,startingVals,A,b,...,data,p,q,gjrType,P,L,false,false)
        try:
            result_joint = minimize(
                lambda x, *a: gogarch_likelihood(x, *a)[0],
                parameters,
                args=(data, p, q, gjr_type, P_mat, L_mat, False, False),
                method='SLSQP',
                bounds=bounds_all,
                constraints=constraints_all,
                options=options,
            )
            parameters = result_joint.x.copy()
        except Exception as exc:
            warnings.warn(
                f"GOGARCH joint optimisation failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Ref: gogarch.m:229 — [ll,~,Ht] = gogarch_likelihood(parameters,...,false,false)
        ll, _, Ht = gogarch_likelihood(
            parameters, data, p, q, gjr_type, P_mat, L_mat, False, False
        )
    else:
        # Ref: gogarch.m:231 — [ll,~,Ht] = gogarch_likelihood(parameters,...,true,false)
        ll, _, Ht = gogarch_likelihood(
            parameters, data, p, q, gjr_type, P_mat, L_mat, True, False
        )

    # Ref: gogarch.m:233 — ll = -ll (negate: objective was neg-LL, convert to LL)
    ll = -float(ll)

    # ==================================================================
    # Phase 5: Inference / VCV Computation
    # Ref: gogarch.m:236-257
    # ==================================================================

    # --- Scores matrix construction ---
    # Ref: gogarch.m:237-238
    k2 = k * (k + 1) // 2
    m = len(parameters)
    v = k2 + m

    scores = np.zeros((T, v), dtype=np.float64)

    # Ref: gogarch.m:240-246 — Fill first k2 columns with vech-ordered
    # covariance moment conditions: data(i,j,:) - S(i,j)
    # MATLAB: for j=1:k, for i=j:k (column-major lower triangle = vech order)
    count = 0
    for j in range(k):
        for i in range(j, k):
            # Ref: gogarch.m:243 — scores(:,count) = squeeze(data(i,j,:) - S(i,j))
            scores[:, count] = data[i, j, :] - S[i, j]
            count += 1

    # Ref: gogarch.m:247 — [~,s] = gradient_2sided(@gogarch_likelihood,parameters',...)
    # gradient_2sided with compute_scores=True expects f to return (scalar, T-array).
    # gogarch_likelihood returns (ll, lls, Ht) — need a wrapper returning (ll, lls).
    def _ll_with_scores(params, *args):
        """Wrapper: returns (ll, lls) for gradient_2sided score computation."""
        ll_val, lls_val, _ = gogarch_likelihood(params, *args)
        return ll_val, lls_val

    _, s = gradient_2sided(
        _ll_with_scores,
        parameters,
        data, p, q, gjr_type, P_mat, L_mat, not is_gogarch, False,
        compute_scores=True,
    )

    # Ref: gogarch.m:248 — scores(:,k2+1:v) = s
    scores[:, k2:v] = s

    # --- B matrix (Newey-West HAC) ---
    # Ref: gogarch.m:249 — B = covnw(scores,ceil(1.2*T^(1/3)))
    nw_lags = int(np.ceil(1.2 * T ** (1.0 / 3.0)))
    B = covnw(scores, nw_lags)

    # --- A matrix construction ---
    # Ref: gogarch.m:250-255
    A_mat = np.zeros((v, v), dtype=np.float64)

    # Ref: gogarch.m:251 — A(1:k2,1:k2) = -eye(k2)
    A_mat[:k2, :k2] = -np.eye(k2)

    # Ref: gogarch.m:252-253 — Augment parameters with vech(S) for inference Hessian
    # hessian_2sided_nrows expects f returning scalar.
    def _ll_scalar(params, *args):
        """Wrapper: returns scalar ll for hessian_2sided_nrows."""
        ll_val, _, _ = gogarch_likelihood(params, *args)
        return ll_val

    # Ref: gogarch.m:253 — parameters = [vech(S)' parameters]'
    vech_S = vech(S).ravel()  # vech returns (N,1) column; ravel to 1-D
    params_augmented = np.concatenate([vech_S, parameters])

    # Ref: gogarch.m:254 — temp = hessian_2sided_nrows(@gogarch_likelihood,parameters,m,...)
    # Note: is_inference=True (last arg) triggers the augmented parameter path
    # in gogarch_likelihood where the first k2 params encode vech(S).
    temp = hessian_2sided_nrows(
        _ll_scalar,
        params_augmented,
        m,
        data, p, q, gjr_type, P_mat, L_mat, not is_gogarch, True,
    )

    # Ref: gogarch.m:255 — A((k2+1):v,:) = temp/T
    A_mat[k2:v, :] = temp / T

    # --- VCV computation ---
    # Ref: gogarch.m:256 — Ainv = A\eye(v)
    try:
        Ainv = np.linalg.solve(A_mat, np.eye(v))
    except np.linalg.LinAlgError:
        warnings.warn(
            "A matrix is singular; VCV may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
        Ainv = np.linalg.pinv(A_mat)

    # Ref: gogarch.m:257 — VCV = Ainv*B*Ainv'/T
    VCV = Ainv @ B @ Ainv.T / T

    # --- Return augmented parameters (matching MATLAB convention) ---
    # Ref: gogarch.m:253 — parameters was augmented with vech(S)
    # The MATLAB code returns this augmented vector as the output.
    parameters = params_augmented

    return parameters, ll, Ht, VCV, scores
