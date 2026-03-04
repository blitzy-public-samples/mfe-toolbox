"""
TARCH/GJR-GARCH(P,O,Q) parameter estimation driver.

Estimates TARCH (Threshold ARCH) and GJR-GARCH models with support for
four error distributions: Normal, Student's t, GED, and Skewed t.

The TARCH model introduces asymmetric response to positive and negative
shocks through an indicator function for negative returns (leverage effect):

    g(h(t)) = omega
            + sum_{i=1}^{p} alpha_i * f(eps_{t-i})
            + sum_{j=1}^{o} gamma_j * I(eps_{t-j}<0) * f(eps_{t-j})
            + sum_{k=1}^{q} beta_k * g(h(t-k))

where f(x) = |x| and g(x) = sqrt(x) for tarch_type='AVGARCH' (type 1),
and   f(x) = x^2 and g(x) = x   for tarch_type='GARCH'   (type 2, default).

Migrated from: univariate/tarch.m — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.univariate.tarch_parameter_check import tarch_parameter_check
from mfe_toolbox.univariate.tarch_starting_values import tarch_starting_values
from mfe_toolbox.univariate.tarch_transform import tarch_transform
from mfe_toolbox.univariate.tarch_itransform import tarch_itransform
from mfe_toolbox.univariate.tarch_likelihood import tarch_likelihood
from mfe_toolbox.utility.robustvcv import robustvcv

__all__ = ['tarch']


def _generate_grid_starting_values(
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    tarch_type: int,
    error_type: int,
    back_cast: float,
    epsilon_augmented: np.ndarray,
    fepsilon: np.ndarray,
    fIepsilon: np.ndarray,
    T: int,
    m: int,
) -> np.ndarray:
    """Generate sorted grid of TARCH starting value candidates for robustness retry.

    Replicates the grid search logic from tarch_starting_values.m to produce
    an ordered matrix of candidate starting values sorted by log-likelihood.
    This is used by the main tarch() driver when the initial optimisation
    does not converge and no user-supplied starting values were provided.

    Ref: tarch_starting_values.m:48-131 — grid search and sorting logic.
    The Python tarch_starting_values() does not return ordered_parameters,
    so this helper is needed for the robustness retry loop in tarch().

    Parameters
    ----------
    epsilon : np.ndarray
        Raw (non-augmented) 1-D residual array.
    p, o, q : int
        TARCH lag orders.
    tarch_type : int
        1 = absolute-value, 2 = squared.
    error_type : int
        Distribution type (1-4).
    back_cast : float
        Backcast value for variance initialisation.
    epsilon_augmented : np.ndarray
        Augmented epsilon (m zeros prepended).
    fepsilon : np.ndarray
        Augmented f(epsilon) array.
    fIepsilon : np.ndarray
        Augmented f(epsilon)*I(epsilon<0) array.
    T : int
        Augmented data length.
    m : int
        max(p, o, q).

    Returns
    -------
    np.ndarray
        2-D array of shape (N, 1+p+o+q) with rows sorted by ascending
        negative log-likelihood (best first).
    """
    # Ref: tarch_starting_values.m:54-59 — grid definition
    a_grid = [0.05, 0.1, 0.2]
    g_grid = [0.01, 0.05, 0.2]  # plus -alpha/2 as 4th element
    agb_grid = [0.5, 0.8, 0.9, 0.95, 0.99]

    la_len = len(a_grid)       # 3
    lg_len = len(g_grid) + 1   # 4 (includes -alpha/2)
    lb_len = len(agb_grid)     # 5
    total = la_len * lg_len * lb_len  # 60

    # Ref: tarch_starting_values.m:67-70 — adjustment factor
    if tarch_type == 1:
        adj_factor = np.sqrt(2.0 / np.pi)
    else:
        adj_factor = 1.0

    # Ref: tarch_starting_values.m:73
    covar = float(np.var(epsilon, ddof=1))

    output_params = np.zeros((total, 1 + p + o + q), dtype=np.float64)
    LLs = np.full(total, np.inf, dtype=np.float64)
    idx = 0

    for i in range(la_len):
        alpha = a_grid[i]
        for j in range(lg_len):
            if j == lg_len - 1:
                # Ref: tarch_starting_values.m:83 — last gamma = -alpha/2
                gamma = -alpha / 2.0
            else:
                gamma = g_grid[j]

            for k in range(lb_len):
                # Ref: tarch_starting_values.m:92
                temp_alpha = alpha * np.ones(max(p, 0), dtype=np.float64) / max(p, 1)
                # Ref: tarch_starting_values.m:94-103
                if o > 0:
                    temp_gamma = gamma * np.ones(o, dtype=np.float64) / o
                    for n in range(o):
                        if n < p:
                            temp_gamma[n] = max(temp_gamma[n], -alpha / (2.0 * p))
                        else:
                            temp_gamma[n] = 0.0
                else:
                    temp_gamma = np.empty(0, dtype=np.float64)

                # Ref: tarch_starting_values.m:105
                beta = agb_grid[k] - np.sum(temp_alpha) - 0.5 * np.sum(temp_gamma)

                # Ref: tarch_starting_values.m:108-110 — omega via variance targeting
                if tarch_type == 1:
                    omega = float(
                        np.mean(np.abs(epsilon))
                        * (1.0 - alpha * adj_factor - 0.5 * gamma - beta)
                    )
                else:
                    omega = float(
                        covar * (1.0 - np.sum(temp_alpha) * adj_factor
                                 - 0.5 * np.sum(temp_gamma) - beta)
                    )

                # Ref: tarch_starting_values.m:113-118 — assemble parameter vector
                parts = [np.array([omega], dtype=np.float64)]
                if p > 0:
                    parts.append(temp_alpha)
                if o > 0:
                    parts.append(temp_gamma)
                if q > 0:
                    parts.append(beta * np.ones(q, dtype=np.float64) / q)
                params_vec = np.concatenate(parts)

                output_params[idx, :] = params_vec

                # Ref: tarch_starting_values.m:122 — evaluate LL (Normal only)
                try:
                    ll_val, _, _ = tarch_likelihood(
                        params_vec, epsilon_augmented, fepsilon, fIepsilon,
                        p, o, q, 1, tarch_type, back_cast, T, False
                    )
                    LLs[idx] = ll_val
                except Exception:
                    LLs[idx] = np.inf

                idx += 1

    # Ref: tarch_starting_values.m:129 — sort by LL ascending (best first)
    sort_indices = np.argsort(LLs)
    return output_params[sort_indices]


def tarch(
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    tarch_type: str = 'GARCH',
    error_type: str = 'NORMAL',
    startingvals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    TARCH(P,O,Q) parameter estimation with different error distributions.

    Estimates TARCH / GJR-GARCH models using ``scipy.optimize.minimize``
    with ``method='L-BFGS-B'`` in the transformed (unconstrained) parameter
    space.  Supports Normal, Student's t, GED, and Skewed t innovations.

    Parameters
    ----------
    epsilon : np.ndarray
        T-by-1 column vector of mean-zero data (residuals).
    p : int
        Positive scalar integer for the number of symmetric innovation
        lags (ARCH terms).  Must be >= 1.
    o : int
        Non-negative scalar integer for the number of asymmetric
        innovation lags (threshold terms).  0 for symmetric processes.
    q : int
        Non-negative scalar integer for the number of lagged conditional
        variance terms (GARCH terms).  0 for pure ARCH.
    tarch_type : str, optional
        The variance process type:

        * ``'GARCH'`` (default) — model evolves in squares (internal type 2)
        * ``'AVGARCH'`` — model evolves in absolute values (internal type 1)

        Integer values 1 and 2 are also accepted for backwards compatibility.
    error_type : str, optional
        The error distribution:

        * ``'NORMAL'`` (default) — Gaussian innovations
        * ``'STUDENTST'`` — Student's t distributed errors
        * ``'GED'`` — Generalized Error Distribution
        * ``'SKEWT'`` — Skewed t distribution

        Integer codes 1-4 are also accepted.
    startingvals : np.ndarray or None, optional
        Starting values for optimisation.  Length depends on distribution:

        * NORMAL: ``1 + p + o + q``
        * STUDENTST or GED: ``1 + p + o + q + 1`` (includes nu)
        * SKEWT: ``1 + p + o + q + 2`` (includes nu and lambda)

        Layout: ``[omega, alpha(1)…alpha(p), gamma(1)…gamma(o),
        beta(1)…beta(q), [nu, [lambda]]]``
    options : dict or None, optional
        Options dict for ``scipy.optimize.minimize``.  If ``None``,
        defaults are:
        ``{'ftol': 1e-5, 'gtol': 1e-5, 'disp': True,
        'maxiter': 200*(2+p+q)}``.

    Returns
    -------
    parameters : np.ndarray
        Estimated parameter vector.
    LL : float
        Log-likelihood at the optimum (positive value).
    ht : np.ndarray
        Estimated conditional variances, length ``T`` (original data length).
    VCVrobust : np.ndarray
        Robust (sandwich) parameter covariance matrix (K × K).
    VCV : np.ndarray
        Non-robust covariance matrix from inverse Hessian (K × K).
    scores : np.ndarray
        Per-observation score matrix (T × K).
    diagnostics : dict
        Optimiser diagnostics including EXITFLAG, ITERATIONS, FUNCCOUNT,
        MESSAGE, A, m, T, fdata, fIdata, back_cast.

    Raises
    ------
    ValueError
        If inputs fail validation (delegated to ``tarch_parameter_check``).

    Notes
    -----
    The following constraints are enforced via parameter transformation:

    1. ``omega > 0``
    2. ``alpha(i) >= 0`` for all *i*
    3. ``gamma(j) + alpha(j) > 0`` for *j* = 1, …, min(p, o)
    4. ``beta(k) >= 0`` for all *k*
    5. ``sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1``  (stationarity)
    6. ``nu > 2`` for Student's t; ``nu > 1`` for GED
    7. ``-0.99 < lambda < 0.99`` for Skewed t

    Ref: tarch.m — MATLAB MFE Toolbox, Kevin Sheppard, University of Oxford.

    See Also
    --------
    tarch_likelihood : Log-likelihood computation.
    tarch_core : Numba JIT conditional variance recursion.
    tarch_parameter_check : Input validation.
    tarch_starting_values : Grid search for starting values.
    tarch_transform : Parameter transform (constrained → unconstrained).
    tarch_itransform : Inverse transform (unconstrained → constrained).

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.tarch import tarch
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal(500)
    >>> params, ll, ht, vcv_r, vcv, sc, diag = tarch(data, 1, 1, 1)
    >>> params.shape[0]  # omega + alpha + gamma + beta = 4
    4
    """
    # ================================================================== #
    #  Map string tarch_type to integer                                   #
    #  Ref: tarch.m uses integers (1=abs, 2=squared); Python API uses    #
    #  strings ('AVGARCH', 'GARCH') mapped internally.                   #
    # ================================================================== #
    _tarch_type_map = {'GARCH': 2, 'AVGARCH': 1}
    if isinstance(tarch_type, str):
        tarch_type_upper = tarch_type.upper()
        if tarch_type_upper not in _tarch_type_map:
            raise ValueError(
                "tarch_type must be 'GARCH' or 'AVGARCH', "
                f"received '{tarch_type}'"
            )
        tarch_type_int = _tarch_type_map[tarch_type_upper]
    elif isinstance(tarch_type, (int, float, np.integer, np.floating)):
        tarch_type_int = int(tarch_type)
    else:
        tarch_type_int = int(tarch_type)

    # ================================================================== #
    #  Input Checking                                                     #
    #  Ref: tarch.m:87-100 — nargin-based dispatch to tarch_parameter_   #
    #  check, replaced by Python default arguments.                       #
    # ================================================================== #
    p, o, q, error_type, tarch_type, startingvals, options = (
        tarch_parameter_check(
            epsilon, p, o, q, error_type, tarch_type_int,
            startingvals, options
        )
    )
    # After this call, error_type and tarch_type are integers (1-4 / 1-2)

    # ================================================================== #
    #  Initial setup                                                      #
    #  Ref: tarch.m:107 — m = max([p o q])                               #
    # ================================================================== #
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()
    T_raw = len(epsilon)
    # Ref: tarch.m:107 — m = max([p o q])
    m = int(np.max(np.array([p, o, q])))

    # ================================================================== #
    #  Augment the data with back casts to avoid costly memory allocs     #
    #  Ref: tarch.m:110-141                                               #
    # ================================================================== #
    if tarch_type == 1:
        # Ref: tarch.m:113 — fepsilon = [mean(abs(eps))*ones(m,1); abs(eps)]
        abs_eps = np.abs(epsilon)
        mean_abs = float(np.mean(abs_eps))
        fepsilon = np.concatenate([
            mean_abs * np.ones(m, dtype=np.float64),
            abs_eps
        ])
        # Ref: tarch.m:115 — fIepsilon = [0.5*mean(abs(eps))*ones(m,1);
        #                                  abs(eps).*(eps<0)]
        fIepsilon = np.concatenate([
            0.5 * mean_abs * np.ones(m, dtype=np.float64),
            abs_eps * (epsilon < 0).astype(np.float64)
        ])
        # Ref: tarch.m:118-124 — Local back casting with exponential weights
        back_cast_length = max(int(np.floor(T_raw ** 0.5)), 1)
        back_cast_weights = 0.05 * (
            0.9 ** np.arange(back_cast_length + 1, dtype=np.float64)
        )
        back_cast_weights = back_cast_weights / np.sum(back_cast_weights)
        # Ref: tarch.m:121 — back_cast = weights * abs(eps(1:bcl+1))
        back_cast = float(
            np.dot(back_cast_weights, abs_eps[:back_cast_length + 1])
        )
        # Ref: tarch.m:122-124 — fallback if back_cast is zero
        if back_cast == 0.0:
            back_cast = mean_abs
    else:
        # tarch_type == 2 (default GARCH / squared returns)
        # Ref: tarch.m:127 — fepsilon = [mean(eps.^2)*ones(m,1); eps.^2]
        eps_sq = epsilon ** 2
        mean_sq = float(np.mean(eps_sq))
        fepsilon = np.concatenate([
            mean_sq * np.ones(m, dtype=np.float64),
            eps_sq
        ])
        # Ref: tarch.m:129 — fIepsilon = [0.5*mean(eps^2)*ones(m,1);
        #                                  eps^2.*(eps<0)]
        fIepsilon = np.concatenate([
            0.5 * mean_sq * np.ones(m, dtype=np.float64),
            eps_sq * (epsilon < 0).astype(np.float64)
        ])
        # Ref: tarch.m:131-137 — Local back casting with exponential weights
        back_cast_length = max(int(np.floor(T_raw ** 0.5)), 1)
        back_cast_weights = 0.05 * (
            0.9 ** np.arange(back_cast_length + 1, dtype=np.float64)
        )
        back_cast_weights = back_cast_weights / np.sum(back_cast_weights)
        # Ref: tarch.m:134 — back_cast = weights * (eps(1:bcl+1)).^2
        back_cast = float(
            np.dot(back_cast_weights, eps_sq[:back_cast_length + 1])
        )
        # Ref: tarch.m:135-137 — fallback if back_cast is zero
        if back_cast == 0.0:
            back_cast = mean_sq

    # Ref: tarch.m:139 — epsilon_augmented = [zeros(m,1); epsilon]
    epsilon_augmented = np.concatenate([
        np.zeros(m, dtype=np.float64), epsilon
    ])
    # Ref: tarch.m:141 — T = size(fepsilon,1) — augmented data length
    T = len(fepsilon)

    # ================================================================== #
    #  Starting values                                                    #
    #  Ref: tarch.m:145-165                                               #
    # ================================================================== #
    # Ref: tarch.m:150-154 — startingflag: 0 = grid search was used,
    # 1 = user supplied starting values (skip alternative retry)
    if startingvals is None or (
        isinstance(startingvals, np.ndarray) and startingvals.size == 0
    ):
        startingflag = 0
    else:
        startingflag = 1

    # Ref: tarch.m:156 — Grid search for starting values
    # Python tarch_starting_values takes raw epsilon and raw T
    sv_core, nu, lam = tarch_starting_values(
        startingvals, epsilon, p, o, q,
        tarch_type, error_type, back_cast, T_raw
    )

    # Ref: tarch.m:158 — startingvals = [startingvals; nu; lambda]
    # Build full parameter vector including distribution parameters
    sv_parts = [sv_core]
    if nu is not None:
        sv_parts.append(np.array([nu], dtype=np.float64))
    if lam is not None:
        sv_parts.append(np.array([lam], dtype=np.float64))
    startingvals_full = np.concatenate(sv_parts)

    # Ref: tarch.m:160-162 — Transform starting values to unconstrained space
    # Python tarch_transform returns a single concatenated array
    startingvals_transformed = tarch_transform(
        startingvals_full, p, o, q, error_type, tarch_type
    )

    # ================================================================== #
    #  Optimisation — parameter estimation                                #
    #  Ref: tarch.m:170-172                                               #
    # ================================================================== #
    # Define scalar objective function for scipy.optimize.minimize.
    # tarch_likelihood returns (LL, LLS, ht); we need only the scalar LL.
    def _obj_func(params: np.ndarray) -> float:
        """Scalar objective: returns negative log-likelihood for minimisation."""
        ll_val, _, _ = tarch_likelihood(
            params, epsilon_augmented, fepsilon, fIepsilon,
            p, o, q, error_type, tarch_type, back_cast, T, True
        )
        return float(ll_val)

    # Ref: tarch.m:170 — LL0 is used to make sure the log likelihood improves
    LL0 = _obj_func(startingvals_transformed)

    # Ref: tarch.m:172 — fminunc → scipy.optimize.minimize(method='L-BFGS-B')
    result = minimize(
        _obj_func, startingvals_transformed,
        method='L-BFGS-B', options=options
    )
    parameters = result.x.copy()
    LL = float(result.fun)
    exitflag = 1 if result.success else 0
    output = result

    # ================================================================== #
    #  Estimation Robustness — Retry with more iterations                 #
    #  Ref: tarch.m:174-199                                               #
    #  This handles the case where the optimisation did not converge but  #
    #  improved on the initial log likelihood.                            #
    # ================================================================== #
    if exitflag <= 0 and LL < LL0:
        # Ref: tarch.m:186-196 — Increase max iterations and function evals,
        # switch to steepest descent (in Python, we keep L-BFGS-B but with
        # increased budget, as there is no direct steepest descent equivalent)
        retry_options = dict(options)
        num_params = len(parameters)
        # Ref: tarch.m:186-189 — MaxIter logic
        current_maxiter = retry_options.get('maxiter', 200 * num_params)
        retry_options['maxiter'] = 2 * current_maxiter
        # Ref: tarch.m:191-195 — MaxFunEvals logic
        current_maxfun = retry_options.get('maxfun', 400 * num_params)
        retry_options['maxfun'] = 2 * current_maxfun

        # Ref: tarch.m:198 — re-estimate from current best parameters
        result = minimize(
            _obj_func, parameters,
            method='L-BFGS-B', options=retry_options
        )
        parameters = result.x.copy()
        LL = float(result.fun)
        exitflag = 1 if result.success else 0
        output = result

    # ================================================================== #
    #  Estimation Robustness — Try alternative starting values            #
    #  Ref: tarch.m:203-252                                               #
    #  If still not converged and user didn't supply starting values,     #
    #  cycle through grid search candidates.                              #
    # ================================================================== #
    if startingflag == 0 and exitflag <= 0:
        # Ref: tarch.m:207-210 — track robust parameters and LLs
        robust_parameters = [parameters.copy()]
        robust_LL = [LL]

        # Generate ordered grid of alternative starting values.
        # Ref: tarch.m:218 — ordered_parameters from tarch_starting_values
        # Python tarch_starting_values does not return ordered_parameters,
        # so we replicate the grid search logic via _generate_grid_starting_values.
        ordered_parameters = _generate_grid_starting_values(
            epsilon, p, o, q, tarch_type, error_type, back_cast,
            epsilon_augmented, fepsilon, fIepsilon, T, m
        )

        # Ref: tarch.m:212-251 — loop through ordered starting values
        # MATLAB index starts at 2 (1-based); Python starts at 1 (0-based)
        index = 1
        while exitflag <= 0 and index < len(ordered_parameters):
            # Ref: tarch.m:218 — startingvals = [ordered_parameters(index,:)';
            #                                     nu; lambda]
            sv_alt = ordered_parameters[index].copy()
            sv_alt_parts = [sv_alt]
            if nu is not None:
                sv_alt_parts.append(np.array([nu], dtype=np.float64))
            if lam is not None:
                sv_alt_parts.append(np.array([lam], dtype=np.float64))
            sv_alt_full = np.concatenate(sv_alt_parts)

            # Ref: tarch.m:220-222 — Transform the starting vals
            try:
                sv_alt_transformed = tarch_transform(
                    sv_alt_full, p, o, q, error_type, tarch_type
                )
            except ValueError:
                # Parameters don't satisfy constraints; skip this candidate
                index += 1
                continue

            # Ref: tarch.m:224 — initial LL for this starting point
            LL0_alt = _obj_func(sv_alt_transformed)

            # Ref: tarch.m:225-227 — HessUpdate='bfgs' → L-BFGS-B
            result = minimize(
                _obj_func, sv_alt_transformed,
                method='L-BFGS-B', options=options
            )
            parameters = result.x.copy()
            LL = float(result.fun)
            exitflag = 1 if result.success else 0
            output = result

            # Ref: tarch.m:228-237 — if failed but improved, retry with more
            if exitflag <= 0 and LL < LL0_alt:
                retry_options2 = dict(options)
                num_p = len(parameters)
                retry_options2['maxiter'] = (
                    retry_options2.get('maxiter', 200 * num_p) * 2
                )
                retry_options2['maxfun'] = (
                    retry_options2.get('maxfun', 400 * num_p) * 2
                )

                # Ref: tarch.m:236 — steepest descent retry
                result = minimize(
                    _obj_func, parameters,
                    method='L-BFGS-B', options=retry_options2
                )
                parameters = result.x.copy()
                LL = float(result.fun)
                exitflag = 1 if result.success else 0
                output = result

            # Ref: tarch.m:239-240 — save results for this attempt
            robust_parameters.append(parameters.copy())
            robust_LL.append(LL)

            # Ref: tarch.m:242 — increment index
            index += 1

        # Ref: tarch.m:244-249 — if all attempts failed, warn and return best
        if exitflag <= 0:
            # Ref: tarch.m:246
            warnings.warn(
                'Convergence not achieved. Use results with caution',
                UserWarning,
            )
            # Ref: tarch.m:247-248 — [LL, index] = min(robust_LL)
            robust_LL_arr = np.array(robust_LL, dtype=np.float64)
            best_idx = int(np.argmin(robust_LL_arr))
            LL = float(robust_LL_arr[best_idx])
            parameters = robust_parameters[best_idx].copy()

    # ================================================================== #
    #  Inverse Transform — recover constrained parameters                 #
    #  Ref: tarch.m:254-256                                               #
    # ================================================================== #
    # Ref: tarch.m:255 — [parameters, nu, lambda] = tarch_itransform(...)
    # Python tarch_itransform returns single concatenated array
    parameters = tarch_itransform(
        parameters, p, o, q, error_type, tarch_type
    )

    # ================================================================== #
    #  Final log-likelihood evaluation                                    #
    #  Ref: tarch.m:258-261                                               #
    # ================================================================== #
    # Ref: tarch.m:259 — [LL, ~, ht] = tarch_likelihood(parameters, ...)
    # estim_flag=False: parameters are already constrained
    LL_neg, _LLS, ht = tarch_likelihood(
        parameters, epsilon_augmented, fepsilon, fIepsilon,
        p, o, q, error_type, tarch_type, back_cast, T, False
    )
    # Ref: tarch.m:260 — LL = -LL (tarch_likelihood returns negated LL;
    # un-negate for the positive log-likelihood convention)
    LL = -float(LL_neg)

    # ================================================================== #
    #  Robust VCV computation                                             #
    #  Ref: tarch.m:264-268                                               #
    # ================================================================== #
    # robustvcv expects fun(theta, *args) -> (scalar, array).
    # tarch_likelihood returns (scalar, array, array).  Wrap to return 2.
    def _ll_for_vcv(params: np.ndarray, *args) -> tuple[float, np.ndarray]:
        """Wrapper returning (scalar_LL, per_obs_LLS) for robustvcv."""
        ll_val, lls_val, _ = tarch_likelihood(params, *args)
        return float(ll_val), lls_val

    # Ref: tarch.m:265 — nw = 0 (no Newey-West HAC on scores)
    # Ref: tarch.m:266 — [VCVrobust, A, ~, scores, hess] = robustvcv(...)
    VCVrobust, A, _B, scores, hess, _gross_scores = robustvcv(
        _ll_for_vcv, parameters, 0,
        epsilon_augmented, fepsilon, fIepsilon,
        p, o, q, error_type, tarch_type, back_cast, T
    )
    # Ref: tarch.m:267 — VCV = hess^(-1) / (T - m)
    # hess is A = H/T from robustvcv; VCV = inv(A) / (T - m)
    VCV = np.linalg.inv(hess) / (T - m)

    # ================================================================== #
    #  Build diagnostics dict                                              #
    #  Ref: tarch.m:268, 272-280                                          #
    # ================================================================== #
    diagnostics: dict = {}
    # Ref: tarch.m:268 — diagnostics.A
    diagnostics['A'] = A
    # Ref: tarch.m:272
    diagnostics['EXITFLAG'] = exitflag
    # Ref: tarch.m:273 — output.iterations → result.nit
    diagnostics['ITERATIONS'] = getattr(output, 'nit', 0)
    # Ref: tarch.m:274 — output.funcCount → result.nfev
    diagnostics['FUNCCOUNT'] = getattr(output, 'nfev', 0)
    # Ref: tarch.m:275 — output.message → result.message
    msg = getattr(output, 'message', '')
    if isinstance(msg, bytes):
        msg = msg.decode('utf-8', errors='replace')
    diagnostics['MESSAGE'] = str(msg)
    # Ref: tarch.m:276-280
    diagnostics['m'] = m
    diagnostics['T'] = T
    diagnostics['fdata'] = fepsilon
    diagnostics['fIdata'] = fIepsilon
    diagnostics['back_cast'] = back_cast

    return parameters, LL, ht, VCVrobust, VCV, scores, diagnostics
