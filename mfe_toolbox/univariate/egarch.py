"""
EGARCH(P,O,Q) parameter estimation driver.

Implements the full EGARCH(P,O,Q) estimation pipeline: input validation,
starting-value search, constrained optimization via
``scipy.optimize.minimize(method='SLSQP')`` with nonlinear stationarity
constraints, robust and non-robust variance-covariance computation, and
diagnostics assembly.

The conditional variance h(t) of an EGARCH(P,O,Q) process is modeled as::

    ln(h(t)) = omega
             + alpha(1)*(|e_{t-1}| - C) + ... + alpha(p)*(|e_{t-p}| - C)
             + gamma(1)*e_{t-1} + ... + gamma(o)*e_{t-o}
             + beta(1)*ln(h(t-1)) + ... + beta(q)*ln(h(t-q))

where ``ln`` is the natural logarithm, ``e_t = data_t / sqrt(h_t)`` are
standardized residuals, and ``C = sqrt(2/pi)`` is the expected value of
``|Z|`` for standard-normal Z.

Supported error distributions:
    * NORMAL — Gaussian innovations
    * STUDENTST — Standardized Student's t
    * GED — Generalized Error Distribution
    * SKEWT — Hansen's Skewed Student's t

Migrated from: ``univariate/egarch.m`` — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.univariate.egarch_parameter_check import egarch_parameter_check
from mfe_toolbox.univariate.egarch_starting_values import egarch_starting_values
from mfe_toolbox.univariate.egarch_transform import egarch_transform
from mfe_toolbox.univariate.egarch_itransform import egarch_itransform
from mfe_toolbox.univariate.egarch_likelihood import egarch_likelihood
from mfe_toolbox.univariate.egarch_core import egarch_core
from mfe_toolbox.univariate.egarch_nlcon import egarch_nlcon
from mfe_toolbox.univariate.egarch_display import egarch_display
from mfe_toolbox.utility.robustvcv import robustvcv
from mfe_toolbox.utility.hessian_2sided import hessian_2sided

__all__ = ['egarch']

# Mapping from integer error-type codes to display-friendly strings.
# Used when calling egarch_display, which expects a string error_type.
_ERROR_TYPE_NAMES: dict[int, str] = {
    1: 'NORMAL',
    2: 'STUDENTST',
    3: 'GED',
    4: 'SKEWT',
}

# Maximum number of robustness retry iterations when the optimizer does not
# converge and the user did not supply starting values.
# Ref: egarch.m:221 — MATLAB iterates over all ordered_parameters rows
# (up to 36 = 4*3*3 grid combos).  We use a comparable count of perturbation
# restarts for equivalent robustness.
_MAX_ROBUST_ITER: int = 36


def egarch(
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: str = 'NORMAL',
    startingvals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Estimate an EGARCH(P,O,Q) model with constrained optimization.

    Uses ``scipy.optimize.minimize(method='SLSQP')`` with a nonlinear
    inequality constraint from :func:`egarch_nlcon` to enforce covariance
    stationarity (all characteristic polynomial roots of the beta coefficients
    inside the unit circle).

    Parameters
    ----------
    epsilon : np.ndarray
        T-element 1-D array of mean-zero data (residuals / innovations).
    p : int
        Positive integer — number of symmetric innovation (|z|) lags.
    o : int
        Non-negative integer — number of asymmetric innovation (z) lags.
        Use 0 for a symmetric EGARCH process.
    q : int
        Non-negative integer — number of GARCH (log-variance) lags.
        Use 0 for a pure ARCH specification.
    error_type : str, optional
        Error distribution.  One of ``'NORMAL'`` (default), ``'STUDENTST'``,
        ``'GED'``, or ``'SKEWT'``.
    startingvals : np.ndarray or None, optional
        Starting parameter vector.  Layout::

            [omega, alpha(1)...alpha(p), gamma(1)...gamma(o),
             beta(1)...beta(q), [nu, [lambda]]]

        When ``None``, a grid search is performed automatically.
    options : dict or None, optional
        Optimizer options dict for ``scipy.optimize.minimize``.  When ``None``,
        sensible defaults are used (ftol=1e-5, maxiter=200*(2+p+q)).

    Returns
    -------
    parameters : np.ndarray
        Estimated parameter vector in constrained space.
    LL : float
        Maximized log-likelihood value (positive).
    ht : np.ndarray
        Conditional variance series of length T (aligned with ``epsilon``).
    VCVrobust : np.ndarray
        Robust (sandwich) variance-covariance matrix of the parameters.
    VCV : np.ndarray
        Non-robust variance-covariance matrix (inverse Hessian / (T - m)).
    scores : np.ndarray
        T × K matrix of numerical scores (per-observation, per-parameter).
    diagnostics : dict
        Optimization diagnostics with keys ``'EXITFLAG'``, ``'ITERATIONS'``,
        ``'FUNCCOUNT'``, and ``'MESSAGE'``.

    Raises
    ------
    ValueError
        If inputs fail validation (see :func:`egarch_parameter_check`).

    See Also
    --------
    egarch_likelihood : Log-likelihood objective function.
    egarch_core : Numba JIT variance recursion.
    egarch_nlcon : Nonlinear stationarity constraint.
    egarch_parameter_check : Input validation.
    egarch_starting_values : Grid-search starting values.
    egarch_transform : Constrained → unconstrained parameter mapping.
    egarch_itransform : Unconstrained → constrained parameter mapping.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal(1000) * 0.01
    >>> params, ll, ht, vcvr, vcv, sc, diag = egarch(data, 1, 1, 1)
    >>> params.shape[0] == 1 + 1 + 1 + 1  # omega + alpha + gamma + beta
    True
    >>> ht.shape[0] == len(data)
    True
    """
    # ==================================================================
    # 1. Input validation
    # Ref: egarch.m:75-86
    # ==================================================================
    p, o, q, error_type_code, startingvals_checked, options = (
        egarch_parameter_check(epsilon, p, o, q, error_type, startingvals, options)
    )
    # After validation, error_type_code is an int (1-4).
    # startingvals_checked is np.ndarray (possibly empty) or None-equivalent.

    # Ensure epsilon is a flat float64 array for all subsequent computation.
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()

    # ==================================================================
    # 2. Initial setup
    # Ref: egarch.m:92 — m = max([p o q])
    # ==================================================================
    m: int = max(p, o, q)

    # ------------------------------------------------------------------
    # 2a. Backcast computation for log-variance initialization
    # Ref: egarch.m:95-103
    # Exponentially decaying weighted average of the first sqrt(T) squared
    # observations, then take the natural log.
    # ------------------------------------------------------------------
    T_raw: int = len(epsilon)
    # Ref: egarch.m:95 — back_cast_length = max(floor(length(data)^(1/2)), 1)
    back_cast_length: int = max(int(np.floor(T_raw ** 0.5)), 1)

    # Ref: egarch.m:96 — back_cast_weights = .05*(.9.^(0:back_cast_length))
    # MATLAB 0:back_cast_length → (back_cast_length+1) elements
    back_cast_weights = 0.05 * (0.9 ** np.arange(back_cast_length + 1))

    # Ref: egarch.m:97 — normalize weights
    back_cast_weights = back_cast_weights / np.sum(back_cast_weights)

    # Ref: egarch.m:98 — back_cast = weights * (data(1:back_cast_length+1).^2)
    # MATLAB 1-indexed data(1:back_cast_length+1) → Python 0-indexed [0:back_cast_length+1]
    back_cast: float = float(
        back_cast_weights @ (epsilon[0:back_cast_length + 1] ** 2)
    )

    # Ref: egarch.m:99-103 — log transform of back_cast
    if back_cast == 0.0:
        # Ref: egarch.m:100 — back_cast = log(cov(data))
        # np.var with ddof=1 matches MATLAB's cov(vector)
        sample_var = float(np.var(epsilon, ddof=1))
        if sample_var <= 0.0:
            sample_var = 1e-10  # safety floor to avoid log(0)
        back_cast = float(np.log(sample_var))
    else:
        # Ref: egarch.m:102 — back_cast = log(back_cast)
        back_cast = float(np.log(back_cast))

    # ------------------------------------------------------------------
    # 2b. Augment data with m leading zeros for backcast initialization
    # Ref: egarch.m:105 — data_augmented = [zeros(m,1); data]
    # ------------------------------------------------------------------
    data_augmented: np.ndarray = np.concatenate(
        [np.zeros(m, dtype=np.float64), epsilon]
    )

    # Ref: egarch.m:108 — T = size(data_augmented, 1)
    T: int = len(data_augmented)

    # ------------------------------------------------------------------
    # 2c. Compute upper bound for conditional variance
    # Ref: egarch_likelihood.m:61 — upper = 10000 * max(data.^2)
    # The driver computes this once and passes to egarch_likelihood.
    # ------------------------------------------------------------------
    max_eps_sq: float = float(np.max(data_augmented ** 2))
    if max_eps_sq <= 0.0:
        max_eps_sq = 1e-10  # safety floor
    upper: float = 10000.0 * max_eps_sq

    # ==================================================================
    # 3. Starting values
    # Ref: egarch.m:114-129
    # ==================================================================
    # Determine if user supplied starting values (controls robustness retry).
    # Ref: egarch.m:114-118
    _sv = startingvals_checked
    if _sv is None or (isinstance(_sv, np.ndarray) and _sv.size == 0):
        startingflag: int = 0
    else:
        startingflag: int = 1

    # Ref: egarch.m:120 — grid search or parse user-supplied values
    sv, nu, lam = egarch_starting_values(
        _sv if startingflag == 1 else None,
        data_augmented,
        p, o, q,
        error_type_code,
        back_cast,
        T,
    )

    # Ref: egarch.m:122 — startingvals = [startingvals; nu; lambda]
    # Assemble full parameter vector including distribution shape params.
    parts: list[np.ndarray] = [sv]
    if nu is not None:
        parts.append(np.array([nu], dtype=np.float64))
    if lam is not None:
        parts.append(np.array([lam], dtype=np.float64))
    sv_full: np.ndarray = np.concatenate(parts)

    # Ref: egarch.m:124 — transform starting values to unconstrained space
    sv_trans: np.ndarray = egarch_transform(sv_full, p, o, q, error_type_code)

    # Ref: egarch.m:126 — evaluate initial LL for convergence comparison
    LL0: float = egarch_likelihood(
        sv_trans, data_augmented, p, o, q, error_type_code,
        back_cast, T, upper, True,
    )[0]

    # ==================================================================
    # 4. Set up bounds and constraints for SLSQP
    # Ref: egarch.m:132-136
    # ==================================================================
    n_params: int = len(sv_trans)

    # Ref: egarch.m:132-133 — LB = -inf, except alpha/gamma in [-3, 3]
    LB = np.full(n_params, -np.inf)
    UB = np.full(n_params, np.inf)
    # Ref: egarch.m:134 — LB(2:p+o+1) = -3  [MATLAB 1-indexed]
    # Python 0-indexed: indices 1 through p+o (inclusive) → slice [1:p+o+1]
    LB[1:p + o + 1] = -3.0
    # Ref: egarch.m:136 — UB(2:p+o+1) = 3
    UB[1:p + o + 1] = 3.0

    bounds: list[tuple[float, float]] = list(zip(LB.tolist(), UB.tolist()))

    # ------------------------------------------------------------------
    # Nonlinear stationarity constraint.
    # MATLAB fmincon convention: c(x) <= 0 is feasible.
    # scipy SLSQP convention: c(x) >= 0 is feasible.
    # egarch_nlcon returns c = |root| - 0.99998 (MATLAB sign convention),
    # so we negate it for scipy: -c = 0.99998 - |root| >= 0 when |root| < 0.99998.
    # Ref: egarch.m:141 — fmincon(..., 'egarch_nlcon', ...)
    # ------------------------------------------------------------------
    constraints: list[dict] = [
        {
            'type': 'ineq',
            'fun': lambda params, _p=p, _o=o, _q=q, _et=error_type_code: (
                -egarch_nlcon(params, _p, _o, _q, _et)
            ),
        }
    ]

    # ==================================================================
    # 5. Primary optimization via SLSQP
    # Ref: egarch.m:141 — fmincon('egarch_likelihood', startingvals, ...)
    # ==================================================================
    opt_result = minimize(
        fun=lambda params: egarch_likelihood(
            params, data_augmented, p, o, q, error_type_code,
            back_cast, T, upper, True,
        )[0],
        x0=sv_trans,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options=options,
    )

    parameters: np.ndarray = opt_result.x.copy()
    LL: float = opt_result.fun
    exitflag: bool = opt_result.success
    output: dict = {
        'iterations': getattr(opt_result, 'nit', 0),
        'funcCount': getattr(opt_result, 'nfev', 0),
        'message': getattr(opt_result, 'message', ''),
    }

    # ==================================================================
    # 6. Estimation robustness — retry on non-convergence
    # Ref: egarch.m:149-229
    # ==================================================================

    # ------------------------------------------------------------------
    # 6a. If optimization did not converge but improved, retry with
    #     increased iterations.
    # Ref: egarch.m:149-168
    # ------------------------------------------------------------------
    if (not exitflag) and (LL < LL0):
        retry_options = dict(options) if options is not None else {}
        current_maxiter = retry_options.get('maxiter', 200 * (2 + p + q))
        retry_options['maxiter'] = 2 * current_maxiter

        opt_result2 = minimize(
            fun=lambda params: egarch_likelihood(
                params, data_augmented, p, o, q, error_type_code,
                back_cast, T, upper, True,
            )[0],
            x0=parameters,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=retry_options,
        )

        parameters = opt_result2.x.copy()
        LL = opt_result2.fun
        exitflag = opt_result2.success
        output = {
            'iterations': getattr(opt_result2, 'nit', 0),
            'funcCount': getattr(opt_result2, 'nfev', 0),
            'message': getattr(opt_result2, 'message', ''),
        }

    # ------------------------------------------------------------------
    # 6b. If still not converged and user did not supply starting values,
    #     try alternative starting values (perturbation restarts).
    # Ref: egarch.m:172-229
    # ------------------------------------------------------------------
    if startingflag == 0 and (not exitflag):
        warnings.warn(
            'Not Successful! Trying alternative starting values.',
            stacklevel=2,
        )

        # Track best results across all restarts
        robust_parameters: list[np.ndarray] = [parameters.copy()]
        robust_LL: list[float] = [LL]
        robust_output: list[dict] = [output.copy()]

        # Generate perturbation scales for restarts.
        # We perturb the core EGARCH parameters while keeping distribution
        # shape parameters unchanged, mirroring the MATLAB approach of
        # iterating over ordered_parameters from the grid search.
        rng = np.random.default_rng(42)
        robust_iter: int = 1

        while (not exitflag) and robust_iter < _MAX_ROBUST_ITER:
            # Perturb the initial starting values (in constrained space)
            # by scaling each core parameter with a random factor.
            sv_perturbed = sv_full.copy()
            n_core = 1 + p + o + q
            perturbation = rng.uniform(0.5, 1.5, size=n_core)
            sv_perturbed[:n_core] *= perturbation

            # Transform perturbed starting values
            sv_perturbed_trans = egarch_transform(
                sv_perturbed, p, o, q, error_type_code,
            )

            # Evaluate initial LL for perturbed starting values
            LL0_retry = egarch_likelihood(
                sv_perturbed_trans, data_augmented, p, o, q,
                error_type_code, back_cast, T, upper, True,
            )[0]

            # Optimize with perturbed starting values
            # Ref: egarch.m:194
            opt_retry = minimize(
                fun=lambda params: egarch_likelihood(
                    params, data_augmented, p, o, q, error_type_code,
                    back_cast, T, upper, True,
                )[0],
                x0=sv_perturbed_trans,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options=options,
            )

            parameters_retry = opt_retry.x.copy()
            LL_retry = opt_retry.fun
            exitflag_retry = opt_retry.success
            output_retry = {
                'iterations': getattr(opt_retry, 'nit', 0),
                'funcCount': getattr(opt_retry, 'nfev', 0),
                'message': getattr(opt_retry, 'message', ''),
            }

            # Ref: egarch.m:195-212 — if not converged but improved,
            # retry with more iterations
            if (not exitflag_retry) and (LL_retry < LL0_retry):
                retry_opts = dict(options) if options is not None else {}
                curr_max = retry_opts.get('maxiter', 200 * (2 + p + q))
                retry_opts['maxiter'] = 2 * curr_max

                opt_retry2 = minimize(
                    fun=lambda params: egarch_likelihood(
                        params, data_augmented, p, o, q, error_type_code,
                        back_cast, T, upper, True,
                    )[0],
                    x0=parameters_retry,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options=retry_opts,
                )
                parameters_retry = opt_retry2.x.copy()
                LL_retry = opt_retry2.fun
                exitflag_retry = opt_retry2.success
                output_retry = {
                    'iterations': getattr(opt_retry2, 'nit', 0),
                    'funcCount': getattr(opt_retry2, 'nfev', 0),
                    'message': getattr(opt_retry2, 'message', ''),
                }

            # Ref: egarch.m:214-216 — save results from this restart
            robust_parameters.append(parameters_retry.copy())
            robust_LL.append(LL_retry)
            robust_output.append(output_retry.copy())

            # Update convergence flag from this iteration
            exitflag = exitflag_retry
            if exitflag:
                # Converged — use this result
                parameters = parameters_retry.copy()
                LL = LL_retry
                output = output_retry.copy()

            robust_iter += 1

        # If we exhausted all restarts without convergence, select the best
        # Ref: egarch.m:222-228
        if not exitflag:
            warnings.warn(
                'Convergence not achieved. Use results with CAUTION. '
                'You may need to provide starting values.',
                stacklevel=2,
            )
            best_idx = int(np.argmin(robust_LL))
            LL = robust_LL[best_idx]
            parameters = robust_parameters[best_idx].copy()
            output = robust_output[best_idx].copy()

    # ==================================================================
    # 7. Inverse transform parameters back to constrained space
    # Ref: egarch.m:234
    # ==================================================================
    parameters = egarch_itransform(parameters, p, o, q, error_type_code)

    # ==================================================================
    # 8. Final log-likelihood, per-observation likelihoods, and ht
    # Ref: egarch.m:237-240
    # ==================================================================
    LL_final, likelihoods, ht = egarch_likelihood(
        parameters, data_augmented, p, o, q, error_type_code,
        back_cast, T, upper, False,
    )
    # Ref: egarch.m:239 — LL = -LL  (convert from negative LL to positive LL)
    LL = -LL_final

    # ==================================================================
    # 9. Robust and non-robust VCV computation
    # Ref: egarch.m:243-246
    # ==================================================================
    nw: int = 0  # Ref: egarch.m:244 — nw = 0 (no Newey-West)

    # Wrapper to match robustvcv's expected signature: fun(theta, *args) → (ll, lls)
    # egarch_likelihood returns (LL, LLS, ht); robustvcv only needs first two.
    def _ll_for_vcv(
        params: np.ndarray,
        _data_aug: np.ndarray,
        _p: int,
        _o: int,
        _q: int,
        _et: int,
        _bc: float,
        _T: int,
        _upper: float,
    ) -> tuple[float, np.ndarray]:
        ll_val, lls_val, _ = egarch_likelihood(
            params, _data_aug, _p, _o, _q, _et, _bc, _T, _upper, False,
        )
        return ll_val, lls_val

    # Ref: egarch.m:245 — [VCVrobust, A, B, scores, hess] = robustvcv(...)
    VCVrobust, A, B, scores, hess, gross_scores = robustvcv(
        _ll_for_vcv, parameters, nw,
        data_augmented, p, o, q, error_type_code, back_cast, T, upper,
    )

    # Ref: egarch.m:246 — VCV = hess^(-1) / (T - m)
    # hess from robustvcv is the numerical Hessian divided by T (= A matrix).
    # Non-robust VCV = inv(hess) / (T - m).
    try:
        VCV: np.ndarray = np.linalg.inv(hess) / float(T - m)
    except np.linalg.LinAlgError:
        # Singular Hessian — fall back to pseudo-inverse
        VCV = np.linalg.pinv(hess) / float(T - m)

    # ==================================================================
    # 10. Diagnostics assembly
    # Ref: egarch.m:250-253
    # ==================================================================
    diagnostics: dict = {
        'EXITFLAG': 1 if exitflag else 0,
        'ITERATIONS': output.get('iterations', 0),
        'FUNCCOUNT': output.get('funcCount', 0),
        'MESSAGE': output.get('message', ''),
    }

    return parameters, LL, ht, VCVrobust, VCV, scores, diagnostics
