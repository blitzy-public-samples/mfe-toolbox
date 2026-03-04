"""
FIGARCH(Q,D,P) parameter estimation with multiple error distributions.

Implements the Fractionally Integrated GARCH model for P={0,1} and Q={0,1}
with Normal, Student's t, Generalized Error Distribution (GED), and
Skewed-t error distributions.  The FIGARCH model captures long-memory
dependence in the conditional variance process through the fractional
differencing parameter d.

The conditional variance h(t) of a FIGARCH(1,d,1) process is modeled as:

    h(t) = omega + [1 - beta*L - phi*L*(1-L)^d] * epsilon^2(t) + beta * h(t-1)

which is estimated using an ARCH(infinity) representation:

    h(t) = (1-beta)^(-1) * omega + sum_{i=1}^{truncLag} lambda(i) * epsilon^2(t-i)

where lambda(i) are functions of d, phi, and beta computed via
:func:`figarch_weights`.

Optimization uses ``scipy.optimize.minimize`` replacing MATLAB's ``fminunc``.
The parameter space is mapped to an unconstrained domain via
:func:`figarch_transform` / :func:`figarch_itransform` to allow unrestricted
optimization.

Migrated from: ``univariate/figarch.m`` (MFE Toolbox Version 4.0, 10424 bytes)

See Also
--------
mfe_toolbox.univariate.figarch_likelihood : FIGARCH log-likelihood.
mfe_toolbox.univariate.figarch_parameter_check : Input validation.
mfe_toolbox.univariate.figarch_starting_values : Grid search initialization.
mfe_toolbox.univariate.figarch_transform : Constrained -> unconstrained.
mfe_toolbox.univariate.figarch_itransform : Unconstrained -> constrained.
mfe_toolbox.univariate.figarch_weights : ARCH(infinity) weight computation.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/13/2009

Constraints (following MATLAB implementation):
    (1) omega > 0
    (2) 0 < d < 1
    (3) 0 <= phi <= (1-d)/2
    (4) 0 <= beta <= d + phi
    (5) nu > 2 for Student's t; nu > 1 for GED
    (6) -0.99 < lambda < 0.99 for Skewed t
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.univariate.figarch_parameter_check import figarch_parameter_check
from mfe_toolbox.univariate.figarch_starting_values import figarch_starting_values
from mfe_toolbox.univariate.figarch_transform import figarch_transform
from mfe_toolbox.univariate.figarch_itransform import figarch_itransform
from mfe_toolbox.univariate.figarch_likelihood import figarch_likelihood
from mfe_toolbox.utility.robustvcv import robustvcv


def figarch(
    epsilon: np.ndarray,
    p: int,
    q: int,
    error_type: str = 'NORMAL',
    trunc_lag: int = 1000,
    startingvals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Estimate FIGARCH(Q,D,P) model parameters via maximum likelihood.

    Performs constrained maximum likelihood estimation of a Fractionally
    Integrated GARCH model.  The estimation pipeline follows the standard
    GARCH pattern: parameter validation, starting value grid search,
    parameter transformation to unconstrained space, numerical optimization,
    inverse transformation, and robust inference.

    Parameters
    ----------
    epsilon : np.ndarray
        T-element array of mean-zero residuals (column vector).  Must be a
        1-D numeric array with more than one observation.
    p : int
        0 or 1 indicating whether the autoregressive (phi) term is present
        in the model.  FIGARCH restricts p to {0, 1}.
    q : int
        0 or 1 indicating whether the moving-average (beta) term is present
        in the model.  FIGARCH restricts q to {0, 1}.
    error_type : str, optional
        Error distribution type.  One of:

        - ``'NORMAL'`` — Gaussian innovations (default)
        - ``'STUDENTST'`` — Student's t distributed errors
        - ``'GED'`` — Generalized Error Distribution
        - ``'SKEWT'`` — Skewed Student's t distribution
    trunc_lag : int, optional
        Number of weights to compute in the ARCH(infinity) representation.
        Default is 1000.  Must be >= 10.
    startingvals : np.ndarray or None, optional
        Initial parameter vector ``[omega, (phi if p=1), d, (beta if q=1),
        (nu), (lambda)]``.  Length is ``2+p+q`` for Normal, plus 1 for
        Student's t or GED (nu), plus 2 for Skewed t (nu, lambda).
        If ``None``, :func:`figarch_starting_values` performs a grid search
        to find reasonable initial values.  Default is ``None``.
    options : dict or None, optional
        Optimizer options for ``scipy.optimize.minimize``.  If ``None``,
        sensible defaults matching the MATLAB ``optimset('fminunc')``
        configuration are used.  Default is ``None``.

    Returns
    -------
    parameters : np.ndarray
        Estimated parameter vector ``[omega, (phi), d, (beta), (nu), (lambda)]``.
    LL : float
        Maximized log-likelihood value (positive).
    ht : np.ndarray
        T-element conditional variance series.
    VCVrobust : np.ndarray
        Robust (sandwich) parameter variance-covariance matrix.
    VCV : np.ndarray
        Non-robust parameter VCV (inverse Hessian / T).
    scores : np.ndarray
        T x num_params matrix of numerical scores (per-observation,
        per-parameter).
    diagnostics : dict
        Optimization diagnostics with keys:

        - ``EXITFLAG`` (int): 1 if converged, 0 otherwise.
        - ``ITERATIONS`` (int): Number of optimizer iterations.
        - ``FUNCCOUNT`` (int): Number of function evaluations.
        - ``MESSAGE`` (str): Optimizer convergence message.

    Raises
    ------
    ValueError
        If input parameters fail validation in :func:`figarch_parameter_check`.

    Notes
    -----
    Ref: figarch.m:74-87 — Input checking via figarch_parameter_check.

    Ref: figarch.m:93-103 — Backcast computation using exponentially decaying
    weights on the first sqrt(T) squared observations.

    Ref: figarch.m:110-125 — Starting value grid search and transformation.

    Ref: figarch.m:133 — Primary optimization via fminunc (replaced by
    scipy.optimize.minimize with method='L-BFGS-B' per AAP Rule 7).

    Ref: figarch.m:140-209 — Robustness retries with alternative starting
    values and increased iteration limits.

    Ref: figarch.m:213-227 — Inverse transform, final likelihood evaluation,
    and robust VCV computation via robustvcv.

    References
    ----------
    .. [1] Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
       Fractionally integrated generalized autoregressive conditional
       heteroskedasticity. *Journal of Econometrics*, 74(1), 3-30.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> eps = rng.standard_normal(500)
    >>> params, ll, ht, vcvr, vcv, sc, diag = figarch(eps, 1, 1)
    >>> params.shape[0] == 4  # [omega, phi, d, beta]
    True
    """
    # ==================================================================
    # Input Checking
    # Ref: figarch.m:74-87 — Validate all inputs via parameter_check
    # ==================================================================
    p, q, error_type_int, trunc_lag, startingvals, options = (
        figarch_parameter_check(
            epsilon, p, q, error_type, trunc_lag, startingvals, options
        )
    )

    # Ensure epsilon is a float64 1-D array for all subsequent operations
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()
    T = len(epsilon)

    # ==================================================================
    # Backcast computation
    # Ref: figarch.m:93-99 — Exponentially decaying weights on initial
    # squared observations for pre-sample variance initialization
    # ==================================================================
    # Ref: figarch.m:93 — backCastLength = max(floor(length(epsilon)^(1/2)),1)
    back_cast_length = max(int(np.floor(np.sqrt(T))), 1)

    # Ref: figarch.m:94 — backCastWeights = .05*(.9.^(0:backCastLength))
    back_cast_weights = 0.05 * (0.9 ** np.arange(back_cast_length + 1))
    # Ref: figarch.m:95 — backCastWeights = backCastWeights/sum(backCastWeights)
    back_cast_weights = back_cast_weights / np.sum(back_cast_weights)
    # Ref: figarch.m:96 — backCast = backCastWeights*((epsilon(1:backCastLength+1)).^2)
    back_cast = float(
        np.dot(back_cast_weights, epsilon[: back_cast_length + 1] ** 2)
    )

    # Ref: figarch.m:97-99 — Fallback if backcast is zero (e.g., data is all zeros)
    if back_cast == 0.0:
        back_cast = float(np.var(epsilon, ddof=1))
        # Ref: figarch.m:98 — MATLAB uses cov(epsilon) which for a vector equals var(x,1)
        # with N-1 normalization; np.var(ddof=1) matches this

    # ==================================================================
    # Starting values
    # Ref: figarch.m:108-125
    # ==================================================================
    # Ref: figarch.m:110-114 — Track whether user supplied starting values
    # to control robustness retry behavior
    if startingvals is None:
        starting_flag = 0
    else:
        starting_flag = 1

    # Ref: figarch.m:116 — Grid search for starting values
    # Python API differs from MATLAB: returns (sv, nu, lam) without
    # orderedParameters.  Robustness retries use perturbations instead.
    sv, nu, lam = figarch_starting_values(
        startingvals, epsilon, p, q, error_type_int, trunc_lag, back_cast, T
    )

    # Ref: figarch.m:118 — startingvals = [startingvals; nu; lambda]
    # Assemble the full starting value vector including distribution params
    parts = [np.asarray(sv, dtype=np.float64).ravel()]
    if nu is not None:
        parts.append(np.array([nu], dtype=np.float64))
    if lam is not None:
        parts.append(np.array([lam], dtype=np.float64))
    sv_full = np.concatenate(parts)

    # Ref: figarch.m:120-122 — Transform starting values to unconstrained space
    sv_transformed = figarch_transform(sv_full, p, q, error_type_int)

    # ==================================================================
    # Optimizer options setup
    # Ref: figarch.m:56-62 — Default options matching MATLAB optimset
    # ==================================================================
    if options is None or len(options) == 0:
        num_vars = len(sv_transformed)
        options = {
            'maxiter': 400 * num_vars,
            'ftol': 1e-5,
            'disp': False,
        }

    # Filter options to only include keys recognized by L-BFGS-B to avoid
    # OptimizeWarning from scipy.  Ref: figarch.m uses fminunc (unconstrained)
    # — AAP maps fminunc → L-BFGS-B (Rule 7).
    _lbfgsb_keys = {'ftol', 'gtol', 'maxiter', 'maxfun', 'disp', 'maxcor', 'maxls', 'eps'}
    _filtered_options = {k: v for k, v in options.items() if k in _lbfgsb_keys}

    # ==================================================================
    # Objective function for optimization
    # Ref: figarch.m:130 — LL0 used to check that log-likelihood improves
    # Ref: figarch.m:133 — fminunc('figarch_likelihood', ..., true)
    # The estim_flag=True causes figarch_likelihood to inverse-transform
    # internally, allowing the optimizer to work in unconstrained space.
    # ==================================================================
    def _objective(params: np.ndarray) -> float:
        """Negative log-likelihood objective for minimization."""
        return figarch_likelihood(
            params, epsilon, p, q, error_type_int,
            trunc_lag, back_cast, T, True
        )[0]

    # Ref: figarch.m:130 — LL0 = initial likelihood for convergence comparison
    LL0 = _objective(sv_transformed)

    # Ref: figarch.m:133 — Primary optimization
    # MATLAB uses fminunc (unconstrained) because parameters are transformed.
    # Python uses L-BFGS-B (per AAP Rule 7: fminunc → L-BFGS-B) which is
    # the correct mapping for unconstrained optimization.
    result = minimize(
        _objective, sv_transformed, method='L-BFGS-B', options=_filtered_options
    )
    parameters = result.x.copy()
    LL = result.fun
    exitflag = 1 if result.success else 0
    last_output = result

    # ==================================================================
    # Estimation Robustness — retry with more iterations
    # Ref: figarch.m:140-159
    # If optimization did not converge but improved on the initial
    # log-likelihood, try again with increased iteration limits.
    # ==================================================================
    if exitflag <= 0 and LL < LL0:
        # Ref: figarch.m:148-156 — Double max iterations and max fun evals
        retry_options = dict(options)
        num_params_opt = len(parameters)
        current_maxiter = options.get('maxiter', 400 * num_params_opt)
        retry_options['maxiter'] = 2 * current_maxiter

        # Ref: figarch.m:158 — Re-optimize from current best parameters
        retry_filtered = {k: v for k, v in retry_options.items() if k in _lbfgsb_keys}
        result2 = minimize(
            _objective, parameters, method='L-BFGS-B', options=retry_filtered
        )
        parameters = result2.x.copy()
        LL = result2.fun
        exitflag = 1 if result2.success else 0
        last_output = result2

    # ==================================================================
    # Estimation Robustness — try alternative starting values
    # Ref: figarch.m:164-209
    # When grid search was used (starting_flag == 0) and optimization
    # still has not converged, try optimization from perturbed starting
    # values.  The MATLAB version uses orderedParameters from the grid
    # search; since the Python figarch_starting_values does not return
    # these, we use small random perturbations of the transformed
    # starting values as diverse restart points.
    # ==================================================================
    if starting_flag == 0 and exitflag <= 0:
        # Ref: figarch.m:167-169 — Track robust parameter estimates
        robust_parameters = [parameters.copy()]
        # Ref: figarch.m:169-170 — Track log-likelihoods
        robust_LL = [LL]

        # Ref: figarch.m:173-203 — Iterate over alternative starting values
        max_retries = 10
        rng_retry = np.random.default_rng(0)
        index = 0

        while exitflag <= 0 and index < max_retries:
            # Generate alternative starting point by perturbing the best
            # transformed starting values with small random offsets
            perturbation = rng_retry.standard_normal(len(sv_transformed)) * 0.1
            alt_transformed = sv_transformed + perturbation

            # Ref: figarch.m:184 — Evaluate initial likelihood at this start
            alt_LL0 = _objective(alt_transformed)

            # Ref: figarch.m:187 — Optimize with alternative starting values
            result_alt = minimize(
                _objective, alt_transformed, method='L-BFGS-B',
                options=_filtered_options,
            )
            alt_params = result_alt.x.copy()
            alt_LL = result_alt.fun
            alt_exitflag = 1 if result_alt.success else 0
            last_output_alt = result_alt

            # Ref: figarch.m:188-197 — If improved but not converged, retry
            # with more iterations
            if alt_exitflag <= 0 and alt_LL < alt_LL0:
                retry_opts = dict(options)
                retry_opts['maxiter'] = 2 * options.get('maxiter', 800)
                retry_opts_filtered = {
                    k: v for k, v in retry_opts.items() if k in _lbfgsb_keys
                }
                result_alt2 = minimize(
                    _objective, alt_params, method='L-BFGS-B',
                    options=retry_opts_filtered,
                )
                alt_params = result_alt2.x.copy()
                alt_LL = result_alt2.fun
                alt_exitflag = 1 if result_alt2.success else 0
                last_output_alt = result_alt2

            # Ref: figarch.m:199-200 — Save results
            robust_parameters.append(alt_params)
            robust_LL.append(alt_LL)

            if alt_exitflag > 0:
                parameters = alt_params
                LL = alt_LL
                exitflag = alt_exitflag
                last_output = last_output_alt

            index += 1

        # Ref: figarch.m:204-209 — If no convergence achieved, use best result
        if exitflag <= 0:
            # Ref: figarch.m:206 — warning('MFEToolbox:Convergence',...)
            warnings.warn(
                'Convergence not achieved. Use results with caution',
                stacklevel=2,
            )
            # Ref: figarch.m:207-208 — Select parameters with best LL
            best_idx = int(np.argmin(robust_LL))
            LL = robust_LL[best_idx]
            parameters = robust_parameters[best_idx]

    # ==================================================================
    # Inverse transform to constrained space
    # Ref: figarch.m:213-214 — Transform from unconstrained back to
    # constrained parameter space
    # ==================================================================
    params_constrained, nu_final, lam_final = figarch_itransform(
        parameters, p, q, error_type_int
    )

    # Ref: figarch.m:214 — parameters = [parameters; nu; lambda]
    # Reassemble the full parameter vector including distribution params
    final_parts = [params_constrained]
    if nu_final is not None:
        final_parts.append(np.array([nu_final], dtype=np.float64))
    if lam_final is not None:
        final_parts.append(np.array([lam_final], dtype=np.float64))
    parameters = np.concatenate(final_parts)

    # ==================================================================
    # Final likelihood evaluation
    # Ref: figarch.m:216-219 — Compute log-likelihood at converged params
    # ==================================================================
    LL_final, likelihoods, ht = figarch_likelihood(
        parameters, epsilon, p, q, error_type_int,
        trunc_lag, back_cast, T, False
    )
    # Ref: figarch.m:218 — LL = -LL (convert from negative to positive)
    LL = -LL_final

    # ==================================================================
    # Robust VCV computation
    # Ref: figarch.m:222-226 — Uses robustvcv for sandwich estimator
    # ==================================================================
    num_params = len(parameters)

    # Wrapper for robustvcv: must return (scalar_LL, individual_LLs)
    # Ref: figarch.m:224 — robustvcv('figarch_likelihood', parameters, nw, ...)
    def _ll_for_vcv(
        params: np.ndarray,
        eps: np.ndarray,
        p_val: int,
        q_val: int,
        et: int,
        tl: int,
        bc: float,
        t_val: int,
    ) -> tuple[float, np.ndarray]:
        """Likelihood wrapper returning (scalar, per-obs) for robustvcv."""
        ll_scalar, lls_vec, _ = figarch_likelihood(
            params, eps, p_val, q_val, et, tl, bc, t_val, False
        )
        return ll_scalar, lls_vec

    try:
        # Ref: figarch.m:223 — nw=0 (no Newey-West on scores)
        nw = 0
        # Ref: figarch.m:224 — [VCVrobust,A,B,scores,hess] = robustvcv(...)
        VCVrobust, A, B, scores_raw, hess, gross_scores = robustvcv(
            _ll_for_vcv, parameters, nw,
            epsilon, p, q, error_type_int, trunc_lag, back_cast, T,
        )
        # Ref: figarch.m:226 — VCV = hess^(-1) / T
        # hess returned by robustvcv is A = H/T, so VCV = inv(A)/T = inv(H/T)/T
        VCV = np.linalg.inv(hess) / T
        # scores from robustvcv are T x num_params
        scores = scores_raw
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        # If Hessian is singular or computation fails, return NaN matrices
        VCVrobust = np.full((num_params, num_params), np.nan)
        VCV = np.full((num_params, num_params), np.nan)
        scores = np.full((T, num_params), np.nan)

    # ==================================================================
    # Diagnostics
    # Ref: figarch.m:230-233
    # ==================================================================
    diagnostics = {
        'EXITFLAG': exitflag,
        'ITERATIONS': getattr(last_output, 'nit', 0),
        'FUNCCOUNT': getattr(last_output, 'nfev', 0),
        'MESSAGE': getattr(last_output, 'message', ''),
    }

    return parameters, LL, ht, VCVrobust, VCV, scores, diagnostics
