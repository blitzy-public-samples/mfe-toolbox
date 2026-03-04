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
mfe_toolbox.univariate.figarch_transform : Constrained → unconstrained.
mfe_toolbox.univariate.figarch_itransform : Unconstrained → constrained.
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
from mfe_toolbox.utility.hessian_2sided import hessian_2sided


def figarch(
    epsilon: np.ndarray,
    p: int,
    q: int,
    error_type: str = 'NORMAL',
    trunc_lag: int = 1000,
    startingvals: np.ndarray = None,
    options: dict = None,
) -> tuple:
    """
    Estimate FIGARCH(Q,D,P) model parameters via maximum likelihood.

    Parameters
    ----------
    epsilon : np.ndarray
        T-element array of mean-zero residuals.
    p : int
        0 or 1 indicating whether the autoregressive (phi) term is present.
    q : int
        0 or 1 indicating whether the moving-average (beta) term is present.
    error_type : str, optional
        Error distribution: ``'NORMAL'``, ``'STUDENTST'``, ``'GED'``, or
        ``'SKEWT'``.  Default is ``'NORMAL'``.
    trunc_lag : int, optional
        Number of ARCH(infinity) weights to compute.  Default is 1000.
    startingvals : np.ndarray or None, optional
        Initial parameter vector ``[omega, (phi), d, (beta), (nu), (lambda)]``.
        If ``None``, :func:`figarch_starting_values` performs a grid search.
    options : dict or None, optional
        Optimizer options for ``scipy.optimize.minimize``.  If ``None``,
        sensible defaults are used.

    Returns
    -------
    tuple
        - ``parameters`` (np.ndarray): Estimated parameter vector.
        - ``LL`` (float): Maximized log-likelihood (positive).
        - ``ht`` (np.ndarray): T-element conditional variance series.
        - ``VCVrobust`` (np.ndarray): Robust (sandwich) parameter VCV matrix.
        - ``VCV`` (np.ndarray): Non-robust VCV (inverse Hessian / T).
        - ``scores`` (np.ndarray): (num_params × T) score matrix.
        - ``diagnostics`` (dict): Optimization diagnostics with keys
          ``EXITFLAG``, ``ITERATIONS``, ``FUNCCOUNT``, ``MESSAGE``.

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

    Ref: figarch.m:133 — Primary optimization via fminunc (→ scipy.optimize.minimize).

    Ref: figarch.m:140-209 — Robustness retries with alternative starting values
    and increased iteration limits.

    Ref: figarch.m:213-227 — Inverse transform, final likelihood evaluation,
    and robust VCV computation.
    """
    # ================================================================
    # Input Checking
    # Ref: figarch.m:74-87
    # ================================================================
    p, q, error_type_int, trunc_lag, startingvals, options = figarch_parameter_check(
        epsilon, p, q, error_type, trunc_lag, startingvals, options
    )

    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()
    T = len(epsilon)

    # ================================================================
    # Backcast computation
    # Ref: figarch.m:93-99
    # ================================================================
    back_cast_length = max(int(np.floor(np.sqrt(T))), 1)
    # Ref: figarch.m:94 — backCastWeights = .05*(.9.^(0:backCastLength))
    back_cast_weights = 0.05 * (0.9 ** np.arange(back_cast_length + 1))
    back_cast_weights = back_cast_weights / np.sum(back_cast_weights)
    # Ref: figarch.m:96 — backCast = backCastWeights * ((epsilon(1:backCastLength+1)).^2)
    back_cast = np.dot(back_cast_weights, epsilon[:back_cast_length + 1] ** 2)

    # Ref: figarch.m:97-99 — fallback if backcast is zero
    if back_cast == 0:
        back_cast = np.var(epsilon, ddof=1)

    # Ref: figarch.m:101-102 — Augmented squared epsilon with backcast
    # epsilon2Augmented = [zeros(truncLag,1); epsilon.^2]
    # epsilon2Augmented(1:truncLag) = backCast
    epsilon2_augmented = np.concatenate([
        np.full(trunc_lag, back_cast),
        epsilon ** 2,
    ])

    # ================================================================
    # Starting values
    # Ref: figarch.m:110-125
    # ================================================================
    # Ref: figarch.m:110-114 — track whether user supplied starting values
    if startingvals is None:
        starting_flag = 0
    else:
        starting_flag = 1

    # Ref: figarch.m:116 — Grid search for starting values
    sv, nu, lam, LLs, ordered_parameters = figarch_starting_values(
        startingvals, epsilon, epsilon2_augmented, p, q, error_type_int, trunc_lag
    )

    # Ref: figarch.m:118 — startingvals = [startingvals; nu; lambda]
    parts = [np.asarray(sv).ravel()]
    if nu is not None:
        parts.append(np.array([nu]))
    if lam is not None:
        parts.append(np.array([lam]))
    sv_full = np.concatenate(parts)

    # Ref: figarch.m:120-122 — Transform starting values to unconstrained space
    sv_transformed = figarch_transform(sv_full, p, q, error_type_int)

    # ================================================================
    # Default optimizer options
    # Ref: figarch.m:56-62
    # ================================================================
    if options is None or len(options) == 0:
        num_params = len(sv_transformed)
        options = {
            'maxiter': 400 * num_params,
            'ftol': 1e-5,
            'gtol': 1e-5,
        }

    # ================================================================
    # Primary optimization
    # Ref: figarch.m:130-133
    # ================================================================
    # Ref: figarch.m:130 — LL0 = initial likelihood for convergence check
    LL0 = figarch_likelihood(
        sv_transformed, p, q, epsilon, epsilon2_augmented,
        trunc_lag, error_type_int, True
    )[0]

    # Ref: figarch.m:133 — fminunc → scipy.optimize.minimize(method='L-BFGS-B')
    # Note: MATLAB figarch.m uses fminunc (unconstrained) because parameters
    # are already transformed. scipy equivalent is L-BFGS-B (quasi-Newton).
    def _objective(params):
        return figarch_likelihood(
            params, p, q, epsilon, epsilon2_augmented,
            trunc_lag, error_type_int, True
        )[0]

    result = minimize(
        _objective, sv_transformed, method='L-BFGS-B', options=options
    )
    parameters = result.x
    LL = result.fun
    exitflag = 1 if result.success else 0

    # ================================================================
    # Estimation Robustness — retry with more iterations
    # Ref: figarch.m:140-159
    # ================================================================
    if exitflag <= 0 and LL < LL0:
        # Ref: figarch.m:148-156 — increase iterations and try again
        retry_options = dict(options)
        num_params = len(parameters)
        retry_options['maxiter'] = 2 * max(options.get('maxiter', 200 * num_params), 200 * num_params)

        result2 = minimize(
            _objective, parameters, method='L-BFGS-B', options=retry_options
        )
        parameters = result2.x
        LL = result2.fun
        exitflag = 1 if result2.success else 0

    # ================================================================
    # Estimation Robustness — try alternative starting values
    # Ref: figarch.m:164-209
    # ================================================================
    if starting_flag == 0 and exitflag <= 0 and ordered_parameters.size > 0:
        # Ref: figarch.m:167-169 — track robust parameter estimates
        robust_parameters = [parameters.copy()]
        robust_LL = [LL]

        # Ref: figarch.m:173-203 — iterate over alternative starting values
        max_retries = min(ordered_parameters.shape[0], 10)
        index = 1
        while exitflag <= 0 and index < max_retries:
            # Ref: figarch.m:178-182 — try next best starting values
            alt_sv = ordered_parameters[index, :].copy()
            alt_parts = [alt_sv]
            if nu is not None:
                alt_parts.append(np.array([nu]))
            if lam is not None:
                alt_parts.append(np.array([lam]))
            alt_full = np.concatenate(alt_parts)

            # Ref: figarch.m:180 — transform alternative starting values
            alt_transformed = figarch_transform(alt_full, p, q, error_type_int)

            # Ref: figarch.m:184 — evaluate initial likelihood
            alt_LL0 = figarch_likelihood(
                alt_transformed, p, q, epsilon, epsilon2_augmented,
                trunc_lag, error_type_int, True
            )[0]

            # Ref: figarch.m:187 — optimize with alternative starting values
            result_alt = minimize(
                _objective, alt_transformed, method='L-BFGS-B', options=options
            )
            alt_params = result_alt.x
            alt_LL = result_alt.fun
            alt_exitflag = 1 if result_alt.success else 0

            # Ref: figarch.m:188-197 — retry with more iterations if improved
            if alt_exitflag <= 0 and alt_LL < alt_LL0:
                retry_options2 = dict(options)
                retry_options2['maxiter'] = 2 * options.get('maxiter', 800)
                result_alt2 = minimize(
                    _objective, alt_params, method='L-BFGS-B', options=retry_options2
                )
                alt_params = result_alt2.x
                alt_LL = result_alt2.fun
                alt_exitflag = 1 if result_alt2.success else 0

            # Ref: figarch.m:199-200 — save results
            robust_parameters.append(alt_params.copy())
            robust_LL.append(alt_LL)

            if alt_exitflag > 0:
                parameters = alt_params
                LL = alt_LL
                exitflag = alt_exitflag

            index += 1

        # Ref: figarch.m:204-209 — if still not converged, use best
        if exitflag <= 0:
            warnings.warn(
                'Convergence not achieved. Use results with caution',
                stacklevel=2,
            )
            best_idx = int(np.argmin(robust_LL))
            LL = robust_LL[best_idx]
            parameters = robust_parameters[best_idx]

    # ================================================================
    # Inverse transform to constrained space
    # Ref: figarch.m:213-214
    # ================================================================
    params_constrained, nu_final, lam_final = figarch_itransform(
        parameters, p, q, error_type_int
    )
    # Ref: figarch.m:214 — parameters = [parameters; nu; lambda]
    final_parts = [params_constrained]
    if nu_final is not None:
        final_parts.append(np.array([nu_final]))
    if lam_final is not None:
        final_parts.append(np.array([lam_final]))
    parameters = np.concatenate(final_parts)

    # ================================================================
    # Final likelihood evaluation
    # Ref: figarch.m:216-219
    # ================================================================
    LL_final, likelihoods, ht = figarch_likelihood(
        parameters, p, q, epsilon, epsilon2_augmented,
        trunc_lag, error_type_int, False
    )
    # Ref: figarch.m:218 — LL = -LL (convert from negative to positive)
    LL = -LL_final

    # ================================================================
    # Robust VCV computation
    # Ref: figarch.m:222-226
    # Uses hessian_2sided for numerical Hessian computation and
    # score-based sandwich estimator for robust standard errors.
    # ================================================================
    num_params = len(parameters)

    # Define the objective for Hessian computation (using constrained params)
    def _ll_for_hessian(params):
        return figarch_likelihood(
            params, p, q, epsilon, epsilon2_augmented,
            trunc_lag, error_type_int, False
        )[0]

    # Ref: figarch.m:224 — Hessian computation
    try:
        hess = hessian_2sided(_ll_for_hessian, parameters)
        # Ref: figarch.m:226 — VCV = hess^(-1) / T
        hess_inv = np.linalg.inv(hess)
        VCV = hess_inv / T

        # Compute scores for robust VCV (sandwich estimator)
        # Ref: figarch.m:224 — robustvcv computes B = (1/T) * sum(scores * scores')
        step_size = np.maximum(np.abs(parameters) * 1e-5, 1e-8)
        scores = np.zeros((num_params, T))
        _, base_lls, _ = figarch_likelihood(
            parameters, p, q, epsilon, epsilon2_augmented,
            trunc_lag, error_type_int, False
        )
        for i in range(num_params):
            params_plus = parameters.copy()
            params_plus[i] += step_size[i]
            _, lls_plus, _ = figarch_likelihood(
                params_plus, p, q, epsilon, epsilon2_augmented,
                trunc_lag, error_type_int, False
            )
            params_minus = parameters.copy()
            params_minus[i] -= step_size[i]
            _, lls_minus, _ = figarch_likelihood(
                params_minus, p, q, epsilon, epsilon2_augmented,
                trunc_lag, error_type_int, False
            )
            # Two-sided numerical gradient of per-observation log-likelihoods
            scores[i, :] = (lls_plus - lls_minus) / (2.0 * step_size[i])

        # Ref: figarch.m:224 — B = scores * scores' / T  (outer product)
        B = scores @ scores.T / T

        # Sandwich estimator: VCVrobust = A^(-1) * B * A^(-1) / T
        # where A = Hessian / T
        A_inv = hess_inv
        VCVrobust = (A_inv @ B @ A_inv) / T

    except (np.linalg.LinAlgError, ValueError):
        # If Hessian computation fails, return NaN matrices
        VCV = np.full((num_params, num_params), np.nan)
        VCVrobust = np.full((num_params, num_params), np.nan)
        scores = np.full((num_params, T), np.nan)

    # ================================================================
    # Diagnostics
    # Ref: figarch.m:230-233
    # ================================================================
    diagnostics = {
        'EXITFLAG': exitflag,
        'ITERATIONS': getattr(result, 'nit', 0),
        'FUNCCOUNT': getattr(result, 'nfev', 0),
        'MESSAGE': getattr(result, 'message', ''),
    }

    return parameters, LL, ht, VCVrobust, VCV, scores, diagnostics
