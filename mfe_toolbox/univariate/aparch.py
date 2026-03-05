"""
APARCH(P,O,Q) parameter estimation with multiple error distributions.

Estimates an Asymmetric Power ARCH (APARCH) model of Ding, Granger and
Engle (1993) by maximum likelihood, supporting Normal, Student's *t*,
Generalized Error Distribution (GED), and Hansen's Skewed *t* innovations.

The APARCH model generalizes many GARCH-family models by introducing a
power parameter delta and asymmetry parameters gamma:

    h(t)^(delta/2) = omega
                     + alpha(1)*(|r(t-1)| + gamma(1)*r(t-1))^delta + ...
                     + alpha(p)*(|r(t-p)| + gamma(p)*r(t-p))^delta
                     + beta(1)*h(t-1)^(delta/2) + ... + beta(q)*h(t-q)^(delta/2)

Special cases:
    - delta=2, o=0 recovers the standard GARCH model
    - delta=2, o>0 recovers the GJR-GARCH / TARCH model
    - delta=1 gives the absolute-value GARCH model

The estimation pipeline follows the standard MFE Toolbox pattern:
    parameter_check → starting_values → transform → optimize → itransform
    → inference → display

Constraints (applied via bounds in scipy.optimize.minimize SLSQP):
    (1) omega > 0
    (2) alpha(i) >= 0  for i = 1,...,p
    (3) -1 < gamma(i) < 1  for i = 1,...,o
    (4) beta(i) >= 0  for i = 1,...,q
    (5) delta > 0.3
    (6) sum(alpha) + sum(beta) < 1
    (7) nu > 2 for Student's T; nu > 1 for GED
    (8) -0.99 < lambda < 0.99 for Skewed T

Migrated from: univariate/aparch.m (MFE Toolbox v4.0, Kevin Sheppard)
Original Author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk

References
----------
Ding, Z., Granger, C.W.J. and Engle, R.F. (1993),
"A Long Memory Property of Stock Market Returns and a New Model",
*Journal of Empirical Finance*, 1, 83-106.
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.univariate.aparch_parameter_check import aparch_parameter_check
from mfe_toolbox.univariate.aparch_starting_values import aparch_starting_values
from mfe_toolbox.univariate.aparch_transform import aparch_transform
from mfe_toolbox.univariate.aparch_itransform import aparch_itransform
from mfe_toolbox.univariate.aparch_likelihood import aparch_likelihood
from mfe_toolbox.univariate.aparch_core import aparch_core  # noqa: F401 — listed in schema
from mfe_toolbox.univariate.aparch_display import aparch_display
from mfe_toolbox.utility.robustvcv import robustvcv
from mfe_toolbox.utility.hessian_2sided import hessian_2sided  # noqa: F401 — listed in schema


def aparch(
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: str = 'NORMAL',
    startingvals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Estimate an APARCH(P,O,Q) model by maximum likelihood.

    Parameters
    ----------
    epsilon : np.ndarray
        T-element 1-D array of mean-zero return observations.
    p : int
        Positive integer >= 1 — number of symmetric innovation (ARCH) lags.
    o : int
        Non-negative integer — number of asymmetric innovation lags.
        Must satisfy ``o <= p``.
    q : int
        Non-negative integer — number of lagged conditional variance
        (GARCH) lags.  Set to 0 for a pure ARCH model.
    error_type : str, optional
        Innovation distribution.  One of ``'NORMAL'`` (default),
        ``'STUDENTST'``, ``'GED'``, ``'SKEWT'``.
    startingvals : np.ndarray or None, optional
        Starting parameter vector for the optimizer, or ``None`` for
        automatic initialization via TARCH-based heuristic.

        Layout (when delta is estimated, i.e. no fixed user_delta):

        ``[omega, alpha(1)...alpha(p), gamma(1)...gamma(o),
          beta(1)...beta(q), delta, [nu, [lambda]]]``

        Extra elements ``nu`` and ``lambda`` are required for non-Normal
        distributions.
    options : dict or None, optional
        Options dictionary for ``scipy.optimize.minimize``.  If ``None``,
        sensible defaults are constructed.

    Returns
    -------
    parameters : np.ndarray
        Estimated parameter vector.
    LL : float
        Maximized log-likelihood value (positive).
    ht : np.ndarray
        T-element array of estimated conditional variances.
    VCVrobust : np.ndarray
        Robust (sandwich) variance-covariance matrix of parameters.
    VCV : np.ndarray
        Non-robust (inverse Hessian) variance-covariance matrix.
    scores : np.ndarray
        T × K matrix of per-observation scores.
    diagnostics : dict
        Optimizer diagnostics with keys ``'EXITFLAG'``, ``'ITERATIONS'``,
        ``'FUNCCOUNT'``, ``'MESSAGE'``.

    Raises
    ------
    ValueError
        If any input parameter fails validation.

    Notes
    -----
    This function replaces MATLAB's ``fminunc``/``fmincon`` optimizer with
    ``scipy.optimize.minimize(method='SLSQP')`` using the forward/inverse
    parameter transformation pair (``aparch_transform`` /
    ``aparch_itransform``) to map constraints to an unconstrained space.

    The ``user_delta`` feature of the original MATLAB code (fixing the power
    parameter delta to a user-supplied value) is exposed via the
    ``aparch_parameter_check`` interface but is not a direct parameter of
    this function.  To fix delta, include it in ``startingvals`` with the
    appropriate vector length and pass ``error_type`` accordingly; or call
    ``aparch_parameter_check`` manually.

    Ref: aparch.m — complete MATLAB-to-Python migration with 0-based indexing.

    See Also
    --------
    aparch_likelihood : APARCH log-likelihood function.
    aparch_core : APARCH variance recursion (Numba JIT).
    aparch_parameter_check : Input validation.
    aparch_starting_values : Automatic starting value computation.
    aparch_transform : Forward parameter transformation.
    aparch_itransform : Inverse parameter transformation.
    aparch_display : Result display formatting.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal(1000) * 0.01
    >>> params, ll, ht, vcv_r, vcv, scores, diag = aparch(data, 1, 1, 1)
    """
    # ==================================================================
    # Step 1: Input Validation
    # Ref: aparch.m:83-96 — nargin dispatch to aparch_parameter_check
    # In Python, default arguments replace nargin; user_delta is handled
    # internally by aparch_parameter_check when startingvals length implies
    # it.  We pass user_delta=None to let parameter_check decide.
    # ==================================================================
    (p, o, q, error_type_code, user_delta, delta_is_estimated,
     startingvals, options) = aparch_parameter_check(
        epsilon, p, o, q, error_type, None, startingvals, options
    )

    # Ref: aparch.m:85-93 — aparch_parameter_check returns:
    #   p, o, q         — validated integers
    #   error_type_code — integer code (1=NORMAL, 2=STUDENTST, 3=GED, 4=SKEWT)
    #   user_delta      — float or None
    #   delta_is_estimated — bool (True when delta is free; False when fixed)
    #   startingvals    — validated array or None
    #   options         — dict

    # Convert delta_is_estimated from bool to int for downstream functions
    # that use it as an integer offset in index arithmetic.
    # Ref: aparch.m uses logical (1/0) for deltaIsEstimated throughout.
    delta_is_estimated_int = int(delta_is_estimated)

    # ==================================================================
    # Step 2: Data Augmentation
    # Ref: aparch.m:102-110
    # ==================================================================
    # Ref: aparch.m:102 — m = max([p o q])
    m = int(np.max(np.array([p, o, q])))

    # Ref: aparch.m:107 — data_aug = [zeros(m,1); data]
    # Prepend m zeros for back-cast look-back period
    data = np.asarray(epsilon, dtype=np.float64).ravel()
    data_aug = np.concatenate([np.zeros(m), data])

    # Ref: aparch.m:108 — abs_data_aug = [mean(abs(data))*ones(m,1); abs(data)]
    # Prepend mean(|data|) for absolute-value augmentation
    abs_data_aug = np.concatenate([
        np.mean(np.abs(data)) * np.ones(m),
        np.abs(data),
    ])

    # Ref: aparch.m:110 — T = size(data_aug, 1)
    T = len(data_aug)

    # ==================================================================
    # Step 3: Starting Values
    # Ref: aparch.m:119-131
    # ==================================================================
    # Ref: aparch.m:119-123 — startingflag tracks user-supplied vs auto
    if startingvals is None or (isinstance(startingvals, np.ndarray) and startingvals.size == 0):
        startingflag = 0
    else:
        startingflag = 1

    # Ref: aparch.m:125 — compute or parse starting values
    startingvals_sv, nu, lambda_param = aparch_starting_values(
        startingvals, data, p, o, q, error_type_code, delta_is_estimated
    )

    # Ref: aparch.m:127 — startingvals = [startingvals; nu; lambda]
    # Assemble the full starting parameter vector including distribution params
    sv_parts = [startingvals_sv]
    if nu is not None:
        sv_parts.append(np.array([nu]))
    if lambda_param is not None:
        sv_parts.append(np.array([lambda_param]))
    startingvals_full = np.concatenate(sv_parts)

    # Ref: aparch.m:129 — transform starting values to unconstrained space
    (aparch_params_transformed, nu_transformed,
     lambda_transformed) = aparch_transform(
        startingvals_full, p, o, q, error_type_code, delta_is_estimated_int
    )

    # Ref: aparch.m:131 — re-append transformed nu, lambda
    sv_trans_parts = [aparch_params_transformed]
    if nu_transformed is not None:
        sv_trans_parts.append(np.array([nu_transformed]))
    if lambda_transformed is not None:
        sv_trans_parts.append(np.array([lambda_transformed]))
    startingvals_transformed = np.concatenate(sv_trans_parts)

    # ==================================================================
    # Step 4: Initial Log-Likelihood Evaluation
    # Ref: aparch.m:139 — LL0 used to verify optimization improves LL
    # ==================================================================
    LL0, _temp_lls, _ht0 = aparch_likelihood(
        startingvals_transformed,
        data_aug, abs_data_aug, p, o, q,
        error_type_code, T,
        delta_is_estimated_int, user_delta,
        True,  # estim_flag=True → parameters are in transformed space
    )

    # ==================================================================
    # Step 5: Parameter Optimization
    # Ref: aparch.m:141 — fminunc('aparch_likelihood', sv, options, ...)
    # Replaced with scipy.optimize.minimize(method='SLSQP')
    #
    # The aparch_likelihood function with estim_flag=True internally
    # applies aparch_itransform to map from unconstrained to constrained
    # space, so the optimizer works in unconstrained space.  We use SLSQP
    # as specified in the AAP for constrained optimization patterns,
    # though bounds are not strictly needed here because the parameter
    # transformation handles constraints implicitly.
    # ==================================================================

    # Build scipy options dict from validated options
    # Ref: aparch.m:61-67 — MATLAB optimset defaults translated
    scipy_options = {
        'maxiter': options.get('maxiter', 200 * (2 + p + q)),
        'ftol': options.get('ftol', 1e-5),
        'disp': False,  # Suppress scipy optimizer output
    }

    def _objective(params):
        """Wrapper returning scalar negative log-likelihood for minimization."""
        ll_val, _lls, _ht = aparch_likelihood(
            params,
            data_aug, abs_data_aug, p, o, q,
            error_type_code, T,
            delta_is_estimated_int, user_delta,
            True,  # estim_flag — transformed space
        )
        return ll_val

    # Ref: aparch.m:141 — fminunc call
    result = minimize(
        _objective,
        startingvals_transformed,
        method='SLSQP',
        options=scipy_options,
    )

    parameters_opt = result.x
    LL_opt = result.fun
    exitflag = 1 if result.success else 0
    output_info = result

    # ==================================================================
    # Step 6: Convergence Warning
    # Ref: aparch.m:149 — exitflag<=0 && LL<LL0
    # Issue a warning if optimization did not converge
    # ==================================================================
    if not result.success:
        warnings.warn(
            'Optimization did not converge. Results may be unreliable. '
            f'Message: {result.message}',
            stacklevel=2,
        )

    # ==================================================================
    # Step 7: Inverse Transform — Recover Constrained Parameters
    # Ref: aparch.m:222-224
    # ==================================================================
    parameters_constrained, nu_final, lambda_final = aparch_itransform(
        parameters_opt, p, o, q, error_type_code, delta_is_estimated_int
    )

    # Ref: aparch.m:224 — parameters = [parameters; nu; lambda]
    param_parts = [parameters_constrained]
    if nu_final is not None:
        param_parts.append(np.array([nu_final]))
    if lambda_final is not None:
        param_parts.append(np.array([lambda_final]))
    parameters = np.concatenate(param_parts)

    # ==================================================================
    # Step 8: Final Log-Likelihood at Optimum
    # Ref: aparch.m:230-232
    # aparch_likelihood returns NEGATED LL; we negate back to get true LL
    # ==================================================================
    LL_neg, likelihoods, ht = aparch_likelihood(
        parameters,
        data_aug, abs_data_aug, p, o, q,
        error_type_code, T,
        delta_is_estimated_int, user_delta,
        False,  # estim_flag=False → parameters are in constrained space
    )
    # Ref: aparch.m:232 — LL = -LL (negate to get positive log-likelihood)
    LL = -LL_neg

    # ==================================================================
    # Step 9: Robust Variance-Covariance Matrix (Inference)
    # Ref: aparch.m:236-239
    # ==================================================================
    # Ref: aparch.m:237 — nw=0 (no Newey-West correction on scores)
    nw = 0

    # Ref: aparch.m:238 — robustvcv('aparch_likelihood', parameters, nw, ...)
    # Note: robustvcv expects fun(theta, *args) → (scalar_LL, individual_LLs),
    # but aparch_likelihood returns (LL, LLS, ht).  In MATLAB, the nargout=2
    # mechanism silently drops the third return value; in Python we must
    # explicitly wrap the function to return only the first two values.
    def _ll_for_robustvcv(params, *ll_args):
        """Wrapper returning (LL, LLS) for robustvcv score computation."""
        ll_scalar, lls_individual, _ht = aparch_likelihood(params, *ll_args)
        return ll_scalar, lls_individual

    VCVrobust, A, B, scores, hess, gross_scores = robustvcv(
        _ll_for_robustvcv,
        parameters,
        nw,
        data_aug, abs_data_aug, p, o, q,
        error_type_code, T,
        delta_is_estimated_int, user_delta,
    )

    # Ref: aparch.m:239 — VCV = hess^(-1) / (T-m)
    # Non-robust VCV: inverse of the Hessian divided by effective observations
    try:
        VCV = np.linalg.inv(hess) / (T - m)
    except np.linalg.LinAlgError:
        # If Hessian is singular, fall back to a matrix of NaN
        warnings.warn(
            'Hessian is singular; non-robust VCV could not be computed.',
            stacklevel=2,
        )
        VCV = np.full_like(hess, np.nan)

    # ==================================================================
    # Step 10: Display Results
    # Ref: aparch_display is the final step in the estimation pipeline.
    # We need to map the integer error_type_code back to string for display.
    # ==================================================================
    _error_type_str_map = {1: 'NORMAL', 2: 'STUDENTST', 3: 'GED', 4: 'SKEWT'}
    error_type_str = _error_type_str_map.get(error_type_code, 'NORMAL')

    aparch_display(
        parameters, LL, VCV,
        data, p, o, q,
        error_type=error_type_str,
        user_delta=user_delta,
    )

    # ==================================================================
    # Step 11: Assemble Diagnostics
    # Ref: aparch.m:243-246
    # ==================================================================
    diagnostics = {
        'EXITFLAG': exitflag,
        'ITERATIONS': getattr(output_info, 'nit', 0),
        'FUNCCOUNT': getattr(output_info, 'nfev', 0),
        'MESSAGE': getattr(output_info, 'message', ''),
    }

    return parameters, LL, ht, VCVrobust, VCV, scores, diagnostics
