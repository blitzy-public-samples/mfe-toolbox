"""
Starting-value selection for EGARCH(P, O, Q) estimation.

Performs a grid search over the EGARCH parameter space and uses bounded
scalar optimisation (replacing MATLAB ``fminbnd``) to find the omega
parameter that minimises the normal-distribution negative log-likelihood
for each combination of alpha, gamma, and beta.  The combination with
the lowest negative log-likelihood is returned as the starting-value
vector.

When the caller supplies starting values, the function parses
distribution shape parameters (``nu``, ``lambda``) and truncates the
vector to the core EGARCH parameters.

Migrated from: ``univariate/egarch_starting_values.m`` — MFE Toolbox
Version 4.0  
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from mfe_toolbox.univariate.egarch_likelihood import egarch_likelihood
from mfe_toolbox.univariate.tarch import tarch

__all__ = ['egarch_starting_values']


def egarch_starting_values(
    startingvals: np.ndarray | None,
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
    back_cast: float,
    T: int,
) -> tuple[np.ndarray, float | None, float | None]:
    """Provide starting values for EGARCH(P, O, Q) estimation.

    When ``startingvals`` is ``None`` (or an empty array), a grid search
    is performed over pre-defined alpha, gamma, and beta values.  For each
    grid combination the omega intercept is optimised using
    :func:`scipy.optimize.minimize_scalar` (with ``method='bounded'``),
    which replaces MATLAB's ``fminbnd``.  The best parameter set (lowest
    negative log-likelihood) is selected as the starting-value vector.

    When ``startingvals`` is user-supplied, distribution shape parameters
    ``nu`` and ``lambda`` are parsed out and the core EGARCH parameter
    vector is truncated to length ``1 + p + o + q``.

    Parameters
    ----------
    startingvals : np.ndarray or None
        A user-supplied starting-value vector, or ``None`` to trigger the
        grid search.  If supplied, layout is::

            [omega, alpha_1, …, alpha_p, gamma_1, …, gamma_o,
             beta_1, …, beta_q, [nu, [lambda]]]

    epsilon : np.ndarray
        1-D array of mean-zero residuals (augmented with leading zeros
        for backcast initialisation).  Length must equal *T*.
    p : int
        Positive integer — number of symmetric ARCH lags.
    o : int
        Non-negative integer — number of asymmetric innovation lags.
    q : int
        Non-negative integer — number of GARCH variance lags.
    error_type : int
        Innovation distribution code:

        * 1 — Normal (Gaussian)
        * 2 — Standardized Student's t
        * 3 — Generalized Error Distribution (GED)
        * 4 — Hansen's Skewed Student's t
    back_cast : float
        Scalar used to initialise the EGARCH log-variance recursion.
        Typically ``log(unconditional variance)`` computed by the main
        EGARCH driver.
    T : int
        Total length of ``epsilon`` (including zero-padding).

    Returns
    -------
    startingvals : np.ndarray
        1-D array of length ``1 + p + o + q`` — the core EGARCH
        starting-value vector ``[omega, alpha…, gamma…, beta…]``.
    nu : float or None
        Distribution kurtosis/shape parameter.  ``None`` when not
        applicable (``error_type == 1``).
    lam : float or None
        Distribution asymmetry parameter.  ``None`` unless
        ``error_type == 4`` (Skewed t).

    Notes
    -----
    * The grid search always evaluates likelihoods under the Normal
      distribution (``error_type=1``) regardless of the requested
      distribution.  Distribution shape parameters are set to sensible
      defaults after the grid search completes.
    * Omega bounds are derived from the log of the sample variance of
      ``epsilon``, matching the MATLAB convention in
      ``egarch_starting_values.m:85-86``.
    * If ``omega_UB < omega_LB`` (i.e. log-variance is negative), the
      bounds are swapped so that the bounded minimiser operates on a
      valid interval.  Ref: egarch_starting_values.m:87-91.

    See Also
    --------
    egarch : Main EGARCH estimation driver.
    egarch_likelihood : Log-likelihood objective used inside the grid
        search.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.egarch_starting_values import (
    ...     egarch_starting_values,
    ... )
    >>> rng = np.random.default_rng(42)
    >>> eps = rng.standard_normal(500) * 0.01
    >>> m = 1
    >>> eps_aug = np.concatenate([np.zeros(m), eps])
    >>> T_aug = len(eps_aug)
    >>> bc = float(np.log(np.var(eps, ddof=1)))
    >>> sv, nu, lam = egarch_starting_values(
    ...     None, eps_aug, 1, 1, 1, 1, bc, T_aug
    ... )
    >>> sv.shape
    (4,)
    """
    # ------------------------------------------------------------------
    # Ensure inputs are well-formed
    # ------------------------------------------------------------------
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Determine whether to perform grid search
    # Ref: egarch_starting_values.m:43 — isempty(startingvals)
    # ------------------------------------------------------------------
    _user_supplied = (
        startingvals is not None
        and isinstance(startingvals, np.ndarray)
        and startingvals.size > 0
    )

    if not _user_supplied:
        # ==============================================================
        # Grid search path
        # Ref: egarch_starting_values.m:44-115
        # ==============================================================
        nu: float | None = None
        lam: float | None = None

        # Grid-search parameter grids
        # Ref: egarch_starting_values.m:50-54
        a = [0.01, 0.05, 0.2, 0.5]
        g = [-0.2, 0.01, 0.1]
        b = [0.01, 0.5, 0.9]

        la = len(a)
        lg = len(g)
        lb = len(b)

        total_combos = la * lg * lb

        # Pre-allocate output storage
        # Ref: egarch_starting_values.m:56-57
        output_parameters = np.zeros(
            (total_combos, 1 + p + o + q), dtype=np.float64
        )
        LLs = np.zeros(total_combos, dtype=np.float64)

        # Compute sample variance (MATLAB cov(vector) → sample variance)
        # Ref: egarch_starting_values.m:59 — covar=cov(data)
        # MATLAB cov of a vector returns scalar variance with N-1 normalization
        covar = float(np.var(epsilon, ddof=1))
        # Guard against zero or negative variance
        if covar <= 0.0:
            covar = 1e-6

        # Ref: egarch_starting_values.m:60 — back_cast=log(covar)
        # Compute local back_cast for the grid search to match MATLAB
        # behaviour (MATLAB starting_values computes its own back_cast)
        back_cast_grid = np.log(covar)

        # Upper bound for variance — egarch_likelihood.m:61
        # Ref: egarch_likelihood.m:61 — upper=10000*max(data.^2)
        max_eps_sq = float(np.max(epsilon ** 2))
        if max_eps_sq <= 0.0:
            max_eps_sq = 1e-6
        upper = 10000.0 * max_eps_sq

        # Options for minimize_scalar (replaces MATLAB optimset('fminbnd'))
        # Ref: egarch_starting_values.m:38-39 — options.TolX=1e-3
        fminbnd_opts: dict = {'xatol': 1e-3}

        # Ref: egarch_starting_values.m:61 — index=1  (MATLAB 1-based)
        index = 0  # Python 0-based

        for i in range(la):
            alpha = a[i]
            for j in range(lg):
                gamma = g[j]
                for k in range(lb):
                    # Build const_parameters vector for this grid point
                    # Ref: egarch_starting_values.m:67-81

                    # Ref: egarch_starting_values.m:67-70
                    if p > 0:
                        temp_alpha = alpha * np.ones(p, dtype=np.float64) / p
                    else:
                        temp_alpha = np.empty(0, dtype=np.float64)

                    # Ref: egarch_starting_values.m:71-74
                    if o > 0:
                        temp_gamma = gamma * np.ones(o, dtype=np.float64) / o
                    else:
                        temp_gamma = np.empty(0, dtype=np.float64)

                    # Ref: egarch_starting_values.m:75-80
                    if q > 0:
                        temp_beta = b[k] * np.ones(q, dtype=np.float64) / q
                    else:
                        temp_beta = np.empty(0, dtype=np.float64)

                    # Compute omega bounds
                    # Ref: egarch_starting_values.m:85-86
                    omega_UB = 1.1 * np.log(covar)
                    omega_LB = np.log(covar) * 0.0001

                    # Swap if UB < LB (happens when log(covar) < 0)
                    # Ref: egarch_starting_values.m:87-91
                    if omega_UB < omega_LB:
                        omega_UB, omega_LB = omega_LB, omega_UB

                    # Assemble const_parameters = [alpha; gamma; beta]
                    # Ref: egarch_starting_values.m:92-94
                    parts: list[np.ndarray] = []
                    if temp_alpha.size > 0:
                        parts.append(temp_alpha)
                    if temp_gamma.size > 0:
                        parts.append(temp_gamma)
                    if temp_beta.size > 0:
                        parts.append(temp_beta)

                    if len(parts) > 0:
                        const_parameters = np.concatenate(parts)
                    else:
                        const_parameters = np.empty(0, dtype=np.float64)

                    # Optimise omega using bounded scalar minimisation
                    # Ref: egarch_starting_values.m:96 —
                    #   fminbnd('egarch_likelihood', omega_LB, omega_UB,
                    #           options, data, p, o, q, 1, back_cast, T,
                    #           const_parameters)
                    #
                    # MATLAB egarch_likelihood mode 3:
                    #   parameters = [omega; const_parameters]
                    # Python equivalent: construct full parameter vector
                    # and call with estim_flag=False, error_type=1 (Normal)

                    # Capture const_parameters for the closure via default arg
                    # to avoid late-binding issues in the loop
                    def _make_objective(
                        _const_params: np.ndarray,
                        _epsilon: np.ndarray,
                        _p: int,
                        _o: int,
                        _q: int,
                        _bc: float,
                        _T: int,
                        _upper: float,
                    ):
                        """Create a closure for minimize_scalar objective."""
                        def _obj(omega_val: float) -> float:
                            full_params = np.concatenate(
                                [np.array([omega_val]), _const_params]
                            )
                            # Call with error_type=1 (Normal), estim_flag=False
                            ll_val, _, _ = egarch_likelihood(
                                full_params,
                                _epsilon,
                                _p,
                                _o,
                                _q,
                                1,       # Normal distribution for grid search
                                _bc,
                                _T,
                                _upper,
                                False,   # estim_flag=False (constrained params)
                            )
                            return ll_val
                        return _obj

                    objective_fn = _make_objective(
                        const_parameters,
                        epsilon,
                        p, o, q,
                        back_cast_grid,
                        T,
                        upper,
                    )

                    try:
                        result = minimize_scalar(
                            objective_fn,
                            bounds=(omega_LB, omega_UB),
                            method='bounded',
                            options=fminbnd_opts,
                        )
                        omega_opt = result.x
                        ll_opt = result.fun
                    except Exception:
                        # If optimisation fails, use midpoint and a large LL
                        omega_opt = 0.5 * (omega_LB + omega_UB)
                        ll_opt = 1e7

                    # Assemble full parameter vector [omega; const_parameters]
                    # Ref: egarch_starting_values.m:97 —
                    #   parameters=[omega; const_parameters]
                    full_parameters = np.concatenate(
                        [np.array([omega_opt]), const_parameters]
                    )

                    # Ref: egarch_starting_values.m:98-100
                    output_parameters[index, :] = full_parameters
                    LLs[index] = ll_opt
                    index += 1

        # Sort by negative log-likelihood (ascending = best first)
        # Ref: egarch_starting_values.m:104 — [LLs,index]=sort(LLs)
        sort_indices = np.argsort(LLs)
        LLs_sorted = LLs[sort_indices]

        # Ref: egarch_starting_values.m:105 —
        #   startingvals=output_parameters(index(1),:)'
        # MATLAB index(1) is the best → Python sort_indices[0]
        startingvals_out = output_parameters[sort_indices[0], :].copy()

        # Ref: egarch_starting_values.m:106 —
        #   output_parameters=output_parameters(index,:)
        # (Not returned in Python, but computed for completeness)

        # Set up distribution shape parameters if needed
        # Ref: egarch_starting_values.m:108-115
        if error_type == 2:
            # Student's t — default nu
            nu = 8.0
        elif error_type == 3:
            # GED — default nu
            nu = 1.9
        elif error_type == 4:
            # Skewed t — default nu and lambda
            nu = 8.0
            lam = -0.1

        return startingvals_out, nu, lam

    else:
        # ==============================================================
        # User-supplied starting values path
        # Ref: egarch_starting_values.m:116-127
        # ==============================================================
        sv = np.asarray(startingvals, dtype=np.float64).ravel()
        nu_out: float | None = None
        lam_out: float | None = None

        # Parse distribution shape parameters
        # Ref: egarch_starting_values.m:120-121
        # MATLAB 1-based: nu=startingvals(p+o+q+2)
        # Python 0-based: nu=sv[p+o+q+1]
        if error_type in (2, 3, 4):
            nu_out = float(sv[p + o + q + 1])

        # Ref: egarch_starting_values.m:123-124
        # MATLAB 1-based: lambda=startingvals(p+o+q+3)
        # Python 0-based: lam=sv[p+o+q+2]
        if error_type == 4:
            lam_out = float(sv[p + o + q + 2])

        # Truncate to core EGARCH parameters
        # Ref: egarch_starting_values.m:126
        # MATLAB: startingvals=startingvals(1:p+o+q+1)
        # Python: sv[:p+o+q+1]  (0-based inclusive end → exclusive slice)
        startingvals_out = sv[: p + o + q + 1].copy()

        return startingvals_out, nu_out, lam_out
