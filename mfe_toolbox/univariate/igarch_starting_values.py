"""
IGARCH(P,Q) starting value computation via grid search.

Provides starting values for IGARCH(P,Q) estimation with unit-root
constraint enforcement.  When no user-supplied starting values are provided,
performs a grid search over candidate alpha values ``[0.05, 0.1, 0.2]``,
distributes each evenly across *p* lags, computes beta to satisfy the
unit-root constraint ``sum(alpha) + sum(beta) = 1``, and selects the
parameter set yielding the lowest negative log-likelihood under the
Normal distribution.

When user-supplied starting values are provided, parses the distribution
shape parameters (``nu``, ``lambda``) from the parameter vector based on
the specified error distribution type.

Migrated from: univariate/igarch_starting_values.m — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1, Date: 7/12/2009

See Also
--------
igarch : IGARCH estimation driver.
igarch_likelihood : IGARCH log-likelihood evaluation.
igarch_core : Numba JIT conditional variance recursion.
"""

import numpy as np

from mfe_toolbox.univariate.igarch_likelihood import igarch_likelihood


def igarch_starting_values(
    startingvals: np.ndarray | None,
    epsilon: np.ndarray,
    p: int,
    q: int,
    igarch_type: int,
    error_type: int,
    constant: int,
) -> tuple[np.ndarray, float | None, float | None]:
    """Compute starting values for IGARCH(P,Q) estimation.

    Performs a grid search to find reasonable starting values for IGARCH(P,Q)
    estimation when no user-supplied values are provided.  The unit-root
    constraint ``sum(alpha) + sum(beta) = 1`` is enforced throughout the
    search by computing ``beta = 1 - sum(alpha)`` and distributing the free
    betas evenly across ``q - 1`` lags (the last beta is implied by the
    constraint inside ``igarch_core``).

    If starting values are user-supplied (non-empty), reformats and parses
    them to extract distribution parameters (``nu``, ``lambda``) based on
    the error distribution type.

    Parameters
    ----------
    startingvals : np.ndarray or None
        User-supplied starting values, or ``None`` / empty array to trigger
        grid search.  When provided, the vector layout depends on
        ``error_type``:

        * ``error_type == 1``:
          ``[omega, alpha_1 … alpha_p, beta_1 … beta_{q-1}]``
        * ``error_type == 2`` or ``error_type == 3``:
          ``[…, nu]``
        * ``error_type == 4``:
          ``[…, nu, lambda]``

    epsilon : np.ndarray
        Column vector (or 1-D array) of mean-zero residuals.
        Ref: igarch_starting_values.m:12 — ``EPSILON — A column of mean
        zero data``
    p : int
        Positive integer — number of ARCH (alpha) lags.
    q : int
        Non-negative integer — number of GARCH (beta) lags (0 for pure
        ARCH).
    igarch_type : int
        Variance process formulation:

        * 1 — absolute-value recursion (IAVARCH):
          ``fepsilon = |epsilon|``
        * 2 — squared recursion (standard IGARCH):
          ``fepsilon = epsilon ** 2``
    error_type : int
        Innovation distribution:

        * 1 — Normal (Gaussian)
        * 2 — Standardised Student's t
        * 3 — Generalised Error Distribution (GED)
        * 4 — Hansen's Skewed t
    constant : int
        * 1 — include an intercept (omega) in the variance equation.
        * 0 — no intercept.

    Returns
    -------
    startingvals_out : np.ndarray
        Parameter vector of length ``constant + p + max(q - 1, 0)``.
    nu : float or None
        Distribution kurtosis parameter.  ``None`` when not applicable
        (``error_type == 1``).
    lambda_param : float or None
        Distribution asymmetry parameter.  ``None`` when not applicable
        (``error_type != 4``).

    Notes
    -----
    The grid search evaluates candidates under the Normal distribution
    (``error_type=1``) regardless of the actual ``error_type`` specified,
    following the MATLAB convention in ``igarch_starting_values.m:100``.

    The unit-root constraint is enforced by computing::

        beta = 1 - sum(alpha)

    and distributing the remaining free betas evenly across ``q - 1`` lags
    (the last beta is implied by the constraint in ``igarch_core``).

    Ref: igarch_starting_values.m:1-131

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.igarch_starting_values import (
    ...     igarch_starting_values,
    ... )
    >>> rng = np.random.default_rng(42)
    >>> eps = rng.standard_normal(500)
    >>> sv, nu, lam = igarch_starting_values(None, eps, 1, 1, 2, 1, 1)
    >>> sv.shape[0] == 1 + 1  # constant + p (q-1=0 free betas for q=1)
    True
    >>> nu is None  # Normal dist → no shape parameter
    True
    """
    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------
    epsilon = np.asarray(epsilon, dtype=np.float64).flatten()
    T = len(epsilon)  # Ref: igarch_starting_values.m:6 — T is length of EPSILON

    # Determine whether a grid search is required.
    # Ref: igarch_starting_values.m:50 — if isempty(startingvals)
    _do_grid_search: bool
    if startingvals is None:
        _do_grid_search = True
    else:
        startingvals = np.asarray(startingvals, dtype=np.float64).flatten()
        _do_grid_search = startingvals.size == 0

    # ==================================================================
    # BRANCH 1: Grid search — no user-supplied starting values
    # Ref: igarch_starting_values.m:50-118
    # ==================================================================
    if _do_grid_search:
        nu: float | None = None
        lambda_param: float | None = None

        # Candidate total-alpha values for the grid
        # Ref: igarch_starting_values.m:57 — a=[.05 .1 .2]
        a = np.array([0.05, 0.1, 0.2])
        la = len(a)

        # Number of free parameters per candidate:
        #   constant (0 or 1) + p (alpha) + max(q-1, 0) (free betas)
        # The last beta is implied by the unit-root constraint in igarch_core.
        # Ref: igarch_starting_values.m:63 — outputParameters=zeros(la*lb,1+p+q-1)
        n_params = constant + p + max(q - 1, 0)

        # Pre-allocate output arrays
        output_parameters = np.zeros((la, n_params))
        lls = np.zeros(la)

        # ------------------------------------------------------------------
        # Compute adjustment factor and back-cast value based on igarch_type.
        # Ref: igarch_starting_values.m:67-73
        # ------------------------------------------------------------------
        if igarch_type == 1:
            # Ref: igarch_starting_values.m:68 — adjFactor=sqrt(2/pi)
            adj_factor = np.sqrt(2.0 / np.pi)
            # Ref: igarch_starting_values.m:69 — backCast=mean(abs(epsilon))
            back_cast = float(np.mean(np.abs(epsilon)))
        else:
            # Ref: igarch_starting_values.m:71 — adjFactor=1
            adj_factor = 1.0
            # Ref: igarch_starting_values.m:72 — backCast=cov(epsilon)
            # MATLAB cov(x) for a vector returns unbiased sample variance (ddof=1).
            back_cast = float(np.var(epsilon, ddof=1))

        # ------------------------------------------------------------------
        # Iterate over candidate alpha values and evaluate likelihood.
        # Ref: igarch_starting_values.m:78-103
        # ------------------------------------------------------------------
        index = 0
        for i in range(la):
            # Ref: igarch_starting_values.m:80 — alpha=a(i)
            alpha = a[i]

            # Distribute alpha evenly across p lags.
            # Ref: igarch_starting_values.m:81 — tempAlpha=alpha*ones(p,1)/p
            temp_alpha = alpha * np.ones(p) / p

            # Beta satisfies the unit-root constraint: sum(alpha)+sum(beta)=1
            # Ref: igarch_starting_values.m:83 — beta=1-sum(tempAlpha)
            beta = 1.0 - np.sum(temp_alpha)

            # Build omega component.
            # Ref: igarch_starting_values.m:85-89
            if constant:
                # Small fraction of back-cast as the intercept.
                # Ref: igarch_starting_values.m:86 — omega=backCast*.01*adjFactor
                omega = np.array([back_cast * 0.01 * adj_factor])
            else:
                # Ref: igarch_starting_values.m:88 — omega = []
                omega = np.array([], dtype=np.float64)

            # Assemble the full parameter vector.
            # Ref: igarch_starting_values.m:92-96
            if q == 0:
                # Pure ARCH — no beta terms in the parameter vector.
                # Ref: igarch_starting_values.m:93 — parameters=[omega; tempAlpha]
                parameters = np.concatenate([omega, temp_alpha])
            else:
                # GARCH — include q-1 free beta terms; the last beta is implied
                # by the unit-root constraint inside igarch_core.
                # Ref: igarch_starting_values.m:95 — parameters=[omega; tempAlpha; beta*ones(q-1,1)/q]
                if q > 1:
                    free_betas = beta * np.ones(q - 1) / q
                else:
                    # q == 1: the single beta is entirely implied, no free betas.
                    free_betas = np.array([], dtype=np.float64)
                parameters = np.concatenate([omega, temp_alpha, free_betas])

            # Store candidate parameters.
            # Ref: igarch_starting_values.m:98 — outputParameters(index,:)=parameters'
            output_parameters[index, :] = parameters

            # Evaluate the negative log-likelihood under the Normal distribution.
            # The grid search always uses error_type=1 (Normal) regardless of the
            # actual error_type, matching the MATLAB convention.
            # Ref: igarch_starting_values.m:100 —
            #   LLs(index)=igarch_likelihood(parameters, epsilon, fepsilon,
            #       p, q, 1, igarchType, constant, backCast, T, false)
            # Python igarch_likelihood computes fepsilon internally from epsilon.
            ll_val, _, _ = igarch_likelihood(
                parameters,
                epsilon,
                p,
                q,
                igarch_type,
                1,           # error_type=1 (Normal) for grid search
                back_cast,
                T,
                constant,
                False,       # estim_flag=False — parameters already in constrained space
            )
            lls[index] = ll_val

            # Ref: igarch_starting_values.m:102 — index=index+1
            index += 1

        # ------------------------------------------------------------------
        # Sort candidates by negative log-likelihood (ascending).
        # The best candidate has the *lowest* value because igarch_likelihood
        # returns the *negative* log-likelihood (i.e. the minimisation
        # objective).
        # Ref: igarch_starting_values.m:105 — [LLs,index]=sort(LLs)
        # ------------------------------------------------------------------
        sorted_indices = np.argsort(lls)
        lls = lls[sorted_indices]

        # Select the best (lowest -LL) candidate as starting values.
        # Ref: igarch_starting_values.m:107 — startingvals=outputParameters(index(1),:)'
        startingvals_out = output_parameters[sorted_indices[0], :].copy()

        # Re-order output_parameters to match sorted LLs.
        # Ref: igarch_starting_values.m:109 — outputParameters=outputParameters(index,:)
        output_parameters = output_parameters[sorted_indices, :]

        # ------------------------------------------------------------------
        # Assign default distribution shape parameters for non-Normal
        # error distributions.
        # Ref: igarch_starting_values.m:111-118
        # ------------------------------------------------------------------
        if error_type == 2:
            # Student's t: default degrees-of-freedom = 8
            nu = 8.0
        elif error_type == 3:
            # GED: default shape parameter = 1.9
            nu = 1.9
        elif error_type == 4:
            # Skewed t: default nu=8, lambda=-0.1
            nu = 8.0
            lambda_param = -0.1

        return startingvals_out, nu, lambda_param

    # ==================================================================
    # BRANCH 2: User-supplied starting values — parse distribution params
    # Ref: igarch_starting_values.m:119-130
    # ==================================================================
    else:
        sv = np.asarray(startingvals, dtype=np.float64).flatten()
        nu: float | None = None
        lambda_param: float | None = None

        # Extract distribution parameters from the tail of the vector.
        # Ref: igarch_starting_values.m:123-128
        if error_type in (2, 3, 4):
            # Ref: igarch_starting_values.m:124 — nu=startingvals(constant+p+q)
            # MATLAB 1-indexed → Python 0-indexed: index = constant + p + q - 1
            nu = float(sv[constant + p + q - 1])

        if error_type == 4:
            # Ref: igarch_starting_values.m:127 — lambda=startingvals(constant+p+q+1)
            # MATLAB 1-indexed → Python 0-indexed: index = constant + p + q
            lambda_param = float(sv[constant + p + q])

        # Trim to variance parameters only (remove nu, lambda from tail).
        # Ref: igarch_starting_values.m:129 — startingvals=startingvals(1:constant+p+q-1)
        # Python 0-indexed slice: [:constant+p+q-1]
        startingvals_out = sv[: constant + p + q - 1].copy()

        return startingvals_out, nu, lambda_param
