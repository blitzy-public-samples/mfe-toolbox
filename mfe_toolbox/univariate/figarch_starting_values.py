"""
Starting value computation for FIGARCH(Q,D,P) estimation.

Performs a grid search over the fractional differencing parameter d and
AR/MA ratios to find starting values that minimize the normal log-likelihood.
If user-supplied starting values are provided, they are parsed and returned
directly without grid search.

Migrated from: ``univariate/figarch_starting_values.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.univariate.figarch : FIGARCH model driver.
mfe_toolbox.univariate.figarch_likelihood : FIGARCH log-likelihood.
mfe_toolbox.univariate.figarch_weights : ARCH(infinity) weight computation.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/12/2009
"""

import numpy as np

from mfe_toolbox.univariate.figarch_likelihood import figarch_likelihood
from mfe_toolbox.univariate.figarch_weights import figarch_weights


def figarch_starting_values(
    startingvals,
    epsilon: np.ndarray,
    epsilon2: np.ndarray,
    p: int,
    q: int,
    error_type: int,
    trunc_lag: int,
) -> tuple:
    """
    Compute or parse starting values for FIGARCH(Q,D,P) estimation.

    When ``startingvals`` is ``None``, performs a grid search over the
    fractional differencing parameter d and, if applicable, the phi and
    beta ratio parameters.  The grid evaluates the normal log-likelihood
    at each candidate and returns the parameter vector with the lowest
    negative log-likelihood.

    When ``startingvals`` is provided, parses the vector to extract core
    model parameters and distribution parameters (nu, lambda).

    Parameters
    ----------
    startingvals : np.ndarray or None
        User-supplied starting values, or ``None`` to trigger grid search.
        When provided, the layout is:
        ``[omega, (phi if p=1), d, (beta if q=1), (nu), (lambda)]``.
    epsilon : np.ndarray
        T-element array of mean-zero residuals.
    epsilon2 : np.ndarray
        (trunc_lag + T)-element array of squared residuals augmented with
        backcast values.
    p : int
        0 or 1 indicating whether the autoregressive (phi) term is present.
    q : int
        0 or 1 indicating whether the moving-average (beta) term is present.
    error_type : int
        Error distribution type: 1=Normal, 2=Student-t, 3=GED, 4=Skewed-t.
    trunc_lag : int
        Truncation lag length for the ARCH(infinity) representation.

    Returns
    -------
    tuple
        - ``startingvals`` (np.ndarray): Best starting parameter vector
          (core parameters only, excluding nu and lambda).
        - ``nu`` (float or None): Distribution shape parameter, or ``None``
          if not applicable.
        - ``lam`` (float or None): Distribution skewness parameter, or
          ``None`` if not applicable (only used for Skewed-t).
        - ``LLs`` (np.ndarray): Log-likelihood values for each grid search
          candidate (empty array if user-supplied).
        - ``ordered_parameters`` (np.ndarray): Grid search parameter matrix
          sorted by log-likelihood (empty array if user-supplied).

    Notes
    -----
    Ref: figarch_starting_values.m:43-112 — Grid search logic.

    Grid search parameters:
    - d in {0.2, 0.5, 0.7}
    - phi_ratio in {0.2, 0.5, 0.8} (fraction of (1-d)/2 upper bound)
    - beta_ratio in {0.1, 0.5, 0.9} (fraction of (d+phi) upper bound)

    For each combination, omega is computed via variance targeting:
    ``omega = sample_variance * (1 - sum(arch_weights))``

    Ref: figarch_starting_values.m:104-112 — Default distribution parameters:
    Student-t: nu=8, GED: nu=1.9, Skewed-t: nu=8, lambda=-0.1.
    """
    # ----------------------------------------------------------------
    # Initialize outputs
    # Ref: figarch_starting_values.m:39-40
    # ----------------------------------------------------------------
    LLs = np.array([])
    ordered_parameters = np.array([])

    if startingvals is None:
        # ============================================================
        # Grid search for starting values
        # Ref: figarch_starting_values.m:43-112
        # ============================================================
        nu = None
        lam = None

        # Ref: figarch_starting_values.m:50-52 — Grid search values
        ds = [0.2, 0.5, 0.7]
        phi_ratio = [0.2, 0.5, 0.8]
        beta_ratio = [0.1, 0.5, 0.9]

        # Ref: figarch_starting_values.m:54 — sample variance for targeting
        covar = np.var(epsilon, ddof=1)

        # Build all parameter combinations
        # Ref: figarch_starting_values.m:61-86
        param_list = []
        for i_phi in range(len(phi_ratio)):
            for j_beta in range(len(beta_ratio)):
                for k_d in range(len(ds)):
                    d = ds[k_d]

                    # Ref: figarch_starting_values.m:65-68
                    if p:
                        phi = (1.0 - d) / 2.0 * phi_ratio[i_phi]
                    else:
                        phi = 0.0

                    # Ref: figarch_starting_values.m:70
                    beta = (d + phi) * beta_ratio[j_beta]

                    # Ref: figarch_starting_values.m:71-79 — build temp vector
                    if p and q:
                        temp = np.array([phi, d, beta])
                    elif p:
                        temp = np.array([phi, d])
                    elif q:
                        temp = np.array([d, beta])
                    else:
                        temp = np.array([d])

                    # Ref: figarch_starting_values.m:80-82 — compute omega
                    # via variance targeting
                    lam_weights = figarch_weights(temp, p, q, trunc_lag)
                    omega = covar * (1.0 - np.sum(lam_weights))

                    # Ref: figarch_starting_values.m:82
                    row = np.concatenate([np.array([omega]), temp])
                    param_list.append(row)

        # Ref: figarch_starting_values.m:88 — unique rows
        all_params = np.array(param_list)
        all_params = np.unique(all_params, axis=0)

        # Ref: figarch_starting_values.m:90-95 — evaluate likelihood at each
        num_candidates = all_params.shape[0]
        LLs = np.zeros(num_candidates)

        for i in range(num_candidates):
            try:
                # Ref: figarch_starting_values.m:94
                # Call with error_type=1 (Normal) for grid search
                ll_val, _, _ = figarch_likelihood(
                    all_params[i, :], p, q, epsilon, epsilon2,
                    trunc_lag, 1, False
                )
                LLs[i] = ll_val
            except Exception:
                # If likelihood evaluation fails for a candidate, assign
                # a very large value so it won't be selected
                LLs[i] = np.inf

        # Ref: figarch_starting_values.m:99-103 — sort by likelihood
        sort_idx = np.argsort(LLs)
        LLs = LLs[sort_idx]
        ordered_parameters = all_params[sort_idx, :]

        # Ref: figarch_starting_values.m:101 — best starting values
        startingvals = ordered_parameters[0, :]

        # Ref: figarch_starting_values.m:104-112 — default distribution params
        if error_type == 2:
            nu = 8.0
        elif error_type == 3:
            nu = 1.9
        elif error_type == 4:
            nu = 8.0
            lam = -0.1

    else:
        # ============================================================
        # User-supplied starting values — parse only
        # Ref: figarch_starting_values.m:113-124
        # ============================================================
        startingvals = np.asarray(startingvals, dtype=np.float64).ravel()
        nu = None
        lam = None

        # Ref: figarch_starting_values.m:117-119
        if error_type in (2, 3, 4):
            # Ref: figarch_starting_values.m:118 — nu = startingvals(p+q+2)
            # MATLAB 1-indexed → Python 0-indexed: index p+q+1 → p+q+2-1
            nu = startingvals[p + q + 1]

        # Ref: figarch_starting_values.m:120-121
        if error_type == 4:
            # Ref: figarch_starting_values.m:121 — lambda = startingvals(p+q+3)
            lam = startingvals[p + q + 2]

        # Ref: figarch_starting_values.m:123 — keep only core params
        startingvals = startingvals[:p + q + 1]

    return startingvals, nu, lam, LLs, ordered_parameters
