"""
Starting value computation for FIGARCH(Q,D,P) estimation.

Performs a grid search over the fractional differencing parameter d and
AR/MA ratios to find starting values that minimize the normal log-likelihood.
If user-supplied starting values are provided, they are parsed and returned
directly without grid search.

The grid search evaluates 27 candidate parameter vectors formed from
combinations of d in {0.2, 0.5, 0.7}, phi_ratio in {0.2, 0.5, 0.8}, and
beta_ratio in {0.1, 0.5, 0.9}.  For each candidate, omega is computed via
variance targeting using the ARCH(infinity) truncation weights:

    omega = sample_variance * (1 - sum(lambda_weights))

The candidate with the lowest normal negative log-likelihood is selected
as the starting value.  Default distribution shape parameters are appended
based on error_type: nu=8 for Student-t, nu=1.9 for GED, and nu=8,
lambda=-0.1 for Skewed-t.

Migrated from: ``univariate/figarch_starting_values.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.univariate.figarch : FIGARCH model driver.
mfe_toolbox.univariate.figarch_likelihood : FIGARCH log-likelihood.
mfe_toolbox.univariate.figarch_weights : ARCH(infinity) weight computation.
mfe_toolbox.univariate.figarch_transform : FIGARCH parameter transformation.
mfe_toolbox.univariate.figarch_itransform : FIGARCH inverse parameter transformation.

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
    startingvals: np.ndarray | None,
    epsilon: np.ndarray,
    p: int,
    q: int,
    error_type: int,
    truncLag: int,
    back_cast: float,
    T: int,
) -> tuple[np.ndarray, float | None, float | None]:
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
        The number of core parameters is ``2 + p + q``; nu and lambda
        follow immediately after.
    epsilon : np.ndarray
        T-element (or longer) array of mean-zero residuals.
    p : int
        0 or 1 indicating whether the autoregressive (phi) term is present.
    q : int
        0 or 1 indicating whether the moving-average (beta) term is present.
    error_type : int
        Error distribution type:

        - 1 — Normal
        - 2 — Student's t
        - 3 — Generalized Error Distribution (GED)
        - 4 — Hansen's Skewed Student's t
    truncLag : int
        Truncation lag length for the ARCH(infinity) representation.
    back_cast : float
        Backcast value used for pre-sample squared residuals. Typically
        the unconditional variance of the residuals.
    T : int
        Number of observations to use from ``epsilon``.

    Returns
    -------
    startingvals : np.ndarray
        Best starting parameter vector containing only core model
        parameters: ``[omega, (phi if p=1), d, (beta if q=1)]``.
        Length is ``2 + p + q``.
    nu : float or None
        Distribution shape parameter. Returns ``None`` for Normal
        (error_type=1).  Defaults: 8.0 for Student-t, 1.9 for GED,
        8.0 for Skewed-t (grid search path); parsed from user-supplied
        vector otherwise.
    lam : float or None
        Distribution skewness parameter. Returns ``None`` except for
        Skewed-t (error_type=4).  Default: -0.1 (grid search path);
        parsed from user-supplied vector otherwise.

    Raises
    ------
    ValueError
        If ``startingvals`` is provided but has insufficient length for
        the specified model and error_type.

    Notes
    -----
    Grid search parameters (Ref: figarch_starting_values.m:50-52):

    - d in {0.2, 0.5, 0.7}
    - phi_ratio in {0.2, 0.5, 0.8} — fraction of ``(1-d)/2`` upper bound
    - beta_ratio in {0.1, 0.5, 0.9} — fraction of ``(d+phi)`` upper bound

    For each combination, omega is computed via variance targeting
    (Ref: figarch_starting_values.m:80-82):

        omega = sample_variance * (1 - sum(arch_weights))

    Default distribution parameters (Ref: figarch_starting_values.m:104-112):

    - Student-t: nu = 8.0
    - GED: nu = 1.9
    - Skewed-t: nu = 8.0, lambda = -0.1

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
    >>> sv, nu, lam = figarch_starting_values(None, eps, 0, 0, 1, 100, 1.0, 500)
    >>> sv.shape[0] == 2  # omega and d
    True
    >>> nu is None and lam is None
    True
    """
    # ------------------------------------------------------------------
    # Input validation and conversion
    # ------------------------------------------------------------------
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()

    # Number of core model parameters: omega + phi(if p) + d + beta(if q)
    # Ref: figarch_starting_values.m:59 — outputParameters=zeros(ld, 2+p+q)
    num_core = 2 + p + q

    if startingvals is None:
        # ==============================================================
        # Grid search for starting values
        # Ref: figarch_starting_values.m:43-112
        # ==============================================================
        nu: float | None = None
        lam: float | None = None

        # Ref: figarch_starting_values.m:50-52 — Grid search values
        ds = [0.2, 0.5, 0.7]
        phi_ratio = [0.2, 0.5, 0.8]
        beta_ratio = [0.1, 0.5, 0.9]

        # Ref: figarch_starting_values.m:54 — sample variance for targeting
        # MATLAB cov(x) for a vector = var(x, ddof=1) in NumPy
        covar = np.var(epsilon[:T], ddof=1)

        # Build all parameter combinations
        # Ref: figarch_starting_values.m:56-86 — triple nested loop
        # MATLAB loop order: i over phiRatio, j over betaRatio, k over ds
        param_list: list[np.ndarray] = []
        for i_phi in range(len(phi_ratio)):
            for j_beta in range(len(beta_ratio)):
                for k_d in range(len(ds)):
                    d = ds[k_d]

                    # Ref: figarch_starting_values.m:65-68
                    # phi = (1-d)/2 * phiRatio(i) if p; else phi = 0
                    if p:
                        phi = (1.0 - d) / 2.0 * phi_ratio[i_phi]
                    else:
                        phi = 0.0

                    # Ref: figarch_starting_values.m:70
                    beta = (d + phi) * beta_ratio[j_beta]

                    # Ref: figarch_starting_values.m:71-79 — build temp vector
                    # containing the weight parameters in the order figarch_weights expects
                    if p and q:
                        temp = np.array([phi, d, beta])
                    elif p:
                        temp = np.array([phi, d])
                    elif q:
                        temp = np.array([d, beta])
                    else:
                        temp = np.array([d])

                    # Ref: figarch_starting_values.m:80 — compute ARCH(infinity) weights
                    lam_weights = figarch_weights(temp, p, q, truncLag)

                    # Ref: figarch_starting_values.m:81 — variance targeting for omega
                    # omega = covar * (1 - sum(lambda))
                    omega = covar * (1.0 - np.sum(lam_weights))

                    # Ref: figarch_starting_values.m:82 — [omega, temp]
                    row = np.concatenate((np.array([omega]), temp))
                    param_list.append(row)

        # Ref: figarch_starting_values.m:88 — unique(outputParameters, 'rows')
        all_params = np.array(param_list)
        all_params = np.unique(all_params, axis=0)

        # Ref: figarch_starting_values.m:90 — LLs=zeros(size(outputParameters,1),1)
        num_candidates = all_params.shape[0]
        LLs = np.zeros(num_candidates)

        # Ref: figarch_starting_values.m:93-95 — evaluate likelihood at each candidate
        # Use error_type=1 (Normal) for grid search, estim_flag=False
        for i in range(num_candidates):
            try:
                # Ref: figarch_starting_values.m:94
                # MATLAB: figarch_likelihood(outputParameters(i,:),p,q,epsilon,epsilon2,truncLag,1,false)
                # Python API: figarch_likelihood(parameters, epsilon, p, q, error_type, truncLag, back_cast, T, estim_flag)
                ll_val, _, _ = figarch_likelihood(
                    all_params[i, :],
                    epsilon,
                    p,
                    q,
                    1,          # error_type = 1 (Normal) for grid search
                    truncLag,
                    back_cast,
                    T,
                    False,      # estim_flag = False — parameters already in constrained space
                )
                LLs[i] = ll_val
            except Exception:
                # If likelihood evaluation fails for a candidate (e.g., negative
                # variance), assign a very large value so it won't be selected
                LLs[i] = np.inf

        # Ref: figarch_starting_values.m:99 — sort by likelihood (ascending)
        # Lower (negative) LL is better since we minimize -1*LL
        sort_idx = np.argsort(LLs)

        # Ref: figarch_starting_values.m:101 — best starting values (first row)
        startingvals_out = all_params[sort_idx[0], :].copy()

        # Ref: figarch_starting_values.m:104-112 — default distribution parameters
        if error_type == 2:
            nu = 8.0
        elif error_type == 3:
            nu = 1.9
        elif error_type == 4:
            nu = 8.0
            lam = -0.1

        return startingvals_out, nu, lam

    else:
        # ==============================================================
        # User-supplied starting values — parse only
        # Ref: figarch_starting_values.m:113-124
        # ==============================================================
        startingvals_arr = np.asarray(startingvals, dtype=np.float64).ravel()
        nu = None
        lam = None

        # Validate minimum length for user-supplied vector
        # Core requires num_core = 2 + p + q elements; distribution params follow
        min_len = num_core
        if error_type in (2, 3):
            min_len = num_core + 1  # + nu
        elif error_type == 4:
            min_len = num_core + 2  # + nu + lambda
        if startingvals_arr.shape[0] < min_len:
            raise ValueError(
                f"startingvals must have at least {min_len} elements for "
                f"p={p}, q={q}, error_type={error_type}. "
                f"Got {startingvals_arr.shape[0]}."
            )

        # Ref: figarch_starting_values.m:117-119
        # MATLAB: nu = startingvals(p+q+2)
        # The core parameter vector has 2+p+q elements (omega + phi(if p) + d + beta(if q)).
        # In 0-based Python indexing, nu is at index num_core = 2+p+q.
        # Ref: figarch_starting_values.m:118 — Python 0-based index correction
        if error_type in (2, 3, 4):
            nu = float(startingvals_arr[num_core])

        # Ref: figarch_starting_values.m:120-121
        # MATLAB: lambda = startingvals(p+q+3)
        # Python 0-based: index num_core + 1 = 2+p+q+1
        if error_type == 4:
            lam = float(startingvals_arr[num_core + 1])

        # Ref: figarch_starting_values.m:123 — keep only core parameters
        # Core params are the first num_core = 2+p+q elements
        startingvals_out = startingvals_arr[:num_core].copy()

        return startingvals_out, nu, lam
