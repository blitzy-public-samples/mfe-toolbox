"""
TARCH/GJR-GARCH starting value computation via grid search.

Provides initial parameter estimates for TARCH(P,O,Q) estimation by
evaluating :func:`tarch_likelihood` over a grid of plausible
alpha / gamma / beta combinations, using variance targeting to set the
intercept (omega).

When user-supplied starting values are provided the function parses
distribution shape parameters (nu, lambda) from the tail of the vector
and returns the core GARCH parameter sub-vector.

Migrated from: ``univariate/tarch_starting_values.m`` — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration.
"""

from __future__ import annotations

import numpy as np

from mfe_toolbox.univariate.tarch_likelihood import tarch_likelihood

__all__ = ['tarch_starting_values']


def tarch_starting_values(
    startingvals: np.ndarray | None,
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    tarch_type: int,
    error_type: int,
    back_cast: float,
    T: int,
) -> tuple[np.ndarray, float | None, float | None]:
    """Compute starting values for TARCH(P, O, Q) estimation.

    Performs a grid search over plausible alpha, gamma, and beta parameter
    combinations, evaluating :func:`tarch_likelihood` at each point using
    the Normal distribution (``error_type=1``).  The best parameter vector
    (lowest negative log-likelihood) is returned as the starting values.

    If starting values are user-supplied (non-empty), they are parsed to
    extract distribution shape parameters *nu* and *lambda* based on
    ``error_type``.

    Parameters
    ----------
    startingvals : np.ndarray or None
        User-supplied starting values vector, or ``None`` to trigger a
        grid search.  When provided and ``error_type > 1``, the vector
        must include distribution parameters at positions
        ``[p+o+q+1]`` (nu) and optionally ``[p+o+q+2]`` (lambda for
        Skewed t).
    epsilon : np.ndarray
        1-D array of mean-zero residuals (raw data, **not** padded with
        backcast values).
    p : int
        Number of symmetric ARCH lags (>= 0).
    o : int
        Number of asymmetric / threshold lags (>= 0).
    q : int
        Number of GARCH (lagged variance) terms (>= 0).
    tarch_type : int
        Variance-process type selector:

        * 1 — absolute-value / standard-deviation model
        * 2 — squared-return / variance model (standard case)
    error_type : int
        Innovation distribution type:

        * 1 — Normal (Gaussian)
        * 2 — Standardized Student's t
        * 3 — Generalized Error Distribution (GED)
        * 4 — Hansen's Skewed Student's t
    back_cast : float
        Scalar used to initialise the first *m* entries of the
        conditional variance recursion.  Precomputed by the caller.

        * ``tarch_type == 1``: typically ``mean(|epsilon|)``
        * ``tarch_type == 2``: typically ``var(epsilon)``
    T : int
        Number of observations in *epsilon*.

    Returns
    -------
    startingvals : np.ndarray
        1-D parameter vector of length ``1 + p + o + q``::

            [omega, alpha_1, …, alpha_p,
             gamma_1, …, gamma_o,
             beta_1,  …, beta_q]

    nu : float or None
        Distribution kurtosis parameter.  ``None`` when
        ``error_type == 1``.  Default 8.0 for Student's t / Skewed t,
        1.9 for GED.
    lam : float or None
        Distribution asymmetry parameter.  ``None`` unless
        ``error_type == 4``.  Default −0.1 for Skewed t.

    Notes
    -----
    * The grid covers alpha ∈ {0.05, 0.1, 0.2},
      gamma ∈ {0.01, 0.05, 0.2, −alpha/2},
      and total persistence (alpha + 0.5·gamma + beta) ∈
      {0.5, 0.8, 0.9, 0.95, 0.99}, giving 60 candidate vectors.
    * Omega is set via variance targeting so that the unconditional
      measure matches the sample estimate.
    * The grid search always evaluates the Normal log-likelihood
      (``error_type=1``) for speed and stability.
    * Distribution shape parameters are set to conventional defaults
      after the grid search completes.
    * Numerical parity with MATLAB within ±1e-6.
    * Ref: tarch_starting_values.m — MFE Toolbox Version 4.0.

    See Also
    --------
    tarch : Main TARCH estimation driver.
    tarch_likelihood : Log-likelihood evaluation used in grid search.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.tarch_starting_values import tarch_starting_values
    >>> rng = np.random.default_rng(42)
    >>> eps = rng.standard_normal(500)
    >>> sv, nu, lam = tarch_starting_values(
    ...     None, eps, 1, 1, 1, 2, 1, float(np.var(eps, ddof=1)), 500
    ... )
    >>> sv.shape
    (4,)
    >>> nu is None  # Normal distribution — no shape parameter
    True
    """
    # ------------------------------------------------------------------ #
    #  Input conditioning                                                 #
    # ------------------------------------------------------------------ #
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()

    # Ref: tarch_starting_values.m:59 — m = max(p, o, q)
    m: int = max(p, o, q)

    # ------------------------------------------------------------------ #
    #  Branch: grid search vs. user-supplied starting values              #
    # ------------------------------------------------------------------ #
    # Ref: tarch_starting_values.m:47 — if isempty(startingvals)
    _is_empty = (
        startingvals is None
        or (isinstance(startingvals, np.ndarray) and startingvals.size == 0)
    )

    if _is_empty:
        # ============================================================== #
        #  GRID SEARCH                                                    #
        # ============================================================== #
        # Ref: tarch_starting_values.m:48-49
        nu: float | None = None
        lam: float | None = None

        # -------------------------------------------------------------- #
        #  Construct padded arrays for tarch_likelihood                   #
        # -------------------------------------------------------------- #
        # tarch_likelihood expects data/fdata/fIdata prepended with
        # m = max(p,o,q) backcast padding values.
        # Ref: tarch_likelihood.py docstring — data is prepended with m values.

        if tarch_type == 1:
            # Ref: tarch_starting_values.m:67 — adj_factor = sqrt(2/pi)
            adj_factor: float = np.sqrt(2.0 / np.pi)
            fdata_raw = np.abs(epsilon)
        else:
            # Ref: tarch_starting_values.m:70 — adj_factor = 1
            adj_factor = 1.0
            fdata_raw = epsilon ** 2

        # Ref: tarch_starting_values.m:14 — fIdata = fdata .* (data < 0)
        fIdata_raw = fdata_raw * (epsilon < 0).astype(np.float64)

        # Build padded arrays — length T_padded = T + m
        data_padded = np.concatenate(
            [np.zeros(m, dtype=np.float64), epsilon]
        )
        fdata_padded = np.concatenate(
            [np.full(m, back_cast, dtype=np.float64), fdata_raw]
        )
        fIdata_padded = np.concatenate(
            [np.full(m, 0.5 * back_cast, dtype=np.float64), fIdata_raw]
        )
        T_padded: int = T + m

        # Ref: tarch_starting_values.m:73 — covar = cov(data)
        # MATLAB cov(x) for a column vector returns scalar sample variance (N-1 denom)
        covar: float = float(np.var(epsilon, ddof=1))

        # -------------------------------------------------------------- #
        #  Grid search parameter space                                    #
        # -------------------------------------------------------------- #
        # Ref: tarch_starting_values.m:54-59
        a = [0.05, 0.1, 0.2]
        la_len: int = len(a)                       # 3
        g = [0.01, 0.05, 0.2]
        lg_len: int = len(g) + 1                   # 4 (includes -alpha/2)
        agb = [0.5, 0.8, 0.9, 0.95, 0.99]
        lb_len: int = len(agb)                     # 5

        total_combinations: int = la_len * lb_len * lg_len   # 60

        # Ref: tarch_starting_values.m:62-63
        output_parameters = np.zeros(
            (total_combinations, 1 + p + o + q), dtype=np.float64
        )
        LLs = np.zeros(total_combinations, dtype=np.float64)

        # Ref: tarch_starting_values.m:76 — index = 1 (MATLAB 1-based)
        index: int = 0

        for i in range(la_len):
            # Ref: tarch_starting_values.m:78 — alpha = a(i)
            alpha: float = a[i]

            for j in range(lg_len):
                # Ref: tarch_starting_values.m:81-87
                if j == lg_len - 1:
                    # Ref: tarch_starting_values.m:83 — last gamma value: -alpha/2
                    gamma: float = -alpha / 2.0
                else:
                    gamma = g[j]

                for k in range(lb_len):
                    # Ref: tarch_starting_values.m:92 — temp_alpha = alpha*ones(p,1)/p
                    if p > 0:
                        temp_alpha = alpha * np.ones(p, dtype=np.float64) / p
                    else:
                        temp_alpha = np.empty(0, dtype=np.float64)

                    # Ref: tarch_starting_values.m:94-103 — Build temp_gamma
                    if o > 0:
                        temp_gamma = gamma * np.ones(o, dtype=np.float64) / o
                        for n in range(o):
                            if n < p:
                                # Ref: tarch_starting_values.m:98
                                # MATLAB 1-based n<=p  →  Python 0-based n<p
                                temp_gamma[n] = np.maximum(
                                    temp_gamma[n], -alpha / (2.0 * p)
                                )
                            else:
                                # Ref: tarch_starting_values.m:100
                                temp_gamma[n] = 0.0
                    else:
                        temp_gamma = np.empty(0, dtype=np.float64)

                    # Ref: tarch_starting_values.m:105
                    # beta = agb(k) - sum(temp_alpha) - 0.5*sum(temp_gamma)
                    beta: float = (
                        agb[k]
                        - np.sum(temp_alpha)
                        - 0.5 * np.sum(temp_gamma)
                    )

                    # -------------------------------------------------- #
                    #  Omega via variance targeting                       #
                    # -------------------------------------------------- #
                    if tarch_type == 1:
                        # Ref: tarch_starting_values.m:108
                        # omega = mean(abs(data)) * (1 - alpha*adj - 0.5*gamma - beta)
                        # Note: uses raw alpha/gamma scalars (not sums of temp arrays)
                        omega: float = float(
                            np.mean(np.abs(epsilon))
                            * (1.0 - alpha * adj_factor - 0.5 * gamma - beta)
                        )
                    else:
                        # Ref: tarch_starting_values.m:110
                        # omega = covar * (1 - sum(temp_alpha)*adj - 0.5*sum(temp_gamma) - sum(beta))
                        # adj_factor == 1.0 for type 2; sum(beta)=beta (scalar)
                        omega = float(
                            covar
                            * (
                                1.0
                                - np.sum(temp_alpha) * adj_factor
                                - 0.5 * np.sum(temp_gamma)
                                - beta
                            )
                        )

                    # -------------------------------------------------- #
                    #  Assemble parameter vector                          #
                    # -------------------------------------------------- #
                    # Ref: tarch_starting_values.m:113-118
                    # parameters = [omega; temp_alpha; temp_gamma; beta*ones(q,1)/q]
                    parts: list[np.ndarray] = [
                        np.array([omega], dtype=np.float64)
                    ]
                    if p > 0:
                        parts.append(temp_alpha)
                    if o > 0:
                        parts.append(temp_gamma)
                    if q > 0:
                        parts.append(
                            beta * np.ones(q, dtype=np.float64) / q
                        )
                    parameters = np.concatenate(parts)

                    # Ref: tarch_starting_values.m:120
                    output_parameters[index, :] = parameters

                    # Ref: tarch_starting_values.m:122
                    # Grid search always uses Normal (error_type=1)
                    ll_val, _, _ = tarch_likelihood(
                        parameters,
                        data_padded,
                        fdata_padded,
                        fIdata_padded,
                        p,
                        o,
                        q,
                        1,           # error_type = 1 (Normal) for grid search
                        tarch_type,
                        back_cast,
                        T_padded,
                    )
                    LLs[index] = ll_val

                    # Ref: tarch_starting_values.m:124 — index = index + 1
                    index += 1

        # -------------------------------------------------------------- #
        #  Sort by log-likelihood and select best                         #
        # -------------------------------------------------------------- #
        # Ref: tarch_starting_values.m:129 — [LLs, index] = sort(LLs)
        # Ascending sort: best (lowest negative LL) first.
        sort_indices = np.argsort(LLs)

        # Ref: tarch_starting_values.m:131 — best starting values
        startingvals_out: np.ndarray = output_parameters[
            sort_indices[0], :
        ].copy()

        # -------------------------------------------------------------- #
        #  Set distribution shape parameters to conventional defaults     #
        # -------------------------------------------------------------- #
        # Ref: tarch_starting_values.m:135-142
        if error_type == 2:
            nu = 8.0
        elif error_type == 3:
            nu = 1.9
        elif error_type == 4:
            nu = 8.0
            lam = -0.1

        return startingvals_out, nu, lam

    else:
        # ============================================================== #
        #  USER-SUPPLIED STARTING VALUES — parse nu / lambda              #
        # ============================================================== #
        # Ref: tarch_starting_values.m:144-154
        startingvals = np.asarray(startingvals, dtype=np.float64).ravel()
        nu_out: float | None = None
        lam_out: float | None = None

        # Ref: tarch_starting_values.m:147-149
        # MATLAB 1-based: nu = startingvals(p+o+q+2) → Python 0-based: [p+o+q+1]
        if error_type in (2, 3, 4):
            nu_out = float(startingvals[p + o + q + 1])

        # Ref: tarch_starting_values.m:150-152
        # MATLAB 1-based: lambda = startingvals(p+o+q+3) → Python 0-based: [p+o+q+2]
        if error_type == 4:
            lam_out = float(startingvals[p + o + q + 2])

        # Ref: tarch_starting_values.m:153
        # MATLAB: startingvals = startingvals(1:p+o+q+1)
        # Python: first 1+p+o+q elements (indices 0 through p+o+q)
        startingvals_core: np.ndarray = startingvals[: 1 + p + o + q].copy()

        return startingvals_core, nu_out, lam_out
