"""
Grid search for starting values for Scalar Variance-Targeting VECH (VT-VECH)
multivariate GARCH estimation.

Performs a systematic grid search over the parameter space of a Scalar
VT-VECH(P,O,Q) model to find starting values that yield the best
(lowest negative) log-likelihood.  If user-supplied starting values are
provided, the grid search is skipped and only the likelihood at the
user-supplied point is evaluated.

Grid Construction
-----------------
The grid is built from the Cartesian product of:

- ``alpha_grid`` ∈ {0.005, 0.01, 0.025, 0.05}  (innovation coefficient)
- ``gamma_grid`` ∈ {0.005, 0.02, 0.5, 0.1}     (asymmetry coefficient)
- ``persistence_grid`` ∈ {0.99, 0.95, 0.90, 0.5} (total persistence)

Each ``alpha`` value is spread equally across ``p`` lags, each ``gamma``
value equally across ``o`` lags, and ``beta`` is derived from the
persistence target minus the alpha and gamma contributions (scaled by
``kappa``).  A stationarity safeguard rescales alpha and gamma when
the residual beta falls below half the persistence target.

Notes
-----
Migrated from ``multivariate/scalar_vt_vech_starting_values.m`` (75 lines).

Author: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005

References
----------
Kevin Sheppard, MFE Toolbox Version 4.0 (2009).
See also: SCALAR_VT_VECH, SCALAR_VT_VECH_LIKELIHOOD
"""

import numpy as np

from mfe_toolbox.multivariate.scalar_vt_vech_likelihood import scalar_vt_vech_likelihood


def scalar_vt_vech_starting_values(
    starting_vals: np.ndarray | None,
    data: np.ndarray,
    data_asym: np.ndarray,
    p: int,
    o: int,
    q: int,
    c: np.ndarray,
    c_asym: np.ndarray,
    kappa: float,
    use_composite: int,
    back_cast: np.ndarray,
    back_cast_asym: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid search for starting values for Scalar VT-VECH(P,O,Q) estimation.

    If ``starting_vals`` is ``None`` (or an empty array), a grid search is
    performed over candidate parameter vectors, evaluating the log-likelihood
    at each point and selecting the vector with the lowest negative
    log-likelihood.  If ``starting_vals`` is already supplied (non-empty),
    the function simply evaluates the likelihood at that point and returns
    immediately.

    Parameters
    ----------
    starting_vals : np.ndarray or None
        User-supplied starting values of length ``p + o + q``, or ``None``
        (or an empty array) to trigger the grid search.
    data : np.ndarray, shape (K, K, T)
        K × K × T array of (symmetric) covariance innovations, typically
        outer products of return vectors: ``data[:, :, t] = r_t @ r_t.T``.
    data_asym : np.ndarray, shape (K, K, T)
        K × K × T array of asymmetric covariance innovations, typically
        ``data_asym[:, :, t] = n_t @ n_t.T`` where ``n_t = r_t * (r_t < 0)``.
    p : int
        Positive integer — number of symmetric innovation lags.
    o : int
        Non-negative integer — number of asymmetric innovation lags.
    q : int
        Non-negative integer — number of lagged conditional covariance terms.
    c : np.ndarray, shape (K, K)
        Unconditional covariance matrix (variance-targeting intercept).
    c_asym : np.ndarray, shape (K, K)
        Unconditional expectation of the asymmetric covariance matrix.
    kappa : float
        Positive scalar — asymmetry scaling factor derived from the
        eigenvalue structure of the unconditional covariance.  Used to
        weight gamma contributions in the stationarity constraint:
        ``sum(alpha) + sum(gamma) / kappa + sum(beta) < 1``.
    use_composite : int
        Likelihood type selector:

        - ``0``: Standard K-dimensional multivariate normal log-likelihood.
        - ``1``: Diagonal composite likelihood (adjacent pairs only).
        - ``2``: Full composite likelihood (all unique pairs).

    back_cast : np.ndarray, shape (K, K)
        Back-cast value for initialising the GARCH recursion when lagged
        values are unavailable (i.e. for early time steps).
    back_cast_asym : np.ndarray, shape (K, K)
        Back-cast value for the asymmetric recursion terms.

    Returns
    -------
    starting_vals : np.ndarray, shape (p + o + q,)
        Best starting values (the parameter vector with the lowest
        negative log-likelihood, or the user-supplied values unchanged).
    lls : np.ndarray
        Sorted negative log-likelihoods for all evaluated grid points
        (length equal to number of unique grid rows), or a single-element
        array if user-supplied starting values were provided.
    output_parameters : np.ndarray, shape (N, p + o + q)
        Matrix of all evaluated parameter vectors, sorted by log-likelihood
        (ascending), or a single-row matrix of the user-supplied values.

    Examples
    --------
    >>> import numpy as np
    >>> k, T = 2, 100
    >>> rng = np.random.default_rng(42)
    >>> data = np.zeros((k, k, T))
    >>> data_asym = np.zeros((k, k, T))
    >>> for t in range(T):
    ...     r = rng.standard_normal(k)
    ...     data[:, :, t] = np.outer(r, r)
    ...     n = r * (r < 0)
    ...     data_asym[:, :, t] = np.outer(n, n)
    >>> c = np.mean(data, axis=2)
    >>> c_asym = np.mean(data_asym, axis=2)
    >>> bc = c.copy()
    >>> bc_asym = c_asym.copy()
    >>> sv, lls, op = scalar_vt_vech_starting_values(
    ...     None, data, data_asym, 1, 0, 1,
    ...     c, c_asym, 2.0, 0, bc, bc_asym
    ... )
    >>> sv.shape
    (2,)

    See Also
    --------
    mfe_toolbox.multivariate.scalar_vt_vech : Driver function.
    mfe_toolbox.multivariate.scalar_vt_vech_likelihood : Likelihood evaluator.
    """
    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_starting_values.m:31 — Time dimension
    # MATLAB: t = size(data, 3);
    # ------------------------------------------------------------------
    t = data.shape[2]  # noqa: F841 — kept for parity reference

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_starting_values.m:32-34 — Grid point arrays
    # a:    Innovation coefficient candidates (alpha)
    # g:    Asymmetry coefficient candidates (gamma)
    # apgpb: Total persistence targets (alpha + gamma/kappa + beta)
    # ------------------------------------------------------------------
    a_grid = np.array([0.005, 0.01, 0.025, 0.05])
    g_grid = np.array([0.005, 0.02, 0.5, 0.1])
    persistence_grid = np.array([0.99, 0.95, 0.90, 0.50])

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_starting_values.m:35-36 — Allocate parameter matrix
    # Total combinations = len(a) * len(g) * len(persistence)
    # ------------------------------------------------------------------
    n_params = p + o + q
    n_combos = len(a_grid) * len(g_grid) * len(persistence_grid)
    params = np.zeros((n_combos, n_params))

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_starting_values.m:37-55 — Triple nested grid search
    # MATLAB uses 1-based loop counters i, j, k; Python uses 0-based
    # i_idx, j_idx, k_idx.
    #
    # Key MATLAB behaviour preserved:
    #   - alpha is reset for each i_idx (outer loop)
    #   - gamma is reset for each j_idx (middle loop)
    #   - If the stationarity safeguard fires (beta <= persistence/2),
    #     alpha and gamma are SCALED DOWN and remain scaled for all
    #     subsequent k_idx iterations within the current (i, j) pair.
    # ------------------------------------------------------------------
    count = 0  # Ref: scalar_vt_vech_starting_values.m:35 — MATLAB count=1 (1-based)

    for i_idx in range(len(a_grid)):
        # Ref: scalar_vt_vech_starting_values.m:38 — alpha = ones(1,p)*a(i)/p
        # Spread the alpha budget equally across p lags.
        # Guard against p == 0 (should not happen per spec, but be safe).
        if p > 0:
            alpha = np.ones(p) * a_grid[i_idx] / p
        else:
            alpha = np.empty(0)

        for j_idx in range(len(g_grid)):
            # Ref: scalar_vt_vech_starting_values.m:40 — gamma = ones(1,o)*g(j)/o
            # Spread the gamma budget equally across o lags.
            # When o == 0 (no asymmetric lags), gamma is empty and
            # sum(gamma) == 0 — matching MATLAB's ones(1,0) behaviour.
            if o > 0:
                gamma = np.ones(o) * g_grid[j_idx] / o
            else:
                gamma = np.empty(0)

            for k_idx in range(len(persistence_grid)):
                # Ref: scalar_vt_vech_starting_values.m:42
                # beta = persistence - sum(alpha) - sum(gamma)/kappa
                sum_alpha = np.sum(alpha) if alpha.size > 0 else 0.0
                sum_gamma = np.sum(gamma) if gamma.size > 0 else 0.0
                beta_scalar = (
                    persistence_grid[k_idx] - sum_alpha - sum_gamma / kappa
                )

                # Ref: scalar_vt_vech_starting_values.m:43-49
                # Stationarity safeguard: if beta is less than half the
                # persistence target, rescale alpha and gamma downward so
                # that their combined contribution consumes exactly half
                # the persistence target, leaving the other half for beta.
                if beta_scalar <= persistence_grid[k_idx] / 2.0:
                    denom = sum_alpha + sum_gamma / kappa
                    if denom > 0.0:
                        scale = 0.5 * persistence_grid[k_idx] / denom
                    else:
                        # Edge case: both alpha and gamma contribute zero;
                        # no scaling needed (beta will equal persistence).
                        scale = 1.0
                    alpha = alpha * scale
                    gamma = gamma * scale
                    # Recompute beta with rescaled alpha/gamma
                    # Ref: scalar_vt_vech_starting_values.m:47
                    sum_alpha = np.sum(alpha) if alpha.size > 0 else 0.0
                    sum_gamma = np.sum(gamma) if gamma.size > 0 else 0.0
                    beta_scalar = (
                        persistence_grid[k_idx]
                        - sum_alpha
                        - sum_gamma / kappa
                    )

                # Ref: scalar_vt_vech_starting_values.m:50
                # Spread beta equally across q lags.
                if q > 0:
                    beta = np.ones(q) * beta_scalar / q
                else:
                    beta = np.empty(0)

                # Ref: scalar_vt_vech_starting_values.m:51
                # Concatenate [alpha, gamma, beta] into a single row.
                parts = [arr for arr in (alpha, gamma, beta) if arr.size > 0]
                if parts:
                    params[count, :] = np.concatenate(parts)
                # else: params[count, :] remains zeros (n_params would be 0)
                count += 1

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_starting_values.m:56 — Remove duplicate rows
    # MATLAB: params = unique(params, 'rows');
    # np.unique with axis=0 deduplicates rows (sorted lexicographically).
    # ------------------------------------------------------------------
    params = np.unique(params, axis=0)

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_starting_values.m:57 — Remove all-zero rows
    # MATLAB: params = params(~all(params==0, 2), :);
    # ------------------------------------------------------------------
    non_zero_mask = ~np.all(params == 0.0, axis=1)
    params = params[non_zero_mask]

    # ------------------------------------------------------------------
    # Ref: scalar_vt_vech_starting_values.m:61-74 — Evaluate likelihoods
    # Two branches:
    #   1. starting_vals is None/empty → grid search → sort → pick best
    #   2. starting_vals provided     → evaluate once → return as-is
    # ------------------------------------------------------------------
    _is_empty = (
        starting_vals is None
        or (isinstance(starting_vals, np.ndarray) and starting_vals.size == 0)
    )

    if _is_empty:
        # ---------------------------------------------------------------
        # Grid search mode
        # Ref: scalar_vt_vech_starting_values.m:62-70
        # ---------------------------------------------------------------
        n_candidates = params.shape[0]
        lls = np.zeros(n_candidates)

        # Ref: scalar_vt_vech_starting_values.m:64-67
        # Evaluate scalar_vt_vech_likelihood at each grid point.
        # The likelihood returns (ll, per_period_lls, Ht); we only need ll.
        for i in range(n_candidates):
            candidate = params[i, :]
            ll_val, _lls_t, _ht = scalar_vt_vech_likelihood(
                candidate,
                data,
                data_asym,
                p,
                o,
                q,
                c,
                c_asym,
                kappa,
                back_cast,
                back_cast_asym,
                False,          # is_joint
                use_composite,
                False,          # estim_flag
            )
            lls[i] = ll_val

        # Ref: scalar_vt_vech_starting_values.m:68
        # Sort by negative log-likelihood (ascending = best first).
        # MATLAB: [lls, pos] = sort(lls);
        pos = np.argsort(lls)
        lls = lls[pos]

        # Ref: scalar_vt_vech_starting_values.m:69-70
        output_parameters = params[pos, :]
        starting_vals_out = output_parameters[0, :].copy()
    else:
        # ---------------------------------------------------------------
        # User-supplied starting values — evaluate once
        # Ref: scalar_vt_vech_starting_values.m:72-73
        # ---------------------------------------------------------------
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        ll_val, _lls_t, _ht = scalar_vt_vech_likelihood(
            starting_vals,
            data,
            data_asym,
            p,
            o,
            q,
            c,
            c_asym,
            kappa,
            back_cast,
            back_cast_asym,
            False,          # is_joint
            use_composite,
            False,          # estim_flag
        )
        # Ref: scalar_vt_vech_starting_values.m:72 — lls is a scalar in MATLAB
        # Return as a 1-element array for consistency
        lls = np.array([ll_val])
        output_parameters = np.atleast_2d(starting_vals)
        starting_vals_out = starting_vals.copy()

    return starting_vals_out, lls, output_parameters
