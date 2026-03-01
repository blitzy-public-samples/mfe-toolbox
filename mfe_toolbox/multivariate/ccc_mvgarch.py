"""
CCC-MVGARCH (Constant Conditional Correlation Multivariate GARCH) Estimation Driver.

Implements Constant Conditional Correlation MV GARCH with TARCH(p,o,q) or
GJR-GARCH(p,o,q) conditional variances using a two-stage estimation procedure:

Stage 1: Fit K univariate TARCH models to obtain conditional variances.
Stage 2: Estimate the constant correlation matrix R from standardized residuals.

Migrated from: multivariate/ccc_mvgarch.m
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 4, Date: 4/13/2012

References
----------
Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange
rates: A multivariate generalized ARCH model. Review of Economics and
Statistics, 72, 498-505.
"""

import warnings

import numpy as np
from scipy.optimize import minimize


# ============================================================================
# LOCAL HELPER: TARCH variance recursion
# Ref: univariate/tarch_core_simple.m and mex_source/tarch_core.c
# ============================================================================
def _tarch_core(fdata, fIdata, parameters, back_cast, p, o, q, m, T, tarch_type):
    """
    Compute TARCH(p,o,q) conditional variance recursion.

    Parameters match the MEX interface: fdata and fIdata are already augmented
    with m back-cast entries prepended. The recursion produces T values;
    the first m are warm-up and the remaining T-m are the valid output.

    Ref: tarch_core_simple.m — adapted to MEX-style augmented-data interface.
    """
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    fdata = np.asarray(fdata, dtype=np.float64).ravel()
    fIdata = np.asarray(fIdata, dtype=np.float64).ravel()

    omega = parameters[0]
    alpha = parameters[1:1 + p]
    gamma = parameters[1 + p:1 + p + o]
    beta = parameters[1 + p + o:1 + p + o + q]

    ht = np.zeros(T, dtype=np.float64)
    for t in range(T):
        ht[t] = omega
        for j in range(p):
            idx = t - j - 1
            if idx >= 0:
                ht[t] += alpha[j] * fdata[idx]
            else:
                ht[t] += alpha[j] * back_cast
        for j in range(o):
            idx = t - j - 1
            if idx >= 0:
                ht[t] += gamma[j] * fIdata[idx]
            else:
                # Ref: tarch.m:115 — asymmetric back-cast = 0.5 * back_cast
                ht[t] += gamma[j] * 0.5 * back_cast
        for j in range(q):
            idx = t - j - 1
            if idx >= 0:
                ht[t] += beta[j] * ht[idx]
            else:
                ht[t] += beta[j] * back_cast
    # Ref: tarch_core_simple.m:79-80 — square ht for absolute-value models
    if tarch_type == 1:
        ht = ht ** 2
    return ht


# ============================================================================
# LOCAL HELPER: Normal log-likelihood
# Ref: distributions/normloglik.m
# ============================================================================
def _normloglik(data, mu, sigma2):
    """
    Compute normal log-likelihood and per-observation log-likelihoods.

    Returns (LL, LLS) where LL is the total log-likelihood (positive)
    and LLS is a T-vector of per-observation log-likelihoods (positive).
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    sigma2 = np.asarray(sigma2, dtype=np.float64).ravel()
    T = len(data)
    # Ref: normloglik.m — LL = -0.5 * sum(log(2*pi) + log(sigma2) + (data-mu)^2/sigma2)
    lls = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + (data - mu) ** 2 / sigma2)
    ll = float(np.sum(lls))
    return ll, lls


# ============================================================================
# LOCAL HELPER: TARCH log-likelihood (normal errors only)
# Ref: univariate/tarch_likelihood.m
# ============================================================================
def _tarch_likelihood(parameters, data_aug, fdata, fIdata, p, o, q,
                      tarch_type, back_cast, T, estim_flag=False):
    """
    Compute negated log-likelihood for TARCH(p,o,q) with normal errors.

    When estim_flag is True, parameters are in unconstrained space and are
    inverse-transformed before evaluation.

    Returns (neg_LL, neg_LLS, ht) where neg_LL is -1 * total log-likelihood.
    """
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    if estim_flag:
        parameters = _tarch_itransform(parameters, p, o, q)

    m = max(p, o, q)
    ht = _tarch_core(fdata, fIdata, parameters, back_cast, p, o, q, m, T, tarch_type)

    # Ref: tarch_likelihood.m:64-65 — trim to valid observations
    ht_valid = ht[m:T]
    data_valid = data_aug[m:T]

    # Ensure positive variances for numerical stability
    ht_valid = np.maximum(ht_valid, 1e-20)

    # Ref: tarch_likelihood.m:69-72 — normal log-likelihood, negated
    ll, lls = _normloglik(data_valid, 0.0, ht_valid)
    return -ll, -lls, ht


def _tarch_likelihood_obj(parameters, data_aug, fdata, fIdata, p, o, q,
                          tarch_type, back_cast, T):
    """Scalar objective for scipy.optimize (transformed parameters)."""
    neg_ll, _, _ = _tarch_likelihood(
        parameters, data_aug, fdata, fIdata, p, o, q,
        tarch_type, back_cast, T, estim_flag=True
    )
    return neg_ll


# ============================================================================
# LOCAL HELPER: TARCH parameter transform (constrained → unconstrained)
# Ref: univariate/tarch_transform.m (normal error_type=1 only)
# ============================================================================
def _tarch_transform(parameters, p, o, q):
    """
    Map constrained TARCH parameters to unconstrained real line.

    Ref: tarch_transform.m — only normal error_type path.
    """
    parameters = np.asarray(parameters, dtype=np.float64).ravel().copy()
    omega = parameters[0]
    alpha = parameters[1:1 + p].copy()
    gamma = parameters[1 + p:1 + p + o].copy()
    beta = parameters[1 + p + o:1 + p + o + q].copy()

    UB = 0.999998
    # Ref: tarch_transform.m:105-106 — avoid exact zeros
    alpha[alpha == 0] = 1e-8
    beta[beta == 0] = 1e-8

    # Ref: tarch_transform.m:108-118 — handle alpha+gamma == 0
    gamma2 = np.concatenate([gamma, np.zeros(max(0, p - o))])
    alpha2 = np.concatenate([alpha, np.zeros(max(0, o - p))])
    pl = np.where((alpha2 + gamma2) == 0)[0]
    if len(pl) > 0:
        if p >= o:
            for idx in pl:
                if idx < p:
                    alpha[idx] += 1e-8
        else:
            for idx in pl:
                if idx < o:
                    gamma[idx] += 1e-8

    UB = UB + 1e-8 * (p + o + q)

    # Ref: tarch_transform.m:78-80 — log transform for omega
    tomega = np.log(omega)

    # Ref: tarch_transform.m:123-133 — alpha transform
    scale = UB
    talpha = np.zeros(p)
    for i in range(p):
        val = alpha[i] / scale
        val = np.clip(val, 1e-15, 1.0 - 1e-15)
        talpha[i] = np.log(val / (1.0 - val))
        scale -= alpha[i]

    # Ref: tarch_transform.m:136-159 — gamma transform
    tgamma = np.zeros(o)
    for i in range(o):
        tg = gamma[i]
        if i < p:
            tg += alpha[i]
            tg /= (2.0 * scale + alpha[i])
        else:
            tg /= (2.0 * scale)
        tg = np.clip(tg, 1e-15, 1.0 - 1e-15)
        tgamma[i] = np.log(tg / (1.0 - tg))
        scale -= gamma[i] / 2.0

    # Ref: tarch_transform.m:162-171 — beta transform
    tbeta = np.zeros(q)
    for i in range(q):
        val = beta[i] / scale
        val = np.clip(val, 1e-15, 1.0 - 1e-15)
        tbeta[i] = np.log(val / (1.0 - val))
        scale -= beta[i]

    return np.concatenate([[tomega], talpha, tgamma, tbeta])


# ============================================================================
# LOCAL HELPER: TARCH inverse parameter transform (unconstrained → constrained)
# Ref: univariate/tarch_itransform.m (normal error_type=1 only)
# ============================================================================
def _tarch_itransform(parameters, p, o, q):
    """
    Map unconstrained parameters back to constrained TARCH parameter space.

    Ref: tarch_itransform.m — only normal error_type path.
    """
    parameters = np.asarray(parameters, dtype=np.float64).ravel().copy()
    # Ref: tarch_itransform.m:50 — cap to prevent overflow
    parameters = np.clip(parameters, -100.0, 100.0)

    omega = parameters[0]
    alpha = parameters[1:1 + p].copy()
    gamma = parameters[1 + p:1 + p + o].copy()
    beta = parameters[1 + p + o:1 + p + o + q].copy()

    UB = 0.9998

    # Ref: tarch_itransform.m:83 — simple transform of omega
    tomega = np.exp(omega)

    scale = UB
    talpha = np.zeros(p)
    for i in range(p):
        # Ref: tarch_itransform.m:95 — logistic map into [0, scale]
        talpha[i] = (np.exp(alpha[i]) / (1.0 + np.exp(alpha[i]))) * scale
        scale -= talpha[i]

    tgamma = np.zeros(o)
    for i in range(o):
        # Ref: tarch_itransform.m:102 — logistic into [0, 1]
        tg = np.exp(gamma[i]) / (1.0 + np.exp(gamma[i]))
        if i < p:
            # Ref: tarch_itransform.m:104-107 — map into [-alpha_i, 2*scale]
            tg = tg * (2.0 * scale + talpha[i])
            tg -= talpha[i]
        else:
            # Ref: tarch_itransform.m:109-110 — map into [0, 2*scale]
            tg = tg * (2.0 * scale)
        tgamma[i] = tg
        scale -= 0.5 * tgamma[i]

    tbeta = np.zeros(q)
    for i in range(q):
        # Ref: tarch_itransform.m:117 — logistic into [0, scale]
        tbeta[i] = (np.exp(beta[i]) / (1.0 + np.exp(beta[i]))) * scale
        scale -= tbeta[i]

    return np.concatenate([[tomega], talpha, tgamma, tbeta])


# ============================================================================
# LOCAL HELPER: TARCH starting values grid search
# Ref: univariate/tarch_starting_values.m (normal error_type=1 only)
# ============================================================================
def _tarch_starting_values(data_aug, fdata, fIdata, p, o, q, T,
                           tarch_type, back_cast):
    """
    Grid search for decent TARCH starting values with normal errors.

    Returns (starting_vals, ordered_params) where starting_vals is the best
    set and ordered_params is the matrix sorted by log-likelihood.

    Ref: tarch_starting_values.m:47-132.
    """
    a_grid = [0.05, 0.1, 0.2]
    g_grid = [0.01, 0.05, 0.2]
    agb_grid = [0.5, 0.8, 0.9, 0.95, 0.99]

    la = len(a_grid)
    lg = len(g_grid) + 1  # +1 for the gamma=-alpha/2 case
    lb = len(agb_grid)
    n_combos = la * lb * lg

    output_parameters = np.zeros((n_combos, 1 + p + o + q))
    lls = np.zeros(n_combos)

    # Ref: tarch_starting_values.m:66-73 — back-cast and adjustment
    if tarch_type == 1:
        adj_factor = np.sqrt(2.0 / np.pi)
    else:
        adj_factor = 1.0
    # data_aug has m prepended values; use all for covariance
    raw_data = data_aug[max(p, o, q):]
    covar = float(np.var(raw_data, ddof=0)) if len(raw_data) > 0 else 1.0

    index = 0
    for i_a, alpha_val in enumerate(a_grid):
        for j_g in range(lg):
            if j_g == lg - 1:
                gamma_val = -alpha_val / 2.0
            else:
                gamma_val = g_grid[j_g]

            for k_b, agb_val in enumerate(agb_grid):
                temp_alpha = (alpha_val * np.ones(p) / p) if p > 0 else np.array([])
                temp_gamma = np.zeros(0)
                if o > 0:
                    temp_gamma = gamma_val * np.ones(o) / o
                    # Ref: tarch_starting_values.m:96-101 — constrain gamma
                    for n_idx in range(o):
                        if n_idx < p:
                            temp_gamma[n_idx] = max(
                                temp_gamma[n_idx], -alpha_val / (2.0 * p)
                            )
                        else:
                            temp_gamma[n_idx] = 0.0

                beta_total = agb_val - np.sum(temp_alpha) - 0.5 * np.sum(temp_gamma)
                if beta_total <= 0:
                    beta_total = 0.01

                # Ref: tarch_starting_values.m:107-111 — omega from unconditional
                if tarch_type == 1:
                    mean_abs = float(np.mean(np.abs(raw_data))) if len(raw_data) > 0 else 1.0
                    omega_val = mean_abs * (
                        1.0 - alpha_val * adj_factor - 0.5 * gamma_val - beta_total
                    )
                else:
                    omega_val = covar * (
                        1.0 - np.sum(temp_alpha) * adj_factor
                        - 0.5 * np.sum(temp_gamma) - beta_total
                    )
                omega_val = max(omega_val, 1e-8)

                params = np.concatenate([
                    [omega_val], temp_alpha, temp_gamma,
                    (beta_total * np.ones(q) / q) if q > 0 else np.array([])
                ])
                output_parameters[index, :] = params
                lls[index], _, _ = _tarch_likelihood(
                    params, data_aug, fdata, fIdata, p, o, q,
                    tarch_type, back_cast, T, estim_flag=False
                )
                index += 1

    # Ref: tarch_starting_values.m:129-133 — sort by likelihood (ascending)
    sort_idx = np.argsort(lls)
    lls = lls[sort_idx]
    output_parameters = output_parameters[sort_idx, :]
    starting_vals = output_parameters[0, :]
    return starting_vals, output_parameters


# ============================================================================
# LOCAL HELPER: Fit a single TARCH model for one series
# Ref: univariate/tarch.m — simplified for normal errors (error_type=1)
# ============================================================================
def _fit_single_tarch(epsilon, p_i, o_i, q_i, tarch_type_i, starting_vals=None):
    """
    Estimate a TARCH(p,o,q) model for a single series with normal errors.

    Returns a dict with keys: parameters, ht, A, scores, fdata, fIdata,
    back_cast, m, T, tarch_type, p, o, q.

    Ref: tarch.m lines 106-280.
    """
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()
    m = max(p_i, o_i, q_i)
    T_raw = len(epsilon)

    # Ref: tarch.m:111-138 — compute augmented fdata, fIdata, back_cast
    if tarch_type_i == 1:
        mean_val = float(np.mean(np.abs(epsilon)))
        if mean_val == 0:
            mean_val = 1e-8
        fepsilon = np.concatenate([mean_val * np.ones(m), np.abs(epsilon)])
        fIepsilon = np.concatenate([
            0.5 * mean_val * np.ones(m),
            np.abs(epsilon) * (epsilon < 0).astype(np.float64)
        ])
        bc_len = max(int(np.floor(np.sqrt(T_raw))), 1)
        bc_weights = 0.05 * (0.9 ** np.arange(bc_len + 1))
        bc_weights /= np.sum(bc_weights)
        back_cast = float(bc_weights @ np.abs(epsilon[:bc_len + 1]))
        if back_cast == 0:
            back_cast = mean_val
    else:
        mean_sq = float(np.mean(epsilon ** 2))
        if mean_sq == 0:
            mean_sq = 1e-8
        fepsilon = np.concatenate([mean_sq * np.ones(m), epsilon ** 2])
        fIepsilon = np.concatenate([
            0.5 * mean_sq * np.ones(m),
            (epsilon ** 2) * (epsilon < 0).astype(np.float64)
        ])
        bc_len = max(int(np.floor(np.sqrt(T_raw))), 1)
        bc_weights = 0.05 * (0.9 ** np.arange(bc_len + 1))
        bc_weights /= np.sum(bc_weights)
        back_cast = float(bc_weights @ (epsilon[:bc_len + 1] ** 2))
        if back_cast == 0:
            back_cast = mean_sq

    # Ref: tarch.m:139-141
    epsilon_augmented = np.concatenate([np.zeros(m), epsilon])
    T = len(fepsilon)

    # Starting values
    if starting_vals is None:
        sv, ordered_params = _tarch_starting_values(
            epsilon_augmented, fepsilon, fIepsilon, p_i, o_i, q_i,
            T, tarch_type_i, back_cast
        )
    else:
        sv = np.asarray(starting_vals, dtype=np.float64).ravel()[:1 + p_i + o_i + q_i]
        ordered_params = None

    # Transform and optimize
    try:
        sv_transformed = _tarch_transform(sv, p_i, o_i, q_i)
    except Exception:
        sv_transformed = np.zeros(1 + p_i + o_i + q_i)

    # Ref: tarch.m:172 — optimize (replaces fminunc)
    try:
        result = minimize(
            _tarch_likelihood_obj, sv_transformed,
            args=(epsilon_augmented, fepsilon, fIepsilon, p_i, o_i, q_i,
                  tarch_type_i, back_cast, T),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 1000, 'disp': False,
                     'ftol': 1e-8, 'gtol': 1e-6}
        )
        opt_params = result.x
        converged = result.success
    except Exception:
        opt_params = sv_transformed
        converged = False

    # Ref: tarch.m:204-252 — robustness with alternative starting values
    if not converged and ordered_params is not None:
        best_ll = _tarch_likelihood_obj(
            opt_params, epsilon_augmented, fepsilon, fIepsilon,
            p_i, o_i, q_i, tarch_type_i, back_cast, T
        )
        best_params = opt_params.copy()
        for alt_idx in range(1, min(len(ordered_params), 5)):
            try:
                alt_sv = _tarch_transform(ordered_params[alt_idx], p_i, o_i, q_i)
                alt_result = minimize(
                    _tarch_likelihood_obj, alt_sv,
                    args=(epsilon_augmented, fepsilon, fIepsilon, p_i, o_i, q_i,
                          tarch_type_i, back_cast, T),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'maxfun': 1000, 'disp': False}
                )
                if alt_result.fun < best_ll:
                    best_ll = alt_result.fun
                    best_params = alt_result.x
                    if alt_result.success:
                        break
            except Exception:
                continue
        opt_params = best_params

    # Ref: tarch.m:255 — inverse transform to constrained space
    parameters_out = _tarch_itransform(opt_params, p_i, o_i, q_i)

    # Compute final ht
    _, _, ht_full = _tarch_likelihood(
        parameters_out, epsilon_augmented, fepsilon, fIepsilon,
        p_i, o_i, q_i, tarch_type_i, back_cast, T, estim_flag=False
    )
    ht_out = ht_full[m:T]

    # -----------------------------------------------------------------------
    # Ref: tarch.m:264-269 + robustvcv.m — Compute A and scores for VCV
    # -----------------------------------------------------------------------
    n_params = 1 + p_i + o_i + q_i
    T_eff = T - m

    def _tarch_ll_for_vcv(params):
        neg_ll, neg_lls, _ = _tarch_likelihood(
            params, epsilon_augmented, fepsilon, fIepsilon,
            p_i, o_i, q_i, tarch_type_i, back_cast, T, estim_flag=False
        )
        return neg_ll, neg_lls

    # Scores: T_eff x n_params via 2-sided finite differences on per-obs LLs
    h_step = np.maximum(
        np.abs(parameters_out) * (np.finfo(np.float64).eps ** (1.0 / 3.0)),
        1e-8
    )
    scores = np.zeros((T_eff, n_params))
    for i in range(n_params):
        p_plus = parameters_out.copy()
        p_plus[i] += h_step[i]
        _, lls_plus = _tarch_ll_for_vcv(p_plus)
        p_minus = parameters_out.copy()
        p_minus[i] -= h_step[i]
        _, lls_minus = _tarch_ll_for_vcv(p_minus)
        scores[:, i] = (lls_plus - lls_minus) / (2.0 * h_step[i])

    # A = hessian / T_eff (average Hessian)
    A = _hessian_2sided_scalar(_tarch_ll_for_vcv, parameters_out, n_params) / T_eff

    return {
        'parameters': parameters_out,
        'ht': ht_out,
        'A': A,
        'scores': scores,
        'fdata': fepsilon,
        'fIdata': fIepsilon,
        'back_cast': back_cast,
        'm': m,
        'T': T,
        'tarch_type': tarch_type_i,
        'p': p_i,
        'o': o_i,
        'q': q_i,
    }


def _hessian_2sided_scalar(func_with_lls, x, n):
    """
    Compute 2-sided Hessian of the scalar objective returned by func_with_lls.

    func_with_lls(x) returns (scalar_obj, per_obs_vec).
    Only the scalar_obj is used for the Hessian.

    Ref: utility/hessian_2sided.m.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.maximum(np.abs(x) * (np.finfo(np.float64).eps ** (1.0 / 3.0)), 1e-8)
    xh = x + h
    h = xh - x  # Exact step after floating-point rounding

    fx = func_with_lls(x)[0]
    gp = np.zeros(n)
    gm = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        gp[i] = func_with_lls(x + ei)[0]
        gm[i] = func_with_lls(x - ei)[0]

    hh = np.outer(h, h)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ei[i] = h[i]
            ej = np.zeros(n)
            ej[j] = h[j]
            fpp = func_with_lls(x + ei + ej)[0]
            fmm = func_with_lls(x - ei - ej)[0]
            H[i, j] = (fpp - gp[i] - gp[j] + fx + fx - gm[i] - gm[j] + fmm) / (2.0 * hh[i, j])
            H[j, i] = H[i, j]
    return H


# ============================================================================
# LOCAL HELPER: Fit K univariate TARCH models
# Ref: multivariate/dcc_fit_variance.m
# ============================================================================
def _dcc_fit_variance(data2d, p, o, q, gjr_type, starting_vals=None):
    """
    Fit K univariate TARCH models to each column of data2d.

    Returns
    -------
    H : ndarray, shape (T, K) — conditional variances
    univariate : list of dicts — per-series estimation results
    """
    T_obs, k = data2d.shape
    H = np.zeros((T_obs, k), dtype=np.float64)
    univariate = []
    offset = 0

    for i in range(k):
        sv_i = None
        if starting_vals is not None:
            count = 1 + int(p[i]) + int(o[i]) + int(q[i])
            if offset + count <= len(starting_vals):
                sv_i = starting_vals[offset:offset + count]
                offset += count

        result = _fit_single_tarch(
            data2d[:, i], int(p[i]), int(o[i]), int(q[i]),
            int(gjr_type[i]), sv_i
        )
        univariate.append(result)
        H[:, i] = result['ht']

    return H, univariate


# ============================================================================
# LOCAL HELPER: Correlation half-vectorization
# Ref: utility/corr_vech.m and utility/corr_ivech.m
# ============================================================================
def _corr_vech(R):
    """
    Extract off-diagonal entries from correlation matrix (upper-tri, row-major).

    Ref: corr_vech.m — sel = ~triu(true(k)); MATLAB column-major lower tri
    = numpy row-major upper tri.
    """
    R = np.asarray(R, dtype=np.float64)
    k = R.shape[0]
    rows, cols = np.triu_indices(k, 1)
    return R[rows, cols]


def _corr_ivech(stacked_data):
    """
    Reconstruct K x K correlation matrix from K*(K-1)/2 off-diagonal elements.

    Ref: corr_ivech.m:48-51.
    """
    stacked_data = np.asarray(stacked_data, dtype=np.float64).ravel()
    k2 = len(stacked_data)
    k = int((-1.0 + np.sqrt(1.0 + 8.0 * k2)) / 2.0) + 1
    if k * (k - 1) // 2 != k2:
        raise ValueError(
            f"The number of elements ({k2}) is not conformable to a "
            "correlation matrix."
        )
    R = np.zeros((k, k), dtype=np.float64)
    rows, cols = np.triu_indices(k, 1)
    R[rows, cols] = stacked_data
    R = R + R.T + np.eye(k, dtype=np.float64)
    return R


# ============================================================================
# LOCAL HELPER: Hessian (last nrows rows) for inference objective
# Ref: utility/hessian_2sided_nrows.m
# ============================================================================
def _hessian_2sided_nrows(func, x, nrows, *args):
    """
    Compute the last *nrows* rows of the 2-sided finite-difference Hessian.

    func(x, *args) returns a scalar.

    Ref: hessian_2sided_nrows.m.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    # Ref: hessian_2sided_nrows.m:52 — step size with 1e-2 minimum
    h = np.maximum(np.abs(x) * (np.finfo(np.float64).eps ** (1.0 / 3.0)), 1e-2)
    xh = x + h
    h = xh - x

    fx = func(x, *args)
    gp = np.zeros(n)
    gm = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        gp[i] = func(x + ei, *args)
        gm[i] = func(x - ei, *args)

    hh = np.outer(h, h)
    H_full = np.zeros((n, n))
    # Ref: hessian_2sided_nrows.m:70-77 — only compute rows n-nrows to n-1
    for i in range(n - nrows, n):
        for j in range(n):
            ei = np.zeros(n)
            ei[i] = h[i]
            ej = np.zeros(n)
            ej[j] = h[j]
            fpp = func(x + ei + ej, *args)
            fmm = func(x - ei - ej, *args)
            H_full[i, j] = (fpp - gp[i] - gp[j] + fx + fx - gm[i] - gm[j] + fmm) / (2.0 * hh[i, j])
            H_full[j, i] = H_full[i, j]

    # Ref: hessian_2sided_nrows.m:88 — return last nrows rows only
    return H_full[n - nrows:n, :]


# ============================================================================
# LOCAL HELPER: Two-sided gradient with per-observation scores
# Ref: utility/gradient_2sided.m
# ============================================================================
def _gradient_2sided(func, x, *args):
    """
    Compute 2-sided gradient and per-observation score matrix.

    func(x, *args) must return (scalar_obj, T_vector_of_per_obs).

    Returns (G, Gt) where G is (n,) gradient and Gt is (T, n) scores.

    Ref: gradient_2sided.m.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    # Ref: gradient_2sided.m:41 — step size with 1e-2 minimum
    h = np.maximum(np.abs(x) * (np.finfo(np.float64).eps ** (1.0 / 3.0)), 1e-2)
    xh = x + h
    h = xh - x

    ee = np.diag(h)
    gf = np.zeros(n)
    gb = np.zeros(n)

    # Evaluate at first perturbation to get T dimension
    f_plus_0, lls_plus_0 = func(x + ee[:, 0], *args)
    T_obs = len(lls_plus_0)
    Gf = np.zeros((T_obs, n))
    Gb = np.zeros((T_obs, n))
    gf[0] = f_plus_0
    Gf[:, 0] = lls_plus_0

    for i in range(1, n):
        gf[i], Gf[:, i] = func(x + ee[:, i], *args)

    for i in range(n):
        gb[i], Gb[:, i] = func(x - ee[:, i], *args)

    G = (gf - gb) / (2.0 * h)
    Gt = (Gf - Gb) / (2.0 * h[np.newaxis, :])
    return G, Gt


# ============================================================================
# LOCAL HELPER: Newey-West HAC covariance estimator
# Ref: utility/covnw.m
# ============================================================================
def _covnw(data, nlag=0):
    """
    Newey-West Bartlett kernel HAC covariance estimator.

    Ref: covnw.m — covnw(data, nlag, demean).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    T, k = data.shape
    # Demean
    data = data - data.mean(axis=0, keepdims=True)

    if nlag == 0:
        # Ref: robustvcv.m:77 — nw=0 uses sample covariance
        return (data.T @ data) / T

    # Bartlett kernel HAC
    V = (data.T @ data) / T
    for lag in range(1, nlag + 1):
        weight = 1.0 - lag / (nlag + 1.0)
        gamma_mat = (data[lag:].T @ data[:T - lag]) / T
        V += weight * (gamma_mat + gamma_mat.T)
    return V


# ============================================================================
# LOCAL HELPER: Reconstruct variance from stored univariate structs
# Ref: multivariate/dcc_reconstruct_variance.m
# ============================================================================
def _dcc_reconstruct_variance(garch_parameters, univariate):
    """
    Recompute T x K conditional variances from garch_parameters and stored
    univariate estimation results.

    Ref: dcc_reconstruct_variance.m.
    """
    k = len(univariate)
    u0 = univariate[0]
    T_eff = u0['T'] - u0['m']
    H = np.zeros((T_eff, k), dtype=np.float64)

    offset = 0
    for i in range(k):
        u = univariate[i]
        count = u['p'] + u['o'] + u['q'] + 1
        vol_params = garch_parameters[offset:offset + count]
        offset += count
        ht = _tarch_core(
            u['fdata'], u['fIdata'], vol_params, u['back_cast'],
            u['p'], u['o'], u['q'], u['m'], u['T'], u['tarch_type']
        )
        H[:, i] = ht[u['m']:u['T']]
    return H


# ============================================================================
# LOCAL HELPER: CCC inference objective (for VCV computation)
# Ref: multivariate/dcc_inference_objective.m (m=0, l=0, n=0 case)
# ============================================================================
def _ccc_inference_objective(parameters, data, data_asym, univariate):
    """
    Compute the "as-if" inference objective for CCC-MVGARCH.

    This is the dcc_inference_objective with m=0, l=0, n=0, reducing to a
    squared-error objective on the standardized residual correlations.

    Parameters
    ----------
    parameters : ndarray — stacked [garch_params, corr_vech(R)]
    data : ndarray, shape (K, K, T)
    data_asym : ndarray, shape (K, K, T)
    univariate : list of dicts

    Returns
    -------
    obj : float — total objective
    objs : ndarray, shape (T,) — per-observation objectives

    Ref: dcc_inference_objective.m:32-78.
    """
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    k = len(univariate)
    T = data.shape[2]

    # Ref: dcc_inference_objective.m:34-39 — parse garch parameters
    count_total = 0
    for i in range(k):
        count_total += univariate[i]['p'] + univariate[i]['o'] + univariate[i]['q'] + 1
    garch_params = parameters[:count_total]
    offset_param = count_total

    # Ref: dcc_inference_objective.m:42-43 — parse correlation parameters
    n_corr = k * (k - 1) // 2
    R = _corr_ivech(parameters[offset_param:offset_param + n_corr])

    # Ref: dcc_inference_objective.m:50 — reconstruct variance
    H = _dcc_reconstruct_variance(garch_params, univariate)

    # Ref: dcc_inference_objective.m:51-57 — standardize data
    std_data = np.zeros_like(data)
    for t in range(T):
        h = np.sqrt(np.maximum(H[t, :], 1e-20))
        h_outer = np.outer(h, h)
        std_data[:, :, t] = data[:, :, t] / h_outer

    # Ref: dcc_inference_objective.m:59-67 — correlation least-squares objective
    scales = np.diag(np.mean(std_data, axis=2))
    objs = np.zeros(T, dtype=np.float64)
    for j in range(k - 1):
        for i in range(j + 1, k):
            scale = np.sqrt(max(scales[i] * scales[j], 1e-20))
            errors = std_data[i, j, :] / scale - R[i, j]
            objs += 0.5 * (errors ** 2)

    obj = float(np.sum(objs))
    return obj, objs


# ============================================================================
# PUBLIC API: CCC-MVGARCH estimation
# ============================================================================
def ccc_mvgarch(data, data_asym=None, p=1, o=1, q=1, gjr_type=None,
                starting_vals=None, options=None):
    """
    Constant Conditional Correlation MV GARCH estimation with TARCH(p,o,q)
    or GJR-GARCH(p,o,q) conditional variances.

    Parameters
    ----------
    data : ndarray
        A T-by-K matrix of zero-mean residuals, or a K-by-K-by-T array of
        covariance estimators (e.g. realized covariance matrices).
    data_asym : ndarray or None, optional
        K-by-K-by-T array of asymmetric outer products. Must be None (or
        empty) when data is T-by-K. When data is 3D and o > 0, this provides
        the asymmetric innovation matrices.
    p : int or array_like
        Positive integer(s) for the number of symmetric innovation lags.
        Scalar is broadcast to all K series; array of length K specifies
        per-series orders.
    o : int or array_like
        Non-negative integer(s) for asymmetric innovation lags.
    q : int or array_like
        Non-negative integer(s) for conditional variance lags.
    gjr_type : int or array_like or None, optional
        1 for TARCH (absolute values), 2 for GJR-GARCH (squares).
        Default is 2. Scalar or K-vector.
    starting_vals : ndarray or None, optional
        Starting values for K TARCH models, stacked:
        [tarch(1)', tarch(2)', ..., tarch(K)'] where each tarch(i) =
        [omega(i), alpha(i,1:p_i), gamma(i,1:o_i), beta(i,1:q_i)]'.
    options : dict or None, optional
        Optimization options (currently unused; reserved for API compatibility).

    Returns
    -------
    parameters : ndarray
        Stacked parameter vector:
        [tarch(1)', ..., tarch(K)', corr_vech(R)].
    ll : float
        Log-likelihood at the optimum (positive; larger is better).
    Ht : ndarray, shape (K, K, T)
        Conditional covariance matrices.
    VCV : ndarray, shape (v, v)
        Robust parameter covariance matrix (A^{-1} B A^{-1}' / T).
    scores : ndarray, shape (T, v)
        Per-observation score matrix.

    Notes
    -----
    The CCC conditional covariance is:

        H(t) = D(t) * R * D(t)

    where D(t) = diag(sqrt(h_1t), ..., sqrt(h_Kt)) and each h_it follows
    a TARCH(p_i, o_i, q_i) process, and R is the constant correlation.

    Migrated from: multivariate/ccc_mvgarch.m (254 lines).

    References
    ----------
    Bollerslev, T. (1990). Modelling the coherence in short-run nominal
    exchange rates: A multivariate generalized ARCH model.
    Review of Economics and Statistics, 72, 498-505.
    """
    # ==================================================================
    # Input validation
    # Ref: ccc_mvgarch.m:55-175
    # ==================================================================
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 2:
        # Ref: ccc_mvgarch.m:72-83 — T x K matrix input
        T_obs, k = data.shape
        if data_asym is not None:
            raise ValueError(
                "If DATA is a T by K matrix, data_asym must be None."
            )
        # Convert to K x K x T outer products
        temp = np.zeros((k, k, T_obs), dtype=np.float64)
        data_asym_3d = np.zeros((k, k, T_obs), dtype=np.float64)
        for t in range(T_obs):
            row = data[t, :]
            temp[:, :, t] = np.outer(row, row)
            neg_row = row * (row < 0).astype(np.float64)
            data_asym_3d[:, :, t] = np.outer(neg_row, neg_row)
        data_3d = temp
        data_asym = data_asym_3d
    elif data.ndim == 3:
        # Ref: ccc_mvgarch.m:84-98 — K x K x T array input
        k = data.shape[0]
        m_dim = data.shape[1]
        T_obs = data.shape[2]
        if m_dim != k:
            raise ValueError("DATA must be K by K by T if a 3D array.")
        data_3d = data.copy()
        if data_asym is not None:
            data_asym = np.asarray(data_asym, dtype=np.float64)
            if data_asym.ndim != 3:
                raise ValueError(
                    "data_asym must be a 3D array with the same dimensions as DATA."
                )
            if data_asym.shape != data_3d.shape:
                raise ValueError(
                    "data_asym must have the same dimensions as DATA."
                )
    else:
        raise ValueError(
            "DATA must be a T by K matrix or a K by K by T 3D array."
        )

    # Ref: ccc_mvgarch.m:100-101
    if min(T_obs, k) < 2 or T_obs < k:
        raise ValueError(
            "DATA must be a T by K matrix or a K by K by T 3D array, T>K>1."
        )

    # Ref: ccc_mvgarch.m:106-137 — validate and broadcast p, o, q
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.int64)).ravel()
    o_arr = np.atleast_1d(np.asarray(o, dtype=np.int64)).ravel()
    q_arr = np.atleast_1d(np.asarray(q, dtype=np.int64)).ravel()

    if len(p_arr) == 1:
        if p_arr[0] < 1:
            raise ValueError("P must be a positive integer if scalar.")
        p_arr = np.full(k, p_arr[0], dtype=np.int64)
    else:
        if len(p_arr) != k or np.any(p_arr < 1):
            raise ValueError("P must contain K positive integer elements.")

    if len(o_arr) == 1:
        if o_arr[0] < 0:
            raise ValueError("O must be a non-negative integer if scalar.")
        o_arr = np.full(k, o_arr[0], dtype=np.int64)
    else:
        if len(o_arr) != k or np.any(o_arr < 0):
            raise ValueError("O must contain K non-negative integer elements.")

    if len(q_arr) == 1:
        if q_arr[0] < 0:
            raise ValueError("Q must be a non-negative integer if scalar.")
        q_arr = np.full(k, q_arr[0], dtype=np.int64)
    else:
        if len(q_arr) != k or np.any(q_arr < 0):
            raise ValueError("Q must contain K non-negative integer elements.")

    # Ref: ccc_mvgarch.m:138-140
    if np.any(o_arr > 0) and data_asym is None:
        raise ValueError("data_asym must be non-empty if O > 0.")

    # Ref: ccc_mvgarch.m:142-150 — gjr_type validation
    if gjr_type is None:
        gjr_type_arr = np.full(k, 2, dtype=np.int64)
    else:
        gjr_type_arr = np.atleast_1d(np.asarray(gjr_type, dtype=np.int64)).ravel()
        if len(gjr_type_arr) == 1:
            gjr_type_arr = np.full(k, gjr_type_arr[0], dtype=np.int64)
        if len(gjr_type_arr) != k or not np.all(np.isin(gjr_type_arr, [1, 2])):
            raise ValueError(
                "gjr_type must be in {1, 2} and must be scalar or K-vector."
            )

    # Ref: ccc_mvgarch.m:154-161 — starting_vals validation
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        min_len = int(k + np.sum(p_arr) + np.sum(o_arr) + np.sum(q_arr))
        if len(starting_vals) < min_len:
            raise ValueError(
                f"starting_vals should have at least "
                f"K+sum(P)+sum(O)+sum(Q) = {min_len} elements."
            )

    # ==================================================================
    # Data preparation
    # Ref: ccc_mvgarch.m:179-198
    # ==================================================================
    if data_asym is None:
        data_asym = np.tile(np.eye(k)[:, :, np.newaxis], (1, 1, T_obs))

    # Ref: ccc_mvgarch.m:184-188 — extract 2D data from 3D
    data2d = np.zeros((T_obs, k), dtype=np.float64)
    for t in range(T_obs):
        diag_vals = np.sqrt(np.maximum(np.diag(data_3d[:, :, t]), 0.0))
        asym_diag = np.diag(data_asym[:, :, t])
        signs = 2.0 * (asym_diag > 0).astype(np.float64) - 1.0
        data2d[t, :] = diag_vals * signs

    # ==================================================================
    # Stage 1: Fit K univariate TARCH models
    # Ref: ccc_mvgarch.m:200
    # ==================================================================
    H, univariate = _dcc_fit_variance(data2d, p_arr, o_arr, q_arr,
                                      gjr_type_arr, starting_vals)

    # ==================================================================
    # Compute htArray: K x K x T of sqrt(H_i * H_j)
    # Ref: ccc_mvgarch.m:201-207
    # ==================================================================
    htArray = np.zeros((k, k, T_obs), dtype=np.float64)
    for i in range(k):
        for j in range(i, k):
            htArray[i, j, :] = np.sqrt(np.maximum(H[:, i] * H[:, j], 0.0))
            htArray[j, i, :] = htArray[i, j, :]

    # ==================================================================
    # Stage 2: Estimate constant correlation R
    # Ref: ccc_mvgarch.m:209-211
    # ==================================================================
    htArray_safe = np.maximum(htArray, 1e-20)
    stdData = data_3d / htArray_safe

    # Ref: ccc_mvgarch.m:210 — mean of standardized data across time
    R = np.mean(stdData, axis=2)

    # Ref: ccc_mvgarch.m:211 — normalize to correlation matrix
    diag_R = np.diag(R).copy()
    diag_R = np.maximum(diag_R, 1e-20)
    R = R / np.sqrt(np.outer(diag_R, diag_R))

    # ==================================================================
    # Compute conditional covariance Ht and log-likelihood
    # Ref: ccc_mvgarch.m:212-219
    # ==================================================================
    Ht = R[:, :, np.newaxis] * htArray

    ll = 0.0
    likConst = k * np.log(2.0 * np.pi)
    for t in range(T_obs):
        Ht_t = Ht[:, :, t]
        sign, logdet = np.linalg.slogdet(Ht_t)
        if sign <= 0:
            logdet = 30.0 * k  # Penalty for non-PD
        try:
            Ht_inv = np.linalg.solve(Ht_t, np.eye(k))
        except np.linalg.LinAlgError:
            Ht_inv = np.linalg.pinv(Ht_t)
        ll += 0.5 * (likConst + logdet
                      + np.sum(np.diag(Ht_inv @ data_3d[:, :, t])))

    # Ref: ccc_mvgarch.m:219 — negate (convention: positive ll = better fit)
    ll = -ll

    # ==================================================================
    # Format parameters
    # Ref: ccc_mvgarch.m:222-231
    # ==================================================================
    n_garch_params = int(np.sum(p_arr) + np.sum(o_arr) + np.sum(q_arr) + k)
    n_corr_params = k * (k - 1) // 2
    v = n_garch_params + n_corr_params
    parameters_out = np.zeros(v, dtype=np.float64)

    offset = 0
    for i in range(k):
        u = univariate[i]
        count = 1 + int(p_arr[i]) + int(o_arr[i]) + int(q_arr[i])
        parameters_out[offset:offset + count] = u['parameters']
        offset += count

    corr_start = offset
    corr_end = offset + n_corr_params
    parameters_out[corr_start:corr_end] = _corr_vech(R)

    # ==================================================================
    # VCV computation
    # Ref: ccc_mvgarch.m:234-254
    # ==================================================================
    A_mat = np.zeros((v, v), dtype=np.float64)
    scores_mat = np.zeros((T_obs, v), dtype=np.float64)

    # Ref: ccc_mvgarch.m:239-246 — fill block-diagonal A and scores
    offset = 0
    for i in range(k):
        u = univariate[i]
        count = 1 + int(p_arr[i]) + int(o_arr[i]) + int(q_arr[i])
        idx_s = offset
        idx_e = offset + count
        offset += count
        A_mat[idx_s:idx_e, idx_s:idx_e] = u['A']
        scores_mat[:, idx_s:idx_e] = u['scores']

    # Ref: ccc_mvgarch.m:247 — Hessian rows for correlation parameters
    def _inf_obj_scalar(params, data_arg, data_asym_arg, univ_arg):
        obj, _ = _ccc_inference_objective(params, data_arg, data_asym_arg, univ_arg)
        return obj

    def _inf_obj_with_lls(params, data_arg, data_asym_arg, univ_arg):
        return _ccc_inference_objective(params, data_arg, data_asym_arg, univ_arg)

    H_corr = _hessian_2sided_nrows(
        _inf_obj_scalar, parameters_out, n_corr_params,
        data_3d, data_asym, univariate
    )

    # Ref: ccc_mvgarch.m:248 — gradient scores for all parameters
    _, s = _gradient_2sided(
        _inf_obj_with_lls, parameters_out,
        data_3d, data_asym, univariate
    )

    # Ref: ccc_mvgarch.m:249 — replace correlation columns in scores
    scores_mat[:, corr_start:corr_end] = s[:, corr_start:corr_end]

    # Ref: ccc_mvgarch.m:250 — correlation rows of A
    A_mat[corr_start:corr_end, :] = H_corr / T_obs

    # Ref: ccc_mvgarch.m:251-253 — sandwich VCV
    B_mat = _covnw(scores_mat, nlag=0)
    try:
        A_inv = np.linalg.solve(A_mat, np.eye(v))
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A_mat)
    VCV = (A_inv @ B_mat @ A_inv.T) / T_obs

    return parameters_out, ll, Ht, VCV, scores_mat
