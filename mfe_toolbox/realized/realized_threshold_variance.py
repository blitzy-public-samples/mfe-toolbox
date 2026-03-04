"""
Threshold realized variance estimator with adaptive local variance proxy.

Implements the threshold realized variance (TRV) estimator of Mancini (2009)
with the iterative adaptive local variance proxy and analytic bias correction
of Corsi, Pirino, and Renò (2010).  Jumps are identified when squared returns
exceed ``c² × V_i``, where *c* is the threshold constant and *V_i* is the
local instantaneous variance estimated via Gaussian kernel smoothing.

The estimator supports subsampled averaging for additional bias reduction.

Migrated from: ``realized/realized_threshold_variance.m`` (MFE Toolbox v4.0)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 5/1/2008

See Also
--------
mfe_toolbox.realized.realized_variance : Standard realized variance.
mfe_toolbox.realized.realized_bipower_variation : Bipower variation estimator.
mfe_toolbox.realized.realized_kernel : Realized kernel estimator.
mfe_toolbox.realized.realized_quantile_variance : Quantile realized variance.
mfe_toolbox.realized.realized_range : Range-based realized variance.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import warnings

import numpy as np
from scipy.special import gamma as _gamma
from scipy.special import gammaincc as _gammaincc
from scipy.stats import norm as _norm

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_subsample import realized_subsample


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_gaussian_kernel(L: int, zero_half_width: int) -> np.ndarray:
    """Build a Gaussian kernel weight array with a zeroed central band.

    Parameters
    ----------
    L : int
        Kernel radius.  The weight array has ``2*L + 1`` elements
        corresponding to offsets ``-L`` through ``+L``.
    zero_half_width : int
        The number of elements on each side of the zero band.  The MATLAB
        source uses ``K(L + (-zero_half_width:zero_half_width)) = 0`` which,
        due to the 1-based index of the centre being at ``L + 1``, zeros
        offsets ``-(zero_half_width + 1)`` through ``(zero_half_width - 1)``.
        This asymmetry is faithfully reproduced for numerical parity.

    Returns
    -------
    np.ndarray
        1-D float64 weight array of length ``2*L + 1``.

    Notes
    -----
    Ref: realized_threshold_variance.m:186-188 — Gaussian kernel construction
    and central zeroing.  The zeroing formula ``K(L+(-s:s)) = 0`` in MATLAB
    (1-based) zeros offsets ``-(s+1)`` to ``(s-1)`` rather than ``-s`` to
    ``+s`` because the centre element is at MATLAB index ``L+1``, not ``L``.
    This off-by-one is preserved for strict numerical parity.
    """
    offsets = np.arange(-L, L + 1, dtype=np.float64)
    # Ref: realized_threshold_variance.m:187
    K = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (offsets / L) ** 2)
    # Ref: realized_threshold_variance.m:188 — K(L+(-s:s)) = 0
    # MATLAB 1-based indices [L-s, L+s] → Python 0-based [L-s-1, L+s-1]
    # Slice: K[L - zero_half_width - 1 : L + zero_half_width]
    lo = L - zero_half_width - 1
    hi = L + zero_half_width
    if lo < 0:
        lo = 0
    K[lo:hi] = 0.0
    return K


def _adaptive_local_variance(
    returns2: np.ndarray,
    c: float,
    L: int,
    zero_half_width: int,
    max_iter: int = 1000,
) -> np.ndarray:
    """Compute iterative adaptive local variance proxy via Gaussian kernel.

    Parameters
    ----------
    returns2 : np.ndarray
        1-D array of squared returns (length *m*).
    c : float
        Threshold scale constant.
    L : int
        Kernel radius.
    zero_half_width : int
        Half-width of the central zero band in the kernel (typically 1 for
        non-subsampled, or ``subsamples`` for the subsampled version).
    max_iter : int, optional
        Maximum number of iterations.  Default 1000.

    Returns
    -------
    np.ndarray
        1-D local variance proxy array of length *m*.

    Notes
    -----
    Ref: realized_threshold_variance.m:191-205 — iterative convergence loop.
    MATLAB uses ``all(V == Vold)`` for convergence (exact equality).
    """
    m = len(returns2)
    V = np.full(m, np.inf, dtype=np.float64)
    K = _build_gaussian_kernel(L, zero_half_width)
    c2 = c * c

    for _ in range(max_iter):
        # Ref: realized_threshold_variance.m:193
        ind = (returns2 < (c2 * V)).astype(np.float64)
        V_old = V.copy()

        for i in range(m):
            # Ref: realized_threshold_variance.m:196 — pl = i-L:i+L (1-based)
            # Python 0-based: pl = i-L to i+L
            pl = np.arange(i - L, i + L + 1)
            # Ref: realized_threshold_variance.m:197 — valid = pl>1 & pl<=m
            # MATLAB 1-based: pl>1 means pl>=2 → Python 0-based: pl>=1 → pl>0
            # MATLAB pl<=m → Python 0-based: pl<=m-1 → pl<m
            valid = (pl > 0) & (pl < m)
            if not np.any(valid):
                continue
            valid_pl = pl[valid]
            temp_ind = ind[valid_pl]
            w = K[valid]
            # Ref: realized_threshold_variance.m:200
            # V(i) = w*(returns2(pl(valid)).*tempInd)/(w*tempInd)
            denom = np.dot(w, temp_ind)
            if denom > 0.0:
                V[i] = np.dot(w, returns2[valid_pl] * temp_ind) / denom
            else:
                # Ref: MATLAB 0/0 = NaN — replicate for IEEE parity
                V[i] = np.nan

        # Ref: realized_threshold_variance.m:202-204
        if np.all(V == V_old):
            break
    else:
        warnings.warn(
            'Adaptive local variance did not converge within '
            f'{max_iter} iterations.',
            stacklevel=2,
        )

    return V


def _compute_bias_correction(c: float, V: np.ndarray) -> np.ndarray:
    """Compute analytic bias correction following Corsi, Pirino, and Renò.

    For each return classified as a jump (squared return exceeds threshold),
    the expected value ``E[r² | r² > c²·V_i]`` is computed analytically using
    the gamma function and the upper regularized incomplete gamma function.

    Parameters
    ----------
    c : float
        Threshold scale constant.
    V : np.ndarray
        Local variance proxy array.

    Returns
    -------
    np.ndarray
        Expected value of squared return conditional on exceeding threshold,
        element-wise for each observation.

    Notes
    -----
    Ref: realized_threshold_variance.m:212 — bias correction formula.

    CRITICAL: MATLAB ``gammainc(x, a, 'upper')`` maps to
    ``scipy.special.gammaincc(a, x)`` with REVERSED argument order.
    MATLAB ``gamma(a)`` maps to ``scipy.special.gamma(a)`` (same order).
    MATLAB ``normcdf(x)`` (from duplication/) maps to
    ``scipy.stats.norm.cdf(x)``.
    """
    # Ref: realized_threshold_variance.m:212
    # expectedValue = 1/(2*normcdf(-c)*sqrt(pi)) * (2/c^2) *
    #     gamma(3/2) .* gammainc(c^2/2, 3/2, 'upper') * c^2 * V
    #
    # The formula is computed exactly as in MATLAB to avoid floating-point
    # differences from algebraic simplification.
    c2 = c * c
    normcdf_neg_c = _norm.cdf(-c)
    gamma_1p5 = _gamma(1.5)
    # CRITICAL: gammaincc(a, x) — argument order reversed from MATLAB gammainc(x, a)
    # Ref: realized_threshold_variance.m:212 — gammainc(c^2/2, 3/2, 'upper')
    upper_inc_gamma = _gammaincc(1.5, c2 / 2.0)

    scalar_factor = (
        (1.0 / (2.0 * normcdf_neg_c * np.sqrt(np.pi)))
        * (2.0 / c2)
        * gamma_1p5
        * upper_inc_gamma
        * c2
    )
    return scalar_factor * V


def _interleave_subsampled_returns(
    returns_mat: np.ndarray,
) -> np.ndarray:
    """Interleave subsampled returns in MATLAB column-major order.

    Parameters
    ----------
    returns_mat : np.ndarray
        2-D array of shape ``(m_orig, subsamples)`` where NaN marks padding.

    Returns
    -------
    np.ndarray
        1-D array of non-NaN returns in MATLAB-compatible interleaved order
        (for each time point, cycle through all subsamples).

    Notes
    -----
    Ref: realized_threshold_variance.m:230-234.
    MATLAB: ``returns = returns'; returns = returns(~isnan(returns));``
    extracts non-NaN elements in column-major order from the transposed
    ``(subsamples, m_orig)`` matrix.
    """
    # Ref: realized_threshold_variance.m:230 — returns = returns'
    returns_t = returns_mat.T  # (subsamples, m_orig)
    # Ref: realized_threshold_variance.m:231 — column-major extraction
    flat = returns_t.ravel(order='F')
    result = flat[~np.isnan(flat)]
    return result


def _interpolate_local_var_for_subsamples(
    local_var: np.ndarray,
    subsamples: int,
    n_returns_ss: int,
) -> np.ndarray:
    """Linearly interpolate user-provided local variance for subsampled grids.

    Parameters
    ----------
    local_var : np.ndarray
        1-D local variance array of length ``m_orig`` (from non-subsampled).
    subsamples : int
        Number of subsamples.
    n_returns_ss : int
        Length of the subsampled returns vector (for truncation).

    Returns
    -------
    np.ndarray
        1-D interpolated and interleaved local variance array of length
        ``n_returns_ss``.

    Notes
    -----
    Ref: realized_threshold_variance.m:261-269.
    """
    m_orig = len(local_var)
    # Ref: realized_threshold_variance.m:262
    V_mat = np.tile(local_var.reshape(-1, 1), (1, subsamples))

    # Ref: realized_threshold_variance.m:263-266
    for i in range(1, subsamples):
        # MATLAB i goes 2..subsamples; (i-1)/subsamples
        # Python i goes 1..subsamples-1; i/subsamples
        w = i / subsamples
        V_mat[: m_orig - 1, i] = (
            (1.0 - w) * V_mat[: m_orig - 1, 0]
            + w * V_mat[1:m_orig, 0]
        )

    # Ref: realized_threshold_variance.m:267-269
    # V=V'; V=V(:); V=V(1:length(returns));
    V_t = V_mat.T  # (subsamples, m_orig)
    V_flat = V_t.ravel(order='F')  # column-major interleave
    return V_flat[:n_returns_ss].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def realized_threshold_variance(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    subsamples: int = 1,
    options: dict | None = None,
) -> tuple[float, float, float, float, dict]:
    """Estimate realized variance using thresholding to remove jumps.

    Computes the threshold realized variance (TRV) following Mancini (2009),
    with an iterative adaptive local variance proxy and analytic bias
    correction following Corsi, Pirino, and Renò (2010).  Returns both
    the plain and subsampled versions, with and without debiasing.

    Parameters
    ----------
    price : array_like
        1-D array of high-frequency prices (not log-prices).
    time : array_like or None, optional
        1-D array of observation times corresponding to *price*.  Must be
        sorted and non-decreasing.  If ``None``, a uniform unit-interval
        grid ``[0, 1]`` is generated automatically (requires
        ``time_type='unit'``).
    time_type : str, optional
        Time format.  One of ``'wall'``, ``'seconds'``, ``'unit'``.
        Default ``'unit'``.
    sampling_type : str, optional
        Sampling scheme.  One of ``'CalendarTime'``, ``'CalendarUniform'``,
        ``'BusinessTime'``, ``'BusinessUniform'``, ``'Fixed'``.
        Default ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Sampling parameter whose interpretation depends on *sampling_type*.
        Default ``1``.
    subsamples : int, optional
        Number of subsampled grids to average.  ``1`` uses only the primary
        grid.  Default ``1``.
    options : dict or None, optional
        Configuration dict with optional keys:

        * ``'thresholdConstant'`` (float) — Scale for the threshold
          (``c`` in ``c² · V_i``).  Default ``3.0``.
        * ``'kernelRadius'`` (int) — Half-width of the Gaussian smoothing
          kernel.  Default ``25``.
        * ``'localVar'`` (np.ndarray or None) — Pre-computed local
          variance array of length equal to the number of filtered returns.
          If ``None``, the adaptive kernel estimator is used.
        * ``'maxIterations'`` (int) — Maximum iterations for the adaptive
          local variance convergence loop.  Default ``1000``.

    Returns
    -------
    trv : float
        Threshold realized variance (plain, no subsampling).
    trv_ss : float
        Subsampled threshold realized variance.
    trv_debiased : float
        Debiased threshold realized variance (following Corsi–Pirino–Renò).
    trv_ss_debiased : float
        Debiased subsampled threshold realized variance.
    diagnostics : dict
        Diagnostic information with keys:

        * ``'localVar'`` — Local variance proxy (non-subsampled).
        * ``'returns'`` — Filtered returns (non-subsampled).
        * ``'localVarSS'`` — Local variance proxy (subsampled).
        * ``'returnsSS'`` — Interleaved subsampled returns.

    Raises
    ------
    ValueError
        If inputs fail validation.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> prices = 100.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(500)))
    >>> times = np.linspace(0, 1, 500)
    >>> trv, trv_ss, trv_d, trv_ss_d, diag = realized_threshold_variance(
    ...     prices, times, 'unit', 'CalendarUniform', 78)

    Notes
    -----
    The debiasing method follows Corsi, Pirino, and Renò (2010).  For each
    return classified as a jump, the expected value ``E[r² | jump]`` is
    computed analytically using the gamma function and the upper regularized
    incomplete gamma function and added back to the TRV estimate.

    The number of returns equals ``len(filtered_price) - 1`` where
    ``filtered_price`` is obtained from
    :func:`~mfe_toolbox.realized.realized_price_filter.realized_price_filter`.
    """
    # ==================================================================
    # 1. Parse options
    # ==================================================================
    if options is None:
        options = {}
    c = float(options.get('thresholdConstant', 3.0))
    kernel_radius = int(options.get('kernelRadius', 25))
    local_var_input = options.get('localVar', None)
    max_iter = int(options.get('maxIterations', 1000))

    # ==================================================================
    # 2. Input validation
    # Ref: realized_threshold_variance.m:77-170
    # ==================================================================

    # --- price ---
    price = np.asarray(price, dtype=np.float64).ravel()
    if price.ndim != 1 or price.size < 2:
        raise ValueError('PRICE must be a 1-D vector with at least 2 elements.')

    # --- time ---
    if time is None:
        # Generate uniform unit-interval time if not provided
        time = np.linspace(0.0, 1.0, len(price))
    time = np.asarray(time, dtype=np.float64).ravel()

    # Ref: realized_threshold_variance.m:98-100
    if time.shape[0] != price.shape[0]:
        raise ValueError('TIME must be a 1-D vector with the same length as PRICE.')
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing.')

    # Ref: realized_threshold_variance.m:108 — protect against integer times
    time = time.astype(np.float64, copy=False)

    # --- time_type ---
    # Ref: realized_threshold_variance.m:110-113
    time_type_lower = time_type.lower()
    if time_type_lower not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # --- sampling_type ---
    # Ref: realized_threshold_variance.m:114-117
    sampling_type_lower = sampling_type.lower()
    _valid_st = (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    )
    if sampling_type_lower not in _valid_st:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # --- sampling_interval ---
    # Ref: realized_threshold_variance.m:119-143
    m_price = len(price)
    t0 = time[0]
    tT = time[m_price - 1]

    if sampling_type_lower in ('calendartime', 'calendaruniform',
                                'businesstime', 'businessuniform'):
        if time_type_lower in ('wall', 'seconds'):
            si_val = float(sampling_interval)
            if (not np.isscalar(sampling_interval)
                    or np.floor(si_val) != si_val
                    or si_val < 1):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for the '
                    "SAMPLINGTYPE selected when using 'wall' or 'seconds' "
                    'as TIMETYPE.'
                )
        else:
            if not np.isscalar(sampling_interval) or float(sampling_interval) < 0:
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive value for the '
                    "SAMPLINGTYPE selected when using 'unit' as TIMETYPE."
                )
    else:
        # Fixed sampling
        sampling_interval = np.asarray(sampling_interval, dtype=np.float64).ravel()
        if not (np.any(sampling_interval >= t0) and np.any(sampling_interval <= tT)):
            raise ValueError(
                'At least one sampling interval must be between min(TIME) '
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )
        if len(sampling_interval) > 1 and np.any(np.diff(sampling_interval) <= 0):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                'times in SAMPLINGINTERVAL must be sorted and strictly '
                'increasing.'
            )

    # --- thresholdConstant (c) ---
    # Ref: realized_threshold_variance.m:147-152
    if c <= 0.0:
        raise ValueError('thresholdConstant (c) must be a positive scalar.')

    # --- localVar ---
    # Ref: realized_threshold_variance.m:155-160
    if local_var_input is not None:
        local_var_input = np.asarray(local_var_input, dtype=np.float64).ravel()

    # --- subsamples ---
    # Ref: realized_threshold_variance.m:163-170
    if subsamples < 0 or int(subsamples) != subsamples:
        raise ValueError('SUBSAMPLES must be a non-negative integer.')
    subsamples = int(subsamples)
    if subsamples == 0:
        subsamples = 1

    # ==================================================================
    # 3. Compute log-prices, filter, and returns
    # Ref: realized_threshold_variance.m:177-182
    # ==================================================================

    log_price = np.log(price)

    # Ref: realized_threshold_variance.m:179
    filtered_result = realized_price_filter(
        log_price, time, time_type, sampling_type, sampling_interval
    )
    # realized_price_filter returns (filtered_price, filtered_time, actual_time)
    filtered_log_price = filtered_result[0]

    # Ref: realized_threshold_variance.m:180
    returns = np.diff(filtered_log_price)
    m = len(returns)
    # Ref: realized_threshold_variance.m:182 — always compute returns2
    # (MATLAB only computes inside the if-block, creating a bug when
    # localVar is provided; we compute it here for correctness)
    returns2 = returns ** 2

    # ==================================================================
    # 4. Compute local variance proxy (non-subsampled)
    # Ref: realized_threshold_variance.m:183-211
    # ==================================================================

    if local_var_input is None:
        # Ref: realized_threshold_variance.m:184-205
        L = kernel_radius
        V = _adaptive_local_variance(
            returns2, c, L, zero_half_width=1, max_iter=max_iter
        )
    else:
        V = local_var_input.copy()
        if len(V) != m:
            raise ValueError(
                'LOCALVAR must have the same number of returns as the '
                f'filtered series. Expected {m}, got {len(V)}.'
            )

    # ==================================================================
    # 5. Bias correction, threshold, and TRV (non-subsampled)
    # Ref: realized_threshold_variance.m:212-217
    # ==================================================================

    # Ref: realized_threshold_variance.m:212
    expected_value = _compute_bias_correction(c, V)

    # Ref: realized_threshold_variance.m:214
    threshold = returns2 > (c * c * V)
    m_orig = m

    # Ref: realized_threshold_variance.m:216
    non_jump_returns = returns[~threshold]
    trv = float(np.dot(non_jump_returns, non_jump_returns))

    # Ref: realized_threshold_variance.m:217
    trv_debiased = trv + float(np.sum(expected_value[threshold]))

    # ==================================================================
    # 6. Diagnostics (non-subsampled)
    # Ref: realized_threshold_variance.m:219-220
    # ==================================================================
    diagnostics: dict = {
        'localVar': V.copy(),
        'returns': returns.copy(),
    }

    # ==================================================================
    # 7. Subsampled returns
    # Ref: realized_threshold_variance.m:223-234
    # ==================================================================

    # Ref: realized_threshold_variance.m:223
    subsampled_data = realized_subsample(
        log_price, time, time_type, sampling_type,
        sampling_interval, subsamples
    )

    # Ref: realized_threshold_variance.m:225-229
    returns_mat = np.full((m_orig, subsamples), np.nan)
    for i in range(subsamples):
        # Ref: realized_threshold_variance.m:227
        temp = np.diff(subsampled_data[i][0])
        returns_mat[:len(temp), i] = temp

    # Ref: realized_threshold_variance.m:230-234
    returns_ss = _interleave_subsampled_returns(returns_mat)
    # Ref: realized_threshold_variance.m:232-234 — ensure column vector
    # In Python returns_ss is already 1-D; no transpose needed.

    # ==================================================================
    # 8. Subsampled local variance proxy
    # Ref: realized_threshold_variance.m:236-270
    # ==================================================================
    returns2_ss = returns_ss ** 2
    m_ss = len(returns_ss)

    if local_var_input is None:
        # Ref: realized_threshold_variance.m:237-259
        L_ss = subsamples * kernel_radius
        V_ss = _adaptive_local_variance(
            returns2_ss, c, L_ss,
            zero_half_width=subsamples,
            max_iter=max_iter,
        )
    else:
        # Ref: realized_threshold_variance.m:260-269
        V_ss = _interpolate_local_var_for_subsamples(
            local_var_input, subsamples, m_ss
        )

    # ==================================================================
    # 9. Subsampled diagnostics
    # Ref: realized_threshold_variance.m:271-272
    # ==================================================================
    diagnostics['localVarSS'] = V_ss.copy()
    diagnostics['returnsSS'] = returns_ss.copy()

    # ==================================================================
    # 10. Bias correction, threshold, and TRV (subsampled)
    # Ref: realized_threshold_variance.m:273-277
    # ==================================================================

    # Ref: realized_threshold_variance.m:273
    expected_value_ss = _compute_bias_correction(c, V_ss)

    # Ref: realized_threshold_variance.m:275
    threshold_ss = returns2_ss > (c * c * V_ss)

    # Ref: realized_threshold_variance.m:276
    non_jump_ss = returns_ss[~threshold_ss]
    scale_factor = m_orig / m_ss if m_ss > 0 else 0.0
    trv_ss = float(scale_factor * np.dot(non_jump_ss, non_jump_ss))

    # Ref: realized_threshold_variance.m:277
    trv_ss_debiased = float(
        scale_factor * (
            np.dot(non_jump_ss, non_jump_ss)
            + np.sum(expected_value_ss[threshold_ss])
        )
    )

    return trv, trv_ss, trv_debiased, trv_ss_debiased, diagnostics
