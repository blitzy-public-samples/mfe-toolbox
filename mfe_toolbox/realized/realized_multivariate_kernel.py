"""
Multivariate realized kernel covariance estimator.

Implements the multivariate version of the realized kernel of Barndorff-Nielsen,
Hansen, Lunde, and Shephard (BNHLS) using non-flat-top kernels and end-point
jittering (pre-averaging) to produce a positive semi-definite covariance matrix
from multiple high-frequency price series.

Calling ``realized_multivariate_kernel`` with a single price series is equivalent
to calling ``realized_kernel`` with ``'BusinessTime'`` sampling.

The only sampling scheme available is refresh time.  The only end-point treatment
is jittering (pre-averaging), which is required to ensure the multivariate
realized kernels are positive semi-definite.

Migrated from: realized/realized_multivariate_kernel.m (495 lines)
Original author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/1/2008

See Also
--------
realized_covariance, realized_hayashi_yoshida, realized_kernel,
realized_variance, realized_quantile_variance, realized_range

References
----------
Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A. and Shephard, N. (2011).
    "Multivariate realised kernels: Consistent positive semi-definite estimators
    of the covariation of equity prices with noise and non-synchronous trading."
    *Journal of Econometrics*, 162(2), 149-169.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_kernel_bandwidth import realized_kernel_bandwidth
from mfe_toolbox.realized.realized_kernel_jitter_lag_length import realized_kernel_jitter_lag_length
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_kernel_weights import realized_kernel_weights
from mfe_toolbox.realized.realized_refresh_time import realized_refresh_time

# ---------------------------------------------------------------------------
# Kernel classification lists (mirrors realized_multivariate_kernel.m:394-402)
# ---------------------------------------------------------------------------
# Ref: realized_multivariate_kernel.m:394-396 — Flat-top kernel list
_FLAT_TOP_KERNEL_LIST: list[str] = [
    'bartlett', 'twoscale', '2ndorder', 'epanechnikov',
    'cubic', 'multiscale', '5thorder', '6thorder', '7thorder', '8thorder',
    'parzen', 'th1', 'th2', 'th5', 'th16',
]

# Ref: realized_multivariate_kernel.m:399 — Non-flat-top kernel list
_NON_FLAT_TOP_KERNEL_LIST: list[str] = [
    'nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls',
]

# Ref: realized_multivariate_kernel.m:402 — Combined kernel list
_KERNEL_LIST: list[str] = _FLAT_TOP_KERNEL_LIST + _NON_FLAT_TOP_KERNEL_LIST

# Valid sampling types for options validation
# Ref: realized_multivariate_kernel.m:469
_VALID_SAMPLING_TYPES: frozenset[str] = frozenset({
    'calendartime', 'calendaruniform', 'businesstime', 'businessuniform', 'fixed',
})


# ---------------------------------------------------------------------------
# Local validation helpers
# Ref: realized_multivariate_kernel.m:488-495
# ---------------------------------------------------------------------------

def _is_nonnegative_scalar_integer(x: object) -> bool:
    """Return True if *x* is a non-empty, scalar integer >= 0.

    Ref: realized_multivariate_kernel.m:488-490
    """
    if x is None:
        return False
    try:
        val = float(x)
    except (TypeError, ValueError):
        return False
    return np.isscalar(x) and val >= 0.0 and val == int(val)


def _is_nonnegative_scalar(x: object) -> bool:
    """Return True if *x* is a non-empty scalar >= 0.

    Ref: realized_multivariate_kernel.m:493-495
    """
    if x is None:
        return False
    try:
        val = float(x)
    except (TypeError, ValueError):
        return False
    return np.isscalar(x) and val >= 0.0


# ---------------------------------------------------------------------------
# Core kernel computation
# Ref: realized_multivariate_kernel.m:209-280
# ---------------------------------------------------------------------------

def _realized_multivariate_kernel_core(
    returns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute the multivariate realized kernel from returns and weights.

    Builds the lagged autocovariance tensor ``gammaH``, computes the 0-lag
    autocovariance ``gamma0``, and returns::

        rmk = gamma0 + sum_j weights[j] * (gammaH[:,:,j] + gammaH[:,:,j].T)

    Parameters
    ----------
    returns : np.ndarray
        ``(m, k)`` matrix of log-returns for *k* assets over *m* periods.
    weights : np.ndarray
        ``(H,)`` vector of kernel weights for lags 1, 2, ..., H.

    Returns
    -------
    np.ndarray
        ``(k, k)`` realized multivariate kernel covariance matrix.
    """
    # ------------------------------------------------------------------
    # Input checking
    # Ref: realized_multivariate_kernel.m:237-255
    # ------------------------------------------------------------------
    if weights.ndim > 1 and weights.shape[1] > 1:
        raise ValueError('WEIGHTS must be a H by 1 vector.')

    weights = np.asarray(weights, dtype=np.float64).ravel()

    # Ref: realized_multivariate_kernel.m:246
    m, k = returns.shape

    # Ref: realized_multivariate_kernel.m:247-249
    if len(weights) >= m:
        warnings.warn(
            'The length of WEIGHTS is longer than the length of RETURNS.\n'
            '  The weights are being truncated at N-1 where N is the '
            'number of returns',
            stacklevel=2,
        )
        weights = weights[: m - 1]
    # Ref: realized_multivariate_kernel.m:250-251
    elif len(weights) > 0.1 * m:
        warnings.warn(
            'The number of WEIGHTS may be excessive.  '
            'Consider using a smaller kernel.',
            stacklevel=2,
        )

    # Ref: realized_multivariate_kernel.m:258
    H = len(weights)

    # ------------------------------------------------------------------
    # Compute lagged autocovariance tensor
    # Ref: realized_multivariate_kernel.m:262-268
    # ------------------------------------------------------------------
    gamma_h = np.zeros((k, k, H), dtype=np.float64)
    for i in range(H):
        # Ref: realized_multivariate_kernel.m:265 — MATLAB loop i=1:H, 1-based
        lag = i + 1  # equivalent MATLAB loop variable
        returns_minus = returns[: m - lag, :]   # returns(1:m-i,:)
        returns_plus = returns[lag:, :]          # returns(i+1:m,:)
        # Ref: realized_multivariate_kernel.m:267
        gamma_h[:, :, i] = returns_minus.T @ returns_plus

    # Ref: realized_multivariate_kernel.m:270
    gamma0 = returns.T @ returns

    # ------------------------------------------------------------------
    # Construct the kernel
    # Ref: realized_multivariate_kernel.m:273-280
    # ------------------------------------------------------------------
    rmk = gamma0.copy()
    if len(weights) > 0:
        for j in range(len(weights)):
            # Ref: realized_multivariate_kernel.m:276
            rmk = rmk + weights[j] * (gamma_h[:, :, j] + gamma_h[:, :, j].T)
    else:
        warnings.warn(
            'The number of lags used was 0.',
            stacklevel=2,
        )

    return rmk


# ---------------------------------------------------------------------------
# Parameter validation (inline, replacing embedded MATLAB subfunction)
# Ref: realized_multivariate_kernel.m:305-495
# ---------------------------------------------------------------------------

def _validate_options(
    options: dict,
    time_type: str,
) -> str | None:
    """Validate realized kernel options dict.  Returns error message or None.

    Ref: realized_multivariate_kernel.m:393-483
    """
    # Ref: realized_multivariate_kernel.m:405-413 — fill missing fields
    default_options = realized_options('Kernel')
    for key, value in default_options.items():
        if key not in options:
            options[key] = value

    # Ref: realized_multivariate_kernel.m:416-483 — per-field validation
    for field_name in list(options.keys()):
        field_value = options[field_name]

        # Lowercase string values
        # Ref: realized_multivariate_kernel.m:419-421
        if isinstance(field_value, str):
            field_value = field_value.lower()
            options[field_name] = field_value

        if field_name == 'kernel':
            # Ref: realized_multivariate_kernel.m:427 — must be non-flat-top
            if field_value not in _NON_FLAT_TOP_KERNEL_LIST:
                return f'OPTIONS.{field_name} must be one of the listed types.'

        elif field_name == 'medFrequencyKernel':
            # Ref: realized_multivariate_kernel.m:432-434
            if field_value not in _KERNEL_LIST:
                return f'OPTIONS.{field_name} must be one of the listed types.'

        elif field_name in ('bandwidth', 'medFrequencyBandwidth'):
            # Ref: realized_multivariate_kernel.m:438-441
            if field_value is not None and not _is_nonnegative_scalar(field_value):
                return f'OPTIONS.{field_name} must a non-negative scalar.'

        elif field_name == 'endTreatment':
            # Ref: realized_multivariate_kernel.m:444-447 — must be 'jitter'
            if field_value != 'jitter':
                return "OPTIONS.endTreatment must be 'Jitter'."

        elif field_name in ('jitterLags', 'maxBandwidth'):
            # Ref: realized_multivariate_kernel.m:450-453
            if field_value is not None and not _is_nonnegative_scalar_integer(field_value):
                return f'OPTIONS.{field_name} must be a non-negative scalar integer.'

        elif field_name in ('useDebiasedNoise', 'useAdjustedNoiseCount'):
            # Ref: realized_multivariate_kernel.m:456-459
            if not isinstance(field_value, (bool, int, np.bool_)):
                return 'OPTIONS.useDebiasedNoise must be a logical value.'
            if isinstance(field_value, (int, np.integer)) and field_value not in (0, 1):
                return 'OPTIONS.useDebiasedNoise must be a logical value.'

        elif field_name == 'maxBandwidthPerc':
            # Ref: realized_multivariate_kernel.m:462-465
            if field_value is not None:
                if not _is_nonnegative_scalar(field_value):
                    return (
                        'OPTIONS.maxBandwidthPerc must be a non-negative '
                        'scalar between 0 and 1.'
                    )
                if float(field_value) > 1:
                    return (
                        'OPTIONS.maxBandwidthPerc must be a non-negative '
                        'scalar between 0 and 1.'
                    )

        elif field_name in (
            'IQEstimationSamplingType',
            'medFrequencySamplingType',
            'noiseVarianceSamplingType',
        ):
            # Ref: realized_multivariate_kernel.m:468-471
            if field_value not in _VALID_SAMPLING_TYPES:
                return (
                    f"OPTIONS.{field_name} must be one of 'CalendarTime', "
                    f"'CalendarUniform', 'BusinessTime', 'BusinessUniform' "
                    f"or 'Fixed'."
                )

        elif field_name in (
            'medFrequencySamplingInterval',
            'noiseVarianceSamplingInterval',
            'IQEstimationSamplingInterval',
        ):
            # Ref: realized_multivariate_kernel.m:474-482
            if field_value is None or not _is_nonnegative_scalar(field_value):
                return (
                    f'OPTIONS.{field_name} must be a non-negative scalar '
                    f'between 0 and 1.'
                )
            if time_type == 'unit' and float(field_value) > 1:
                return (
                    f"OPTIONS.{field_name} must be less than 1 if "
                    f"TIMETYPE when 'unit'."
                )

    return None  # no error


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def realized_multivariate_kernel(
    prices: list[np.ndarray] | np.ndarray,
    times: list[np.ndarray] | np.ndarray,
    time_type: str,
    sampling_interval: int,
    options: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute the multivariate realized kernel covariance estimator.

    Produces a positive semi-definite (PSD) covariance matrix from multiple
    high-frequency price series using the BNHLS realized kernel with
    non-flat-top kernel functions and endpoint jitter pre-averaging.

    Parameters
    ----------
    prices : list of np.ndarray or np.ndarray
        Either a Python list of *K* one-dimensional price arrays (one per
        asset, possibly of different lengths), or a 2-D ``(m, K)`` array
        where each column is an asset's price series.
    times : list of np.ndarray or np.ndarray
        Corresponding time arrays.  A list of *K* one-dimensional arrays
        matching *prices*, or a 2-D ``(m, K)`` array.  Each time series
        must be sorted in ascending order.
    time_type : str
        Time measurement format:

        * ``'wall'``    — 24-hour clock HHMMSS (e.g. 101543)
        * ``'seconds'`` — Seconds past midnight
        * ``'unit'``    — Unit-normalised [0, 1]
    sampling_interval : int
        Scalar positive integer controlling subsampling of the
        refresh-time synchronized prices.  1 uses every observation
        (recommended).
    options : dict or None, optional
        Realized kernel options dictionary.  If *None* (default), defaults
        from ``realized_options('Multivariate Kernel')`` are used.  See
        :func:`realized_options` for valid fields.

    Returns
    -------
    rmk : np.ndarray
        ``(K, K)`` PSD covariance matrix.
    diagnostics : dict
        Diagnostic information with keys:

        * ``'kernel'``      — kernel type string
        * ``'bandwidth'``   — bandwidth used
        * ``'jitterLags'``  — per-asset jitter lag array
        * ``'weights'``     — kernel weight vector
        * ``'rtPrice'``     — refresh-time synchronised prices
        * ``'rtTime'``      — refresh-time timestamps

    Raises
    ------
    ValueError
        If any input fails validation (invalid dimensions, unsorted times,
        unrecognised kernel type, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> p1 = 100 + np.cumsum(rng.standard_normal(500) * 0.01)
    >>> p2 = 50 + np.cumsum(rng.standard_normal(500) * 0.01)
    >>> t = np.linspace(0.0, 1.0, 500)
    >>> rmk, diag = realized_multivariate_kernel([p1, p2], [t, t], 'unit', 1)
    >>> rmk.shape
    (2, 2)
    """

    # ==================================================================
    # Input Parsing
    # Ref: realized_multivariate_kernel.m:74-97
    # ==================================================================

    # --- At least 4 inputs required ---
    # Ref: realized_multivariate_kernel.m:74-76
    # (enforced by the function signature)

    # --- Handle cell-array / list / 2D-array input formats ---
    # Ref: realized_multivariate_kernel.m:79-91
    if isinstance(prices, np.ndarray) and prices.ndim == 2:
        # 2-D array input: columns are assets
        num_prices = prices.shape[1]
        prices_list: list[np.ndarray] = [
            np.asarray(prices[:, i], dtype=np.float64).ravel()
            for i in range(num_prices)
        ]
        if isinstance(times, np.ndarray) and times.ndim == 2:
            times_list: list[np.ndarray] = [
                np.asarray(times[:, i], dtype=np.float64).ravel()
                for i in range(num_prices)
            ]
        elif isinstance(times, (list, tuple)):
            times_list = [
                np.asarray(times[i], dtype=np.float64).ravel()
                for i in range(num_prices)
            ]
        else:
            raise ValueError(
                'TIMES must be a list of arrays or a 2D array matching PRICES.'
            )
    elif isinstance(prices, (list, tuple)):
        num_prices = len(prices)
        prices_list = [
            np.asarray(p, dtype=np.float64).ravel() for p in prices
        ]
        if not isinstance(times, (list, tuple)):
            raise ValueError('TIMES must be a list of arrays when PRICES is a list.')
        if len(times) != num_prices:
            raise ValueError(
                'TIMES must have the same number of elements as PRICES.'
            )
        times_list = [
            np.asarray(t, dtype=np.float64).ravel() for t in times
        ]
    else:
        raise ValueError(
            'PRICES must be a list of 1-D arrays or a 2-D numpy array.'
        )

    if num_prices < 1:
        raise ValueError('At least one price series is required.')

    # ==================================================================
    # Options Initialization
    # Ref: realized_multivariate_kernel.m:337-346
    # ==================================================================
    if options is None:
        # Ref: realized_multivariate_kernel.m:342
        options = realized_options('Multivariate Kernel')
    else:
        # Work on a copy to avoid mutating caller's dict
        options = dict(options)

    # ==================================================================
    # Validate Price/Time Pairs
    # Ref: realized_multivariate_kernel.m:353-376
    # ==================================================================
    for i in range(num_prices):
        price_i = prices_list[i]
        time_i = times_list[i]

        # Ref: realized_multivariate_kernel.m:356-362
        if price_i.ndim != 1 or price_i.size == 0:
            raise ValueError(
                f'PRICE series {i + 1} must be a m(i) by 1 vector.'
            )

        # Ref: realized_multivariate_kernel.m:368-369
        if np.any(np.diff(time_i) < 0):
            raise ValueError(
                f'TIME for series {i + 1} must be sorted and increasing'
            )

        # Ref: realized_multivariate_kernel.m:372-374
        if time_i.ndim != 1 or len(time_i) != len(price_i):
            raise ValueError(
                f'TIME for series {i + 1} must be a m(i) by 1 vector '
                f'the same size as PRICE series (i).'
            )

    # ==================================================================
    # Validate time_type
    # Ref: realized_multivariate_kernel.m:379-383
    # ==================================================================
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # ==================================================================
    # Validate sampling_interval
    # Ref: realized_multivariate_kernel.m:386-389
    # ==================================================================
    if not np.isscalar(sampling_interval) or int(sampling_interval) < 1:
        raise ValueError(
            'SAMPLINGINTERVAL must be a positive integer (usually 1).'
        )
    sampling_interval = int(sampling_interval)

    # ==================================================================
    # Validate options fields
    # Ref: realized_multivariate_kernel.m:393-483
    # ==================================================================
    err_msg = _validate_options(options, time_type)
    if err_msg is not None:
        raise ValueError(err_msg)

    # ==================================================================
    # Step 0: Quick filter using refresh time to estimate nMax
    # Ref: realized_multivariate_kernel.m:105-109
    # ==================================================================
    if num_prices >= 2:
        rt_prices_initial: np.ndarray = realized_refresh_time(
            prices_list, times_list, time_type,
        )[0]
    else:
        # Single asset: no synchronization needed
        # Ref: realized_multivariate_kernel.m:58-62
        rt_prices_initial = prices_list[0].reshape(-1, 1)

    # Ref: realized_multivariate_kernel.m:106-108 — subsample
    if sampling_interval != 1:
        rt_prices_initial = rt_prices_initial[::sampling_interval]

    # Ref: realized_multivariate_kernel.m:109 — double subsampling for nMax
    n_max: int = rt_prices_initial[::sampling_interval].shape[0]

    # Ref: realized_multivariate_kernel.m:114
    options['filteredN'] = n_max

    # ==================================================================
    # Steps 1-2: Determine bandwidth and jitter lags
    # Ref: realized_multivariate_kernel.m:116-170
    # ==================================================================
    jitter_lags_is_none = options.get('jitterLags') is None
    bandwidth_is_none = options.get('bandwidth') is None

    # Retrieve kernel name for jitter-lag computation
    # Ref: realized_multivariate_kernel.m:133 — 'kernel' variable used
    # (MATLAB code uses bare 'kernel' which is options.kernel)
    kernel_name: str = options['kernel']

    bandwidth_arr = np.zeros(num_prices, dtype=np.float64)
    jitter_lags = np.zeros(num_prices, dtype=np.int64)
    noise_variance = np.zeros(num_prices, dtype=np.float64)
    iq_estimate = np.zeros(num_prices, dtype=np.float64)

    if jitter_lags_is_none and bandwidth_is_none:
        # ---- Branch 1: Need both ----
        # Ref: realized_multivariate_kernel.m:116-136
        for i in range(num_prices):
            price_i = prices_list[i]
            time_i = times_list[i]

            # Ref: realized_multivariate_kernel.m:126
            nv_i, _dbg_nv, iq_i, _oomen = realized_noise_estimate(
                price_i, time_i, time_type, options,
            )
            noise_variance[i] = nv_i
            iq_estimate[i] = iq_i

            # Ref: realized_multivariate_kernel.m:127
            bandwidth_arr[i] = realized_kernel_bandwidth(
                nv_i, iq_i, options,
            )

            # Ref: realized_multivariate_kernel.m:129-131
            # Per-element bandwidth truncation at 25% of data
            if bandwidth_arr[i] > 0.25 * n_max:
                bandwidth_arr[i] = float(np.round(0.25 * n_max))
                warnings.warn(
                    'The estimated bandwidth requires a lag length larger '
                    'than 25% of the available data.  Bandwidth has been '
                    'truncated to 25% of data.',
                    stacklevel=2,
                )

            # Ref: realized_multivariate_kernel.m:133-134
            jitter_lags[i] = realized_kernel_jitter_lag_length(
                nv_i, iq_i, kernel_name, n_max,
            )
            jitter_lags[i] = max(jitter_lags[i], 1)

        # Ref: realized_multivariate_kernel.m:136
        options['bandwidth'] = int(np.round(float(np.mean(bandwidth_arr))))

    elif jitter_lags_is_none and not bandwidth_is_none:
        # ---- Branch 2: Need jitterLags only ----
        # Ref: realized_multivariate_kernel.m:137-150
        for i in range(num_prices):
            price_i = prices_list[i]
            time_i = times_list[i]

            # Ref: realized_multivariate_kernel.m:146
            nv_i, _dbg_nv, iq_i, _oomen = realized_noise_estimate(
                price_i, time_i, time_type, options,
            )
            noise_variance[i] = nv_i
            iq_estimate[i] = iq_i

            # Ref: realized_multivariate_kernel.m:148-149
            jitter_lags[i] = realized_kernel_jitter_lag_length(
                nv_i, iq_i, kernel_name, n_max,
            )
            jitter_lags[i] = max(jitter_lags[i], 1)

    elif not jitter_lags_is_none and bandwidth_is_none:
        # ---- Branch 3: Need bandwidth only ----
        # Ref: realized_multivariate_kernel.m:151-169
        for i in range(num_prices):
            price_i = prices_list[i]
            time_i = times_list[i]

            # Ref: realized_multivariate_kernel.m:160
            nv_i, _dbg_nv, iq_i, _oomen = realized_noise_estimate(
                price_i, time_i, time_type, options,
            )
            noise_variance[i] = nv_i
            iq_estimate[i] = iq_i

            # Ref: realized_multivariate_kernel.m:161
            bandwidth_arr[i] = realized_kernel_bandwidth(
                nv_i, iq_i, options,
            )

            # Ref: realized_multivariate_kernel.m:163-165
            if bandwidth_arr[i] > 0.25 * n_max:
                bandwidth_arr[i] = float(np.round(0.25 * n_max))
                warnings.warn(
                    'The estimated bandwidth requires a lag length larger '
                    'than 25% of the available data.  Bandwidth has been '
                    'truncated to 25% of data.',
                    stacklevel=2,
                )

        # Ref: realized_multivariate_kernel.m:168
        options['bandwidth'] = int(np.round(float(np.mean(bandwidth_arr))))
        # Ref: realized_multivariate_kernel.m:169
        jitter_lags = (
            np.ones(num_prices, dtype=np.int64) * int(options['jitterLags'])
        )

    else:
        # ---- Branch 4: Both provided (implicit) ----
        # The MATLAB code does not explicitly handle this branch, which
        # would leave 'jitterLags' undefined and cause a runtime error.
        # We handle it by expanding the scalar jitterLags to per-asset.
        jitter_lags = (
            np.ones(num_prices, dtype=np.int64) * int(options['jitterLags'])
        )

    # ==================================================================
    # Step 2b: Jitter pre-averaging of endpoints
    # Ref: realized_multivariate_kernel.m:172-184
    # ==================================================================
    jittered_prices: list[np.ndarray] = []
    jittered_times: list[np.ndarray] = []

    for i in range(num_prices):
        price_i = prices_list[i].copy()
        time_i = times_list[i].copy()
        m_i: int = len(price_i)
        jl_i: int = int(jitter_lags[i])

        # Guard: if jitter lags exceed or equal half the data, clamp
        if jl_i >= m_i // 2:
            jl_i = max(m_i // 2 - 1, 1)
            jitter_lags[i] = jl_i

        # Ref: realized_multivariate_kernel.m:177 — p0 = mean(price(1:jL))
        p0 = float(np.mean(price_i[:jl_i]))
        # Ref: realized_multivariate_kernel.m:178 — p1 = mean(price(m-jL+1:m))
        p1 = float(np.mean(price_i[m_i - jl_i:]))

        # Ref: realized_multivariate_kernel.m:179
        # t0 = time(ceil(mean(1:jL)))  — MATLAB 1-based indices
        # Mean of [1..jL] = (jL+1)/2; ceil → Python 0-based = ceil((jL+1)/2) - 1
        t0_idx = int(np.ceil((jl_i + 1) / 2.0)) - 1
        # Ref: realized_multivariate_kernel.m:180
        # t1 = time(floor(mean(m-jL+1:m)))  — MATLAB 1-based indices
        # Mean of [m-jL+1..m] = m - (jL-1)/2; floor → Python 0-based = floor(m-(jL-1)/2) - 1
        t1_idx = int(np.floor(m_i - (jl_i - 1) / 2.0)) - 1

        t0 = time_i[t0_idx]
        t1 = time_i[t1_idx]

        # Ref: realized_multivariate_kernel.m:181
        # price = [p0; price(jL+1 : m-jL); p1]  — MATLAB 1-based
        # Python: price[jl_i : m_i - jl_i] = indices jL (0-based) to m-jL-1
        inner_idx = np.arange(jl_i, m_i - jl_i)
        inner_price = price_i[inner_idx]
        new_price = np.empty(len(inner_price) + 2, dtype=np.float64)
        new_price[0] = p0
        new_price[1:-1] = inner_price
        new_price[-1] = p1

        # Ref: realized_multivariate_kernel.m:182
        inner_time = time_i[inner_idx]
        new_time = np.empty(len(inner_time) + 2, dtype=np.float64)
        new_time[0] = t0
        new_time[1:-1] = inner_time
        new_time[-1] = t1

        jittered_prices.append(new_price)
        jittered_times.append(new_time)

    # ==================================================================
    # Step 3: Put into refresh time (after jittering)
    # Ref: realized_multivariate_kernel.m:187
    # ==================================================================
    if num_prices >= 2:
        rt_prices, rt_times, _actual_times = realized_refresh_time(
            jittered_prices, jittered_times, time_type,
        )
    else:
        # Single asset: no synchronization needed
        rt_prices = jittered_prices[0].reshape(-1, 1)
        rt_times = jittered_times[0]

    # Ref: realized_multivariate_kernel.m:189-191 — subsample
    if sampling_interval != 1:
        rt_prices = rt_prices[::sampling_interval]

    # ==================================================================
    # Step 4: Compute the weights
    # Ref: realized_multivariate_kernel.m:193
    # ==================================================================
    weights: np.ndarray = realized_kernel_weights(options)

    # ==================================================================
    # Step 5: Compute the kernel
    # Ref: realized_multivariate_kernel.m:195-196
    # ==================================================================
    # Ref: realized_multivariate_kernel.m:195 — diff(log(rtPrice))
    rt_returns: np.ndarray = np.diff(np.log(rt_prices), axis=0)

    # Ref: realized_multivariate_kernel.m:196
    rmk: np.ndarray = _realized_multivariate_kernel_core(rt_returns, weights)

    # ==================================================================
    # Diagnostics
    # Ref: realized_multivariate_kernel.m:198-203
    # ==================================================================
    diagnostics: dict = {
        'kernel': options['kernel'],
        'bandwidth': options['bandwidth'],
        'jitterLags': jitter_lags,
        'weights': weights,
        'rtPrice': rt_prices,
        'rtTime': rt_times,
    }

    return rmk, diagnostics
