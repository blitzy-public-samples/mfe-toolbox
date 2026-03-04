"""
Pre-averaged realized variance estimator (Christensen-Oomen-Podolski).

Implements the pre-averaged realized variance estimator with the
Hautsch-Podolski noise variance correction.  The pre-averaging kernel is
the Bartlett (triangular) kernel ``g(x) = min(x, 1-x)`` and the window
size is ``K = ceil(theta * sqrt(m))`` where *m* is the number of log
returns and *theta* is a tuning parameter from *options*.

Migrated from: ``realized/realized_preaveraged_variance.m``
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 2/27/2014

References
----------
.. [1] Christensen, K., Oomen, R. C. A. and Podolskij, M. (2014).
   "Fact or friction: Jumps at ultra high frequency." *Journal of
   Financial Economics*, 114(3), 576-599.
.. [2] Hautsch, N. and Podolskij, M. (2013). "Pre-averaging based
   estimation of quadratic variation in the presence of noise and
   jumps: theory, implementation, and empirical evidence."
   *Journal of Business and Economic Statistics*, 31(2), 165-183.

See Also
--------
realized_options : Default option factory for realized estimators.
realized_kernel : Realized kernel quadratic variation estimator.
realized_noise_estimate : Microstructure noise variance estimation.
realized_variance : Standard realized variance estimator.
realized_quantile_variance : Quantile-based realized variance.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_price_filter import realized_price_filter


def realized_preaveraged_variance(
    price: np.ndarray,
    time: np.ndarray | None = None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval: int | float | np.ndarray = 1,
    subsamples: int = 1,
    options: dict | None = None,
) -> tuple[float, float, dict]:
    """Compute the pre-averaged realized variance estimator.

    Estimates the integrated variance from high-frequency price data using
    pre-averaging of log returns with a triangular kernel
    ``g(x) = min(x, 1-x)``.  The estimator applies a bias correction
    based on estimated microstructure noise variance (Hautsch-Podolski
    method).

    Parameters
    ----------
    price : array_like
        1-D array of *m* high-frequency prices.  Must contain at least 2
        observations.
    time : array_like or None, optional
        1-D array of *m* times corresponding to ``price``.  When ``None``
        (default), a default grid spanning 09:30-16:00 in seconds past
        midnight is generated and ``time_type`` is set to ``'seconds'``,
        ``sampling_type`` to ``'BusinessTime'`` and ``sampling_interval``
        to ``1``.
        Ref: realized_preaveraged_variance.m:58-62
    time_type : str, optional
        Time format descriptor (case-insensitive):

        * ``'wall'``    — 24-hour clock HHMMSS (e.g. 101543)
        * ``'seconds'`` — seconds past midnight
        * ``'unit'``    — unit-normalized [0, 1] interval (default)
    sampling_type : str, optional
        Sampling strategy for price filtering (case-insensitive):

        * ``'CalendarTime'``     — calendar-time sampling
        * ``'CalendarUniform'``  — uniformly spaced in calendar time
        * ``'BusinessTime'``     — tick-time sampling
        * ``'BusinessUniform'``  — uniformly spaced in tick time
        * ``'Fixed'``            — user-specified sample times

        Default is ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Sampling interval whose semantics depend on *sampling_type*.
        Default is ``1``.
    subsamples : int, optional
        Number of subsampled grids to average.  When ``1`` (default), the
        standard (non-subsampled) estimator matching the original MATLAB
        implementation is used.  When greater than ``1``, *subsamples*
        shifted grids of pre-averaged returns are computed and averaged
        to reduce finite-sample variance.
    options : dict or None, optional
        Pre-averaging option dictionary.  If ``None`` (default),
        ``realized_options('preaveraging')`` is called to obtain defaults
        (``theta=1``, etc.).  Required key: ``'theta'`` — the tuning
        parameter that determines the window size ``K``.
        Ref: realized_preaveraged_variance.m:63,65

    Returns
    -------
    pav : float
        Raw pre-averaged realized variance estimate **before** noise bias
        correction: ``const1 * const2 * sum(Y_i^2)`` where ``Y_i`` are the
        pre-averaged returns.
    pav_debiased : float
        Bias-corrected pre-averaged realized variance estimate (this matches
        the single return value ``rpav`` of the original MATLAB function):
        ``pav - psi1K / (theta^2 * psi2K) * omega^2``.
    diagnostics : dict
        Dictionary of intermediate quantities useful for diagnostics:

        * ``'theta'``                — theta parameter used
        * ``'K'``                    — pre-averaging window size
        * ``'psi_1'``               — psi_1 kernel constant
        * ``'psi_2'``               — psi_2 kernel constant
        * ``'omega'``               — noise standard-deviation proxy used
        * ``'bias'``                — total noise bias subtracted
        * ``'noise_variance'``      — Bandi-Russell noise variance
        * ``'noise_estimate_oomen'``— Oomen AC(1) noise estimate
        * ``'n_returns'``           — number of log returns (m)
        * ``'n_preaveraged'``       — number of pre-averaged returns
        * ``'const1'``              — finite-sample scaling constant
        * ``'const2'``              — kernel normalisation constant
        * ``'subsamples'``          — number of subsamples used

    Raises
    ------
    ValueError
        If *price* has fewer than 2 observations, if *price* and *time*
        have different lengths, if *time* is not sorted in non-decreasing
        order, or if any parameter takes an invalid value.

    Warnings
    --------
    UserWarning
        Emitted when the Oomen noise estimate is negative and the function
        falls back to the Bandi-Russell estimate, or when there are
        insufficient returns for the chosen pre-averaging window size.

    Notes
    -----
    The core algorithm follows Christensen, Oomen and Podolskij (2014)
    with the noise estimator of Hautsch and Podolskij (2013):

    1. Filter raw prices at the desired sampling frequency.
    2. Compute log returns ``r_i = log(P_{i+1}) - log(P_i)``.
    3. Pre-average returns with the triangular kernel:
       ``Y_i = sum_{j=1}^{K-1} g(j/K) * r_{i+j-1}`` for
       ``i = 0, ..., m-K+1``   (0-based indexing).
    4. Compute kernel constants ``psi_1`` and ``psi_2``.
    5. Estimate noise variance ``omega^2`` via
       :func:`realized_noise_estimate`.
    6. Compute the bias-corrected estimator:
       ``PAV = const1 * const2 * sum(Y_i^2) - psi_1/(theta^2 * psi_2) * omega^2``

    MATLAB uses 1-based indexing throughout; all array indices in this
    module use 0-based Python indexing with equivalent ranges.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> prices = 100.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(500)))
    >>> pav, pav_db, diag = realized_preaveraged_variance(prices)
    """
    # ==================================================================
    # 1. Input coercion and validation
    # Ref: realized_preaveraged_variance.m:56-70
    # ==================================================================

    # --- Price ---
    price = np.asarray(price, dtype=np.float64).ravel()
    m_price = price.shape[0]
    if m_price < 2:
        raise ValueError(
            'PRICE must be a vector with at least 2 observations.'
        )

    # --- Time defaults ---
    # Ref: realized_preaveraged_variance.m:57-62
    # When only price is provided, MATLAB defaults to a seconds-past-midnight
    # grid spanning NYSE 09:30-16:00 with BusinessTime tick sampling.
    if time is None:
        time = np.linspace(9.5 * 3600.0, 16.0 * 3600.0, m_price)
        time_type = 'seconds'
        sampling_type = 'BusinessTime'
        sampling_interval = 1

    # Ref: realized_preaveraged_variance.m:77 — time = double(time)
    time = np.asarray(time, dtype=np.float64).ravel()

    # --- Validate time ---
    if time.shape[0] != m_price:
        raise ValueError(
            'PRICE and TIME must have the same number of elements.'
        )
    if m_price > 1 and np.any(np.diff(time) < 0.0):
        raise ValueError('TIME must be sorted and non-decreasing.')

    # --- Validate time_type ---
    time_type_lower = time_type.lower()
    if time_type_lower not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds', or 'unit'."
        )

    # --- Validate sampling_type ---
    _valid_sampling = {
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    }
    sampling_type_lower = sampling_type.lower()
    if sampling_type_lower not in _valid_sampling:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform', or 'Fixed'."
        )

    # --- Validate subsamples ---
    subsamples = int(subsamples)
    if subsamples < 1:
        raise ValueError('SUBSAMPLES must be a positive integer.')

    # --- Default options ---
    # Ref: realized_preaveraged_variance.m:63,65
    if options is None:
        options = realized_options('preaveraging')

    # --- Normalise time to unit interval for internal bookkeeping ---
    # realized_convert2unit handles wall / seconds → unit conversion and
    # rescales the sampling interval accordingly.  The converted values
    # are used for any internal time-based arithmetic; downstream helpers
    # (realized_price_filter, realized_noise_estimate) receive the
    # *original* time and time_type so their own validation is unaffected.
    # Ref: realized_convert2unit.m — called implicitly in MATLAB pipeline.
    if time_type_lower in ('wall', 'seconds'):
        _unit_time, _t0, _t1, _conv_si = realized_convert2unit(
            time, time_type_lower, sampling_type, sampling_interval,
        )
    else:
        # time_type == 'unit' — already in [0, 1]
        _unit_time = time

    # ==================================================================
    # 2. Filter prices
    # Ref: realized_preaveraged_variance.m:80
    # ==================================================================
    filter_result = realized_price_filter(
        price, time, time_type_lower,
        sampling_type, sampling_interval,
    )
    # realized_price_filter returns (filtered_price, ...) as a tuple
    if isinstance(filter_result, tuple):
        filtered_price = np.asarray(
            filter_result[0], dtype=np.float64,
        ).ravel()
    else:
        filtered_price = np.asarray(
            filter_result, dtype=np.float64,
        ).ravel()

    # ==================================================================
    # 3. Compute log returns
    # Ref: realized_preaveraged_variance.m:81
    # ==================================================================
    returns = np.diff(np.log(filtered_price))
    m = returns.shape[0]  # number of returns

    if m < 1:
        warnings.warn(
            'Filtered price series has fewer than 2 observations; '
            'cannot compute returns.',
            stacklevel=2,
        )
        return (
            np.nan,
            np.nan,
            _build_diagnostics(
                options.get('theta', 1.0), 0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, m, 0, 0.0, 0.0, subsamples,
            ),
        )

    # ==================================================================
    # 4. Theta and window size K
    # Ref: realized_preaveraged_variance.m:83-85
    # ==================================================================
    theta = float(options.get('theta', 1.0))
    K = int(np.ceil(theta * np.sqrt(float(m))))
    # Ref: K must be at least 2 for a non-degenerate kernel window
    if K < 2:
        K = 2

    # Number of pre-averaged returns
    # Ref: realized_preaveraged_variance.m:89 — nan(m-K+2,1)
    n_preav = m - K + 2
    if n_preav < 1:
        # Not enough returns for the chosen window size
        warnings.warn(
            f'Not enough returns (m={m}) for pre-averaging window K={K}. '
            'Returning NaN.',
            stacklevel=2,
        )
        return (
            np.nan,
            np.nan,
            _build_diagnostics(
                theta, K, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, m, 0, 0.0, 0.0, subsamples,
            ),
        )

    # ==================================================================
    # 5. Kernel weights  g(x) = min(x, 1-x)
    # Ref: realized_preaveraged_variance.m:86-87
    # w = g((1:(K-1))/K)  — length K-1
    # ==================================================================
    # Ref: realized_preaveraged_variance.m:86 — g = @(x) min(x,1-x)
    j_w = np.arange(1, K, dtype=np.float64) / float(K)
    w = np.minimum(j_w, 1.0 - j_w)

    # ==================================================================
    # 6. Psi constants
    # Ref: realized_preaveraged_variance.m:95-96
    # ==================================================================
    # psi_1 = K * sum((g((1:K)/K) - g((0:(K-1))/K)).^2)
    j1 = np.arange(1, K + 1, dtype=np.float64) / float(K)
    j0 = np.arange(0, K, dtype=np.float64) / float(K)
    g1 = np.minimum(j1, 1.0 - j1)
    g0 = np.minimum(j0, 1.0 - j0)
    psi_1 = float(K) * float(np.sum((g1 - g0) ** 2))

    # psi_2 = (1/K) * sum(g((1:(K-1))/K).^2)
    psi_2 = (1.0 / float(K)) * float(np.sum(w ** 2))

    # Guard against psi_2 == 0 (should not happen for K >= 2, but be safe)
    if psi_2 <= 0.0:
        warnings.warn(
            'psi_2 is zero or negative; cannot normalise pre-averaged '
            'variance.  Returning NaN.',
            stacklevel=2,
        )
        return (
            np.nan,
            np.nan,
            _build_diagnostics(
                theta, K, psi_1, psi_2,
                0.0, 0.0, 0.0, 0.0, m, n_preav, 0.0, 0.0, subsamples,
            ),
        )

    # ==================================================================
    # 7. Noise estimation
    # Ref: realized_preaveraged_variance.m:98-103
    # Uses *original* (unfiltered) price and time.
    # ==================================================================
    noise_variance, _debiased_nv, _iq_est, noise_estimate_oomen = (
        realized_noise_estimate(price, time, time_type_lower, options)
    )

    # Ref: realized_preaveraged_variance.m:100-103
    omega = noise_estimate_oomen
    if omega < 0.0:
        # Oomen estimate is negative — fall back to Bandi-Russell
        # Ref: realized_preaveraged_variance.m:101-103
        omega = noise_variance
        warnings.warn(
            'Oomen noise estimate is negative; falling back to '
            'Bandi-Russell noise variance estimate.',
            stacklevel=2,
        )

    # ==================================================================
    # 8. Compute pre-averaged variance
    # ==================================================================
    if subsamples == 1:
        # ----------------------------------------------------------
        # Standard (non-subsampled) estimator — exact MATLAB match
        # Ref: realized_preaveraged_variance.m:89-92, 105-109
        # ----------------------------------------------------------
        pav_raw, pav_debiased = _pav_core(
            returns, w, K, m, n_preav, theta, psi_1, psi_2, omega,
        )
    else:
        # ----------------------------------------------------------
        # Subsampled pre-averaged variance
        # Compute S estimates by shifting the starting offset of the
        # pre-averaging window within the return vector and averaging.
        # When the shift is larger than the available return window,
        # it is skipped.
        # ----------------------------------------------------------
        pav_accum = 0.0
        pav_db_accum = 0.0
        count = 0
        for s in range(subsamples):
            # Use a shifted slice of returns starting at index s
            shifted_returns = returns[s:]
            m_s = shifted_returns.shape[0]
            K_s = int(np.ceil(theta * np.sqrt(float(m_s))))
            if K_s < 2:
                K_s = 2
            n_s = m_s - K_s + 2
            if n_s < 1 or m_s < 1:
                continue  # skip if insufficient data

            # Recompute weights and psi for this shifted grid length
            j_s = np.arange(1, K_s, dtype=np.float64) / float(K_s)
            w_s = np.minimum(j_s, 1.0 - j_s)
            j1_s = np.arange(1, K_s + 1, dtype=np.float64) / float(K_s)
            j0_s = np.arange(0, K_s, dtype=np.float64) / float(K_s)
            g1_s = np.minimum(j1_s, 1.0 - j1_s)
            g0_s = np.minimum(j0_s, 1.0 - j0_s)
            psi_1_s = float(K_s) * float(np.sum((g1_s - g0_s) ** 2))
            psi_2_s = (1.0 / float(K_s)) * float(np.sum(w_s ** 2))
            if psi_2_s <= 0.0:
                continue

            p_raw, p_db = _pav_core(
                shifted_returns, w_s, K_s, m_s, n_s,
                theta, psi_1_s, psi_2_s, omega,
            )
            if not (np.isnan(p_raw) or np.isnan(p_db)):
                pav_accum += p_raw
                pav_db_accum += p_db
                count += 1

        if count > 0:
            pav_raw = pav_accum / count
            pav_debiased = pav_db_accum / count
        else:
            pav_raw = np.nan
            pav_debiased = np.nan

    # ==================================================================
    # 9. Build diagnostics dictionary
    # ==================================================================
    diagnostics = _build_diagnostics(
        theta, K, psi_1, psi_2,
        omega,
        psi_1 / (theta ** 2 * psi_2) * omega ** 2,  # bias
        noise_variance, noise_estimate_oomen,
        m, n_preav,
        m / (m - K + 2),   # const1
        1.0 / (K * psi_2),  # const2
        subsamples,
    )

    return float(pav_raw), float(pav_debiased), diagnostics


# ======================================================================
# Private helpers
# ======================================================================


def _pav_core(
    returns: np.ndarray,
    w: np.ndarray,
    K: int,
    m: int,
    n_preav: int,
    theta: float,
    psi_1: float,
    psi_2: float,
    omega: float,
) -> tuple[float, float]:
    """Core pre-averaged variance computation (single grid).

    Parameters
    ----------
    returns : np.ndarray
        1-D array of *m* log returns.
    w : np.ndarray
        Kernel weight vector of length ``K-1``.
    K : int
        Pre-averaging window size.
    m : int
        Number of returns.
    n_preav : int
        Number of pre-averaged returns (``m - K + 2``).
    theta : float
        Theta tuning parameter.
    psi_1 : float
        First kernel constant.
    psi_2 : float
        Second kernel constant.
    omega : float
        Noise standard-deviation proxy (omega ≥ 0).

    Returns
    -------
    pav_raw : float
        Raw pre-averaged variance (before bias correction).
    pav_debiased : float
        Bias-corrected pre-averaged variance.

    Notes
    -----
    Ref: realized_preaveraged_variance.m:89-92, 105-109
    """
    # ------------------------------------------------------------------
    # Pre-averaged returns
    # MATLAB: for i=1:m-K+2,  preav_returns(i) = w*returns(i:i+K-2);  end
    # Python 0-based: for i in range(n_preav):
    #     preav_returns[i] = w @ returns[i : i + K - 1]
    # Ref: realized_preaveraged_variance.m:89-92
    # ------------------------------------------------------------------
    preav_returns = np.zeros(n_preav, dtype=np.float64)
    w_len = w.shape[0]  # K - 1
    for i in range(n_preav):
        # Ref: realized_preaveraged_variance.m:91 — w*returns(i:i+K-2)
        # Python slice [i : i + K - 1] yields K-1 elements matching w_len
        preav_returns[i] = np.dot(w, returns[i: i + w_len])

    # ------------------------------------------------------------------
    # Scaling constants
    # Ref: realized_preaveraged_variance.m:105-106
    # ------------------------------------------------------------------
    const1 = float(m) / float(n_preav)  # m / (m - K + 2)
    const2 = 1.0 / (float(K) * psi_2)

    # ------------------------------------------------------------------
    # Bias
    # Ref: realized_preaveraged_variance.m:107
    # bias = psi_1 / (theta^2 * psi_2) * omega^2
    # ------------------------------------------------------------------
    bias = (psi_1 / (theta ** 2 * psi_2)) * omega ** 2

    # ------------------------------------------------------------------
    # Raw and debiased pre-averaged variance
    # Ref: realized_preaveraged_variance.m:109
    # rpav = const1 * const2 * sum(preav_returns.^2) - bias
    # ------------------------------------------------------------------
    sum_sq = float(np.sum(preav_returns ** 2))
    pav_raw = const1 * const2 * sum_sq
    pav_debiased = pav_raw - bias

    return pav_raw, pav_debiased


def _build_diagnostics(
    theta: float,
    K: int,
    psi_1: float,
    psi_2: float,
    omega: float,
    bias: float,
    noise_variance: float,
    noise_estimate_oomen: float,
    n_returns: int,
    n_preaveraged: int,
    const1: float,
    const2: float,
    subsamples: int,
) -> dict:
    """Construct the diagnostics dictionary returned to the caller.

    All arguments are scalar values describing the estimation state.
    """
    return {
        'theta': theta,
        'K': K,
        'psi_1': psi_1,
        'psi_2': psi_2,
        'omega': omega,
        'bias': bias,
        'noise_variance': noise_variance,
        'noise_estimate_oomen': noise_estimate_oomen,
        'n_returns': n_returns,
        'n_preaveraged': n_preaveraged,
        'const1': const1,
        'const2': const2,
        'subsamples': subsamples,
    }
