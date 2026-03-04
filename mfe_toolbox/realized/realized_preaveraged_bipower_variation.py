"""
Pre-averaged bipower variation estimator for realized volatility.

Implements the pre-averaged bipower variation (PABPV) estimator following
Christensen, Oomen and Podolskij (2014), with the noise variance estimator
from Hautsch and Podolskij (2013).  The estimator applies a triangular
preaveraging kernel g(x)=min(x,1-x) to high-frequency log-returns, then
computes bipower variation from absolute products of lead-lag pre-averaged
returns separated by a gap of K-1 to ensure near-independence.

A debiased variant subtracts the estimated microstructure noise contribution
using the Hautsch-Podolskij formula.

Migrated from: realized/realized_preaveraged_bipower_variation.m
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 2/27/2014

See Also
--------
realized_preaveraged_variance : Pre-averaged realized variance estimator.
realized_options : Default options for realized estimators.
realized_kernel : Realized kernel quadratic variation estimator.
realized_noise_estimate : Microstructure noise estimation.
realized_variance : Standard realized variance estimator.
realized_range : Range-based realized volatility estimator.
realized_quantile_variance : Quantile realized variance estimator.

References
----------
.. [1] Christensen, K., Oomen, R.C.A. and Podolskij, M. (2014).
   "Fact or friction: Jumps at ultra high frequency."  *Journal of
   Financial Economics*, 114(3), 576-599.
.. [2] Hautsch, N. and Podolskij, M. (2013).  "Preaveraging-based
   estimation of quadratic variation in the presence of noise and
   jumps: Theory, implementation, and empirical evidence."
   *Journal of Business and Economic Statistics*, 31(2), 165-183.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_price_filter import realized_price_filter


def realized_preaveraged_bipower_variation(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    subsamples: int = 1,
    options: dict | None = None,
) -> tuple[float, float, dict]:
    """Estimate integrated variance using pre-averaged bipower variation.

    Computes the pre-averaged bipower variation (PABPV) estimator which
    is robust to both microstructure noise and jumps.  The estimator applies
    a triangular preaveraging kernel to high-frequency returns, then
    computes bipower variation from absolute products of lead-lag
    pre-averaged returns.

    Parameters
    ----------
    price : array_like
        m-element vector of high-frequency prices.
    time : array_like or None, optional
        m-element vector of times corresponding to ``price``.  If ``None``,
        default NYSE trading hours (9:30 AM – 4:00 PM) are assumed with
        times in seconds past midnight, using BusinessTime sampling with
        interval 1 (all ticks).
    time_type : str, optional
        Time format descriptor.  One of:

        * ``'wall'``    — 24-hour clock HHMMSS (e.g. 101543)
        * ``'seconds'`` — seconds past midnight
        * ``'unit'``    — unit-normalised [0, 1]

        Default is ``'unit'``.
    sampling_type : str, optional
        Sampling scheme for price filtering.  One of:

        * ``'CalendarTime'``    — fixed calendar-time interval
        * ``'CalendarUniform'`` — uniform calendar-time spacing
        * ``'BusinessTime'``    — fixed tick interval
        * ``'BusinessUniform'`` — uniform tick spacing
        * ``'Fixed'``           — specific time points

        Default is ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Sampling parameter whose meaning depends on ``sampling_type``.
        Default is ``1``.
    subsamples : int, optional
        Number of subsamples (currently only ``1`` is supported).
        Default is ``1``.
    options : dict or None, optional
        Preaveraging options dictionary from
        ``realized_options('preaveraging')``.  Required key:

        * ``'theta'`` : float — controls window size K = ceil(theta * sqrt(m)).

        If ``None``, default preaveraging options are used.

    Returns
    -------
    pabpv : float
        Raw pre-averaged bipower variation estimate (before noise
        correction).
    pabpv_debiased : float
        Debiased pre-averaged bipower variation estimate with
        Hautsch-Podolskij noise correction subtracted.
    diagnostics : dict
        Dictionary of diagnostic information:

        * ``'K'`` : int — preaveraging window size
        * ``'theta'`` : float — theta parameter controlling K
        * ``'psi_1'`` : float — first preaveraging constant
        * ``'psi_2'`` : float — second preaveraging constant
        * ``'mu_1'`` : float — sqrt(2/pi) constant
        * ``'omega'`` : float — noise variance estimate used
        * ``'noise_variance'`` : float — Bandi-Russell noise variance
        * ``'noise_estimate_oomen'`` : float — Oomen AC(1) noise estimate
        * ``'bias'`` : float — noise bias term subtracted
        * ``'n_returns'`` : int — number of log returns
        * ``'n_preav_returns'`` : int — number of pre-averaged returns
        * ``'n_bpv_terms'`` : int — number of lead-lag product terms
        * ``'const1'`` : float — finite-sample correction factor
        * ``'const2'`` : float — scaling constant

    Raises
    ------
    ValueError
        If ``price`` has fewer than 2 elements.
        If ``time`` length does not match ``price``.

    Notes
    -----
    The pre-averaged bipower variation uses a triangular kernel
    ``g(x) = min(x, 1-x)`` with window size ``K = ceil(theta * sqrt(m))``
    where ``m`` is the number of log returns after price filtering.

    The BPV is computed from absolute products of lead-lag pre-averaged
    returns separated by a gap of K-1, which ensures near-independence:

    .. math::

        PABPV = \\frac{m}{m - 2K + 2} \\cdot
                \\frac{1}{K \\, \\psi_{2K} \\, \\mu_1^2}
                \\sum_{i=1}^{m-2K+3} |\\bar{Y}_i| \\, |\\bar{Y}_{i+K-1}|

    where ``mu_1 = sqrt(2 / pi)`` is the first absolute moment of the
    standard normal, and the noise bias correction is:

    .. math::

        \\text{bias} = \\frac{\\psi_{1K}}{\\theta^2 \\, \\psi_{2K}} \\,
                       \\hat{\\omega}^2

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> prices = np.cumsum(rng.standard_normal(1000)) + 100
    >>> pabpv, pabpv_d, diag = realized_preaveraged_bipower_variation(prices)
    """
    # ================================================================
    # Input Validation and Default Handling
    # Ref: realized_preaveraged_bipower_variation.m:57-71
    # ================================================================

    # Coerce price to float64 1-D array
    price = np.asarray(price, dtype=np.float64).ravel()
    m_price = len(price)

    if m_price < 2:
        raise ValueError("PRICE must contain at least 2 elements.")

    # ------------------------------------------------------------------
    # Handle default time generation when time is None
    # Ref: realized_preaveraged_bipower_variation.m:59-64
    # MATLAB 1-arg default: linspace(9.5*3600, 16*3600, m)', 'seconds',
    #   'businesstime', 1
    # ------------------------------------------------------------------
    if time is None:
        # Ref: realized_preaveraged_bipower_variation.m:60 — NYSE 9:30–16:00
        time = np.linspace(9.5 * 3600.0, 16.0 * 3600.0, m_price)
        time_type = 'seconds'
        sampling_type = 'BusinessTime'
        sampling_interval = 1

    # Coerce time to float64 1-D array
    # Ref: realized_preaveraged_bipower_variation.m:78 — time = double(time)
    time = np.asarray(time, dtype=np.float64).ravel()

    if len(time) != m_price:
        raise ValueError(
            "TIME must be a 1-D array with the same length as PRICE."
        )

    # ------------------------------------------------------------------
    # Default options
    # Ref: realized_preaveraged_bipower_variation.m:64-67
    # ------------------------------------------------------------------
    if options is None:
        options = realized_options('preaveraging')

    # ------------------------------------------------------------------
    # Warn about unsupported subsampling
    # ------------------------------------------------------------------
    if subsamples is not None and int(subsamples) > 1:
        warnings.warn(
            "Subsampling (subsamples > 1) is not implemented for the "
            "pre-averaged bipower variation estimator. Using subsamples=1.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Validate time format via realized_convert2unit
    # This ensures time and sampling_interval are consistent and valid
    # before downstream processing.
    # Ref: realized_convert2unit — normalises time inputs
    # ------------------------------------------------------------------
    realized_convert2unit(time, time_type, sampling_type, sampling_interval)

    # ================================================================
    # Step 1: Filter prices at the specified sampling frequency
    # Ref: realized_preaveraged_bipower_variation.m:81
    # ================================================================
    filter_result = realized_price_filter(
        price, time, time_type, sampling_type, sampling_interval
    )
    # realized_price_filter returns (filtered_price, filtered_time, actual_time)
    if isinstance(filter_result, tuple):
        filtered_price = np.asarray(filter_result[0], dtype=np.float64).ravel()
    else:
        filtered_price = np.asarray(filter_result, dtype=np.float64).ravel()

    # ================================================================
    # Step 2: Compute log returns from filtered prices
    # Ref: realized_preaveraged_bipower_variation.m:82
    # ================================================================
    returns = np.diff(np.log(filtered_price))
    m = len(returns)

    if m < 2:
        # Degenerate case — insufficient data after filtering
        warnings.warn(
            "Insufficient returns after price filtering for pre-averaged "
            "bipower variation; returning NaN.",
            stacklevel=2,
        )
        return np.nan, np.nan, {
            'K': 0,
            'theta': float(options.get('theta', 1)),
            'psi_1': np.nan,
            'psi_2': np.nan,
            'mu_1': float(np.sqrt(2.0 / np.pi)),
            'omega': np.nan,
            'noise_variance': np.nan,
            'noise_estimate_oomen': np.nan,
            'bias': np.nan,
            'n_returns': m,
            'n_preav_returns': 0,
            'n_bpv_terms': 0,
            'const1': np.nan,
            'const2': np.nan,
        }

    # ================================================================
    # Step 3: Compute preaveraging window and kernel weights
    # Ref: realized_preaveraged_bipower_variation.m:84-88
    # ================================================================
    theta = float(options.get('theta', 1))

    # Ref: realized_preaveraged_bipower_variation.m:86 — K = ceil(theta*sqrt(m))
    K = int(np.ceil(theta * np.sqrt(m)))

    # Ensure K >= 2 for meaningful preaveraging
    # Ref: g(x) = min(x, 1-x) requires at least 1 weight → K-1 >= 1
    if K < 2:
        K = 2
        warnings.warn(
            "Preaveraging window K was less than 2; increased to K=2 for "
            "minimum viable computation.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Check that we have enough returns for the lead-lag BPV structure.
    # The finite-sample correction const1 = m / (m - 2K + 2) requires
    # m - 2K + 2 > 0, i.e. m >= 2K - 1.
    # The number of lead-lag terms is m - 2K + 3, requiring m >= 2K - 2
    # for at least 1 term.  We enforce the stricter constraint m >= 2K - 1.
    # ------------------------------------------------------------------
    if m - 2 * K + 2 <= 0:
        warnings.warn(
            f"Insufficient returns (m={m}) for preaveraging window K={K}. "
            f"Need m >= {2 * K - 1} for finite-sample correction. "
            f"Returning NaN.",
            stacklevel=2,
        )
        return np.nan, np.nan, {
            'K': K,
            'theta': theta,
            'psi_1': np.nan,
            'psi_2': np.nan,
            'mu_1': float(np.sqrt(2.0 / np.pi)),
            'omega': np.nan,
            'noise_variance': np.nan,
            'noise_estimate_oomen': np.nan,
            'bias': np.nan,
            'n_returns': m,
            'n_preav_returns': max(0, m - K + 2),
            'n_bpv_terms': 0,
            'const1': np.nan,
            'const2': np.nan,
        }

    # ------------------------------------------------------------------
    # Triangular kernel: g(x) = min(x, 1-x)
    # Ref: realized_preaveraged_bipower_variation.m:87 — g = @(x) min(x,1-x)
    # Weights for j = 1, 2, ..., K-1
    # Ref: realized_preaveraged_bipower_variation.m:88 — w = g((1:(K-1))/K)
    # ------------------------------------------------------------------
    j_arr = np.arange(1, K, dtype=np.float64)  # [1, 2, ..., K-1]
    w = np.minimum(j_arr / K, 1.0 - j_arr / K)

    # ================================================================
    # Step 4: Compute pre-averaged returns via convolution
    # Ref: realized_preaveraged_bipower_variation.m:90-93
    #
    # MATLAB loop:
    #   for i = 1:m-K+2
    #       preav_returns(i) = w * returns(i:i+K-2);
    #   end
    #
    # This is a sliding inner product:
    #   Y_i = sum_{j=0}^{K-2} w[j] * returns[i+j]
    #
    # Equivalent to np.convolve(returns, w[::-1], 'valid') because the
    # convolution with reversed kernel produces the sliding dot product.
    # Note: g(x) = min(x, 1-x) is symmetric so w[::-1] == w, but we
    # use w[::-1] for mathematical correctness.
    # ================================================================
    preav_returns = np.convolve(returns, w[::-1], mode='valid')
    n_preav = len(preav_returns)
    # Verify: n_preav should equal m - K + 2
    # Ref: realized_preaveraged_bipower_variation.m:90 — nan(m-K+2, 1)

    # ================================================================
    # Step 5: Compute psi_1 and psi_2 constants
    # Ref: realized_preaveraged_bipower_variation.m:96-97
    #
    # psi_1 = K * sum((g(j/K) - g((j-1)/K))^2)  for j = 1..K
    # psi_2 = (1/K) * sum(g(j/K)^2)              for j = 1..K-1
    # ================================================================

    # psi_1 computation
    # Ref: realized_preaveraged_bipower_variation.m:96
    # MATLAB: psi_1 = K*sum((g((1:K)/K) - g((0:(K-1))/K)).^2)
    j_full = np.arange(1, K + 1, dtype=np.float64)   # [1, ..., K]
    j_prev = np.arange(0, K, dtype=np.float64)        # [0, ..., K-1]
    g_full = np.minimum(j_full / K, 1.0 - j_full / K)
    g_prev = np.minimum(j_prev / K, 1.0 - j_prev / K)
    psi_1 = float(K * np.sum((g_full - g_prev) ** 2))

    # psi_2 computation
    # Ref: realized_preaveraged_bipower_variation.m:97
    # MATLAB: psi_2 = 1/K*sum((g((1:(K-1))/K)).^2)
    g_inner = np.minimum(j_arr / K, 1.0 - j_arr / K)  # reuse j_arr from Step 3
    psi_2 = float((1.0 / K) * np.sum(g_inner ** 2))

    # Guard against degenerate psi_2 (should not happen for K >= 2)
    if psi_2 <= 0.0:
        warnings.warn(
            "psi_2 is non-positive; degenerate kernel. Returning NaN.",
            stacklevel=2,
        )
        return np.nan, np.nan, {
            'K': K, 'theta': theta, 'psi_1': psi_1, 'psi_2': psi_2,
            'mu_1': float(np.sqrt(2.0 / np.pi)),
            'omega': np.nan, 'noise_variance': np.nan,
            'noise_estimate_oomen': np.nan, 'bias': np.nan,
            'n_returns': m, 'n_preav_returns': n_preav, 'n_bpv_terms': 0,
            'const1': np.nan, 'const2': np.nan,
        }

    # ================================================================
    # Step 6: Estimate microstructure noise variance
    # Ref: realized_preaveraged_bipower_variation.m:99-104
    # Uses original (unfiltered) price and time arrays.
    # ================================================================
    noise_variance, _, _, noise_estimate_oomen = realized_noise_estimate(
        price, time, time_type, options
    )

    # Use Oomen AC(1) estimator if non-negative; else fall back to
    # Bandi-Russell estimator.
    # Ref: realized_preaveraged_bipower_variation.m:101-103
    omega = float(noise_estimate_oomen)
    if omega < 0:
        # Ref: realized_preaveraged_bipower_variation.m:103
        omega = float(noise_variance)

    # ================================================================
    # Step 7: Compute bipower variation
    # Ref: realized_preaveraged_bipower_variation.m:107-116
    # ================================================================

    # mu_1 = sqrt(2/pi) — first absolute moment of the standard normal
    # E[|Z|] for Z ~ N(0, 1)
    # Ref: realized_preaveraged_bipower_variation.m:107
    mu_1 = np.sqrt(2.0 / np.pi)

    # Finite-sample correction factor
    # Ref: realized_preaveraged_bipower_variation.m:108
    const1 = float(m) / float(m - 2 * K + 2)

    # Scaling constant incorporating psi_2 and mu_1
    # Ref: realized_preaveraged_bipower_variation.m:109
    const2 = 1.0 / (K * psi_2 * mu_1 ** 2)

    # Noise bias term
    # Ref: realized_preaveraged_bipower_variation.m:111
    bias = psi_1 / (theta ** 2 * psi_2) * omega ** 2

    # ------------------------------------------------------------------
    # Lead-lag products of pre-averaged returns with gap K-1
    # Ref: realized_preaveraged_bipower_variation.m:113-114
    #
    # MATLAB (1-based indexing):
    #   lead = preav_returns(K:end)            → elements K to m-K+2
    #   lag  = preav_returns(1:end-K+1)        → elements 1 to m-2K+3
    #
    # Python (0-based indexing):
    #   lead = preav_returns[K-1:]             → indices K-1 to n_preav-1
    #   lag  = preav_returns[:n_preav - K + 1] → indices 0 to n_preav-K
    #
    # Both have length m - 2K + 3.
    # ------------------------------------------------------------------
    lead = preav_returns[K - 1:]
    lag = preav_returns[:n_preav - K + 1]
    n_bpv_terms = len(lead)

    # Sum of absolute products of lead-lag pre-averaged returns
    # Ref: realized_preaveraged_bipower_variation.m:116
    # MATLAB: sum(abs(lead.*lag))
    bpv_sum = float(np.sum(np.abs(lead * lag)))

    # Raw PABPV (without noise correction)
    pabpv = const1 * const2 * bpv_sum

    # Debiased PABPV (with Hautsch-Podolskij noise correction)
    # Ref: realized_preaveraged_bipower_variation.m:116
    # MATLAB: rpav = const1 * const2 * sum(abs(lead.*lag)) - bias
    pabpv_debiased = pabpv - bias

    # Check for unexpected negative debiased result
    if pabpv_debiased < 0:
        warnings.warn(
            "Debiased pre-averaged bipower variation is negative "
            f"({pabpv_debiased:.6e}). This may indicate the noise "
            "correction is too large relative to the signal.",
            stacklevel=2,
        )

    # ================================================================
    # Step 8: Build diagnostics dictionary
    # ================================================================
    diagnostics: dict = {
        'K': K,
        'theta': theta,
        'psi_1': psi_1,
        'psi_2': psi_2,
        'mu_1': float(mu_1),
        'omega': omega,
        'noise_variance': float(noise_variance),
        'noise_estimate_oomen': float(noise_estimate_oomen),
        'bias': bias,
        'n_returns': m,
        'n_preav_returns': n_preav,
        'n_bpv_terms': n_bpv_terms,
        'const1': const1,
        'const2': const2,
    }

    return float(pabpv), float(pabpv_debiased), diagnostics
