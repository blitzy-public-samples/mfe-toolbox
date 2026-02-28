"""
Microstructure noise variance estimation for realized kernel bandwidth selection.

Implements the Bandi-Russell noise variance estimator and the BNHLS (Barndorff-Nielsen,
Hansen, Lunde, Shephard) debiased variant, plus the Oomen (2006) AC(1) alternative
estimator. This module is a helper for the realized kernel estimator and related
noise-sensitive realized volatility estimators.

Migrated from: realized/realized_noise_estimate.m
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 5/1/2008

See Also
--------
realized_kernel : Realized kernel quadratic variation estimator.
realized_price_filter : Price filtering for realized estimators.
realized_kernel_weights : Kernel weight computation.
realized_variance : Standard realized variance estimator.
"""

import numpy as np


def realized_noise_estimate(
    price: np.ndarray,
    time: np.ndarray,
    time_type: str,
    options: dict,
) -> tuple[float, float, float, float]:
    """Estimate microstructure noise variance from high-frequency price data.

    Computes the Bandi-Russell noise variance estimate, the BNHLS debiased
    noise variance, an IQ (integrated quarticity) lower bound, and the
    Oomen (2006) AC(1) noise estimator.

    Parameters
    ----------
    price : np.ndarray
        1-D array of m prices. If a row vector is provided, it is automatically
        transposed to a column vector.
    time : np.ndarray
        1-D array of m times corresponding to ``price``. Must be sorted in
        strictly ascending order.
    time_type : str
        String indicating the time format:
        - ``'wall'``    : 24-hour clock HHMMSS (e.g. 101543)
        - ``'seconds'`` : seconds past midnight
        - ``'unit'``    : unit-normalized [0, 1] interval
    options : dict
        Realized kernel options dictionary. Required keys:

        - ``'medFrequencyKernel'`` : str — kernel type for medium-frequency RK
        - ``'medFrequencySamplingType'`` : str — sampling type for med-freq RK
        - ``'medFrequencySamplingInterval'`` : int — sampling interval for med-freq RK
        - ``'medFrequencyBandwidth'`` : int or None — bandwidth for med-freq RK
        - ``'noiseVarianceSamplingType'`` : str — sampling type for noise RV
        - ``'noiseVarianceSamplingInterval'`` : int — sampling interval for noise RV
        - ``'IQEstimationSamplingType'`` : str — sampling type for IQ estimation
        - ``'IQEstimationSamplingInterval'`` : int — sampling interval for IQ estimation
        - ``'useAdjustedNoiseCount'`` : bool — if True, only count non-zero returns

    Returns
    -------
    noise_variance : float
        Bandi-Russell noise variance estimate: RV / (2 * n) where RV is the
        realized variance at the noise-frequency sampling and n is the
        effective number of returns.
    debiased_noise_variance : float
        BNHLS debiased noise variance: ``exp(log(noise_variance) - RK / RV)``
        where RK is a medium-frequency realized kernel and RV is the
        corresponding realized variance.
    iq_estimate : float
        Lower bound estimate of the integrated quarticity based on
        low-frequency realized variance squared.
    noise_estimate_oomen : float
        Oomen (2006) AC(1) noise variance estimate:
        ``-1/(n-1) * r[:-1].T @ r[1:]`` where r are log returns of filtered
        prices.

    Raises
    ------
    ValueError
        If inputs are invalid (wrong shape, unsorted time, invalid time_type,
        missing options fields).

    Notes
    -----
    This is a helper function for :func:`realized_kernel`. See
    Barndorff-Nielsen, Hansen, Lunde and Shephard (2008) for details about
    the optimal selection of bandwidth.

    References
    ----------
    .. [1] Bandi, F. M. and Russell, J. R. (2008). "Microstructure noise,
       realized variance, and optimal sampling." *Review of Economic Studies*,
       75(2), 339–369.
    .. [2] Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N.
       (2008). "Designing realized kernels to measure the ex post variation
       of equity prices in the presence of noise." *Econometrica*, 76(6),
       1481–1536.
    .. [3] Oomen, R. C. A. (2006). "Properties of realized variance under
       alternative sampling schemes." *Journal of Business and Economic
       Statistics*, 24(2), 219–237.
    """
    # ----------------------------------------------------------------
    # Import sibling modules lazily to avoid circular imports and to
    # match MATLAB's implicit path-based function resolution.
    # These modules are all part of the mfe_toolbox.realized subpackage.
    # ----------------------------------------------------------------
    from mfe_toolbox.realized.realized_variance import realized_variance
    from mfe_toolbox.realized.realized_price_filter import realized_price_filter
    from mfe_toolbox.realized.realized_options import realized_options
    from mfe_toolbox.realized.realized_kernel import realized_kernel

    # ================================================================
    # Input Checking — Ref: realized_noise_estimate.m:39-93
    # ================================================================

    # Validate price array
    # Ref: realized_noise_estimate.m:42-46 — auto-transpose row to column
    price = np.asarray(price, dtype=np.float64).ravel()
    if price.ndim != 1 or price.size < 2:
        raise ValueError("PRICE must be a m by 1 vector with at least 2 elements.")

    # Validate time array
    # Ref: realized_noise_estimate.m:48-56 — auto-transpose, sort check, length match
    time = np.asarray(time, dtype=np.float64).ravel()
    if np.any(np.diff(time) < 0):
        raise ValueError("TIME must be sorted and increasing")
    if time.ndim != 1 or len(time) != len(price):
        raise ValueError("TIME must be a m by 1 vector.")

    # Validate time_type
    # Ref: realized_noise_estimate.m:58-61
    # Note: MATLAB source checks for {'wall','seconds','matlab'} but the
    # Python codebase standardizes on 'unit' instead of 'matlab'.
    # Accept both for backward compatibility, normalizing 'matlab' → 'unit'.
    time_type = time_type.lower()
    valid_time_types = {"wall", "seconds", "unit", "matlab"}
    if time_type not in valid_time_types:
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )
    # Normalize 'matlab' to 'unit' for downstream Python modules
    if time_type == "matlab":
        time_type = "unit"

    # ----------------------------------------------------------------
    # Kernel type validation
    # Ref: realized_noise_estimate.m:64-76
    # ----------------------------------------------------------------

    # Flat-top kernel list
    flat_top_kernel_list = [
        "bartlett", "twoscale", "2ndorder", "epanechnikov",
        "cubic", "multiscale", "5thorder", "6thorder", "7thorder",
        "8thorder", "parzen", "th1", "th2", "th5", "th16",
    ]

    # Non-flat-top kernel list
    non_flat_top_kernel_list = [
        "nonflatparzen", "qs", "fejer", "thinf", "bnhls",
    ]

    # Combined kernel list
    kernel_list = flat_top_kernel_list + non_flat_top_kernel_list

    if "medFrequencyKernel" not in options or options["medFrequencyKernel"] not in kernel_list:
        raise ValueError(
            "KERNEL must be a field of OPTIONS and one of the listed types."
        )

    # ----------------------------------------------------------------
    # Extract options fields
    # Ref: realized_noise_estimate.m:78-90
    # ----------------------------------------------------------------
    med_frequency_sampling_type = options["medFrequencySamplingType"]
    med_frequency_sampling_interval = options["medFrequencySamplingInterval"]
    med_frequency_kernel = options["medFrequencyKernel"]
    med_frequency_bandwidth = options["medFrequencyBandwidth"]

    # Validate medFrequencySamplingInterval
    # Ref: realized_noise_estimate.m:82-84
    if (
        not np.isscalar(med_frequency_sampling_interval)
        or int(med_frequency_sampling_interval) != med_frequency_sampling_interval
        or med_frequency_sampling_interval <= 0
    ):
        raise ValueError("MEDIUMFREQUENCYTIME must be a positive integer scalar value.")

    iq_estimation_sampling_type = options["IQEstimationSamplingType"]
    iq_estimation_sampling_interval = options["IQEstimationSamplingInterval"]

    # Validate IQEstimationSamplingInterval
    # Ref: realized_noise_estimate.m:88-90
    if (
        not np.isscalar(iq_estimation_sampling_interval)
        or int(iq_estimation_sampling_interval) != iq_estimation_sampling_interval
        or iq_estimation_sampling_interval <= 0
    ):
        raise ValueError("LOWFREQUENCYTIME must be a positive integer scalar value.")

    # ================================================================
    # Computation
    # ================================================================

    # ------------------------------------------------------------------
    # Step 1: Low-frequency realized variance for IQ estimate
    # Ref: realized_noise_estimate.m:96
    # ------------------------------------------------------------------
    low_frequency_rv_result = realized_variance(
        price, time, time_type,
        iq_estimation_sampling_type, iq_estimation_sampling_interval,
    )
    # realized_variance returns (rv, ...) — extract scalar RV
    if isinstance(low_frequency_rv_result, tuple):
        low_frequency_realized_variance = float(low_frequency_rv_result[0])
    else:
        low_frequency_realized_variance = float(low_frequency_rv_result)

    # ------------------------------------------------------------------
    # Step 2: Noise variance via Bandi-Russell estimator (all prices)
    # Ref: realized_noise_estimate.m:98-111
    # ------------------------------------------------------------------
    noise_variance_sampling_type = options["noiseVarianceSamplingType"]
    noise_variance_sampling_interval = options["noiseVarianceSamplingInterval"]

    # Compute realized variance at noise-frequency sampling
    # Ref: realized_noise_estimate.m:102
    noise_rv_result = realized_variance(
        price, time, time_type,
        noise_variance_sampling_type, noise_variance_sampling_interval,
    )
    if isinstance(noise_rv_result, tuple):
        noise_variance = float(noise_rv_result[0])
    else:
        noise_variance = float(noise_rv_result)

    # ------------------------------------------------------------------
    # Step 3: Filter prices and compute effective n
    # Ref: realized_noise_estimate.m:104-110
    # ------------------------------------------------------------------
    filter_result = realized_price_filter(
        price, time, time_type,
        noise_variance_sampling_type, noise_variance_sampling_interval,
    )
    # realized_price_filter returns (filtered_price, filtered_time) or
    # (filtered_price, filtered_time, actual_time)
    if isinstance(filter_result, tuple):
        noise_filtered_price = np.asarray(filter_result[0], dtype=np.float64).ravel()
    else:
        noise_filtered_price = np.asarray(filter_result, dtype=np.float64).ravel()

    # Compute effective n for Bandi-Russell divisor
    # Ref: realized_noise_estimate.m:105-110
    use_adjusted_noise_count = options.get("useAdjustedNoiseCount", False)
    if use_adjusted_noise_count:
        # Ref: realized_noise_estimate.m:107 — only count non-zero returns
        n = int(np.sum(np.diff(noise_filtered_price) != 0))
    else:
        # Ref: realized_noise_estimate.m:109
        n = len(noise_filtered_price) - 1

    # Ensure n > 0 to prevent division by zero
    if n <= 0:
        n = 1

    # Bandi-Russell noise variance = RV / (2*n)
    # Ref: realized_noise_estimate.m:111
    noise_variance = noise_variance / (2.0 * n)

    # ------------------------------------------------------------------
    # Step 4: Oomen (2006) AC(1) noise estimator
    # Ref: realized_noise_estimate.m:113-115
    # ------------------------------------------------------------------
    noise_returns = np.diff(np.log(noise_filtered_price))
    n_returns = len(noise_returns)

    # Ref: realized_noise_estimate.m:115
    # noiseEstimateOomen = -1/(n-1) * noiseReturns(1:end-1)' * noiseReturns(2:end)
    # Note: MATLAB 1-indexed slicing [1:end-1] and [2:end] → Python [:-1] and [1:]
    if n_returns >= 2:
        noise_estimate_oomen = float(
            -1.0 / (n_returns - 1) * noise_returns[:-1] @ noise_returns[1:]
        )
    else:
        noise_estimate_oomen = 0.0

    # ------------------------------------------------------------------
    # Step 5: Medium-frequency realized kernel for BNHLS debiasing
    # Ref: realized_noise_estimate.m:118-124
    # ------------------------------------------------------------------

    # Build medium-frequency options for the realized kernel call
    # Ref: realized_noise_estimate.m:118-121
    med_frequency_options = realized_options("kernel")
    med_frequency_options["kernel"] = med_frequency_kernel
    med_frequency_options["bandwidth"] = med_frequency_bandwidth
    med_frequency_options["endTreatment"] = "stagger"

    # Compute medium-frequency realized kernel
    # Ref: realized_noise_estimate.m:123
    rk_result = realized_kernel(
        price, time, time_type,
        med_frequency_sampling_type, int(med_frequency_sampling_interval),
        med_frequency_options,
    )
    # realized_kernel returns (rk, rk_adjusted, diagnostics) — extract first value
    if isinstance(rk_result, tuple):
        med_frequency_realized_kernel = float(rk_result[0])
    else:
        med_frequency_realized_kernel = float(rk_result)

    # Compute medium-frequency realized variance
    # Ref: realized_noise_estimate.m:124
    med_rv_result = realized_variance(
        price, time, time_type,
        med_frequency_sampling_type, int(med_frequency_sampling_interval),
    )
    if isinstance(med_rv_result, tuple):
        med_frequency_realized_variance = float(med_rv_result[0])
    else:
        med_frequency_realized_variance = float(med_rv_result)

    # ------------------------------------------------------------------
    # Step 6: BNHLS debiased noise variance
    # Ref: realized_noise_estimate.m:126
    # debiasedNoiseVariance = exp(log(noiseVariance) - RK/RV)
    # ------------------------------------------------------------------
    if med_frequency_realized_variance != 0.0 and noise_variance > 0.0:
        debiased_noise_variance = float(
            np.exp(
                np.log(noise_variance)
                - med_frequency_realized_kernel / med_frequency_realized_variance
            )
        )
    else:
        # Fallback: if RV is zero or noise variance non-positive, use raw estimate
        debiased_noise_variance = noise_variance

    # ------------------------------------------------------------------
    # Step 7: IQ estimate as low-frequency RV squared
    # Ref: realized_noise_estimate.m:127
    # ------------------------------------------------------------------
    iq_estimate = low_frequency_realized_variance ** 2

    return noise_variance, debiased_noise_variance, iq_estimate, noise_estimate_oomen
