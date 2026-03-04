"""
Quantile Realized Variance (QRV) estimator.

Implements the Quantile Realized Variance of Christensen, Oomen and Podolskij
(2008), the MinRV and MedRV estimators of Andersen, Dobrev and Schaumburg, and
the general symmetrized version suggested by Sheppard.

The estimator partitions filtered returns into blocks, selects order-statistic
quantiles from each block, and combines them using minimum-variance (GMVP)
weights derived from precomputed (or on-the-fly Monte Carlo) scale factors.

Migrated from: realized/realized_quantile_variance.m (359 lines)
Original author: Kevin Sheppard
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008

See Also
--------
mfe_toolbox.realized.realized_min_med_variance : Convenience MinRV/MedRV.
mfe_toolbox.realized.realized_kernel : Realized kernel estimator.
mfe_toolbox.realized.realized_variance : Standard realized variance.
mfe_toolbox.realized.realized_bipower_variation : Bipower variation.
mfe_toolbox.realized.realized_threshold_variance : Threshold RV.
"""

import os
import warnings
from pathlib import Path

import numpy as np

from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_subsample import realized_subsample
from mfe_toolbox.realized.realized_quantile_variance_scale import (
    realized_quantile_variance_scale,
)


# ---------------------------------------------------------------------------
# Precomputed block sizes (from realized_quantile_scales.mat via fixtures)
# Ref: realized_quantile_variance.m:74-81 (COMMENTS section)
# ---------------------------------------------------------------------------
_ASYMMETRIC_PRECOMPUTED_BLOCK_SIZES = np.array(
    [4, 5, 6, 10, 13, 15, 18, 20, 25, 26, 30, 36, 39, 50, 60, 65, 72, 75, 100, 144],
    dtype=np.int64,
)
_SYMMETRIC_PRECOMPUTED_BLOCK_SIZES = np.array(
    [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 18, 20, 24, 25, 26, 30, 36, 39, 40,
     50, 60, 65, 72, 75, 100, 144],
    dtype=np.int64,
)


def _resolve_fixture_dir() -> Path:
    """Resolve the directory containing precomputed quantile scale fixtures.

    Checks the optional ``MFE_FIXTURE_DIR`` environment variable first
    (per AAP §0.7.2), then falls back to the default test fixtures location
    relative to the repository root.

    Returns
    -------
    Path
        Directory path where ``realized_quantile_scales_*.npy`` files reside.
    """
    # Ref: AAP §0.7.2 — optional MFE_FIXTURE_DIR environment variable
    env_dir = os.environ.get("MFE_FIXTURE_DIR")
    if env_dir is not None:
        candidate = Path(env_dir) / "realized"
        if candidate.exists():
            return candidate
        # Also check if MFE_FIXTURE_DIR already points to the realized subdir
        candidate_root = Path(env_dir)
        if candidate_root.exists():
            return candidate_root

    # Default: tests/fixtures/realized relative to project root
    # Walk up from this module's location to find the repo root
    module_dir = Path(__file__).resolve().parent  # mfe_toolbox/realized/
    repo_root = module_dir.parent.parent           # mfe_toolbox/../..
    default_dir = repo_root / "tests" / "fixtures" / "realized"
    if default_dir.exists():
        return default_dir

    # Fallback: return the default path even if it doesn't exist yet
    return default_dir


def _load_precomputed_scales(
    block_size: int, symmetric: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Load precomputed quantile scales, quantile values, and covariance.

    Parameters
    ----------
    block_size : int
        Number of returns per block (SAMPLESPERBIN).
    symmetric : bool
        Whether to load symmetric or asymmetric scales.

    Returns
    -------
    expected_quantiles : np.ndarray
        Expected values of squared order statistics (scale factors).
    simulation_quantile : np.ndarray
        Quantile values used in the simulation.
    expected_covariance : np.ndarray
        Non-scaled covariance matrix.
    found : bool
        True if precomputed data was found and loaded successfully.
    """
    fixture_dir = _resolve_fixture_dir()
    prefix = "symmetric" if symmetric else "asymmetric"
    precomputed_sizes = (
        _SYMMETRIC_PRECOMPUTED_BLOCK_SIZES if symmetric
        else _ASYMMETRIC_PRECOMPUTED_BLOCK_SIZES
    )

    if block_size not in precomputed_sizes:
        return np.array([]), np.array([]), np.array([[]]), False

    # Find the cell index for this block size
    # Ref: realized_quantile_variance.m:333 — [~,pl]=ismember(blockSize,simulationBlockSize)
    # The cell index corresponds to the position in the sorted block-size list.
    cell_idx = int(np.where(precomputed_sizes == block_size)[0][0])

    try:
        eq_path = fixture_dir / f"realized_quantile_scales_{prefix}ExpectedQuantiles_cell{cell_idx}.npy"
        sq_path = fixture_dir / f"realized_quantile_scales_{prefix}SimulationQuantile_cell{cell_idx}.npy"
        cov_path = fixture_dir / f"realized_quantile_scales_{prefix}ExpectedCovariance_cell{cell_idx}.npy"

        if not (eq_path.exists() and sq_path.exists() and cov_path.exists()):
            return np.array([]), np.array([]), np.array([[]]), False

        expected_quantiles = np.load(str(eq_path))
        simulation_quantile = np.load(str(sq_path))
        expected_covariance = np.load(str(cov_path))

        return expected_quantiles, simulation_quantile, expected_covariance, True
    except Exception:
        return np.array([]), np.array([]), np.array([[]]), False


def _realized_quantile_variance_core(
    rqindiv: np.ndarray,
    block_size: int,
    quantiles: np.ndarray,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute weighted quantile RV from individual quantile estimates.

    Inner helper implementing lines 313-360 of the MATLAB source.
    Loads precomputed scale factors and covariance matrices, or computes
    them on-the-fly via Monte Carlo if the requested block_size is not
    among the precomputed values.

    Parameters
    ----------
    rqindiv : np.ndarray
        Shape ``(q,)`` vector of averaged individual quantile RV estimates
        (one per quantile).
    block_size : int
        Number of returns per block (SAMPLESPERBIN).
    quantiles : np.ndarray
        Shape ``(q,)`` vector of quantile values.
    symmetric : bool
        Whether the estimator uses absolute returns (symmetric) or
        upper/lower pairs (asymmetric).

    Returns
    -------
    rq : np.ndarray
        Scalar (or 1-element array) — the minimum-variance combined QRV.
    rqindiv : np.ndarray
        Shape ``(q,)`` — scaled individual quantile estimates.
    rqcov : np.ndarray
        Shape ``(q, q)`` — non-scaled covariance matrix.
    rqweights : np.ndarray
        Shape ``(q,)`` — GMVP weights.

    Notes
    -----
    Ref: realized_quantile_variance.m:313-360
    """
    q = len(quantiles)

    # Ref: realized_quantile_variance.m:315 — load('realized_quantile_scales')
    expected_quantiles, simulation_quantile, expected_covariance, found = (
        _load_precomputed_scales(block_size, symmetric)
    )

    if found:
        # Ref: realized_quantile_variance.m:329-346
        this_scale = expected_quantiles
        this_quantile = simulation_quantile
        this_covariance = expected_covariance

        # Ref: realized_quantile_variance.m:338-341
        # Find closest matching quantile positions
        # MATLAB: for i = 1:q, [~,indicator(i)] = min(abs(thisQuantile-quantiles(i))); end
        indicator = np.zeros(q, dtype=np.intp)
        for i in range(q):
            indicator[i] = int(np.argmin(np.abs(this_quantile - quantiles[i])))

        # Ref: realized_quantile_variance.m:342
        rq_expected_squared_quantile_value = this_scale[indicator]
        # Ref: realized_quantile_variance.m:344
        rqcov = this_covariance[np.ix_(indicator, indicator)]

        # Ref: realized_quantile_variance.m:345-346
        # rqCovInv = rqcov \ eye(q)  →  np.linalg.solve(rqcov, np.eye(q))
        rq_cov_inv = np.linalg.solve(rqcov, np.eye(q))
        ones_q = np.ones(q)
        # Ref: realized_quantile_variance.m:346
        # rqweights = rqCovInv*ones(q,1)/(ones(1,q)*rqCovInv*ones(q,1))
        numerator = rq_cov_inv @ ones_q
        denominator = ones_q @ rq_cov_inv @ ones_q
        rqweights = numerator / denominator
    else:
        # Ref: realized_quantile_variance.m:348-354
        # Need to simulate using 1,000,000 simulations
        if block_size > 100:
            warnings.warn(
                "Computing the scales needed.  Since SAMPLESPERBIN is very "
                "large this may take a long time.\nConsider pre-computing "
                "this value, especially if using this value of SAMPLESPERBIN "
                "many times.",
                stacklevel=3,
            )
        else:
            warnings.warn(
                "Computing the scales needed.\nConsider pre-computing this "
                "value, especially if using this value of SAMPLESPERBIN "
                "many times.",
                stacklevel=3,
            )
        # Ref: realized_quantile_variance.m:354
        # MATLAB uses undefined adjSamplesPerBin; corrected to block_size.
        # Also symmetric parameter was missing in MATLAB call; added here.
        # Ref: realized_quantile_variance.m:354 — MATLAB uses undefined
        # adjSamplesPerBin; corrected to block_size; symmetric parameter added
        rq_expected_squared_quantile_value, rqweights, rqcov = (
            realized_quantile_variance_scale(
                block_size, quantiles, 1_000_000, symmetric=symmetric
            )
        )

    # Ref: realized_quantile_variance.m:358
    # rqindiv = rqindiv ./ rqExpectedSquaredQuantileValue
    rqindiv = rqindiv / rq_expected_squared_quantile_value

    # Ref: realized_quantile_variance.m:359
    # rq = rqweights' * rqindiv'
    # MATLAB: rqweights is (q,1), rqindiv is (1,q), rqweights' is (1,q)
    # rqweights' * rqindiv' = (1,q) * (q,1) = scalar
    rq = np.dot(rqweights, rqindiv)
    rq = np.atleast_1d(np.asarray(rq, dtype=np.float64))

    return rq, rqindiv, rqcov, rqweights


def realized_quantile_variance(
    price,
    time,
    time_type: str,
    sampling_type: str,
    sampling_interval,
    quantiles,
    block_size: int,
    symmetric: bool = False,
    overlap: bool = True,
    subsamples: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute the Quantile Realized Variance (QRV).

    Implements the QRV estimator of Christensen, Oomen and Podolskij (2008),
    the MinRV and MedRV estimators of Andersen, Dobrev and Schaumburg, and
    the general symmetrized version suggested by Sheppard.

    Parameters
    ----------
    price : array_like
        m-element 1-D vector of high-frequency prices.
    time : array_like
        m-element 1-D vector of observation times corresponding to *price*.
        Must be sorted and increasing.
    time_type : str
        Time format descriptor.  Case-insensitive.  One of:

        * ``'wall'``    — 24-hour clock in HHMMSS format (e.g. 101543).
        * ``'seconds'`` — Seconds past midnight.
        * ``'unit'``    — Unit-normalised [0, 1].
    sampling_type : str
        Sampling scheme.  Case-insensitive.  One of:

        * ``'CalendarTime'``
        * ``'CalendarUniform'``
        * ``'BusinessTime'``
        * ``'BusinessUniform'``
        * ``'Fixed'``
    sampling_interval : int, float, or array_like
        Meaning depends on *sampling_type*.
    quantiles : array_like
        k-element vector of quantile values.

        * When ``symmetric=False``: ``0.5 < quantiles <= 1``.
        * When ``symmetric=True``:  ``0 < quantiles <= 1``.

        In either case, ``quantiles * block_size`` must produce integer values.
    block_size : int
        Number of returns per block (SAMPLESPERBIN).  Must be >= 2.
    symmetric : bool, optional
        If True, base the estimator on absolute returns.  Default is False
        (Christensen-Oomen-Podolskij convention).
    overlap : bool, optional
        If True (default), use all overlapping blocks.  If False, use only
        non-overlapping blocks.
    subsamples : int, optional
        Number of subsample RV estimators to average.  Default is 1
        (no subsampling beyond the primary grid).

    Returns
    -------
    rq : numpy.ndarray
        Quantile realized variance (scalar wrapped in 1-D array).
    rq_ss : numpy.ndarray
        Subsample-based QRV.  Equals *rq* if ``subsamples <= 1``.
    diagnostics : dict
        Diagnostic information with keys:

        * ``'rqindiv'`` — k-element scaled individual quantile RV estimates.
        * ``'rqcov'``   — (k, k) non-scaled covariance matrix.
        * ``'rqweights'`` — k-element GMVP combination weights.
        * ``'rqindiv_ss'`` — k-element subsample individual estimates.

    Raises
    ------
    ValueError
        If any input parameter is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_quantile_variance import (
    ...     realized_quantile_variance,
    ... )
    >>> prices = np.exp(np.cumsum(np.random.default_rng(42).standard_normal(200) * 0.001))
    >>> times = np.linspace(0.0, 1.0, 200)
    >>> rq, rq_ss, diag = realized_quantile_variance(
    ...     prices, times, 'unit', 'CalendarUniform', 100,
    ...     np.array([0.6, 0.7333, 0.9]), 30, symmetric=False,
    ... )

    Notes
    -----
    The cases where ``block_size`` is in
    ``{4, 5, 6, 10, 13, 15, 18, 20, 25, 26, 30, 36, 39, 50, 60, 65, 72,
    75, 100, 144}`` (asymmetric) or
    ``{2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 18, 20, 24, 25, 26, 30, 36,
    39, 40, 50, 60, 65, 72, 75, 100, 144}`` (symmetric) use precomputed
    scales from 100,000,000 Monte Carlo simulations.  Other block sizes
    trigger an on-the-fly computation with 1,000,000 simulations.

    Migrated from ``realized/realized_quantile_variance.m``.
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_quantile_variance.m:112-218
    # ==================================================================

    # --- PRICE ---
    # Ref: realized_quantile_variance.m:131-136
    price = np.asarray(price, dtype=np.float64).ravel()
    if price.ndim != 1 or price.size < 2:
        raise ValueError("PRICE must be a 1-D vector with at least 2 elements.")

    # --- TIME ---
    # Ref: realized_quantile_variance.m:137-145
    time = np.asarray(time, dtype=np.float64).ravel()
    if np.any(np.diff(time) < 0):
        raise ValueError("TIME must be sorted and increasing")
    if len(time) != len(price):
        raise ValueError("TIME must be a m by 1 vector.")
    # Ref: realized_quantile_variance.m:147 — protect against integer times
    # Already handled by np.float64 coercion above.

    # --- TIMETYPE ---
    # Ref: realized_quantile_variance.m:149-152
    time_type = time_type.lower()
    if time_type not in ("wall", "seconds", "unit"):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # --- SAMPLINGTYPE ---
    # Ref: realized_quantile_variance.m:153-156
    sampling_type = sampling_type.lower()
    valid_sampling_types = (
        "calendartime", "calendaruniform",
        "businesstime", "businessuniform", "fixed",
    )
    if sampling_type not in valid_sampling_types:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # --- SAMPLINGINTERVAL ---
    # Ref: realized_quantile_variance.m:158-176
    m = len(price)
    t0 = time[0]
    tT = time[m - 1]
    if sampling_type in ("calendartime", "calendaruniform",
                         "businesstime", "businessuniform"):
        # Ref: realized_quantile_variance.m:163-165
        if (not np.isscalar(sampling_interval)
                or np.floor(float(sampling_interval)) != float(sampling_interval)
                or float(sampling_interval) < 1):
            raise ValueError(
                "SAMPLINGINTERVAL must be a positive integer for the "
                "SAMPLINGTYPE selected."
            )
    else:
        # Ref: realized_quantile_variance.m:167-176 — Fixed sampling
        sampling_interval = np.asarray(sampling_interval, dtype=np.float64).ravel()
        if not (np.any(sampling_interval >= t0) and np.any(sampling_interval <= tT)):
            raise ValueError(
                "At least one sampling interval must be between min(TIME) "
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )
        if len(sampling_interval) > 1 and np.any(np.diff(sampling_interval) <= 0):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                "times in SAMPLINGINTERVAL must be sorted and strictly "
                "increasing."
            )

    # --- BLOCKSIZE ---
    # Ref: realized_quantile_variance.m:178-181
    block_size = int(block_size)
    if block_size < 2:
        raise ValueError(
            "SAMPLESPERBIN must be a positive integer greater than or equal "
            "to 2 (and usually at least 20)"
        )

    # --- SYMMETRIC ---
    # Ref: realized_quantile_variance.m:183-190
    if symmetric is None:
        symmetric = False
    symmetric = bool(symmetric)

    # --- QUANTILES ---
    # Ref: realized_quantile_variance.m:192-201
    quantiles = np.asarray(quantiles, dtype=np.float64).ravel()
    if symmetric:
        if (np.any(quantiles <= 0) or np.any(quantiles > 1)
                or len(quantiles) > block_size
                or len(np.unique(np.round(quantiles * block_size).astype(np.int64))) != len(quantiles)):
            raise ValueError(
                "QUANTILES must be unique, greater than 0 and less than or "
                "equal to 1, and the number of quantiles must be smaller "
                "than SAMPLESPERBIN when SYMMETRIC = true."
            )
    else:
        if (np.any(quantiles <= 0.5) or np.any(quantiles > 1)
                or len(quantiles) > (block_size // 2)
                or len(np.unique(np.round(quantiles * block_size).astype(np.int64))) != len(quantiles)):
            raise ValueError(
                "QUANTILES must be unique, greater than 0.5 and less than "
                "or equal to 1, and the number of quantiles must be smaller "
                "than 0.5*SAMPLESPERBIN when SYMMETRIC = false."
            )

    # --- OVERLAP ---
    # Ref: realized_quantile_variance.m:203-210
    if overlap is None:
        overlap = True
    overlap = bool(overlap)

    # --- SUBSAMPLES ---
    # Ref: realized_quantile_variance.m:212-218
    if subsamples is None:
        subsamples = 1
    subsamples = int(subsamples)
    if subsamples < 0:
        raise ValueError("SUBSAMPLES must be a non-negative integer.")

    # ==================================================================
    # 1. Compute returns
    # Ref: realized_quantile_variance.m:224-234
    # ==================================================================

    # Ref: realized_quantile_variance.m:225
    log_price = np.log(price)

    # Ref: realized_quantile_variance.m:226
    # filteredLogPrice = realized_price_filter(logPrice, time, timeType,
    #                                          samplingType, samplingInterval)
    filtered_log_price = realized_price_filter(
        log_price, time, time_type, sampling_type, sampling_interval
    )
    # realized_price_filter returns (filtered_price, filtered_time, actual_time)
    # Ref: realized_quantile_variance.m:226 — only uses the first output
    if isinstance(filtered_log_price, tuple):
        filtered_log_price = filtered_log_price[0]

    # Ref: realized_quantile_variance.m:227 — returns = diff(filteredLogPrice)
    returns = np.diff(filtered_log_price, axis=0)
    # Ref: realized_quantile_variance.m:229
    n = len(returns)

    # Ref: realized_quantile_variance.m:230-234 — non-overlap compatibility check
    if not overlap:
        if n % block_size != 0:
            raise ValueError(
                "The number of returns computed from the prices returned "
                "from realized_price_filter must be an integer multiple of "
                f"SAMPLESPERBIN.  The number of returns produced is {n}"
            )

    # ==================================================================
    # 2. Build blocks and compute quantile RV
    # Ref: realized_quantile_variance.m:236-272
    # ==================================================================

    # Ref: realized_quantile_variance.m:236-239 — bin start indices
    # MATLAB: binStart = 1:n-blockSize+1  (1-based, overlapping)
    # Python: range(0, n - block_size + 1)  (0-based)
    if overlap:
        bin_starts = np.arange(0, n - block_size + 1)
    else:
        # Ref: realized_quantile_variance.m:239
        # MATLAB: binStart = 1:blockSize:n  →  Python: range(0, n, block_size)
        bin_starts = np.arange(0, n, block_size)

    # Ref: realized_quantile_variance.m:241 — scale returns
    returns = returns * np.sqrt(n)

    # Ref: realized_quantile_variance.m:242-243 — symmetric → absolute values
    if symmetric:
        returns = np.abs(returns)

    q = len(quantiles)

    # Ref: realized_quantile_variance.m:246
    rqindiv_mat = np.zeros((len(bin_starts), q), dtype=np.float64)

    # Ref: realized_quantile_variance.m:250-256 — compute quantile indices
    if symmetric:
        # Ref: realized_quantile_variance.m:252
        # MATLAB: indices = round(quantiles*blockSize)  (1-based)
        # Python: subtract 1 for 0-based indexing
        indices = np.round(quantiles * block_size).astype(np.intp) - 1
    else:
        # Ref: realized_quantile_variance.m:254-255
        # MATLAB: upperIndices = round(quantiles*blockSize)  (1-based)
        # MATLAB: lowerIndices = round((1-quantiles)*blockSize+1)  (1-based)
        # Python: subtract 1 for 0-based indexing
        upper_indices = np.round(quantiles * block_size).astype(np.intp) - 1
        lower_indices = np.round((1.0 - quantiles) * block_size + 1.0).astype(np.intp) - 1

    # Ref: realized_quantile_variance.m:261-271 — loop over bins
    for count, j in enumerate(bin_starts):
        # Ref: realized_quantile_variance.m:263
        # MATLAB: binreturns = returns(j:j+blockSize-1)
        # Python: 0-based slicing [j : j + block_size]
        bin_returns = returns[j: j + block_size]

        # Ref: realized_quantile_variance.m:264
        # MATLAB: binreturns = sort(binreturns)'  → sorted column → row
        # Python: np.sort along axis 0 (same effect for 1-D)
        bin_returns = np.sort(bin_returns)

        if symmetric:
            # Ref: realized_quantile_variance.m:266
            # MATLAB: rqindiv(count,:) = binreturns(indices).^2
            # Python: 0-based indices already adjusted above
            rqindiv_mat[count, :] = bin_returns[indices] ** 2
        else:
            # Ref: realized_quantile_variance.m:268
            # MATLAB: rqindiv(count,:) = binreturns(lowerIndices).^2 + binreturns(upperIndices).^2
            rqindiv_mat[count, :] = (
                bin_returns[lower_indices] ** 2 + bin_returns[upper_indices] ** 2
            )

    # Ref: realized_quantile_variance.m:272 — average over bins
    # MATLAB: mean(rqindiv) computes column means → (1, q)
    rqindiv_avg = np.mean(rqindiv_mat, axis=0)

    # Ref: realized_quantile_variance.m:274
    rq, rqindiv_scaled, rqcov, rqweights = _realized_quantile_variance_core(
        rqindiv_avg, block_size, quantiles, symmetric
    )

    # ==================================================================
    # 3. Build diagnostics
    # Ref: realized_quantile_variance.m:276-278
    # ==================================================================
    diagnostics: dict = {
        "rqindiv": rqindiv_scaled,
        "rqcov": rqcov,
        "rqweights": rqweights,
    }

    # ==================================================================
    # 4. Subsampling
    # Ref: realized_quantile_variance.m:281-308
    # ==================================================================

    if subsamples <= 0:
        # No subsampling — rq_ss equals rq
        rq_ss = rq.copy()
        diagnostics["rqindiv_ss"] = rqindiv_scaled.copy()
        return rq, rq_ss, diagnostics

    # Ref: realized_quantile_variance.m:281
    # subsampledLogPrice = realized_subsample(logPrice, time, timeType,
    #     samplingType, samplingInterval, subsamples)
    subsampled_data = realized_subsample(
        log_price, time, time_type, sampling_type, sampling_interval, subsamples
    )

    # Ref: realized_quantile_variance.m:282
    # rqindiv = nan(subsamples * length(binStart), q)
    # Pre-allocate with NaN — will trim unused rows later
    max_rows = subsamples * len(bin_starts) * 2  # generous upper bound
    rqindiv_ss_mat = np.full((max_rows, q), np.nan, dtype=np.float64)

    count_ss = 0

    # Ref: realized_quantile_variance.m:284-303 — loop over subsamples
    # MATLAB: for i = 1:subsamples
    for i in range(subsamples):
        # Ref: realized_quantile_variance.m:285
        # MATLAB: filteredLogPrice = subsampledLogPrice{i}
        # Python: realized_subsample returns list of tuples (prices, times, bc, tc)
        # Ref: realized_quantile_variance.m:285 — cell{i} → list[i] (0-based)
        ss_filtered_log_price = subsampled_data[i][0]

        # Ref: realized_quantile_variance.m:286
        ss_returns = np.diff(ss_filtered_log_price, axis=0)
        ss_n = len(ss_returns)

        if ss_n < block_size:
            # Not enough returns for even one block — skip this subsample
            continue

        # Ref: realized_quantile_variance.m:288
        ss_returns = ss_returns * np.sqrt(ss_n)

        # Ref: realized_quantile_variance.m:289-291
        if symmetric:
            ss_returns = np.abs(ss_returns)

        # Ref: realized_quantile_variance.m:292
        # MATLAB: binStart = 1:n-blockSize+1  (1-based, always overlapping for subsamples)
        # Python: range(0, ss_n - block_size + 1)
        ss_bin_starts = np.arange(0, ss_n - block_size + 1)

        # Ref: realized_quantile_variance.m:293-302 — inner block loop
        for j in ss_bin_starts:
            # Expand storage if needed
            if count_ss >= rqindiv_ss_mat.shape[0]:
                extra = np.full((max_rows, q), np.nan, dtype=np.float64)
                rqindiv_ss_mat = np.concatenate([rqindiv_ss_mat, extra], axis=0)

            # Ref: realized_quantile_variance.m:294
            ss_bin_returns = ss_returns[j: j + block_size]
            # Ref: realized_quantile_variance.m:295
            ss_bin_returns = np.sort(ss_bin_returns)

            if symmetric:
                # Ref: realized_quantile_variance.m:297
                rqindiv_ss_mat[count_ss, :] = ss_bin_returns[indices] ** 2
            else:
                # Ref: realized_quantile_variance.m:299
                rqindiv_ss_mat[count_ss, :] = (
                    ss_bin_returns[lower_indices] ** 2
                    + ss_bin_returns[upper_indices] ** 2
                )
            count_ss += 1

    # Ref: realized_quantile_variance.m:304
    # rqindiv = rqindiv(1:count-1,:)  → trim unused NaN rows
    rqindiv_ss_mat = rqindiv_ss_mat[:count_ss, :]

    # Ref: realized_quantile_variance.m:305
    # rqindiv = mean(rqindiv)
    if count_ss > 0:
        rqindiv_ss_avg = np.mean(rqindiv_ss_mat, axis=0)
    else:
        # Edge case: no valid blocks in subsamples
        rqindiv_ss_avg = rqindiv_avg.copy()

    # Ref: realized_quantile_variance.m:307
    rq_ss, rqindiv_ss_scaled, _, _ = _realized_quantile_variance_core(
        rqindiv_ss_avg, block_size, quantiles, symmetric
    )

    # Ref: realized_quantile_variance.m:308
    diagnostics["rqindiv_ss"] = rqindiv_ss_scaled

    return rq, rq_ss, diagnostics
