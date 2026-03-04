"""
Threshold multipower variation estimator for realized volatility.

Implements thresholded multipower variation using an adaptive Gaussian kernel
local variance proxy to identify and separate jumps from continuous price
variation.  The estimator computes both raw and bias-corrected (debiased)
versions of the threshold multipower variation.

The adaptive local variance is estimated iteratively:

1. Initialize V_i = Inf for all returns.
2. Classify returns: non-jump if r_i^2 < c^2 * V_i, jump otherwise.
3. Re-estimate V_i using a Gaussian-kernel-weighted average of squared
   non-jump returns in a neighbourhood of radius L around each return.
4. Repeat steps 2-3 until convergence (V does not change).

Bias correction adds the expected contribution from jump returns under the
null of Gaussian returns, following Corsi, Pirino, and Renò.

Migrated from: ``realized/realized_threshold_multipower_variation.m``

Original author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk

Notes
-----
**Known MATLAB source issues faithfully reproduced:**

* Ref: realized_threshold_multipower_variation.m:9,13 — The result of
  ``realized_price_filter`` is immediately overwritten with ``log(price)``,
  so the filtering step has no effect on the output.  This behaviour is
  preserved for strict numerical parity.

* Ref: realized_threshold_multipower_variation.m — The second output
  ``rvSS`` (subsampled threshold multipower variation) is declared in the
  function signature but never assigned in the MATLAB source.  The Python
  version returns ``np.nan`` for ``tmpv_ss`` and ``tmpv_ss_debiased`` to
  faithfully reproduce this known issue.

* Ref: realized_threshold_multipower_variation.m — The ``gamma`` (powers)
  parameter is declared but never used in the MATLAB implementation.
  The function computes a standard threshold variance (not multipower
  products).  The ``powers`` parameter is accepted for API compatibility
  but has no effect on computation.

See Also
--------
mfe_toolbox.realized.realized_threshold_variance :
    Threshold variance estimator with full subsampling support.
mfe_toolbox.realized.realized_variance :
    Standard realized variance estimator.
mfe_toolbox.realized.realized_bipower_variation :
    Bipower variation estimator.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import special
from scipy.stats import norm

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_subsample import realized_subsample


def realized_threshold_multipower_variation(
    price: np.ndarray,
    time: np.ndarray | None = None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval: int | float | np.ndarray = 1,
    powers: np.ndarray | None = None,
    subsamples: int = 1,
    options: dict | None = None,
) -> tuple[float, float, float, float, dict]:
    """Estimate threshold multipower variation with adaptive local variance.

    Uses an iterative Gaussian-kernel-weighted local variance proxy to
    identify jumps, then computes the threshold multipower variation
    (sum of squared non-jump returns) and a debiased version that adds
    the expected contribution of jump returns under the null hypothesis
    of Gaussian innovations.

    Parameters
    ----------
    price : array_like
        An m-element 1-D vector of high-frequency prices.  Must contain
        at least 2 elements to compute at least one return.
    time : array_like or None, optional
        An m-element 1-D vector of observation times where ``time[i]``
        corresponds to ``price[i]``.  Must be sorted in non-decreasing
        order.  Format must match *time_type*.  If ``None``, a default
        unit-interval grid ``linspace(0, 1, m)`` is used and *time_type*
        is forced to ``'unit'``.
    time_type : str, optional
        Time format descriptor (case-insensitive).  One of:

        * ``'wall'``    — 24-hour clock in HHMMSS format.
        * ``'seconds'`` — Seconds past midnight.
        * ``'unit'``    — Unit-normalised [0, 1].

        Default is ``'unit'``.
    sampling_type : str, optional
        Sampling scheme (case-insensitive).  One of:

        * ``'CalendarTime'``     — Fixed calendar-time interval.
        * ``'CalendarUniform'``  — Uniform spacing in calendar time.
        * ``'BusinessTime'``     — Every N-th tick.
        * ``'BusinessUniform'``  — Uniform tick spacing.
        * ``'Fixed'``            — Specific user-supplied times.

        Default is ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Interpretation depends on *sampling_type*.  Default is ``1``.
    powers : array_like or None, optional
        Power exponents for multipower variation.  **Not used** in the
        current implementation (MATLAB source does not use the ``gamma``
        parameter).  Accepted for API compatibility.
    subsamples : int, optional
        Number of subsamples.  **Not used** in the current implementation
        (the MATLAB source never assigns the subsampled output).
        Default is ``1``.
    options : dict or None, optional
        Estimation options.  Supported keys:

        * ``'thresholdConstant'`` (float) — Multiplier *c* for the
          threshold ``c^2 * V_i``.  Default ``3.0``.
        * ``'kernelRadius'`` (int) — Half-width *L* of the Gaussian
          kernel window used for local variance estimation.  Default
          ``25``.
        * ``'maxIterations'`` (int) — Maximum number of convergence
          iterations for the local variance estimation loop.  Default
          ``100``.

    Returns
    -------
    tmpv : float
        Raw threshold multipower variation (sum of squared non-jump
        returns, without bias correction).
    tmpv_ss : float
        Subsampled threshold multipower variation.  Always ``np.nan``
        because the MATLAB source never assigns ``rvSS`` (known issue).
    tmpv_debiased : float
        Debiased threshold multipower variation (sum of squared non-jump
        returns plus expected contribution from jump returns).
    tmpv_ss_debiased : float
        Subsampled debiased threshold multipower variation.  Always
        ``np.nan`` (MATLAB source known issue).
    diagnostics : dict
        Diagnostic information with keys:

        * ``'localVar'`` — ``np.ndarray`` (m-1,) of converged local
          variance estimates.
        * ``'returns'`` — ``np.ndarray`` (m-1,) of log returns used
          in the computation.

    Raises
    ------
    ValueError
        If *price* has fewer than 2 elements.
        If *time* is provided and has a different length than *price*.
        If *time* is not sorted in non-decreasing order.
        If *time_type* is not a recognised format.
        If *options* ``'thresholdConstant'`` is not a positive scalar.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([100.0, 100.5, 101.0, 100.8, 101.2, 101.5])
    >>> tmpv, tmpv_ss, tmpv_d, tmpv_ss_d, diag = (
    ...     realized_threshold_multipower_variation(prices)
    ... )
    >>> isinstance(tmpv, float)
    True
    >>> np.isnan(tmpv_ss)  # Known MATLAB bug: rvSS never assigned
    True

    Notes
    -----
    **MATLAB → Python translation details:**

    * ``normcdf(-c)`` → ``scipy.stats.norm.cdf(-c)``
      (Ref: duplication/normcdf.m replaced per AAP §0.5.2)
    * ``gamma(3/2)`` → ``scipy.special.gamma(1.5)``
    * ``gammainc(x, a, 'upper')`` → ``scipy.special.gammaincc(a, x)``
      (argument order swap; Ref: AAP §0.7.3)
    * ``error()`` → ``raise ValueError()``
    * MATLAB 1-based loop indexing → Python 0-based (Ref: AAP §0.7.3)
    * ``pl > 1`` in MATLAB (1-based) → ``pl > 0`` in Python (0-based)
    """
    # ==================================================================
    # Input Coercion and Validation
    # ==================================================================

    # --- price ---
    # Ref: realized_threshold_multipower_variation.m — no explicit validation,
    # but following the pattern from realized_threshold_variance.m:92-96.
    price = np.asarray(price, dtype=np.float64).ravel()
    if price.size < 2:
        raise ValueError(
            'PRICE must contain at least 2 elements to compute returns.'
        )

    # --- time ---
    # If time is not provided, create a default unit-interval grid.
    if time is None:
        n_obs = price.size
        time = np.linspace(0.0, 1.0, n_obs)
        time_type = 'unit'
    else:
        time = np.asarray(time, dtype=np.float64).ravel()

    # Validate time length matches price
    if time.size != price.size:
        raise ValueError(
            'TIME must have the same number of elements as PRICE.'
        )

    # Validate monotonicity
    if time.size > 1:
        time_diffs = np.diff(time)
        if np.any(time_diffs < 0):
            raise ValueError('TIME must be sorted and increasing.')

    # Ref: realized_threshold_variance.m:108 — cast to float64 (protect ints)
    time = time.astype(np.float64, copy=False)

    # --- time_type ---
    time_type_lower = time_type.lower()
    if time_type_lower not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # --- sampling_type ---
    sampling_type_lower = sampling_type.lower()
    _valid_sampling = (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    )
    if sampling_type_lower not in _valid_sampling:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # --- powers ---
    # Ref: realized_threshold_multipower_variation.m — the MATLAB `gamma`
    # parameter is declared but never used in the body.  Accept it for API
    # compatibility; issue a diagnostic warning if a non-None value is
    # supplied, since the current implementation does not compute
    # multipower products.
    if powers is not None:
        powers = np.asarray(powers, dtype=np.float64).ravel()
        # The MATLAB source never uses the gamma parameter.  Preserve
        # this behaviour: accept powers but do not use them.

    # --- options ---
    if options is None:
        options = {}

    # Extract threshold constant c (MATLAB: thresholdScale, default 3)
    # Ref: realized_threshold_variance.m:147-152 — default 3, must be > 0
    c = float(options.get('thresholdConstant', 3.0))
    if c <= 0.0:
        raise ValueError(
            "options['thresholdConstant'] must be a positive scalar."
        )

    # Extract kernel radius L (default 25)
    # Ref: realized_threshold_multipower_variation.m:22
    L = int(options.get('kernelRadius', 25))
    if L < 1:
        raise ValueError(
            "options['kernelRadius'] must be a positive integer."
        )

    # Maximum iterations for convergence loop (safety limit)
    max_iterations = int(options.get('maxIterations', 100))

    # ==================================================================
    # Compute Log Prices and Filter
    # ==================================================================

    # Ref: realized_threshold_multipower_variation.m:7
    log_price = np.log(price)

    # Ref: realized_threshold_multipower_variation.m:9
    # Call realized_price_filter on log prices.  The MATLAB code calls this
    # function but the result is overwritten on line 13.  We call it here
    # to faithfully reproduce MATLAB behaviour (including any side effects).
    _filtered_log_price = realized_price_filter(
        log_price, time, time_type_lower, sampling_type_lower,
        sampling_interval
    )[0]

    # Ref: realized_threshold_multipower_variation.m:13
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # filteredLogPrice  = log(price);
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # KNOWN BUG: The MATLAB source overwrites the filtered result with the
    # raw log(price).  This means price filtering has no effect.  We
    # preserve this bug for strict numerical parity.
    filtered_log_price = np.log(price)

    # ==================================================================
    # Compute Returns
    # Ref: realized_threshold_multipower_variation.m:20
    # ==================================================================
    returns = np.diff(filtered_log_price)
    m = returns.shape[0]

    if m < 1:
        # Edge case: not enough data for any returns
        diagnostics: dict = {
            'localVar': np.array([], dtype=np.float64),
            'returns': np.array([], dtype=np.float64),
        }
        return 0.0, np.nan, 0.0, np.nan, diagnostics

    # ==================================================================
    # Gaussian Kernel Setup
    # Ref: realized_threshold_multipower_variation.m:22-28
    # ==================================================================

    # Ref: m:22 — L = 25 (default, now from options)
    # Ref: m:23 — V = inf*ones(size(returns))
    V = np.full(m, np.inf, dtype=np.float64)

    # Ref: m:24 — K = -L:L (creates 2L+1 element vector)
    K_positions = np.arange(-L, L + 1, dtype=np.float64)

    # Ref: m:25 — Gaussian kernel weights
    # K = 1/sqrt(2*pi) * exp(-(K/L).^2 / 2)
    K_weights = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(
        -(K_positions / float(L)) ** 2 / 2.0
    )

    # Ref: m:26 — K(L+(-1:1)) = 0
    # MATLAB L+[-1, 0, 1] = [L-1, L, L+1] (1-based indices)
    # Python: K_weights has 2L+1 elements indexed 0..2L.
    #   Position -2 → index L-2 (0-based)
    #   Position -1 → index L-1 (0-based)
    #   Position  0 → index L   (0-based)
    # MATLAB 1-based [L-1, L, L+1] → Python 0-based [L-2, L-1, L]
    # Ref: realized_threshold_multipower_variation.m:26 — zero center 3 weights
    K_weights[L - 2: L + 1] = 0.0

    # Ref: m:28 — returns2 = returns.^2
    returns2 = returns ** 2

    # ==================================================================
    # Iterative Local Variance Estimation
    # Ref: realized_threshold_multipower_variation.m:30-44
    # ==================================================================

    # Ref: m:30 — finished = false
    ind = np.zeros(m, dtype=np.bool_)

    for _iteration in range(max_iterations):
        # Ref: m:32 — ind = returns2 < (c^2 .* V)
        ind = returns2 < (c ** 2 * V)
        V_old = V.copy()

        # Ref: m:34-40 — for i=1:m (MATLAB 1-based → Python 0-based)
        for i in range(m):
            # Ref: m:35 — pl = i-L:i+L (MATLAB 1-based indices)
            # In Python 0-based: window centered at i
            pl = np.arange(i - L, i + L + 1)

            # Ref: m:36 — valid = pl>1 & pl<=m (MATLAB 1-based)
            # MATLAB pl>1 means index >= 2 (1-based) = index >= 1 (0-based)
            # MATLAB pl<=m means index <= m (1-based) = index <= m-1 (0-based)
            # Python: pl > 0 & pl <= m-1
            valid_mask = (pl > 0) & (pl <= m - 1)

            if not np.any(valid_mask):
                continue

            # Select valid neighbour indices and their kernel weights
            valid_pl = pl[valid_mask]
            temp_ind = ind[valid_pl].astype(np.float64)
            w = K_weights[valid_mask]

            # Ref: m:39 — V(i) = w*(returns2(pl(valid)).*tempInd) / (w*tempInd)
            # Numerator: dot product of kernel weights with (squared returns * indicator)
            # Denominator: dot product of kernel weights with indicator
            denominator = np.dot(w, temp_ind)

            if denominator > 0.0:
                numerator = np.dot(w, returns2[valid_pl] * temp_ind)
                V[i] = numerator / denominator
            # If denominator is 0 (all neighbours are jumps), V[i] retains
            # its previous value.  MATLAB division by zero yields Inf, which
            # would be overwritten in the next iteration if any neighbours
            # become non-jumps.

        # Ref: m:41-43 — convergence check
        if np.all(V == V_old):
            break
    else:
        # Convergence not achieved within max_iterations
        warnings.warn(
            f'Local variance estimation did not converge within '
            f'{max_iterations} iterations.  Results may be inaccurate.',
            stacklevel=2,
        )

    # ==================================================================
    # Bias Correction — Expected Value for Jump Returns
    # Ref: realized_threshold_multipower_variation.m:46
    # ==================================================================
    #
    # MATLAB expression:
    #   expectedValue = 1./(2*normcdf(-c)*sqrt(pi)) * (2/c^2)
    #                   * gamma(3/2) .* gammainc(c^2/2, 3/2, 'upper')
    #                   * c^2 * V;
    #
    # Translation notes:
    #   normcdf(-c)                    → norm.cdf(-c)
    #     (Ref: duplication/normcdf.m → scipy.stats.norm per AAP §0.6.2)
    #   gamma(3/2)                     → special.gamma(1.5)
    #   gammainc(c^2/2, 3/2, 'upper')  → special.gammaincc(1.5, c**2/2)
    #     (MATLAB gammainc(x, a, 'upper') → scipy gammaincc(a, x) — arg swap)

    norm_cdf_neg_c = norm.cdf(-c)
    gamma_three_halves = special.gamma(1.5)
    # Ref: MATLAB gammainc(c^2/2, 3/2, 'upper') with arg order swap
    gammainc_upper = special.gammaincc(1.5, c ** 2 / 2.0)

    # Assemble the expected value as a per-return vector
    # Ref: realized_threshold_multipower_variation.m:46
    expected_value = (
        (1.0 / (2.0 * norm_cdf_neg_c * np.sqrt(np.pi)))
        * (2.0 / c ** 2)
        * gamma_three_halves
        * gammainc_upper
        * c ** 2
        * V
    )

    # ==================================================================
    # Compute Threshold Multipower Variation
    # Ref: realized_threshold_multipower_variation.m:49
    # ==================================================================
    #
    # MATLAB: rv = returns(ind)'*returns(ind) + sum(expectedValue(~ind))
    #
    # ind = True for non-jump returns
    # returns(ind)'*returns(ind) = sum of squared non-jump returns
    # expectedValue(~ind) = expected values for jump returns
    #
    # The MATLAB rv is the DEBIASED version (includes expected-value
    # correction).  We also compute the raw (non-debiased) version.

    non_jump_mask = ind
    non_jump_returns = returns[non_jump_mask]

    # Raw threshold multipower variation (no bias correction)
    tmpv = float(np.dot(non_jump_returns, non_jump_returns))

    # Debiased version (with expected-value correction for jumps)
    jump_expected_sum = float(np.sum(expected_value[~non_jump_mask]))
    tmpv_debiased = tmpv + jump_expected_sum

    # ==================================================================
    # Subsampled Outputs — MATLAB Bug: rvSS Never Assigned
    # Ref: realized_threshold_multipower_variation.m — rvSS declared in
    # function signature but never assigned in the body.
    # ==================================================================
    tmpv_ss = np.nan
    tmpv_ss_debiased = np.nan

    # ==================================================================
    # Diagnostics
    # ==================================================================
    diagnostics = {
        'localVar': V.copy(),
        'returns': returns.copy(),
    }

    return tmpv, tmpv_ss, tmpv_debiased, tmpv_ss_debiased, diagnostics
