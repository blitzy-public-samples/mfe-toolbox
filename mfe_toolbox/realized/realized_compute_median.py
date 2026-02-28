"""
Median price computation for tick data deduplication.

Computes the median price at each unique timestamp for a vector of prices
which may have multiple observations at the same timestamp. This is a helper
function consumed by many other Realized toolkit modules.

Migrated from realized/realized_compute_median.m (MFE Toolbox, Version 4.0)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 2, Date: 6/12/2011

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""
from __future__ import annotations

import numpy as np


def realized_compute_median(
    price: np.ndarray,
    time: np.ndarray,
    volume: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the median price at each unique timestamp for tick data deduplication.

    Most functions in the Realized toolkit expect timestamps to be unique.
    Median price is a reasonable and fairly robust (to noise) method of
    computing a unique price for each timestamp.

    Parameters
    ----------
    price : np.ndarray
        m-element 1-D array (or column vector) of high-frequency prices.
    time : np.ndarray
        m-element 1-D array of timestamps corresponding to ``price``, sorted
        in ascending order.  Timestamps may contain duplicates.
    volume : np.ndarray or None, optional
        m-element 1-D array of transaction volumes or bid/ask size.
        If ``None`` (default), ``total_vol`` in the output will be all zeros.

    Returns
    -------
    median_price : np.ndarray
        n-element 1-D array of median prices where *n* is the number of
        unique elements of ``time``.
    median_time : np.ndarray
        n-element 1-D array of unique timestamps corresponding to
        ``median_price``.
    total_vol : np.ndarray
        n-element 1-D array capturing the total volume at each unique
        timestamp.
    n_obs : np.ndarray
        n-element 1-D integer array indicating the number of raw price
        observations at each unique timestamp.

    Raises
    ------
    ValueError
        If ``price`` is not a 1-D vector, ``time`` is not sorted and
        increasing, or array lengths are mismatched.

    Notes
    -----
    The algorithm groups observations by timestamp, then computes the median
    price for each group using an efficient batched approach that processes
    all groups of the same size simultaneously:

    * Groups of size 1 — the single price is the median.
    * Groups of size 2 — the arithmetic mean of the two prices.
    * Groups of size > 2 — the standard median (middle element for odd,
      mean of two middle elements for even).

    References
    ----------
    Migrated from ``realized_compute_median.m``, MFE Toolbox Version 4.0,
    Kevin Sheppard, University of Oxford.
    """
    # ------------------------------------------------------------------ #
    # Input Checking  (Ref: realized_compute_median.m:30-67)
    # ------------------------------------------------------------------ #

    # --- price validation (Ref: realized_compute_median.m:36-42) ---
    price = np.asarray(price, dtype=np.float64)
    # Ref: realized_compute_median.m:36 — auto-transpose row vector
    if price.ndim == 2 and price.shape[1] > price.shape[0]:
        price = price.T
    # Flatten to 1-D (handles (m,1) column vectors)
    price = price.ravel()
    # Ref: realized_compute_median.m:39-40 — must be 1-D after processing
    if price.ndim != 1:
        raise ValueError("PRICE must be a m by 1 vector.")
    m: int = price.shape[0]

    # --- time validation (Ref: realized_compute_median.m:44-53) ---
    time = np.asarray(time, dtype=np.float64)
    # Ref: realized_compute_median.m:44 — auto-transpose row vector
    if time.ndim == 2 and time.shape[1] > time.shape[0]:
        time = time.T
    time = time.ravel()
    # Ref: realized_compute_median.m:47 — must be sorted ascending
    if np.any(np.diff(time) < 0):
        raise ValueError("TIME must be sorted and increasing")
    # Ref: realized_compute_median.m:50-51 — shape and length check
    if time.ndim != 1 or len(time) != m:
        raise ValueError("TIME must be a m by 1 vector.")
    # Ref: realized_compute_median.m:53,70 — ensure float64
    time = np.asarray(time, dtype=np.float64)

    # --- volume validation (Ref: realized_compute_median.m:55-64) ---
    if volume is not None:
        volume = np.asarray(volume, dtype=np.float64)
        # Ref: realized_compute_median.m:56 — auto-transpose row vector
        if volume.ndim == 2 and volume.shape[1] > volume.shape[0]:
            volume = volume.T
        volume = volume.ravel()
        # Ref: realized_compute_median.m:59-60 — shape and length check
        if volume.ndim != 1 or len(volume) != m:
            raise ValueError("VOLUME must be a m by 1 vector.")
    else:
        # Ref: realized_compute_median.m:62-63 — default to zeros
        volume = np.zeros(m, dtype=np.float64)

    # Ref: realized_compute_median.m:70-71 — ensure double precision
    volume = np.asarray(volume, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Core Algorithm  (Ref: realized_compute_median.m:73-107)
    # ------------------------------------------------------------------ #

    # Ref: realized_compute_median.m:74 — find indices where time value changes
    # MATLAB: pl = find(diff(time)) returns 1-indexed positions where diff != 0
    # Python: np.where returns 0-indexed positions
    change_idx = np.where(np.diff(time) != 0)[0]

    # Ref: realized_compute_median.m:75 — build boundary array
    # MATLAB: pl = [1; pl+1; length(time)+1]  (1-indexed start of each group + past-end)
    # Python: [0, change_idx+1, len(time)]    (0-indexed start of each group + past-end)
    pl = np.concatenate(([0], change_idx + 1, [len(time)]))

    # Ref: realized_compute_median.m:78 — number of boundary markers
    n_boundaries: int = len(pl)

    # Number of unique timestamps (groups)
    n_groups: int = n_boundaries - 1

    # Ref: realized_compute_median.m:81-84 — pre-allocate output arrays
    median_price = np.zeros(n_groups, dtype=np.float64)
    # Ref: realized_compute_median.m:82 — time at start of each group
    median_time = time[pl[:n_groups]]
    total_vol = np.zeros(n_groups, dtype=np.float64)
    n_obs = np.zeros(n_groups, dtype=np.int64)

    # Ref: realized_compute_median.m:87 — compute group sizes and find unique sizes
    # MATLAB: [sizes, ~, ind] = unique(diff(pl))
    group_sizes = np.diff(pl)
    unique_sizes, inv_indices = np.unique(group_sizes, return_inverse=True)

    # Ref: realized_compute_median.m:88-107 — loop over unique group sizes
    for i in range(len(unique_sizes)):
        s = int(unique_sizes[i])  # current group size
        # Ref: realized_compute_median.m:89 — find all groups with this size
        # MATLAB: j = find(ind == i)  (1-indexed)
        # Python: np.where gives 0-indexed group indices
        j = np.where(inv_indices == i)[0]

        # Ref: realized_compute_median.m:90 — build index matrix for all groups
        # MATLAB: loc = bsxfun(@plus, pl(j), 0:sizes(i)-1)'  → sizes(i) x len(j)
        # Python: rows are groups, columns are within-group offsets → (len(j), s)
        loc = pl[j][:, np.newaxis] + np.arange(s)

        # Ref: realized_compute_median.m:91-103 — compute median price per group
        if s == 1:
            # Ref: realized_compute_median.m:93 — single observation: price IS the median
            median_price[j] = price[pl[j]]
        elif s == 2:
            # Ref: realized_compute_median.m:96 — two observations: arithmetic mean
            median_price[j] = (price[pl[j]] + price[pl[j] + 1]) / 2.0
        else:
            # Ref: realized_compute_median.m:98 — sort prices within each group
            # MATLAB sorts columns; Python sorts along axis=1 (within each row = group)
            sorted_prices = np.sort(price[loc], axis=1)
            if s % 2 == 0:
                # Ref: realized_compute_median.m:100 — even group size: mean of two middle
                # MATLAB 1-indexed: sortedPrice([sizes(i)/2; sizes(i)/2+1], :)
                # Python 0-indexed: columns s//2 - 1 and s//2
                median_price[j] = np.mean(
                    sorted_prices[:, [s // 2 - 1, s // 2]], axis=1
                )
            else:
                # Ref: realized_compute_median.m:102 — odd group size: middle element
                # MATLAB 1-indexed: ceil(sizes(i)/2) → Python 0-indexed: s // 2
                median_price[j] = sorted_prices[:, s // 2]

        # Ref: realized_compute_median.m:105 — total volume for ALL group sizes
        # MATLAB: totalVol(j) = sum(volume(loc))'
        # Python: sum along axis=1 (within each group row)
        total_vol[j] = np.sum(volume[loc], axis=1)

        # Ref: realized_compute_median.m:106 — observation count per group
        n_obs[j] = s

    return median_price, median_time, total_vol, n_obs
