"""
Core realized kernel computation.

Computes the value of a realized kernel given a set of returns and kernel
weights. This is a helper function for the main realized_kernel estimator,
implementing the weighted sum of realized autocovariances at various lags
with support for both 'jitter' and 'stagger' endpoint treatments.

Migrated from: realized/realized_kernel_core.m
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 5/1/2008

See Also
--------
realized_kernel : Main realized kernel estimator orchestrator.
realized_options : Default options factory for realized estimators.
realized_kernel_weights : Kernel weight function lookup.
realized_kernel_bandwidth : Optimal bandwidth selection.
"""

import warnings

import numpy as np


def realized_kernel_core(
    returns: np.ndarray,
    weights: np.ndarray,
    options: dict,
) -> float:
    """
    Compute the realized kernel value from returns and kernel weights.

    Implements the Barndorff-Nielsen, Hansen, Lunde, Shephard (BNHLS) realized
    kernel estimator core computation as a weighted sum of realized
    autocovariances at various lags. Supports both 'jitter' (using all returns)
    and 'stagger' (using central returns only) endpoint treatment strategies.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of m returns (typically log returns of filtered prices).
    weights : np.ndarray
        1-D array of H kernel weights corresponding to lags 1, 2, ..., H.
        H should be much smaller than m.
    options : dict
        A realized kernel options dictionary. Relevant keys include:

        - ``'endTreatment'`` : str
            Either ``'jitter'`` or ``'stagger'``. Controls how endpoints are
            handled when computing autocovariances.
        - ``'maxBandwidthPerc'`` : float or None
            Maximum bandwidth as a percentage of ``filteredN``. If the weight
            vector exceeds this fraction of available data, it is truncated.
        - ``'maxBandwidth'`` : int or None
            Absolute maximum bandwidth. If the weight vector exceeds this
            value, it is truncated.
        - ``'filteredN'`` : int
            Number of filtered observations, used with ``maxBandwidthPerc``.

    Returns
    -------
    float
        The realized kernel estimate (a scalar).

    Raises
    ------
    ValueError
        If ``returns`` or ``weights`` cannot be coerced to a 1-D vector,
        or if any required input is ``None``.

    Notes
    -----
    When ``endTreatment`` is ``'jitter'``, the autocovariances are computed
    using *all* returns, so that ``gamma_minus(h) = returns[:m-h]' @
    returns[h:]`` and ``gamma0 = returns' @ returns``. Both ``gamma_minus``
    and ``gamma_plus`` are identical in this case by construction.

    When ``endTreatment`` is *not* ``'jitter'`` (i.e. ``'stagger'``), a
    *central* subset of returns is used: ``returnsBase = returns[H:m-H]``,
    and the lead/lag autocovariances are computed relative to this base,
    avoiding potential endpoint bias.

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N.
    (2008). Designing realized kernels to measure the ex-post variation
    of equity prices in the presence of noise. *Econometrica*, 76(6),
    1481-1536.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> r = rng.standard_normal(100) * 0.01
    >>> w = np.array([0.9, 0.7, 0.4, 0.1])
    >>> opts = {'endTreatment': 'jitter'}
    >>> rk = realized_kernel_core(r, w, opts)
    """
    # ------------------------------------------------------------------
    # Input Checking
    # Ref: realized_kernel_core.m:29-31 — Three inputs required
    # ------------------------------------------------------------------
    if returns is None or weights is None or options is None:
        raise ValueError("Three inputs required.")

    # ------------------------------------------------------------------
    # Coerce returns to 1-D float64 array
    # Ref: realized_kernel_core.m:32-37 — transpose row vectors; reject
    # multi-column matrices
    # ------------------------------------------------------------------
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim == 2:
        # Ref: realized_kernel_core.m:32-34 — if more columns than rows,
        # transpose to column
        if returns.shape[1] > returns.shape[0]:
            returns = returns.T
        # Ref: realized_kernel_core.m:35-37
        if returns.shape[1] > 1:
            raise ValueError("RETURNS must be a m by 1 vector.")
        returns = returns.ravel()
    elif returns.ndim == 0:
        returns = returns.reshape(1)
    elif returns.ndim > 2:
        raise ValueError("RETURNS must be a m by 1 vector.")

    # ------------------------------------------------------------------
    # Coerce weights to 1-D float64 array
    # Ref: realized_kernel_core.m:38-43
    # ------------------------------------------------------------------
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 2:
        if weights.shape[1] > weights.shape[0]:
            weights = weights.T
        if weights.shape[1] > 1:
            raise ValueError("WEIGHTS must be a H by 1 vector.")
        weights = weights.ravel()
    elif weights.ndim == 0:
        weights = weights.reshape(1)
    elif weights.ndim > 2:
        raise ValueError("WEIGHTS must be a H by 1 vector.")

    # ------------------------------------------------------------------
    # Get the number of returns
    # Ref: realized_kernel_core.m:46
    # ------------------------------------------------------------------
    m: int = len(returns)

    # ------------------------------------------------------------------
    # Weight truncation logic
    # Ref: realized_kernel_core.m:47-67
    # ------------------------------------------------------------------
    if len(weights) >= m:
        # Ref: realized_kernel_core.m:48-49 — truncate weights to m-1
        warnings.warn(
            "The length of WEIGHTS is longer than the length of RETURNS.\n"
            "  The weights are being truncated at N-1 where N is the "
            "number of returns",
            stacklevel=2,
        )
        weights = weights[: m - 1]
    else:
        # Ref: realized_kernel_core.m:50-67 — maxBandwidthPerc /
        # maxBandwidth guard
        max_bw_perc = options.get("maxBandwidthPerc", None)
        max_bw = options.get("maxBandwidth", None)

        if max_bw_perc is not None or max_bw is not None:
            warning_string: str | None = None
            filtered_n = options.get("filteredN", None)

            # Ref: realized_kernel_core.m:53-57 — percentage-based cap
            if (
                max_bw_perc is not None
                and filtered_n is not None
                and len(weights) > (max_bw_perc * filtered_n)
            ):
                max_bw = int(round(max_bw_perc * filtered_n))
                options["maxBandwidth"] = max_bw
                warning_string = (
                    f"The estimated bandwidth requires a lag length larger "
                    f"than {100 * max_bw_perc} % of the available data.  "
                    f"Bandwidth has been truncated to {max_bw}."
                )
            # Ref: realized_kernel_core.m:58-62 — absolute cap
            elif max_bw is not None and len(weights) > max_bw:
                warning_string = (
                    f"The estimated bandwidth requires a lag length larger "
                    f"than {max_bw}.  Bandwidth has been truncated to "
                    f"{max_bw}."
                )

            # Ref: realized_kernel_core.m:64-67 — apply truncation
            if warning_string is not None:
                warnings.warn(warning_string, stacklevel=2)
                weights = weights[: int(max_bw)]

    # ------------------------------------------------------------------
    # Extract relevant fields from OPTIONS
    # Ref: realized_kernel_core.m:74 — 'jitter' vs 'stagger'
    # ------------------------------------------------------------------
    is_jittered: bool = (
        options.get("endTreatment", "").lower() == "jitter"
    )

    # Ref: realized_kernel_core.m:76 — Get the size of the kernel
    h_len: int = len(weights)

    # ------------------------------------------------------------------
    # Compute autocovariances
    # ------------------------------------------------------------------
    if is_jittered:
        # ==============================================================
        # Jittered endpoint treatment — use ALL returns
        # Ref: realized_kernel_core.m:78-89
        # ==============================================================
        gamma_minus = np.zeros(h_len, dtype=np.float64)
        gamma_plus = np.zeros(h_len, dtype=np.float64)

        for i in range(1, h_len + 1):
            # Ref: realized_kernel_core.m:83 — MATLAB returns(1:m-i)
            # → Python returns[:m-i]  (0-indexed slicing)
            returns_minus = returns[: m - i]
            # Ref: realized_kernel_core.m:84 — MATLAB returns(i+1:m)
            # → Python returns[i:]  (0-indexed slicing)
            returns_plus = returns[i:]
            # Ref: realized_kernel_core.m:85-86 — Both gammaMinus and
            # gammaPlus are set to the same dot product (symmetric by
            # construction in the jitter case).
            dot_val = np.dot(returns_minus, returns_plus)
            gamma_minus[i - 1] = dot_val
            gamma_plus[i - 1] = dot_val

        # Ref: realized_kernel_core.m:89 — gamma0 = returns' * returns
        gamma0: float = float(np.dot(returns, returns))
    else:
        # ==============================================================
        # Stagger endpoint treatment — use CENTRAL returns
        # Ref: realized_kernel_core.m:91-107
        # ==============================================================
        # Ref: realized_kernel_core.m:94 — MATLAB returns(H+1:m-H)
        # → Python returns[H:m-H]  (0-indexed slicing)
        returns_base = returns[h_len: m - h_len]

        gamma_minus = np.zeros(h_len, dtype=np.float64)
        gamma_plus = np.zeros(h_len, dtype=np.float64)

        for i in range(1, h_len + 1):
            # Ref: realized_kernel_core.m:100 — MATLAB
            # returns(H+1-i:m-H-i)  →  Python returns[H-i:m-H-i]
            returns_minus = returns[h_len - i: m - h_len - i]
            # Ref: realized_kernel_core.m:101 — MATLAB
            # returns(H+1+i:m-H+i)  →  Python returns[H+i:m-H+i]
            returns_plus = returns[h_len + i: m - h_len + i]
            # Ref: realized_kernel_core.m:102
            gamma_minus[i - 1] = np.dot(returns_minus, returns_base)
            # Ref: realized_kernel_core.m:103
            gamma_plus[i - 1] = np.dot(returns_base, returns_plus)

        # Ref: realized_kernel_core.m:106 — gamma0 = returnsBase' *
        # returnsBase
        gamma0 = float(np.dot(returns_base, returns_base))

    # ------------------------------------------------------------------
    # Construct the kernel
    # Ref: realized_kernel_core.m:110-115
    # ------------------------------------------------------------------
    if len(weights) > 0:
        # Ref: realized_kernel_core.m:111 — rk = gamma0 +
        # weights'*(gammaMinus+gammaPlus)
        rk: float = gamma0 + float(
            np.dot(weights, gamma_minus + gamma_plus)
        )
    else:
        # Ref: realized_kernel_core.m:113-114 — zero lags used
        warnings.warn("The number of lags used was 0.", stacklevel=2)
        rk = gamma0

    return float(rk)
