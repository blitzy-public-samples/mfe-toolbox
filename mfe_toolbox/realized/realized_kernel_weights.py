"""
Kernel weight computation for realized kernels.

Computes the weight vector for a specified kernel type and bandwidth,
supporting both flat-top and non-flat-top kernels as defined in
realized_options.

Migrated from: realized/realized_kernel_weights.m
Original Author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008

References
----------
Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A. and Shephard, N. (2008a, 2008b)
for details about realized kernels and their properties.
"""

import numpy as np


# Ref: realized_kernel_weights.m:32-34 — List of flat-top kernel types
# Flat-top kernels have k(0)=1 and k'(0)=0 (zero derivative at origin)
FLAT_TOP_KERNEL_LIST: list[str] = [
    'bartlett', 'twoscale', '2ndorder', 'epanechnikov',
    'cubic', 'multiscale', '5thorder', '6thorder', '7thorder', '8thorder',
    'parzen', 'th1', 'th2', 'th5', 'th16',
]

# Ref: realized_kernel_weights.m:37 — List of non-flat-top kernel types
# Non-flat-top kernels may have k'(0)!=0 and use different normalization
NON_FLAT_TOP_KERNEL_LIST: list[str] = [
    'nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls',
]

# Ref: realized_kernel_weights.m:40 — Combined kernel list for validation
KERNEL_LIST: list[str] = FLAT_TOP_KERNEL_LIST + NON_FLAT_TOP_KERNEL_LIST


def realized_kernel_weights(options: dict) -> np.ndarray:
    """
    Compute weights for realized kernels.

    Parameters
    ----------
    options : dict
        A realized kernel options dictionary containing at minimum:

        - ``'kernel'`` : str
            Kernel type. Must be one of the supported flat-top or non-flat-top
            kernel names listed in ``KERNEL_LIST``.
        - ``'bandwidth'`` : float
            Non-negative scalar bandwidth value. For flat-top kernels this is
            rounded to the nearest integer to determine the number of lags.
            For non-flat-top kernels the fractional bandwidth is used directly.

    Returns
    -------
    np.ndarray
        1-D array of kernel weights with dtype float64. The length H depends
        on the bandwidth and kernel type:

        - For flat-top kernels: ``H = round(bandwidth)`` weights for
          normalized lags ``0/H, 1/H, ..., (H-1)/H``.
        - For non-flat-top kernels: H depends on the truncation rule
          specific to each kernel (e.g., ``30*bandwidth`` for QS/Fejér).
        - Returns an empty array (shape ``(0,)``) when bandwidth is zero
          or rounds to zero.

    Raises
    ------
    ValueError
        If ``'kernel'`` is missing from *options* or is not a recognized type.
        If ``'bandwidth'`` is missing from *options* or is negative.

    Notes
    -----
    This is a helper function for ``realized_kernel``. All kernel formulas
    follow Barndorff-Nielsen, Hansen, Lunde and Shephard (2008a, 2008b).

    **Supported flat-top kernels** (first lag at ``x = 0``, weight = 1):

    +-----------------------+----------------------------------------------+
    | Name(s)               | Formula ``k(x)``                             |
    +=======================+==============================================+
    | bartlett / twoscale    | ``1 - x``                                    |
    +-----------------------+----------------------------------------------+
    | 2ndorder              | ``1 - 2x + x²``                              |
    +-----------------------+----------------------------------------------+
    | epanechnikov          | ``1 - x²``                                   |
    +-----------------------+----------------------------------------------+
    | cubic / multiscale    | ``1 - 2x² + 2x³``                            |
    +-----------------------+----------------------------------------------+
    | 5thorder              | ``1 - 10x³ + 15x⁴ - 6x⁵``                   |
    +-----------------------+----------------------------------------------+
    | 6thorder              | ``1 - 15x⁴ + 24x⁵ - 10x⁶``                  |
    +-----------------------+----------------------------------------------+
    | 7thorder              | ``1 - 21x⁵ + 35x⁶ - 15x⁷``                  |
    +-----------------------+----------------------------------------------+
    | 8thorder              | ``1 - 28x⁶ + 48x⁷ - 21x⁸``                  |
    +-----------------------+----------------------------------------------+
    | parzen                | piecewise: see source                        |
    +-----------------------+----------------------------------------------+
    | th1, th2, th5, th16   | ``sin(π/2 · (1-x)^p)²``  (p = 1,2,5,16)    |
    +-----------------------+----------------------------------------------+

    **Supported non-flat-top kernels** (normalized by ``H + 1``):

    +------------------+---------------------------------------------------+
    | Name             | Formula / truncation                              |
    +==================+===================================================+
    | nonflatparzen    | Parzen on ``x = j/(H+1)``, j=1..⌊H⌋             |
    +------------------+---------------------------------------------------+
    | qs               | Quadratic Spectral, truncation 30H                |
    +------------------+---------------------------------------------------+
    | fejer            | Fejér (sinc²), truncation 30H                     |
    +------------------+---------------------------------------------------+
    | thinf            | Tukey-Hanning ∞, truncation 4H                    |
    +------------------+---------------------------------------------------+
    | bnhls            | BNHLS ``(1+x)exp(-x)``, truncation 10H           |
    +------------------+---------------------------------------------------+

    Examples
    --------
    >>> import numpy as np
    >>> opts = {'kernel': 'bartlett', 'bandwidth': 5}
    >>> w = realized_kernel_weights(opts)
    >>> len(w)
    5
    >>> np.allclose(w, [1.0, 0.8, 0.6, 0.4, 0.2])
    True
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: realized_kernel_weights.m:27-48
    # ------------------------------------------------------------------
    if 'kernel' not in options or options['kernel'] not in KERNEL_LIST:
        raise ValueError(
            'KERNEL must be a field of OPTIONS and one of the listed types.'
        )

    if 'bandwidth' not in options or options['bandwidth'] < 0:
        raise ValueError(
            'BANDWIDTH must be a field of OPTIONS and a non-negative scalar.'
        )

    # Ref: realized_kernel_weights.m:54-55 — Extract relevant fields
    kernel: str = options['kernel']
    bandwidth: float = float(options['bandwidth'])

    # ------------------------------------------------------------------
    # Weight computation
    # Ref: realized_kernel_weights.m:58-123
    # ------------------------------------------------------------------
    if bandwidth <= 0.0:
        # Ref: realized_kernel_weights.m:123 — Zero bandwidth → empty weights
        return np.array([], dtype=np.float64)

    # Determine the normalized grid x based on kernel family
    if kernel in FLAT_TOP_KERNEL_LIST:
        # Ref: realized_kernel_weights.m:61-63
        # Flat-top: H = round(bandwidth); x = (0, 1, ..., H-1) / H
        H_int: int = int(np.round(bandwidth))
        if H_int <= 0:
            # Edge case: bandwidth rounds to 0 (e.g. bandwidth=0.4)
            return np.array([], dtype=np.float64)
        # Ref: realized_kernel_weights.m:62 — MATLAB 1-indexed (1:H)',
        # then shifted: x = ((1:H)' - 1) / H  →  [0/H, 1/H, ..., (H-1)/H]
        x = np.arange(0, H_int, dtype=np.float64) / H_int
    else:
        # Ref: realized_kernel_weights.m:65-66
        # Non-flat-top kernels keep fractional bandwidth; x computed per-case
        H_float: float = bandwidth

    # ------------------------------------------------------------------
    # Kernel-specific weight formulas (case-insensitive lookup)
    # Ref: realized_kernel_weights.m:68-121
    # ------------------------------------------------------------------
    kernel_lower: str = kernel.lower()

    # === Flat-top kernels ===

    if kernel_lower in ('bartlett', 'twoscale'):
        # Ref: realized_kernel_weights.m:70 — Bartlett / Two-scale kernel
        weights = 1.0 - x

    elif kernel_lower == '2ndorder':
        # Ref: realized_kernel_weights.m:72 — Second-order (quadratic) kernel
        weights = 1.0 - 2.0 * x + x ** 2

    elif kernel_lower == 'epanechnikov':
        # Ref: realized_kernel_weights.m:74 — Epanechnikov kernel
        weights = 1.0 - x ** 2

    elif kernel_lower in ('cubic', 'multiscale'):
        # Ref: realized_kernel_weights.m:76 — Cubic / Multiscale kernel
        weights = 1.0 - 2.0 * x ** 2 + 2.0 * x ** 3

    elif kernel_lower == '5thorder':
        # Ref: realized_kernel_weights.m:78 — Fifth-order kernel
        weights = 1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5

    elif kernel_lower == '6thorder':
        # Ref: realized_kernel_weights.m:80 — Sixth-order kernel
        weights = 1.0 - 15.0 * x ** 4 + 24.0 * x ** 5 - 10.0 * x ** 6

    elif kernel_lower == '7thorder':
        # Ref: realized_kernel_weights.m:82 — Seventh-order kernel
        weights = 1.0 - 21.0 * x ** 5 + 35.0 * x ** 6 - 15.0 * x ** 7

    elif kernel_lower == '8thorder':
        # Ref: realized_kernel_weights.m:84 — Eighth-order kernel
        weights = 1.0 - 28.0 * x ** 6 + 48.0 * x ** 7 - 21.0 * x ** 8

    elif kernel_lower == 'parzen':
        # Ref: realized_kernel_weights.m:86 — Parzen kernel (flat-top variant)
        # Piecewise definition:
        #   k(x) = 1 - 6x² + 6x³       for 0 ≤ x ≤ 1/2
        #   k(x) = 2(1 - x)³            for 1/2 < x < 1
        mask_low = (x >= 0.0) & (x <= 0.5)
        mask_high = (x > 0.5) & (x < 1.0)
        weights = (
            (1.0 - 6.0 * x ** 2 + 6.0 * x ** 3) * mask_low
            + 2.0 * (1.0 - x) ** 3 * mask_high
        )

    elif kernel_lower == 'th1':
        # Ref: realized_kernel_weights.m:88 — Tukey-Hanning kernel, power 1
        weights = np.sin(np.pi / 2.0 * (1.0 - x)) ** 2

    elif kernel_lower == 'th2':
        # Ref: realized_kernel_weights.m:90 — Tukey-Hanning kernel, power 2
        weights = np.sin(np.pi / 2.0 * (1.0 - x) ** 2) ** 2

    elif kernel_lower == 'th5':
        # Ref: realized_kernel_weights.m:92 — Tukey-Hanning kernel, power 5
        weights = np.sin(np.pi / 2.0 * (1.0 - x) ** 5) ** 2

    elif kernel_lower == 'th16':
        # Ref: realized_kernel_weights.m:94 — Tukey-Hanning kernel, power 16
        weights = np.sin(np.pi / 2.0 * (1.0 - x) ** 16) ** 2

    # === Non-flat-top kernels ===
    # Each non-flat-top kernel computes its own x grid internally.

    elif kernel_lower == 'nonflatparzen':
        # Ref: realized_kernel_weights.m:96-98 — Non-flat-top Parzen kernel
        # x = j / (H + 1) for j = 1, 2, ..., floor(H)
        n_lags = int(np.floor(H_float))
        if n_lags < 1:
            return np.array([], dtype=np.float64)
        x = np.arange(1, n_lags + 1, dtype=np.float64) / (H_float + 1.0)
        mask_low = (x >= 0.0) & (x <= 0.5)
        mask_high = (x > 0.5) & (x < 1.0)
        weights = (
            (1.0 - 6.0 * x ** 2 + 6.0 * x ** 3) * mask_low
            + 2.0 * (1.0 - x) ** 3 * mask_high
        )

    elif kernel_lower == 'qs':
        # Ref: realized_kernel_weights.m:100-103 — Quadratic Spectral kernel
        # Truncation at 30 * H lags; x = j / (H + 1)
        max_lag = int(np.floor(30.0 * H_float))
        if max_lag < 1:
            return np.array([], dtype=np.float64)
        # Ref: realized_kernel_weights.m:101 — unique(round(...)) preserves
        # integer sequence; kept for fidelity to MATLAB source
        raw_lags = np.arange(1, max_lag + 1, dtype=np.float64)
        x = np.unique(np.round(raw_lags)) / (H_float + 1.0)
        # Ref: realized_kernel_weights.m:103 — QS formula:
        # k(x) = 3/x² · (sin(x)/x − cos(x))
        weights = 3.0 / (x ** 2) * (np.sin(x) / x - np.cos(x))

    elif kernel_lower == 'fejer':
        # Ref: realized_kernel_weights.m:105-108 — Fejér kernel (sinc squared)
        # Truncation at 30 * H lags
        max_lag = int(np.floor(30.0 * H_float))
        if max_lag < 1:
            return np.array([], dtype=np.float64)
        raw_lags = np.arange(1, max_lag + 1, dtype=np.float64)
        x = np.unique(np.round(raw_lags)) / (H_float + 1.0)
        # Ref: realized_kernel_weights.m:108 — Fejér formula: k(x) = (sin(x)/x)²
        weights = (np.sin(x) / x) ** 2

    elif kernel_lower == 'thinf':
        # Ref: realized_kernel_weights.m:110-113 — Tukey-Hanning infinite kernel
        # Truncation at 4 * H lags
        max_lag = int(np.floor(4.0 * H_float))
        if max_lag < 1:
            return np.array([], dtype=np.float64)
        raw_lags = np.arange(1, max_lag + 1, dtype=np.float64)
        x = np.unique(np.round(raw_lags)) / (H_float + 1.0)
        # Ref: realized_kernel_weights.m:113 — THinf formula:
        # k(x) = sin(π/2 · exp(−x))²
        weights = np.sin((np.pi / 2.0) * np.exp(-x)) ** 2

    elif kernel_lower == 'bnhls':
        # Ref: realized_kernel_weights.m:115-118 — BNHLS kernel
        # Barndorff-Nielsen, Hansen, Lunde and Shephard kernel
        # Truncation at 10 * H lags
        max_lag = int(np.floor(10.0 * H_float))
        if max_lag < 1:
            return np.array([], dtype=np.float64)
        raw_lags = np.arange(1, max_lag + 1, dtype=np.float64)
        x = np.unique(np.round(raw_lags)) / (H_float + 1.0)
        # Ref: realized_kernel_weights.m:118 — BNHLS formula: k(x) = (1+x)exp(−x)
        weights = (1.0 + x) * np.exp(-x)

    else:
        # Ref: realized_kernel_weights.m:120 — Should not be reachable
        # due to initial validation, but included for defensive programming
        raise ValueError('OPTIONS.KERNEL must be one of the listed types.')

    return weights
