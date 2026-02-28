"""
Optimal bandwidth computation for realized kernels.

Migrated from: realized/realized_kernel_bandwidth.m

Implements bandwidth selection following Barndorff-Nielsen, Hansen, Lunde
and Shephard (2008) for use with realized kernel estimators. The optimal
bandwidth is computed using plug-in estimates of noise variance and
integrated quarticity, combined with kernel-specific rate constants and
the effective sample size.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008
"""

import math
import warnings

# ---------------------------------------------------------------------------
# Kernel classification lists
# ---------------------------------------------------------------------------
# Ref: realized_kernel_bandwidth.m:33-35 — Flat-top kernel list
FLAT_TOP_KERNEL_LIST: tuple[str, ...] = (
    'bartlett', 'twoscale', '2ndorder', 'epanechnikov',
    'cubic', 'multiscale', '5thorder', '6thorder', '7thorder', '8thorder',
    'parzen', 'th1', 'th2', 'th5', 'th16',
)

# Ref: realized_kernel_bandwidth.m:38 — Non-flat-top kernel list
NON_FLAT_TOP_KERNEL_LIST: tuple[str, ...] = (
    'nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls',
)

# Ref: realized_kernel_bandwidth.m:41 — Combined kernel list
KERNEL_LIST: tuple[str, ...] = FLAT_TOP_KERNEL_LIST + NON_FLAT_TOP_KERNEL_LIST

# ---------------------------------------------------------------------------
# Kernel-specific optimal bandwidth constant (c*) and kernel type mapping
# ---------------------------------------------------------------------------
# Ref: realized_kernel_bandwidth.m:51-108
#
# Each kernel maps to (c_star, kernel_type) where:
#   kernel_type == 1  →  flat-top higher-order:  bw = c* · ξ^(1/2) · n^(1/2)
#   kernel_type == 2  →  flat-top Bartlett-type:  bw = c* · ξ^(2/3) · n^(2/3)
#   kernel_type == 3  →  non-flat-top:           bw = c* · ξ^(2/5) · n^(3/5)
#
# c* values for flat-top kernels are tabulated constants from
# Barndorff-Nielsen, Hansen, Lunde, and Shephard (2008).
# c* values for non-flat-top kernels are computed from analytical formulas
# given in the MATLAB source.
_KERNEL_CONSTANTS: dict[str, tuple[float, int]] = {
    # --- Flat-top, kernel_type 2 (Bartlett-like) ---
    # Ref: realized_kernel_bandwidth.m:52-53
    'bartlett': (2.28, 2),
    'twoscale': (2.28, 2),
    # Ref: realized_kernel_bandwidth.m:55-56
    '2ndorder': (3.42, 2),
    # Ref: realized_kernel_bandwidth.m:58-59
    'epanechnikov': (2.46, 2),
    # --- Flat-top, kernel_type 1 (higher-order) ---
    # Ref: realized_kernel_bandwidth.m:61-62
    'cubic': (3.68, 1),
    'multiscale': (3.68, 1),
    # Ref: realized_kernel_bandwidth.m:64-65
    '5thorder': (3.70, 1),
    # Ref: realized_kernel_bandwidth.m:67-68
    '6thorder': (3.97, 1),
    # Ref: realized_kernel_bandwidth.m:70-71
    '7thorder': (4.11, 1),
    # Ref: realized_kernel_bandwidth.m:73-74
    '8thorder': (4.31, 1),
    # Ref: realized_kernel_bandwidth.m:76-77
    'parzen': (4.77, 1),
    # Ref: realized_kernel_bandwidth.m:79-80
    'th1': (3.70, 1),
    # Ref: realized_kernel_bandwidth.m:82-83
    'th2': (5.74, 1),
    # Ref: realized_kernel_bandwidth.m:85-86
    'th5': (8.07, 1),
    # Ref: realized_kernel_bandwidth.m:88-89
    'th16': (39.16, 1),
    # --- Non-flat-top, kernel_type 3 ---
    # Ref: realized_kernel_bandwidth.m:91-92 — ((12)^2 / 0.269)^(1/5)
    'nonflatparzen': ((12.0 ** 2 / 0.269) ** (1.0 / 5.0), 3),
    # Ref: realized_kernel_bandwidth.m:94-95 — ((1/5)^2 / (3*pi/5))^(1/5)
    'qs': (((1.0 / 5.0) ** 2 / (3.0 * math.pi / 5.0)) ** (1.0 / 5.0), 3),
    # Ref: realized_kernel_bandwidth.m:97-98 — ((2/3)^2 / (pi/3))^(1/5)
    'fejer': (((2.0 / 3.0) ** 2 / (math.pi / 3.0)) ** (1.0 / 5.0), 3),
    # Ref: realized_kernel_bandwidth.m:100-101 — ((pi^2/2)^2 / 0.52)^(1/5)
    'thinf': (((math.pi ** 2 / 2.0) ** 2 / 0.52) ** (1.0 / 5.0), 3),
    # Ref: realized_kernel_bandwidth.m:103-104 — (1^2 / (5/4))^(1/5)
    'bnhls': ((1.0 ** 2 / (5.0 / 4.0)) ** (1.0 / 5.0), 3),
}


def realized_kernel_bandwidth(
    noise_variance: float,
    IQ_estimate: float,
    options: dict,
) -> float:
    """
    Compute optimal bandwidth for realized kernel estimation.

    Estimates the optimal bandwidth for use in realized kernels, following
    Barndorff-Nielsen, Hansen, Lunde, and Shephard (2008). The bandwidth
    is computed using plug-in estimates of noise variance and integrated
    quarticity, combined with kernel-specific rate constants.

    Parameters
    ----------
    noise_variance : float
        Estimate of the market microstructure noise variance. Must be a
        non-negative finite scalar.
    IQ_estimate : float
        Estimate of the integrated quarticity. Must be a positive finite
        scalar (used under a square root).
    options : dict
        Realized kernel options dictionary. Must contain at minimum:

        - ``'kernel'`` : str
            Name of the kernel function. Must be one of the valid kernel
            types listed in ``KERNEL_LIST``. Comparison is
            case-insensitive.
        - ``'filteredN'`` : int or float
            Number of filtered observations (effective sample size after
            any pre-filtering). Must be positive.

        See ``realized_options`` for details on constructing the full
        options dictionary.

    Returns
    -------
    float
        Optimal bandwidth computed using plug-in estimates of unknown
        quantities. The bandwidth is always a positive real number.

    Raises
    ------
    ValueError
        If any required input is ``None``, if ``'kernel'`` is not present
        in *options*, if the kernel type string is not recognized, or if
        ``'filteredN'`` is not present in *options*.

    Warnings
    --------
    UserWarning
        Issued when ``xi_squared`` (= ``noise_variance / sqrt(IQ_estimate)``)
        exceeds 1.0, indicating that the integrated quarticity may be
        estimated too close to zero. In this case ``xi_squared`` is clamped
        to 1.0.

    Notes
    -----
    This is a helper function for ``realized_kernel``. The bandwidth
    formula depends on the kernel type:

    * **Type 1** (flat-top, higher-order kernels such as *parzen*, *cubic*,
      *th1–th16*):

      .. math:: H = c^{\\ast} \\, \\xi \\, \\sqrt{n}

    * **Type 2** (flat-top, Bartlett-type kernels such as *bartlett*,
      *2ndorder*, *epanechnikov*):

      .. math:: H = c^{\\ast} \\, \\xi^{2/3} \\, n^{2/3}

    * **Type 3** (non-flat-top kernels such as *nonflatparzen*, *qs*,
      *fejer*, *thinf*, *bnhls*):

      .. math:: H = c^{\\ast} \\, \\xi^{2/5} \\, n^{3/5}

    where :math:`\\xi^{2} = \\sigma^{2}_{\\text{noise}} / \\sqrt{IQ}`,
    *n* is the filtered sample size, and :math:`c^{\\ast}` is a
    kernel-specific constant tabulated or computed from the formulas in
    Barndorff-Nielsen et al. (2008).

    References
    ----------
    Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., and Shephard, N.
    (2008). "Designing realized kernels to measure the ex-post variation
    of equity prices in the presence of noise." *Econometrica*, 76(6),
    1481–1536.

    See Also
    --------
    realized_kernel : Main realized kernel estimator.
    realized_noise_estimate : Noise variance estimation.
    realized_options : Default options constructor.

    Examples
    --------
    >>> opts = {'kernel': 'parzen', 'filteredN': 1000}
    >>> bw = realized_kernel_bandwidth(0.001, 0.5, opts)
    >>> isinstance(bw, float)
    True
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: realized_kernel_bandwidth.m:28-30
    # ------------------------------------------------------------------
    if noise_variance is None or IQ_estimate is None or options is None:
        raise ValueError('Three inputs required.')

    # ------------------------------------------------------------------
    # Validate kernel option
    # Ref: realized_kernel_bandwidth.m:43-45
    # ------------------------------------------------------------------
    if 'kernel' not in options:
        raise ValueError(
            'KERNEL must be a field of OPTIONS and one of the listed types.'
        )

    # Ref: realized_kernel_bandwidth.m:51 — lower(options.kernel)
    kernel = options['kernel'].lower()

    if kernel not in _KERNEL_CONSTANTS:
        raise ValueError(
            'KERNEL must be a field of OPTIONS and one of the listed types.'
        )

    # ------------------------------------------------------------------
    # Retrieve kernel-specific constants
    # Ref: realized_kernel_bandwidth.m:51-108
    # ------------------------------------------------------------------
    c_star, kernel_type = _KERNEL_CONSTANTS[kernel]

    # ------------------------------------------------------------------
    # Compute xi squared
    # Ref: realized_kernel_bandwidth.m:112
    # xiSquared = noiseVariance / sqrt(IQEstimate)
    # ------------------------------------------------------------------
    if IQ_estimate <= 0.0:
        raise ValueError(
            'IQ_estimate must be a positive number (used under square root).'
        )
    xi_squared: float = float(noise_variance) / math.sqrt(float(IQ_estimate))

    # ------------------------------------------------------------------
    # Clamp xi_squared if it exceeds 1
    # Ref: realized_kernel_bandwidth.m:113-116
    # MATLAB warns: 'IQ estimaed o be close to 0.  Please check results'
    # (note: original MATLAB has typos — reproduced intent, not typos)
    # ------------------------------------------------------------------
    if xi_squared > 1.0:
        warnings.warn(
            'IQ estimated to be close to 0. Please check results',
            stacklevel=2,
        )
        xi_squared = 1.0

    # ------------------------------------------------------------------
    # Get the size of the filtered data
    # Ref: realized_kernel_bandwidth.m:118
    # nMax = options.filteredN
    # ------------------------------------------------------------------
    if 'filteredN' not in options:
        raise ValueError(
            "'filteredN' must be a field of OPTIONS."
        )
    n_max: float = float(options['filteredN'])

    if n_max <= 0.0:
        raise ValueError(
            "'filteredN' must be a positive number."
        )

    # ------------------------------------------------------------------
    # Compute the bandwidth
    # Ref: realized_kernel_bandwidth.m:120-126
    # ------------------------------------------------------------------
    if kernel_type == 1:
        # Flat-top higher-order kernels: n^(1/2) rate
        # Ref: realized_kernel_bandwidth.m:121
        bandwidth = c_star * (xi_squared ** (1.0 / 2.0)) * (n_max ** (1.0 / 2.0))
    elif kernel_type == 2:
        # Flat-top Bartlett-type kernels: n^(2/3) rate
        # Ref: realized_kernel_bandwidth.m:123
        bandwidth = c_star * (xi_squared ** (2.0 / 3.0)) * (n_max ** (2.0 / 3.0))
    elif kernel_type == 3:
        # Non-flat-top kernels: n^(3/5) rate
        # Ref: realized_kernel_bandwidth.m:125
        bandwidth = c_star * (xi_squared ** (2.0 / 5.0)) * (n_max ** (3.0 / 5.0))
    else:
        # Defensive — should never reach here given _KERNEL_CONSTANTS validation
        raise ValueError(f'Unknown kernel type: {kernel_type}')

    return float(bandwidth)
