"""
Optimal jitter lag length computation for realized kernel end-point treatment.

Computes the MSE-minimizing number of lags for endpoint jittering used by the
realized kernel estimator. The optimal jitter lag length balances the noise
reduction from endpoint pre-averaging against the loss of information from
discarding observations.

Migrated from: realized/realized_kernel_jitter_lag_length.m
Original author: Kevin Sheppard

See Also
--------
realized_kernel : Main realized kernel estimator that calls this function.
realized_kernel_bandwidth : Optimal bandwidth selection for realized kernels.
realized_kernel_weights : Kernel weight function lookup.

References
----------
Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., and Shephard, N. (2008a).
    "Designing realized kernels to measure the ex-post variation of equity
    prices in the presence of noise."
Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., and Shephard, N. (2008b).
    "Realized kernels in practice: Trades and quotes."
"""

import math
import numpy as np


# --- Supported kernel names ---
# Ref: realized_kernel_jitter_lag_length.m:65 — complete list of recognized kernels
_VALID_KERNELS = frozenset({
    'bartlett', 'twoscale', '2ndorder', 'epanechnikov',
    'cubic', 'multiscale',
    '5thorder', '6thorder', '7thorder', '8thorder',
    'parzen', 'th1', 'th2', 'th5', 'th16',
    'nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls',
})


def realized_kernel_jitter_lag_length(
    noise_estimate: float,
    iq_estimate: float,
    kernel: str,
    n: int,
) -> int:
    """Compute optimal jitter lag length for realized kernel endpoint treatment.

    Determines the optimal number of observations to pre-average at the
    endpoints of the price series, minimising a bias-variance MSE criterion
    that depends on the kernel type, noise level, integrated quarticity, and
    sample size.

    Parameters
    ----------
    noise_estimate : float
        Estimated variance of the microstructure noise present in the
        high-frequency asset price.  Must be a non-negative scalar.
    iq_estimate : float
        Estimated integrated quarticity.  For practical purposes it is often
        reasonable to use the square of a low-frequency integrated-variance
        measure.  Must be a non-negative scalar.
    kernel : str
        Kernel type identifier.  Recognised values (case-insensitive):

        *Non-flat-top, weakly positive, n^{1/5} rate:*
            ``'nonflatparzen'`` (recommended), ``'qs'``, ``'fejer'``,
            ``'thinf'``, ``'bnhls'``

        *Flat-top, n^{1/4} rate:*
            ``'parzen'``, ``'th1'``, ``'th2'``, ``'th5'``, ``'th16'``,
            ``'cubic'`` / ``'multiscale'``, ``'5thorder'``, ``'6thorder'``,
            ``'7thorder'``, ``'8thorder'``

        *Flat-top, n^{1/6} rate:*
            ``'bartlett'`` / ``'twoscale'``, ``'2ndorder'``,
            ``'epanechnikov'``
    n : int
        Number of price observations (not returns).  Must be a positive
        integer.

    Returns
    -------
    int
        Optimal number of observations to jitter (pre-average) at each
        endpoint.  A value of 1 indicates that no pre-averaging is needed.

    Raises
    ------
    ValueError
        If any input fails validation.

    Notes
    -----
    The function minimises the composite MSE criterion over candidate lag
    lengths *m* = 1, …, *N*-1:

    .. math::

        \\text{MSE}(m) = 8 \\cdot \\hat\\omega^4 \\cdot m^{-2}
                        + \\text{avar} \\cdot (N - m)^{-2p}

    where *p* depends on the kernel convergence rate (1/4, 1/5, or 1/6) and
    *avar* encodes the asymptotic variance contribution.

    Examples
    --------
    >>> realized_kernel_jitter_lag_length(1e-6, 1e-4, 'nonflatparzen', 5000)
    3
    """

    # ------------------------------------------------------------------
    # Input Checking — Ref: realized_kernel_jitter_lag_length.m:56-71
    # ------------------------------------------------------------------

    # Validate noise_estimate
    # Ref: realized_kernel_jitter_lag_length.m:59-60
    if not np.isscalar(noise_estimate):
        raise ValueError('NOISEESTIMATE must be a non-negative scalar.')
    noise_estimate = float(noise_estimate)
    if noise_estimate < 0.0:
        raise ValueError('NOISEESTIMATE must be a non-negative scalar.')

    # Validate iq_estimate
    # Ref: realized_kernel_jitter_lag_length.m:62-63
    if not np.isscalar(iq_estimate):
        raise ValueError('IQESTIMATE must be a non-negative scalar.')
    iq_estimate = float(iq_estimate)
    if iq_estimate < 0.0:
        raise ValueError('IQESTIMATE must be a non-negative scalar.')

    # Validate kernel
    # Ref: realized_kernel_jitter_lag_length.m:65-66
    if not isinstance(kernel, str):
        raise ValueError('KERNEL must be one of the listed types.')
    kernel_lower = kernel.lower()
    if kernel_lower not in _VALID_KERNELS:
        raise ValueError('KERNEL must be one of the listed types.')

    # Validate n
    # Ref: realized_kernel_jitter_lag_length.m:69-71
    if not np.isscalar(n):
        raise ValueError('N must be an integer greater than 1.')
    n_int = int(n)
    if n_int < 1 or n_int != n:
        raise ValueError('N must be an integer greater than 1.')

    # ------------------------------------------------------------------
    # Kernel constant look-up table
    # Ref: realized_kernel_jitter_lag_length.m:77-172
    # ------------------------------------------------------------------
    c_star: float
    k00: float
    kernel_type: int
    # k11, k22 only used for kernel_type == 1
    k11: float = 0.0
    k22: float = 0.0

    if kernel_lower in ('bartlett', 'twoscale'):
        # Ref: realized_kernel_jitter_lag_length.m:78-81
        c_star = 2.28
        k00 = 1.0 / 3.0
        kernel_type = 2
    elif kernel_lower == '2ndorder':
        # Ref: realized_kernel_jitter_lag_length.m:83-85
        c_star = 3.42
        k00 = 1.0 / 5.0
        kernel_type = 2
    elif kernel_lower == 'epanechnikov':
        # Ref: realized_kernel_jitter_lag_length.m:87-89
        c_star = 2.46
        k00 = 8.0 / 15.0
        kernel_type = 2
    elif kernel_lower in ('cubic', 'multiscale'):
        # Ref: realized_kernel_jitter_lag_length.m:90-95
        c_star = 3.68
        k00 = 0.371
        k11 = 1.20
        k22 = 12.0
        kernel_type = 1
    elif kernel_lower == '5thorder':
        # Ref: realized_kernel_jitter_lag_length.m:96-101
        k00 = 0.391
        k11 = 1.42
        k22 = 17.1
        c_star = 3.70
        kernel_type = 1
    elif kernel_lower == '6thorder':
        # Ref: realized_kernel_jitter_lag_length.m:102-107
        k00 = 0.471
        k11 = 1.55
        k22 = 22.8
        c_star = 3.97
        kernel_type = 1
    elif kernel_lower == '7thorder':
        # Ref: realized_kernel_jitter_lag_length.m:108-113
        k00 = 0.533
        k11 = 1.71
        k22 = 31.8
        c_star = 4.11
        kernel_type = 1
    elif kernel_lower == '8thorder':
        # Ref: realized_kernel_jitter_lag_length.m:114-119
        k00 = 0.582
        k11 = 1.87
        k22 = 43.8
        c_star = 4.31
        kernel_type = 1
    elif kernel_lower == 'parzen':
        # Ref: realized_kernel_jitter_lag_length.m:120-125
        k00 = 0.269
        k11 = 1.50
        k22 = 24.0
        c_star = 4.77
        kernel_type = 1
    elif kernel_lower == 'th1':
        # Ref: realized_kernel_jitter_lag_length.m:126-131
        k00 = 0.375
        k11 = 1.23
        k22 = 12.1
        c_star = 3.70
        kernel_type = 1
    elif kernel_lower == 'th2':
        # Ref: realized_kernel_jitter_lag_length.m:132-137
        k00 = 0.219
        k11 = 1.71
        k22 = 41.7
        c_star = 5.74
        kernel_type = 1
    elif kernel_lower == 'th5':
        # Ref: realized_kernel_jitter_lag_length.m:138-143
        k00 = 0.097
        k11 = 3.50
        k22 = 489.0
        c_star = 8.07
        kernel_type = 1
    elif kernel_lower == 'th16':
        # Ref: realized_kernel_jitter_lag_length.m:144-149
        k00 = 0.032
        k11 = 10.26
        k22 = 14374.0
        c_star = 39.16
        kernel_type = 1
    elif kernel_lower == 'nonflatparzen':
        # Ref: realized_kernel_jitter_lag_length.m:150-153
        c_star = (12.0 ** 2 / 0.269) ** (1.0 / 5.0)
        k00 = 0.269
        kernel_type = 3
    elif kernel_lower == 'qs':
        # Ref: realized_kernel_jitter_lag_length.m:154-157
        c_star = ((1.0 / 5.0) ** 2 / (3.0 * math.pi / 5.0)) ** (1.0 / 5.0)
        k00 = 3.0 * math.pi / 5.0
        kernel_type = 3
    elif kernel_lower == 'fejer':
        # Ref: realized_kernel_jitter_lag_length.m:158-161
        c_star = ((2.0 / 3.0) ** 2 / (math.pi / 3.0)) ** (1.0 / 5.0)
        k00 = math.pi / 3.0
        kernel_type = 3
    elif kernel_lower == 'thinf':
        # Ref: realized_kernel_jitter_lag_length.m:162-165
        c_star = ((math.pi ** 2 / 2.0) ** 2 / 0.52) ** (1.0 / 5.0)
        k00 = 0.52
        kernel_type = 3
    elif kernel_lower == 'bnhls':
        # Ref: realized_kernel_jitter_lag_length.m:166-169
        c_star = (1.0 ** 2 / (5.0 / 4.0)) ** (1.0 / 5.0)
        k00 = 5.0 / 4.0
        kernel_type = 3
    else:
        # Ref: realized_kernel_jitter_lag_length.m:170-172
        raise ValueError('KERNEL must be one of the listed types.')

    # ------------------------------------------------------------------
    # Compute the asymptotic variance contribution (avar)
    # Ref: realized_kernel_jitter_lag_length.m:175-192
    # ------------------------------------------------------------------
    if kernel_type == 1:
        # n^{1/4} flat-top kernels
        # Ref: realized_kernel_jitter_lag_length.m:176-183
        # Approximate at the constant variance solution
        d = k00 * k22 / (k11 ** 2)
        fd = math.sqrt(1.0 + math.sqrt(1.0 + 3.0 * d))
        g = math.sqrt(k00 * k11) * (1.0 / fd + fd)
        avar = (16.0 / 3.0) * g * noise_estimate * (iq_estimate ** (3.0 / 4.0))
        power = 1.0 / 4.0
    elif kernel_type == 2:
        # n^{1/6} flat-top kernels
        # Ref: realized_kernel_jitter_lag_length.m:184-187
        avar = 6.0 * c_star * k00 * (noise_estimate ** (4.0 / 3.0)) * (iq_estimate ** (2.0 / 3.0))
        power = 1.0 / 6.0
    elif kernel_type == 3:
        # n^{1/5} non-flat-top kernels
        # Ref: realized_kernel_jitter_lag_length.m:188-192
        avar = 5.0 * c_star * k00 * (noise_estimate ** (4.0 / 5.0)) * (iq_estimate ** (4.0 / 5.0))
        power = 1.0 / 5.0
    else:
        # This branch should be unreachable given the kernel validation above,
        # but is included for defensive completeness.
        raise ValueError('KERNEL must be one of the listed types.')

    # ------------------------------------------------------------------
    # Minimise the MSE over candidate jitter lag lengths m = 1 … N-1
    #
    # MSE(m) = 8 * noiseEstimate^2 * m^{-2}
    #        + avar * (N - m)^{-2*power}
    #
    # Ref: realized_kernel_jitter_lag_length.m:194-202
    # ------------------------------------------------------------------
    noise_sq = noise_estimate ** 2

    best_mse = math.inf
    best_m = 1  # Default to 1 (no jittering)

    # Ref: realized_kernel_jitter_lag_length.m:197-199  — for m=1:(N-1)
    # MATLAB uses 1-based loop; Python range produces same m values.
    for m in range(1, n_int):
        mse_val = 8.0 * noise_sq * (m ** (-2)) + avar * ((n_int - m) ** (-2.0 * power))
        if mse_val < best_mse:
            best_mse = mse_val
            best_m = m

    # Ref: realized_kernel_jitter_lag_length.m:201-202  — [~,jitterLags] = min(MSE)
    return best_m
