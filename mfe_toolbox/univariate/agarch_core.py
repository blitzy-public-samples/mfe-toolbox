"""
Conditional variance recursion kernel for AGARCH(P,Q) and NAGARCH(P,Q) processes.

This module provides the core computational kernel for the Asymmetric GARCH
(AGARCH) and Nonlinear Asymmetric GARCH (NAGARCH) models. The implementation
is a direct translation of the C MEX acceleration kernel
``mex_source/agarch_core.c`` into a Numba JIT-compiled Python function,
preserving identical numerical behavior and 0-based indexing conventions.

The MATLAB wrapper ``univariate/agarch_core.m`` served as the pure-MATLAB
fallback; the C MEX provided 10-100x speedup. In Python, ``@numba.jit``
with ``nopython=True, cache=True`` replaces the C MEX gateway while
maintaining equivalent performance characteristics.

Migrated from:
    - ``univariate/agarch_core.m`` (MATLAB recursion, 83 lines)
    - ``mex_source/agarch_core.c`` (C MEX acceleration kernel, 105 lines)

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/12/2009
"""

import math

import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def agarch_core(
    data: np.ndarray,
    parameters: np.ndarray,
    back_cast: float,
    p: int,
    q: int,
    m: int,
    T: int,
    model_type: int,
) -> np.ndarray:
    """
    Conditional variance computation for AGARCH(P,Q) and NAGARCH(P,Q) processes.

    Computes the conditional variance sequence ``ht`` by applying the AGARCH or
    NAGARCH recursion over the input data series, using the supplied parameter
    vector and back-cast initialization values.

    Parameters
    ----------
    data : np.ndarray
        A 1-D array of length ``T`` containing mean-zero return data, augmented
        with ``m`` back-cast values at the front.
    parameters : np.ndarray
        A 1-D array of length ``p + q + 2`` containing the model parameters in
        the layout:
        ``[omega, alpha_1, ..., alpha_p, gamma, beta_1, ..., beta_q]``

        - ``parameters[0]``: omega (constant/intercept term)
        - ``parameters[1:p+1]``: alpha coefficients (symmetric innovation weights)
        - ``parameters[p+1]``: gamma (asymmetry parameter)
        - ``parameters[p+2:p+q+2]``: beta coefficients (variance persistence weights)
    back_cast : float
        The value used to initialize the conditional variance recursion for the
        first ``m`` observations. Typically set to the unconditional variance
        of the data.
    p : int
        Positive integer representing the number of symmetric innovation terms
        (ARCH order). Must satisfy ``p >= 1``.
    q : int
        Non-negative integer representing the number of lagged conditional
        variance terms (GARCH order). ``q = 0`` yields a pure ARCH model.
    m : int
        The number of back-cast observations required to initialize the
        recursion. Equals ``max(p, q)``.
    T : int
        Total length of ``data``, including any prepended back-cast entries.
    model_type : int
        Selects the variance process type:

        - ``1``: AGARCH — shock term is ``(r_t - gamma)^2``
        - ``2``: NAGARCH — shock term is ``(r_t - gamma * sqrt(h_t))^2``

    Returns
    -------
    ht : np.ndarray
        A 1-D float64 array of length ``T`` containing the conditional
        variances. The first ``m`` entries are set to ``back_cast``; entries
        ``m`` through ``T-1`` are computed via the recursion.

    Notes
    -----
    **AGARCH(P,Q) model** (``model_type == 1``):

    .. math::

        h_t = \\omega
              + \\sum_{j=1}^{p} \\alpha_j (r_{t-j} - \\gamma)^2
              + \\sum_{j=1}^{q} \\beta_j h_{t-j}

    **NAGARCH(P,Q) model** (``model_type == 2``):

    .. math::

        h_t = \\omega
              + \\sum_{j=1}^{p} \\alpha_j (r_{t-j} - \\gamma \\sqrt{h_{t-j}})^2
              + \\sum_{j=1}^{q} \\beta_j h_{t-j}

    The implementation translates directly from the C MEX source
    ``mex_source/agarch_core.c`` (lines 15-59), preserving 0-based array
    indexing. All loop bounds, parameter offsets, and array accesses are
    identical to the C source — no MATLAB 1-based indexing adjustments are
    applied.

    References
    ----------
    - Source: ``univariate/agarch_core.m``, ``mex_source/agarch_core.c``
    - Engle, R.F. and Ng, V.K. (1993). Measuring and Testing the Impact of
      News on Volatility. *Journal of Finance*, 48(5), 1749-1778.
    """
    # Allocate output arrays of length T — Ref: agarch_core.c:92-93
    ht = np.zeros(T)
    shock = np.zeros(T)

    # Extract asymmetry parameter and precompute its square
    # Ref: agarch_core.c:19-20
    gamma = parameters[p + 1]
    gamma2 = gamma * gamma

    if model_type == 1:
        # ======================================================================
        # AGARCH branch: shock term is (r_t - gamma)^2
        # ======================================================================

        # Initialization of back-cast periods
        # Ref: agarch_core.c:23-27
        for j in range(m):
            ht[j] = back_cast                 # Ref: agarch_core.c:25
            shock[j] = back_cast + gamma2     # Ref: agarch_core.c:26

        # Main AGARCH variance recursion
        # Ref: agarch_core.c:29-39
        for i in range(m, T):
            # Start with omega (constant term)
            # Ref: agarch_core.c:30
            ht[i] = parameters[0]

            # Add alpha * shock (ARCH component)
            # Ref: agarch_core.c:31-33
            for j in range(p):
                ht[i] += parameters[j + 1] * shock[i - 1 - j]

            # Add beta * lagged variance (GARCH component)
            # Ref: agarch_core.c:34-36
            for j in range(q):
                ht[i] += parameters[j + p + 2] * ht[i - 1 - j]

            # Compute AGARCH shock: (data[i] - gamma)^2
            # Ref: agarch_core.c:37-38
            shock[i] = data[i] - gamma
            shock[i] = shock[i] * shock[i]
    else:
        # ======================================================================
        # NAGARCH branch: shock term is (r_t - gamma * sqrt(h_t))^2
        # ======================================================================

        # Initialization of back-cast periods
        # Ref: agarch_core.c:42-46
        for j in range(m):
            ht[j] = back_cast                         # Ref: agarch_core.c:44
            shock[j] = back_cast * (1.0 + gamma2)     # Ref: agarch_core.c:45

        # Main NAGARCH variance recursion
        # Ref: agarch_core.c:48-58
        for i in range(m, T):
            # Start with omega (constant term)
            # Ref: agarch_core.c:49
            ht[i] = parameters[0]

            # Add alpha * shock (ARCH component)
            # Ref: agarch_core.c:50-52
            for j in range(p):
                ht[i] += parameters[j + 1] * shock[i - 1 - j]

            # Add beta * lagged variance (GARCH component)
            # Ref: agarch_core.c:53-55
            for j in range(q):
                ht[i] += parameters[j + p + 2] * ht[i - 1 - j]

            # Compute NAGARCH shock: (data[i] - gamma * sqrt(ht[i]))^2
            # KEY NAGARCH DIFFERENCE from AGARCH: uses sqrt(ht[i])
            # Ref: agarch_core.c:56-57
            shock[i] = data[i] - gamma * math.sqrt(ht[i])
            shock[i] = shock[i] * shock[i]

    return ht
