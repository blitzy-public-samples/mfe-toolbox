"""
Conditional variance recursion kernel for the APARCH(P,O,Q) process.

This module provides the core computational kernel for the Asymmetric Power
ARCH (APARCH) model of Ding, Granger, and Engle (1993). The APARCH model
generalises many GARCH-family models by introducing a power parameter delta
and asymmetry parameters gamma, producing the recursion:

    h(t)^(delta/2) = omega
                     + sum_{j=1}^{p} alpha_j * (|e_{t-j}| + gamma_j * e_{t-j})^delta
                     + sum_{j=1}^{q} beta_j * h(t-j)^(delta/2)

Unlike agarch_core, egarch_core, igarch_core, and tarch_core, APARCH has
**no C MEX file** in the original toolbox — the MATLAB ``.m`` file
(``univariate/aparch_core.m``) IS the sole reference implementation. Numba
``@jit(nopython=True, cache=True)`` provides inner-loop acceleration
equivalent to the C MEX pattern used by the other GARCH families.

Migrated from:
    - ``univariate/aparch_core.m`` (MATLAB recursion, 74 lines, 2350 bytes)

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005
"""

import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def aparch_core(
    data: np.ndarray,
    parameters: np.ndarray,
    back_cast: float,
    p: int,
    o: int,
    q: int,
    m: int,
    T: int,
) -> np.ndarray:
    """
    Compute conditional variance for an APARCH(P,O,Q) process.

    Implements the full APARCH power-ARCH recursion, translating the MATLAB
    reference ``univariate/aparch_core.m`` into a Numba JIT-compiled function
    with 0-based indexing.

    Parameters
    ----------
    data : np.ndarray
        A 1-D float64 array of length ``T`` containing mean-zero residuals,
        augmented with ``m`` leading zeros for the look-back period.  Replaces
        the separate ``data_aug`` and ``abs_data_aug`` inputs in the MATLAB
        source; absolute values are computed internally via ``np.abs``.
    parameters : np.ndarray
        A 1-D float64 array of length ``1 + p + o + q + 1`` containing model
        parameters in the layout::

            [omega, alpha_1, ..., alpha_p,
             gamma_1, ..., gamma_o,
             beta_1, ..., beta_q,
             delta]

        - ``parameters[0]``                   : omega   (intercept)
        - ``parameters[1 : 1+p]``             : alpha   (symmetric ARCH weights)
        - ``parameters[1+p : 1+p+o]``         : gamma   (asymmetry / leverage)
        - ``parameters[1+p+o : 1+p+o+q]``     : beta    (GARCH persistence)
        - ``parameters[1+p+o+q]``             : delta   (power parameter)
    back_cast : float
        Back-cast initialisation value for ``htdelta`` (i.e. ``h^(delta/2)``)
        during the first ``m`` observations.  Typically the unconditional
        variance raised to the ``delta/2`` power.
    p : int
        Positive integer — number of symmetric innovation (ARCH) terms.
    o : int
        Non-negative integer — number of asymmetric innovation terms.
        Set to 0 for a symmetric power-ARCH process.
    q : int
        Non-negative integer — number of lagged conditional variance (GARCH)
        terms.  Set to 0 for a pure power-ARCH model.
    m : int
        Number of back-cast observations required, typically ``max(p, q)``.
    T : int
        Total length of ``data`` including the ``m`` prepended back-cast
        entries.

    Returns
    -------
    ht : np.ndarray
        A 1-D float64 array of length ``T`` containing conditional variances.
        The first ``m`` entries are set to the sample variance of the
        non-augmented data portion; entries ``m`` through ``T-1`` are
        computed via the APARCH recursion.

    Notes
    -----
    **Model specification (Ding, Granger and Engle, 1993):**

    The APARCH recursion is expressed in terms of ``htdelta = h^(delta/2)``:

    .. math::

        htdelta_t = \\omega
                    + \\sum_{j=1}^{p} \\alpha_j
                      \\bigl(|\\varepsilon_{t-j}|
                             + \\gamma_j \\, \\varepsilon_{t-j}\\bigr)^{\\delta}
                    + \\sum_{j=1}^{q} \\beta_j \\, htdelta_{t-j}

    The conditional variance is then recovered as:

    .. math::

        h_t = htdelta_t^{2/\\delta}

    Special cases:
    - ``delta = 2, o = 0`` recovers the standard GARCH model.
    - ``delta = 2, o > 0`` recovers the GJR-GARCH / TARCH model.
    - ``delta = 1`` gives the absolute-value GARCH model.

    The sign convention for gamma follows the original MATLAB source:
    ``|e| + gamma * e``.  With ``|gamma| < 1``, the base is guaranteed
    non-negative:

    - ``e >= 0``:  ``|e| + gamma * e = (1 + gamma) * e  >= 0``
    - ``e <  0``:  ``|e| + gamma * e = (1 - gamma) * |e| >= 0``

    Ref: aparch_core.m lines 1-74.  All index transformations from MATLAB
    1-based to Python 0-based are annotated inline.
    """
    # ------------------------------------------------------------------
    # Allocate output arrays — Ref: aparch_core.m:38-39
    # htdelta stores h^(delta/2);  ht stores the actual variance h.
    # ------------------------------------------------------------------
    htdelta = np.zeros(T)
    ht = np.zeros(T)

    # ------------------------------------------------------------------
    # Extract delta (power parameter) from end of parameter vector
    # MATLAB: parameters(1+p+o+q+1) → Python 0-based: parameters[p+o+q+1]
    # Ref: aparch_core.m:40
    # ------------------------------------------------------------------
    delta = parameters[p + o + q + 1]

    # ------------------------------------------------------------------
    # Compute absolute values internally (replaces MATLAB abs_data_aug)
    # Ref: aparch_core.m uses a separate abs_data_aug input argument;
    # the Python API computes it here via np.abs for a simpler signature.
    # ------------------------------------------------------------------
    abs_data = np.abs(data)

    # ------------------------------------------------------------------
    # Initialise htdelta for the back-cast period [0, m)
    # MATLAB: htdelta(1:m) = back_cast  — Ref: aparch_core.m:43
    # ------------------------------------------------------------------
    for i in range(m):
        htdelta[i] = back_cast

    # ------------------------------------------------------------------
    # Initialise ht[0:m] with the sample variance of the non-augmented
    # data portion, i.e. the inner product data[m:T]' * data[m:T] / (T-m).
    # MATLAB: ht(1:m) = data_aug(m+1:T)' * data_aug(m+1:T) / (T-m)
    # Ref: aparch_core.m:44
    # ------------------------------------------------------------------
    sample_var = 0.0
    for i in range(m, T):
        sample_var += data[i] * data[i]
    if T > m:
        sample_var = sample_var / float(T - m)
    for i in range(m):
        ht[i] = sample_var

    # ------------------------------------------------------------------
    # Precompute inverse-delta for htdelta → ht conversion
    # MATLAB: deltainv = 2/delta  — Ref: aparch_core.m:46
    # ------------------------------------------------------------------
    deltainv = 2.0 / delta

    # ------------------------------------------------------------------
    # Extract omega (constant intercept)
    # MATLAB: omega = parameters(1)  — Ref: aparch_core.m:48
    # ------------------------------------------------------------------
    omega = parameters[0]

    # ==================================================================
    # Main APARCH recursion loop
    # MATLAB: for i = m+1 : T  (1-based)  →  Python: range(m, T)  (0-based)
    # Ref: aparch_core.m:52-74
    # ==================================================================
    for i in range(m, T):
        # Start with the constant/intercept term
        # Ref: aparch_core.m:53
        htdelta[i] = omega

        # --------------------------------------------------------------
        # ARCH terms with optional asymmetry
        # MATLAB: for j = 1:p  →  Python: j in range(1, p+1)
        # alpha(j) in MATLAB → parameters[j] in Python (0-based)
        # gamma(j) in MATLAB → parameters[p + j] in Python (0-based)
        # Ref: aparch_core.m:54-59
        # --------------------------------------------------------------
        for j in range(1, p + 1):
            if o >= j:
                # With asymmetry: (|e_{t-j}| + gamma_j * e_{t-j})^delta
                # Ref: aparch_core.m:56
                # gamma_j is at parameters[1+p+(j-1)] = parameters[p+j] (0-based)
                gamma_j = parameters[p + j]
                htdelta[i] += parameters[j] * (
                    abs_data[i - j] + gamma_j * data[i - j]
                ) ** delta
            else:
                # Without asymmetry: (|e_{t-j}|)^delta
                # Ref: aparch_core.m:58
                htdelta[i] += parameters[j] * (abs_data[i - j]) ** delta

        # --------------------------------------------------------------
        # GARCH terms (lagged htdelta persistence)
        # MATLAB: for j = 1:q  →  Python: j in range(1, q+1)
        # beta(j) in MATLAB → parameters[p+o+j] in Python (0-based)
        # MATLAB: parameters(p+o+2 : p+o+q+1) → parameters[p+o+1 : p+o+q+1]
        # Individual: beta(j) = parameters(p+o+1+j) → parameters[p+o+j]
        # Ref: aparch_core.m:61-63
        # --------------------------------------------------------------
        for j in range(1, q + 1):
            htdelta[i] += parameters[p + o + j] * htdelta[i - j]

        # --------------------------------------------------------------
        # Convert htdelta to ht via power transformation
        # MATLAB: htTemp = htdelta(i)^deltainv  — Ref: aparch_core.m:65
        # ht = htdelta^(2/delta) since htdelta represents h^(delta/2)
        # Ref: aparch_core.m:65-73
        #
        # Note: The original MATLAB code clamps ht between LB and UB
        # (lines 66-72).  The Python signature omits LB/UB; the calling
        # driver (aparch.py) is responsible for parameter-space bounds.
        # A small numerical floor is applied to prevent negative-base
        # power operations that would produce NaN.
        # --------------------------------------------------------------
        if htdelta[i] > 0.0:
            ht[i] = htdelta[i] ** deltainv
        else:
            # Guard: if htdelta goes non-positive (possible during
            # optimizer exploration with out-of-range parameters), clamp
            # to a tiny positive value to prevent NaN propagation.
            htdelta[i] = 1e-20
            ht[i] = htdelta[i] ** deltainv

    return ht
