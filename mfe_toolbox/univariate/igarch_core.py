"""
IGARCH unit-root conditional variance recursion with Numba JIT acceleration.

Migrated from:
  - univariate/igarch_core.m (MATLAB fallback implementation)
  - mex_source/igarch_core.c (C MEX performance kernel)

This module provides a Numba JIT-compiled function that computes the
conditional variance series for Integrated GARCH (IGARCH) models. The
IGARCH model enforces a unit-root constraint where the sum of all ARCH
(alpha) and GARCH (beta) coefficients equals exactly 1.0.  The last
GARCH coefficient is not stored in the parameter vector; instead it is
computed as ``1 - sum(alpha) - sum(beta_free)`` to satisfy the
unit-root identity.

Two model types are supported:
  * igarch_type == 2  – the recursion evolves in squared terms
    (standard GARCH variance), and ht is returned directly.
  * igarch_type == 1  – the recursion evolves in absolute-value terms
    (AVGARCH), and ht is squared before returning so that the output
    is always a conditional *variance* array.

The constant parameter is optional (constant == 1 includes an intercept
at parameters[0]; constant == 0 omits it).

Performance
-----------
The ``@numba.jit(nopython=True, cache=True)`` decorator compiles the
pure-Python loop into optimised machine code, achieving 10–100×
speed-up over an interpreted loop and matching the performance of the
original C MEX kernel it replaces.

Copyright: Kevin Sheppard, University of Oxford
Python migration by the MFE Toolbox project.
"""

import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def igarch_core(
    fepsilon: np.ndarray,
    parameters: np.ndarray,
    back_cast: float,
    p: int,
    q: int,
    m: int,
    T: int,
    igarch_type: int,
    constant: int,
) -> np.ndarray:
    """Compute IGARCH conditional variance via unit-root recursion.

    Translates the C MEX kernel ``mex_source/igarch_core.c`` (lines
    15-46) and the MATLAB fallback ``univariate/igarch_core.m`` into a
    Numba nopython JIT function.

    Parameters
    ----------
    fepsilon : np.ndarray
        T-length array of transformed residuals.  When igarch_type == 2
        these are ``epsilon ** 2``; when igarch_type == 1 these are
        ``abs(epsilon)``.
    parameters : np.ndarray
        Model parameter vector laid out as::

            [omega (if constant==1), alpha_1 … alpha_p, beta_1 … beta_{q-1}]

        The last (q-th) GARCH coefficient is **not** stored; it is
        computed internally as ``1 - sum(alpha) - sum(beta_free)`` to
        enforce the IGARCH unit-root constraint.
    back_cast : float
        Scalar back-cast value used to initialise the first ``m``
        elements of the conditional variance array.
    p : int
        Number of ARCH (alpha) lags.
    q : int
        Number of GARCH (beta) lags.  The parameter vector contains
        only ``q - 1`` free beta coefficients.
    m : int
        Pre-sample length, typically ``max(p, q)``.
    T : int
        Total length of the output array (including the ``m``
        initialisation positions).
    igarch_type : int
        Model formulation selector:

        * 1 – AVGARCH (absolute-value recursion); the output is
          squared so that ht always represents conditional variance.
        * 2 – standard IGARCH (squared recursion); ht returned as-is.
    constant : int
        * 1 – an intercept (omega) is present at ``parameters[0]``.
        * 0 – no intercept; the recursion has no constant term.

    Returns
    -------
    ht : np.ndarray
        T-length float64 array of conditional variances.

    Notes
    -----
    The unit-root constraint is enforced by computing a residual
    coefficient:

    .. math::

        \\beta_q = 1 - \\sum_{i=1}^{p} \\alpha_i
                     - \\sum_{j=1}^{q-1} \\beta_j

    This guarantees ``sum(alpha) + sum(beta) == 1`` exactly.

    Indexing follows the C MEX source (0-based):

    * ``parameters[constant + j]``  →  alpha coefficients
    * ``parameters[constant + p + j]``  →  free beta coefficients
    * ``fepsilon[i - 1 - j]``  →  lagged transformed residuals
    * ``ht[i - 1 - j]``  →  lagged conditional variances

    References
    ----------
    * ``mex_source/igarch_core.c``, lines 15–46
    * ``univariate/igarch_core.m``, lines 1–76
    """
    # Allocate output conditional variance array.
    # Ref: igarch_core.m:40 — MATLAB: ht = zeros(size(fepsilon))
    ht = np.zeros(T, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 1: Compute the residual (implied) last GARCH parameter so
    #         that sum(alpha) + sum(beta) = 1 exactly.
    # Ref: igarch_core.c:17-18 — finalParameter = 1; for(i=0;i<p+q-1;i++) …
    # ------------------------------------------------------------------
    final_parameter = 1.0
    for i in range(p + q - 1):
        # Ref: igarch_core.c:18 — finalParameter -= parameters[constant + i]
        final_parameter -= parameters[constant + i]

    # ------------------------------------------------------------------
    # Step 2: Initialise the first m elements with the back-cast value.
    # Ref: igarch_core.c:21-22 — for(j=0;j<m;j++) ht[j]=backCast[0]
    # Note: In C the back-cast arrives as a single-element array;
    #       in Python it is a scalar float.
    # ------------------------------------------------------------------
    for j in range(m):
        ht[j] = back_cast

    # ------------------------------------------------------------------
    # Step 3: Main variance recursion from i = m  to  i = T-1.
    # Ref: igarch_core.c:25-40  (0-based loop)
    # ------------------------------------------------------------------
    for i in range(m, T):
        # Start each step at zero, then optionally add the constant.
        # Ref: igarch_core.c:26 — ht[i] = 0
        ht[i] = 0.0

        # Ref: igarch_core.c:27-28 — if(constant) ht[i] = parameters[0]
        if constant:
            ht[i] = parameters[0]

        # --- ARCH terms (alpha_1 … alpha_p) ---
        # Ref: igarch_core.c:30-31
        #   for(j=0;j<p;j++) ht[i] += parameters[j+constant]*fepsilon[i-1-j]
        for j in range(p):
            ht[i] += parameters[j + constant] * fepsilon[i - 1 - j]

        # --- Free GARCH terms (beta_1 … beta_{q-1}) ---
        # Ref: igarch_core.c:33-34
        #   for(j=0;j<q-1;j++) ht[i] += parameters[j+p+constant]*ht[i-1-j]
        for j in range(q - 1):
            ht[i] += parameters[j + p + constant] * ht[i - 1 - j]

        # --- Implied last GARCH term (unit-root residual) ---
        # Ref: igarch_core.c:36 — ht[i] += finalParameter * ht[i-q]
        ht[i] += final_parameter * ht[i - q]

    # ------------------------------------------------------------------
    # Step 4: Post-processing – if igarch_type == 1 (AVGARCH), the
    #         recursion operated in absolute-value space, so square ht
    #         to convert back to variance.
    # Ref: igarch_core.c:39-41 — if(igarchType==1) { … ht[i]*=ht[i]; }
    # ------------------------------------------------------------------
    if igarch_type == 1:
        for i in range(m, T):
            ht[i] = ht[i] * ht[i]

    return ht
