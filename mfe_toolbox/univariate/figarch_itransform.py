"""
FIGARCH(q,d,p) inverse parameter transformation.

Transforms unconstrained parameters (real line) back to constrained parameter
space satisfying FIGARCH non-negativity and boundedness constraints.  This is
the exact inverse of :func:`figarch_transform` and is used during FIGARCH
estimation to recover economically meaningful parameters from the optimizer's
unconstrained search space.

Migrated from: univariate/figarch_itransform.m
Copyright: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Original Revision: 1  Date: 7/12/2009
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def figarch_itransform(
    parameters: np.ndarray,
    p: int,
    q: int,
    error_type: int,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """Inverse-transform unconstrained FIGARCH parameters to the constrained space.

    Maps parameters from the real line to values satisfying the FIGARCH
    non-negativity and boundedness constraints.  Used in estimation of
    FIGARCH models to convert optimizer output back to the constrained
    parameter space.

    Parameters
    ----------
    parameters : np.ndarray
        Column vector of unconstrained parameters.  Layout (0-indexed)::

            [0]     : omega  (unconstrained)
            [1]     : phi    (unconstrained, present only when p=1)
            [1+p]   : d      (unconstrained)
            [2+p]   : beta   (unconstrained, present only when q=1)
            [2+p+q] : nu     (unconstrained, present when error_type in {2,3,4})
            [3+p+q] : lambda (unconstrained, present only when error_type=4)

    p : int
        0 or 1 indicating whether the autoregressive (phi) term is present.
    q : int
        0 or 1 indicating whether the moving-average (beta) term is present.
    error_type : int
        Innovation distribution type:

        * 1 — Gaussian
        * 2 — Student's *t*
        * 3 — Generalized Error Distribution (GED)
        * 4 — Skewed Student's *t*

    Returns
    -------
    trans_parameters : np.ndarray
        Constrained parameter vector ``[omega, phi, d, beta]`` where only
        elements corresponding to active terms (``p``, ``q``) are included.
        Satisfies:

        * omega > 0
        * 0 < d < 1
        * 0 <= phi <= (1 - d) / 2
        * 0 <= beta <= d + phi
    nu : float or None
        Distribution shape parameter.  ``None`` when ``error_type == 1``.

        * Student's *t* (error_type 2, 4): nu > 2  (via ``2.01 + x**2``)
        * GED (error_type 3): 1.01 < nu < 50.01  (via logistic scaling)
    lam : float or None
        Distribution asymmetry parameter.  ``None`` unless ``error_type == 4``.
        Constrained to (-0.99, 0.99) via logistic mapping.

    Notes
    -----
    This function is the exact inverse of :func:`figarch_transform`.

    The parameter layout in the input vector depends on ``p`` and ``q``:

    +-------+----------+----------+
    | Index | p=0      | p=1      |
    +=======+==========+==========+
    | 0     | omega    | omega    |
    +-------+----------+----------+
    | 1     | d        | phi      |
    +-------+----------+----------+
    | 2     | beta*    | d        |
    +-------+----------+----------+
    | 3     | —        | beta*    |
    +-------+----------+----------+

    \\* beta present only when q=1.

    References
    ----------
    .. [1] Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
       Fractionally integrated generalized autoregressive conditional
       heteroskedasticity. *Journal of Econometrics*, 74(1), 3-30.

    See Also
    --------
    figarch_transform : Forward (constrained → unconstrained) transform.
    figarch : FIGARCH model driver.
    """
    # Ensure input is a contiguous float64 numpy array (defensive copy)
    parameters = np.array(parameters, dtype=np.float64)

    # ------------------------------------------------------------------
    # Distribution parameter transforms (nu and lambda)
    # Ref: figarch_itransform.m:42-65 — handle nu and lambda first
    # ------------------------------------------------------------------
    nu: Optional[float] = None
    lam: Optional[float] = None

    if error_type == 2 or error_type == 4:
        # Student's t or Skewed-t: nu = 2.01 + x^2  (ensures nu > 2.01)
        # Ref: figarch_itransform.m:45-46
        nu_unc = float(parameters[p + q + 2])  # 0-indexed from MATLAB p+q+3
        nu = 2.01 + nu_unc * nu_unc
    elif error_type == 3:
        # GED: logistic mapping to (1.01, ~50.01) with overflow protection
        # Ref: figarch_itransform.m:48-54
        nu_unc = float(parameters[p + q + 2])
        # Overflow protection: cap at 100 before exp
        # Ref: figarch_itransform.m:50-52
        if nu_unc > 100.0:
            nu_unc = 100.0
        exp_nu = np.exp(nu_unc)
        nu = float(49.0 * (exp_nu / (1.0 + exp_nu)) + 1.01)

    if error_type == 4:
        # Skewed-t: logistic mapping to (-0.99, 0.99)
        # Ref: figarch_itransform.m:59-62
        lam_unc = float(parameters[p + q + 3])  # 0-indexed from MATLAB p+q+4
        exp_lam = np.exp(lam_unc)
        lam = float(1.98 * (exp_lam / (1.0 + exp_lam)) - 0.99)

    # ------------------------------------------------------------------
    # Core parameter transforms
    # ------------------------------------------------------------------

    # omega: exp transform ensures omega > 0
    # Ref: figarch_itransform.m:74
    omega = float(np.exp(parameters[0]))

    # d: logistic/sigmoid maps to (0, 1)
    # Ref: figarch_itransform.m:76-82
    # In MATLAB 1-indexed: d at position 3 when p=1, position 2 when p=0
    # In Python 0-indexed: d at position 2 when p=1, position 1 when p=0
    if p:
        d_unc = float(parameters[2])
    else:
        d_unc = float(parameters[1])
    exp_d = np.exp(d_unc)
    d = float(exp_d / (1.0 + exp_d))

    # phi: logistic scaled to (0, (1-d)/2)
    # Ref: figarch_itransform.m:84-92
    phi: Optional[float] = None
    if p:
        # phi is at position 1 (0-indexed), position 2 in MATLAB
        phi_unc = float(parameters[1])
        exp_phi = np.exp(phi_unc)
        phi_sigmoid = float(exp_phi / (1.0 + exp_phi))
        phi = (1.0 - d) / 2.0 * phi_sigmoid
        phiplusd = phi + d
    else:
        phiplusd = d

    # beta: logistic scaled to (0, phi + d)
    # Ref: figarch_itransform.m:93-100
    beta: Optional[float] = None
    if q:
        # beta is at position 2+p (0-indexed), position 3+p in MATLAB
        beta_unc = float(parameters[2 + p])
        exp_beta = np.exp(beta_unc)
        beta_sigmoid = float(exp_beta / (1.0 + exp_beta))
        beta = beta_sigmoid * phiplusd

    # ------------------------------------------------------------------
    # Assemble output vector: [omega, phi, d, beta]
    # Ref: figarch_itransform.m:101 — parameters = [omega;phi;d;beta]
    # In MATLAB, empty [] elements are silently dropped during concatenation.
    # Pre-allocate with np.empty for known output size, then fill in order.
    # ------------------------------------------------------------------
    n_out = 1 + p + 1 + q  # omega + (phi if p) + d + (beta if q)
    trans_parameters = np.empty(n_out, dtype=np.float64)
    idx = 0
    trans_parameters[idx] = omega
    idx += 1
    if phi is not None:
        trans_parameters[idx] = phi
        idx += 1
    trans_parameters[idx] = d
    idx += 1
    if beta is not None:
        trans_parameters[idx] = beta

    return trans_parameters, nu, lam
