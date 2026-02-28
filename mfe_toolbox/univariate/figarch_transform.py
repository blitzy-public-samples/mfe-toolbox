"""
FIGARCH constrained-to-unconstrained parameter transformation.

Migrated from univariate/figarch_transform.m (Kevin Sheppard, University of Oxford).
Maps constrained FIGARCH(p,d,q) model parameters to unconstrained real-line space
for use with scipy.optimize.minimize during maximum likelihood estimation.

This is the forward transform, paired with figarch_itransform (inverse transform).
The logit (logistic) transform is used for parameters bounded in (0,1) or sub-intervals
thereof, ensuring the optimizer works in unconstrained space while the inverse transform
always recovers valid constrained parameters.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/12/2009
"""

import numpy as np


def figarch_transform(
    parameters: np.ndarray, p: int, q: int, error_type: int
) -> np.ndarray:
    """
    Transform constrained FIGARCH parameters to unconstrained space.

    Maps the constrained parameter space of the FIGARCH(p,d,q) model to an
    unconstrained real-line space suitable for numerical optimization. Each
    bounded parameter is transformed via a logit or log map so that the
    optimizer faces no box constraints.

    Parameters
    ----------
    parameters : np.ndarray
        Constrained parameter vector. The layout depends on (p, q) and
        error_type:

        - p=0, q=0: ``[omega, d, (nu), (lambda)]``
        - p=1, q=0: ``[omega, phi, d, (nu), (lambda)]``
        - p=0, q=1: ``[omega, d, beta, (nu), (lambda)]``
        - p=1, q=1: ``[omega, phi, d, beta, (nu), (lambda)]``

        Distribution parameters ``nu`` and ``lambda`` are appended only when
        required by *error_type* (2, 3, or 4 for ``nu``; 4 for ``lambda``).
    p : int
        FIGARCH phi (autoregressive) order. Must be 0 or 1.
    q : int
        FIGARCH beta (moving-average) order. Must be 0 or 1.
    error_type : int
        Error distribution type:

        - 1 = Normal (Gaussian)
        - 2 = Student's t
        - 3 = Generalized Error Distribution (GED)
        - 4 = Skewed t

    Returns
    -------
    np.ndarray
        Unconstrained (transformed) parameter vector preserving the same
        element ordering as the input. Every bounded parameter has been
        mapped to the full real line.

    Notes
    -----
    Transforms applied per parameter:

    +-----------+-------------------------+-------------------------------+
    | Parameter | Constraint              | Forward Transform             |
    +===========+=========================+===============================+
    | omega     | omega > 0               | log(omega)                    |
    +-----------+-------------------------+-------------------------------+
    | d         | 0 < d < 1               | log(d / (1 - d))              |
    +-----------+-------------------------+-------------------------------+
    | phi       | 0 < phi < (1 - d) / 2   | logit(phi / ((1 - d) / 2))    |
    +-----------+-------------------------+-------------------------------+
    | beta      | 0 < beta < phi + d      | logit(beta / (phi + d))       |
    +-----------+-------------------------+-------------------------------+
    | nu (t)    | nu > 2.01               | sqrt(nu - 2.01)               |
    +-----------+-------------------------+-------------------------------+
    | nu (GED)  | 1 < nu < 50             | logit((nu - 1) / 49)          |
    +-----------+-------------------------+-------------------------------+
    | lambda    | -0.995 < lambda < 0.995  | logit((lambda + 0.995)/1.99) |
    +-----------+-------------------------+-------------------------------+

    The fractional differencing parameter *d* must be strictly in (0, 1).
    The logistic (logit) transform is essential to maintain this constraint
    throughout optimization.

    References
    ----------
    Ref: figarch_transform.m — Kevin Sheppard, MFE Toolbox Version 4.0
    """
    # ----------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------
    if not isinstance(parameters, np.ndarray):
        raise ValueError("parameters must be a numpy ndarray.")
    if p not in (0, 1):
        raise ValueError("p must be 0 or 1.")
    if q not in (0, 1):
        raise ValueError("q must be 0 or 1.")
    if error_type not in (1, 2, 3, 4):
        raise ValueError("error_type must be 1, 2, 3, or 4.")

    # Determine expected parameter count
    # Base model: omega + d = 2, plus p (phi) and q (beta)
    num_core = 2 + p + q
    num_dist = 0
    if error_type in (2, 3, 4):
        num_dist += 1  # nu
    if error_type == 4:
        num_dist += 1  # lambda
    expected_length = num_core + num_dist
    if parameters.size < expected_length:
        raise ValueError(
            f"parameters has {parameters.size} elements but expected at least "
            f"{expected_length} for p={p}, q={q}, error_type={error_type}."
        )

    # Work on a float64 copy to avoid modifying the caller's array
    # Ref: figarch_transform.m — MATLAB passes by value; Python numpy by reference
    trans = np.array(parameters, dtype=np.float64).ravel().copy()

    # ----------------------------------------------------------------
    # Distribution parameter transforms (nu, lambda)
    # Ref: figarch_transform.m:41-61
    # Index of nu in MATLAB 1-based: p+q+3 → Python 0-based: p+q+2
    # ----------------------------------------------------------------
    nu_idx = p + q + 2  # 0-based index of nu in the parameter vector

    if error_type == 2 or error_type == 4:
        # Student's t or Skewed t: nu > 2.01
        # Ref: figarch_transform.m:43-45 — nu = sqrt(nu - 2.01)
        nu_val = trans[nu_idx]
        trans[nu_idx] = np.sqrt(nu_val - 2.01)
    elif error_type == 3:
        # GED: 1 < nu < 50
        # Ref: figarch_transform.m:46-49 — logistic transform mapping (1, 50) -> R
        nu_val = trans[nu_idx]
        temp = (nu_val - 1.0) / 49.0
        trans[nu_idx] = np.log(temp / (1.0 - temp))

    if error_type == 4:
        # Skewed t: lambda in (-0.995, 0.995)
        # Ref: figarch_transform.m:55-58 — logistic transform mapping (-0.995, 0.995) -> R
        lam_idx = nu_idx + 1
        lam_val = trans[lam_idx]
        temp = (lam_val + 0.995) / 1.99
        trans[lam_idx] = np.log(temp / (1.0 - temp))

    # ----------------------------------------------------------------
    # Core parameter transforms
    # Ref: figarch_transform.m:64-94
    # ----------------------------------------------------------------

    # omega > 0 → log(omega)
    # Ref: figarch_transform.m:67 — omegaTrans = log(parameters(1))
    trans[0] = np.log(trans[0])

    # Extract the constrained d value *before* we overwrite it.
    # In MATLAB (1-indexed): d is at index 3 if p==1, else index 2.
    # In Python (0-indexed): d is at index 2 if p==1, else index 1.
    # Ref: figarch_transform.m:69-73
    if p:
        d_idx = 2
    else:
        d_idx = 1
    d = parameters.ravel()[d_idx]  # read from original (untransformed) copy

    # d in (0, 1) → logit(d) = log(d / (1 - d))
    # Ref: figarch_transform.m:75 — dTrans = log(d/(1-d))
    trans[d_idx] = np.log(d / (1.0 - d))

    # phi transform (only when p == 1)
    # phi in (0, (1-d)/2) → logit(phi / ((1-d)/2))
    # Ref: figarch_transform.m:77-85
    if p:
        # Ref: figarch_transform.m:78 — phi = parameters(2)  (MATLAB 1-indexed)
        phi = parameters.ravel()[1]  # original constrained value
        upper_phi = (1.0 - d) / 2.0
        phi_scaled = phi / upper_phi
        # Ref: figarch_transform.m:80 — phiTrans = log(phiTrans/(1-phiTrans))
        trans[1] = np.log(phi_scaled / (1.0 - phi_scaled))
        phiplusd = phi + d
    else:
        # Ref: figarch_transform.m:84 — phiplusd = d when phi absent
        phiplusd = d

    # beta transform (only when q == 1)
    # beta in (0, phi + d) → logit(beta / (phi + d))
    # Ref: figarch_transform.m:86-93
    if q:
        # Ref: figarch_transform.m:87 — beta = parameters(3+p)  (MATLAB 1-indexed)
        # Python 0-indexed: beta at index 2+p
        beta_idx = 2 + p
        beta = parameters.ravel()[beta_idx]  # original constrained value
        beta_scaled = beta / phiplusd
        # Ref: figarch_transform.m:90 — betaTrans = log(betaTrans/(1-betaTrans))
        trans[beta_idx] = np.log(beta_scaled / (1.0 - beta_scaled))

    return trans
