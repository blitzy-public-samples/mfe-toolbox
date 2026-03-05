"""
MFE Toolbox — Multivariate GARCH Models

Provides 11 multivariate GARCH model families for conditional covariance
estimation, spanning a wide range of parameterisation strategies:

- **BEKK**: Full, Diagonal, and Scalar BEKK(p,o,q) models with symmetric and
  asymmetric dynamics (Engle & Kroner, 1995).
- **CCC-MVGARCH**: Constant Conditional Correlation with per-series TARCH/GJR
  conditional variances (Bollerslev, 1990).
- **DCC/ADCC**: Dynamic Conditional Correlation with symmetric and asymmetric
  correlation dynamics and optional composite likelihood (Engle, 2002).
- **GO-GARCH**: Generalized Orthogonal GARCH with PCA-based decomposition and
  optional Givens rotation angles (van der Weide, 2002).
- **Matrix GARCH**: Hadamard product multivariate GARCH with full K×K parameter
  matrices for covariance dynamics.
- **O-MVGARCH**: Orthogonal/Factor GARCH using PCA-based dimension reduction
  with per-factor univariate GARCH fitting (Carroll, 2000).
- **RARCH**: Rotated ARCH with Scalar, Common Persistence, and Diagonal
  parameterisations in the rotated covariance space (Noureldin, Shephard &
  Sheppard).
- **RCC**: Regime-switching Conditional Correlation with RARCH-style dynamics
  in the correlation space and TARCH conditional variances.
- **RiskMetrics**: EWMA exponential smoothing covariance estimation — pure
  filtering with no optimisation required (J.P. Morgan, 1996).
- **RiskMetrics 2006**: Multi-frequency EWMA covariance estimation using
  weighted averages across multiple half-lives.
- **Scalar VT-VECH**: Scalar Variance Targeting VECH with symmetric/asymmetric
  dynamics and optional composite likelihood.

All model drivers are re-exported here for convenient access::

    from mfe_toolbox.multivariate import bekk, dcc, ccc_mvgarch

Helper modules (likelihoods, constraints, transforms, simulations, displays)
are not imported at the subpackage level — import them explicitly when needed,
e.g.::

    from mfe_toolbox.multivariate.bekk_likelihood import bekk_likelihood

Per AAP Section 0.4.1: 1:1 migration from multivariate/*.m driver functions.
"""

# ---------------------------------------------------------------------------
# Primary driver function imports — one per model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.bekk import bekk
from mfe_toolbox.multivariate.ccc_mvgarch import ccc_mvgarch
from mfe_toolbox.multivariate.dcc import dcc
from mfe_toolbox.multivariate.gogarch import gogarch
from mfe_toolbox.multivariate.matrix_garch import matrix_garch
from mfe_toolbox.multivariate.o_mvgarch import o_mvgarch
from mfe_toolbox.multivariate.rarch import rarch
from mfe_toolbox.multivariate.rcc import rcc
from mfe_toolbox.multivariate.riskmetrics import riskmetrics
from mfe_toolbox.multivariate.riskmetrics2006 import riskmetrics2006
from mfe_toolbox.multivariate.scalar_vt_vech import scalar_vt_vech

__all__ = [
    "bekk",
    "ccc_mvgarch",
    "dcc",
    "gogarch",
    "matrix_garch",
    "o_mvgarch",
    "rarch",
    "rcc",
    "riskmetrics",
    "riskmetrics2006",
    "scalar_vt_vech",
]
