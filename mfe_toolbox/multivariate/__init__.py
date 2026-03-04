"""
MFE Toolbox multivariate subpackage — multivariate GARCH models.

Provides BEKK, CCC-MVGARCH, DCC, GO-GARCH, Matrix GARCH, O-GARCH, RARCH,
RCC, RiskMetrics, and Scalar VT-VECH model implementations:

- **BEKK**: bekk_constraint, bekk_likelihood, bekk_parameter_transform, bekk_simulate
- **CCC-MVGARCH**: ccc_mvgarch, ccc_mvgarch_joint_likelihood, ccc_mvgarch_likelihood,
  ccc_mvgarch_simulate
- **DCC**: dcc_reconstruct_variance
- **GO-GARCH**: gogarch_likelihood
- **Matrix GARCH**: matrix_garch_display, matrix_garch_likelihood, matrix_garch_simulate
- **O-GARCH**: ogarch_likelihood
- **RARCH**: rarch_constraint, rarch_likelihood, rarch_parameter_transform, rarch_simulate
- **RCC**: rcc_constraint
- **RiskMetrics**: riskmetrics, riskmetrics2006
- **Scalar VT-VECH**: scalar_vt_vech_itransform, scalar_vt_vech_likelihood,
  scalar_vt_vech_simulate, scalar_vt_vech_transform

All modules are re-exported here for convenient access via
``from mfe_toolbox.multivariate import ccc_mvgarch, riskmetrics`` etc.

Per AAP Section 0.4.1: 1:1 migration from multivariate/*.m.
"""

# ---------------------------------------------------------------------------
# BEKK model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.bekk_constraint import bekk_constraint
from mfe_toolbox.multivariate.bekk_likelihood import bekk_likelihood
from mfe_toolbox.multivariate.bekk_parameter_transform import bekk_parameter_transform
from mfe_toolbox.multivariate.bekk_simulate import bekk_simulate

# ---------------------------------------------------------------------------
# CCC-MVGARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.ccc_mvgarch import ccc_mvgarch
from mfe_toolbox.multivariate.ccc_mvgarch_joint_likelihood import ccc_mvgarch_joint_likelihood
from mfe_toolbox.multivariate.ccc_mvgarch_likelihood import ccc_mvgarch_likelihood
from mfe_toolbox.multivariate.ccc_mvgarch_simulate import ccc_mvgarch_simulate

# ---------------------------------------------------------------------------
# DCC model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.dcc_reconstruct_variance import dcc_reconstruct_variance

# ---------------------------------------------------------------------------
# GO-GARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.gogarch_likelihood import gogarch_likelihood

# ---------------------------------------------------------------------------
# Matrix GARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.matrix_garch_display import matrix_garch_display
from mfe_toolbox.multivariate.matrix_garch_likelihood import matrix_garch_likelihood
from mfe_toolbox.multivariate.matrix_garch_simulate import matrix_garch_simulate

# ---------------------------------------------------------------------------
# O-GARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.ogarch_likelihood import ogarch_likelihood

# ---------------------------------------------------------------------------
# RARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.rarch_constraint import rarch_constraint
from mfe_toolbox.multivariate.rarch_likelihood import rarch_likelihood
from mfe_toolbox.multivariate.rarch_parameter_transform import rarch_parameter_transform
from mfe_toolbox.multivariate.rarch_simulate import rarch_simulate

# ---------------------------------------------------------------------------
# RCC model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.rcc_constraint import rcc_constraint

# ---------------------------------------------------------------------------
# RiskMetrics
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.riskmetrics import riskmetrics
from mfe_toolbox.multivariate.riskmetrics2006 import riskmetrics2006

# ---------------------------------------------------------------------------
# Scalar VT-VECH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.multivariate.scalar_vt_vech_itransform import scalar_vt_vech_itransform
from mfe_toolbox.multivariate.scalar_vt_vech_likelihood import scalar_vt_vech_likelihood
from mfe_toolbox.multivariate.scalar_vt_vech_simulate import scalar_vt_vech_simulate
from mfe_toolbox.multivariate.scalar_vt_vech_transform import scalar_vt_vech_transform

__all__ = [
    # BEKK
    'bekk_constraint',
    'bekk_likelihood',
    'bekk_parameter_transform',
    'bekk_simulate',
    # CCC-MVGARCH
    'ccc_mvgarch',
    'ccc_mvgarch_joint_likelihood',
    'ccc_mvgarch_likelihood',
    'ccc_mvgarch_simulate',
    # DCC
    'dcc_reconstruct_variance',
    # GO-GARCH
    'gogarch_likelihood',
    # Matrix GARCH
    'matrix_garch_display',
    'matrix_garch_likelihood',
    'matrix_garch_simulate',
    # O-GARCH
    'ogarch_likelihood',
    # RARCH
    'rarch_constraint',
    'rarch_likelihood',
    'rarch_parameter_transform',
    'rarch_simulate',
    # RCC
    'rcc_constraint',
    # RiskMetrics
    'riskmetrics',
    'riskmetrics2006',
    # Scalar VT-VECH
    'scalar_vt_vech_itransform',
    'scalar_vt_vech_likelihood',
    'scalar_vt_vech_simulate',
    'scalar_vt_vech_transform',
]
