"""
MFE Toolbox utility subpackage — foundational helper functions.

Provides matrix parameterization, correlation/covariance transforms, HAC estimators,
numerical derivatives, date conversion, robust VCV, display, and plotting helpers.

This is the foundational subpackage imported by nearly all other subpackages in the
mfe_toolbox ecosystem. All 29 public utility functions are re-exported here for
convenient access via ``from mfe_toolbox.utility import vech, ivech, covnw`` etc.

Functional Groups
-----------------
Matrix parameterization:
    vech, ivech, chol2vec, vec2chol, corr_ivech, corr_vech

Correlation / Covariance transforms:
    cov2corr, phi2r, phi2u, r2phi, r2z, z2r

HAC / Covariance estimators:
    covnw, covvar, robustvcv

Numerical derivatives:
    gradient_2sided, hessian_2sided, hessian_2sided_nrows

Data processing:
    demean, standardize, mvstandardize, newlagmatrix

Date conversion:
    c2mdate, m2cdate, x2mdate

Display / Plotting:
    mprint, pltdens

Other utilities:
    dirod, randchar
"""

# ---------------------------------------------------------------------------
# Matrix parameterization
# ---------------------------------------------------------------------------
from mfe_toolbox.utility.vech import vech
from mfe_toolbox.utility.ivech import ivech
from mfe_toolbox.utility.chol2vec import chol2vec
from mfe_toolbox.utility.vec2chol import vec2chol
from mfe_toolbox.utility.corr_ivech import corr_ivech
from mfe_toolbox.utility.corr_vech import corr_vech

# ---------------------------------------------------------------------------
# Correlation / Covariance transforms
# ---------------------------------------------------------------------------
from mfe_toolbox.utility.cov2corr import cov2corr
from mfe_toolbox.utility.phi2r import phi2r
from mfe_toolbox.utility.phi2u import phi2u
from mfe_toolbox.utility.r2phi import r2phi
from mfe_toolbox.utility.r2z import r2z
from mfe_toolbox.utility.z2r import z2r

# ---------------------------------------------------------------------------
# HAC / Covariance estimators
# ---------------------------------------------------------------------------
from mfe_toolbox.utility.covnw import covnw
from mfe_toolbox.utility.covvar import covvar
from mfe_toolbox.utility.robustvcv import robustvcv

# ---------------------------------------------------------------------------
# Numerical derivatives
# ---------------------------------------------------------------------------
from mfe_toolbox.utility.gradient_2sided import gradient_2sided
from mfe_toolbox.utility.hessian_2sided import hessian_2sided
from mfe_toolbox.utility.hessian_2sided_nrows import hessian_2sided_nrows

# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------
from mfe_toolbox.utility.demean import demean
from mfe_toolbox.utility.standardize import standardize
from mfe_toolbox.utility.mvstandardize import mvstandardize
from mfe_toolbox.utility.newlagmatrix import newlagmatrix

# ---------------------------------------------------------------------------
# Date conversion
# ---------------------------------------------------------------------------
from mfe_toolbox.utility.c2mdate import c2mdate
from mfe_toolbox.utility.m2cdate import m2cdate
from mfe_toolbox.utility.x2mdate import x2mdate

# ---------------------------------------------------------------------------
# Display / Plotting
# ---------------------------------------------------------------------------
from mfe_toolbox.utility.mprint import mprint
from mfe_toolbox.utility.pltdens import pltdens

# ---------------------------------------------------------------------------
# Other utilities
# ---------------------------------------------------------------------------
from mfe_toolbox.utility.dirod import dirod
from mfe_toolbox.utility.randchar import randchar

__all__ = [
    # Matrix parameterization
    'vech',
    'ivech',
    'chol2vec',
    'vec2chol',
    'corr_ivech',
    'corr_vech',
    # Correlation / Covariance transforms
    'cov2corr',
    'phi2r',
    'phi2u',
    'r2phi',
    'r2z',
    'z2r',
    # HAC / Covariance estimators
    'covnw',
    'covvar',
    'robustvcv',
    # Numerical derivatives
    'gradient_2sided',
    'hessian_2sided',
    'hessian_2sided_nrows',
    # Data processing
    'demean',
    'standardize',
    'mvstandardize',
    'newlagmatrix',
    # Date conversion
    'c2mdate',
    'm2cdate',
    'x2mdate',
    # Display / Plotting
    'mprint',
    'pltdens',
    # Other utilities
    'dirod',
    'randchar',
]
