"""
MFE Toolbox distributions subpackage — statistical distribution functions.

Provides probability density functions (PDF), cumulative distribution functions
(CDF), inverse CDF (quantile) functions, log-likelihood evaluators, and random
variate generators for the following distribution families:

- **Generalized Error Distribution (GED)**: gedcdf, gedinv, gedpdf, gedloglik, gedrnd
- **Standardized Student's t**: stdtcdf, stdtinv, stdtpdf, stdtloglik, stdtrnd
- **Hansen's Skewed Student's t**: skewtcdf, skewtinv, skewtpdf, skewtloglik, skewtrnd
- **Normal / Multivariate Normal**: normloglik, mvnormloglik
- **Composite Likelihood**: composite_likelihood (Numba JIT-accelerated)
- **Shape Compatibility**: iscompatible (broadcasting validation)

All 19 public functions are re-exported here for convenient access via
``from mfe_toolbox.distributions import gedcdf, stdtpdf`` etc.

Per AAP Section 0.4.1: 1:1 migration from distributions/*.m (19 modules).
"""

# ---------------------------------------------------------------------------
# GED (Generalized Error Distribution)
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.gedcdf import gedcdf
from mfe_toolbox.distributions.gedinv import gedinv
from mfe_toolbox.distributions.gedpdf import gedpdf
from mfe_toolbox.distributions.gedloglik import gedloglik
from mfe_toolbox.distributions.gedrnd import gedrnd

# ---------------------------------------------------------------------------
# Standardized Student's t
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.stdtcdf import stdtcdf
from mfe_toolbox.distributions.stdtinv import stdtinv
from mfe_toolbox.distributions.stdtpdf import stdtpdf
from mfe_toolbox.distributions.stdtloglik import stdtloglik
from mfe_toolbox.distributions.stdtrnd import stdtrnd

# ---------------------------------------------------------------------------
# Hansen's Skewed Student's t
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.skewtcdf import skewtcdf
from mfe_toolbox.distributions.skewtinv import skewtinv
from mfe_toolbox.distributions.skewtpdf import skewtpdf
from mfe_toolbox.distributions.skewtloglik import skewtloglik

# ---------------------------------------------------------------------------
# Normal / Multivariate Normal
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.normloglik import normloglik
from mfe_toolbox.distributions.mvnormloglik import mvnormloglik

# ---------------------------------------------------------------------------
# Composite Likelihood (Numba JIT)
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.composite_likelihood import composite_likelihood

# ---------------------------------------------------------------------------
# Shape Compatibility
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.iscompatible import iscompatible

__all__ = [
    # GED
    'gedcdf',
    'gedinv',
    'gedpdf',
    'gedloglik',
    'gedrnd',
    # Standardized Student's t
    'stdtcdf',
    'stdtinv',
    'stdtpdf',
    'stdtloglik',
    'stdtrnd',
    # Hansen's Skewed Student's t
    'skewtcdf',
    'skewtinv',
    'skewtpdf',
    'skewtloglik',
    # Normal / Multivariate Normal
    'normloglik',
    'mvnormloglik',
    # Composite Likelihood
    'composite_likelihood',
    # Shape Compatibility
    'iscompatible',
]
