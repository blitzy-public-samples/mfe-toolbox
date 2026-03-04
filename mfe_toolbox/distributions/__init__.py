"""
Statistical distribution functions for the MFE Toolbox.

Provides GED, standardized Student's t, skewed Student's t (Hansen 1994),
normal, multivariate normal distributions with PDF, CDF, quantile,
log-likelihood, and random variate generation, plus composite likelihood
and shape compatibility validation.

Distribution Families
---------------------
**Generalized Error Distribution (GED)**
    gedcdf, gedinv, gedloglik, gedpdf, gedrnd

**Standardized Student's t** (unit variance)
    stdtcdf, stdtinv, stdtloglik, stdtpdf, stdtrnd

**Hansen's (1994) Skewed Student's t**
    skewtcdf, skewtinv, skewtloglik, skewtpdf, skewtrnd

**Normal / Multivariate Normal**
    normloglik, mvnormloglik

**Composite Likelihood** (Numba JIT-accelerated pairwise bivariate evaluator)
    composite_likelihood

**Shape Compatibility**
    iscompatible

All 19 public functions are re-exported here for convenient access::

    from mfe_toolbox.distributions import gedcdf, stdtpdf, composite_likelihood

Migrated from: distributions/*.m (19 MATLAB modules, MFE Toolbox Version 4.0)
"""

# ---------------------------------------------------------------------------
# Composite Likelihood (Numba JIT)
# Ref: distributions/composite_likelihood.m + mex_source/composite_likelihood.c
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.composite_likelihood import composite_likelihood

# ---------------------------------------------------------------------------
# GED (Generalized Error Distribution)
# Ref: distributions/gedcdf.m, gedinv.m, gedloglik.m, gedpdf.m, gedrnd.m
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.gedcdf import gedcdf
from mfe_toolbox.distributions.gedinv import gedinv
from mfe_toolbox.distributions.gedloglik import gedloglik
from mfe_toolbox.distributions.gedpdf import gedpdf
from mfe_toolbox.distributions.gedrnd import gedrnd

# ---------------------------------------------------------------------------
# Shape Compatibility
# Ref: distributions/iscompatible.m
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.iscompatible import iscompatible

# ---------------------------------------------------------------------------
# Normal / Multivariate Normal
# Ref: distributions/mvnormloglik.m, normloglik.m
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.mvnormloglik import mvnormloglik
from mfe_toolbox.distributions.normloglik import normloglik

# ---------------------------------------------------------------------------
# Hansen's (1994) Skewed Student's t
# Ref: distributions/skewtcdf.m, skewtinv.m, skewtloglik.m, skewtpdf.m, skewtrnd.m
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.skewtcdf import skewtcdf
from mfe_toolbox.distributions.skewtinv import skewtinv
from mfe_toolbox.distributions.skewtloglik import skewtloglik
from mfe_toolbox.distributions.skewtpdf import skewtpdf
from mfe_toolbox.distributions.skewtrnd import skewtrnd

# ---------------------------------------------------------------------------
# Standardized Student's t (unit variance)
# Ref: distributions/stdtcdf.m, stdtinv.m, stdtloglik.m, stdtpdf.m, stdtrnd.m
# ---------------------------------------------------------------------------
from mfe_toolbox.distributions.stdtcdf import stdtcdf
from mfe_toolbox.distributions.stdtinv import stdtinv
from mfe_toolbox.distributions.stdtloglik import stdtloglik
from mfe_toolbox.distributions.stdtpdf import stdtpdf
from mfe_toolbox.distributions.stdtrnd import stdtrnd

# ---------------------------------------------------------------------------
# Public API — all 19 functions in alphabetical order
# ---------------------------------------------------------------------------
__all__ = [
    'composite_likelihood',
    'gedcdf',
    'gedinv',
    'gedloglik',
    'gedpdf',
    'gedrnd',
    'iscompatible',
    'mvnormloglik',
    'normloglik',
    'skewtcdf',
    'skewtinv',
    'skewtloglik',
    'skewtpdf',
    'skewtrnd',
    'stdtcdf',
    'stdtinv',
    'stdtloglik',
    'stdtpdf',
    'stdtrnd',
]
