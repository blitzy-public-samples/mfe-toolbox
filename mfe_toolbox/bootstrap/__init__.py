"""
MFE Toolbox bootstrap subpackage — resampling inference methods.

Provides block and stationary bootstrap index generators plus inference routines
for model comparison and predictive ability testing:

- **block_bootstrap**: Circular block bootstrap for dependent series
- **stationary_bootstrap**: Stationary bootstrap with geometric block lengths
- **bsds**: Bootstrap Data Snooping / Hansen-White Superior Predictive Ability
- **mcs**: Model Confidence Set of Hansen, Lunde and Nason

All 4 public functions are re-exported here for convenient access via
``from mfe_toolbox.bootstrap import block_bootstrap, stationary_bootstrap`` etc.

Per AAP Section 0.4.1: 1:1 migration from bootstrap/*.m (4 modules).
"""

# ---------------------------------------------------------------------------
# Bootstrap Index Generators
# ---------------------------------------------------------------------------
from mfe_toolbox.bootstrap.block_bootstrap import block_bootstrap
from mfe_toolbox.bootstrap.stationary_bootstrap import stationary_bootstrap

# ---------------------------------------------------------------------------
# Bootstrap Inference Routines
# ---------------------------------------------------------------------------
from mfe_toolbox.bootstrap.bsds import bsds
from mfe_toolbox.bootstrap.mcs import mcs

__all__ = [
    'block_bootstrap',
    'stationary_bootstrap',
    'bsds',
    'mcs',
]
