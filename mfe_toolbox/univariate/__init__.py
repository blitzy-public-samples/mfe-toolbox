"""
MFE Toolbox univariate subpackage — univariate GARCH model families.

Provides seven GARCH model families (AGARCH, APARCH, EGARCH, FIGARCH, HEAVY,
IGARCH, TARCH) with their helper modules:

- **AGARCH**: agarch_core, agarch_display, agarch_itransform, agarch_likelihood,
  agarch_parameter_check, agarch_transform
- **APARCH**: aparch_core, aparch_display, aparch_itransform, aparch_likelihood,
  aparch_loglikelihood, aparch_parameter_check, aparch_transform
- **EGARCH**: egarch_core, egarch_display, egarch_itransform, egarch_likelihood,
  egarch_nlcon, egarch_parameter_check, egarch_transform
- **FIGARCH**: figarch, figarch_itransform, figarch_likelihood,
  figarch_parameter_check, figarch_starting_values, figarch_transform, figarch_weights
- **HEAVY**: heavy_likelihood, heavy_parameter_transform, heavy_simulate
- **IGARCH**: igarch_core, igarch_display, igarch_itransform, igarch_likelihood,
  igarch_parameter_check, igarch_transform
- **TARCH**: tarch_core, tarch_core_simple, tarch_display, tarch_itransform,
  tarch_likelihood, tarch_parameter_check, tarch_transform

All modules are re-exported here for convenient access via
``from mfe_toolbox.univariate import tarch_core, agarch_core`` etc.

Per AAP Section 0.4.1: 1:1 migration from univariate/*.m.
"""

# ---------------------------------------------------------------------------
# AGARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.univariate.agarch_core import agarch_core
from mfe_toolbox.univariate.agarch_display import agarch_display
from mfe_toolbox.univariate.agarch_itransform import agarch_itransform
from mfe_toolbox.univariate.agarch_likelihood import agarch_likelihood
from mfe_toolbox.univariate.agarch_parameter_check import agarch_parameter_check
from mfe_toolbox.univariate.agarch_transform import agarch_transform

# ---------------------------------------------------------------------------
# APARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.univariate.aparch_core import aparch_core
from mfe_toolbox.univariate.aparch_display import aparch_display
from mfe_toolbox.univariate.aparch_itransform import aparch_itransform
from mfe_toolbox.univariate.aparch_likelihood import aparch_likelihood
from mfe_toolbox.univariate.aparch_parameter_check import aparch_parameter_check
from mfe_toolbox.univariate.aparch_transform import aparch_transform

# ---------------------------------------------------------------------------
# EGARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.univariate.egarch_core import egarch_core
from mfe_toolbox.univariate.egarch_display import egarch_display
from mfe_toolbox.univariate.egarch_itransform import egarch_itransform
from mfe_toolbox.univariate.egarch_likelihood import egarch_likelihood
from mfe_toolbox.univariate.egarch_nlcon import egarch_nlcon
from mfe_toolbox.univariate.egarch_parameter_check import egarch_parameter_check
from mfe_toolbox.univariate.egarch_transform import egarch_transform

# ---------------------------------------------------------------------------
# FIGARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.univariate.figarch import figarch
from mfe_toolbox.univariate.figarch_itransform import figarch_itransform
from mfe_toolbox.univariate.figarch_likelihood import figarch_likelihood
from mfe_toolbox.univariate.figarch_parameter_check import figarch_parameter_check
from mfe_toolbox.univariate.figarch_starting_values import figarch_starting_values
from mfe_toolbox.univariate.figarch_transform import figarch_transform
from mfe_toolbox.univariate.figarch_weights import figarch_weights

# ---------------------------------------------------------------------------
# HEAVY model family
# ---------------------------------------------------------------------------
from mfe_toolbox.univariate.heavy_likelihood import heavy_likelihood
from mfe_toolbox.univariate.heavy_parameter_transform import heavy_parameter_transform
from mfe_toolbox.univariate.heavy_simulate import heavy_simulate

# ---------------------------------------------------------------------------
# IGARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.univariate.igarch_core import igarch_core
from mfe_toolbox.univariate.igarch_display import igarch_display
from mfe_toolbox.univariate.igarch_itransform import igarch_itransform
from mfe_toolbox.univariate.igarch_likelihood import igarch_likelihood
from mfe_toolbox.univariate.igarch_parameter_check import igarch_parameter_check
from mfe_toolbox.univariate.igarch_transform import igarch_transform

# ---------------------------------------------------------------------------
# TARCH / GJR-GARCH model family
# ---------------------------------------------------------------------------
from mfe_toolbox.univariate.tarch_core import tarch_core
from mfe_toolbox.univariate.tarch_core_simple import tarch_core_simple
from mfe_toolbox.univariate.tarch_display import tarch_display
from mfe_toolbox.univariate.tarch_itransform import tarch_itransform
from mfe_toolbox.univariate.tarch_likelihood import tarch_likelihood
from mfe_toolbox.univariate.tarch_parameter_check import tarch_parameter_check
from mfe_toolbox.univariate.tarch_transform import tarch_transform

__all__ = [
    # AGARCH
    'agarch_core',
    'agarch_display',
    'agarch_itransform',
    'agarch_likelihood',
    'agarch_parameter_check',
    'agarch_transform',
    # APARCH
    'aparch_core',
    'aparch_display',
    'aparch_itransform',
    'aparch_likelihood',
    'aparch_parameter_check',
    'aparch_transform',
    # EGARCH
    'egarch_core',
    'egarch_display',
    'egarch_itransform',
    'egarch_likelihood',
    'egarch_nlcon',
    'egarch_parameter_check',
    'egarch_transform',
    # FIGARCH
    'figarch',
    'figarch_itransform',
    'figarch_likelihood',
    'figarch_parameter_check',
    'figarch_starting_values',
    'figarch_transform',
    'figarch_weights',
    # HEAVY
    'heavy_likelihood',
    'heavy_parameter_transform',
    'heavy_simulate',
    # IGARCH
    'igarch_core',
    'igarch_display',
    'igarch_itransform',
    'igarch_likelihood',
    'igarch_parameter_check',
    'igarch_transform',
    # TARCH
    'tarch_core',
    'tarch_core_simple',
    'tarch_display',
    'tarch_itransform',
    'tarch_likelihood',
    'tarch_parameter_check',
    'tarch_transform',
]
