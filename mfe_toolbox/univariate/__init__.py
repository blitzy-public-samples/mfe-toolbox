"""
MFE Toolbox — Univariate GARCH Models.

Implements seven families of univariate GARCH volatility models migrated from
the MATLAB MFE Toolbox (Version 4.0) ``univariate/`` directory:

* **AGARCH / NAGARCH** — Asymmetric GARCH (Engle 1990) and Nonlinear
  Asymmetric GARCH (Engle & Ng 1993).
* **APARCH** — Asymmetric Power ARCH (Ding, Granger & Engle 1993),
  nesting standard GARCH, GJR-GARCH, and absolute-value GARCH as
  special cases via the power parameter delta.
* **EGARCH** — Exponential GARCH (Nelson 1991) with log-variance
  recursion and leverage effects.
* **FIGARCH** — Fractionally Integrated GARCH (Baillie, Bollerslev &
  Mikkelsen 1996) capturing long-memory dependence in conditional
  variance.
* **HEAVY** — High-frEquency bAsed VolatilitY model (Shephard &
  Sheppard 2010) jointly modelling returns and realized measures.
* **IGARCH** — Integrated GARCH with unit-root constraint
  ``sum(alpha) + sum(beta) = 1``.
* **TARCH / GJR-GARCH** — Threshold ARCH (Zakoian 1994) and
  GJR-GARCH (Glosten, Jagannathan & Runkle 1993) with asymmetric
  leverage effects.

Each model family exposes a single public driver function through this
subpackage.  Internal helper modules (``*_core``, ``*_likelihood``,
``*_transform``, ``*_itransform``, ``*_display``, ``*_parameter_check``,
``*_starting_values``, ``*_simulate``) are accessible via their full
import path (e.g., ``from mfe_toolbox.univariate.tarch_core import
tarch_core``) but are **not** re-exported at the subpackage level.

Usage
-----
>>> from mfe_toolbox.univariate import tarch
>>> results = tarch(data, p=1, o=1, q=1)

Or import all public drivers at once:

>>> from mfe_toolbox.univariate import *  # imports agarch … tarch
"""

from mfe_toolbox.univariate.agarch import agarch
from mfe_toolbox.univariate.aparch import aparch
from mfe_toolbox.univariate.egarch import egarch
from mfe_toolbox.univariate.figarch import figarch
from mfe_toolbox.univariate.heavy import heavy
from mfe_toolbox.univariate.igarch import igarch
from mfe_toolbox.univariate.tarch import tarch

__all__ = [
    "agarch",
    "aparch",
    "egarch",
    "figarch",
    "heavy",
    "igarch",
    "tarch",
]
