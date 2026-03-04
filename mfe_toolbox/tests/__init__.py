"""
Diagnostic statistical tests subpackage.

This subpackage provides statistical tests for distributional correctness,
normality, serial correlation, and heteroskedasticity. Migrated from the
MATLAB MFE Toolbox tests/ directory.

Functions
---------
berkowitz : Berkowitz density forecast test
jarquebera : Jarque-Bera normality test
kolmogorov : Kolmogorov-Smirnov distributional test
ljungbox : Ljung-Box serial correlation test
lmtest1 : LM test for serial correlation (heteroskedasticity-robust)
"""

from mfe_toolbox.tests.berkowitz import berkowitz
from mfe_toolbox.tests.jarquebera import jarquebera
from mfe_toolbox.tests.kolmogorov import kolmogorov
from mfe_toolbox.tests.ljungbox import ljungbox
from mfe_toolbox.tests.lmtest1 import lmtest1

__all__ = [
    'berkowitz',
    'jarquebera',
    'kolmogorov',
    'ljungbox',
    'lmtest1',
]
