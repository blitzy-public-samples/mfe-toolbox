"""MFE Toolbox cross-sectional analysis subpackage.

Provides OLS regression with robust standard errors and PCA with multiple
normalization modes, migrated from the MATLAB MFE Toolbox crosssection/ module.

Functions
---------
ols : OLS regression with White heteroskedasticity-robust standard errors
pca : Principal Component Analysis with outer-product, covariance, and correlation modes
"""

from mfe_toolbox.crosssection.ols import ols
from mfe_toolbox.crosssection.pca import pca

__all__ = [
    'ols',
    'pca',
]
