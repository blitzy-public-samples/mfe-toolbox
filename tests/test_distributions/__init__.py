"""
Pytest test suite for mfe_toolbox.distributions subpackage.

Covers all 19 distribution modules:
- GED family: gedcdf, gedinv, gedloglik, gedpdf, gedrnd
- Standardized Student's t family: stdtcdf, stdtinv, stdtloglik, stdtpdf, stdtrnd
- Hansen's Skewed Student's t family: skewtcdf, skewtinv, skewtloglik, skewtpdf, skewtrnd
- Normal log-likelihood: normloglik
- Multivariate normal log-likelihood: mvnormloglik
- Composite likelihood (Numba JIT): composite_likelihood
- Shape compatibility checker: iscompatible
"""
