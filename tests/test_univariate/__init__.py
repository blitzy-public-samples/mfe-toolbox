"""Pytest test package for mfe_toolbox.univariate GARCH model families.

Contains parametrized unit, integration, and numerical parity tests for all 7
univariate GARCH model families:
- AGARCH/NAGARCH (Asymmetric/Nonlinear Asymmetric GARCH)
- APARCH (Asymmetric Power ARCH)
- EGARCH (Exponential GARCH)
- FIGARCH (Fractionally Integrated GARCH)
- HEAVY (High-frEquency-bAsed VolatilitY)
- IGARCH (Integrated GARCH)
- TARCH/GJR-GARCH (Threshold ARCH / Glosten-Jagannathan-Runkle GARCH)

All parity tests compare Python outputs against MATLAB reference fixtures
with tolerances atol=1e-6, rtol=1e-4 per AAP Section 0.7.1.
"""
