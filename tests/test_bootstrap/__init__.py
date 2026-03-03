"""Pytest test suite for mfe_toolbox.bootstrap subpackage.

Tests cover all 4 bootstrap modules migrated from MATLAB:
- test_block_bootstrap.py — Circular block bootstrap (T=200, B=100, w=10)
- test_stationary_bootstrap.py — Politis-Romano stationary bootstrap (T=200, B=100, p=0.1)
- test_bsds.py — Hansen-White SPA/BSDS test with loss differentials
- test_mcs.py — Hansen-Lunde-Nason Model Confidence Set with loss matrix

All tests verify:
- Input validation error messages match MATLAB originals
- Output shapes and types are correct
- Statistical properties are preserved
- Numerical parity against MATLAB fixtures (atol=1e-6, rtol=1e-4)
- Bootstrap index generation reproducibility with seeded RNG
"""
