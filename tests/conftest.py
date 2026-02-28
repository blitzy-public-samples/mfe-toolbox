"""Shared test fixtures and configuration for MFE Toolbox test suite."""
import os
import pytest
import numpy as np

# Numerical parity tolerance constants
ATOL = 1e-6
RTOL = 1e-4

# Fixture directory
FIXTURE_DIR = os.environ.get(
    "MFE_FIXTURE_DIR",
    os.path.join(os.path.dirname(__file__), "fixtures")
)


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def fixture_dir():
    """Return the path to the test fixtures directory."""
    return FIXTURE_DIR
