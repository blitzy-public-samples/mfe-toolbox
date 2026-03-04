"""
Pytest tests for ``mfe_toolbox.utility.pltdens`` — nonparametric kernel density
estimation using binning + FFT convolution.

Tests cover:
- Gaussian, Epanechnikov, Biweight, and Triangular kernel types
- Silverman rule-of-thumb default bandwidth (with ``ddof=1`` matching MATLAB)
- Custom bandwidth passthrough
- Return type and tuple structure validation
- Density non-negativity (accounting for FFT rounding)
- Density integrates-to-one property
- Grid size (256 points) and output shape verification
- Positive constraint with reflection boundary correction
- Error handling for negative data with ``positive=True``
- MATLAB fixture parity using ``assert_allclose(atol=1e-6, rtol=1e-4)``

Source reference: ``utility/pltdens.m`` (86 lines, Anders Holtsberg, 18-Nov-1993).

Notes
-----
- ``matplotlib.use('Agg')`` is called before any ``pyplot`` import to prevent
  display windows during automated test execution.
- MATLAB ``std(x)`` uses ``ddof=1`` (sample standard deviation) by default;
  bandwidth tests use ``np.std(x, ddof=1)`` to match.
- ``np.trapezoid`` is used for numerical integration (``np.trapz`` removed in
  NumPy 2.x).
"""

import os

import matplotlib
matplotlib.use("Agg")  # Must precede any pyplot import — prevents display during tests
import matplotlib.pyplot as plt  # noqa: E402 — after backend set

import numpy as np
import pytest

from mfe_toolbox.utility.pltdens import pltdens

# Import shared tolerance constants and helpers from conftest.py
# (auto-discovered by pytest; explicit import for direct use)
from tests.conftest import ATOL, RTOL, assert_allclose, load_fixture_npy

# ---------------------------------------------------------------------------
# Fixture Directory
# ---------------------------------------------------------------------------
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "utility")

# ---------------------------------------------------------------------------
# Module-Level Test Data
# ---------------------------------------------------------------------------
# Seeded random generator for reproducible test data
_RNG = np.random.default_rng(42)

# Standard normal sample (500 points) — reused across basic tests
_NORMAL_DATA = _RNG.standard_normal(500)

# Positive-only data (lognormal) for positive constraint tests
_POSITIVE_DATA = np.exp(_RNG.standard_normal(200) * 0.5 + 1.0)


# ---------------------------------------------------------------------------
# Cleanup Fixture — close matplotlib figures after each test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _cleanup_plots():
    """Close all matplotlib figures after each test to prevent resource leaks."""
    yield
    plt.close("all")


# ============================================================================
# 3.1 Basic Functionality Tests
# ============================================================================


def test_pltdens_gaussian_kernel():
    """Call pltdens with kernel=1 (Gaussian) on normal data.

    Verifies:
    - Returns a tuple of 3 elements.
    - Density values are finite and have a plausible bell-curve shape.

    Ref: pltdens.m:55-56 — Gaussian kernel: ``exp(-0.5*(xk/h).^2)``
    """
    result = pltdens(_NORMAL_DATA, kernel=1)

    # Must return tuple of length 3
    assert isinstance(result, tuple), "pltdens must return a tuple"
    assert len(result) == 3, "pltdens must return exactly 3 elements"

    h, f, xx = result

    # Density values must be finite
    assert np.all(np.isfinite(f)), "Density values contain non-finite entries"
    assert np.all(np.isfinite(xx)), "Grid values contain non-finite entries"

    # Density should have a plausible maximum (bell-curve shape for normal data)
    assert f.max() > 0.1, "Gaussian density peak is unreasonably low"
    assert f.max() < 2.0, "Gaussian density peak is unreasonably high"


def test_pltdens_default_bandwidth():
    """Call pltdens(x) with no ``h`` argument and verify returned bandwidth.

    CRITICAL: MATLAB ``std(x)`` uses ``ddof=1`` by default (sample standard
    deviation). Python ``np.std(x)`` uses ``ddof=0`` by default.  The
    bandwidth formula MUST use ``ddof=1`` to match MATLAB.

    Ref: pltdens.m:35-36 — ``h = 1.06 * std(x) * n^(-1/5)``
    """
    x = _NORMAL_DATA.copy()
    n = len(x)

    # Expected Silverman bandwidth with ddof=1 matching MATLAB std()
    # Ref: pltdens.m:36 — Silverman page 45
    expected_h = 1.06 * np.std(x, ddof=1) * n ** (-1.0 / 5.0)

    h, _f, _xx = pltdens(x)

    # Use tight tolerance — bandwidth is a deterministic scalar computation
    np.testing.assert_allclose(
        h,
        expected_h,
        atol=1e-12,
        err_msg=(
            "Default bandwidth does not match Silverman rule: "
            "h = 1.06 * std(x, ddof=1) * n^(-1/5)"
        ),
    )


def test_pltdens_custom_bandwidth():
    """Call pltdens(x, h=0.5) and verify returned ``h`` equals 0.5 exactly.

    Ref: pltdens.m:35 — ``if isempty(h) ... end`` — when h is provided, use it as-is.
    """
    h, _f, _xx = pltdens(_NORMAL_DATA, h=0.5)

    assert h == 0.5, f"Custom bandwidth not returned exactly: got {h}, expected 0.5"


def test_pltdens_returns_tuple():
    """Call pltdens and verify return type is a tuple of (float, ndarray, ndarray).

    Ref: pltdens.m:1 — ``function [h,f,xx] = plotdens(x,h,positive,kernel)``
    """
    result = pltdens(_NORMAL_DATA)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3 elements, got {len(result)}"

    h, f, xx = result

    # h should be a Python float or numeric scalar
    assert isinstance(h, (float, int, np.floating)), (
        f"Bandwidth h should be scalar, got {type(h)}"
    )

    # f and xx should be numpy arrays
    assert isinstance(f, np.ndarray), f"Density f should be ndarray, got {type(f)}"
    assert isinstance(xx, np.ndarray), f"Grid xx should be ndarray, got {type(xx)}"


# ============================================================================
# 3.2 Density Property Tests
# ============================================================================


@pytest.mark.parametrize("kernel", [1, 2, 3, 4])
def test_pltdens_density_nonnegative(kernel):
    """For all kernel types, verify density values f >= 0.

    Non-Gaussian kernels can produce tiny negative values (~1e-17) from FFT
    rounding, so we allow a small tolerance below zero.

    Ref: pltdens.m:55-65 — kernel selection block
    """
    _h, f, _xx = pltdens(_NORMAL_DATA, kernel=kernel)

    # Allow tiny negative values from FFT numerical rounding
    assert np.all(f >= -1e-10), (
        f"Kernel {kernel}: density has values below -1e-10; "
        f"min = {f.min():.2e}"
    )


def test_pltdens_density_integrates_to_one():
    """Verify ``np.trapezoid(f, xx)`` ≈ 1.0 for Gaussian kernel.

    The 256-point grid with extended support should give an integral very
    close to 1.0.  We use a wider tolerance (atol=0.05) to account for
    discretization effects, though in practice the integral is extremely
    close to 1.0 for sufficient data.

    Ref: pltdens.m:66 — ``K = K / (sum(K)*d*n)`` ensures proper normalization.
    """
    _h, f, xx = pltdens(_NORMAL_DATA)

    integral = np.trapezoid(f, xx)
    np.testing.assert_allclose(
        integral,
        1.0,
        atol=0.05,
        err_msg=f"Density integral = {integral}, expected ≈ 1.0",
    )


def test_pltdens_grid_size():
    """Verify the evaluation grid has exactly 256 points.

    Ref: pltdens.m:44 — ``gridsize = 256``
    """
    _h, _f, xx = pltdens(_NORMAL_DATA)

    assert len(xx) == 256, f"Grid should have 256 points, got {len(xx)}"


def test_pltdens_output_shapes():
    """Verify f.shape == xx.shape == (256,).

    Both the density values and the support grid must be 1-D arrays of
    length 256.

    Ref: pltdens.m:44-45,68 — gridsize=256, xx is linspace, f is real part of IFFT
    """
    _h, f, xx = pltdens(_NORMAL_DATA)

    assert f.shape == (256,), f"Density shape should be (256,), got {f.shape}"
    assert xx.shape == (256,), f"Grid shape should be (256,), got {xx.shape}"


# ============================================================================
# 3.3 Kernel Type Tests
# ============================================================================


def test_pltdens_epanechnikov():
    """Call with kernel=2 (Epanechnikov), verify non-negative and integrates near 1.

    Ref: pltdens.m:57-58 — ``K = max(0, 1-(xk/h).^2/5)``
    """
    _h, f, xx = pltdens(_NORMAL_DATA, kernel=2)

    # Non-negative (with FFT tolerance)
    assert np.all(f >= -1e-10), (
        f"Epanechnikov density has values below -1e-10; min = {f.min():.2e}"
    )

    # Integrates to approximately 1
    integral = np.trapezoid(f, xx)
    np.testing.assert_allclose(
        integral, 1.0, atol=0.05,
        err_msg=f"Epanechnikov integral = {integral}, expected ≈ 1.0",
    )


def test_pltdens_biweight():
    """Call with kernel=3 (Biweight), verify non-negative and integrates near 1.

    Ref: pltdens.m:59-61 — Biweight kernel with ``c = sqrt(1/7)``
    """
    _h, f, xx = pltdens(_NORMAL_DATA, kernel=3)

    # Non-negative (with FFT tolerance)
    assert np.all(f >= -1e-10), (
        f"Biweight density has values below -1e-10; min = {f.min():.2e}"
    )

    # Integrates to approximately 1
    integral = np.trapezoid(f, xx)
    np.testing.assert_allclose(
        integral, 1.0, atol=0.05,
        err_msg=f"Biweight integral = {integral}, expected ≈ 1.0",
    )


def test_pltdens_triangular():
    """Call with kernel=4 (Triangular), verify non-negative and integrates near 1.

    Ref: pltdens.m:62-64 — Triangular kernel with ``c = sqrt(1/6)``
    """
    _h, f, xx = pltdens(_NORMAL_DATA, kernel=4)

    # Non-negative (with FFT tolerance)
    assert np.all(f >= -1e-10), (
        f"Triangular density has values below -1e-10; min = {f.min():.2e}"
    )

    # Integrates to approximately 1
    integral = np.trapezoid(f, xx)
    np.testing.assert_allclose(
        integral, 1.0, atol=0.05,
        err_msg=f"Triangular integral = {integral}, expected ≈ 1.0",
    )


# ============================================================================
# 3.4 Positive Constraint Tests
# ============================================================================


def test_pltdens_positive_flag():
    """Call with positive=True on positive data.

    Verifies:
    - Density is 0 (or near-zero) for grid points xx < 0.
    - The reflection boundary correction folds negative-domain mass onto
      the positive axis.

    Ref: pltdens.m:69-74 — positive boundary correction block
    """
    _h, f, xx = pltdens(_POSITIVE_DATA, positive=True)

    # All density values at negative grid points should be zero
    neg_mask = xx < 0.0
    if np.any(neg_mask):
        assert np.all(np.abs(f[neg_mask]) < 1e-10), (
            f"Density should be ~0 for xx < 0 when positive=True; "
            f"max |f| in negative region = {np.abs(f[neg_mask]).max():.2e}"
        )


def test_pltdens_positive_negative_raises():
    """Call with positive=True on data containing negative values.

    Verifies ValueError is raised with appropriate message.

    Ref: pltdens.m:38-39 — ``error('There is a negative element in X')``
    """
    # Data with negative values
    x_with_neg = np.array([1.0, 2.0, -0.5, 3.0, 4.0])

    with pytest.raises(ValueError, match="negative element"):
        pltdens(x_with_neg, positive=True)


# ============================================================================
# 3.5 Fixture Parity Test
# ============================================================================


# Check if fixture file exists for skipif decorator
_FIXTURE_PATH = os.path.join(FIXTURE_DIR, "pltdens.npy")


@pytest.mark.skipif(
    not os.path.exists(_FIXTURE_PATH),
    reason=f"Fixture file not found: {_FIXTURE_PATH}",
)
@pytest.mark.parametrize(
    "case_name",
    [
        "gaussian_default_1000",
        "gaussian_default_100",
        "gaussian_bw05_100",
        "epanechnikov_100",
        "biweight_100",
        "triangular_100",
        "gaussian_positive_100",
        "gaussian_default_500",
    ],
)
def test_pltdens_fixture_parity(case_name):
    """Compare Python pltdens output against MATLAB-generated reference fixture.

    Loads test cases from ``tests/fixtures/utility/pltdens.npy`` which contains
    input data, expected bandwidth, expected grid points, and expected density
    values for each kernel type and configuration.

    Tolerance: ``atol=1e-6, rtol=1e-4`` per AAP Section 0.7.1.

    Ref: pltdens.m — full function; fixture generated by exercising MATLAB
    pltdens with identical inputs and saving outputs.
    """
    # Load fixture dictionary
    fixture = np.load(_FIXTURE_PATH, allow_pickle=True).item()

    if case_name not in fixture:
        pytest.skip(f"Test case '{case_name}' not found in fixture file")

    tc = fixture[case_name]

    # Extract test case parameters
    input_data = tc["input_data"]
    expected_h = tc["bandwidth"]
    expected_xx = tc["grid_points"]
    expected_f = tc["density_values"]
    kernel = tc["kernel"]
    positive = bool(tc["positive"])

    # Determine bandwidth argument: if case uses custom bandwidth, pass it
    # "gaussian_bw05_100" uses h=0.5 (custom bandwidth)
    if "bw05" in case_name:
        h_arg = 0.5
    else:
        h_arg = None

    # Run pltdens with the fixture's input data
    h, f, xx = pltdens(input_data, h=h_arg, positive=positive, kernel=kernel)

    # Verify bandwidth parity
    assert_allclose(
        np.array([h]),
        np.array([expected_h]),
        atol=ATOL,
        rtol=RTOL,
        err_msg=f"[{case_name}] Bandwidth mismatch",
    )

    # Verify grid points parity
    assert_allclose(
        xx,
        expected_xx,
        atol=ATOL,
        rtol=RTOL,
        err_msg=f"[{case_name}] Grid points mismatch",
    )

    # Verify density values parity
    assert_allclose(
        f,
        expected_f,
        atol=ATOL,
        rtol=RTOL,
        err_msg=f"[{case_name}] Density values mismatch",
    )
