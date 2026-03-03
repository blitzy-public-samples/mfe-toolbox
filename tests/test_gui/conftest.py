"""GUI-specific pytest fixtures for MFE Toolbox PyQt6 test suite.

Provides:
- Sample time series data for ARMAX estimation
- Mock estimation results for viewer testing
- SSIM-based image comparison helper for plot snapshot testing
- Common widget interaction helpers via pytest-qt

Per AAP Section 0.7.1:
- Coverage threshold >=80% for GUI modules
- All GUIDE callbacks exercised via pytest-qt
- Plot output compared via saved .png snapshots with SSIM >= 0.95
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Conditional PyQt6 imports — skip tests gracefully if PyQt6 not installed.
# When pytest collects this conftest, importorskip will cause all tests in
# tests/test_gui/ to be skipped if PyQt6 is not available in the environment.
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402


# ---------------------------------------------------------------------------
# SSIM Threshold Constant
# ---------------------------------------------------------------------------
# Per AAP Section 0.7.1: Plot output compared via saved .png snapshots
# with perceptual diff tolerance SSIM >= 0.95
SSIM_THRESHOLD: float = 0.95


# ---------------------------------------------------------------------------
# QApplication Session Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp_cls() -> QApplication:
    """Provide a QApplication instance for the test session.

    pytest-qt typically manages this via the ``qtbot`` fixture, but this
    fixture serves as a safety fallback to guarantee a QApplication singleton
    exists before any widget construction.  Session-scoped because
    QApplication must be a singleton in any Qt process.

    Returns
    -------
    QApplication
        The existing QApplication singleton, or a newly created one.
    """
    app = QApplication.instance()
    if app is None:
        # Ref: QApplication requires at least an empty argv list.
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_y() -> np.ndarray:
    """Generate sample time series data for ARMAX GUI testing.

    Ref: GUI/ARMAX.m:62 — ``handles.y = varargin{1}``
    Uses a seeded random generator for full reproducibility across test
    runs.  Returns a T=500 demeaned Gaussian series that mirrors the kind
    of data the ARMAX GUI expects as its primary input.

    Returns
    -------
    np.ndarray
        Shape ``(500,)`` mean-zero Gaussian series.
    """
    rng = np.random.default_rng(42)
    T: int = 500
    y: np.ndarray = rng.standard_normal(T)
    # Ref: GUI/ARMAX.m:64 — ACF_lags = min(24, floor(length(handles.y)/8))
    # Demeaning mirrors standard pre-processing applied to financial returns.
    y = y - y.mean()
    return y


@pytest.fixture
def sample_results(sample_y: np.ndarray) -> list[dict[str, Any]]:
    """Generate mock estimation results matching ARMAX_viewer's expected structure.

    Ref: GUI/ARMAX_viewer.m:59 — ``handles.Results = varargin{1}``
    Ref: GUI/ARMAX.m:239-300 — model result dict structure stored in
    ``handles.models{modelno}``

    Each result dict contains the following keys, mirroring the MATLAB struct
    fields stored by ``estimate_pushbutton_Callback`` in ARMAX.m:

    - StringID : str — Model label, e.g., ``"ARMA(2,1) w/ constant"``
    - parameters : ndarray — Parameter estimates
    - errors : ndarray — Residuals (T-vector)
    - ll : float — Log-likelihood
    - seregression : float — Regression standard error (sigma^2)
    - diagnostics : dict — Model diagnostics
    - covariance : ndarray — VCV matrix (K x K)
    - likelihoods : ndarray — Per-observation log-likelihoods (T-vector)
    - scores : ndarray — Score matrix (T x K)
    - ARlags : ndarray — AR lag indices
    - MAlags : ndarray — MA lag indices
    - constant : int — 1 if constant included, 0 otherwise
    - StdErr : ndarray — Standard errors (K-vector)
    - Tstat : ndarray — T-statistics (K-vector)
    - Pval : ndarray — P-values (K-vector)
    - AIC : float — Akaike Information Criterion
    - BIC : float — Bayesian Information Criterion
    - K : int — Number of parameters

    Parameters
    ----------
    sample_y : np.ndarray
        The sample time series from the ``sample_y`` fixture (used for
        length ``T``).

    Returns
    -------
    list[dict[str, Any]]
        List of two model result dictionaries.
    """
    rng = np.random.default_rng(123)
    T: int = len(sample_y)

    results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Model 1: ARMA(1,0) with constant
    # Ref: GUI/ARMAX.m:278-295 — StringID built as "ARMA(<maxAR>,<maxMA>)"
    # ------------------------------------------------------------------
    K1: int = 2  # constant + 1 AR parameter
    params1: np.ndarray = np.array([0.1, 0.3])  # constant, phi_1
    errors1: np.ndarray = rng.standard_normal(T)
    # Ref: GUI/ARMAX.m:244-247 — covariance selected by InferenceMethod
    vcv1: np.ndarray = np.eye(K1) * 0.01
    # Ref: GUI/ARMAX.m:260 — StdErr = sqrt(diag(covariance))
    stde1: np.ndarray = np.sqrt(np.diag(vcv1))
    # Ref: GUI/ARMAX.m:261 — Tstat = parameters ./ StdErr
    tstat1: np.ndarray = params1 / stde1
    # Ref: GUI/ARMAX.m:262 — Pval = 2 - 2*normcdf(abs(Tstat))
    pval1: np.ndarray = np.array([0.01, 0.001])

    results.append({
        'StringID': 'ARMA(1,0) w/ constant',
        'parameters': params1,
        'errors': errors1,
        'll': -650.0,
        'seregression': float(np.var(errors1)),
        'diagnostics': {},
        'covariance': vcv1,
        'likelihoods': rng.standard_normal(T) - 1.0,
        'scores': rng.standard_normal((T, K1)),
        'ARlags': np.array([1]),
        'MAlags': np.array([], dtype=np.int64),
        'constant': 1,
        'StdErr': stde1,
        'Tstat': tstat1,
        'Pval': pval1,
        'AIC': 1304.0,
        'BIC': 1312.4,
        'K': K1,
    })

    # ------------------------------------------------------------------
    # Model 2: ARMA(2,1) with constant
    # ------------------------------------------------------------------
    K2: int = 4  # constant + 2 AR + 1 MA parameters
    params2: np.ndarray = np.array([0.05, 0.25, -0.1, 0.15])
    errors2: np.ndarray = rng.standard_normal(T)
    vcv2: np.ndarray = np.eye(K2) * 0.005
    stde2: np.ndarray = np.sqrt(np.diag(vcv2))
    tstat2: np.ndarray = params2 / stde2
    pval2: np.ndarray = np.array([0.05, 0.001, 0.1, 0.02])

    results.append({
        'StringID': 'ARMA(2,1) w/ constant',
        'parameters': params2,
        'errors': errors2,
        'll': -640.0,
        'seregression': float(np.var(errors2)),
        'diagnostics': {},
        'covariance': vcv2,
        'likelihoods': rng.standard_normal(T) - 1.0,
        'scores': rng.standard_normal((T, K2)),
        'ARlags': np.array([1, 2]),
        'MAlags': np.array([1]),
        'constant': 1,
        'StdErr': stde2,
        'Tstat': tstat2,
        'Pval': pval2,
        'AIC': 1288.0,
        'BIC': 1305.0,
        'K': K2,
    })

    return results


# ---------------------------------------------------------------------------
# SSIM-Based Image Comparison Helper
# ---------------------------------------------------------------------------

def compare_images_ssim(
    image_path_1: Path,
    image_path_2: Path,
    threshold: float = SSIM_THRESHOLD,
) -> bool:
    """Compare two PNG images using Structural Similarity Index (SSIM).

    Per AAP Section 0.7.1: Plot output compared via saved ``.png`` snapshots
    with perceptual diff tolerance SSIM >= 0.95.

    The function lazily imports ``scikit-image`` so that it remains an optional
    dependency.  If ``scikit-image`` is not installed, the calling test is
    gracefully skipped via ``pytest.skip()``.

    Parameters
    ----------
    image_path_1 : Path
        Path to the first (reference) image.
    image_path_2 : Path
        Path to the second (rendered) image.
    threshold : float, optional
        Minimum SSIM value for images to be considered visually similar.
        Defaults to :data:`SSIM_THRESHOLD` (0.95).

    Returns
    -------
    bool
        ``True`` if SSIM >= *threshold*, ``False`` otherwise.

    Raises
    ------
    pytest.skip
        If ``scikit-image`` is not installed.
    FileNotFoundError
        If either image path does not exist.
    """
    # Lazy import — scikit-image is NOT a hard dependency of the test suite.
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.io import imread
        from skimage.color import rgb2gray
    except ImportError:
        pytest.skip("scikit-image not available for SSIM comparison")

    # Validate paths exist before attempting I/O.
    if not Path(image_path_1).exists():
        raise FileNotFoundError(
            f"Reference image not found: {image_path_1}"
        )
    if not Path(image_path_2).exists():
        raise FileNotFoundError(
            f"Rendered image not found: {image_path_2}"
        )

    img1: np.ndarray = imread(str(image_path_1))
    img2: np.ndarray = imread(str(image_path_2))

    # Convert to grayscale for SSIM comparison — colour differences are
    # less relevant for structural layout verification of plots.
    if img1.ndim == 3:
        img1 = rgb2gray(img1)
    if img2.ndim == 3:
        img2 = rgb2gray(img2)

    # Resize if dimensions don't match (handle minor rendering-size
    # differences between reference and rendered images).
    if img1.shape != img2.shape:
        try:
            from skimage.transform import resize
        except ImportError:
            pytest.skip("scikit-image transform module not available")
        img2 = resize(img2, img1.shape, anti_aliasing=True)

    score: float = ssim(img1, img2, data_range=1.0)
    return score >= threshold


# ---------------------------------------------------------------------------
# Snapshot Directory Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for plot snapshot testing.

    Creates a ``snapshots/`` subdirectory under pytest's per-test
    ``tmp_path`` so that each test gets an isolated location for saving
    rendered ``.png`` images.

    Parameters
    ----------
    tmp_path : Path
        pytest-provided per-test temporary directory.

    Returns
    -------
    Path
        Path to the ``snapshots/`` subdirectory (guaranteed to exist).
    """
    snap_dir: Path = tmp_path / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    return snap_dir
