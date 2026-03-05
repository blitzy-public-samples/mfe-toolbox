"""
Realized volatility estimators and high-frequency data utilities.

This subpackage provides a comprehensive suite of realized volatility
estimators, kernel-based covariance estimators, high-frequency data
filtering and sampling utilities, refresh-time synchronization for
multi-asset data, time-unit conversion helpers, and Monte Carlo simulation
routines for scale-factor calibration.

The 42 public functions are organized into the following categories:

**Variance Estimators**

- :func:`realized_variance` — Standard realized variance with optional
  subsampling.
- :func:`realized_bipower_variation` — Jump-robust bipower variation
  (Barndorff-Nielsen & Shephard 2004) with skip-k support.
- :func:`realized_semivariance` — Realized semivariance with positive
  and negative components (Barndorff-Nielsen, Kinnebrock, Shephard).
- :func:`realized_quarticity` — Realized quarticity (BNS, tripower,
  quadpower) with skip-k support.
- :func:`realized_multiscale_variance` — Zhang multiscale realized
  variance (MSRV) estimator.
- :func:`realized_twoscale_variance` — Ait-Sahalia/Mykland/Zhang
  two-scale realized variance (TSRV).
- :func:`realized_qmle_variance` — Quasi-maximum likelihood estimated
  realized variance (Xiu 2010) with EM iterations.
- :func:`realized_preaveraged_variance` — Christensen-Oomen-Podolski
  pre-averaged realized variance.
- :func:`realized_preaveraged_bipower_variation` — Pre-averaged bipower
  variation for noise-robust jump detection.
- :func:`realized_quantile_variance` — Quantile realized variance
  (MinRV, MedRV, symmetrized variants).
- :func:`realized_threshold_variance` — Adaptive threshold realized
  variance with Gaussian kernel local variance proxy.
- :func:`realized_threshold_multipower_variation` — Thresholded
  multipower variation for jump-robust higher-order estimation.
- :func:`realized_min_med_variance` — Minimum and median realized
  variance (Andersen, Dobrev, Schaumburg 2012).

**Covariance / Multivariate Estimators**

- :func:`realized_covariance` — Realized covariance with subsampling.
- :func:`realized_hayashi_yoshida` — Hayashi-Yoshida asynchronous
  covariation estimator with lead-lag correction.
- :func:`realized_multivariate_kernel` — Multivariate realized kernel
  covariance estimator producing PSD covariance matrices.

**Kernel Estimator Components**

- :func:`realized_kernel` — Orchestrator for the Barndorff-Nielsen,
  Hansen, Lunde and Shephard realized kernel estimator.
- :func:`realized_kernel_bandwidth` — Optimal bandwidth selection.
- :func:`realized_kernel_core` — Core weighted autocovariance
  summation.
- :func:`realized_kernel_jitter_lag_length` — MSE-minimizing jitter
  lag length selection.
- :func:`realized_kernel_weights` — Kernel weight function lookup.

**Range and Noise Estimation**

- :func:`realized_range` — Realized range estimator using scaled
  max-min ranges within blocks.
- :func:`realized_noise_estimate` — Microstructure noise variance
  estimation.
- :func:`realized_variance_optimal_sampling` — Bandi-Russell optimal
  sampling frequency selection.

**Data Filtering and Sampling**

- :func:`realized_price_filter` — Core price filtering supporting 5
  sampling types.
- :func:`realized_return_filter` — Return filtering producing log
  returns and time intervals.
- :func:`realized_refresh_time` — Multi-asset refresh-time
  synchronization.
- :func:`realized_refresh_time_bivariate` — Bivariate refresh time
  synchronization (BNHLS 2008).
- :func:`realized_subsample` — Subsampling grid generation for
  bias-reduction averaging.

**Configuration and Helpers**

- :func:`realized_options` — Default options factory for estimator
  configuration dicts.
- :func:`realized_compute_median` — Median price computation for tick
  deduplication.
- :func:`realized_convert2unit` — Time-to-unit-interval conversion
  normalizing timestamps to [0, 1].
- :func:`realized_test` — Smoke test exercising all estimators with
  sample data.

**Simulation / Calibration**

- :func:`realized_range_simulation` — Monte Carlo simulation for range
  scale factors.
- :func:`realized_quantile_variance_scale` — Monte Carlo quantile RV
  scale factors, covariances, GMVP weights.
- :func:`realized_quantile_weight_simulation` — Two-phase Monte Carlo
  for QRV weights and scales.

**Time Conversion Utilities**

- :func:`seconds2unit` — Seconds past midnight → unit [0, 1].
- :func:`seconds2wall` — Seconds past midnight → wall-clock HHMMSS.
- :func:`unit2seconds` — Unit [0, 1] → seconds past midnight.
- :func:`unit2wall` — Unit [0, 1] → wall-clock HHMMSS.
- :func:`wall2seconds` — Wall-clock HHMMSS → seconds past midnight.
- :func:`wall2unit` — Wall-clock HHMMSS → unit [0, 1].

All 42 functions are re-exported at package level for convenient access::

    from mfe_toolbox.realized import realized_variance, realized_kernel

Migrated from: ``realized/*.m`` — MFE Toolbox Version 4.0
(Kevin Sheppard, University of Oxford)
"""

# ---------------------------------------------------------------------------
# Variance estimators
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_variance import realized_variance
from mfe_toolbox.realized.realized_bipower_variation import realized_bipower_variation
from mfe_toolbox.realized.realized_semivariance import realized_semivariance
from mfe_toolbox.realized.realized_quarticity import realized_quarticity
from mfe_toolbox.realized.realized_multiscale_variance import realized_multiscale_variance
from mfe_toolbox.realized.realized_twoscale_variance import realized_twoscale_variance
from mfe_toolbox.realized.realized_qmle_variance import realized_qmle_variance
from mfe_toolbox.realized.realized_preaveraged_variance import realized_preaveraged_variance
from mfe_toolbox.realized.realized_preaveraged_bipower_variation import realized_preaveraged_bipower_variation
from mfe_toolbox.realized.realized_quantile_variance import realized_quantile_variance
from mfe_toolbox.realized.realized_threshold_variance import realized_threshold_variance
from mfe_toolbox.realized.realized_threshold_multipower_variation import realized_threshold_multipower_variation
from mfe_toolbox.realized.realized_min_med_variance import realized_min_med_variance

# ---------------------------------------------------------------------------
# Covariance / multivariate estimators
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_covariance import realized_covariance
from mfe_toolbox.realized.realized_hayashi_yoshida import realized_hayashi_yoshida
from mfe_toolbox.realized.realized_multivariate_kernel import realized_multivariate_kernel

# ---------------------------------------------------------------------------
# Kernel estimator components
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_kernel import realized_kernel
from mfe_toolbox.realized.realized_kernel_bandwidth import realized_kernel_bandwidth
from mfe_toolbox.realized.realized_kernel_core import realized_kernel_core
from mfe_toolbox.realized.realized_kernel_jitter_lag_length import realized_kernel_jitter_lag_length
from mfe_toolbox.realized.realized_kernel_weights import realized_kernel_weights

# ---------------------------------------------------------------------------
# Range and noise estimation
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_range import realized_range
from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_variance_optimal_sampling import realized_variance_optimal_sampling

# ---------------------------------------------------------------------------
# Data filtering and sampling
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_return_filter import realized_return_filter
from mfe_toolbox.realized.realized_refresh_time import realized_refresh_time
from mfe_toolbox.realized.realized_refresh_time_bivariate import realized_refresh_time_bivariate
from mfe_toolbox.realized.realized_subsample import realized_subsample

# ---------------------------------------------------------------------------
# Configuration and helpers
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_compute_median import realized_compute_median
from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_test import realized_test

# ---------------------------------------------------------------------------
# Simulation / calibration
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_range_simulation import realized_range_simulation
from mfe_toolbox.realized.realized_quantile_variance_scale import realized_quantile_variance_scale
from mfe_toolbox.realized.realized_quantile_weight_simulation import realized_quantile_weight_simulation

# ---------------------------------------------------------------------------
# Time conversion utilities
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.seconds2unit import seconds2unit
from mfe_toolbox.realized.seconds2wall import seconds2wall
from mfe_toolbox.realized.unit2seconds import unit2seconds
from mfe_toolbox.realized.unit2wall import unit2wall
from mfe_toolbox.realized.wall2seconds import wall2seconds
from mfe_toolbox.realized.wall2unit import wall2unit

# ---------------------------------------------------------------------------
# Public API — explicit star-import control
# ---------------------------------------------------------------------------
__all__ = [
    # Variance estimators
    "realized_variance",
    "realized_bipower_variation",
    "realized_semivariance",
    "realized_quarticity",
    "realized_multiscale_variance",
    "realized_twoscale_variance",
    "realized_qmle_variance",
    "realized_preaveraged_variance",
    "realized_preaveraged_bipower_variation",
    "realized_quantile_variance",
    "realized_threshold_variance",
    "realized_threshold_multipower_variation",
    "realized_min_med_variance",
    # Covariance / multivariate
    "realized_covariance",
    "realized_hayashi_yoshida",
    "realized_multivariate_kernel",
    # Kernel estimator components
    "realized_kernel",
    "realized_kernel_bandwidth",
    "realized_kernel_core",
    "realized_kernel_jitter_lag_length",
    "realized_kernel_weights",
    # Range and noise estimation
    "realized_range",
    "realized_noise_estimate",
    "realized_variance_optimal_sampling",
    # Data filtering and sampling
    "realized_price_filter",
    "realized_return_filter",
    "realized_refresh_time",
    "realized_refresh_time_bivariate",
    "realized_subsample",
    # Configuration and helpers
    "realized_options",
    "realized_compute_median",
    "realized_convert2unit",
    "realized_test",
    # Simulation / calibration
    "realized_range_simulation",
    "realized_quantile_variance_scale",
    "realized_quantile_weight_simulation",
    # Time conversion utilities
    "seconds2unit",
    "seconds2wall",
    "unit2seconds",
    "unit2wall",
    "wall2seconds",
    "wall2unit",
]
