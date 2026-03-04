"""
MFE Toolbox realized subpackage — realized volatility estimators and helpers.

Provides high-frequency variance/covariance estimators, kernel-based estimators,
time-unit conversion utilities, price/return filtering helpers, and simulation
routines:

- **Kernel estimators**: realized_kernel, realized_kernel_bandwidth,
  realized_kernel_core, realized_kernel_jitter_lag_length, realized_kernel_weights
- **Noise / options**: realized_noise_estimate, realized_options
- **Filtering**: realized_price_filter, realized_return_filter
- **Refresh time**: realized_refresh_time, realized_refresh_time_bivariate
- **Quantile / range**: realized_quantile_variance_scale,
  realized_quantile_weight_simulation, realized_range_simulation
- **Helpers**: realized_compute_median
- **Time conversions**: seconds2unit, seconds2wall, unit2seconds, unit2wall,
  wall2seconds, wall2unit

All modules are re-exported here for convenient access via
``from mfe_toolbox.realized import realized_kernel, seconds2unit`` etc.

Per AAP Section 0.4.1: 1:1 migration from realized/*.m.
"""

# ---------------------------------------------------------------------------
# Kernel estimators
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_kernel import realized_kernel
from mfe_toolbox.realized.realized_kernel_bandwidth import realized_kernel_bandwidth
from mfe_toolbox.realized.realized_kernel_core import realized_kernel_core
from mfe_toolbox.realized.realized_kernel_jitter_lag_length import realized_kernel_jitter_lag_length
from mfe_toolbox.realized.realized_kernel_weights import realized_kernel_weights

# ---------------------------------------------------------------------------
# Noise estimation and options
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_options import realized_options

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_return_filter import realized_return_filter

# ---------------------------------------------------------------------------
# Refresh time sampling
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_refresh_time import realized_refresh_time
from mfe_toolbox.realized.realized_refresh_time_bivariate import realized_refresh_time_bivariate

# ---------------------------------------------------------------------------
# Quantile / range / simulation
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_quantile_variance_scale import realized_quantile_variance_scale
from mfe_toolbox.realized.realized_quantile_weight_simulation import realized_quantile_weight_simulation
from mfe_toolbox.realized.realized_range_simulation import realized_range_simulation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_compute_median import realized_compute_median

# ---------------------------------------------------------------------------
# Time conversions
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.seconds2unit import seconds2unit
from mfe_toolbox.realized.seconds2wall import seconds2wall
from mfe_toolbox.realized.unit2seconds import unit2seconds
from mfe_toolbox.realized.unit2wall import unit2wall
from mfe_toolbox.realized.wall2seconds import wall2seconds
from mfe_toolbox.realized.wall2unit import wall2unit

__all__ = [
    # Kernel estimators
    'realized_kernel',
    'realized_kernel_bandwidth',
    'realized_kernel_core',
    'realized_kernel_jitter_lag_length',
    'realized_kernel_weights',
    # Noise / options
    'realized_noise_estimate',
    'realized_options',
    # Filtering
    'realized_price_filter',
    'realized_return_filter',
    # Refresh time
    'realized_refresh_time',
    'realized_refresh_time_bivariate',
    # Quantile / range / simulation
    'realized_quantile_variance_scale',
    'realized_quantile_weight_simulation',
    'realized_range_simulation',
    # Helpers
    'realized_compute_median',
    # Time conversions
    'seconds2unit',
    'seconds2wall',
    'unit2seconds',
    'unit2wall',
    'wall2seconds',
    'wall2unit',
]
