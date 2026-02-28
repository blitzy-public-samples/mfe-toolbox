# MFE Toolbox

A comprehensive Python 3.12 package for financial econometrics, providing production-grade
implementations of volatility models, time series analysis, realized volatility estimators,
statistical distributions, bootstrap inference, and diagnostic tests.

## Features

- **Univariate GARCH Models** — AGARCH, APARCH, EGARCH, FIGARCH, HEAVY, IGARCH, and
  TARCH/GJR-GARCH with full estimation, simulation, parameter transforms, and display
- **Multivariate GARCH Models** — BEKK, CCC-MVGARCH, DCC/ADCC, GO-GARCH, Matrix GARCH,
  O-MVGARCH, RARCH, RCC, RiskMetrics, RiskMetrics 2006, and Scalar VT-VECH
- **Time Series Analysis** — ARMAX filtering and estimation, VAR, HAR, Augmented
  Dickey-Fuller unit root tests, Hodrick-Prescott and Baxter-King filters, impulse
  response functions, Granger causality tests, ACF/PACF
- **Realized Volatility** — 19+ estimators including realized variance, bipower variation,
  kernel estimators, pre-averaged variance, quantile variance, two-scale variance,
  multivariate kernel, Hayashi-Yoshida, and range-based estimators with microstructure
  noise handling
- **Statistical Distributions** — Generalized Error Distribution (GED), standardized-t,
  skewed-t (PDF, CDF, quantile, random generation, log-likelihood), multivariate normal
  log-likelihood, and composite likelihood
- **Bootstrap Methods** — Block bootstrap, stationary bootstrap, Hansen-White SPA/BSDS
  test, and Model Confidence Set (MCS)
- **Diagnostic Tests** — Berkowitz density test, Jarque-Bera normality test,
  Kolmogorov-Smirnov test, Ljung-Box serial correlation test, and LM ARCH test
- **Cross-Sectional Analysis** — OLS with White and HAC robust standard errors, PCA with
  three normalization modes
- **Utility Functions** — Newey-West HAC covariance estimation, numerical gradient and
  Hessian computation, matrix parameterizations (vech, ivech, chol2vec, vec2chol),
  correlation transforms, robust sandwich VCV, and data standardization
- **GUI Application** — PyQt6-based ARMAX estimation front-end with interactive plotting
  via embedded Matplotlib canvases

## Installation

### Requirements

- Python 3.12 or later

### Standard Install

```bash
pip install mfe-toolbox
```

### Development Install

```bash
# Clone the repository
git clone https://github.com/bashtage/mfe-toolbox.git
cd mfe-toolbox

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from mfe_toolbox.univariate.tarch import tarch

# Simulate or load return data
returns = np.random.default_rng(42).standard_normal(1000)

# Estimate a TARCH(1,1,1) model with normal errors
result = tarch(returns, p=1, o=1, q=1)
```

```python
from mfe_toolbox.timeseries.armaxfilter import armaxfilter

# Estimate an ARMA(1,1) model
result = armaxfilter(y, constant=1, p=1, q=1)
```

```python
from mfe_toolbox.realized.realized_variance import realized_variance

# Compute realized variance from high-frequency price data
rv = realized_variance(prices, times)
```

## GUI Launch

The package includes a PyQt6-based graphical interface for ARMAX model estimation
and result visualization. Launch it with either:

```bash
# Using the console entry point
mfe-toolbox-gui

# Or using the Python module
python -m mfe_toolbox.gui
```

## Testing

Run the full test suite with:

```bash
# Quick test run (stop on first failure)
pytest -x --tb=short

# Full test run with coverage reporting
pytest --cov=mfe_toolbox --cov-report=term-missing

# Generate an HTML coverage report
pytest --cov=mfe_toolbox --cov-report=html
```

The project targets **≥90% line coverage** across all non-GUI modules and ≥80% for GUI
modules.

## Dependencies

### Runtime

| Package | Version | Purpose |
| --- | --- | --- |
| [numpy](https://numpy.org/) | ≥1.26 | Core numerical arrays, linear algebra, FFT, random generation |
| [scipy](https://scipy.org/) | ≥1.12 | Optimization, statistical distributions, special functions, sparse matrices |
| [pandas](https://pandas.pydata.org/) | ≥2.2 | DataFrame return types, time series indexing |
| [numba](https://numba.pydata.org/) | ≥0.59 | JIT compilation for performance-critical GARCH recursion loops |
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) | ≥6.6 | GUI framework for ARMAX estimation front-end |
| [statsmodels](https://www.statsmodels.org/) | ≥0.14 | Supplementary econometric utilities |
| [matplotlib](https://matplotlib.org/) | ≥3.8 | Plotting and embedded GUI plot canvases |

### Development

| Package | Version | Purpose |
| --- | --- | --- |
| [pytest](https://docs.pytest.org/) | ≥8.0 | Test framework with parametrized parity tests |
| [pytest-cov](https://pytest-cov.readthedocs.io/) | ≥5.0 | Coverage reporting and enforcement |
| [pytest-qt](https://pytest-qt.readthedocs.io/) | ≥4.4 | PyQt6 GUI integration testing |

## Project Structure

```
mfe_toolbox/
├── univariate/       # Univariate GARCH model families (59 modules)
├── multivariate/     # Multivariate GARCH models (38 modules)
├── timeseries/       # Time series models and filters (31 modules)
├── realized/         # Realized volatility estimators (42 modules)
├── distributions/    # Statistical distributions (19 modules)
├── utility/          # Helper functions (29 modules)
├── bootstrap/        # Resampling inference (4 modules)
├── tests/            # Diagnostic statistical tests (5 modules)
├── crosssection/     # Cross-sectional models (2 modules)
├── sandbox/          # Experimental/prototype modules (6 modules)
└── gui/              # PyQt6 ARMAX GUI application (4 widget modules)
```

## License

This project is licensed under the BSD License. See the [LICENSE](LICENSE) file for details.

## Credits

The MFE Toolbox was originally developed as a MATLAB toolbox (Version 4.0, 28-Oct-2009)
by **Kevin Sheppard** at the University of Oxford. This Python package is a faithful
migration of the original MATLAB codebase, preserving all public function interfaces,
numerical behavior, and estimation algorithms.
