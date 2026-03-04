"""
HEAVY model test script.

Experimental sandbox script that orchestrates a complete HEAVY model
simulation, estimation, and visual comparison workflow.  Demonstrates the
full pipeline: parameter definition → simulation → estimation → plot.

The script:

1. Defines a bivariate HEAVY model with K=2 (one return series, one
   realized-measure series with ``m[1]=390`` underlying observations).
2. Simulates T=1000 observations via :func:`heavy_simulate`.
3. Computes auxiliary data transformations matching the original MATLAB
   test script (squared returns, back-cast initialization, volatility
   bounds).
4. Estimates the model twice using :func:`heavy` — once with default
   starting values and once with the true simulation parameters.
5. Plots estimated vs. original latent volatility paths.

Migrated from ``sandbox/heavy_test.m`` (Version 4.0, 30 lines).
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk).

Migration notes:
    - ``clear all; clc; close all`` environment reset removed (no Python
      equivalent needed)
    - MATLAB 1-based array indexing converted to Python 0-based throughout
    - ``fminunc`` / ``optimset`` references preserved as comments with scipy
      equivalents noted
    - ``plot`` replaced by ``matplotlib.pyplot``
    - All variable names converted to snake_case
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from mfe_toolbox.univariate.heavy_simulate import heavy_simulate
from mfe_toolbox.univariate.heavy import heavy
# from mfe_toolbox.univariate.heavy_likelihood import heavy_likelihood  # Ref: heavy_test.m:16 — commented out per original source


def heavy_test() -> None:
    """Run a HEAVY model simulation-estimation-comparison test workflow.

    This function exercises the full HEAVY model pipeline:

    1. Defines model parameters for a K=2 bivariate system where series 0
       is return-like (``m[0]=1``) and series 1 is a realized measure with
       390 underlying intraday observations (``m[1]=390``).
    2. Simulates ``T=1000`` observations using :func:`heavy_simulate`.
    3. Computes data transformations (squared returns, column means for
       back-casting, extreme volatility bounds) as in the original MATLAB
       test script.
    4. Estimates the model using :func:`heavy` first with automatically
       generated starting values, then with the true simulation parameters
       as starting values.
    5. Plots the first series' estimated conditional volatility against the
       true simulated latent volatility for visual comparison.

    Returns
    -------
    None

    Notes
    -----
    This is a sandbox/experimental script preserved from the original MATLAB
    toolbox for development and testing purposes.  It is not part of the
    production estimation API.

    The commented-out code blocks (Phase 4 below) faithfully reproduce the
    structure of the original MATLAB source, where direct likelihood
    evaluation and manual ``fminunc`` optimization were commented out in
    favour of the high-level ``heavy()`` driver.

    See Also
    --------
    mfe_toolbox.univariate.heavy : HEAVY model estimation driver.
    mfe_toolbox.univariate.heavy_simulate : HEAVY model simulation.
    mfe_toolbox.univariate.heavy_likelihood : HEAVY log-likelihood function.

    Examples
    --------
    >>> from mfe_toolbox.sandbox.heavy_test import heavy_test
    >>> # heavy_test()  # Runs full simulation + estimation + plot workflow
    """
    # ==================================================================
    # Phase 1: Model Setup
    # Ref: heavy_test.m:4-8
    # ==================================================================

    # Ref: heavy_test.m:4 — parameters = [.3 .05 .2 .4 .7 .55]
    # 6-element parameter vector for a K=2 HEAVY model:
    #   [O1, O2, A(0,1), A(1,1), B(0,0), B(1,1)]
    # where O are intercepts, A are innovation coefficients, B are smoothing
    # coefficients following the ordering convention of heavy_parameter_transform.
    parameters = np.array([0.3, 0.05, 0.2, 0.4, 0.7, 0.55])

    # Ref: heavy_test.m:5 — p = [0 1;0 1] → 2×2 lag structure for innovations
    # Row i, column j: number of lags of series j innovations in equation for
    # series i.  Both equations use 1 lag of series 1 (the realized measure)
    # and 0 lags of series 0 (returns).
    p = np.array([[0, 1], [0, 1]])

    # Ref: heavy_test.m:6 — q = eye(2) → 2×2 identity lag structure for
    # conditional variances; each equation uses 1 lag of its own variance.
    q = np.eye(2, dtype=int)

    # Ref: heavy_test.m:7 — m = [1 390]
    # m[0]=1 → series 0 is return-like (mean zero, variance h);
    # m[1]=390 → series 1 is a realized measure with 390 intraday observations.
    m = np.array([1, 390])

    # Ref: heavy_test.m:8 — T = 1000
    T = 1000

    # ==================================================================
    # Phase 2: Simulation
    # Ref: heavy_test.m:9-11
    # ==================================================================

    # Ref: heavy_test.m:9 — [data, htOrig] = heavy_simulate(T, 2, parameters, p, q, m)
    # Simulate T observations of a bivariate HEAVY process.
    # data: (T, K) array of simulated return + realized-measure series
    # ht_orig: (T, K) array of true latent conditional variances
    data, ht_orig = heavy_simulate(T, 2, parameters, p, q, m)

    # Ref: heavy_test.m:10 — parametersOrig = parameters
    # Save a copy of the true simulation parameters before they are
    # overwritten by estimation results.
    parameters_orig = parameters.copy()

    # Ref: heavy_test.m:11 — data2 = data
    # Create a working copy of data for manual transformation.
    data2 = data.copy()

    # ==================================================================
    # Phase 3: Data Transformation
    # Ref: heavy_test.m:12-15
    # These variables (data2, back_cast, lb, ub) were used by the
    # commented-out direct likelihood / manual optimization calls below.
    # They are computed here for completeness and parity with the MATLAB
    # source but are not consumed by the heavy() driver calls.
    # ==================================================================

    # Ref: heavy_test.m:12 — data2(:,1) = data2(:,1).^2
    # MATLAB 1-indexed (:,1) → Python 0-indexed [:, 0]
    # Square the return series to match the internal representation used
    # by the HEAVY likelihood function.
    data2[:, 0] = data2[:, 0] ** 2  # Ref: heavy_test.m:12

    # Ref: heavy_test.m:13 — backCast = mean(data2)
    # Column-wise means used as initial back-cast values for the recursion.
    back_cast = np.mean(data2, axis=0)

    # Ref: heavy_test.m:14 — lb = min(data2)'/10000
    # MATLAB min(data2) returns a row vector of column minimums;
    # the transpose (') converts to column; Python np.min with axis=0
    # returns a 1-D array directly.
    lb = np.min(data2, axis=0) / 10000.0

    # Ref: heavy_test.m:15 — ub = max(data2)'*100000
    ub = np.max(data2, axis=0) * 100000.0

    # ==================================================================
    # Phase 4: Commented-out Direct Likelihood & Manual Optimization
    # Ref: heavy_test.m:16-23
    # The original MATLAB source had these lines partially commented out.
    # They are preserved here as comments with Python equivalents for
    # reference, matching the original experimental workflow.
    # ==================================================================

    # Ref: heavy_test.m:16 — %[ll,lls,h] = heavy_likelihood(parameters,data2,p,q,backCast,lb,ub);
    # ll, lls, h = heavy_likelihood(parameters, data2.T, p, q, back_cast, lb, ub)

    # Ref: heavy_test.m:18-19 — options = optimset('fminunc'); options.Display = 'iter';
    # In Python, scipy.optimize.minimize replaces fminunc:
    # options = {'disp': True}

    # Ref: heavy_test.m:21 — sv = parameters;
    # sv = parameters.copy()

    # Ref: heavy_test.m:22 — sv(1:2) = log(sv(1:2))
    # MATLAB 1-based sv(1:2) selects elements 1 and 2 → Python 0-based sv[0:2]
    # Log-transform the intercept parameters for unconstrained optimization.
    # sv[0:2] = np.log(sv[0:2])  # Ref: heavy_test.m:22

    # Ref: heavy_test.m:23 — %out = fminunc(@heavy_likelihood,sv,options,data2,p,q,backCast,lb,ub);
    # In Python with scipy:
    # from scipy.optimize import minimize
    # out = minimize(heavy_likelihood, sv, args=(data2.T, p, q, back_cast, lb, ub),
    #                method='L-BFGS-B', options=options)

    # ==================================================================
    # Phase 5: First Estimation (default starting values)
    # Ref: heavy_test.m:25
    # ==================================================================

    # Ref: heavy_test.m:25 — [parameters, ll, ht, VCV, scores] = heavy(data,p,q,'None')
    # Estimate the HEAVY model from simulated data using automatically
    # generated starting values.  The 'None' constraint mode only enforces
    # positive intercepts.
    parameters_est, ll, ht, vcv, scores = heavy(data, p, q, 'None')

    # ==================================================================
    # Phase 6: Visualization
    # Ref: heavy_test.m:26
    # ==================================================================

    # Ref: heavy_test.m:26 — plot([ht(:,1), htOrig(:,1)])
    # MATLAB (:,1) → Python [:, 0]; plots the first series' (return) estimated
    # conditional volatility alongside the true simulated volatility path.
    plt.plot(np.column_stack([ht[:, 0], ht_orig[:, 0]]))
    plt.show()

    # ==================================================================
    # Phase 7: Second Estimation (original parameters as starting values)
    # Ref: heavy_test.m:29
    # ==================================================================

    # Ref: heavy_test.m:29 — [parameters, ll, ht, VCV, scores] = heavy(data,p,q,'None',parametersOrig')
    # Re-estimate using the true simulation parameters as starting values.
    # MATLAB parametersOrig' transposes row vector → column vector; in Python
    # the 1-D array is passed directly (no transpose needed).
    parameters_est2, ll2, ht2, vcv2, scores2 = heavy(
        data, p, q, 'None', parameters_orig
    )


if __name__ == '__main__':
    heavy_test()
