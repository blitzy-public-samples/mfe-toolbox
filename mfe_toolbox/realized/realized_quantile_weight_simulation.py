"""
Two-phase Monte Carlo simulation for precomputing Quantile Realized Variance
(QRV) weight scales, covariances, and GMVP weights.

This module runs an extensive Monte Carlo simulation across a predefined set of
block sizes to precompute the scale factors, covariance matrices, and
Global Minimum Variance Portfolio (GMVP) weights used by the
``realized_quantile_variance`` estimator.  The precomputed data is returned as a
Python ``dict`` and may optionally be persisted to an ``.npz`` file via
:func:`numpy.savez`.

The simulation proceeds in two phases:

Phase 1 — **Symmetric** quantiles  ``(1:M)/M``  with ``symmetric=True``.
Phase 2 — **Asymmetric** quantiles ``(ceil(M/2)+1:M)/M`` with ``symmetric=False``.

For block sizes *M* ≥ 10, a convex-regression smoothing step is applied to the
raw scale estimates.  The smoothing solves::

    min  ||x − y||²   subject to  −x[i] + 2·x[i+1] − x[i+2] ≤ 0  ∀ i

using ``scipy.optimize.minimize`` with the SLSQP method, replacing MATLAB's
``lsqlin`` constrained least-squares solver.

Migrated from: realized/realized_quantile_weight_simulation.m
Original author: Kevin Sheppard
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.realized.realized_quantile_variance_scale import (
    realized_quantile_variance_scale,
)

# ---------------------------------------------------------------------------
# Predefined block-size sets — matches MATLAB source exactly
# Ref: realized_quantile_weight_simulation.m:9
# ---------------------------------------------------------------------------
_SYMMETRIC_BLOCK_SIZES: list[int] = [
    2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 18, 20,
    24, 25, 26, 30, 36, 39, 40, 50, 60, 65, 72, 75, 100, 144,
]

# Ref: realized_quantile_weight_simulation.m:67
_ASYMMETRIC_BLOCK_SIZES: list[int] = [
    4, 5, 6, 10, 13, 15, 18, 20, 25, 26, 30, 36,
    39, 50, 60, 65, 72, 75, 100, 144,
]


# ===================================================================== #
# Public API                                                             #
# ===================================================================== #


def realized_quantile_weight_simulation(
    output_file: str | None = None,
    simulations: int = 100_000_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Precompute QRV scales, covariances, and GMVP weights for standard
    block sizes.

    Runs a two-phase Monte Carlo simulation over predefined sets of block
    sizes to compute the quantile-variance scale factors, covariance
    matrices, and GMVP weights required by
    :func:`~mfe_toolbox.realized.realized_quantile_variance.realized_quantile_variance`.

    Parameters
    ----------
    output_file : str or None, optional
        Path to save results as a ``.npz`` file.  If *None*, results are
        only returned as a dict.
    simulations : int, optional
        Number of Monte Carlo simulations per block size.  Default is
        100 000 000 (matching the MATLAB source).
    seed : int, optional
        Random seed passed to each call of
        :func:`realized_quantile_variance_scale` for reproducibility.
        Default is 0.

    Returns
    -------
    dict
        Dictionary with the following keys (each value is a Python list
        whose *i*-th element corresponds to the *i*-th block size):

        * ``'symmetricSimulationSamplerperbin'`` – list of ``int`` block
          sizes used in Phase 1.
        * ``'symmetricSimulationQuantile'`` – list of 1-D
          :class:`numpy.ndarray` quantile vectors.
        * ``'symmetricExpectedQuantiles'`` – list of 1-D
          :class:`numpy.ndarray` scale-factor vectors (smoothed when
          *M* ≥ 10).
        * ``'symmetricExpectedCovariance'`` – list of 2-D
          :class:`numpy.ndarray` covariance matrices.
        * ``'asymmetricSimulationSamplerperbin'`` – list of ``int``
          block sizes used in Phase 2.
        * ``'asymmetricSimulationQuantile'`` – list of 1-D
          :class:`numpy.ndarray` quantile vectors.
        * ``'asymmetricExpectedQuantiles'`` – list of 1-D
          :class:`numpy.ndarray` scale-factor vectors.
        * ``'asymmetricExpectedCovariance'`` – list of 2-D
          :class:`numpy.ndarray` covariance matrices.

    Raises
    ------
    ValueError
        If *simulations* < 1 or *seed* < 0.

    Notes
    -----
    This function replicates the precomputation performed by the MATLAB
    script ``realized_quantile_weight_simulation.m`` which generates the
    ``realized_quantile_scales.mat`` data file consumed by
    ``realized_quantile_variance``.

    The convex-regression smoothing enforces that the scale factors form
    a convex sequence, using SLSQP optimisation with second-difference
    inequality constraints (Ref: realized_quantile_weight_simulation.m
    lines 46–51 and 84–89).

    See Also
    --------
    realized_quantile_variance_scale :
        Computes scales for a single block size.
    """
    # ------------------------------------------------------------------ #
    # Input validation                                                    #
    # ------------------------------------------------------------------ #
    if not isinstance(simulations, (int, np.integer)) or simulations < 1:
        raise ValueError(
            "simulations must be a positive integer; got "
            f"{simulations!r}."
        )
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(
            f"seed must be a non-negative integer; got {seed!r}."
        )

    # ------------------------------------------------------------------ #
    # Phase 1 — Symmetric quantile scales                                #
    # Ref: realized_quantile_weight_simulation.m:29-57                   #
    # ------------------------------------------------------------------ #
    sym_samplerperbin: list[int] = []
    sym_quantile: list[np.ndarray] = []
    sym_expected_quantiles: list[np.ndarray] = []
    sym_expected_covariance: list[np.ndarray] = []

    t_start: float = time.time()

    for block_m in _SYMMETRIC_BLOCK_SIZES:
        # Ref: realized_quantile_weight_simulation.m:36-37
        # MATLAB: (1:M)/M  →  Python: np.arange(1, M+1) / M
        quantiles = np.arange(1, block_m + 1, dtype=np.float64) / block_m

        sym_samplerperbin.append(block_m)
        sym_quantile.append(quantiles)

        # Ref: realized_quantile_weight_simulation.m:39
        scales, _weights, covar = realized_quantile_variance_scale(
            block_m, quantiles, simulations, symmetric=True, seed=seed,
        )

        elapsed = time.time() - t_start
        print(
            f"Symmetric  M={block_m:>4d}  "
            f"(elapsed {elapsed:10.1f}s)"
        )

        # Ref: realized_quantile_weight_simulation.m:44-53
        if block_m >= 10:
            smoothed = _convex_regression_smooth(scales)
            max_diff = float(np.max(np.abs(smoothed - scales)))
            # Ref: realized_quantile_weight_simulation.m:54
            print(f"  Convex regression max |Δ| = {max_diff:.6e}")
            scales = smoothed

        sym_expected_quantiles.append(scales)
        sym_expected_covariance.append(covar)

    # ------------------------------------------------------------------ #
    # Phase 2 — Asymmetric quantile scales                               #
    # Ref: realized_quantile_weight_simulation.m:64-95                   #
    # ------------------------------------------------------------------ #
    asym_samplerperbin: list[int] = []
    asym_quantile: list[np.ndarray] = []
    asym_expected_quantiles: list[np.ndarray] = []
    asym_expected_covariance: list[np.ndarray] = []

    for block_m in _ASYMMETRIC_BLOCK_SIZES:
        # Ref: realized_quantile_weight_simulation.m:74
        # MATLAB: (ceil(M/2)+1:M)/M  →  Python: np.arange(ceil(M/2)+1, M+1)/M
        start_idx = int(np.ceil(block_m / 2)) + 1
        quantiles = np.arange(start_idx, block_m + 1, dtype=np.float64) / block_m

        asym_samplerperbin.append(block_m)
        asym_quantile.append(quantiles)

        # Ref: realized_quantile_weight_simulation.m:75
        scales, _weights, covar = realized_quantile_variance_scale(
            block_m, quantiles, simulations, symmetric=False, seed=seed,
        )

        elapsed = time.time() - t_start
        print(
            f"Asymmetric M={block_m:>4d}  "
            f"(elapsed {elapsed:10.1f}s)"
        )

        # Ref: realized_quantile_weight_simulation.m:80-89
        if block_m >= 10:
            smoothed = _convex_regression_smooth(scales)
            max_diff = float(np.max(np.abs(smoothed - scales)))
            print(f"  Convex regression max |Δ| = {max_diff:.6e}")
            scales = smoothed

        asym_expected_quantiles.append(scales)
        asym_expected_covariance.append(covar)

    # ------------------------------------------------------------------ #
    # Assemble result dict using MATLAB-compatible key names              #
    # ------------------------------------------------------------------ #
    result: dict[str, Any] = {
        "symmetricSimulationSamplerperbin": sym_samplerperbin,
        "symmetricSimulationQuantile": sym_quantile,
        "symmetricExpectedQuantiles": sym_expected_quantiles,
        "symmetricExpectedCovariance": sym_expected_covariance,
        "asymmetricSimulationSamplerperbin": asym_samplerperbin,
        "asymmetricSimulationQuantile": asym_quantile,
        "asymmetricExpectedQuantiles": asym_expected_quantiles,
        "asymmetricExpectedCovariance": asym_expected_covariance,
    }

    # ------------------------------------------------------------------ #
    # Optionally persist to .npz                                          #
    # Ref: realized_quantile_weight_simulation.m:56 (save ...)           #
    # ------------------------------------------------------------------ #
    if output_file is not None:
        _save_results(result, output_file)

    return result


# ===================================================================== #
# Internal helpers                                                       #
# ===================================================================== #


def _convex_regression_smooth(y: np.ndarray) -> np.ndarray:
    """Project *y* onto the set of convex sequences (L₂ projection).

    Solves the constrained least-squares problem::

        min  Σ (x_i − y_i)²
        s.t. −x[i] + 2·x[i+1] − x[i+2] ≤ 0   ∀ i ∈ {0, …, n−3}

    using ``scipy.optimize.minimize`` with the SLSQP method.  This
    replaces MATLAB ``lsqlin(eye(n), y, A, zeros(n−2,1))`` where *A* is
    the second-difference matrix.

    Ref: realized_quantile_weight_simulation.m:46-51 and :84-89

    Parameters
    ----------
    y : numpy.ndarray
        1-D array of raw scale estimates (length *n*).

    Returns
    -------
    numpy.ndarray
        1-D array of smoothed scale estimates forming a convex sequence.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if n < 3:
        # Fewer than 3 points: convexity constraint is vacuous.
        return y.copy()

    # ------------------------------------------------------------------ #
    # Build the second-difference constraint matrix  A  (n-2 × n)        #
    # Ref: realized_quantile_weight_simulation.m:49                      #
    # MATLAB:                                                             #
    #   A = -[eye(n-2) zeros(n-2,2)]                                     #
    #       + [zeros(n-2,1) 2*eye(n-2) zeros(n-2,1)]                    #
    #       - [zeros(n-2,2) eye(n-2)];                                   #
    # Row i encodes: −x[i] + 2·x[i+1] − x[i+2] ≤ 0                    #
    # ------------------------------------------------------------------ #
    m = n - 2
    eye_m = np.eye(m, dtype=np.float64)
    z2 = np.zeros((m, 2), dtype=np.float64)
    z1 = np.zeros((m, 1), dtype=np.float64)
    constraint_A = (
        -np.hstack([eye_m, z2])
        + np.hstack([z1, 2.0 * eye_m, z1])
        - np.hstack([z2, eye_m])
    )

    # ------------------------------------------------------------------ #
    # Objective: min ||x − y||² and its gradient                         #
    # ------------------------------------------------------------------ #
    def _objective(x: np.ndarray) -> np.float64:
        diff = x - y
        return np.float64(np.sum(diff * diff))

    def _jacobian(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - y)

    # Constraint: A @ x ≤ 0  ⟺  -(A @ x) ≥ 0  (scipy ineq: fun ≥ 0)
    constraint = {
        "type": "ineq",
        "fun": lambda x: -(constraint_A @ x),
        "jac": lambda _x: -constraint_A,
    }

    opt_result = minimize(
        _objective,
        y.copy(),
        method="SLSQP",
        jac=_jacobian,
        constraints=constraint,
        options={"ftol": 1e-15, "maxiter": 2000, "disp": False},
    )
    return np.asarray(opt_result.x, dtype=np.float64)


def _save_results(result: dict[str, Any], output_file: str) -> None:
    """Persist simulation results to a ``.npz`` file.

    Variable-length arrays are stored under indexed keys (e.g.
    ``symmetric_quantile_0``, ``symmetric_scale_0``, …) so that
    ``numpy.load`` can reconstruct them without pickle.

    Parameters
    ----------
    result : dict
        The dict returned by :func:`realized_quantile_weight_simulation`.
    output_file : str
        File path for the output ``.npz`` archive.
    """
    save_dict: dict[str, np.ndarray] = {}

    # --- Symmetric ---------------------------------------------------- #
    sym_sizes = result["symmetricSimulationSamplerperbin"]
    n_sym = len(sym_sizes)
    save_dict["symmetric_count"] = np.array(n_sym, dtype=np.intp)
    save_dict["symmetric_block_sizes"] = np.array(sym_sizes, dtype=np.intp)

    for idx in range(n_sym):
        save_dict[f"symmetric_quantile_{idx}"] = np.asarray(
            result["symmetricSimulationQuantile"][idx], dtype=np.float64,
        )
        save_dict[f"symmetric_scale_{idx}"] = np.asarray(
            result["symmetricExpectedQuantiles"][idx], dtype=np.float64,
        )
        save_dict[f"symmetric_covar_{idx}"] = np.asarray(
            result["symmetricExpectedCovariance"][idx], dtype=np.float64,
        )

    # --- Asymmetric --------------------------------------------------- #
    asym_sizes = result["asymmetricSimulationSamplerperbin"]
    n_asym = len(asym_sizes)
    save_dict["asymmetric_count"] = np.array(n_asym, dtype=np.intp)
    save_dict["asymmetric_block_sizes"] = np.array(asym_sizes, dtype=np.intp)

    for idx in range(n_asym):
        save_dict[f"asymmetric_quantile_{idx}"] = np.asarray(
            result["asymmetricSimulationQuantile"][idx], dtype=np.float64,
        )
        save_dict[f"asymmetric_scale_{idx}"] = np.asarray(
            result["asymmetricExpectedQuantiles"][idx], dtype=np.float64,
        )
        save_dict[f"asymmetric_covar_{idx}"] = np.asarray(
            result["asymmetricExpectedCovariance"][idx], dtype=np.float64,
        )

    np.savez(output_file, **save_dict)


def load_simulation_results(npz_path: str) -> dict[str, Any]:
    """Load precomputed QRV simulation results from a ``.npz`` file.

    This is the inverse of the ``_save_results`` helper — it
    reconstructs the dictionary originally returned by
    :func:`realized_quantile_weight_simulation` from an on-disk
    ``.npz`` archive.

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` file written by
        :func:`realized_quantile_weight_simulation`.

    Returns
    -------
    dict
        Same structure as the return value of
        :func:`realized_quantile_weight_simulation`.

    Raises
    ------
    FileNotFoundError
        If *npz_path* does not exist.
    KeyError
        If the archive is missing expected keys.
    """
    data = np.load(npz_path, allow_pickle=False)

    # --- Symmetric ---------------------------------------------------- #
    n_sym = int(data["symmetric_count"])
    sym_block_sizes = data["symmetric_block_sizes"].tolist()
    sym_quantile = [data[f"symmetric_quantile_{i}"] for i in range(n_sym)]
    sym_scale = [data[f"symmetric_scale_{i}"] for i in range(n_sym)]
    sym_covar = [data[f"symmetric_covar_{i}"] for i in range(n_sym)]

    # --- Asymmetric --------------------------------------------------- #
    n_asym = int(data["asymmetric_count"])
    asym_block_sizes = data["asymmetric_block_sizes"].tolist()
    asym_quantile = [data[f"asymmetric_quantile_{i}"] for i in range(n_asym)]
    asym_scale = [data[f"asymmetric_scale_{i}"] for i in range(n_asym)]
    asym_covar = [data[f"asymmetric_covar_{i}"] for i in range(n_asym)]

    return {
        "symmetricSimulationSamplerperbin": sym_block_sizes,
        "symmetricSimulationQuantile": sym_quantile,
        "symmetricExpectedQuantiles": sym_scale,
        "symmetricExpectedCovariance": sym_covar,
        "asymmetricSimulationSamplerperbin": asym_block_sizes,
        "asymmetricSimulationQuantile": asym_quantile,
        "asymmetricExpectedQuantiles": asym_scale,
        "asymmetricExpectedCovariance": asym_covar,
    }
