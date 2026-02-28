"""
Monte Carlo simulation for range-based volatility scale factors.

Migrated from realized/realized_range_simulation.m (MFE Toolbox v4.0).
This module computes scale factors used by the realized range estimator
via Monte Carlo simulation of Brownian motion paths, with concave regression
smoothing to enforce monotonicity and concavity.

The scale factors represent E[(max(BM) - min(BM))^2] where BM is a standard
Brownian motion on [0, 1] observed at m equally-spaced points. As the number
of observation points m increases, the expected squared range approaches
the asymptotic value 4*log(2) ≈ 2.7726.
"""

import numpy as np
from scipy.optimize import minimize

# Asymptotic value of E[range(BM)^2] as observation frequency → ∞
# Ref: realized_range_simulation.m:10 — "The asymptotic value is 4*log(2)"
ASYMPTOTIC_VALUE: float = 4.0 * np.log(2.0)

# Module-level __all__ for explicit public API
__all__ = ["realized_range_simulation", "ASYMPTOTIC_VALUE"]


def realized_range_simulation(
    BB: int = 1_000_000,
    max_m: int = 100,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation for realized range scale factors.

    Computes scale factors E[range(BM)^2] for different numbers of observation
    points per block via Monte Carlo simulation, then applies concave regression
    smoothing to enforce non-decreasing and concavity constraints.

    The function simulates BB independent Brownian motion paths for each block
    size m = 2, 3, ..., max_m, computes the mean squared range, and then
    smooths the raw Monte Carlo estimates using constrained least-squares
    (concave regression) that enforces monotonicity and concavity.

    Parameters
    ----------
    BB : int, optional
        Number of Monte Carlo simulation paths per block size. Default is
        1,000,000. Must be a positive integer >= 1.
        Ref: realized_range_simulation.m:8 — ``BB = 1000000``
    max_m : int, optional
        Maximum number of observation points per block (prices per window).
        Scale factors are computed for m = 2, 3, ..., max_m. Default is 100.
        Must be an integer >= 2.
    seed : int or None, optional
        Random seed for numpy.random.default_rng for reproducibility.
        Default is None (non-deterministic).

    Returns
    -------
    scale_factors : numpy.ndarray
        Concave-regression-smoothed scale factors, shape (max_m - 1,).
        Index i corresponds to block size m = i + 2 observation points.
        Values are guaranteed non-decreasing and concave.
        Ref: realized_range_simulation.m:56 — ``concave_mean_MC``
    simulation_results : numpy.ndarray
        Raw Monte Carlo mean squared range values (with y[0]=1 theoretical
        override), shape (max_m - 1,). Before concave regression smoothing.
        Ref: realized_range_simulation.m:50 — ``orig_mean_MC``

    Raises
    ------
    ValueError
        If BB < 1 or is not an integer.
        If max_m < 2 or is not an integer.

    Notes
    -----
    For m observation points on a standard Brownian motion on [0, 1]:

    - m = 2 (start and end only): E[range^2] = 1 (theoretical, set exactly)
    - As m → ∞: E[range^2] → 4*log(2) ≈ 2.7726

    The concave regression (Ref: realized_range_simulation.m:47-56) solves:

        minimize  ||x - y||^2
        subject to  A1 * x <= 0  (non-decreasing)
                    A2 * x <= 0  (concavity)

    where A1 encodes first-difference constraints and A2 encodes
    second-difference constraints. This is equivalent to MATLAB's
    ``lsqlin(eye(n), y, [A1;A2], zeros(2*n-3, 1))``.

    The MATLAB implementation uses subsampling of a single long Brownian path
    at different resolutions corresponding to factors of 23400 (the number of
    seconds in a standard US trading day). The Python implementation simplifies
    this by generating independent paths for each block size m, which is
    statistically equivalent but more straightforward to implement.

    Examples
    --------
    >>> scale, raw = realized_range_simulation(BB=10000, max_m=10, seed=42)
    >>> scale.shape
    (9,)
    >>> scale[0]  # m=2 observation points: E[range^2] = 1
    1.0
    """
    # --- Parameter validation ---
    # Ref: MATLAB error() → Python raise ValueError()
    if not isinstance(BB, (int, np.integer)) or BB < 1:
        raise ValueError(
            f"BB must be a positive integer >= 1. Received: {BB}"
        )
    if not isinstance(max_m, (int, np.integer)) or max_m < 2:
        raise ValueError(
            f"max_m must be an integer >= 2. Received: {max_m}"
        )

    # Initialize reproducible random number generator
    # Ref: MATLAB randn → Python numpy.random.default_rng().standard_normal()
    rng = np.random.default_rng(seed)

    # Number of scale factors to compute (for m = 2, 3, ..., max_m)
    n = max_m - 1

    # Array to hold raw Monte Carlo mean squared ranges
    # Ref: realized_range_simulation.m:28 — ``MC = zeros(BB, M)``
    raw_means = np.zeros(n)

    # --- Monte Carlo simulation ---
    # For each block size m (number of observation points per window):
    #   - Generate BB independent Brownian motion paths on [0, 1] with m points
    #   - Each path has (m-1) increments, each ~ N(0, 1/(m-1))
    #   - Total variance at endpoint = (m-1) * (1/(m-1)) = 1
    #   - Compute squared range and average across all BB paths
    #
    # In the MATLAB source (realized_range_simulation.m:30-44), a single long
    # BM of maxM steps is generated per simulation and subsampled at different
    # gap sizes to produce paths of different lengths. The Python version uses
    # an equivalent approach of generating independent paths per block size.
    for m in range(2, max_m + 1):
        # Number of BM increments for m observation points
        # Ref: realized_range_simulation.m:33 — randn(maxM, 1)/sqrt(maxM)
        # MATLAB generates maxM increments for maxM+1 points;
        # Python generates (m-1) increments for m observation points.
        # 0-based indexing: result stored at index m-2
        num_increments = m - 1

        # Generate BM increments: each ~ N(0, 1/(m-1)) so total path
        # variance at the endpoint equals 1.
        # Ref: realized_range_simulation.m:33 — randn(maxM, 1)/sqrt(maxM)
        increments = rng.standard_normal((BB, num_increments)) / np.sqrt(
            num_increments
        )

        # Construct paths starting from 0
        # Ref: realized_range_simulation.m:33 — [0; cumsum(randn(...))]
        # np.column_stack treats the 1D np.zeros(BB) as a column vector,
        # then concatenates with the (BB, num_increments) cumsum result
        # to produce a (BB, m) array of paths.
        paths = np.column_stack(
            [np.zeros(BB), np.cumsum(increments, axis=1)]
        )

        # Compute squared range for each path
        # Ref: realized_range_simulation.m:36 — (max(x2)-min(x2))^2
        # MATLAB: max(x,[],2) - min(x,[],2) operates along dim 2;
        # Python: np.max(paths, axis=1) - np.min(paths, axis=1) along axis 1
        ranges = np.max(paths, axis=1) - np.min(paths, axis=1)

        # Store mean squared range
        # 0-based indexing: m=2 stored at index 0, m=3 at index 1, etc.
        # Ref: realized_range_simulation.m:48 — y = mean(MC)
        raw_means[m - 2] = np.mean(ranges ** 2)

    # Override first element with exact theoretical value for m=2
    # For m=2 (start and end point only, i.e., one increment), the range
    # is |BM(1)|, so E[range^2] = E[BM(1)^2] = Var(BM(1)) = 1.
    # Ref: realized_range_simulation.m:49 — ``y(1) = 1``
    # Note: MATLAB 1-based y(1) corresponds to Python 0-based raw_means[0]
    raw_means[0] = 1.0

    # Save raw results (including the y[0]=1 override) as simulation_results
    # Ref: realized_range_simulation.m:50 — ``orig_mean_MC = y``
    simulation_results = raw_means.copy()

    # --- Concave regression smoothing ---
    # Minimize ||x - y||^2 subject to:
    #   A1 * x <= 0 (non-decreasing constraint)
    #   A2 * x <= 0 (concavity constraint)
    # Ref: realized_range_simulation.m:47-56
    y = raw_means  # target vector for regression

    # Build identity matrix for constraint construction
    # Ref: realized_range_simulation.m:52 — ``x = eye(n)``
    I_n = np.eye(n)

    # A1: first-difference constraint matrix (n-1 rows × n cols)
    # Row i: [0...0, 1, -1, 0...0] → x[i] - x[i+1] <= 0 → non-decreasing
    # Constructed as I[:-1] - I[1:], matching the MATLAB pattern:
    # Ref: realized_range_simulation.m:53
    # ``A1 = [eye(n-1) zeros(n-1,1)] + [zeros(n-1,1) -eye(n-1)]``
    if n >= 2:
        A1 = I_n[:-1] - I_n[1:]
    else:
        A1 = np.zeros((0, n))

    # A2: second-difference constraint matrix (n-2 rows × n cols)
    # Row i: [0...0, 1, -2, 1, 0...0] → x[i] - 2*x[i+1] + x[i+2] <= 0
    # This enforces concavity (diminishing marginal increase).
    # Ref: realized_range_simulation.m:54
    # ``A2 = [eye(n-2) zeros(n-2,2)] + [zeros(n-2,1) -2*eye(n-2)
    #         zeros(n-2,1)] + [zeros(n-2,2) eye(n-2)]``
    if n >= 3:
        A2 = I_n[:-2] - 2.0 * I_n[1:-1] + I_n[2:]
    else:
        A2 = np.zeros((0, n))

    # Stack all inequality constraints into a single matrix
    # Ref: realized_range_simulation.m:56 — ``[A1; A2]``
    A_ub = np.vstack([A1, A2])

    # Number of constraint rows: (n-1) + (n-2) = 2*n-3 for n >= 3
    num_constraints = A_ub.shape[0]

    if num_constraints > 0:
        # Solve constrained least-squares via scipy.optimize.minimize (SLSQP)
        # Replaces MATLAB: lsqlin(eye(n), y, [A1;A2], zeros(2*n-3, 1))
        #
        # lsqlin(C, d, A, b) minimizes ||C*x - d||^2 s.t. A*x <= b
        # With C=eye(n), d=y, b=0: minimize ||x - y||^2 s.t. A_ub*x <= 0
        #
        # In scipy SLSQP, 'ineq' constraint means fun(x) >= 0.
        # We need A_ub @ x <= 0, equivalent to -(A_ub @ x) >= 0.
        # Ref: realized_range_simulation.m:55-56

        def _objective(x: np.ndarray) -> float:
            """Objective: 0.5 * ||x - y||^2."""
            diff = x - y
            return 0.5 * float(np.sum(diff ** 2))

        def _gradient(x: np.ndarray) -> np.ndarray:
            """Gradient of objective: x - y."""
            return x - y

        # Linear inequality constraints: -(A_ub @ x) >= 0 ⟺ A_ub @ x <= 0
        # Ref: realized_range_simulation.m:55 — ``b = zeros(2*n-3, 1)``
        constraints = {
            "type": "ineq",
            "fun": lambda x: -(A_ub @ x),
            "jac": lambda x: -A_ub,
        }

        # Solve the quadratic program using SLSQP
        result = minimize(
            _objective,
            y.copy(),
            method="SLSQP",
            jac=_gradient,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        scale_factors = np.array(result.x)
    else:
        # No constraints (n <= 1): unconstrained solution is exactly y
        scale_factors = np.array(y.copy())

    return scale_factors, simulation_results
