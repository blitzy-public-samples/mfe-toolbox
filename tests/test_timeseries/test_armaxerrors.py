"""
Pytest parity and correctness tests for mfe_toolbox.timeseries.armaxerrors.

This module tests the ARMAX residual filtering function, which was migrated from:
  - timeseries/armaxerrors.m   (MATLAB fallback wrapper)
  - mex_source/armaxerrors.c   (C MEX acceleration kernel)
to a Numba ``@jit(nopython=True, cache=True)`` decorated Python function.

The test suite covers:
  1. Input validation and boundary conditions
  2. Correctness of AR, MA, exogenous, constant, and sigma-scaling logic
  3. Numba JIT compilation verification (no object-mode fallback)
  4. MATLAB parity via .npy fixture comparison (ATOL=1e-6, RTOL=1e-4)
  5. Edge cases: single observation, high-order AR, non-contiguous lags

Per AAP Section 0.7.1:
  - Every migrated function MUST pass numpy.testing.assert_allclose(atol=1e-6, rtol=1e-4)
  - Numba @jit(nopython=True, cache=True) — no object mode fallback permitted
  - Execution time per function MUST NOT exceed 2x MATLAB baseline

Ref: timeseries/armaxerrors.m (Kevin Sheppard, Revision 3, 10/19/2009)
Ref: mex_source/armaxerrors.c (Kevin Sheppard, Revision 3, 10/16/2009)
"""

import time

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.armaxerrors import armaxerrors

# ---------------------------------------------------------------------------
# Numerical parity tolerances per AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Pytest Fixtures — reproducible test data
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Return a seeded random number generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def ar1_data(rng):
    """Generate AR(1) data: y(t) = 0.5*y(t-1) + e(t), T=200.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y, e)`` where ``y`` is the observed series and ``e`` is the true
        innovation sequence.
    """
    T = 200
    y = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + e[t]
    return y, e


@pytest.fixture
def arma11_data(rng):
    """Generate ARMA(1,1) data: y(t) = 0.7*y(t-1) + e(t) + 0.3*e(t-1), T=300.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y, e)`` where ``y`` is the observed series and ``e`` is the true
        innovation sequence.
    """
    T = 300
    y = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(1, T):
        y[t] = 0.7 * y[t - 1] + e[t] + 0.3 * e[t - 1]
    return y, e


# ===================================================================
# Phase 2: Input Validation Tests
# ===================================================================


class TestInputValidation:
    """Tests for basic input handling and boundary conditions."""

    def test_armaxerrors_empty_p_q(self, rng):
        """Both p and q empty, constant=0 → errors ≈ y / sigma.

        With no AR, MA, or constant terms, the residual is simply y(t)/sigma(t)
        for t >= m.  The first m elements remain zero.
        """
        T = 100
        y = rng.standard_normal(T)
        sigma = np.ones(T)
        # No constant, no AR, no MA → parameters is empty
        params = np.array([], dtype=np.float64)
        p = np.array([], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        m = 0  # Ref: armaxerrors.c — m=0 means start from first element

        errors = armaxerrors(params, p, q, 0, y, np.zeros(T), m, sigma)

        # All errors should equal y / sigma = y (since sigma=1)
        npt.assert_allclose(errors, y, atol=ATOL, rtol=RTOL)

    def test_armaxerrors_constant_only(self, rng):
        """Only constant parameter → errors = (y - constant) / sigma.

        When there are no AR, MA, or exogenous terms and constant=1,
        the residual at each t >= m is (y(t) - c) / sigma(t).
        """
        T = 100
        y = rng.standard_normal(T) + 2.0  # shift mean
        sigma = np.ones(T)
        constant_val = 2.0
        params = np.array([constant_val])
        p = np.array([], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        m = 0

        errors = armaxerrors(params, p, q, 1, y, np.zeros(T), m, sigma)

        expected = (y - constant_val) / sigma
        npt.assert_allclose(errors, expected, atol=ATOL, rtol=RTOL)


# ===================================================================
# Phase 3: Correctness Tests
# ===================================================================


class TestCorrectness:
    """Tests verifying the mathematical correctness of ARMAX residual filtering."""

    def test_armaxerrors_ar1_residuals(self, ar1_data):
        """AR(1): known parameters → recovered residuals match original innovations.

        Ref: armaxerrors.m line 42-44 — AR term subtraction loop.
        For an AR(1) with phi=0.5, errors(t) = y(t) - 0.5*y(t-1) = e(t).
        """
        y, true_e = ar1_data
        T = len(y)
        # Parameters: no constant (constant=0), phi=0.5
        params = np.array([0.5])
        p = np.array([1.0])  # MATLAB stores lags as doubles
        q = np.array([], dtype=np.float64)
        sigma = np.ones(T)
        m = 1  # Ref: armaxerrors.c — start from index 1 (max(p)=1)

        errors = armaxerrors(params, p, q, 0, y, np.zeros(T), m, sigma)

        # First element should be zero (burn-in)
        assert errors[0] == 0.0
        # From index 1 onward: errors should match true innovations
        npt.assert_allclose(errors[1:], true_e[1:], atol=1e-10, rtol=1e-10)

    def test_armaxerrors_arma11_residuals(self, arma11_data):
        """ARMA(1,1) with known params: recursive residuals converge.

        Ref: armaxerrors.m lines 42-50 — full AR + MA recursion.
        For ARMA(1,1): y(t) = 0.7*y(t-1) + e(t) + 0.3*e(t-1)
        With correct parameters and zero initial errors, the MA recursion
        should recover the true innovations after the burn-in period.
        """
        y, true_e = arma11_data
        T = len(y)
        # Parameters: no constant, phi=0.7, theta=0.3
        params = np.array([0.7, 0.3])
        p = np.array([1.0])
        q = np.array([1.0])
        sigma = np.ones(T)
        m = 1

        errors = armaxerrors(params, p, q, 0, y, np.zeros(T), m, sigma)

        # First element should be zero
        assert errors[0] == 0.0
        # MA recursion: errors(t) = y(t) - phi*y(t-1) - theta*errors(t-1)
        # Due to zero initial conditions, the recovered errors should closely
        # approximate true innovations after the initial transient
        npt.assert_allclose(errors[5:], true_e[5:], atol=1e-1, rtol=1e-1)

    def test_armaxerrors_with_exogenous(self, rng):
        """Include exogenous regressors, verify subtraction.

        Ref: armaxerrors.c lines 55-59 — exogenous term loop.
        With a single exogenous regressor and known coefficient beta,
        errors(t) = y(t) - beta * x(t).
        """
        T = 150
        x_col = rng.standard_normal(T)
        beta = 1.5
        e = rng.standard_normal(T)
        y = beta * x_col + e  # Pure exogenous + noise model

        # Parameters: no constant, no AR, beta=1.5, no MA
        params = np.array([1.5])
        p = np.array([], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        sigma = np.ones(T)
        m = 0

        # Pass x as 2D (T, 1) for exogenous regressors
        x_2d = x_col.reshape(-1, 1)
        errors = armaxerrors(params, p, q, 0, y, x_2d, m, sigma)

        # errors should equal e (the true innovations)
        npt.assert_allclose(errors, e, atol=ATOL, rtol=RTOL)

    def test_armaxerrors_sigma_scaling(self, rng):
        """Non-unit sigma scales residuals correctly: errors = raw_resid / sigma.

        Ref: armaxerrors.c lines 66-69 — post-loop sigma normalisation.
        The sigma division happens AFTER the recursion, so
        errors(t) = (y(t) - model(t)) / sigma(t).
        """
        T = 100
        y = rng.standard_normal(T)
        sigma = np.abs(rng.standard_normal(T)) + 0.5  # Positive sigma values

        params = np.array([], dtype=np.float64)
        p = np.array([], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        m = 0

        errors = armaxerrors(params, p, q, 0, y, np.zeros(T), m, sigma)

        # With no parameters, errors = y / sigma
        expected = y / sigma
        npt.assert_allclose(errors, expected, atol=ATOL, rtol=RTOL)

    def test_armaxerrors_ma_recursion(self, rng):
        """MA component: verify errors(t) -= theta * errors(t-1) recursion.

        Ref: armaxerrors.c lines 60-64 — MA term recursion with lagged errors.
        Construct data from a pure MA(1) process and verify the recursive
        residual computation recovers the innovations.
        """
        T = 200
        theta = 0.4
        e = rng.standard_normal(T)
        y = np.zeros(T)
        # Pure MA(1): y(t) = e(t) + theta * e(t-1)
        y[0] = e[0]
        for t in range(1, T):
            y[t] = e[t] + theta * e[t - 1]

        # Parameters: no constant, no AR, theta=0.4
        params = np.array([theta])
        p = np.array([], dtype=np.float64)
        q = np.array([1.0])
        sigma = np.ones(T)
        m = 1

        errors = armaxerrors(params, p, q, 0, y, np.zeros(T), m, sigma)

        # The MA recursion should recover the innovations after initial transient
        # errors(0) = 0 (burn-in)
        assert errors[0] == 0.0
        # Manually check: errors(1) = y(1) - theta * errors(0)
        # = (e[1] + theta*e[0]) - theta*0 = e[1] + theta*e[0]
        # This differs from e[1] because the initial error is zero, not e[0].
        # But subsequent errors converge to true innovations.
        # Check convergence after transient
        npt.assert_allclose(errors[10:], e[10:], atol=5e-2, rtol=5e-2)

    def test_armaxerrors_output_shape(self, rng):
        """Output shape matches input y shape (T,).

        Per AAP: All return types must be numpy.ndarray.
        """
        T = 50
        y = rng.standard_normal(T)
        sigma = np.ones(T)
        params = np.array([0.1, 0.5])  # constant + 1 AR
        p = np.array([1.0])
        q = np.array([], dtype=np.float64)

        errors = armaxerrors(params, p, q, 1, y, np.zeros(T), 1, sigma)

        assert isinstance(errors, np.ndarray)
        assert errors.shape == (T,)

    def test_armaxerrors_m_parameter(self, rng):
        """Varying m: errors before index m should be zero.

        Ref: armaxerrors.c lines 40-43 — zero burn-in for first m elements.
        """
        T = 100
        y = rng.standard_normal(T)
        sigma = np.ones(T)
        params = np.array([], dtype=np.float64)
        p = np.array([], dtype=np.float64)
        q = np.array([], dtype=np.float64)

        for m_val in [0, 1, 5, 10, 50]:
            errors = armaxerrors(params, p, q, 0, y, np.zeros(T), m_val, sigma)
            # First m_val elements should be zero
            npt.assert_array_equal(errors[:m_val], np.zeros(m_val))
            # Remaining elements should equal y / sigma = y
            if m_val < T:
                npt.assert_allclose(
                    errors[m_val:], y[m_val:], atol=ATOL, rtol=RTOL
                )

    def test_armaxerrors_manual_recursion(self, rng):
        """Verify ARMAX recursion against manual step-by-step computation.

        Build an ARMA(1,1) with constant, exogenous, and sigma, then manually
        compute each step to verify the function produces identical results.
        Ref: armaxerrors.c lines 44-69 — full recursion + sigma division.
        """
        T = 20
        rng_local = np.random.default_rng(99)
        y = rng_local.standard_normal(T)
        x_col = rng_local.standard_normal(T)
        sigma = np.abs(rng_local.standard_normal(T)) + 0.5

        # ARMA(1,1) with constant + exogenous
        # params = [const, phi1, beta1, theta1]
        const_val = 0.1
        phi1 = 0.6
        beta1 = 0.3
        theta1 = 0.2
        params = np.array([const_val, phi1, beta1, theta1])
        p = np.array([1.0])
        q = np.array([1.0])
        m = 1

        # Manual computation matching the C algorithm
        e_manual = np.zeros(T)
        for t in range(m, T):
            e_manual[t] = y[t]
            e_manual[t] -= const_val                     # constant
            e_manual[t] -= phi1 * y[t - 1]               # AR(1)
            e_manual[t] -= beta1 * x_col[t]              # exogenous
            e_manual[t] -= theta1 * e_manual[t - 1]      # MA(1)
        # Post-loop sigma normalisation
        for t in range(m, T):
            e_manual[t] = e_manual[t] / sigma[t]

        x_2d = x_col.reshape(-1, 1)
        errors = armaxerrors(params, p, q, 1, y, x_2d, m, sigma)

        npt.assert_allclose(errors, e_manual, atol=1e-12, rtol=1e-12)


# ===================================================================
# Phase 4: Numba JIT Specific Tests
# ===================================================================


class TestNumbaJIT:
    """Tests verifying Numba JIT compilation and performance characteristics."""

    def test_armaxerrors_numba_compiled(self):
        """Verify the core function is Numba JIT-compiled.

        Per AAP Section 0.7.1: @numba.jit(nopython=True, cache=True)
        is required for all routines where MEX C was used — no object
        mode fallback permitted.
        """
        import numba
        from mfe_toolbox.timeseries.armaxerrors import _armaxerrors_core

        # The inner function should be a Numba Dispatcher (JIT-compiled)
        assert isinstance(
            _armaxerrors_core, numba.core.dispatcher.Dispatcher
        ), (
            "_armaxerrors_core must be a Numba Dispatcher, not a plain Python "
            f"function. Got: {type(_armaxerrors_core)}"
        )

    def test_armaxerrors_nopython_mode(self):
        """Verify nopython mode is active (no object-mode fallback).

        Per AAP Section 0.7.1: No object mode fallback is permitted.
        A successful call with typed arrays confirms nopython compilation.
        """
        T = 50
        y = np.zeros(T, dtype=np.float64)
        y[1:] = np.random.default_rng(0).standard_normal(T - 1)
        sigma = np.ones(T, dtype=np.float64)
        params = np.array([0.5], dtype=np.float64)
        p = np.array([1.0], dtype=np.float64)
        q = np.array([], dtype=np.float64)

        # This should succeed without Numba falling back to object mode
        errors = armaxerrors(params, p, q, 0, y, np.zeros(T), 1, sigma)
        assert errors.dtype == np.float64

    def test_armaxerrors_large_data_performance(self):
        """T=10000, ensure runs in reasonable time (< 2 seconds).

        Per AAP Section 0.7.1: Execution time MUST NOT exceed 2× MATLAB
        baseline. For T=10000 the MATLAB MEX completes in ~0.01s, so
        Numba JIT should be well under 1s (even including first-call
        compilation overhead).
        """
        T = 10000
        rng_perf = np.random.default_rng(123)
        y = rng_perf.standard_normal(T)
        sigma = np.ones(T)
        params = np.array([0.1, 0.5, 0.3])  # constant + AR(1) + MA(1)
        p = np.array([1.0])
        q = np.array([1.0])

        # Warm up JIT (first call may be slower)
        _ = armaxerrors(params, p, q, 1, y[:100], np.zeros(100), 1, sigma[:100])

        start = time.perf_counter()
        errors = armaxerrors(params, p, q, 1, y, np.zeros(T), 1, sigma)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, (
            f"armaxerrors with T={T} took {elapsed:.3f}s, "
            f"exceeding the 2s performance threshold"
        )
        assert errors.shape == (T,)


# ===================================================================
# Phase 5: Parity Tests — MATLAB fixture comparison
# ===================================================================


class TestParity:
    """MATLAB parity tests comparing Python outputs against Octave/MATLAB fixtures.

    Per AAP Section 0.7.1: Every migrated function MUST pass
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
    against MATLAB-generated fixtures.
    """

    @pytest.mark.parity
    def test_armaxerrors_parity_arma11(self, timeseries_fixture_dir):
        """ARMA(1,1) parity: scenario1 from fixtures.

        Fixture scenario1: ARMA(1,1) p=[1], q=[1], constant=1, T=1000,
        parameters=[0.05, 0.7, 0.3], sigma=ones, no exogenous.
        """
        fixture_path = timeseries_fixture_dir / "armaxerrors.npy"
        if not fixture_path.exists():
            pytest.skip("Fixture file not found: armaxerrors.npy")
        data = np.load(fixture_path, allow_pickle=True).item()
        sc = data["scenario1"]
        y = data["y"]
        T = int(data["T"])

        errors = armaxerrors(
            parameters=sc["parameters"],
            p=sc["p"].astype(np.float64),
            q=sc["q"].astype(np.float64),
            constant=int(sc["constant"]),
            y=y,
            x=np.zeros(T),  # no exogenous → 1D triggers k=0
            m=int(sc["m"]),
            sigma=np.ones(T),
        )

        npt.assert_allclose(errors, sc["errors"], atol=ATOL, rtol=RTOL)

    @pytest.mark.parity
    def test_armaxerrors_parity_arma21(self, timeseries_fixture_dir):
        """ARMA(2,1) parity: scenario2 from fixtures.

        Fixture scenario2: ARMA(2,1) p=[1,2], q=[1], constant=1, T=1000,
        parameters=[0.05, 0.5, 0.2, 0.3], m=2, sigma=ones, no exogenous.
        """
        fixture_path = timeseries_fixture_dir / "armaxerrors.npy"
        if not fixture_path.exists():
            pytest.skip("Fixture file not found: armaxerrors.npy")
        data = np.load(fixture_path, allow_pickle=True).item()
        sc = data["scenario2"]
        y = data["y"]
        T = int(data["T"])

        errors = armaxerrors(
            parameters=sc["parameters"],
            p=sc["p"].astype(np.float64),
            q=sc["q"].astype(np.float64),
            constant=int(sc["constant"]),
            y=y,
            x=np.zeros(T),
            m=int(sc["m"]),
            sigma=np.ones(T),
        )

        npt.assert_allclose(errors, sc["errors"], atol=ATOL, rtol=RTOL)

    @pytest.mark.parity
    def test_armaxerrors_parity_armax_exog(self, timeseries_fixture_dir):
        """ARMAX(1,1) with exogenous parity: scenario3 from fixtures.

        Fixture scenario3: ARMAX(1,1) p=[1], q=[1], constant=1, T=1000,
        parameters=[0.05, 0.7, 0.5, 0.3], x=T×1 exogenous, sigma=ones.
        """
        fixture_path = timeseries_fixture_dir / "armaxerrors.npy"
        if not fixture_path.exists():
            pytest.skip("Fixture file not found: armaxerrors.npy")
        data = np.load(fixture_path, allow_pickle=True).item()
        sc = data["scenario3"]
        y = data["y"]
        T = int(data["T"])
        x_exog = data["x_exog"].reshape(-1, 1)  # T×1 exogenous

        errors = armaxerrors(
            parameters=sc["parameters"],
            p=sc["p"].astype(np.float64),
            q=sc["q"].astype(np.float64),
            constant=int(sc["constant"]),
            y=y,
            x=x_exog,
            m=int(sc["m"]),
            sigma=np.ones(T),
        )

        npt.assert_allclose(errors, sc["errors"], atol=ATOL, rtol=RTOL)

    @pytest.mark.parity
    def test_armaxerrors_parity_ar2_no_ma(self, timeseries_fixture_dir):
        """AR(2) no MA parity: scenario4 from fixtures.

        Fixture scenario4: AR(2) p=[1,2], q=[], constant=1, T=1000,
        parameters=[0.05, 0.5, 0.2], m=2, sigma=ones, no exogenous.
        """
        fixture_path = timeseries_fixture_dir / "armaxerrors.npy"
        if not fixture_path.exists():
            pytest.skip("Fixture file not found: armaxerrors.npy")
        data = np.load(fixture_path, allow_pickle=True).item()
        sc = data["scenario4"]
        y = data["y"]
        T = int(data["T"])

        errors = armaxerrors(
            parameters=sc["parameters"],
            p=sc["p"].astype(np.float64),
            q=sc["q"].astype(np.float64),
            constant=int(sc["constant"]),
            y=y,
            x=np.zeros(T),
            m=int(sc["m"]),
            sigma=np.ones(T),
        )

        npt.assert_allclose(errors, sc["errors"], atol=ATOL, rtol=RTOL)


# ===================================================================
# Phase 6: Edge Cases
# ===================================================================


class TestEdgeCases:
    """Edge-case tests for boundary conditions and unusual configurations."""

    def test_armaxerrors_single_observation(self):
        """T=1 edge case: single observation, m=0 → one-element output.

        With T=1 and m=0, the function should return a single-element array
        containing y[0] / sigma[0] (no AR/MA terms apply with 0 lags).
        """
        y = np.array([3.0])
        sigma = np.array([2.0])
        params = np.array([], dtype=np.float64)
        p = np.array([], dtype=np.float64)
        q = np.array([], dtype=np.float64)

        errors = armaxerrors(params, p, q, 0, y, np.zeros(1), 0, sigma)

        assert errors.shape == (1,)
        npt.assert_allclose(errors[0], 3.0 / 2.0, atol=ATOL)

    def test_armaxerrors_single_observation_with_m_equal_t(self):
        """T=1, m=1 → all elements are zero (entire series is burn-in)."""
        y = np.array([5.0])
        sigma = np.array([1.0])
        params = np.array([0.5], dtype=np.float64)
        p = np.array([1.0], dtype=np.float64)
        q = np.array([], dtype=np.float64)

        errors = armaxerrors(params, p, q, 0, y, np.zeros(1), 1, sigma)

        assert errors.shape == (1,)
        # m=T=1 → range(1, 1) is empty → all zeros
        npt.assert_array_equal(errors, np.zeros(1))

    def test_armaxerrors_high_order_ar(self, rng):
        """p=[1,2,3,4,5], large AR order.

        Ref: armaxerrors.c lines 51-54 — AR loop iterates over all lags.
        Verify the function handles AR(5) without error and produces
        correct residuals.
        """
        T = 200
        y = rng.standard_normal(T)
        sigma = np.ones(T)
        # AR(5) with known coefficients
        phi = np.array([0.3, -0.1, 0.05, 0.02, -0.01])
        params = phi.copy()
        p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = np.array([], dtype=np.float64)
        m = 5  # max lag = 5

        errors = armaxerrors(params, p, q, 0, y, np.zeros(T), m, sigma)

        # First 5 elements should be zero
        npt.assert_array_equal(errors[:5], np.zeros(5))
        # Manually verify one value: errors[5]
        expected_5 = y[5]
        for i in range(5):
            expected_5 -= phi[i] * y[5 - int(p[i])]
        expected_5 /= sigma[5]
        npt.assert_allclose(errors[5], expected_5, atol=1e-12)

    def test_armaxerrors_non_contiguous_lags(self, rng):
        """p=[1,3,5], q=[2,4] (non-contiguous lag indices).

        Ref: armaxerrors.c — uses p[i] and q[i] as arbitrary lag indices,
        not necessarily contiguous. Verify the function handles gaps in
        lag specification correctly.
        """
        T = 100
        y = rng.standard_normal(T)
        sigma = np.ones(T)
        # params = [phi_1, phi_3, phi_5, theta_2, theta_4]
        params = np.array([0.3, -0.1, 0.05, 0.2, -0.1])
        p = np.array([1.0, 3.0, 5.0])
        q = np.array([2.0, 4.0])
        m = 5  # max(max(p), max(q)) = max(5, 4) = 5

        errors = armaxerrors(params, p, q, 0, y, np.zeros(T), m, sigma)

        assert errors.shape == (T,)
        # First m elements zero
        npt.assert_array_equal(errors[:m], np.zeros(m))

        # Manually verify errors[5]
        e_manual = np.zeros(T)
        for t in range(m, T):
            e_manual[t] = y[t]
            e_manual[t] -= 0.3 * y[t - 1]    # phi_1 * y(t-1)
            e_manual[t] -= (-0.1) * y[t - 3]  # phi_3 * y(t-3)
            e_manual[t] -= 0.05 * y[t - 5]    # phi_5 * y(t-5)
            e_manual[t] -= 0.2 * e_manual[t - 2]    # theta_2 * e(t-2)
            e_manual[t] -= (-0.1) * e_manual[t - 4]  # theta_4 * e(t-4)
        # Sigma normalisation
        for t in range(m, T):
            e_manual[t] /= sigma[t]

        npt.assert_allclose(errors, e_manual, atol=1e-12, rtol=1e-12)

    def test_armaxerrors_all_m_zero_burn_in(self, rng):
        """When m equals T, entire output should be zeros.

        Ref: armaxerrors.c — loop range(m, T) is empty when m >= T.
        Sigma division loop range(m, T) is also empty, so zeros stay.
        """
        T = 50
        y = rng.standard_normal(T)
        sigma = np.ones(T)
        params = np.array([0.5], dtype=np.float64)
        p = np.array([1.0])
        q = np.array([], dtype=np.float64)

        errors = armaxerrors(params, p, q, 0, y, np.zeros(T), T, sigma)

        npt.assert_array_equal(errors, np.zeros(T))

    def test_armaxerrors_multiple_exogenous(self, rng):
        """Verify correct handling of multiple exogenous regressors (k > 1).

        Ref: armaxerrors.c lines 55-59 — exogenous loop iterates over k columns.
        """
        T = 100
        k = 3
        x = rng.standard_normal((T, k))
        beta = np.array([0.5, -0.3, 0.8])
        e_true = rng.standard_normal(T)
        y = x @ beta + e_true  # y = X*beta + e

        # No constant, no AR, no MA — just exogenous
        params = beta.copy()
        p = np.array([], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        sigma = np.ones(T)
        m = 0

        errors = armaxerrors(params, p, q, 0, y, x, m, sigma)

        npt.assert_allclose(errors, e_true, atol=ATOL, rtol=RTOL)

    def test_armaxerrors_combined_all_components(self, rng):
        """Full ARMAX with constant, AR, MA, exogenous, and non-unit sigma.

        This test exercises ALL code paths simultaneously, verifying the
        complete algorithm matches manual computation.
        """
        T = 80
        rng_local = np.random.default_rng(77)
        y = rng_local.standard_normal(T)
        x_col = rng_local.standard_normal(T)
        sigma = np.abs(rng_local.standard_normal(T)) + 1.0

        # ARMAX(2,1) with constant and 1 exogenous
        # params = [const, phi1, phi2, beta1, theta1]
        const_val = 0.2
        phi1, phi2 = 0.5, -0.15
        beta1 = 0.7
        theta1 = 0.25
        params = np.array([const_val, phi1, phi2, beta1, theta1])
        p = np.array([1.0, 2.0])
        q = np.array([1.0])
        m = 2  # max(max(p), max(q)) = max(2, 1) = 2

        # Manual computation
        e_manual = np.zeros(T)
        for t in range(m, T):
            e_manual[t] = y[t]
            e_manual[t] -= const_val
            e_manual[t] -= phi1 * y[t - 1]
            e_manual[t] -= phi2 * y[t - 2]
            e_manual[t] -= beta1 * x_col[t]
            e_manual[t] -= theta1 * e_manual[t - 1]
        for t in range(m, T):
            e_manual[t] /= sigma[t]

        x_2d = x_col.reshape(-1, 1)
        errors = armaxerrors(params, p, q, 1, y, x_2d, m, sigma)

        npt.assert_allclose(errors, e_manual, atol=1e-12, rtol=1e-12)
