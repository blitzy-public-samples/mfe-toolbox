"""Pytest tests for the Ljung-Box serial correlation test.

Comprehensive test suite for ``mfe_toolbox.tests.ljungbox.ljungbox()``,
verifying numerical parity (±1e-6 atol, ±1e-4 rtol) against MATLAB reference
fixtures, input validation, return types/shapes, Q-stat monotonicity, and
chi-squared degrees of freedom.

Migrated from tests/ljungbox.m — MFE Toolbox v4.0 (Kevin Sheppard)
Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.

Test organisation
-----------------
Phase 1 — Unit tests (no fixture dependency)
Phase 2 — Input validation tests
Phase 3 — Parity tests (against MATLAB fixtures)
Phase 4 — Return type and shape tests
"""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import chi2

from mfe_toolbox.tests.ljungbox import ljungbox
from tests.conftest import ATOL, RTOL, load_fixture_npy, assert_allclose


# ===========================================================================
# Phase 1: Unit Tests (no fixture dependency)
# ===========================================================================


class TestLjungboxUnit:
    """Unit tests for ljungbox() — no MATLAB fixture dependency."""

    def test_ljungbox_iid_data(self, rng: np.random.Generator) -> None:
        """IID data should produce high p-values (no serial correlation).

        Generate 500 IID standard-normal observations with the shared seeded
        RNG.  Under H0 most p-values should exceed the 5 % significance
        threshold.
        """
        data = rng.standard_normal(500)
        q, pval = ljungbox(data, 10)

        # Shape checks
        assert q.shape == (10,), f"Expected q shape (10,), got {q.shape}"
        assert pval.shape == (10,), f"Expected pval shape (10,), got {pval.shape}"

        # Under H0 at least half the p-values should be > 0.05
        n_not_rejected = int(np.sum(pval > 0.05))
        assert n_not_rejected >= 5, (
            f"Expected most p-values > 0.05 for IID data; "
            f"only {n_not_rejected}/10 exceeded 0.05"
        )

    def test_ljungbox_serially_correlated_data(self, rng: np.random.Generator) -> None:
        """AR(1) data with phi=0.8 should show significant serial correlation.

        The Ljung-Box Q statistic at lag 1 should be large and the p-value
        small, indicating rejection of the no-serial-correlation null.
        """
        T = 500
        y = np.zeros(T)
        e = rng.standard_normal(T)
        # Ref: AR(1) recursion y[t] = 0.8*y[t-1] + e[t]
        for t in range(1, T):
            y[t] = 0.8 * y[t - 1] + e[t]

        q, pval = ljungbox(y, 10)

        # AR(1) with phi=0.8 should be easily detected
        assert pval[0] < 0.05, (
            f"Expected p-value at lag 1 < 0.05 for AR(1), got {pval[0]:.6f}"
        )
        assert q[0] > 10.0, (
            f"Expected large Q-stat at lag 1 for AR(1), got {q[0]:.4f}"
        )

    def test_ljungbox_q_monotonically_nondecreasing(self, rng: np.random.Generator) -> None:
        """Q-statistics must be monotonically non-decreasing across lags.

        This is a structural invariant of the Ljung-Box formula: each
        Q(L) = Q(L-1) + T*(T+2)*ac[L]^2/(T-L), adding a non-negative term.
        """
        data = rng.standard_normal(300)
        q, _ = ljungbox(data, 15)
        diffs = np.diff(q)
        assert np.all(diffs >= -1e-12), (
            f"Q-stats must be non-decreasing; min diff = {diffs.min():.2e}"
        )

    def test_ljungbox_single_lag(self, rng: np.random.Generator) -> None:
        """lags=1 should return single-element arrays for both q and pval."""
        data = rng.standard_normal(100)
        q, pval = ljungbox(data, 1)
        assert q.shape == (1,), f"Expected q shape (1,), got {q.shape}"
        assert pval.shape == (1,), f"Expected pval shape (1,), got {pval.shape}"

    @pytest.mark.parametrize("lags", [1, 5, 10, 20])
    def test_ljungbox_multiple_lag_specs(self, rng: np.random.Generator, lags: int) -> None:
        """Parametrised test: output shapes must match the requested lag count."""
        data = rng.standard_normal(500)
        q, pval = ljungbox(data, lags)
        assert q.shape == (lags,), f"Expected q shape ({lags},), got {q.shape}"
        assert pval.shape == (lags,), f"Expected pval shape ({lags},), got {pval.shape}"

    def test_ljungbox_formula_manual_check(self) -> None:
        """Verify the Ljung-Box formula on a small dataset by manual computation.

        Manually compute sample autocorrelations using the same OLS-regression
        method that ``sacf`` uses internally (regress data[L:] on
        [ones, data[0:T-L]] and take the slope), then apply the Ljung-Box
        Q-statistic formula and compare against ljungbox() output.

        Ljung-Box Q(L) = T*(T+2) * sum_{k=1}^{L} rho_hat_k^2 / (T - k)

        Ref: ljungbox.m:58-59 — MATLAB 1-indexed loop, Python 0-indexed.
        Ref: sacf.m:77-81 — OLS regression for autocorrelation.
        """
        # Deterministic small dataset for reproducibility
        local_rng = np.random.default_rng(123)
        data = local_rng.standard_normal(50)
        T = len(data)
        lags = 3

        q, pval = ljungbox(data, lags)

        # Manually compute autocorrelations using the OLS method from sacf:
        # For each lag L, regress data[L:] on [ones(T-L,1), data[0:T-L]]
        # and extract the slope coefficient (phi[1]).
        # Ref: sacf.m:77-81
        ac_manual = np.zeros(lags)
        for k in range(1, lags + 1):
            y = data[k:T]
            n_obs = T - k
            x = np.ones((n_obs, 2))
            x[:, 1] = data[0:T - k]
            phi, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            ac_manual[k - 1] = phi[1]

        # Compute Q statistics manually using the Ljung-Box formula
        # Ref: ljungbox.m:58-59
        for L in range(1, lags + 1):
            divisors = T - np.arange(1, L + 1, dtype=np.float64)
            q_manual = T * (T + 2) * np.sum(ac_manual[:L] ** 2 / divisors)
            npt.assert_allclose(
                q[L - 1], q_manual, atol=1e-6, rtol=1e-4,
                err_msg=f"Q-stat manual check mismatch at lag {L}"
            )

    def test_ljungbox_chi2_degrees_of_freedom(self) -> None:
        """Verify that the k-th p-value uses chi2(k) degrees of freedom.

        For each lag k, the p-value should satisfy:
            pval[k-1] = 1 - chi2.cdf(q[k-1], k)

        Ref: ljungbox.m:61 — pval = 1 - chi2cdf(q, (1:lags)')
        """
        local_rng = np.random.default_rng(99)
        data = local_rng.standard_normal(200)
        lags = 10
        q, pval = ljungbox(data, lags)

        for k in range(1, lags + 1):
            # Independently compute the p-value using chi2(k) df
            expected_pval = 1.0 - chi2.cdf(q[k - 1], k)
            npt.assert_allclose(
                pval[k - 1], expected_pval, atol=1e-12,
                err_msg=(
                    f"p-value at lag {k}: expected chi2({k}) df. "
                    f"Got pval={pval[k - 1]:.10e}, expected={expected_pval:.10e}"
                )
            )


# ===========================================================================
# Phase 2: Input Validation Tests
# ===========================================================================


class TestLjungboxValidation:
    """Input validation tests for ljungbox() — error handling and edge cases."""

    def test_ljungbox_requires_two_args(self) -> None:
        """Both data and lags are mandatory positional arguments.

        Calling with fewer than 2 arguments must raise TypeError (Python's
        standard missing-argument error).
        """
        with pytest.raises(TypeError):
            ljungbox()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            ljungbox(np.array([1.0, 2.0, 3.0]))  # type: ignore[call-arg]

    def test_ljungbox_lags_too_large_raises(self) -> None:
        """T ≤ lags must raise ValueError.

        Ref: ljungbox.m:38-39 — 'At least LAGS observations required'
        """
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # T == lags → error
        with pytest.raises(ValueError, match="[Oo]bservations|LAGS"):
            ljungbox(data, 5)
        # T < lags → error
        with pytest.raises(ValueError, match="[Oo]bservations|LAGS"):
            ljungbox(data, 10)

    def test_ljungbox_lags_not_positive_raises(self) -> None:
        """lags=0 or negative lags must raise ValueError.

        Ref: ljungbox.m:47-48 — 'LAGS must be a positive integer'
        """
        data = np.random.default_rng(42).standard_normal(100)
        with pytest.raises(ValueError, match="positive integer"):
            ljungbox(data, 0)
        with pytest.raises(ValueError, match="positive integer"):
            ljungbox(data, -1)
        with pytest.raises(ValueError, match="positive integer"):
            ljungbox(data, -10)

    def test_ljungbox_lags_noninteger_raises(self) -> None:
        """Float lags must raise ValueError.

        Ref: ljungbox.m:47 — floor(lags)==lags check; Python implementation
        requires isinstance(lags, (int, np.integer)).
        """
        data = np.random.default_rng(42).standard_normal(100)
        with pytest.raises(ValueError, match="positive integer"):
            ljungbox(data, 5.0)
        with pytest.raises(ValueError, match="positive integer"):
            ljungbox(data, 5.5)

    def test_ljungbox_empty_data_raises(self) -> None:
        """Empty data array must raise ValueError.

        An empty array has T=0, so T <= lags is always true for any
        positive lags value.
        """
        with pytest.raises(ValueError):
            ljungbox(np.array([]), 1)
        with pytest.raises(ValueError):
            ljungbox(np.array([]), 5)

    def test_ljungbox_2d_data_flattened(self) -> None:
        """2D column-vector data should be flattened (ravel) to 1-D.

        Ref: ljungbox.m:41-42 — MATLAB transposes row vectors to columns.
        The Python implementation uses ``np.asarray(data).ravel()``, so both
        (T,1) and (1,T) shaped inputs should produce the same result as a
        flat (T,) input.
        """
        local_rng = np.random.default_rng(77)
        data_1d = local_rng.standard_normal(100)
        data_col = data_1d.reshape(-1, 1)  # Column vector (100, 1)
        data_row = data_1d.reshape(1, -1)  # Row vector (1, 100)

        q_1d, pval_1d = ljungbox(data_1d, 5)
        q_col, pval_col = ljungbox(data_col, 5)
        q_row, pval_row = ljungbox(data_row, 5)

        npt.assert_allclose(
            q_1d, q_col, atol=1e-15,
            err_msg="Column-vector (T,1) input should match flat (T,) input"
        )
        npt.assert_allclose(
            pval_1d, pval_col, atol=1e-15,
            err_msg="Column-vector pval should match flat input"
        )
        npt.assert_allclose(
            q_1d, q_row, atol=1e-15,
            err_msg="Row-vector (1,T) input should match flat (T,) input"
        )
        npt.assert_allclose(
            pval_1d, pval_row, atol=1e-15,
            err_msg="Row-vector pval should match flat input"
        )


# ===========================================================================
# Phase 3: Parity Tests (against MATLAB fixtures)
# ===========================================================================


class TestLjungboxParity:
    """MATLAB parity tests for ljungbox() — require fixture files.

    All tests in this class are marked with ``@pytest.mark.parity`` and
    gracefully skip when fixture files are not found on disk.
    """

    @pytest.mark.parity
    def test_ljungbox_parity_iid(self, tests_fixture_dir) -> None:
        """Compare IID data outputs against MATLAB fixtures.

        Loads the main ljungbox fixture and verifies Q-statistics and p-values
        at lags 5, 10, and 20 for IID (white noise) data against the MATLAB
        reference outputs.
        """
        fixture = load_fixture_npy(tests_fixture_dir, "ljungbox")
        fixture_dict = fixture.item()

        data = fixture_dict["iid_data"].ravel()

        for lags_val in fixture_dict["lags_values"]:
            lags_val = int(lags_val)
            expected_q = fixture_dict[f"iid_q{lags_val}"].ravel()
            expected_pval = fixture_dict[f"iid_pval{lags_val}"].ravel()

            q, pval = ljungbox(data, lags_val)

            assert_allclose(
                q, expected_q,
                err_msg=f"Q-stat parity failed for IID data, lags={lags_val}",
            )
            assert_allclose(
                pval, expected_pval,
                err_msg=f"p-value parity failed for IID data, lags={lags_val}",
            )

    @pytest.mark.parity
    def test_ljungbox_parity_ar1(self, tests_fixture_dir) -> None:
        """Compare AR(1) data outputs against MATLAB fixtures.

        Verifies per-element numerical parity for all lags in the AR(1) test
        case.  AR(1) data produces large Q-statistics and very small p-values.
        """
        fixture = load_fixture_npy(tests_fixture_dir, "ljungbox")
        fixture_dict = fixture.item()

        data = fixture_dict["ar_data"].ravel()

        for lags_val in fixture_dict["lags_values"]:
            lags_val = int(lags_val)
            expected_q = fixture_dict[f"ar_q{lags_val}"].ravel()
            expected_pval = fixture_dict[f"ar_pval{lags_val}"].ravel()

            q, pval = ljungbox(data, lags_val)

            assert_allclose(
                q, expected_q,
                err_msg=f"Q-stat parity failed for AR(1) data, lags={lags_val}",
            )
            assert_allclose(
                pval, expected_pval,
                err_msg=f"p-value parity failed for AR(1) data, lags={lags_val}",
            )

    @pytest.mark.parity
    @pytest.mark.parametrize(
        "data_type", ["iid", "ar", "garch"],
        ids=["iid", "ar1", "garch"],
    )
    @pytest.mark.parametrize(
        "lags_val", [5, 10, 20],
        ids=["lags5", "lags10", "lags20"],
    )
    def test_ljungbox_parity_multiple_lags(
        self, tests_fixture_dir, data_type: str, lags_val: int
    ) -> None:
        """Parametrised parity test across data types and lag specifications.

        Each combination of {iid, ar, garch} × {5, 10, 20} is compared
        independently against the MATLAB reference output.
        """
        fixture = load_fixture_npy(tests_fixture_dir, "ljungbox")
        fixture_dict = fixture.item()

        data = fixture_dict[f"{data_type}_data"].ravel()
        expected_q = fixture_dict[f"{data_type}_q{lags_val}"].ravel()
        expected_pval = fixture_dict[f"{data_type}_pval{lags_val}"].ravel()

        q, pval = ljungbox(data, lags_val)

        assert_allclose(
            q, expected_q,
            err_msg=f"Q-stat parity: {data_type}, lags={lags_val}",
        )
        assert_allclose(
            pval, expected_pval,
            err_msg=f"p-value parity: {data_type}, lags={lags_val}",
        )

    @pytest.mark.parity
    def test_ljungbox_parity_split_fixtures(self, tests_fixture_dir) -> None:
        """Compare against the split fixture files (ljungbox_lb_*).

        The split fixtures provide a separate data vector and corresponding
        Q-statistics and p-values for lags=10.
        """
        lb_data = load_fixture_npy(tests_fixture_dir, "ljungbox_lb_data")
        lb_q = load_fixture_npy(tests_fixture_dir, "ljungbox_lb_q")
        lb_pval = load_fixture_npy(tests_fixture_dir, "ljungbox_lb_pval")

        data = lb_data.ravel()
        expected_q = lb_q.ravel()
        expected_pval = lb_pval.ravel()
        lags = len(expected_q)

        q, pval = ljungbox(data, lags)

        assert_allclose(
            q, expected_q,
            err_msg="Q-stat parity failed against split fixture (ljungbox_lb_q)",
        )
        assert_allclose(
            pval, expected_pval,
            err_msg="p-value parity failed against split fixture (ljungbox_lb_pval)",
        )

    @pytest.mark.parity
    def test_ljungbox_parity_garch(self, tests_fixture_dir) -> None:
        """Compare GARCH data outputs against MATLAB fixtures.

        GARCH data exhibits heteroskedasticity but may not show strong serial
        correlation in levels.  This test verifies Q and p-value parity
        regardless of the rejection outcome.
        """
        fixture = load_fixture_npy(tests_fixture_dir, "ljungbox")
        fixture_dict = fixture.item()

        data = fixture_dict["garch_data"].ravel()

        for lags_val in fixture_dict["lags_values"]:
            lags_val = int(lags_val)
            expected_q = fixture_dict[f"garch_q{lags_val}"].ravel()
            expected_pval = fixture_dict[f"garch_pval{lags_val}"].ravel()

            q, pval = ljungbox(data, lags_val)

            assert_allclose(
                q, expected_q,
                err_msg=f"Q-stat parity failed for GARCH data, lags={lags_val}",
            )
            assert_allclose(
                pval, expected_pval,
                err_msg=f"p-value parity failed for GARCH data, lags={lags_val}",
            )


# ===========================================================================
# Phase 4: Return Type and Shape Tests
# ===========================================================================


class TestLjungboxReturnTypes:
    """Return type, shape, and invariant validation tests."""

    def test_ljungbox_return_types(self, rng: np.random.Generator) -> None:
        """Both q and pval must be numpy.ndarray instances."""
        data = rng.standard_normal(200)
        q, pval = ljungbox(data, 5)
        assert isinstance(q, np.ndarray), (
            f"q should be np.ndarray, got {type(q).__name__}"
        )
        assert isinstance(pval, np.ndarray), (
            f"pval should be np.ndarray, got {type(pval).__name__}"
        )

    @pytest.mark.parametrize("lags", [1, 5, 10, 20])
    def test_ljungbox_return_shapes(
        self, rng: np.random.Generator, lags: int
    ) -> None:
        """Both q and pval must have shape (lags,) for all tested lag counts."""
        data = rng.standard_normal(500)
        q, pval = ljungbox(data, lags)
        assert q.shape == (lags,), (
            f"q shape: expected ({lags},), got {q.shape}"
        )
        assert pval.shape == (lags,), (
            f"pval shape: expected ({lags},), got {pval.shape}"
        )

    def test_ljungbox_q_nonnegative(self, rng: np.random.Generator) -> None:
        """All Q-statistics must be ≥ 0.

        The Q-stat is a sum of squared autocorrelations divided by positive
        denominators, scaled by T*(T+2) > 0, so it is non-negative by
        construction.
        """
        data = rng.standard_normal(200)
        q, _ = ljungbox(data, 10)
        assert np.all(q >= 0), (
            f"Q-stats must be non-negative: min = {q.min():.2e}"
        )

    def test_ljungbox_pval_range(self, rng: np.random.Generator) -> None:
        """All p-values must lie in the interval [0, 1].

        P-values are 1 - chi2.cdf(q, k), which is always in [0, 1] for q ≥ 0
        and k ≥ 1.
        """
        data = rng.standard_normal(200)
        _, pval = ljungbox(data, 10)
        assert np.all(pval >= 0.0), (
            f"p-values must be >= 0: min = {pval.min():.2e}"
        )
        assert np.all(pval <= 1.0), (
            f"p-values must be <= 1: max = {pval.max():.2e}"
        )

    def test_ljungbox_return_dtypes(self, rng: np.random.Generator) -> None:
        """Both q and pval should have float64 dtype."""
        data = rng.standard_normal(100)
        q, pval = ljungbox(data, 5)
        assert q.dtype == np.float64, (
            f"q dtype: expected float64, got {q.dtype}"
        )
        assert pval.dtype == np.float64, (
            f"pval dtype: expected float64, got {pval.dtype}"
        )

    def test_ljungbox_tuple_return(self, rng: np.random.Generator) -> None:
        """ljungbox() must return a tuple of exactly 2 arrays."""
        data = rng.standard_normal(100)
        result = ljungbox(data, 5)
        assert isinstance(result, tuple), (
            f"Expected tuple return, got {type(result).__name__}"
        )
        assert len(result) == 2, (
            f"Expected 2-element tuple, got length {len(result)}"
        )
