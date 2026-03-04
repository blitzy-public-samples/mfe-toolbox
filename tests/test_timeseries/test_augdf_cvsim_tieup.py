"""
Pytest tests for mfe_toolbox.timeseries.augdf_cvsim_tieup

Tests the ADF critical value simulation tie-up function, which aggregates
Monte Carlo simulation results into quantile tables for the Augmented
Dickey-Fuller test.  The MATLAB source (augdf_cvsim_tieup.m) loads two
halves of pre-simulated t-statistics, concatenates, sorts column-wise,
and extracts critical values at 103 quantile points for each of 10
sample sizes.  The Python implementation processes one case at a time,
returning ``(cv_table, cvs)`` where ``cv_table`` has shape
``(num_quantiles, num_T)`` and ``cvs`` is the 103-element quantile
vector.

References
----------
- timeseries/augdf_cvsim_tieup.m  (MATLAB source)
- mfe_toolbox/timeseries/augdf_cvsim_tieup.py  (Python implementation)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.augdf_cvsim_tieup import augdf_cvsim_tieup

# ---------------------------------------------------------------------------
# Project-standard tolerances for MATLAB numerical parity verification
# Ref: tests/conftest.py — ATOL = 1e-6, RTOL = 1e-4
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# ---------------------------------------------------------------------------
# Expected dimensions derived from MATLAB source augdf_cvsim_tieup.m
# Cvs = [.001 .005 .01:.01:.99 .995 .999]  →  103 quantile points
# Ts  = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]  →  10 sizes
# ---------------------------------------------------------------------------
NUM_QUANTILES: int = 103
MATLAB_NUM_T: int = 10

# Expected quantile grid (from MATLAB Cvs vector)
EXPECTED_CVS = np.concatenate([
    np.array([0.001, 0.005]),
    np.arange(0.01, 1.0, 0.01),
    np.array([0.995, 0.999]),
])

# Expected sample-size vector
EXPECTED_TS = np.array(
    [10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
)


# ===================================================================== #
#                          Module-scoped fixtures                        #
# ===================================================================== #

@pytest.fixture(scope="module")
def synthetic_data():
    """Generate reproducible synthetic t-statistic halves for testing.

    Returns two half-matrices drawn from a standard normal distribution
    (seed 42) with B = 5000 replications and num_T = 5 sample-size
    columns, along with B itself.
    """
    rng = np.random.default_rng(42)
    B = 5000
    num_T = 5
    half1 = rng.standard_normal((B, num_T))
    half2 = rng.standard_normal((B, num_T))
    return half1, half2, B


@pytest.fixture(scope="module")
def function_result(synthetic_data):
    """Call ``augdf_cvsim_tieup`` once with synthetic data for reuse."""
    half1, half2, B = synthetic_data
    cv_table, cvs = augdf_cvsim_tieup(half1, half2, B)
    return cv_table, cvs


# ===================================================================== #
#                            Core tests (8+)                             #
# ===================================================================== #


class TestAugdfCvsimTieupBasic:
    """Basic sanity tests for function existence, types, and shapes."""

    def test_augdf_cvsim_tieup_callable(self) -> None:
        """Verify that ``augdf_cvsim_tieup`` is a callable function."""
        assert callable(augdf_cvsim_tieup), (
            "augdf_cvsim_tieup should be callable"
        )

    def test_augdf_cvsim_tieup_output_type(self, function_result) -> None:
        """Return value must be a tuple of two numpy arrays.

        The first element is the critical-value table (2-D) and the
        second is the quantile vector (1-D).
        """
        cv_table, cvs = function_result
        assert isinstance(cv_table, np.ndarray), (
            f"cv_table should be np.ndarray, got {type(cv_table)}"
        )
        assert isinstance(cvs, np.ndarray), (
            f"cvs should be np.ndarray, got {type(cvs)}"
        )
        assert cv_table.ndim == 2, (
            f"cv_table should be 2-D, got ndim={cv_table.ndim}"
        )
        assert cvs.ndim == 1, (
            f"cvs should be 1-D, got ndim={cvs.ndim}"
        )

    def test_augdf_cvsim_tieup_table_shapes(
        self, function_result, synthetic_data
    ) -> None:
        """CV table must have (103, num_T) rows; cvs vector must have 103 elements.

        The 103 quantile points come from the MATLAB Cvs vector:
        [0.001, 0.005, 0.01:0.01:0.99, 0.995, 0.999].
        """
        half1, _half2, _B = synthetic_data
        cv_table, cvs = function_result
        num_T = half1.shape[1]
        assert cv_table.shape == (NUM_QUANTILES, num_T), (
            f"cv_table shape {cv_table.shape} != expected "
            f"({NUM_QUANTILES}, {num_T})"
        )
        assert cvs.shape == (NUM_QUANTILES,), (
            f"cvs shape {cvs.shape} != expected ({NUM_QUANTILES},)"
        )


class TestAugdfCvsimTieupProperties:
    """Statistical and structural property tests."""

    def test_augdf_cvsim_tieup_quantiles_ordered(
        self, function_result
    ) -> None:
        """For every sample-size column, critical values must be
        monotonically non-decreasing across quantile rows (lower
        quantiles → more negative, upper quantiles → more positive).
        """
        cv_table, _cvs = function_result
        for col_idx in range(cv_table.shape[1]):
            diffs = np.diff(cv_table[:, col_idx])
            assert np.all(diffs >= -1e-10), (
                f"Column {col_idx}: critical values are not monotonically "
                f"non-decreasing across quantile rows. "
                f"Min diff = {diffs.min():.2e}"
            )

    def test_augdf_cvsim_tieup_negative_lower_tail(
        self, function_result
    ) -> None:
        """Lower-tail quantiles (1 %, 5 %, 10 %) must have negative
        critical values when the input data has zero or negative mean
        (as is the case for the Dickey-Fuller distribution and for
        standard-normal synthetic data).

        Quantile positions in Cvs:
          0.01  → index 2
          0.05  → index 6
          0.10  → index 11
        """
        cv_table, cvs = function_result

        # Locate quantile positions dynamically via the cvs vector
        q_01_idx = int(np.argmin(np.abs(cvs - 0.01)))
        q_05_idx = int(np.argmin(np.abs(cvs - 0.05)))
        q_10_idx = int(np.argmin(np.abs(cvs - 0.10)))

        for col_idx in range(cv_table.shape[1]):
            assert cv_table[q_01_idx, col_idx] < 0.0, (
                f"1 % quantile (row {q_01_idx}) should be negative in "
                f"column {col_idx}, got {cv_table[q_01_idx, col_idx]:.4f}"
            )
            assert cv_table[q_05_idx, col_idx] < 0.0, (
                f"5 % quantile (row {q_05_idx}) should be negative in "
                f"column {col_idx}, got {cv_table[q_05_idx, col_idx]:.4f}"
            )
            assert cv_table[q_10_idx, col_idx] < 0.0, (
                f"10 % quantile (row {q_10_idx}) should be negative in "
                f"column {col_idx}, got {cv_table[q_10_idx, col_idx]:.4f}"
            )

    def test_augdf_cvsim_tieup_cases_differ(self) -> None:
        """When the function is called with t-statistics drawn from
        different distributions (simulating different ADF deterministic
        cases), the resulting CV tables must differ.

        In the MATLAB toolbox, case 1 (no deterministic), case 2
        (constant), and case 4 (constant + trend) produce distinct
        distributions of t-statistics, and hence distinct critical-value
        tables.
        """
        rng = np.random.default_rng(123)
        B = 2000
        num_T = 3

        # "Case A" — standard normal t-statistics (no shift)
        half1_a = rng.standard_normal((B, num_T))
        half2_a = rng.standard_normal((B, num_T))

        # "Case B" — shifted distribution (simulates a more negative DF dist)
        half1_b = rng.standard_normal((B, num_T)) - 2.0
        half2_b = rng.standard_normal((B, num_T)) - 2.0

        # "Case C" — heavier-tailed distribution
        half1_c = rng.standard_t(df=5, size=(B, num_T))
        half2_c = rng.standard_t(df=5, size=(B, num_T))

        cv_a, _ = augdf_cvsim_tieup(half1_a, half2_a, B)
        cv_b, _ = augdf_cvsim_tieup(half1_b, half2_b, B)
        cv_c, _ = augdf_cvsim_tieup(half1_c, half2_c, B)

        assert not np.allclose(cv_a, cv_b), (
            "Case A (standard normal) and Case B (shifted) should "
            "produce different CV tables"
        )
        assert not np.allclose(cv_a, cv_c), (
            "Case A (standard normal) and Case C (t-distributed) should "
            "produce different CV tables"
        )
        assert not np.allclose(cv_b, cv_c), (
            "Case B (shifted) and Case C (t-distributed) should "
            "produce different CV tables"
        )

    def test_augdf_cvsim_tieup_sample_size_effect(
        self, timeseries_fixture_dir
    ) -> None:
        """Critical values should converge as sample size T increases
        (asymptotic theory).

        Using the MATLAB fixture data where columns correspond to T in
        [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000], the
        mean absolute distance between T = 100 and T = 10 000 should
        be strictly less than between T = 10 and T = 10 000.
        """
        fixture_path = Path(timeseries_fixture_dir) / "augdf_cvsim_tieup.npy"
        if not fixture_path.exists():
            pytest.skip("Fixture augdf_cvsim_tieup.npy not found")

        data = np.load(str(fixture_path), allow_pickle=True).item()
        cv1 = data["augdf_case1_cv"]  # shape (103, 10)
        ts = data["Ts"]  # [10, 25, ..., 10000]

        # Locate column indices for T = 10, T = 100, T = 10 000
        idx_10 = int(np.argmin(np.abs(ts - 10)))
        idx_100 = int(np.argmin(np.abs(ts - 100)))
        idx_10000 = int(np.argmin(np.abs(ts - 10000)))

        dist_100_10000 = float(
            np.mean(np.abs(cv1[:, idx_100] - cv1[:, idx_10000]))
        )
        dist_10_10000 = float(
            np.mean(np.abs(cv1[:, idx_10] - cv1[:, idx_10000]))
        )

        assert dist_100_10000 < dist_10_10000, (
            f"T=100 vs T=10000 mean abs distance ({dist_100_10000:.6f}) "
            f"should be less than T=10 vs T=10000 ({dist_10_10000:.6f})"
        )


class TestAugdfCvsimTieupEdgeCases:
    """Edge-case and input-validation tests."""

    def test_augdf_cvsim_tieup_1d_input(self) -> None:
        """The function should gracefully handle 1-D input arrays by
        promoting them to 2-D column vectors (single sample-size column).
        """
        rng = np.random.default_rng(77)
        B = 500
        half1 = rng.standard_normal(B)
        half2 = rng.standard_normal(B)

        cv_table, cvs = augdf_cvsim_tieup(half1, half2, B)

        assert cv_table.shape == (NUM_QUANTILES, 1), (
            f"1-D input should yield cv_table shape ({NUM_QUANTILES}, 1), "
            f"got {cv_table.shape}"
        )
        assert cvs.shape == (NUM_QUANTILES,)

    def test_augdf_cvsim_tieup_single_column(self) -> None:
        """Explicitly 2-D input with a single column should work."""
        rng = np.random.default_rng(88)
        B = 600
        half1 = rng.standard_normal((B, 1))
        half2 = rng.standard_normal((B, 1))

        cv_table, cvs = augdf_cvsim_tieup(half1, half2, B)

        assert cv_table.shape == (NUM_QUANTILES, 1)
        assert cvs.shape == (NUM_QUANTILES,)
        # Monotonicity should still hold
        diffs = np.diff(cv_table[:, 0])
        assert np.all(diffs >= -1e-10)

    def test_augdf_cvsim_tieup_invalid_type_raises(self) -> None:
        """Non-array inputs must raise ``ValueError``."""
        with pytest.raises(ValueError, match="must be numpy arrays"):
            augdf_cvsim_tieup("bad", np.ones(10), 10)

        with pytest.raises(ValueError, match="must be numpy arrays"):
            augdf_cvsim_tieup(np.ones(10), [1, 2, 3], 3)

    def test_augdf_cvsim_tieup_negative_B_raises(self) -> None:
        """A non-positive B must raise ``ValueError``."""
        h = np.ones(10)
        with pytest.raises(ValueError, match="positive integer"):
            augdf_cvsim_tieup(h, h, -1)

        with pytest.raises(ValueError, match="positive integer"):
            augdf_cvsim_tieup(h, h, 0)

    def test_augdf_cvsim_tieup_shape_mismatch_raises(self) -> None:
        """Mismatched half shapes must raise ``ValueError``."""
        rng = np.random.default_rng(99)
        h1 = rng.standard_normal((500, 3))
        h2 = rng.standard_normal((500, 4))
        with pytest.raises(ValueError, match="same shape"):
            augdf_cvsim_tieup(h1, h2, 500)

    def test_augdf_cvsim_tieup_quantile_vector_values(
        self, function_result
    ) -> None:
        """The returned cvs vector must match the MATLAB quantile grid
        [0.001, 0.005, 0.01, 0.02, ..., 0.99, 0.995, 0.999] to
        machine precision.
        """
        _cv_table, cvs = function_result
        npt.assert_allclose(
            cvs,
            EXPECTED_CVS,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Quantile vector does not match expected MATLAB Cvs",
        )


class TestAugdfCvsimTieupParity:
    """MATLAB fixture parity tests.

    These tests load pre-computed MATLAB reference data from
    ``tests/fixtures/timeseries/augdf_cvsim_tieup.npy`` and verify
    structural and numerical agreement with the Python implementation.
    """

    @pytest.mark.parity
    def test_augdf_cvsim_tieup_parity(
        self, timeseries_fixture_dir
    ) -> None:
        """Comprehensive parity check against MATLAB-generated fixture.

        Verifies:
        1. All expected dict keys present (augdf_case1_cv, case2, case4, Cvs, Ts)
        2. Table shapes are (103, 10)
        3. Python-computed quantile vector matches MATLAB Cvs
        4. MATLAB Ts matches expected sample-size vector
        5. Monotonicity holds in every MATLAB table column
        6. The three cases produce distinct tables
        """
        fixture_path = Path(timeseries_fixture_dir) / "augdf_cvsim_tieup.npy"
        if not fixture_path.exists():
            pytest.skip("Fixture augdf_cvsim_tieup.npy not found")

        data = np.load(str(fixture_path), allow_pickle=True).item()

        # ---- 1. Verify all expected keys are present ----
        expected_keys = {
            "augdf_case1_cv",
            "augdf_case2_cv",
            "augdf_case4_cv",
            "Cvs",
            "Ts",
        }
        actual_keys = set(data.keys())
        missing = expected_keys - actual_keys
        assert not missing, f"Missing fixture keys: {missing}"

        # ---- 2. Verify table shapes ----
        for case_key in ["augdf_case1_cv", "augdf_case2_cv", "augdf_case4_cv"]:
            shape = data[case_key].shape
            assert shape == (NUM_QUANTILES, MATLAB_NUM_T), (
                f"{case_key} shape {shape} != "
                f"expected ({NUM_QUANTILES}, {MATLAB_NUM_T})"
            )

        # ---- 3. Quantile vector parity ----
        # Call the Python function with minimal data just to retrieve the
        # deterministic quantile vector produced by the implementation.
        rng_local = np.random.default_rng(999)
        B_probe = 1000
        h1_probe = rng_local.standard_normal((B_probe, 1))
        h2_probe = rng_local.standard_normal((B_probe, 1))
        _, python_cvs = augdf_cvsim_tieup(h1_probe, h2_probe, B_probe)

        npt.assert_allclose(
            python_cvs,
            data["Cvs"],
            atol=ATOL,
            rtol=RTOL,
            err_msg="Python Cvs vector does not match MATLAB fixture Cvs",
        )

        # ---- 4. Sample-size vector parity ----
        npt.assert_allclose(
            data["Ts"],
            EXPECTED_TS,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Fixture Ts does not match expected sample sizes",
        )

        # ---- 5. Monotonicity of every MATLAB table column ----
        for case_key in ["augdf_case1_cv", "augdf_case2_cv", "augdf_case4_cv"]:
            table = data[case_key]
            for col_idx in range(table.shape[1]):
                diffs = np.diff(table[:, col_idx])
                assert np.all(diffs >= -1e-10), (
                    f"MATLAB {case_key} col {col_idx}: not monotonically "
                    f"non-decreasing (min diff = {diffs.min():.2e})"
                )

        # ---- 6. Distinct tables for distinct ADF cases ----
        assert not np.allclose(
            data["augdf_case1_cv"], data["augdf_case2_cv"]
        ), "MATLAB case1 and case2 tables should differ"
        assert not np.allclose(
            data["augdf_case2_cv"], data["augdf_case4_cv"]
        ), "MATLAB case2 and case4 tables should differ"
        assert not np.allclose(
            data["augdf_case1_cv"], data["augdf_case4_cv"]
        ), "MATLAB case1 and case4 tables should differ"

    @pytest.mark.parity
    def test_augdf_cvsim_tieup_parity_lower_tail_negative(
        self, timeseries_fixture_dir
    ) -> None:
        """MATLAB fixture: lower-tail critical values (1 %, 5 %, 10 %)
        must be negative for all cases and all sample sizes, consistent
        with the left-tailed Dickey-Fuller distribution.
        """
        fixture_path = Path(timeseries_fixture_dir) / "augdf_cvsim_tieup.npy"
        if not fixture_path.exists():
            pytest.skip("Fixture augdf_cvsim_tieup.npy not found")

        data = np.load(str(fixture_path), allow_pickle=True).item()
        cvs = data["Cvs"]

        q_01_idx = int(np.argmin(np.abs(cvs - 0.01)))
        q_05_idx = int(np.argmin(np.abs(cvs - 0.05)))
        q_10_idx = int(np.argmin(np.abs(cvs - 0.10)))

        for case_key in ["augdf_case1_cv", "augdf_case2_cv", "augdf_case4_cv"]:
            table = data[case_key]
            for col_idx in range(table.shape[1]):
                assert table[q_01_idx, col_idx] < 0.0, (
                    f"MATLAB {case_key} 1% quantile col {col_idx} should "
                    f"be negative, got {table[q_01_idx, col_idx]:.4f}"
                )
                assert table[q_05_idx, col_idx] < 0.0, (
                    f"MATLAB {case_key} 5% quantile col {col_idx} should "
                    f"be negative, got {table[q_05_idx, col_idx]:.4f}"
                )
                assert table[q_10_idx, col_idx] < 0.0, (
                    f"MATLAB {case_key} 10% quantile col {col_idx} should "
                    f"be negative, got {table[q_10_idx, col_idx]:.4f}"
                )

    @pytest.mark.parity
    def test_augdf_cvsim_tieup_parity_convergence_all_cases(
        self, timeseries_fixture_dir
    ) -> None:
        """MATLAB fixture: asymptotic convergence should hold for all
        three ADF cases, not only case 1.

        For each case, the mean absolute distance between T = 100 and
        T = 10 000 columns should be strictly less than between T = 10
        and T = 10 000.
        """
        fixture_path = Path(timeseries_fixture_dir) / "augdf_cvsim_tieup.npy"
        if not fixture_path.exists():
            pytest.skip("Fixture augdf_cvsim_tieup.npy not found")

        data = np.load(str(fixture_path), allow_pickle=True).item()
        ts = data["Ts"]

        idx_10 = int(np.argmin(np.abs(ts - 10)))
        idx_100 = int(np.argmin(np.abs(ts - 100)))
        idx_10000 = int(np.argmin(np.abs(ts - 10000)))

        for case_key in ["augdf_case1_cv", "augdf_case2_cv", "augdf_case4_cv"]:
            table = data[case_key]
            dist_close = float(
                np.mean(np.abs(table[:, idx_100] - table[:, idx_10000]))
            )
            dist_far = float(
                np.mean(np.abs(table[:, idx_10] - table[:, idx_10000]))
            )
            assert dist_close < dist_far, (
                f"{case_key}: T=100 vs T=10000 distance ({dist_close:.6f}) "
                f"should be less than T=10 vs T=10000 ({dist_far:.6f})"
            )
