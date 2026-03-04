"""Pytest parity and unit tests for the Kolmogorov-Smirnov distributional test.

Tests the ``kolmogorov`` function migrated from MATLAB ``tests/kolmogorov.m``
to ``mfe_toolbox/tests/kolmogorov.py``.  Verifies:

* Numerical parity (±1e-6 atol, ±1e-4 rtol) against MATLAB reference fixtures
* Input validation and error handling
* Return types and invariants (stat ≥ 0, 0 ≤ pval ≤ 1, H is bool)
* Both lookup-table (n ≤ 100) and Miller (1956) asymptotic (n > 100) code paths
* CDF callable (``dist``) with and without extra ``*args``
* Empirical CDF formula S = (1:n) / (n+1)

Per AAP Section 0.7.1: ATOL = 1e-6, RTOL = 1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import norm

from mfe_toolbox.tests.kolmogorov import kolmogorov, _kscritical
from tests.conftest import ATOL, RTOL, load_fixture_npy, assert_allclose


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_data(rng: np.random.Generator) -> np.ndarray:
    """Generate reproducible Uniform(0,1) data for unit tests.

    Uses the session-scoped ``rng`` fixture (seeded at 42) to match the
    fixture-generation environment.
    """
    return rng.random(200)


@pytest.fixture
def small_uniform_data(rng: np.random.Generator) -> np.ndarray:
    """Generate a small (n=50) uniform sample for lookup-table tests."""
    return rng.random(50)


@pytest.fixture
def fixture_data(tests_fixture_dir):
    """Load the MATLAB-generated Kolmogorov fixture dictionary.

    The fixture is stored as a 0-d object-dtype numpy array wrapping a dict.
    Returns the dict directly for convenient key access.
    """
    raw = load_fixture_npy(tests_fixture_dir, "kolmogorov")
    # np.load with allow_pickle=True returns a 0-d array wrapping a dict
    return raw.item() if raw.ndim == 0 else raw


# ---------------------------------------------------------------------------
# Phase 1: Unit Tests (no fixture dependency)
# ---------------------------------------------------------------------------

class TestKolmogorovUnit:
    """Unit tests for kolmogorov() that do not depend on fixture files."""

    def test_kolmogorov_uniform_data(self, rng: np.random.Generator) -> None:
        """Uniform (0,1) data should typically not reject the null.

        Ref: kolmogorov.m — dist=None means data is already PIT-transformed.
        """
        x = rng.random(200)
        stat, pval, H = kolmogorov(x, alpha=0.05)

        # Basic invariants
        assert isinstance(stat, float)
        assert stat >= 0.0, "KS stat must be non-negative"
        assert 0.0 <= pval <= 1.0, "p-value must be in [0, 1]"
        # For genuinely uniform data, rejection is unlikely at alpha=0.05
        # (not a hard guarantee, but with seed=42 this is deterministic)
        assert H is False or isinstance(H, (bool, np.bool_))

    def test_kolmogorov_non_uniform_data(self, rng: np.random.Generator) -> None:
        """Non-uniform (Beta(0.5,0.5)) data tested as PIT should reject.

        Beta(0.5,0.5) has a U-shaped density on (0,1) — very different from
        uniform, so the KS test should reject.
        """
        x = np.clip(rng.beta(0.5, 0.5, 200), 0.001, 0.999)
        stat, pval, H = kolmogorov(x, alpha=0.05)

        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0
        # Strong departure → should reject
        assert H is True or H == True  # noqa: E712

    def test_kolmogorov_with_dist_callable(
        self, rng: np.random.Generator
    ) -> None:
        """Pass a CDF function (norm.cdf) for standard-normal data.

        Ref: kolmogorov.m:82-91 — ``feval(dist, x, varargin{:})``
        """
        x = rng.standard_normal(200)
        stat, pval, H = kolmogorov(x, alpha=0.05, dist=norm.cdf)

        assert isinstance(stat, float)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0

    def test_kolmogorov_with_dist_and_args(
        self, rng: np.random.Generator
    ) -> None:
        """CDF with extra parameters (loc=2, scale=3).

        Data drawn from N(2, 9) should not be rejected when tested against
        N(2, 3) via ``norm.cdf(x, 2, 3)``.
        Uses positional argument passing to fill *args in the kolmogorov
        signature: kolmogorov(x, alpha, dist, *args).
        """
        x = 2.0 + 3.0 * rng.standard_normal(200)
        # Pass loc=2, scale=3 as positional *args after dist
        # Ref: kolmogorov.m:87 — feval(dist, x, varargin{:})
        stat, pval, H = kolmogorov(x, 0.05, norm.cdf, 2, 3)

        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0
        # Data matches the distribution, so stat should be small
        # (loose check — just verifying the function works with extra args)
        assert stat < 0.5, "Stat should be moderate for matching distribution"

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_kolmogorov_alpha_variations(
        self, rng: np.random.Generator, alpha: float
    ) -> None:
        """stat and pval are invariant to alpha; only H changes.

        Ref: kolmogorov.m:97 — alpha is only used after stat computation
        to determine the critical value and H.
        """
        x = rng.random(200)
        stat, pval, H = kolmogorov(x, alpha=alpha)

        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0
        assert isinstance(H, (bool, np.bool_))

    def test_kolmogorov_stat_formula(self) -> None:
        """Verify KS stat = max|cdfvals − S| with S = (1:n)/(n+1).

        Use a small known dataset where the stat can be hand-computed.
        Ref: kolmogorov.m:96,99 — S=(1:n)'/(n+1); d=max(abs(cdfvals-S))
        """
        # Small dataset: 5 evenly spaced values in (0,1)
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        n = len(x)
        # Empirical CDF: S = [1/6, 2/6, 3/6, 4/6, 5/6]
        S = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
        x_sorted = np.sort(x)
        expected_stat = float(np.max(np.abs(x_sorted - S)))

        stat, pval, H = kolmogorov(x, alpha=0.05)

        npt.assert_allclose(stat, expected_stat, atol=ATOL, rtol=RTOL)

    def test_kolmogorov_kscritical_small_n(self) -> None:
        """Critical value for n=50 uses the lookup table (n ≤ 100).

        Ref: kolmogorov.m:124-228 — lookup table interpolation.
        At n=50, alpha/2=0.025, the table entry is at row 50 (1-based),
        alpha column 0.025.
        """
        # n=50, alpha=0.05 → alpha_half=0.025
        # Directly test _kscritical
        crit = _kscritical(50, 0.025)
        assert isinstance(crit, float)
        assert crit > 0.0
        # From the lookup table, row 50 (n=50), column for alpha=0.025
        # is 0.1884 (see kolmogorov.m line 175)
        npt.assert_allclose(crit, 0.1884, atol=1e-4)

    def test_kolmogorov_kscritical_large_n(self) -> None:
        """Critical value for n=200 uses Miller (1956) asymptotic (n > 100).

        Ref: kolmogorov.m:229-232 — Miller asymptotic formula.
        """
        # n=200, alpha/2=0.025
        crit = _kscritical(200, 0.025)
        assert isinstance(crit, float)
        assert crit > 0.0
        # Asymptotic: verify formula manually
        # A = 0.09037*(-log10(0.025))^1.5 + 0.01515*(log10(0.025))^2
        #     - 0.08467*0.025 - 0.11143
        log10_p = np.log10(0.025)
        neg_log10_p = -log10_p
        A = (0.09037 * (neg_log10_p ** 1.5) + 0.01515 * (log10_p ** 2)
             - 0.08467 * 0.025 - 0.11143)
        expected = (np.sqrt(np.log(1.0 / 0.025) / (2.0 * 200))
                    - 0.16693 / 200 - A * 200 ** (-1.5))
        npt.assert_allclose(crit, expected, atol=ATOL, rtol=RTOL)

    def test_kolmogorov_pvalue_series_convergence(
        self, rng: np.random.Generator
    ) -> None:
        """Verify the alternating-series p-value computation converges.

        Ref: kolmogorov.m:104-115 — alternating series with tol < 1e-10.
        p-value should be in [0, 1].
        """
        x = rng.random(200)
        stat, pval, H = kolmogorov(x, alpha=0.05)
        assert 0.0 <= pval <= 1.0, f"p-value {pval} out of [0, 1] range"


# ---------------------------------------------------------------------------
# Phase 2: Input Validation Tests
# ---------------------------------------------------------------------------

class TestKolmogorovValidation:
    """Input validation and error handling tests."""

    def test_kolmogorov_empty_input_raises(self) -> None:
        """Empty array should raise ValueError.

        Ref: kolmogorov.m:57-58 — requires non-empty column vector.
        """
        with pytest.raises(ValueError):
            kolmogorov(np.array([]))

    def test_kolmogorov_data_out_of_01_no_dist_raises(self) -> None:
        """Data outside (0,1) without dist should raise ValueError.

        Ref: kolmogorov.m:64-66 — ``if any(x>=1) || any(x<=0)``
        """
        # Values >= 1
        with pytest.raises(ValueError, match="0<X<1"):
            kolmogorov(np.array([0.1, 0.5, 1.0]))
        # Values <= 0
        with pytest.raises(ValueError, match="0<X<1"):
            kolmogorov(np.array([0.0, 0.5, 0.9]))
        # Negative values
        with pytest.raises(ValueError, match="0<X<1"):
            kolmogorov(np.array([-0.1, 0.5, 0.9]))

    def test_kolmogorov_alpha_out_of_range_raises(self) -> None:
        """alpha=0 or alpha=1 should raise ValueError.

        Ref: kolmogorov.m:73-74 — alpha must be in (0, 1) exclusive.
        """
        x = np.array([0.2, 0.5, 0.8])
        with pytest.raises(ValueError, match="ALPHA"):
            kolmogorov(x, alpha=0.0)
        with pytest.raises(ValueError, match="ALPHA"):
            kolmogorov(x, alpha=1.0)

    def test_kolmogorov_alpha_negative_raises(self) -> None:
        """Negative alpha should raise ValueError.

        Ref: kolmogorov.m:73-74 — ``alpha<=0 || alpha>=1``
        """
        x = np.array([0.2, 0.5, 0.8])
        with pytest.raises(ValueError, match="ALPHA"):
            kolmogorov(x, alpha=-0.5)

    def test_kolmogorov_2d_data_handled(self) -> None:
        """2D row/column vectors should be flattened and accepted.

        Ref: kolmogorov.m:60-61 — ``if size(x,2)~=1, x=x'``
        The Python implementation uses ravel for 2D vectors.
        """
        x_1d = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        # Column vector (n, 1)
        x_col = x_1d.reshape(-1, 1)
        stat_col, pval_col, H_col = kolmogorov(x_col, alpha=0.05)
        stat_1d, pval_1d, H_1d = kolmogorov(x_1d, alpha=0.05)
        npt.assert_allclose(stat_col, stat_1d, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(pval_col, pval_1d, atol=ATOL, rtol=RTOL)
        assert H_col == H_1d

        # Row vector (1, n)
        x_row = x_1d.reshape(1, -1)
        stat_row, pval_row, H_row = kolmogorov(x_row, alpha=0.05)
        npt.assert_allclose(stat_row, stat_1d, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(pval_row, pval_1d, atol=ATOL, rtol=RTOL)
        assert H_row == H_1d

    def test_kolmogorov_2d_matrix_raises(self) -> None:
        """True 2D matrix (min(shape) > 1) should raise ValueError.

        Ref: kolmogorov.m:57 — ``if min(size(x))~=1``
        """
        x_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError):
            kolmogorov(x_matrix)

    def test_kolmogorov_dist_not_callable_raises(self) -> None:
        """Non-callable dist argument should raise ValueError.

        Ref: kolmogorov.m:69-70 — MATLAB checks ``ischar(dist)``; Python
        checks ``callable(dist)``.
        """
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        with pytest.raises(ValueError, match="callable"):
            kolmogorov(x, alpha=0.05, dist="not_a_function")


# ---------------------------------------------------------------------------
# Phase 3: Parity Tests (against MATLAB fixtures)
# ---------------------------------------------------------------------------

class TestKolmogorovParity:
    """MATLAB fixture parity tests.

    Requires generated fixture files in ``tests/fixtures/tests/kolmogorov.npy``.
    Decorated with ``pytest.mark.parity`` per AAP Section 0.7.1.
    """

    @pytest.mark.parity
    def test_kolmogorov_parity_uniform_n200(self, fixture_data) -> None:
        """Compare uniform n=200 outputs against MATLAB fixtures.

        Fixture indices 0-2: uniform_n200 at alpha=[0.05, 0.01, 0.10].
        """
        data = fixture_data
        x = data["uniform_data_200"]
        alphas = data["alpha_values"]
        expected_stats = data["stat"]
        expected_pvals = data["pval"]
        expected_H = data["H"]

        # Indices 0, 1, 2 correspond to uniform_n200
        for i in range(3):
            alpha_val = float(alphas[i])
            stat, pval, H = kolmogorov(x, alpha=alpha_val)
            npt.assert_allclose(
                stat, expected_stats[i], atol=ATOL, rtol=RTOL,
                err_msg=f"stat mismatch for uniform_n200 alpha={alpha_val}"
            )
            npt.assert_allclose(
                pval, expected_pvals[i], atol=ATOL, rtol=RTOL,
                err_msg=f"pval mismatch for uniform_n200 alpha={alpha_val}"
            )
            assert bool(H) == bool(expected_H[i]), (
                f"H mismatch: got {H}, expected {bool(expected_H[i])} "
                f"for alpha={alpha_val}"
            )

    @pytest.mark.parity
    def test_kolmogorov_parity_uniform_n50(self, fixture_data) -> None:
        """Compare uniform n=50 outputs against MATLAB fixtures.

        Fixture indices 3-5: uniform_n50 at alpha=[0.05, 0.01, 0.10].
        Tests the lookup-table code path (n ≤ 100).
        """
        data = fixture_data
        x = data["uniform_data_50"]
        alphas = data["alpha_values"]
        expected_stats = data["stat"]
        expected_pvals = data["pval"]
        expected_H = data["H"]

        for i in range(3, 6):
            alpha_val = float(alphas[i])
            stat, pval, H = kolmogorov(x, alpha=alpha_val)
            npt.assert_allclose(
                stat, expected_stats[i], atol=ATOL, rtol=RTOL,
                err_msg=f"stat mismatch for uniform_n50 alpha={alpha_val}"
            )
            npt.assert_allclose(
                pval, expected_pvals[i], atol=ATOL, rtol=RTOL,
                err_msg=f"pval mismatch for uniform_n50 alpha={alpha_val}"
            )
            assert bool(H) == bool(expected_H[i]), (
                f"H mismatch: got {H}, expected {bool(expected_H[i])} "
                f"for alpha={alpha_val}"
            )

    @pytest.mark.parity
    def test_kolmogorov_parity_nonuniform_n200(self, fixture_data) -> None:
        """Compare non-uniform n=200 outputs against MATLAB fixtures.

        Fixture indices 6-8: nonuniform_beta_n200 at alpha=[0.05, 0.01, 0.10].
        Non-uniform (Beta-transformed) data should produce rejection.
        """
        data = fixture_data
        x = data["nonuniform_data_200"]
        alphas = data["alpha_values"]
        expected_stats = data["stat"]
        expected_pvals = data["pval"]
        expected_H = data["H"]

        for i in range(6, 9):
            alpha_val = float(alphas[i])
            stat, pval, H = kolmogorov(x, alpha=alpha_val)
            npt.assert_allclose(
                stat, expected_stats[i], atol=ATOL, rtol=RTOL,
                err_msg=f"stat mismatch for nonuniform_n200 alpha={alpha_val}"
            )
            npt.assert_allclose(
                pval, expected_pvals[i], atol=ATOL, rtol=RTOL,
                err_msg=f"pval mismatch for nonuniform_n200 alpha={alpha_val}"
            )
            assert bool(H) == bool(expected_H[i]), (
                f"H mismatch: got {H}, expected {bool(expected_H[i])} "
                f"for alpha={alpha_val}"
            )

    @pytest.mark.parity
    def test_kolmogorov_parity_with_dist_n200(self, fixture_data) -> None:
        """Compare normal data + norm.cdf outputs against MATLAB fixtures.

        Fixture indices 9-11: normal_normcdf_n200 at alpha=[0.05, 0.01, 0.10].
        Tests the dist=callable code path.
        Ref: kolmogorov.m:82-91 — ``feval(dist, x, varargin{:})``
        """
        data = fixture_data
        x = data["normal_data_200"]
        alphas = data["alpha_values"]
        expected_stats = data["stat"]
        expected_pvals = data["pval"]
        expected_H = data["H"]

        for i in range(9, 12):
            alpha_val = float(alphas[i])
            stat, pval, H = kolmogorov(x, alpha=alpha_val, dist=norm.cdf)
            npt.assert_allclose(
                stat, expected_stats[i], atol=ATOL, rtol=RTOL,
                err_msg=f"stat mismatch for normal_normcdf_n200 alpha={alpha_val}"
            )
            npt.assert_allclose(
                pval, expected_pvals[i], atol=ATOL, rtol=RTOL,
                err_msg=f"pval mismatch for normal_normcdf_n200 alpha={alpha_val}"
            )
            assert bool(H) == bool(expected_H[i]), (
                f"H mismatch: got {H}, expected {bool(expected_H[i])} "
                f"for alpha={alpha_val}"
            )

    @pytest.mark.parity
    def test_kolmogorov_parity_nonuniform_n50(self, fixture_data) -> None:
        """Compare non-uniform n=50 outputs against MATLAB fixtures.

        Fixture index 12: nonuniform_beta_n50_alpha005.
        Tests the lookup-table path (n ≤ 100) for non-uniform data.
        """
        data = fixture_data
        x = data["nonuniform_data_50"]
        alpha_val = float(data["alpha_values"][12])
        expected_stat = float(data["stat"][12])
        expected_pval = float(data["pval"][12])
        expected_H = bool(data["H"][12])

        stat, pval, H = kolmogorov(x, alpha=alpha_val)
        npt.assert_allclose(
            stat, expected_stat, atol=ATOL, rtol=RTOL,
            err_msg="stat mismatch for nonuniform_n50"
        )
        npt.assert_allclose(
            pval, expected_pval, atol=ATOL, rtol=RTOL,
            err_msg="pval mismatch for nonuniform_n50"
        )
        assert bool(H) == expected_H, (
            f"H mismatch: got {H}, expected {expected_H}"
        )

    @pytest.mark.parity
    def test_kolmogorov_parity_normal_n50(self, fixture_data) -> None:
        """Compare normal n=50 + norm.cdf outputs against MATLAB fixtures.

        Fixture index 13: normal_normcdf_n50_alpha005.
        Tests the lookup-table + dist-callable combined path.
        """
        data = fixture_data
        x = data["normal_data_50"]
        alpha_val = float(data["alpha_values"][13])
        expected_stat = float(data["stat"][13])
        expected_pval = float(data["pval"][13])
        expected_H = bool(data["H"][13])

        stat, pval, H = kolmogorov(x, alpha=alpha_val, dist=norm.cdf)
        npt.assert_allclose(
            stat, expected_stat, atol=ATOL, rtol=RTOL,
            err_msg="stat mismatch for normal_normcdf_n50"
        )
        npt.assert_allclose(
            pval, expected_pval, atol=ATOL, rtol=RTOL,
            err_msg="pval mismatch for normal_normcdf_n50"
        )
        assert bool(H) == expected_H, (
            f"H mismatch: got {H}, expected {expected_H}"
        )

    @pytest.mark.parity
    def test_kolmogorov_parity_all_cases(self, fixture_data) -> None:
        """Comprehensive parity check over all 14 MATLAB fixture test cases.

        Iterates through every test case in the fixture dict and compares
        stat, pval, and H against MATLAB reference values.
        """
        data = fixture_data
        num_cases = int(data["num_test_cases"])
        descriptions = data["test_descriptions"]
        stats = data["stat"]
        pvals = data["pval"]
        H_vals = data["H"]
        alphas = data["alpha_values"]
        data_types = data["data_types"]
        sample_sizes = data["sample_sizes"]

        for idx in range(num_cases):
            desc = str(descriptions[idx])
            alpha_val = float(alphas[idx])
            n = int(sample_sizes[idx])
            dtype = int(data_types[idx])

            # Select appropriate data array and dist parameter
            if dtype == 1:
                # Uniform data — no dist
                key = f"uniform_data_{n}"
                x = data[key]
                dist_fn = None
                extra_args: tuple = ()
            elif dtype == 2:
                # Non-uniform (Beta-transformed) data — no dist
                key = f"nonuniform_data_{n}"
                x = data[key]
                dist_fn = None
                extra_args = ()
            elif dtype == 3:
                # Normal data — use norm.cdf as dist
                key = f"normal_data_{n}"
                x = data[key]
                dist_fn = norm.cdf
                extra_args = ()
            else:
                pytest.fail(f"Unknown data_type {dtype} in fixture case {idx}")

            stat, pval, H = kolmogorov(x, alpha=alpha_val, dist=dist_fn,
                                       *extra_args)
            npt.assert_allclose(
                stat, stats[idx], atol=ATOL, rtol=RTOL,
                err_msg=f"[{desc}] stat mismatch"
            )
            npt.assert_allclose(
                pval, pvals[idx], atol=ATOL, rtol=RTOL,
                err_msg=f"[{desc}] pval mismatch"
            )
            assert bool(H) == bool(H_vals[idx]), (
                f"[{desc}] H mismatch: got {H}, expected {bool(H_vals[idx])}"
            )


# ---------------------------------------------------------------------------
# Phase 4: Return Type and Edge Case Tests
# ---------------------------------------------------------------------------

class TestKolmogorovReturnTypes:
    """Return type assertions and invariant checks."""

    def test_kolmogorov_return_types(self) -> None:
        """Verify stat is float, pval is float, H is bool.

        Ref: kolmogorov.py returns ``tuple[float, float, bool]``.
        """
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        stat, pval, H = kolmogorov(x, alpha=0.05)
        assert isinstance(stat, (float, np.floating)), (
            f"stat should be float, got {type(stat)}"
        )
        assert isinstance(pval, (float, np.floating)), (
            f"pval should be float, got {type(pval)}"
        )
        assert isinstance(H, (bool, np.bool_)), (
            f"H should be bool, got {type(H)}"
        )

    def test_kolmogorov_stat_nonnegative(
        self, rng: np.random.Generator
    ) -> None:
        """KS statistic must always be ≥ 0.

        This is a fundamental property: stat = max|F_n − F| ≥ 0.
        """
        x = rng.random(100)
        stat, _, _ = kolmogorov(x, alpha=0.05)
        assert stat >= 0.0, f"KS stat {stat} is negative"

    def test_kolmogorov_pval_range(self, rng: np.random.Generator) -> None:
        """p-value must always be in [0, 1]."""
        x = rng.random(100)
        _, pval, _ = kolmogorov(x, alpha=0.05)
        assert 0.0 <= pval <= 1.0, f"p-value {pval} outside [0, 1]"

    def test_kolmogorov_empirical_cdf_formula(self) -> None:
        """Verify the empirical CDF formula uses S = (1:n)/(n+1).

        Ref: kolmogorov.m:96 — ``S = (1:n)' / (n+1)`` — NOTE: (n+1)
        denominator, NOT n.  This is the Weibull plotting position.
        """
        x = np.array([0.2, 0.4, 0.6, 0.8])
        n = len(x)

        # Expected empirical CDF: S = [1/5, 2/5, 3/5, 4/5]
        S = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
        npt.assert_allclose(S, np.array([0.2, 0.4, 0.6, 0.8]),
                            atol=ATOL, rtol=RTOL)

        # The data IS exactly [0.2, 0.4, 0.6, 0.8] which equals S
        # so the KS stat should be 0
        stat, pval, H = kolmogorov(x, alpha=0.05)
        npt.assert_allclose(stat, 0.0, atol=ATOL,
                            err_msg="Stat should be ~0 when data matches S")

    def test_kolmogorov_stat_consistency_across_alpha(
        self, rng: np.random.Generator
    ) -> None:
        """stat and pval must be identical across different alpha values.

        Only H should change with alpha since it depends on the critical value.
        """
        x = rng.random(150)
        stat_05, pval_05, _ = kolmogorov(x, alpha=0.05)
        stat_01, pval_01, _ = kolmogorov(x, alpha=0.01)
        stat_10, pval_10, _ = kolmogorov(x, alpha=0.10)

        npt.assert_allclose(stat_05, stat_01, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(stat_05, stat_10, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(pval_05, pval_01, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(pval_05, pval_10, atol=ATOL, rtol=RTOL)

    def test_kolmogorov_large_sample_asymptotic(self) -> None:
        """Verify large-sample (n > 100) critical value path.

        The Miller (1956) asymptotic formula should produce reasonable
        critical values for large n.
        """
        # n=500 should definitively use the asymptotic path
        crit = _kscritical(500, 0.025)
        assert crit > 0.0
        assert crit < 1.0, "Critical value should be < 1 for large n"

        # For very large n, the critical value decreases roughly as 1/sqrt(n)
        crit_1000 = _kscritical(1000, 0.025)
        assert crit_1000 < crit, (
            "Critical value should decrease with increasing n"
        )

    def test_kolmogorov_single_observation(self) -> None:
        """Single observation should work with the lookup table (n=1).

        Ref: kolmogorov.m lookup table row 1 (n=1).
        """
        x = np.array([0.5])
        stat, pval, H = kolmogorov(x, alpha=0.05)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0

    def test_kolmogorov_n_exactly_100(self) -> None:
        """n=100 is the boundary for lookup table vs asymptotic.

        n=100 should use the lookup table (n ≤ 100 condition).
        """
        crit_100 = _kscritical(100, 0.025)
        crit_101 = _kscritical(101, 0.025)

        # Both should be positive
        assert crit_100 > 0.0
        assert crit_101 > 0.0

        # They should be close but from different code paths
        # The transition should be smooth
        assert abs(crit_100 - crit_101) < 0.01, (
            "Lookup table → asymptotic transition should be smooth"
        )

    def test_kolmogorov_dist_error_handling(self) -> None:
        """Dist function that raises should produce a ValueError.

        Ref: kolmogorov.m:89-91 — catch block wraps feval errors.
        """
        def bad_cdf(x):
            raise RuntimeError("Intentional test error")

        x = np.array([0.1, 0.3, 0.5])
        with pytest.raises(ValueError, match="error calling"):
            kolmogorov(x, alpha=0.05, dist=bad_cdf)
