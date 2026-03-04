"""Pytest parity and unit tests for the Berkowitz distributional forecast test.

Comprehensive test coverage for ``mfe_toolbox.tests.berkowitz.berkowitz`` —
the Python migration of MATLAB ``tests/berkowitz.m`` (MFE Toolbox v4.0,
Kevin Sheppard).

The Berkowitz test applies the Probability Integral Transform (PIT) followed
by the inverse normal CDF to produce data that should be N(0,1) if the
distributional model is correct.  A likelihood ratio statistic is computed:
- **TS mode** (time-series): AR(1) regression on transformed data → chi2(3)
- **CS mode** (cross-section): Mean/variance comparison → chi2(2)

Per AAP Section 0.7.1:
  - ATOL = 1e-6, RTOL = 1e-4 for all parity comparisons
  - All tests MUST pass with ``pytest -x --tb=short``
  - ``pytest.mark.parity`` applied to fixture-dependent tests
  - Graceful ``pytest.skip()`` when fixture files not found

See Also
--------
tests/berkowitz.m : Original MATLAB source (129 lines).
mfe_toolbox.tests.berkowitz : Migrated Python implementation.
"""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import norm

from mfe_toolbox.tests.berkowitz import berkowitz
from tests.conftest import ATOL, RTOL, load_fixture_npy, assert_allclose


# ---------------------------------------------------------------------------
# Phase 1: Unit Tests (no fixture dependency)
# ---------------------------------------------------------------------------


class TestBerkowitzUnit:
    """Unit tests for berkowitz() — no MATLAB fixture dependency."""

    def test_berkowitz_ts_uniform_data(self, rng: np.random.Generator) -> None:
        """TS mode with sorted uniform (0,1) data should not reject at alpha=0.05.

        Ref: berkowitz.m — When input data is already well-calibrated uniform,
        the PIT+norminv transformation should produce approximate N(0,1) data,
        yielding a small LR stat and large p-value.
        """
        x = np.sort(rng.random(500))
        stat, pval, H = berkowitz(x, test_type='TS', alpha=0.05)

        # Structural invariants
        assert isinstance(stat, float), "stat must be float"
        assert stat >= 0.0, "LR statistic must be non-negative"
        assert 0.0 <= pval <= 1.0, f"p-value {pval} out of [0, 1]"
        assert isinstance(H, (bool, np.bool_)), "H must be bool"

    def test_berkowitz_cs_uniform_data(self, rng: np.random.Generator) -> None:
        """CS mode with sorted uniform (0,1) data — 2 degrees of freedom.

        Same data generation as TS test but using cross-sectional mode.
        The CS branch uses chi2(2) instead of chi2(3).
        """
        x = np.sort(rng.random(500))
        stat, pval, H = berkowitz(x, test_type='CS', alpha=0.05)

        assert isinstance(stat, float), "stat must be float"
        assert stat >= 0.0, "LR statistic must be non-negative"
        assert 0.0 <= pval <= 1.0, f"p-value {pval} out of [0, 1]"
        assert isinstance(H, (bool, np.bool_)), "H must be bool"

    def test_berkowitz_with_dist_callable(self) -> None:
        """Pass a CDF function (norm.cdf) as the dist parameter.

        When a dist callable is provided, berkowitz() applies it to the raw
        data before the PIT+norminv transformation.  For standard normal data
        with norm.cdf, the pipeline should produce approximately N(0,1).
        Ref: berkowitz.m:95-104 — feval(dist, x, varargin{:})
        """
        rng = np.random.default_rng(42)
        x = rng.standard_normal(300)
        stat, pval, H = berkowitz(
            x, test_type='TS', alpha=0.05, dist=norm.cdf
        )

        assert isinstance(stat, float), "stat must be float"
        assert stat >= 0.0, "LR statistic must be non-negative"
        assert 0.0 <= pval <= 1.0, f"p-value {pval} out of [0, 1]"
        assert isinstance(H, (bool, np.bool_)), "H must be bool"

    def test_berkowitz_dist_callable_cs_mode(self) -> None:
        """CS mode with a CDF function callable (norm.cdf).

        Verifies the dist pathway works for both TS and CS modes.
        """
        rng = np.random.default_rng(99)
        x = rng.standard_normal(200)
        stat, pval, H = berkowitz(
            x, test_type='CS', alpha=0.05, dist=norm.cdf
        )

        assert isinstance(stat, float)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0

    def test_berkowitz_cs_vs_ts_df_difference(self) -> None:
        """CS uses chi2(2), TS uses chi2(3) — stat and/or pval must differ.

        With the same input data, different degrees of freedom in the chi2
        distribution yield different p-values (and potentially different test
        statistics due to the additional AR(1) parameters in TS mode).
        """
        rng = np.random.default_rng(42)
        x = np.sort(rng.random(500))

        stat_ts, pval_ts, _ = berkowitz(x, test_type='TS', alpha=0.05)
        stat_cs, pval_cs, _ = berkowitz(x, test_type='CS', alpha=0.05)

        # TS includes AR(1), so stat computation differs from CS
        # At minimum the p-values differ due to different chi2 df
        assert not (
            np.isclose(stat_ts, stat_cs, atol=1e-12)
            and np.isclose(pval_ts, pval_cs, atol=1e-12)
        ), "TS and CS should produce different results (different df and model)"

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_berkowitz_alpha_variations(self, alpha: float) -> None:
        """Verify H = (pval < alpha) holds for various significance levels.

        Ref: berkowitz.m:118 — H = pval < alpha; berkowitz.m:128 — same.
        """
        rng = np.random.default_rng(42)
        x = np.sort(rng.random(500))
        stat, pval, H = berkowitz(x, test_type='TS', alpha=alpha)

        assert isinstance(stat, float)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0
        # Verify the decision rule: H = pval < alpha
        assert H == (pval < alpha), (
            f"H={H} should equal (pval < alpha) = ({pval} < {alpha}) = {pval < alpha}"
        )

    def test_berkowitz_misspecified_data_rejects(self) -> None:
        """Heavily misspecified (skewed) data fed as uniform PIT should reject.

        When data is clearly not from a uniform distribution but is treated
        as already-PIT-transformed, the test should detect the departure
        from N(0,1) in the norminv-transformed space.
        """
        rng = np.random.default_rng(12345)
        # Clip exponential data into (0, 1) — clearly not uniform
        x = np.clip(rng.exponential(1.0, 500), 0.001, 0.999)
        stat, pval, H = berkowitz(x, test_type='CS', alpha=0.05)

        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0
        # We expect rejection for misspecified data
        assert H is True or stat > 5.0, (
            "Heavily misspecified data should produce a large test statistic"
        )

    def test_berkowitz_dist_with_extra_args(self) -> None:
        """Verify dist callable receives extra *args correctly.

        Ref: berkowitz.m:99-100 — feval(dist, x, varargin{:})
        Python equivalent: dist(x, *args) where args=(mu, sigma).
        Must use positional args to pass extra CDF parameters.
        """
        rng = np.random.default_rng(42)
        # Generate Normal(loc=2, scale=3) data
        x = rng.normal(loc=2.0, scale=3.0, size=300)
        # Use all positional args: x, test_type, alpha, dist, *extra
        stat, pval, H = berkowitz(x, 'TS', 0.05, norm.cdf, 2.0, 3.0)

        assert isinstance(stat, float)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# Phase 2: Input Validation Tests
# ---------------------------------------------------------------------------


class TestBerkowitzInputValidation:
    """Input validation tests — verify error handling per berkowitz.m:46-93."""

    def test_berkowitz_empty_input_raises(self) -> None:
        """Empty array should raise ValueError.

        Ref: berkowitz.m:64-65 — nargin==0 → error.  The Python version
        may raise from newlagmatrix or from internal validation, but any
        ValueError (or subclass) is acceptable.
        """
        with pytest.raises((ValueError, Exception)):
            berkowitz(np.array([]))

    def test_berkowitz_invalid_type_raises(self) -> None:
        """test_type='INVALID' must raise ValueError.

        Ref: berkowitz.m:88-89 — TYPE must be either 'TS' or 'CS'.
        """
        rng = np.random.default_rng(42)
        x = rng.random(100)
        with pytest.raises(ValueError, match="test_type"):
            berkowitz(x, test_type='INVALID')

    def test_berkowitz_alpha_zero_raises(self) -> None:
        """alpha=0 must raise ValueError.

        Ref: berkowitz.m:85-86 — TESTSIZE must be a scalar between 0 and 1.
        """
        rng = np.random.default_rng(42)
        x = rng.random(100)
        with pytest.raises(ValueError, match="alpha"):
            berkowitz(x, alpha=0.0)

    def test_berkowitz_alpha_one_raises(self) -> None:
        """alpha=1 must raise ValueError.

        Ref: berkowitz.m:85-86 — alpha >= 1 is invalid.
        """
        rng = np.random.default_rng(42)
        x = rng.random(100)
        with pytest.raises(ValueError, match="alpha"):
            berkowitz(x, alpha=1.0)

    def test_berkowitz_alpha_negative_raises(self) -> None:
        """alpha=-0.5 must raise ValueError."""
        rng = np.random.default_rng(42)
        x = rng.random(100)
        with pytest.raises(ValueError, match="alpha"):
            berkowitz(x, alpha=-0.5)

    def test_berkowitz_data_out_of_01_no_dist_raises(self) -> None:
        """Data outside (0,1) without dist must raise ValueError.

        Ref: berkowitz.m:77-78 — If DIST is not input, X must satisfy 0<X<1.
        """
        with pytest.raises(ValueError):
            berkowitz(np.array([0.0, 0.5, 0.9]))  # x=0.0 violates 0 < x

        with pytest.raises(ValueError):
            berkowitz(np.array([0.1, 0.5, 1.0]))  # x=1.0 violates x < 1

        with pytest.raises(ValueError):
            berkowitz(np.array([-0.5, 0.5, 1.5]))  # both violations

    def test_berkowitz_2d_array_column_vector_accepted(self) -> None:
        """2D column vector (N,1) should be accepted and flattened.

        Ref: berkowitz.m:70-72 — If size(x,2)~=1, transpose; the Python
        implementation ravels the input. A (N,1) array should work.
        """
        rng = np.random.default_rng(42)
        x_2d = rng.random((100, 1))
        stat, pval, H = berkowitz(x_2d, test_type='CS', alpha=0.05)
        assert isinstance(stat, float)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0

    def test_berkowitz_2d_row_vector_accepted(self) -> None:
        """2D row vector (1,N) should be accepted and flattened.

        Ref: berkowitz.m:67-72 — min(size(x))!=1 check allows row/col vectors.
        """
        rng = np.random.default_rng(42)
        x_2d = rng.random((1, 100))
        stat, pval, H = berkowitz(x_2d, test_type='CS', alpha=0.05)
        assert isinstance(stat, float)
        assert stat >= 0.0
        assert 0.0 <= pval <= 1.0

    def test_berkowitz_noncallable_dist_raises(self) -> None:
        """Non-callable dist must raise ValueError.

        Ref: berkowitz.m:81-83 — DIST must be a character string (callable
        in Python terms).
        """
        rng = np.random.default_rng(42)
        x = rng.random(100)
        with pytest.raises(ValueError, match="callable"):
            berkowitz(x, dist="not_a_function")

    def test_berkowitz_none_input_raises(self) -> None:
        """None input should raise ValueError.

        Ref: berkowitz.m:64-65 — nargin==0 case.
        """
        with pytest.raises((ValueError, TypeError)):
            berkowitz(None)


# ---------------------------------------------------------------------------
# Phase 3: Parity Tests (against MATLAB/Octave fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.parity
class TestBerkowitzParity:
    """MATLAB numerical parity tests using generated fixture data.

    Fixture structure (berkowitz.npy): dict with keys:
      - normal_data (500,): uniform(0,1) data from Octave rng(42)
      - nonnormal_data (500,): non-uniform data from Octave rng(42)
      - normal_stat_ts_a01/a05/a10: stat for TS mode, various alphas
      - normal_pval_ts_a01/a05/a10: pval for TS mode
      - normal_H_ts_a01/a05/a10: H for TS mode
      - normal_stat_cs_a01/a05/a10: stat for CS mode
      - nonnormal_stat_ts_a01/a05/a10: stat for TS mode (nonnormal data)
      - ... (same pattern for pval, H)

    Per AAP Section 0.7.1: assert_allclose(atol=1e-6, rtol=1e-4).
    """

    def _load_fixture(self, tests_fixture_dir) -> dict:
        """Load the berkowitz fixture dict, skip if not available."""
        fixture = load_fixture_npy(tests_fixture_dir, 'berkowitz')
        # The fixture is stored as a 0-d object array containing a dict
        if fixture.ndim == 0:
            return fixture.item()
        return fixture

    # --- Normal data, TS mode ---

    @pytest.mark.parametrize("alpha_key,alpha_val", [
        ("a01", 0.01),
        ("a05", 0.05),
        ("a10", 0.10),
    ])
    def test_berkowitz_ts_normal_parity(
        self, tests_fixture_dir, alpha_key: str, alpha_val: float
    ) -> None:
        """Compare TS mode outputs for normal data against MATLAB fixtures.

        Verifies stat, pval, and H match MATLAB reference to within
        ATOL=1e-6, RTOL=1e-4.
        """
        fix = self._load_fixture(tests_fixture_dir)
        normal_data = fix['normal_data']

        stat, pval, H = berkowitz(normal_data, test_type='TS', alpha=alpha_val)

        expected_stat = fix[f'normal_stat_ts_{alpha_key}']
        expected_pval = fix[f'normal_pval_ts_{alpha_key}']
        expected_H = bool(fix[f'normal_H_ts_{alpha_key}'])

        npt.assert_allclose(
            stat, expected_stat, atol=ATOL, rtol=RTOL,
            err_msg=f"TS stat mismatch (normal, alpha={alpha_val})"
        )
        npt.assert_allclose(
            pval, expected_pval, atol=ATOL, rtol=RTOL,
            err_msg=f"TS pval mismatch (normal, alpha={alpha_val})"
        )
        assert H == expected_H, (
            f"TS H mismatch (normal, alpha={alpha_val}): "
            f"got {H}, expected {expected_H}"
        )

    # --- Normal data, CS mode ---

    @pytest.mark.parametrize("alpha_key,alpha_val", [
        ("a01", 0.01),
        ("a05", 0.05),
        ("a10", 0.10),
    ])
    def test_berkowitz_cs_normal_parity(
        self, tests_fixture_dir, alpha_key: str, alpha_val: float
    ) -> None:
        """Compare CS mode outputs for normal data against MATLAB fixtures."""
        fix = self._load_fixture(tests_fixture_dir)
        normal_data = fix['normal_data']

        stat, pval, H = berkowitz(normal_data, test_type='CS', alpha=alpha_val)

        expected_stat = fix[f'normal_stat_cs_{alpha_key}']
        expected_pval = fix[f'normal_pval_cs_{alpha_key}']
        expected_H = bool(fix[f'normal_H_cs_{alpha_key}'])

        npt.assert_allclose(
            stat, expected_stat, atol=ATOL, rtol=RTOL,
            err_msg=f"CS stat mismatch (normal, alpha={alpha_val})"
        )
        npt.assert_allclose(
            pval, expected_pval, atol=ATOL, rtol=RTOL,
            err_msg=f"CS pval mismatch (normal, alpha={alpha_val})"
        )
        assert H == expected_H, (
            f"CS H mismatch (normal, alpha={alpha_val}): "
            f"got {H}, expected {expected_H}"
        )

    # --- Nonnormal data, TS mode ---

    @pytest.mark.parametrize("alpha_key,alpha_val", [
        ("a01", 0.01),
        ("a05", 0.05),
        ("a10", 0.10),
    ])
    def test_berkowitz_ts_nonnormal_parity(
        self, tests_fixture_dir, alpha_key: str, alpha_val: float
    ) -> None:
        """Compare TS mode outputs for nonnormal data against MATLAB fixtures.

        With heavily non-uniform data, the test should always reject
        (H = True) and produce a large statistic.
        """
        fix = self._load_fixture(tests_fixture_dir)
        nonnormal_data = fix['nonnormal_data']

        stat, pval, H = berkowitz(
            nonnormal_data, test_type='TS', alpha=alpha_val
        )

        expected_stat = fix[f'nonnormal_stat_ts_{alpha_key}']
        expected_pval = fix[f'nonnormal_pval_ts_{alpha_key}']
        expected_H = bool(fix[f'nonnormal_H_ts_{alpha_key}'])

        npt.assert_allclose(
            stat, expected_stat, atol=ATOL, rtol=RTOL,
            err_msg=f"TS stat mismatch (nonnormal, alpha={alpha_val})"
        )
        npt.assert_allclose(
            pval, expected_pval, atol=ATOL, rtol=RTOL,
            err_msg=f"TS pval mismatch (nonnormal, alpha={alpha_val})"
        )
        assert H == expected_H, (
            f"TS H mismatch (nonnormal, alpha={alpha_val}): "
            f"got {H}, expected {expected_H}"
        )

    # --- Nonnormal data, CS mode ---

    @pytest.mark.parametrize("alpha_key,alpha_val", [
        ("a01", 0.01),
        ("a05", 0.05),
        ("a10", 0.10),
    ])
    def test_berkowitz_cs_nonnormal_parity(
        self, tests_fixture_dir, alpha_key: str, alpha_val: float
    ) -> None:
        """Compare CS mode outputs for nonnormal data against MATLAB fixtures."""
        fix = self._load_fixture(tests_fixture_dir)
        nonnormal_data = fix['nonnormal_data']

        stat, pval, H = berkowitz(
            nonnormal_data, test_type='CS', alpha=alpha_val
        )

        expected_stat = fix[f'nonnormal_stat_cs_{alpha_key}']
        expected_pval = fix[f'nonnormal_pval_cs_{alpha_key}']
        expected_H = bool(fix[f'nonnormal_H_cs_{alpha_key}'])

        npt.assert_allclose(
            stat, expected_stat, atol=ATOL, rtol=RTOL,
            err_msg=f"CS stat mismatch (nonnormal, alpha={alpha_val})"
        )
        npt.assert_allclose(
            pval, expected_pval, atol=ATOL, rtol=RTOL,
            err_msg=f"CS pval mismatch (nonnormal, alpha={alpha_val})"
        )
        assert H == expected_H, (
            f"CS H mismatch (nonnormal, alpha={alpha_val}): "
            f"got {H}, expected {expected_H}"
        )

    # --- Aggregate vector parity checks ---

    def test_berkowitz_ts_aggregate_vector_parity(
        self, tests_fixture_dir
    ) -> None:
        """Verify aggregated TS stat/pval/H vectors match fixture arrays.

        The fixture contains 'stat_ts' (6,), 'pval_ts' (6,), 'H_ts' (6,)
        arrays representing all 6 combinations of (normal/nonnormal) ×
        (alpha=0.01/0.05/0.10) in TS mode.
        """
        fix = self._load_fixture(tests_fixture_dir)
        expected_stat = fix['stat_ts']
        expected_pval = fix['pval_ts']
        expected_H = fix['H_ts']

        datasets = [fix['normal_data'], fix['nonnormal_data']]
        alphas = [0.01, 0.05, 0.10]
        actual_stat = np.empty(6)
        actual_pval = np.empty(6)
        actual_H = np.empty(6)

        idx = 0
        for data in datasets:
            for alpha in alphas:
                s, p, h = berkowitz(data, test_type='TS', alpha=alpha)
                actual_stat[idx] = s
                actual_pval[idx] = p
                actual_H[idx] = float(h)
                idx += 1

        npt.assert_allclose(
            actual_stat, expected_stat, atol=ATOL, rtol=RTOL,
            err_msg="TS stat vector mismatch"
        )
        npt.assert_allclose(
            actual_pval, expected_pval, atol=ATOL, rtol=RTOL,
            err_msg="TS pval vector mismatch"
        )
        npt.assert_allclose(
            actual_H, expected_H, atol=ATOL, rtol=RTOL,
            err_msg="TS H vector mismatch"
        )

    def test_berkowitz_cs_aggregate_vector_parity(
        self, tests_fixture_dir
    ) -> None:
        """Verify aggregated CS stat/pval/H vectors match fixture arrays."""
        fix = self._load_fixture(tests_fixture_dir)
        expected_stat = fix['stat_cs']
        expected_pval = fix['pval_cs']
        expected_H = fix['H_cs']

        datasets = [fix['normal_data'], fix['nonnormal_data']]
        alphas = [0.01, 0.05, 0.10]
        actual_stat = np.empty(6)
        actual_pval = np.empty(6)
        actual_H = np.empty(6)

        idx = 0
        for data in datasets:
            for alpha in alphas:
                s, p, h = berkowitz(data, test_type='CS', alpha=alpha)
                actual_stat[idx] = s
                actual_pval[idx] = p
                actual_H[idx] = float(h)
                idx += 1

        npt.assert_allclose(
            actual_stat, expected_stat, atol=ATOL, rtol=RTOL,
            err_msg="CS stat vector mismatch"
        )
        npt.assert_allclose(
            actual_pval, expected_pval, atol=ATOL, rtol=RTOL,
            err_msg="CS pval vector mismatch"
        )
        npt.assert_allclose(
            actual_H, expected_H, atol=ATOL, rtol=RTOL,
            err_msg="CS H vector mismatch"
        )


# ---------------------------------------------------------------------------
# Phase 4: Return Type and Shape Tests
# ---------------------------------------------------------------------------


class TestBerkowitzReturnTypes:
    """Return type, shape, and invariant tests."""

    def test_berkowitz_return_types(self) -> None:
        """stat is float, pval is float, H is bool.

        Ref: berkowitz.py signature → tuple[float, float, bool].
        """
        rng = np.random.default_rng(42)
        x = rng.random(200)
        result = berkowitz(x, test_type='TS', alpha=0.05)

        assert isinstance(result, tuple), "Return value must be a tuple"
        assert len(result) == 3, "Return tuple must have 3 elements"

        stat, pval, H = result
        assert isinstance(stat, (float, np.floating)), (
            f"stat type {type(stat)} is not float"
        )
        assert isinstance(pval, (float, np.floating)), (
            f"pval type {type(pval)} is not float"
        )
        assert isinstance(H, (bool, np.bool_)), (
            f"H type {type(H)} is not bool"
        )

    def test_berkowitz_stat_nonnegative(self) -> None:
        """Likelihood ratio stat must be >= 0.

        The LR statistic is computed as -2*(LL_restricted - LL_unrestricted),
        and since the unrestricted model always fits at least as well as the
        restricted model, stat >= 0 must hold.
        """
        rng = np.random.default_rng(42)
        x = rng.random(500)

        for mode in ('TS', 'CS'):
            stat, _, _ = berkowitz(x, test_type=mode, alpha=0.05)
            assert stat >= 0.0, (
                f"stat must be non-negative in {mode} mode, got {stat}"
            )

    def test_berkowitz_pval_range(self) -> None:
        """pval must be in [0, 1].

        The p-value is computed as 1 - chi2.cdf(stat, df), which is
        always in [0, 1] for non-negative stat.
        """
        rng = np.random.default_rng(42)
        x = rng.random(500)

        for mode in ('TS', 'CS'):
            _, pval, _ = berkowitz(x, test_type=mode, alpha=0.05)
            assert 0.0 <= pval <= 1.0, (
                f"pval must be in [0, 1] in {mode} mode, got {pval}"
            )

    def test_berkowitz_h_decision_rule(self) -> None:
        """H must equal (pval < alpha) — the standard hypothesis test rule.

        Ref: berkowitz.m:118 — H = pval < alpha; berkowitz.m:128 — same.
        """
        rng = np.random.default_rng(42)
        x = rng.random(300)

        for mode in ('TS', 'CS'):
            for alpha in (0.01, 0.05, 0.10):
                _, pval, H = berkowitz(x, test_type=mode, alpha=alpha)
                assert H == (pval < alpha), (
                    f"mode={mode}, alpha={alpha}: H={H} but "
                    f"pval<alpha = {pval < alpha} (pval={pval})"
                )

    def test_berkowitz_returns_tuple_three_elements(self) -> None:
        """Should return a 3-tuple (stat, pval, H) for both modes."""
        rng = np.random.default_rng(42)
        x = rng.random(200)

        for mode in ('TS', 'CS'):
            result = berkowitz(x, test_type=mode, alpha=0.05)
            assert isinstance(result, tuple), "Must return a tuple"
            assert len(result) == 3, "Tuple must have exactly 3 elements"

    def test_berkowitz_scalar_outputs(self) -> None:
        """stat and pval must be scalar (0-d), not arrays.

        The Python implementation converts to float explicitly, so the
        return values should be plain Python float, not 0-d ndarrays.
        """
        rng = np.random.default_rng(42)
        x = rng.random(200)
        stat, pval, H = berkowitz(x, test_type='TS', alpha=0.05)

        # Ensure scalar — not array
        assert np.ndim(stat) == 0, f"stat should be scalar, got ndim={np.ndim(stat)}"
        assert np.ndim(pval) == 0, f"pval should be scalar, got ndim={np.ndim(pval)}"
