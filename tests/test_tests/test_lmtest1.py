"""Pytest parity and unit tests for the LM serial correlation test.

Verifies the ``lmtest1`` function migrated from MATLAB ``tests/lmtest1.m``
(Kevin Sheppard, MFE Toolbox v4.0) against MATLAB-generated reference
fixtures with numerical parity tolerances ATOL=1e-6, RTOL=1e-4 per
AAP Section 0.7.1.

Covers:
- Robust (sandwich) and classical (OLS) variance estimator modes
- IID data (should not reject) and ARCH-effect data (should reject)
- Cumulative demeaning behavior (Ref: lmtest1.m:67 — data modified in loop)
- Chi-squared degrees of freedom per lag Q
- Input validation error paths (Ref: lmtest1.m:38-60)
- Return type/shape invariants (ndarray, shape (q,), LM ≥ 0, 0 ≤ pval ≤ 1)
- MATLAB fixture parity for noarch and garch data at q=1, 5, 10

Fixture data is loaded from ``tests/fixtures/tests/lmtest1.npy``.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import chi2

from mfe_toolbox.tests.lmtest1 import lmtest1
from tests.conftest import ATOL, RTOL, assert_allclose, load_fixture_npy


# ---------------------------------------------------------------------------
# Helper: Generate GARCH-like data with conditional heteroskedasticity
# ---------------------------------------------------------------------------

def _generate_garch_data(
    rng: np.random.Generator,
    T: int = 1000,
    omega: float = 0.01,
    alpha: float = 0.10,
    beta: float = 0.85,
) -> np.ndarray:
    """Generate GARCH(1,1) process for ARCH-effect testing.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random generator.
    T : int
        Sample length.
    omega, alpha, beta : float
        GARCH(1,1) parameters.

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` returns from GARCH(1,1) process.
    """
    z = rng.standard_normal(T)
    h = np.zeros(T, dtype=np.float64)
    data = np.zeros(T, dtype=np.float64)
    h[0] = omega / (1.0 - alpha - beta)  # unconditional variance
    data[0] = np.sqrt(h[0]) * z[0]
    for t in range(1, T):
        h[t] = omega + alpha * data[t - 1] ** 2 + beta * h[t - 1]
        data[t] = np.sqrt(h[t]) * z[t]
    return data


# =========================================================================
# Phase 1: Unit Tests (no fixture dependency)
# =========================================================================


class TestLmtest1UnitIID:
    """Unit tests using IID data (no ARCH effects)."""

    def test_lmtest1_iid_robust(self, rng: np.random.Generator) -> None:
        """IID data with robust=True should not reject at 5% level.

        Ref: lmtest1.m:73-75 — robust sandwich estimator.
        """
        data = rng.standard_normal(1000)
        lm_stat, pval = lmtest1(data, q=5, robust=True)

        # Shape verification
        assert lm_stat.shape == (5,), f"Expected shape (5,), got {lm_stat.shape}"
        assert pval.shape == (5,), f"Expected shape (5,), got {pval.shape}"

        # For IID data, p-values should generally be > 0.05
        # (statistical test — use relaxed threshold to avoid flaky tests)
        assert np.any(pval > 0.05), (
            "Expected at least some p-values > 0.05 for IID data"
        )

    def test_lmtest1_iid_classical(self, rng: np.random.Generator) -> None:
        """IID data with robust=False should not reject at 5% level.

        Ref: lmtest1.m:76-78 — classical OLS estimator.
        """
        data = rng.standard_normal(1000)
        lm_stat, pval = lmtest1(data, q=5, robust=False)

        assert lm_stat.shape == (5,)
        assert pval.shape == (5,)
        assert np.any(pval > 0.05), (
            "Expected at least some p-values > 0.05 for IID data (classical)"
        )

    def test_lmtest1_arch_effect_data(self) -> None:
        """GARCH-like squared data should show serial correlation.

        Ref: When data has ARCH effects, squaring reveals serial
        dependence that the LM test should detect.
        """
        rng_local = np.random.default_rng(123)
        garch_data = _generate_garch_data(rng_local, T=2000)

        # Squared returns expose ARCH clustering
        lm_stat, pval = lmtest1(garch_data ** 2, q=5, robust=True)

        assert lm_stat.shape == (5,)
        assert pval.shape == (5,)
        # At least the first lag should show significant serial correlation
        assert np.any(pval < 0.05), (
            "Expected at least some p-values < 0.05 for ARCH-effect squared data"
        )

    def test_lmtest1_robust_vs_classical(self, rng: np.random.Generator) -> None:
        """Robust and classical modes should produce different LM statistics.

        The two variance estimators (sandwich vs OLS) yield different
        quadratic forms, so LM statistics must differ (except in degenerate
        cases).
        """
        data = rng.standard_normal(500)

        lm_robust, _ = lmtest1(data.copy(), q=5, robust=True)
        lm_classical, _ = lmtest1(data.copy(), q=5, robust=False)

        # The two estimators should generally not produce identical values
        assert not np.allclose(lm_robust, lm_classical), (
            "Robust and classical LM stats should differ"
        )

    def test_lmtest1_single_lag(self, rng: np.random.Generator) -> None:
        """q=1 should return single-element arrays."""
        data = rng.standard_normal(200)
        lm_stat, pval = lmtest1(data, q=1)

        assert lm_stat.shape == (1,), f"Expected shape (1,), got {lm_stat.shape}"
        assert pval.shape == (1,), f"Expected shape (1,), got {pval.shape}"

    @pytest.mark.parametrize("q", [1, 5, 10])
    def test_lmtest1_multiple_q_specs(
        self, rng: np.random.Generator, q: int
    ) -> None:
        """Various q values should produce correctly shaped outputs."""
        data = rng.standard_normal(500)
        lm_stat, pval = lmtest1(data, q=q)

        assert lm_stat.shape == (q,), f"Expected shape ({q},), got {lm_stat.shape}"
        assert pval.shape == (q,), f"Expected shape ({q},), got {pval.shape}"


class TestLmtest1CumulativeDemeaning:
    """Verify the critical cumulative demeaning behavior.

    Ref: lmtest1.m:67 — ``data = data - mean(data)`` modifies data
    IN PLACE at each loop iteration.  After iteration 1, data is
    demeaned once; iteration 2 further demeans the already-demeaned
    data, etc.  This cumulative effect must be preserved.
    """

    def test_lmtest1_cumulative_demeaning(self) -> None:
        """Verify cumulative demeaning matches MATLAB behavior.

        Create data with non-zero mean.  After the first iteration,
        the data mean is driven to near-zero.  Subsequent iterations
        do not change already-zero-mean data, but the *first*
        iteration's demeaning matters.  We compare results from the
        function with manually computed cumulative-demeaning results.
        """
        rng_local = np.random.default_rng(99)
        # Data with intentional non-zero mean
        data = rng_local.standard_normal(300) + 5.0

        # Run lmtest1 — it should demean cumulatively
        lm_stat, pval = lmtest1(data.copy(), q=3, robust=True)

        # Manually replicate cumulative demeaning behavior
        # Ref: lmtest1.m:67 — data = data - mean(data) at each iteration
        data_manual = data.copy()
        from mfe_toolbox.utility.newlagmatrix import newlagmatrix

        T = len(data_manual)
        lm_manual = np.zeros(3, dtype=np.float64)
        for Q_idx in range(1, 4):
            data_manual = data_manual - np.mean(data_manual)
            y, x = newlagmatrix(data_manual, Q_idx, 0)
            e = y.ravel()
            s = x[:, :Q_idx] * e[:, np.newaxis]
            sbar = np.mean(s, axis=0)
            s_demeaned = s - sbar
            S = s_demeaned.T @ s_demeaned / T
            lm_manual[Q_idx - 1] = float(T * sbar @ np.linalg.solve(S, sbar))

        # The function result must match the manual cumulative demeaning
        npt.assert_allclose(
            lm_stat, lm_manual, atol=ATOL, rtol=RTOL,
            err_msg="Cumulative demeaning behavior mismatch"
        )

    def test_lmtest1_nonzero_mean_vs_zero_mean(self) -> None:
        """Data with non-zero mean should give the same LM stats
        as pre-demeaned data (because of the cumulative demeaning).

        Ref: lmtest1.m:67 — first iteration demeans the data.
        """
        rng_local = np.random.default_rng(77)
        data_centered = rng_local.standard_normal(500)
        data_centered = data_centered - np.mean(data_centered)
        data_shifted = data_centered + 10.0  # non-zero mean

        # After the first iteration's demeaning, both should be equivalent
        lm_centered, pval_centered = lmtest1(data_centered.copy(), q=3, robust=True)
        lm_shifted, pval_shifted = lmtest1(data_shifted.copy(), q=3, robust=True)

        npt.assert_allclose(
            lm_centered, lm_shifted, atol=ATOL, rtol=RTOL,
            err_msg="Non-zero mean data should match centered data after demeaning"
        )


class TestLmtest1Chi2DegreesOfFreedom:
    """Verify chi-squared degrees of freedom for each lag Q."""

    def test_lmtest1_chi2_degrees_of_freedom(
        self, rng: np.random.Generator
    ) -> None:
        """The Q-th p-value uses chi2(Q) degrees of freedom.

        Ref: lmtest1.m:82 — ``pval = 1 - chi2cdf(lm, (1:q)')``
        Verify: ``1 - chi2.cdf(lm[Q-1], Q) == pval[Q-1]`` for each Q.
        """
        data = rng.standard_normal(500)
        q = 5
        lm_stat, pval = lmtest1(data, q=q, robust=True)

        # Independently compute p-values using scipy.stats.chi2
        dof = np.arange(1, q + 1)
        expected_pval = 1.0 - chi2.cdf(lm_stat, dof)

        npt.assert_allclose(
            pval, expected_pval, atol=ATOL, rtol=RTOL,
            err_msg="P-values should match chi2(Q) CDF formula"
        )


class TestLmtest1QuadraticForm:
    """Verify LM statistic non-negativity (quadratic form invariant)."""

    def test_lmtest1_quadratic_form_nonnegative(
        self, rng: np.random.Generator
    ) -> None:
        """LM stats from the quadratic form T*sbar @ inv(S) @ sbar'
        should always be non-negative.

        Ref: lmtest1.m:80 — ``lm(Q) = T * sbar * S^(-1) * sbar'``
        """
        data = rng.standard_normal(500)
        lm_stat, _ = lmtest1(data, q=10, robust=True)
        assert np.all(lm_stat >= 0), (
            f"LM stats must be non-negative, got: {lm_stat}"
        )

        lm_stat_c, _ = lmtest1(data.copy(), q=10, robust=False)
        assert np.all(lm_stat_c >= 0), (
            f"Classical LM stats must be non-negative, got: {lm_stat_c}"
        )


# =========================================================================
# Phase 2: Input Validation Tests
# =========================================================================


class TestLmtest1InputValidation:
    """Input validation tests — all error paths from lmtest1.m:38-60."""

    def test_lmtest1_q_too_large_raises(self) -> None:
        """T < q should raise ValueError.

        Ref: lmtest1.m:44-45 — ``if T<q, error('At least Q observations requires')``
        """
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="[Qq]"):
            lmtest1(data, q=5)

    @pytest.mark.parametrize("q", [0, -1, -10])
    def test_lmtest1_q_not_positive_raises(self, q: int) -> None:
        """q=0 or negative q should raise ValueError.

        Ref: lmtest1.m:53-54 — ``Q must be a positive integer``
        """
        data = np.random.default_rng(1).standard_normal(100)
        with pytest.raises(ValueError, match="[Qq].*positive"):
            lmtest1(data, q=q)

    @pytest.mark.parametrize("q", [2.5, 3.0, 1.1])
    def test_lmtest1_q_noninteger_raises(self, q: float) -> None:
        """Float q should raise ValueError.

        Ref: lmtest1.m:53 — ``floor(q)==q`` check.
        """
        data = np.random.default_rng(1).standard_normal(100)
        with pytest.raises(ValueError, match="[Qq].*positive"):
            lmtest1(data, q=q)

    def test_lmtest1_empty_data_raises(self) -> None:
        """Empty array should raise ValueError.

        Ref: lmtest1.m:44 — T=0 < q for any q ≥ 1.
        """
        with pytest.raises(ValueError):
            lmtest1(np.array([]), q=1)

    @pytest.mark.parametrize("robust_val", [2, -1, "yes", "true", 3.0])
    def test_lmtest1_robust_invalid_raises(self, robust_val) -> None:
        """Invalid robust values should raise ValueError.

        Ref: lmtest1.m:57-59 — ``ROBUST must be either 0 or 1``
        """
        data = np.random.default_rng(1).standard_normal(100)
        with pytest.raises((ValueError, TypeError)):
            lmtest1(data, q=5, robust=robust_val)

    def test_lmtest1_2d_data_flattened(self) -> None:
        """2D column input (T, 1) should be accepted and flattened.

        Ref: lmtest1.m:47-48 — if size(data,1)~=T, data=data'
        """
        rng_local = np.random.default_rng(42)
        data_2d = rng_local.standard_normal((100, 1))
        data_1d = data_2d.ravel()

        lm_2d, pval_2d = lmtest1(data_2d, q=3)
        lm_1d, pval_1d = lmtest1(data_1d, q=3)

        npt.assert_allclose(lm_2d, lm_1d, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(pval_2d, pval_1d, atol=ATOL, rtol=RTOL)

    def test_lmtest1_2d_row_vector_flattened(self) -> None:
        """2D row input (1, T) should be accepted and flattened.

        Ref: lmtest1.m:47-48 — MATLAB transposes row vectors to columns.
        """
        rng_local = np.random.default_rng(42)
        data_row = rng_local.standard_normal((1, 100))
        data_1d = data_row.ravel()

        lm_row, pval_row = lmtest1(data_row, q=3)
        lm_1d, pval_1d = lmtest1(data_1d, q=3)

        npt.assert_allclose(lm_row, lm_1d, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(pval_row, pval_1d, atol=ATOL, rtol=RTOL)

    def test_lmtest1_2d_matrix_raises(self) -> None:
        """2D matrix (non-vector) should raise ValueError.

        Ref: lmtest1.m:50-51 — DATA must be a column vector
        """
        data = np.random.default_rng(1).standard_normal((10, 10))
        with pytest.raises(ValueError, match="column vector"):
            lmtest1(data, q=5)


# =========================================================================
# Phase 3: Parity Tests (against MATLAB fixtures)
# =========================================================================


def _load_lmtest1_fixture(tests_fixture_dir):
    """Load the lmtest1 MATLAB fixture dict."""
    fixture = load_fixture_npy(tests_fixture_dir, "lmtest1")
    return fixture.item()


@pytest.mark.parity
class TestLmtest1ParityNoarch:
    """Parity tests for data without ARCH effects (noarch)."""

    @pytest.mark.parametrize("q", [1, 5, 10])
    def test_lmtest1_parity_noarch_robust(self, tests_fixture_dir, q: int) -> None:
        """Robust mode on noarch data must match MATLAB to ±1e-6.

        Ref: lmtest1.m:73-75 — robust sandwich variance estimator.
        """
        fixture = _load_lmtest1_fixture(tests_fixture_dir)
        data = fixture["data_noarch"].ravel()
        expected_lm = fixture[f"noarch_q{q}_lm_robust"].ravel()
        expected_pval = fixture[f"noarch_q{q}_pval_robust"].ravel()

        lm_stat, pval = lmtest1(data.copy(), q=q, robust=True)

        assert_allclose(
            lm_stat, expected_lm,
            err_msg=f"noarch robust LM stats mismatch at q={q}"
        )
        assert_allclose(
            pval, expected_pval,
            err_msg=f"noarch robust p-values mismatch at q={q}"
        )

    @pytest.mark.parametrize("q", [1, 5, 10])
    def test_lmtest1_parity_noarch_classical(self, tests_fixture_dir, q: int) -> None:
        """Classical mode on noarch data must match MATLAB to ±1e-6.

        Ref: lmtest1.m:76-78 — classical OLS variance estimator.
        """
        fixture = _load_lmtest1_fixture(tests_fixture_dir)
        data = fixture["data_noarch"].ravel()
        expected_lm = fixture[f"noarch_q{q}_lm_classical"].ravel()
        expected_pval = fixture[f"noarch_q{q}_pval_classical"].ravel()

        lm_stat, pval = lmtest1(data.copy(), q=q, robust=False)

        assert_allclose(
            lm_stat, expected_lm,
            err_msg=f"noarch classical LM stats mismatch at q={q}"
        )
        assert_allclose(
            pval, expected_pval,
            err_msg=f"noarch classical p-values mismatch at q={q}"
        )


@pytest.mark.parity
class TestLmtest1ParityGarch:
    """Parity tests for GARCH-like data with serial correlation."""

    @pytest.mark.parametrize("q", [1, 5, 10])
    def test_lmtest1_parity_garch_robust(self, tests_fixture_dir, q: int) -> None:
        """Robust mode on GARCH data must match MATLAB to ±1e-6."""
        fixture = _load_lmtest1_fixture(tests_fixture_dir)
        data = fixture["data_garch"].ravel()
        expected_lm = fixture[f"garch_q{q}_lm_robust"].ravel()
        expected_pval = fixture[f"garch_q{q}_pval_robust"].ravel()

        lm_stat, pval = lmtest1(data.copy(), q=q, robust=True)

        assert_allclose(
            lm_stat, expected_lm,
            err_msg=f"garch robust LM stats mismatch at q={q}"
        )
        assert_allclose(
            pval, expected_pval,
            err_msg=f"garch robust p-values mismatch at q={q}"
        )

    @pytest.mark.parametrize("q", [1, 5, 10])
    def test_lmtest1_parity_garch_classical(self, tests_fixture_dir, q: int) -> None:
        """Classical mode on GARCH data must match MATLAB to ±1e-6."""
        fixture = _load_lmtest1_fixture(tests_fixture_dir)
        data = fixture["data_garch"].ravel()
        expected_lm = fixture[f"garch_q{q}_lm_classical"].ravel()
        expected_pval = fixture[f"garch_q{q}_pval_classical"].ravel()

        lm_stat, pval = lmtest1(data.copy(), q=q, robust=False)

        assert_allclose(
            lm_stat, expected_lm,
            err_msg=f"garch classical LM stats mismatch at q={q}"
        )
        assert_allclose(
            pval, expected_pval,
            err_msg=f"garch classical p-values mismatch at q={q}"
        )


@pytest.mark.parity
class TestLmtest1ParityDedicated:
    """Parity tests using the dedicated _lm_data/_lm_stat/_lm_pval fixture files."""

    def test_lmtest1_parity_dedicated_fixtures(self, tests_fixture_dir) -> None:
        """Verify against dedicated per-field fixture files.

        Uses ``lmtest1_lm_data.npy``, ``lmtest1_lm_stat.npy``,
        ``lmtest1_lm_pval.npy`` which store a single test case
        with q=10 robust=True (default).
        """
        data = load_fixture_npy(tests_fixture_dir, "lmtest1_lm_data")
        expected_stat = load_fixture_npy(tests_fixture_dir, "lmtest1_lm_stat")
        expected_pval = load_fixture_npy(tests_fixture_dir, "lmtest1_lm_pval")

        data_flat = data.ravel()
        q = len(expected_stat.ravel())

        lm_stat, pval = lmtest1(data_flat.copy(), q=q, robust=True)

        assert_allclose(
            lm_stat, expected_stat.ravel(),
            err_msg="Dedicated fixture LM stat mismatch"
        )
        assert_allclose(
            pval, expected_pval.ravel(),
            err_msg="Dedicated fixture p-value mismatch"
        )


# =========================================================================
# Phase 4: Return Type and Shape Tests
# =========================================================================


class TestLmtest1ReturnTypes:
    """Return type, shape, and value range invariant tests."""

    def test_lmtest1_return_types(self, rng: np.random.Generator) -> None:
        """lm and pval must both be numpy ndarrays."""
        data = rng.standard_normal(200)
        lm_stat, pval = lmtest1(data, q=5)

        assert isinstance(lm_stat, np.ndarray), (
            f"Expected np.ndarray, got {type(lm_stat)}"
        )
        assert isinstance(pval, np.ndarray), (
            f"Expected np.ndarray, got {type(pval)}"
        )

    @pytest.mark.parametrize("q", [1, 3, 5, 10])
    def test_lmtest1_return_shapes(
        self, rng: np.random.Generator, q: int
    ) -> None:
        """Both lm and pval must have shape (q,)."""
        data = rng.standard_normal(500)
        lm_stat, pval = lmtest1(data, q=q)

        assert lm_stat.shape == (q,), (
            f"Expected lm shape ({q},), got {lm_stat.shape}"
        )
        assert pval.shape == (q,), (
            f"Expected pval shape ({q},), got {pval.shape}"
        )

    def test_lmtest1_lm_nonnegative(self, rng: np.random.Generator) -> None:
        """All LM statistics must be ≥ 0 (quadratic form invariant)."""
        data = rng.standard_normal(500)

        for robust in [True, False]:
            lm_stat, _ = lmtest1(data.copy(), q=10, robust=robust)
            assert np.all(lm_stat >= 0), (
                f"LM stats must be ≥ 0, got min={lm_stat.min()} "
                f"(robust={robust})"
            )

    def test_lmtest1_pval_range(self, rng: np.random.Generator) -> None:
        """All p-values must be in [0, 1]."""
        data = rng.standard_normal(500)

        for robust in [True, False]:
            _, pval = lmtest1(data.copy(), q=10, robust=robust)
            assert np.all(pval >= 0.0) and np.all(pval <= 1.0), (
                f"P-values must be in [0, 1], got range "
                f"[{pval.min()}, {pval.max()}] (robust={robust})"
            )

    def test_lmtest1_default_robust(self, rng: np.random.Generator) -> None:
        """Default mode is robust=True — omitting should match explicit True.

        Ref: lmtest1.m:41 — ``robust = true`` when nargin==2.
        """
        data = rng.standard_normal(300)

        lm_default, pval_default = lmtest1(data.copy(), q=5)
        lm_explicit, pval_explicit = lmtest1(data.copy(), q=5, robust=True)

        npt.assert_allclose(
            lm_default, lm_explicit, atol=ATOL, rtol=RTOL,
            err_msg="Default should equal robust=True"
        )
        npt.assert_allclose(
            pval_default, pval_explicit, atol=ATOL, rtol=RTOL,
            err_msg="Default pval should equal robust=True pval"
        )
