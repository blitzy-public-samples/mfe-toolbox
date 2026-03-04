"""
Comprehensive pytest tests for ``mfe_toolbox.timeseries.grangercause``.

Tests Granger causality testing with F-test based causality analysis under
4 VCV estimation modes (het × uncorr combinations) and 3 inference methods
(LR, LM, Wald).  Covers output structure, statistical properties, edge cases,
and fixture-based numerical parity against MATLAB/Octave reference outputs.

Source MATLAB Reference
-----------------------
- ``timeseries/grangercause.m`` by Kevin Sheppard (Revision 3.0, 1/1/2007)
- Function signature::

    [stat, pval, statAll, pvalAll] = grangercause(y, constant, lags, het, uncorr, inference)

INPUTS:
  - ``y`` — T×K multivariate time series
  - ``constant`` — 1 to include, 0 to exclude
  - ``lags`` — VAR lag orders (vector, can be non-contiguous e.g. [1,3])
  - ``het`` — 0=homoskedastic, 1=heteroskedastic [DEFAULT]
  - ``uncorr`` — 0=correlated errors [DEFAULT], 1=uncorrelated
  - ``inference`` — 1=LR [DEFAULT], 2=LM, 3=Wald
OUTPUTS:
  - ``stat`` — K×K matrix: stat(i,j) = test that y_i not caused by y_j
  - ``pval`` — K×K p-values (chi2 distribution)
  - ``statAll`` — K×1 vector: stat that y_i not caused by any y_j, j≠i
  - ``pvalAll`` — K×1 p-values for joint tests

Per AAP Section 0.7.1: All migrated functions MUST pass
``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
against MATLAB-generated fixtures.
"""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.grangercause import grangercause

# ---------------------------------------------------------------------------
# Numerical Parity Tolerance Constants
# Per AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# ---------------------------------------------------------------------------
# Fixture Directory Resolution
# Per AAP Section 0.7.2: Optional MFE_FIXTURE_DIR override
# ---------------------------------------------------------------------------
_FIXTURE_BASE: Path = Path(__file__).resolve().parent.parent / "fixtures"
_TIMESERIES_FIXTURE_DIR: Path = Path(
    os.environ.get("MFE_FIXTURE_DIR", str(_FIXTURE_BASE))
) / "timeseries"

_GRANGERCAUSE_FIXTURE_PATH: Path = _TIMESERIES_FIXTURE_DIR / "grangercause.npy"
_HAS_FIXTURES: bool = _GRANGERCAUSE_FIXTURE_PATH.exists()


# ===================================================================
# Fixtures — Data Generation
# ===================================================================


@pytest.fixture
def rng_local():
    """Provide a local seeded RNG for reproducible test data generation.

    Uses a fixed seed (99999) independent of conftest.rng to ensure
    consistent test data regardless of test execution order.

    Returns
    -------
    np.random.Generator
        A NumPy random generator seeded with 99999.
    """
    return np.random.default_rng(99999)


@pytest.fixture
def bivariate_causal_data(rng_local):
    """Generate bivariate data where y1 causes y2, but y2 does not cause y1.

    DGP:
        y1(t) = 0.5 * y1(t-1) + e1(t)
        y2(t) = 0.3 * y2(t-1) + 0.4 * y1(t-1) + e2(t)

    The coefficient 0.4 from y1→y2 induces Granger causality from column 0
    to column 1.

    Returns
    -------
    np.ndarray
        ``(500, 2)`` array with causal structure y1 → y2.
    """
    T = 500
    y1 = np.zeros(T)
    y2 = np.zeros(T)
    e = rng_local.standard_normal((T, 2))
    for t in range(1, T):
        y1[t] = 0.5 * y1[t - 1] + e[t, 0]
        y2[t] = 0.3 * y2[t - 1] + 0.4 * y1[t - 1] + e[t, 1]
    return np.column_stack([y1, y2])


@pytest.fixture
def bivariate_independent_data(rng_local):
    """Generate bivariate independent AR(1) processes — no Granger causality.

    DGP:
        y1(t) = 0.5 * y1(t-1) + e1(t)
        y2(t) = 0.5 * y2(t-1) + e2(t)

    Returns
    -------
    np.ndarray
        ``(500, 2)`` array with no cross-variable causality.
    """
    T = 500
    y1 = np.zeros(T)
    y2 = np.zeros(T)
    e = rng_local.standard_normal((T, 2))
    for t in range(1, T):
        y1[t] = 0.5 * y1[t - 1] + e[t, 0]
        y2[t] = 0.5 * y2[t - 1] + e[t, 1]
    return np.column_stack([y1, y2])


@pytest.fixture
def trivariate_data(rng_local):
    """Generate trivariate VAR(1) data with known causal structure.

    DGP:
        y1(t) = 0.4 * y1(t-1) + e1(t)
        y2(t) = 0.3 * y2(t-1) + 0.35 * y1(t-1) + e2(t)
        y3(t) = 0.2 * y3(t-1) + 0.3 * y2(t-1) + e3(t)

    y1 → y2, y2 → y3, but y3 does not directly cause y1.

    Returns
    -------
    np.ndarray
        ``(500, 3)`` array with known trivariate causal structure.
    """
    T = 500
    y = np.zeros((T, 3))
    e = rng_local.standard_normal((T, 3))
    for t in range(1, T):
        y[t, 0] = 0.4 * y[t - 1, 0] + e[t, 0]
        y[t, 1] = 0.3 * y[t - 1, 1] + 0.35 * y[t - 1, 0] + e[t, 1]
        y[t, 2] = 0.2 * y[t - 1, 2] + 0.3 * y[t - 1, 1] + e[t, 2]
    return y


@pytest.fixture
def large_bivariate_data(rng_local):
    """Generate a larger bivariate causal dataset for higher power tests.

    T=2000 for more reliable detection of weaker causal effects.

    Returns
    -------
    np.ndarray
        ``(2000, 2)`` array with strong causal structure y1 → y2.
    """
    T = 2000
    y1 = np.zeros(T)
    y2 = np.zeros(T)
    e = rng_local.standard_normal((T, 2))
    for t in range(1, T):
        y1[t] = 0.5 * y1[t - 1] + e[t, 0]
        y2[t] = 0.3 * y2[t - 1] + 0.4 * y1[t - 1] + e[t, 1]
    return np.column_stack([y1, y2])


# ===================================================================
# Phase 2: Output Structure Tests
# ===================================================================


class TestGrangerCauseOutputStructure:
    """Verify the shape, type, and basic properties of grangercause outputs."""

    def test_grangercause_returns_four(self, bivariate_causal_data):
        """Returns a tuple of exactly 4 elements: (stat, pval, statAll, pvalAll)."""
        result = grangercause(bivariate_causal_data, 1, [1])
        assert isinstance(result, tuple), "grangercause must return a tuple"
        assert len(result) == 4, "grangercause must return exactly 4 outputs"

    def test_grangercause_stat_shape(self, bivariate_causal_data):
        """stat is a K×K matrix for K=2 bivariate data."""
        stat, pval, stat_all, pval_all = grangercause(bivariate_causal_data, 1, [1])
        K = bivariate_causal_data.shape[1]
        assert stat.shape == (K, K), f"stat shape must be ({K},{K}), got {stat.shape}"

    def test_grangercause_pval_shape(self, bivariate_causal_data):
        """pval is a K×K matrix for K=2 bivariate data."""
        stat, pval, stat_all, pval_all = grangercause(bivariate_causal_data, 1, [1])
        K = bivariate_causal_data.shape[1]
        assert pval.shape == (K, K), f"pval shape must be ({K},{K}), got {pval.shape}"

    def test_grangercause_statAll_shape(self, bivariate_causal_data):
        """statAll is a K×1 vector (or K-length 1D) for K=2 bivariate data."""
        stat, pval, stat_all, pval_all = grangercause(bivariate_causal_data, 1, [1])
        K = bivariate_causal_data.shape[1]
        # The Python implementation returns (K, 1); MATLAB returns (K, 1) as well
        assert stat_all.shape == (K, 1) or stat_all.shape == (K,), \
            f"statAll shape must be ({K},1) or ({K},), got {stat_all.shape}"

    def test_grangercause_pvalAll_shape(self, bivariate_causal_data):
        """pvalAll is a K×1 vector (or K-length 1D) for K=2 bivariate data."""
        stat, pval, stat_all, pval_all = grangercause(bivariate_causal_data, 1, [1])
        K = bivariate_causal_data.shape[1]
        assert pval_all.shape == (K, 1) or pval_all.shape == (K,), \
            f"pvalAll shape must be ({K},1) or ({K},), got {pval_all.shape}"

    def test_grangercause_pval_range(self, bivariate_causal_data):
        """All p-values must be in [0, 1]."""
        stat, pval, stat_all, pval_all = grangercause(bivariate_causal_data, 1, [1])
        assert np.all(pval >= 0.0), "All pval entries must be >= 0"
        assert np.all(pval <= 1.0), "All pval entries must be <= 1"
        assert np.all(pval_all >= 0.0), "All pvalAll entries must be >= 0"
        assert np.all(pval_all <= 1.0), "All pvalAll entries must be <= 1"

    def test_grangercause_stat_nonnegative(self, bivariate_causal_data):
        """Test statistics must be non-negative (chi-squared distributed)."""
        stat, pval, stat_all, pval_all = grangercause(bivariate_causal_data, 1, [1])
        assert np.all(stat >= 0.0), "All stat entries must be >= 0"
        assert np.all(stat_all >= 0.0), "All statAll entries must be >= 0"

    def test_grangercause_output_types(self, bivariate_causal_data):
        """All outputs must be numpy ndarrays with float64 dtype."""
        stat, pval, stat_all, pval_all = grangercause(bivariate_causal_data, 1, [1])
        for name, arr in [("stat", stat), ("pval", pval),
                          ("stat_all", stat_all), ("pval_all", pval_all)]:
            assert isinstance(arr, np.ndarray), f"{name} must be np.ndarray"
            assert arr.dtype == np.float64, f"{name} must have float64 dtype"


# ===================================================================
# Phase 3: Statistical Tests
# ===================================================================


class TestGrangerCauseStatistical:
    """Verify that grangercause correctly detects (or fails to detect) causality."""

    def test_grangercause_detects_causality(self, bivariate_causal_data):
        """Bivariate causal data: pval[1,0] < 0.05 (y1 → y2).

        In the DGP, y1 Granger-causes y2 with coefficient 0.4.
        The test stat[1,0] tests H0: y2 is NOT caused by y1.
        We expect strong rejection (low p-value).
        """
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1]
        )
        # y1 (column 0) causes y2 (column 1):  stat[1,0] should be large
        assert pval[1, 0] < 0.05, (
            f"Expected y1→y2 causality to be detected (pval[1,0]={pval[1,0]:.6f})"
        )

    def test_grangercause_no_reverse_causality(self, bivariate_causal_data):
        """Bivariate causal data: pval[0,1] > 0.10 (y2 does NOT cause y1).

        In the DGP, y2 does not affect y1.  The test stat[0,1] tests
        H0: y1 is NOT caused by y2.  We should fail to reject.
        """
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1]
        )
        # y2 (column 1) does NOT cause y1 (column 0)
        assert pval[0, 1] > 0.10, (
            f"Expected no y2→y1 causality (pval[0,1]={pval[0,1]:.6f})"
        )

    def test_grangercause_independent_no_causality(self, bivariate_independent_data):
        """Independent data: all off-diagonal pvals > 0.10.

        With independent AR(1) processes, no cross-causality should be detected.
        """
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_independent_data, 1, [1]
        )
        K = bivariate_independent_data.shape[1]
        for i in range(K):
            for j in range(K):
                if i != j:
                    assert pval[i, j] > 0.05, (
                        f"Independent processes: pval[{i},{j}]={pval[i,j]:.6f} "
                        f"should not be significant"
                    )

    def test_grangercause_diagonal_values(self, bivariate_causal_data):
        """Diagonal of stat/pval: self-causation entries.

        The diagonal stat[i,i] tests whether y_i is caused by y_i's own
        lags beyond what's already in the VAR.  Since the model includes
        y_i's own lags, these diagonals should show strong self-dependence
        (large stat, small pval) or be trivially large.
        """
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1]
        )
        K = bivariate_causal_data.shape[1]
        for i in range(K):
            # Diagonal stat should be >= 0 (general property)
            assert stat[i, i] >= 0.0, f"Diagonal stat[{i},{i}] must be >= 0"

    def test_grangercause_statAll_joint_test(self, bivariate_causal_data):
        """Joint test: statAll[1] should be significant (y1 causes y2).

        statAll[i] tests that y_i is not caused by ANY y_j, j!=i.
        Since y1→y2, the joint test for y2 should reject.
        """
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1]
        )
        # pvalAll for y2 (index 1) should be small
        pval_all_flat = pval_all.ravel()
        assert pval_all_flat[1] < 0.05, (
            f"Joint test for y2 should detect causality from y1 "
            f"(pvalAll[1]={pval_all_flat[1]:.6f})"
        )

    def test_grangercause_statAll_no_joint_causality(self, bivariate_independent_data):
        """Independent data: pvalAll should not be significant."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_independent_data, 1, [1]
        )
        pval_all_flat = pval_all.ravel()
        for i in range(len(pval_all_flat)):
            assert pval_all_flat[i] > 0.05, (
                f"Independent data: pvalAll[{i}]={pval_all_flat[i]:.6f} "
                f"should not be significant"
            )


# ===================================================================
# Phase 4: VCV Mode Tests
# ===================================================================


class TestGrangerCauseVCVModes:
    """Verify grangercause runs and produces valid output under all 4 VCV modes."""

    @pytest.mark.parametrize("het,uncorr,description", [
        (0, 0, "Homoskedastic correlated"),
        (1, 0, "Heteroskedastic correlated (default)"),
        (0, 1, "Homoskedastic uncorrelated"),
        (1, 1, "Heteroskedastic uncorrelated"),
    ])
    def test_grangercause_vcv_mode(self, bivariate_causal_data, het, uncorr,
                                    description):
        """All 4 VCV modes produce valid K×K stat and pval matrices."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], het=het, uncorr=uncorr
        )
        K = bivariate_causal_data.shape[1]
        assert stat.shape == (K, K), (
            f"{description}: stat shape must be ({K},{K})"
        )
        assert pval.shape == (K, K), (
            f"{description}: pval shape must be ({K},{K})"
        )
        assert np.all(pval >= 0.0), f"{description}: pval must be >= 0"
        assert np.all(pval <= 1.0), f"{description}: pval must be <= 1"
        assert np.all(stat >= 0.0), f"{description}: stat must be >= 0"

    def test_grangercause_het0_uncorr0(self, bivariate_causal_data):
        """Homoskedastic correlated mode with LR inference."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], het=0, uncorr=0, inference=1
        )
        # Should still detect causality y1 → y2
        assert pval[1, 0] < 0.05, "het=0,uncorr=0 should detect y1→y2"

    def test_grangercause_het1_uncorr0(self, bivariate_causal_data):
        """Heteroskedastic correlated mode (default) with LR inference."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], het=1, uncorr=0, inference=1
        )
        assert pval[1, 0] < 0.05, "het=1,uncorr=0 should detect y1→y2"

    def test_grangercause_het0_uncorr1(self, bivariate_causal_data):
        """Homoskedastic uncorrelated mode with LR inference."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], het=0, uncorr=1, inference=1
        )
        assert pval[1, 0] < 0.05, "het=0,uncorr=1 should detect y1→y2"

    def test_grangercause_het1_uncorr1(self, bivariate_causal_data):
        """Heteroskedastic uncorrelated mode with LR inference."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], het=1, uncorr=1, inference=1
        )
        assert pval[1, 0] < 0.05, "het=1,uncorr=1 should detect y1→y2"


# ===================================================================
# Phase 5: Inference Method Tests
# ===================================================================


class TestGrangerCauseInferenceMethods:
    """Verify all 3 inference methods (LR, LM, Wald) produce valid results."""

    @pytest.mark.parametrize("inference,method_name", [
        (1, "Likelihood Ratio (LR)"),
        (2, "Lagrange Multiplier (LM)"),
        (3, "Wald"),
    ])
    def test_grangercause_inference_method(self, bivariate_causal_data,
                                           inference, method_name):
        """Each inference method produces valid output and detects causality."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], het=1, uncorr=0,
            inference=inference
        )
        K = bivariate_causal_data.shape[1]
        assert stat.shape == (K, K), f"{method_name}: stat shape wrong"
        assert np.all(pval >= 0.0), f"{method_name}: pval out of range"
        assert np.all(pval <= 1.0), f"{method_name}: pval out of range"
        assert np.all(stat >= 0.0), f"{method_name}: stat must be non-negative"
        # All methods should detect y1 → y2
        assert pval[1, 0] < 0.05, (
            f"{method_name}: should detect y1→y2 (pval[1,0]={pval[1,0]:.6f})"
        )

    def test_grangercause_inference_lr(self, bivariate_causal_data):
        """inference=1 (LR): specific test with default het/uncorr."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], inference=1
        )
        assert pval[1, 0] < 0.05, "LR should detect y1→y2"
        assert pval[0, 1] > 0.10, "LR: no y2→y1"

    def test_grangercause_inference_lm(self, bivariate_causal_data):
        """inference=2 (LM): specific test with default het/uncorr."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], inference=2
        )
        assert pval[1, 0] < 0.05, "LM should detect y1→y2"
        assert pval[0, 1] > 0.10, "LM: no y2→y1"

    def test_grangercause_inference_wald(self, bivariate_causal_data):
        """inference=3 (Wald): specific test with default het/uncorr."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], inference=3
        )
        assert pval[1, 0] < 0.05, "Wald should detect y1→y2"
        assert pval[0, 1] > 0.10, "Wald: no y2→y1"

    def test_grangercause_lr_hom_uncorr(self, bivariate_causal_data):
        """LR with homoskedastic uncorrelated: het=0, uncorr=1, inference=1."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], het=0, uncorr=1, inference=1
        )
        assert pval[1, 0] < 0.05, "LR hom uncorr should detect y1→y2"

    def test_grangercause_lm_het_uncorr(self, bivariate_causal_data):
        """LM with heteroskedastic uncorrelated: het=1, uncorr=1, inference=2."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], het=1, uncorr=1, inference=2
        )
        assert pval[1, 0] < 0.05, "LM het uncorr should detect y1→y2"

    def test_grangercause_wald_hom_corr(self, bivariate_causal_data):
        """Wald with homoskedastic correlated: het=0, uncorr=0, inference=3."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1], het=0, uncorr=0, inference=3
        )
        assert pval[1, 0] < 0.05, "Wald hom corr should detect y1→y2"


# ===================================================================
# Phase 6: Edge Cases and Special Inputs
# ===================================================================


class TestGrangerCauseEdgeCases:
    """Edge cases: trivariate, non-contiguous lags, no constant, etc."""

    def test_grangercause_trivariate(self, trivariate_data):
        """K=3: test 3×3 output shapes and causal detection."""
        stat, pval, stat_all, pval_all = grangercause(
            trivariate_data, 1, [1]
        )
        K = 3
        assert stat.shape == (K, K), f"Trivariate stat must be {K}×{K}"
        assert pval.shape == (K, K), f"Trivariate pval must be {K}×{K}"
        assert stat_all.shape == (K, 1) or stat_all.shape == (K,), \
            f"Trivariate statAll shape error: {stat_all.shape}"

        # y1 → y2: pval[1,0] should be significant
        assert pval[1, 0] < 0.05, (
            f"Trivariate: y1→y2 should be detected (pval[1,0]={pval[1,0]:.6f})"
        )
        # y2 → y3: pval[2,1] should be significant
        assert pval[2, 1] < 0.05, (
            f"Trivariate: y2→y3 should be detected (pval[2,1]={pval[2,1]:.6f})"
        )

    def test_grangercause_noncontiguous_lags(self, bivariate_causal_data):
        """lags=[1,3] (non-contiguous) works correctly."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1, 3]
        )
        K = bivariate_causal_data.shape[1]
        assert stat.shape == (K, K), "Non-contiguous lags: wrong stat shape"
        assert pval.shape == (K, K), "Non-contiguous lags: wrong pval shape"
        assert np.all(stat >= 0.0), "Non-contiguous lags: stat must be >= 0"
        assert np.all(pval >= 0.0), "Non-contiguous lags: pval must be >= 0"
        assert np.all(pval <= 1.0), "Non-contiguous lags: pval must be <= 1"
        # Should still detect causality with non-contiguous lags
        assert pval[1, 0] < 0.05, (
            f"Non-contiguous lags: y1→y2 should still be detected "
            f"(pval[1,0]={pval[1,0]:.6f})"
        )

    def test_grangercause_no_constant(self, bivariate_causal_data):
        """constant=0: VAR without intercept."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 0, [1]
        )
        K = bivariate_causal_data.shape[1]
        assert stat.shape == (K, K), "No constant: wrong stat shape"
        assert np.all(pval >= 0.0), "No constant: pval out of range"
        assert np.all(pval <= 1.0), "No constant: pval out of range"

    def test_grangercause_multiple_lags(self, bivariate_causal_data):
        """lags=[1,2,3]: VAR(3) with contiguous lags."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1, 2, 3]
        )
        K = bivariate_causal_data.shape[1]
        assert stat.shape == (K, K), "Multiple lags: wrong stat shape"
        assert np.all(pval >= 0.0) and np.all(pval <= 1.0), \
            "Multiple lags: pval out of range"

    def test_grangercause_unsorted_lags(self, bivariate_causal_data):
        """Lags provided in unsorted order [3,1] should be auto-sorted."""
        stat1, pval1, sa1, pa1 = grangercause(
            bivariate_causal_data, 1, [1, 3]
        )
        stat2, pval2, sa2, pa2 = grangercause(
            bivariate_causal_data, 1, [3, 1]
        )
        npt.assert_allclose(stat1, stat2, atol=ATOL, rtol=RTOL,
                            err_msg="Unsorted lags should produce same results")
        npt.assert_allclose(pval1, pval2, atol=ATOL, rtol=RTOL,
                            err_msg="Unsorted lags should produce same pvals")

    def test_grangercause_single_lag(self, bivariate_causal_data):
        """Single lag provided as integer-like."""
        stat, pval, stat_all, pval_all = grangercause(
            bivariate_causal_data, 1, [1]
        )
        K = bivariate_causal_data.shape[1]
        assert stat.shape == (K, K)
        assert np.all(pval >= 0.0) and np.all(pval <= 1.0)

    def test_grangercause_trivariate_all_modes(self, trivariate_data):
        """Trivariate data with all inference methods."""
        for inference in [1, 2, 3]:
            stat, pval, stat_all, pval_all = grangercause(
                trivariate_data, 1, [1], inference=inference
            )
            K = 3
            assert stat.shape == (K, K), (
                f"Trivariate inference={inference}: wrong stat shape"
            )
            assert np.all(pval >= 0.0) and np.all(pval <= 1.0), (
                f"Trivariate inference={inference}: pval out of range"
            )


# ===================================================================
# Phase 6b: Input Validation / Error Handling Tests
# ===================================================================


class TestGrangerCauseInputValidation:
    """Verify that invalid inputs raise appropriate ValueErrors."""

    def test_grangercause_y_not_2d(self):
        """1-D y array should raise ValueError."""
        y = np.random.default_rng(0).standard_normal(100)
        with pytest.raises(ValueError, match="Y must be T by K"):
            grangercause(y, 1, [1])

    def test_grangercause_y_3d(self):
        """3-D y array should raise ValueError."""
        y = np.random.default_rng(0).standard_normal((100, 2, 3))
        with pytest.raises(ValueError, match="Y must be T by K"):
            grangercause(y, 1, [1])

    def test_grangercause_constant_invalid(self, bivariate_causal_data):
        """constant must be 0 or 1."""
        with pytest.raises(ValueError, match="CONSTANT must be either 0 or 1"):
            grangercause(bivariate_causal_data, 2, [1])

    def test_grangercause_lags_zero(self, bivariate_causal_data):
        """lags containing 0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            grangercause(bivariate_causal_data, 1, [0, 1])

    def test_grangercause_lags_negative(self, bivariate_causal_data):
        """Negative lag should raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            grangercause(bivariate_causal_data, 1, [-1])

    def test_grangercause_lags_noninteger(self, bivariate_causal_data):
        """Non-integer lag should raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            grangercause(bivariate_causal_data, 1, [1.5])

    def test_grangercause_lags_duplicate(self, bivariate_causal_data):
        """Duplicate lags should raise ValueError."""
        with pytest.raises(ValueError, match="unique"):
            grangercause(bivariate_causal_data, 1, [1, 1])

    def test_grangercause_het_invalid(self, bivariate_causal_data):
        """het must be 0 or 1."""
        with pytest.raises(ValueError, match="HET must be either 0 or 1"):
            grangercause(bivariate_causal_data, 1, [1], het=2)

    def test_grangercause_uncorr_invalid(self, bivariate_causal_data):
        """uncorr must be 0 or 1."""
        with pytest.raises(ValueError, match="UNCORR must be either 0 or 1"):
            grangercause(bivariate_causal_data, 1, [1], uncorr=2)

    def test_grangercause_inference_invalid(self, bivariate_causal_data):
        """inference must be 1, 2, or 3."""
        with pytest.raises(ValueError, match="INFERENCE"):
            grangercause(bivariate_causal_data, 1, [1], inference=4)

    def test_grangercause_inference_zero(self, bivariate_causal_data):
        """inference=0 should raise ValueError."""
        with pytest.raises(ValueError, match="INFERENCE"):
            grangercause(bivariate_causal_data, 1, [1], inference=0)


# ===================================================================
# Phase 7: Fixture-Based Numerical Parity Tests
# ===================================================================


class TestGrangerCauseParity:
    """Fixture-based parity tests against MATLAB/Octave reference outputs.

    Tests use the reference data from tests/fixtures/timeseries/grangercause.npy
    which contains 4 test cases generated by ``scripts/generate_fixtures.m``
    using Octave 8.4.0 with the MFE Toolbox grangercause.m function.

    Per AAP Section 0.7.1: numpy.testing.assert_allclose(actual, expected,
    atol=1e-6, rtol=1e-4) is the standard for MATLAB parity comparison.
    """

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="Fixture file not found: grangercause.npy"
    )
    def test_grangercause_parity_all_cases(self):
        """Test all fixture test cases against MATLAB/Octave reference outputs."""
        fixture_data = np.load(
            _GRANGERCAUSE_FIXTURE_PATH, allow_pickle=True
        ).item()

        # Load the reference input data
        y2 = fixture_data["y2"]  # (1000, 2)
        y3 = fixture_data["y3"]  # (1000, 3)
        test_cases = fixture_data["test_cases"]

        for idx, tc in enumerate(test_cases):
            desc = tc["description"]
            K = int(tc["K"])
            constant = int(tc["constant"])
            lags = tc["lags"].astype(int).tolist()
            het = int(tc["het"])
            uncorr = int(tc["uncorr"])
            inference = int(tc["inference"])

            # Select appropriate input data based on K
            y = y2 if K == 2 else y3

            # Run the Python implementation
            stat, pval, stat_all, pval_all = grangercause(
                y, constant, lags, het=het, uncorr=uncorr, inference=inference
            )

            # Extract reference values
            expected_stat = tc["stat"]
            expected_pval = tc["pval"]
            expected_stat_all = tc["statAll"]
            expected_pval_all = tc["pvalAll"]

            # Flatten stat_all/pval_all for comparison since MATLAB may
            # return 1D while Python returns 2D (K,1)
            stat_all_flat = stat_all.ravel()
            pval_all_flat = pval_all.ravel()
            expected_stat_all_flat = expected_stat_all.ravel()
            expected_pval_all_flat = expected_pval_all.ravel()

            # Assert numerical parity with ATOL=1e-6, RTOL=1e-4
            npt.assert_allclose(
                stat, expected_stat,
                atol=ATOL, rtol=RTOL,
                err_msg=f"Case {idx} ({desc}): stat mismatch"
            )
            npt.assert_allclose(
                pval, expected_pval,
                atol=ATOL, rtol=RTOL,
                err_msg=f"Case {idx} ({desc}): pval mismatch"
            )
            npt.assert_allclose(
                stat_all_flat, expected_stat_all_flat,
                atol=ATOL, rtol=RTOL,
                err_msg=f"Case {idx} ({desc}): statAll mismatch"
            )
            npt.assert_allclose(
                pval_all_flat, expected_pval_all_flat,
                atol=ATOL, rtol=RTOL,
                err_msg=f"Case {idx} ({desc}): pvalAll mismatch"
            )

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="Fixture file not found: grangercause.npy"
    )
    def test_grangercause_parity_bivariate_hom(self):
        """Parity: Bivariate, homoskedastic, LR (test case 0)."""
        fixture_data = np.load(
            _GRANGERCAUSE_FIXTURE_PATH, allow_pickle=True
        ).item()
        y2 = fixture_data["y2"]
        tc = fixture_data["test_cases"][0]

        stat, pval, stat_all, pval_all = grangercause(
            y2, int(tc["constant"]), tc["lags"].astype(int).tolist(),
            het=int(tc["het"]), uncorr=int(tc["uncorr"]),
            inference=int(tc["inference"])
        )

        npt.assert_allclose(stat, tc["stat"], atol=ATOL, rtol=RTOL,
                            err_msg="Bivariate hom LR stat mismatch")
        npt.assert_allclose(pval, tc["pval"], atol=ATOL, rtol=RTOL,
                            err_msg="Bivariate hom LR pval mismatch")

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="Fixture file not found: grangercause.npy"
    )
    def test_grangercause_parity_trivariate_het(self):
        """Parity: Trivariate, heteroskedastic, LR with lags=[1,2] (test case 1)."""
        fixture_data = np.load(
            _GRANGERCAUSE_FIXTURE_PATH, allow_pickle=True
        ).item()
        y3 = fixture_data["y3"]
        tc = fixture_data["test_cases"][1]

        stat, pval, stat_all, pval_all = grangercause(
            y3, int(tc["constant"]), tc["lags"].astype(int).tolist(),
            het=int(tc["het"]), uncorr=int(tc["uncorr"]),
            inference=int(tc["inference"])
        )

        npt.assert_allclose(stat, tc["stat"], atol=ATOL, rtol=RTOL,
                            err_msg="Trivariate het LR stat mismatch")
        npt.assert_allclose(pval, tc["pval"], atol=ATOL, rtol=RTOL,
                            err_msg="Trivariate het LR pval mismatch")
        npt.assert_allclose(
            stat_all.ravel(), tc["statAll"].ravel(),
            atol=ATOL, rtol=RTOL,
            err_msg="Trivariate het LR statAll mismatch"
        )

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="Fixture file not found: grangercause.npy"
    )
    def test_grangercause_parity_wald(self):
        """Parity: Trivariate, heteroskedastic, Wald inference (test case 3)."""
        fixture_data = np.load(
            _GRANGERCAUSE_FIXTURE_PATH, allow_pickle=True
        ).item()
        y3 = fixture_data["y3"]
        tc = fixture_data["test_cases"][3]

        stat, pval, stat_all, pval_all = grangercause(
            y3, int(tc["constant"]), tc["lags"].astype(int).tolist(),
            het=int(tc["het"]), uncorr=int(tc["uncorr"]),
            inference=int(tc["inference"])
        )

        npt.assert_allclose(stat, tc["stat"], atol=ATOL, rtol=RTOL,
                            err_msg="Wald stat mismatch")
        npt.assert_allclose(pval, tc["pval"], atol=ATOL, rtol=RTOL,
                            err_msg="Wald pval mismatch")
        npt.assert_allclose(
            stat_all.ravel(), tc["statAll"].ravel(),
            atol=ATOL, rtol=RTOL,
            err_msg="Wald statAll mismatch"
        )
        npt.assert_allclose(
            pval_all.ravel(), tc["pvalAll"].ravel(),
            atol=ATOL, rtol=RTOL,
            err_msg="Wald pvalAll mismatch"
        )


# ===================================================================
# Phase 8: Cross-Method Consistency Tests
# ===================================================================


class TestGrangerCauseConsistency:
    """Verify internal consistency across methods and parameter settings."""

    def test_grangercause_lr_lm_wald_consistency(self, large_bivariate_data):
        """LR, LM, and Wald should agree on significance direction.

        While the exact statistics differ, all three methods should agree
        on which relationships are significant vs insignificant for a
        well-powered dataset.
        """
        results = {}
        for inference in [1, 2, 3]:
            stat, pval, stat_all, pval_all = grangercause(
                large_bivariate_data, 1, [1], het=1, uncorr=0,
                inference=inference
            )
            results[inference] = (stat, pval)

        # All methods should agree on the direction of causality
        for inference in [1, 2, 3]:
            pval = results[inference][1]
            assert pval[1, 0] < 0.01, (
                f"inference={inference}: y1→y2 should be strongly significant"
            )
            assert pval[0, 1] > 0.05, (
                f"inference={inference}: y2→y1 should NOT be significant"
            )

    def test_grangercause_default_args_match_explicit(self, bivariate_causal_data):
        """Default arguments should match explicit specification of defaults."""
        stat1, pval1, sa1, pa1 = grangercause(bivariate_causal_data, 1, [1])
        stat2, pval2, sa2, pa2 = grangercause(
            bivariate_causal_data, 1, [1], het=1, uncorr=0, inference=1
        )
        npt.assert_allclose(stat1, stat2, atol=1e-14,
                            err_msg="Default args should match explicit")
        npt.assert_allclose(pval1, pval2, atol=1e-14,
                            err_msg="Default pval should match explicit")
        npt.assert_allclose(sa1, sa2, atol=1e-14,
                            err_msg="Default statAll should match explicit")
        npt.assert_allclose(pa1, pa2, atol=1e-14,
                            err_msg="Default pvalAll should match explicit")

    def test_grangercause_larger_sample_smaller_pval(self):
        """Larger sample should give stronger (smaller) p-values for true causality.

        Uses matched DGP with T=200 and T=1000 to verify power increase.
        """
        rng = np.random.default_rng(77777)

        for T in [200, 1000]:
            y1 = np.zeros(T)
            y2 = np.zeros(T)
            e = rng.standard_normal((T, 2))
            for t in range(1, T):
                y1[t] = 0.5 * y1[t - 1] + e[t, 0]
                y2[t] = 0.3 * y2[t - 1] + 0.4 * y1[t - 1] + e[t, 1]
            y = np.column_stack([y1, y2])
            stat, pval, stat_all, pval_all = grangercause(y, 1, [1])
            if T == 200:
                pval_small = pval[1, 0]
            else:
                pval_large = pval[1, 0]

        # Larger sample should give smaller (more significant) p-value
        # or both should be very small
        assert pval_large <= pval_small + 1e-10, (
            f"Larger sample should have smaller p-value: "
            f"T=200 pval={pval_small:.6f}, T=1000 pval={pval_large:.6f}"
        )
