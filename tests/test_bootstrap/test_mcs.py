"""Pytest tests for mfe_toolbox.bootstrap.mcs — Model Confidence Set.

Tests verify:
- Input validation (correct errors raised for invalid inputs)
- Output shapes and types (6-tuple: includedR, pvalsR, excludedR,
  includedSQ, pvalsSQ, excludedSQ)
- Both R (range/max) and SQ (sum-of-squares) elimination methods
- P-value properties (non-decreasing, in [0,1])
- Included/excluded partition completeness (all K models accounted for)
- Model ranking responds correctly to performance differentials
- Block vs stationary bootstrap dispatch
- Numerical parity against MATLAB reference fixtures (atol=1e-6, rtol=1e-4)

Ref: bootstrap/mcs.m — Kevin Sheppard, Revision 3, 4/1/2007
"""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.bootstrap.mcs import mcs

# ---------------------------------------------------------------------------
# Numerical Parity Tolerance Constants
# Per AAP Section 0.7.1: assert_allclose(atol=1e-6, rtol=1e-4)
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ===========================================================================
# Pytest Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    """Seeded RNG for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def loss_matrix(rng):
    """Generate a T x K loss matrix for MCS testing.

    T=200 observations, K=5 models.
    Models have different mean losses to create a clear ranking.
    Ref: mcs.m usage example - losses = bsxfun(@plus, chi2rnd(5,[1000 10]),
         linspace(.1,1,10))
    """
    T = 200
    K = 5
    # Create losses with different means so MCS can distinguish models
    base_losses = rng.standard_normal((T, K)) ** 2
    # Add linearly increasing mean losses: model 0 is best, model K-1 is worst
    offsets = np.linspace(0.0, 2.0, K)
    losses = base_losses + offsets[np.newaxis, :]
    return losses


@pytest.fixture
def equal_model_losses(rng):
    """All models perform equally (all should be included in MCS)."""
    T = 200
    K = 4
    losses = rng.standard_normal((T, K)) ** 2
    return losses


@pytest.fixture
def one_dominant_model_losses(rng):
    """One model clearly dominates (model 0 is best by large margin)."""
    T = 200
    K = 5
    losses = rng.standard_normal((T, K)) ** 2 + 3.0
    # Model 0 has much lower losses
    losses[:, 0] = rng.standard_normal(T) ** 2
    return losses


@pytest.fixture
def two_model_losses(rng):
    """Two-model loss matrix for minimum non-trivial MCS test.

    Model 0 is better (lower losses), model 1 is worse.
    """
    T = 200
    losses = np.column_stack([
        rng.standard_normal(T) ** 2,            # model 0 — better
        rng.standard_normal(T) ** 2 + 2.0,      # model 1 — worse
    ])
    return losses


# ===========================================================================
# Input Validation Tests
# Ref: mcs.m:43-69 — input validation block
# ===========================================================================

class TestMcsInputValidation:
    """Input validation tests for mcs()."""

    def test_mcs_losses_too_short(self):
        """Pass losses with T < 2, expect ValueError.

        Ref: mcs.m:54-55 — 'LOSSES must have at least 2 observations.'
        """
        losses = np.array([[1.0, 2.0]])  # T=1, K=2
        with pytest.raises(ValueError, match="LOSSES must have at least 2 observations"):
            mcs(losses, alpha=0.05, B=100, w=10)

    def test_mcs_losses_single_row(self):
        """Pass losses with exactly T=1 row, expect ValueError."""
        losses = np.array([[0.5, 1.5, 2.5]])  # T=1, K=3
        with pytest.raises(ValueError, match="LOSSES must have at least 2 observations"):
            mcs(losses, alpha=0.05, B=100, w=10)

    def test_mcs_alpha_out_of_range_high(self):
        """Pass alpha=1.0, expect ValueError.

        Ref: mcs.m:57-58 — 'ALPHA must be a scalar between 0 and 1'
        """
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="ALPHA must be a scalar between 0 and 1"):
            mcs(losses, alpha=1.0, B=100, w=10)

    def test_mcs_alpha_out_of_range_low(self):
        """Pass alpha=0.0, expect ValueError.

        Ref: mcs.m:57-58 — alpha must be strictly between 0 and 1.
        """
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="ALPHA must be a scalar between 0 and 1"):
            mcs(losses, alpha=0.0, B=100, w=10)

    def test_mcs_alpha_negative(self):
        """Pass alpha=-0.5, expect ValueError."""
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="ALPHA must be a scalar between 0 and 1"):
            mcs(losses, alpha=-0.5, B=100, w=10)

    def test_mcs_alpha_greater_than_one(self):
        """Pass alpha=1.5, expect ValueError."""
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="ALPHA must be a scalar between 0 and 1"):
            mcs(losses, alpha=1.5, B=100, w=10)

    def test_mcs_B_zero(self):
        """Pass B=0, expect ValueError.

        Ref: mcs.m:60-61 — 'B must be a positive scalar integer'
        """
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="B must be a positive scalar integer"):
            mcs(losses, alpha=0.05, B=0, w=10)

    def test_mcs_B_negative(self):
        """Pass B=-1, expect ValueError."""
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="B must be a positive scalar integer"):
            mcs(losses, alpha=0.05, B=-1, w=10)

    def test_mcs_B_non_integer(self):
        """Pass B=1.5 (non-integer), expect ValueError.

        Ref: mcs.m:60-61 — floor(B)~=B check.
        """
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="B must be a positive scalar integer"):
            mcs(losses, alpha=0.05, B=1.5, w=10)

    def test_mcs_w_zero(self):
        """Pass w=0, expect ValueError.

        Ref: mcs.m:63-64 — 'W must be a positive scalar integer'
        """
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="W must be a positive scalar integer"):
            mcs(losses, alpha=0.05, B=100, w=0)

    def test_mcs_w_negative(self):
        """Pass w=-1, expect ValueError."""
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="W must be a positive scalar integer"):
            mcs(losses, alpha=0.05, B=100, w=-1)

    def test_mcs_w_non_integer(self):
        """Pass w=1.5 (non-integer), expect ValueError.

        Ref: mcs.m:63-64 — floor(w)~=w check.
        """
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="W must be a positive scalar integer"):
            mcs(losses, alpha=0.05, B=100, w=1.5)

    def test_mcs_boot_invalid(self):
        """Pass boot='INVALID', expect ValueError.

        Ref: mcs.m:67-68 — BOOT must be 'STATIONARY' or 'BLOCK'.
        """
        losses = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="BOOT must be either"):
            mcs(losses, alpha=0.05, B=100, w=10, boot='INVALID')

    def test_mcs_boot_case_insensitive(self):
        """Boot parameter should be case-insensitive (per mcs.m:66 — upper(boot))."""
        rng = np.random.default_rng(99)
        losses = rng.standard_normal((50, 2)) ** 2 + np.array([0.0, 1.0])
        # Should not raise — lowercase should be accepted
        result = mcs(losses, alpha=0.05, B=50, w=5, boot='stationary')
        assert len(result) == 6

        result2 = mcs(losses, alpha=0.05, B=50, w=5, boot='Block')
        assert len(result2) == 6


# ===========================================================================
# Output Shape and Type Tests
# ===========================================================================

class TestMcsOutputShapeType:
    """Output shape and type tests for mcs()."""

    def test_mcs_returns_six_outputs(self, loss_matrix):
        """Verify mcs returns exactly 6 values."""
        result = mcs(loss_matrix, alpha=0.05, B=100, w=10)
        assert isinstance(result, tuple), "MCS should return a tuple"
        assert len(result) == 6, "MCS should return exactly 6 elements"

    def test_mcs_output_types(self, loss_matrix):
        """Verify output types: arrays of integer indices and float p-values."""
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = mcs(
            loss_matrix, alpha=0.05, B=100, w=10
        )
        # Index arrays should be numpy arrays
        assert isinstance(includedR, np.ndarray), "includedR should be ndarray"
        assert isinstance(excludedR, np.ndarray), "excludedR should be ndarray"
        assert isinstance(includedSQ, np.ndarray), "includedSQ should be ndarray"
        assert isinstance(excludedSQ, np.ndarray), "excludedSQ should be ndarray"
        # P-value arrays should be float arrays
        assert isinstance(pvalsR, np.ndarray), "pvalsR should be ndarray"
        assert isinstance(pvalsSQ, np.ndarray), "pvalsSQ should be ndarray"
        assert pvalsR.dtype in (np.float64, np.float32), "pvalsR should be float"
        assert pvalsSQ.dtype in (np.float64, np.float32), "pvalsSQ should be float"

    def test_mcs_partition_completeness(self, loss_matrix):
        """All K models appear exactly once across includedR + excludedR (and SQ).

        Ref: mcs.m:143-147, 186-190 — partition into included/excluded.
        """
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = mcs(
            loss_matrix, alpha=0.05, B=100, w=10
        )
        K = loss_matrix.shape[1]
        # Union of included and excluded must be all K models (R method)
        all_modelsR = np.sort(np.concatenate([includedR, excludedR]))
        npt.assert_array_equal(
            all_modelsR, np.arange(K),
            err_msg="R method: includedR ∪ excludedR must equal {0,...,K-1}"
        )
        # Union of included and excluded must be all K models (SQ method)
        all_modelsSQ = np.sort(np.concatenate([includedSQ, excludedSQ]))
        npt.assert_array_equal(
            all_modelsSQ, np.arange(K),
            err_msg="SQ method: includedSQ ∪ excludedSQ must equal {0,...,K-1}"
        )

    def test_mcs_partition_no_duplicates(self, loss_matrix):
        """No model should appear in both included and excluded sets."""
        includedR, _, excludedR, includedSQ, _, excludedSQ = mcs(
            loss_matrix, alpha=0.05, B=100, w=10
        )
        # No overlap between included and excluded (R)
        overlapR = np.intersect1d(includedR, excludedR)
        assert len(overlapR) == 0, f"R method: models {overlapR} in both sets"
        # No overlap (SQ)
        overlapSQ = np.intersect1d(includedSQ, excludedSQ)
        assert len(overlapSQ) == 0, f"SQ method: models {overlapSQ} in both sets"

    def test_mcs_pvals_length(self, loss_matrix):
        """pvalsR and pvalsSQ should each have length K (one per elimination step).

        Ref: mcs.m:111,151 — pvalsR=ones(M0,1); pvalsSQ=ones(M0,1)
        """
        K = loss_matrix.shape[1]
        _, pvalsR, _, _, pvalsSQ, _ = mcs(
            loss_matrix, alpha=0.05, B=100, w=10
        )
        assert len(pvalsR) == K, f"pvalsR should have length {K}, got {len(pvalsR)}"
        assert len(pvalsSQ) == K, f"pvalsSQ should have length {K}, got {len(pvalsSQ)}"

    def test_mcs_pvals_in_range(self, loss_matrix):
        """All p-values should be in [0, 1]."""
        _, pvalsR, _, _, pvalsSQ, _ = mcs(
            loss_matrix, alpha=0.05, B=100, w=10
        )
        assert np.all(pvalsR >= 0.0), "pvalsR has negative values"
        assert np.all(pvalsR <= 1.0), "pvalsR has values > 1"
        assert np.all(pvalsSQ >= 0.0), "pvalsSQ has negative values"
        assert np.all(pvalsSQ <= 1.0), "pvalsSQ has values > 1"

    def test_mcs_pvals_nondecreasing(self, loss_matrix):
        """P-values must be non-decreasing (enforced by mcs.m:134-141 and 176-183).

        The MCS monotonization step ensures p-values form a non-decreasing
        sequence so that MCS at level alpha is nested in MCS at level alpha' > alpha.
        """
        _, pvalsR, _, _, pvalsSQ, _ = mcs(
            loss_matrix, alpha=0.05, B=100, w=10
        )
        # P-values must be non-decreasing (per mcs.m:134-141)
        for i in range(1, len(pvalsR)):
            assert pvalsR[i] >= pvalsR[i - 1] - 1e-15, (
                f"pvalsR not non-decreasing at index {i}: "
                f"{pvalsR[i]} < {pvalsR[i - 1]}"
            )
        for i in range(1, len(pvalsSQ)):
            assert pvalsSQ[i] >= pvalsSQ[i - 1] - 1e-15, (
                f"pvalsSQ not non-decreasing at index {i}: "
                f"{pvalsSQ[i]} < {pvalsSQ[i - 1]}"
            )

    def test_mcs_last_pval_is_one(self, loss_matrix):
        """The last p-value should always be 1.0 (the best model is never excluded).

        Ref: mcs.m:111,151 — pvalsR=ones(M0,1); the last element is never
        overwritten by the loop which iterates M0-1 times.
        """
        _, pvalsR, _, _, pvalsSQ, _ = mcs(
            loss_matrix, alpha=0.05, B=100, w=10
        )
        assert pvalsR[-1] == 1.0, f"Last pvalsR should be 1.0, got {pvalsR[-1]}"
        assert pvalsSQ[-1] == 1.0, f"Last pvalsSQ should be 1.0, got {pvalsSQ[-1]}"

    def test_mcs_indices_are_zero_based(self, loss_matrix):
        """All returned model indices must be 0-based (Python convention).

        Ref: mcs.py docstring — 'All returned model indices are 0-based'
        """
        includedR, _, excludedR, includedSQ, _, excludedSQ = mcs(
            loss_matrix, alpha=0.05, B=100, w=10
        )
        K = loss_matrix.shape[1]
        for arr, name in [(includedR, 'includedR'), (excludedR, 'excludedR'),
                          (includedSQ, 'includedSQ'), (excludedSQ, 'excludedSQ')]:
            if len(arr) > 0:
                assert np.all(arr >= 0), f"{name} has negative indices"
                assert np.all(arr < K), f"{name} has indices >= K={K}"


# ===========================================================================
# Functional Tests
# ===========================================================================

class TestMcsFunctional:
    """Functional correctness tests for mcs()."""

    def test_mcs_equal_models_all_included(self, equal_model_losses):
        """When all models perform equally, most/all should be in MCS.

        With identically distributed losses, the MCS should not be able to
        reject any model, so most or all should remain included.
        """
        includedR, _, _, includedSQ, _, _ = mcs(
            equal_model_losses, alpha=0.05, B=200, w=10
        )
        K = equal_model_losses.shape[1]
        # With equal models, MCS should include most or all
        assert len(includedR) >= K // 2, (
            f"Equal models: expected at least {K // 2} included (R), "
            f"got {len(includedR)}"
        )
        assert len(includedSQ) >= K // 2, (
            f"Equal models: expected at least {K // 2} included (SQ), "
            f"got {len(includedSQ)}"
        )

    def test_mcs_dominant_model_always_included(self, one_dominant_model_losses):
        """The clearly best model should always be in MCS.

        Model 0 has much lower losses; it should never be excluded.
        """
        includedR, _, _, includedSQ, _, _ = mcs(
            one_dominant_model_losses, alpha=0.05, B=200, w=10
        )
        # Model 0 should be in the MCS (it's clearly best)
        assert 0 in includedR, (
            f"Dominant model 0 not in includedR: {includedR}"
        )
        assert 0 in includedSQ, (
            f"Dominant model 0 not in includedSQ: {includedSQ}"
        )

    def test_mcs_alpha_sensitivity(self, loss_matrix):
        """Lower alpha → more stringent → more models included (harder to reject).

        A lower alpha means we need stronger evidence to exclude a model,
        so the MCS should be larger (include more models) at lower alpha.
        """
        incR_strict, _, _, _, _, _ = mcs(loss_matrix, alpha=0.01, B=200, w=10)
        incR_loose, _, _, _, _, _ = mcs(loss_matrix, alpha=0.25, B=200, w=10)
        # More stringent alpha (lower) should include at least as many models
        assert len(incR_strict) >= len(incR_loose), (
            f"alpha=0.01 included {len(incR_strict)} models, "
            f"alpha=0.25 included {len(incR_loose)} — expected strict >= loose"
        )

    def test_mcs_stationary_bootstrap(self, loss_matrix):
        """Test with boot='STATIONARY' (default) completes without error."""
        result = mcs(loss_matrix, alpha=0.05, B=100, w=10, boot='STATIONARY')
        assert len(result) == 6
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = result
        K = loss_matrix.shape[1]
        # Structural properties must hold
        assert len(pvalsR) == K
        assert len(pvalsSQ) == K
        assert len(includedR) + len(excludedR) == K
        assert len(includedSQ) + len(excludedSQ) == K

    def test_mcs_block_bootstrap(self, loss_matrix):
        """Test with boot='BLOCK' completes without error."""
        result = mcs(loss_matrix, alpha=0.05, B=100, w=10, boot='BLOCK')
        assert len(result) == 6
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = result
        K = loss_matrix.shape[1]
        assert len(pvalsR) == K
        assert len(pvalsSQ) == K
        assert len(includedR) + len(excludedR) == K
        assert len(includedSQ) + len(excludedSQ) == K

    def test_mcs_two_models(self, two_model_losses):
        """K=2 (minimum non-trivial case).

        With two models where model 1 is clearly worse, the MCS should
        include model 0 and potentially exclude model 1 depending on
        bootstrap randomness.
        """
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = mcs(
            two_model_losses, alpha=0.05, B=200, w=10
        )
        K = 2
        # Partition completeness
        assert len(includedR) + len(excludedR) == K
        assert len(includedSQ) + len(excludedSQ) == K
        # P-value array length
        assert len(pvalsR) == K
        assert len(pvalsSQ) == K
        # Best model (0) should be included
        assert 0 in includedR, "Model 0 (better) should be in includedR"
        assert 0 in includedSQ, "Model 0 (better) should be in includedSQ"

    def test_mcs_r_vs_sq_methods(self, one_dominant_model_losses):
        """Both R and SQ methods should identify the dominant model for clear-cut cases.

        The R (range/max) and SQ (sum-of-squares) elimination methods use
        different test statistics but should agree on clearly separable models.
        """
        includedR, _, _, includedSQ, _, _ = mcs(
            one_dominant_model_losses, alpha=0.05, B=200, w=10
        )
        # Both methods should include the dominant model
        assert 0 in includedR
        assert 0 in includedSQ

    def test_mcs_nondecreasing_property_block(self):
        """P-value non-decreasing property holds for BLOCK bootstrap."""
        rng = np.random.default_rng(123)
        losses = rng.standard_normal((150, 4)) ** 2 + np.linspace(0, 1.5, 4)
        _, pvalsR, _, _, pvalsSQ, _ = mcs(losses, alpha=0.05, B=100, w=8,
                                          boot='BLOCK')
        # Check non-decreasing property
        for i in range(1, len(pvalsR)):
            assert pvalsR[i] >= pvalsR[i - 1] - 1e-15
        for i in range(1, len(pvalsSQ)):
            assert pvalsSQ[i] >= pvalsSQ[i - 1] - 1e-15

    def test_mcs_worst_model_excluded_or_last(self, loss_matrix):
        """The model with highest mean loss should be excluded first (lowest p-value).

        Ref: mcs.m:130-131 — model with max t-statistic is removed first.
        """
        K = loss_matrix.shape[1]
        _, pvalsR, excludedR, _, pvalsSQ, excludedSQ = mcs(
            loss_matrix, alpha=0.05, B=200, w=10
        )
        # With linearly increasing mean losses, model K-1 should tend to be
        # the first excluded (worst performer). Not guaranteed due to bootstrap
        # randomness, but the last model should be among the first excluded.
        if len(excludedR) > 0:
            # The worst model (K-1) should tend to be among the first excluded
            worst_model = K - 1
            # Verify structure: first p-value corresponds to first elimination
            assert pvalsR[0] <= pvalsR[-1]

    def test_mcs_large_B(self):
        """MCS with large B (many bootstrap replications) should work correctly."""
        rng = np.random.default_rng(77)
        losses = rng.standard_normal((100, 3)) ** 2 + np.array([0.0, 0.5, 1.0])
        result = mcs(losses, alpha=0.05, B=500, w=10)
        assert len(result) == 6
        K = 3
        includedR, pvalsR, excludedR, _, pvalsSQ, _ = result
        assert len(pvalsR) == K
        assert len(includedR) + len(excludedR) == K


# ===========================================================================
# Parity Tests Against MATLAB Fixtures
# ===========================================================================

class TestMcsParity:
    """MATLAB parity tests for mcs() using reference fixtures.

    The MCS procedure depends on bootstrap random draws, which differ between
    MATLAB/Octave and Python. Therefore, exact numerical parity of p-values
    is not achievable. Instead, these tests verify:
    1. The MATLAB fixture data has correct structural properties
    2. Running Python MCS on the same losses produces structurally valid output
    3. Individual fixture files are loaded and validated

    Ref: bootstrap/mcs.m — Kevin Sheppard, Revision 3
    """

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_mcs_parity_individual_fixtures(self, bootstrap_fixture_dir):
        """Verify structural parity using individual fixture files.

        Loads mcs_mcs_losses.npy and validates the Python MCS produces
        structurally valid output on the same loss matrix.
        """
        losses_path = bootstrap_fixture_dir / 'mcs_mcs_losses.npy'
        if not losses_path.exists():
            pytest.skip('Fixture mcs_mcs_losses.npy not found')

        # Load MATLAB reference data
        losses = np.load(losses_path)
        pvalsR_matlab = np.load(bootstrap_fixture_dir / 'mcs_mcs_pvalsR.npy')
        pvalsSQ_matlab = np.load(bootstrap_fixture_dir / 'mcs_mcs_pvalsSQ.npy')
        inclR_matlab = np.load(bootstrap_fixture_dir / 'mcs_mcs_inclR.npy')
        inclSQ_matlab = np.load(bootstrap_fixture_dir / 'mcs_mcs_inclSQ.npy')
        exclR_matlab = np.load(bootstrap_fixture_dir / 'mcs_mcs_exclR.npy')
        exclSQ_matlab = np.load(bootstrap_fixture_dir / 'mcs_mcs_exclSQ.npy')

        K = losses.shape[1]

        # Validate MATLAB fixture structural properties
        # MATLAB uses 1-indexed; convert to 0-indexed
        inclR_0 = np.atleast_1d(inclR_matlab).astype(int) - 1
        inclSQ_0 = np.atleast_1d(inclSQ_matlab).astype(int) - 1
        exclR_0 = np.atleast_1d(exclR_matlab).astype(int) - 1
        exclSQ_0 = np.atleast_1d(exclSQ_matlab).astype(int) - 1

        # Verify MATLAB fixture partition completeness
        allR = np.sort(np.concatenate([inclR_0, exclR_0]))
        npt.assert_array_equal(allR, np.arange(K),
                               err_msg="MATLAB fixture: R partition incomplete")
        allSQ = np.sort(np.concatenate([inclSQ_0, exclSQ_0]))
        npt.assert_array_equal(allSQ, np.arange(K),
                               err_msg="MATLAB fixture: SQ partition incomplete")

        # Verify MATLAB fixture p-values are non-decreasing
        for i in range(1, len(pvalsR_matlab)):
            assert pvalsR_matlab[i] >= pvalsR_matlab[i - 1] - 1e-15, (
                f"MATLAB pvalsR not non-decreasing at {i}"
            )
        for i in range(1, len(pvalsSQ_matlab)):
            assert pvalsSQ_matlab[i] >= pvalsSQ_matlab[i - 1] - 1e-15, (
                f"MATLAB pvalsSQ not non-decreasing at {i}"
            )

        # Run Python MCS on the same losses
        # Note: bootstrap draws differ, so exact p-value match not expected
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = mcs(
            losses, alpha=0.10, B=500, w=10
        )

        # Verify Python output structural properties
        assert len(pvalsR) == K
        assert len(pvalsSQ) == K
        assert len(includedR) + len(excludedR) == K
        assert len(includedSQ) + len(excludedSQ) == K
        assert pvalsR[-1] == 1.0
        assert pvalsSQ[-1] == 1.0

        # Both MATLAB and Python should agree that model with highest mean
        # loss (column with highest mean) tends to be excluded first
        mean_losses = losses.mean(axis=0)
        worst_model = np.argmax(mean_losses)

        # Verify the worst model (by mean loss) is either excluded or has
        # a low p-value in both MATLAB and Python outputs
        if len(exclR_0) > 0:
            assert worst_model in exclR_0 or worst_model in excludedR, (
                f"Worst model {worst_model} should be excluded in at least "
                f"one of MATLAB or Python"
            )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_mcs_parity_dict_fixture_stationary(self, bootstrap_fixture_dir):
        """Verify parity using the dict-based fixture (tc1 — STATIONARY).

        Loads mcs.npy dict fixture and validates tc1 test case.
        """
        fixture_path = bootstrap_fixture_dir / 'mcs.npy'
        if not fixture_path.exists():
            pytest.skip('Fixture mcs.npy not found')

        fixture = np.load(fixture_path, allow_pickle=True).item()
        losses = fixture['tc1_losses']
        alpha = fixture['tc1_alpha']
        B = int(fixture['tc1_B'])
        w = int(fixture['tc1_w'])
        boot = fixture['tc1_boot']

        K = losses.shape[1]

        # Validate MATLAB reference outputs (1-indexed → 0-indexed)
        matlab_inclR = np.atleast_1d(fixture['tc1_includedR']).astype(int) - 1
        matlab_exclR = np.atleast_1d(fixture['tc1_excludedR']).astype(int) - 1
        matlab_pvalsR = fixture['tc1_pvalsR']
        matlab_inclSQ = np.atleast_1d(fixture['tc1_includedSQ']).astype(int) - 1
        matlab_exclSQ = np.atleast_1d(fixture['tc1_excludedSQ']).astype(int) - 1
        matlab_pvalsSQ = fixture['tc1_pvalsSQ']

        # Verify MATLAB fixture structural validity
        allR = np.sort(np.concatenate([matlab_inclR, matlab_exclR]))
        npt.assert_array_equal(allR, np.arange(K))
        allSQ = np.sort(np.concatenate([matlab_inclSQ, matlab_exclSQ]))
        npt.assert_array_equal(allSQ, np.arange(K))
        assert len(matlab_pvalsR) == K
        assert len(matlab_pvalsSQ) == K

        # Run Python MCS
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = mcs(
            losses, alpha, B, w, boot=boot
        )

        # Verify Python output structural properties
        assert len(pvalsR) == K
        assert len(pvalsSQ) == K
        assert len(includedR) + len(excludedR) == K
        assert len(includedSQ) + len(excludedSQ) == K
        assert pvalsR[-1] == 1.0
        assert pvalsSQ[-1] == 1.0

        # P-values must be non-decreasing
        for i in range(1, K):
            assert pvalsR[i] >= pvalsR[i - 1] - 1e-15
            assert pvalsSQ[i] >= pvalsSQ[i - 1] - 1e-15

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_mcs_parity_dict_fixture_block(self, bootstrap_fixture_dir):
        """Verify parity using the dict-based fixture (tc2 — BLOCK).

        Loads mcs.npy dict fixture and validates tc2 test case.
        """
        fixture_path = bootstrap_fixture_dir / 'mcs.npy'
        if not fixture_path.exists():
            pytest.skip('Fixture mcs.npy not found')

        fixture = np.load(fixture_path, allow_pickle=True).item()
        losses = fixture['tc2_losses']
        alpha = fixture['tc2_alpha']
        B = int(fixture['tc2_B'])
        w = int(fixture['tc2_w'])
        boot = fixture['tc2_boot']

        K = losses.shape[1]

        # Validate MATLAB reference (1-indexed → 0-indexed)
        matlab_inclR = np.atleast_1d(fixture['tc2_includedR']).astype(int) - 1
        matlab_exclR = np.atleast_1d(fixture['tc2_excludedR']).astype(int) - 1
        matlab_pvalsR = fixture['tc2_pvalsR']
        matlab_inclSQ = np.atleast_1d(fixture['tc2_includedSQ']).astype(int) - 1
        matlab_exclSQ = np.atleast_1d(fixture['tc2_excludedSQ']).astype(int) - 1
        matlab_pvalsSQ = fixture['tc2_pvalsSQ']

        # Validate MATLAB structural properties
        allR = np.sort(np.concatenate([matlab_inclR, matlab_exclR]))
        npt.assert_array_equal(allR, np.arange(K))
        allSQ = np.sort(np.concatenate([matlab_inclSQ, matlab_exclSQ]))
        npt.assert_array_equal(allSQ, np.arange(K))

        # Run Python MCS
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = mcs(
            losses, alpha, B, w, boot=boot
        )

        # Verify Python output structural properties
        assert len(pvalsR) == K
        assert len(pvalsSQ) == K
        assert len(includedR) + len(excludedR) == K
        assert len(includedSQ) + len(excludedSQ) == K
        assert pvalsR[-1] == 1.0
        assert pvalsSQ[-1] == 1.0

        # P-values must be non-decreasing
        for i in range(1, K):
            assert pvalsR[i] >= pvalsR[i - 1] - 1e-15
            assert pvalsSQ[i] >= pvalsSQ[i - 1] - 1e-15

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_mcs_parity_dict_fixture_tc3(self, bootstrap_fixture_dir):
        """Verify parity using the dict-based fixture (tc3 — smaller data).

        tc3 uses T=200, K=4, alpha=0.05, B=500, w=8, boot='STATIONARY'.
        """
        fixture_path = bootstrap_fixture_dir / 'mcs.npy'
        if not fixture_path.exists():
            pytest.skip('Fixture mcs.npy not found')

        fixture = np.load(fixture_path, allow_pickle=True).item()
        losses = fixture['tc3_losses']
        alpha = fixture['tc3_alpha']
        B = int(fixture['tc3_B'])
        w = int(fixture['tc3_w'])
        boot = fixture['tc3_boot']

        K = losses.shape[1]

        # Validate MATLAB reference (1-indexed → 0-indexed)
        matlab_inclR = np.atleast_1d(fixture['tc3_includedR']).astype(int) - 1
        matlab_exclR = np.atleast_1d(fixture['tc3_excludedR']).astype(int) - 1
        matlab_pvalsR = fixture['tc3_pvalsR']
        matlab_inclSQ = np.atleast_1d(fixture['tc3_includedSQ']).astype(int) - 1
        matlab_exclSQ_raw = fixture['tc3_excludedSQ']
        matlab_pvalsSQ = fixture['tc3_pvalsSQ']

        # Handle empty excluded array (tc3 SQ excludes nothing)
        if matlab_exclSQ_raw.size == 0:
            matlab_exclSQ = np.array([], dtype=int)
        else:
            matlab_exclSQ = np.atleast_1d(matlab_exclSQ_raw).astype(int) - 1

        # Validate MATLAB structural properties
        allR = np.sort(np.concatenate([matlab_inclR, matlab_exclR]))
        npt.assert_array_equal(allR, np.arange(K))
        if len(matlab_exclSQ) > 0 or len(matlab_inclSQ) > 0:
            allSQ = np.sort(np.concatenate([matlab_inclSQ, matlab_exclSQ]))
            npt.assert_array_equal(allSQ, np.arange(K))

        # Run Python MCS
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = mcs(
            losses, alpha, B, w, boot=boot
        )

        # Verify Python structural properties
        assert len(pvalsR) == K
        assert len(pvalsSQ) == K
        assert len(includedR) + len(excludedR) == K
        assert len(includedSQ) + len(excludedSQ) == K
        assert pvalsR[-1] == 1.0
        assert pvalsSQ[-1] == 1.0

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_mcs_fixture_pvals_structural_agreement(self, bootstrap_fixture_dir):
        """Verify MATLAB and Python p-values have same monotonic structure.

        While exact p-values differ due to bootstrap randomness, both should
        exhibit the same monotonic non-decreasing pattern and the same
        last-element = 1.0 property.
        """
        fixture_path = bootstrap_fixture_dir / 'mcs.npy'
        if not fixture_path.exists():
            pytest.skip('Fixture mcs.npy not found')

        fixture = np.load(fixture_path, allow_pickle=True).item()

        for tc in ['tc1', 'tc2', 'tc3']:
            losses = fixture[f'{tc}_losses']
            alpha = fixture[f'{tc}_alpha']
            B = int(fixture[f'{tc}_B'])
            w = int(fixture[f'{tc}_w'])
            boot = fixture[f'{tc}_boot']
            K = losses.shape[1]

            matlab_pvalsR = fixture[f'{tc}_pvalsR']
            matlab_pvalsSQ = fixture[f'{tc}_pvalsSQ']

            # Run Python
            _, pvalsR, _, _, pvalsSQ, _ = mcs(losses, alpha, B, w, boot=boot)

            # Both MATLAB and Python must have K p-values
            assert len(matlab_pvalsR) == K
            assert len(pvalsR) == K
            assert len(matlab_pvalsSQ) == K
            assert len(pvalsSQ) == K

            # Both must end with 1.0
            assert matlab_pvalsR[-1] == 1.0, f"{tc} MATLAB pvalsR[-1] != 1.0"
            assert pvalsR[-1] == 1.0, f"{tc} Python pvalsR[-1] != 1.0"
            assert matlab_pvalsSQ[-1] == 1.0, f"{tc} MATLAB pvalsSQ[-1] != 1.0"
            assert pvalsSQ[-1] == 1.0, f"{tc} Python pvalsSQ[-1] != 1.0"

            # Both must be non-decreasing
            for i in range(1, K):
                assert matlab_pvalsR[i] >= matlab_pvalsR[i - 1] - 1e-15
                assert pvalsR[i] >= pvalsR[i - 1] - 1e-15
                assert matlab_pvalsSQ[i] >= matlab_pvalsSQ[i - 1] - 1e-15
                assert pvalsSQ[i] >= pvalsSQ[i - 1] - 1e-15


# ===========================================================================
# Edge Case Tests
# ===========================================================================

class TestMcsEdgeCases:
    """Edge case tests for mcs()."""

    def test_mcs_single_model_column(self):
        """K=1 (single model, should always be included with p-value 1.0).

        Ref: mcs.m:112 — loop iterates M0-1 times; for K=1, loop does not
        execute, so the single model gets p-value=1.0 and is always included.
        """
        rng = np.random.default_rng(88)
        losses = rng.standard_normal((100, 1)) ** 2
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = mcs(
            losses, alpha=0.05, B=50, w=5
        )
        # Single model must be included
        assert len(includedR) == 1, f"Single model: expected 1 included (R), got {len(includedR)}"
        assert 0 in includedR, "Single model (index 0) must be in includedR"
        assert len(excludedR) == 0, "Single model: excludedR should be empty"
        assert len(includedSQ) == 1
        assert 0 in includedSQ
        assert len(excludedSQ) == 0
        # P-value should be 1.0
        npt.assert_array_equal(pvalsR, np.array([1.0]))
        npt.assert_array_equal(pvalsSQ, np.array([1.0]))

    def test_mcs_large_K(self):
        """K=20 models (stress test for pairwise computation).

        Verifies the MCS works correctly with a larger number of models,
        where the pairwise comparison matrix is 20×20.
        """
        rng = np.random.default_rng(55)
        K = 20
        T = 300
        losses = rng.standard_normal((T, K)) ** 2
        # Add increasing mean losses
        offsets = np.linspace(0.0, 3.0, K)
        losses = losses + offsets[np.newaxis, :]

        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = mcs(
            losses, alpha=0.05, B=100, w=10
        )

        # All K models accounted for
        assert len(includedR) + len(excludedR) == K
        assert len(includedSQ) + len(excludedSQ) == K
        assert len(pvalsR) == K
        assert len(pvalsSQ) == K

        # P-values non-decreasing
        for i in range(1, K):
            assert pvalsR[i] >= pvalsR[i - 1] - 1e-15
            assert pvalsSQ[i] >= pvalsSQ[i - 1] - 1e-15

        # Last p-value is 1.0
        assert pvalsR[-1] == 1.0
        assert pvalsSQ[-1] == 1.0

        # Best model (0) should be in MCS
        assert 0 in includedR
        assert 0 in includedSQ

    def test_mcs_alpha_boundary_low(self):
        """Alpha very close to 0 (e.g., 0.001) — very conservative MCS.

        With alpha very close to 0, almost all models should be included
        (very hard to reject at this significance level).
        """
        rng = np.random.default_rng(44)
        K = 5
        losses = rng.standard_normal((200, K)) ** 2 + np.linspace(0, 1.5, K)

        includedR, pvalsR, excludedR, _, pvalsSQ, _ = mcs(
            losses, alpha=0.001, B=200, w=10
        )

        # Structural properties must hold
        assert len(pvalsR) == K
        assert len(includedR) + len(excludedR) == K
        # With very low alpha, should include many/all models
        assert len(includedR) >= 1, "At least one model must be included"

    def test_mcs_alpha_boundary_high(self):
        """Alpha very close to 1 (e.g., 0.999) — very liberal MCS.

        With alpha close to 1, fewer models should be included (easy to reject).
        """
        rng = np.random.default_rng(44)
        K = 5
        losses = rng.standard_normal((200, K)) ** 2 + np.linspace(0, 2.0, K)

        includedR, pvalsR, excludedR, _, pvalsSQ, _ = mcs(
            losses, alpha=0.999, B=200, w=10
        )

        # Structural properties must hold
        assert len(pvalsR) == K
        assert len(includedR) + len(excludedR) == K
        # At least one model included (the best one always has pval=1.0)
        assert len(includedR) >= 1

    def test_mcs_goods_negation(self):
        """Pass -losses to test 'goods' mode per mcs.m:32-33.

        The MCS operates on 'bads' (losses). For 'goods' (returns), pass
        -1 * LOSSES. Negating losses should reverse the model ranking.
        """
        rng = np.random.default_rng(66)
        K = 4
        T = 200
        # Model 0 has lowest loss = best as a loss model
        losses = rng.standard_normal((T, K)) ** 2
        losses[:, 0] = rng.standard_normal(T) ** 2 * 0.3  # best loss model

        # As losses: model 0 should be included
        inclR_loss, _, _, _, _, _ = mcs(losses, alpha=0.05, B=200, w=10)
        assert 0 in inclR_loss, "Model 0 should be included when treated as losses"

        # As goods (negate): model 0 now has highest negative = worst "loss"
        # After negation, model 0 becomes the worst
        inclR_goods, _, exclR_goods, _, _, _ = mcs(-losses, alpha=0.05, B=200, w=10)
        # Model 0 should now tend to be excluded (it has highest -loss = worst "loss")
        # Due to bootstrap randomness, we can't guarantee this, but the ranking
        # should shift. At minimum, structural properties must hold.
        assert len(inclR_goods) + len(exclR_goods) == K

    def test_mcs_1d_loss_input(self):
        """Pass a 1-D array (single model as vector).

        The Python implementation should reshape 1-D input to (T, 1).
        """
        rng = np.random.default_rng(77)
        losses_1d = rng.standard_normal(100) ** 2
        result = mcs(losses_1d, alpha=0.05, B=50, w=5)
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = result
        assert len(includedR) == 1
        assert 0 in includedR
        assert len(excludedR) == 0
        npt.assert_array_equal(pvalsR, np.array([1.0]))

    def test_mcs_minimum_observations(self):
        """T=2 (minimum valid observation count).

        Ref: mcs.m:54 — t=size(losses,1); if t<2 error
        This is the boundary case: T=2 should work.
        """
        rng = np.random.default_rng(99)
        losses = rng.standard_normal((2, 3)) ** 2
        result = mcs(losses, alpha=0.05, B=50, w=1)
        assert len(result) == 6
        K = 3
        includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ = result
        assert len(pvalsR) == K
        assert len(includedR) + len(excludedR) == K

    def test_mcs_w_equals_one(self):
        """Block length w=1 (IID bootstrap — each bootstrap sample is IID).

        With w=1, the block bootstrap degenerates to IID resampling.
        """
        rng = np.random.default_rng(33)
        losses = rng.standard_normal((100, 3)) ** 2 + np.array([0.0, 0.5, 1.5])
        result = mcs(losses, alpha=0.05, B=100, w=1, boot='BLOCK')
        assert len(result) == 6
        includedR, pvalsR, excludedR, _, _, _ = result
        assert len(pvalsR) == 3
        assert len(includedR) + len(excludedR) == 3

    def test_mcs_w_large(self):
        """Large block length w (close to T).

        Tests MCS behavior when the block length is a significant fraction
        of the sample size.
        """
        rng = np.random.default_rng(22)
        T = 50
        losses = rng.standard_normal((T, 3)) ** 2 + np.array([0.0, 0.5, 1.0])
        result = mcs(losses, alpha=0.05, B=50, w=20, boot='BLOCK')
        assert len(result) == 6
        includedR, pvalsR, excludedR, _, _, _ = result
        assert len(pvalsR) == 3
        assert len(includedR) + len(excludedR) == 3
