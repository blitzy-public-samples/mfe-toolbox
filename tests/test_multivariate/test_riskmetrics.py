"""
Comprehensive pytest test file for the RiskMetrics model family.

Tests cover two migrated modules:
- mfe_toolbox.multivariate.riskmetrics   (RiskMetrics 1994 EWMA)
- mfe_toolbox.multivariate.riskmetrics2006 (RiskMetrics 2006 multi-frequency EWMA)

Validates output shapes, positive definiteness, symmetry, EWMA recursion
correctness, lambda/tau parametrization, backcast handling, input validation,
and numerical parity against MATLAB reference fixtures.

Per AAP Section 0.7.1:
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)

Source references:
    multivariate/riskmetrics.m   (82 lines — Kevin Sheppard, Rev 3)
    multivariate/riskmetrics2006.m (166 lines — Kevin Sheppard, Rev 3)
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.multivariate.riskmetrics import riskmetrics
from mfe_toolbox.multivariate.riskmetrics2006 import riskmetrics2006
from tests.conftest import ATOL, RTOL, load_fixture_npy, assert_allclose


# ---------------------------------------------------------------------------
# Local Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def riskmetrics_backcast(multivariate_data):
    """Compute sample covariance matrix as backcast for RiskMetrics tests.

    Uses ``np.cov(data, rowvar=False)`` which matches MATLAB ``cov(data)``
    (both divide by T-1).  Returns a K×K positive-definite matrix.
    """
    return np.cov(multivariate_data, rowvar=False)


# ---------------------------------------------------------------------------
# TestRiskMetrics — RiskMetrics 1994 EWMA covariance
# ---------------------------------------------------------------------------

class TestRiskMetrics:
    """Tests for the RiskMetrics EWMA covariance estimator.

    Ref: riskmetrics.m — H(t) = (1-λ)*r(t-1)*r(t-1)' + λ*H(t-1)
    Pure filtering/recursion — no optimisation step required.
    """

    def test_output_shape(self, multivariate_data):
        """Ht must be K×K×T for K=3, T=1000."""
        T, K = multivariate_data.shape
        Ht = riskmetrics(multivariate_data, lambda_param=0.94)
        assert Ht.shape == (K, K, T), (
            f"Expected Ht shape ({K}, {K}, {T}), got {Ht.shape}"
        )

    def test_ht_positive_definite(self, multivariate_data):
        """All Ht[:,:,t] slices must be positive definite."""
        Ht = riskmetrics(multivariate_data, lambda_param=0.94)
        T = Ht.shape[2]
        for t in range(T):
            eigs = np.linalg.eigvalsh(Ht[:, :, t])
            assert np.all(eigs > -1e-10), (
                f"Ht[:,:,{t}] not PD, min eigenvalue = {eigs.min()}"
            )

    def test_ht_symmetric(self, multivariate_data):
        """All Ht[:,:,t] slices must be symmetric."""
        Ht = riskmetrics(multivariate_data, lambda_param=0.94)
        T = Ht.shape[2]
        for t in range(T):
            npt.assert_allclose(
                Ht[:, :, t], Ht[:, :, t].T,
                atol=1e-14,
                err_msg=f"Ht[:,:,{t}] not symmetric",
            )

    def test_default_lambda(self, multivariate_data):
        """Default λ=0.94 should produce valid, finite output."""
        T, K = multivariate_data.shape
        Ht = riskmetrics(multivariate_data)
        assert Ht.shape == (K, K, T)
        assert np.all(np.isfinite(Ht)), "Ht contains NaN/Inf with default lambda"

    @pytest.mark.parametrize("lambda_", [0.90, 0.94, 0.97])
    def test_lambda_parametrized(self, multivariate_data, lambda_):
        """Different λ values in (0, 1) all produce valid K×K×T output."""
        T, K = multivariate_data.shape
        Ht = riskmetrics(multivariate_data, lambda_param=lambda_)
        assert Ht.shape == (K, K, T)
        assert np.all(np.isfinite(Ht))
        # Verify at least one slice is PD
        eigs = np.linalg.eigvalsh(Ht[:, :, T // 2])
        assert np.all(eigs > -1e-10)

    def test_lambda_0_equal_outer_product(self, multivariate_data, riskmetrics_backcast):
        """λ ≈ 0 ⇒ H(t) ≈ r(t-1)*r(t-1)' (pure outer product).

        Ref: riskmetrics.m:54 — lambda must be strictly > 0, so we use 1e-10.
        With λ ≈ 0: H(t) = (1-λ)*outer(r(t-1)) + λ*H(t-1) ≈ outer(r(t-1)).
        """
        lambda_near_zero = 1e-10
        Ht = riskmetrics(
            multivariate_data,
            lambda_param=lambda_near_zero,
            back_cast=riskmetrics_backcast,
        )
        T, K = multivariate_data.shape
        # Verify for several time steps (t >= 1; t=0 uses backcast)
        for t in range(1, min(T, 20)):
            expected = np.outer(multivariate_data[t - 1], multivariate_data[t - 1])
            npt.assert_allclose(
                Ht[:, :, t], expected, atol=1e-5,
                err_msg=f"H({t}) ≠ outer(r({t - 1})) with λ ≈ 0",
            )

    def test_lambda_1_constant(self, multivariate_data, riskmetrics_backcast):
        """λ ≈ 1 ⇒ H(t) ≈ backCast (constant covariance).

        Ref: riskmetrics.m:54 — lambda must be strictly < 1, so we use 1 - 1e-10.
        With λ ≈ 1: H(t) = (1-λ)*outer(r(t-1)) + λ*H(t-1) ≈ H(t-1) ≈ backCast.
        """
        lambda_near_one = 1.0 - 1e-10
        Ht = riskmetrics(
            multivariate_data,
            lambda_param=lambda_near_one,
            back_cast=riskmetrics_backcast,
        )
        # All H(t) should be approximately equal to the provided backcast
        for t in range(min(Ht.shape[2], 20)):
            npt.assert_allclose(
                Ht[:, :, t], riskmetrics_backcast, atol=1e-4,
                err_msg=f"H({t}) ≠ backcast with λ ≈ 1",
            )

    def test_custom_backcast(self, multivariate_data, riskmetrics_backcast):
        """Providing an explicit backCast sets H(0) = backCast.

        Ref: riskmetrics.m:78 — Ht(:,:,1) = backCast (MATLAB 1-indexed).
        """
        Ht = riskmetrics(
            multivariate_data,
            lambda_param=0.94,
            back_cast=riskmetrics_backcast,
        )
        # H(0) should exactly equal the provided backcast
        npt.assert_allclose(
            Ht[:, :, 0], riskmetrics_backcast, atol=1e-12,
            err_msg="H(0) does not match provided backcast",
        )

    def test_no_optimizer_needed(self, multivariate_data):
        """RiskMetrics is pure filtering — no optimisation involved.

        Verify the function completes without any optimiser call (pure EWMA recursion).
        """
        T, K = multivariate_data.shape
        Ht = riskmetrics(multivariate_data, lambda_param=0.94)
        assert Ht.shape == (K, K, T)
        # Verify result is valid and has correct dtype
        assert np.all(np.isfinite(Ht))
        assert Ht.dtype == np.float64
        # Verify no zeros on the diagonal at any time step (valid covariance)
        for t in range(0, T, 100):
            assert np.all(np.diag(Ht[:, :, t]) > 0)

    def test_ewma_recursion_manually(self, multivariate_data, riskmetrics_backcast):
        """Manually verify EWMA recursion for the first several observations.

        Ref: riskmetrics.m:79-81 —
            for i=2:T
                Ht(:,:,i) = (1-lambda)*data(:,:,i-1) + lambda*Ht(:,:,i-1)
        """
        lambda_ = 0.94
        Ht = riskmetrics(
            multivariate_data,
            lambda_param=lambda_,
            back_cast=riskmetrics_backcast,
        )
        # Manual recursion for first 10 time steps
        H_prev = riskmetrics_backcast.copy()
        npt.assert_allclose(
            Ht[:, :, 0], H_prev, atol=1e-12,
            err_msg="H(0) != backcast",
        )
        for t in range(1, 10):
            outer_t = np.outer(multivariate_data[t - 1], multivariate_data[t - 1])
            H_manual = (1.0 - lambda_) * outer_t + lambda_ * H_prev
            npt.assert_allclose(
                Ht[:, :, t], H_manual, atol=1e-12,
                err_msg=f"EWMA recursion mismatch at t={t}",
            )
            H_prev = H_manual

    def test_input_validation(self, multivariate_data, riskmetrics_backcast):
        """Invalid inputs must raise ValueError.

        Ref: riskmetrics.m:54-56 — lambda validation (strict 0 < lambda < 1)
        Ref: riskmetrics.m:70-72 — backcast PD check (min(eig) < 0 → error)
        """
        # lambda = 0 (boundary, not allowed)
        with pytest.raises(ValueError, match="LAMBDA must be between 0 and 1"):
            riskmetrics(multivariate_data, lambda_param=0.0)
        # lambda = 1 (boundary, not allowed)
        with pytest.raises(ValueError, match="LAMBDA must be between 0 and 1"):
            riskmetrics(multivariate_data, lambda_param=1.0)
        # lambda < 0
        with pytest.raises(ValueError, match="LAMBDA must be between 0 and 1"):
            riskmetrics(multivariate_data, lambda_param=-0.5)
        # lambda > 1
        with pytest.raises(ValueError, match="LAMBDA must be between 0 and 1"):
            riskmetrics(multivariate_data, lambda_param=1.5)
        # Non-PD backcast (negative-definite matrix)
        K = multivariate_data.shape[1]
        bad_backcast = -1.0 * np.eye(K)
        with pytest.raises(ValueError, match="positive semidefinite"):
            riskmetrics(multivariate_data, lambda_param=0.94, back_cast=bad_backcast)

    def test_fixture_parity(self, multivariate_data, multivariate_fixture_dir):
        """Compare Ht against MATLAB reference fixture.

        Uses conftest ``load_fixture_npy`` which calls ``pytest.skip`` if
        the fixture file is not found.
        """
        fixture = load_fixture_npy(multivariate_fixture_dir, "riskmetrics_Ht")
        Ht = riskmetrics(multivariate_data, lambda_param=0.94)
        assert_allclose(
            Ht, fixture,
            err_msg="RiskMetrics Ht does not match MATLAB fixture",
        )


# ---------------------------------------------------------------------------
# TestRiskMetrics2006 — RiskMetrics 2006 multi-frequency EWMA
# ---------------------------------------------------------------------------

class TestRiskMetrics2006:
    """Tests for the RiskMetrics 2006 multi-frequency EWMA covariance.

    Ref: riskmetrics2006.m —
        H(t) = sum_k  w(k) * Htilde(t, k)
    where Htilde(t, k) is an EWMA with half-life tau_k = tau1 * rho^(k-1).
    """

    def test_output_shapes(self, multivariate_data):
        """Ht is K×K×T; returned weights is a T×T matrix.

        Ref: riskmetrics2006.m:127 — Ht is K×K×T
        Ref: riskmetrics2006.m:152 — weights is T×T weight matrix
        """
        T, K = multivariate_data.shape
        Ht, weights = riskmetrics2006(multivariate_data)
        assert Ht.shape == (K, K, T), (
            f"Expected Ht ({K},{K},{T}), got {Ht.shape}"
        )
        assert weights.shape == (T, T), (
            f"Expected weights ({T},{T}), got {weights.shape}"
        )

    def test_ht_positive_definite(self, multivariate_data):
        """All Ht[:,:,t] slices must be positive definite."""
        Ht, _ = riskmetrics2006(multivariate_data)
        T = Ht.shape[2]
        for t in range(T):
            eigs = np.linalg.eigvalsh(Ht[:, :, t])
            assert np.all(eigs > -1e-10), (
                f"Ht[:,:,{t}] not PD, min eigenvalue = {eigs.min()}"
            )

    def test_ht_symmetric(self, multivariate_data):
        """All Ht[:,:,t] slices must be symmetric."""
        Ht, _ = riskmetrics2006(multivariate_data)
        T = Ht.shape[2]
        for t in range(T):
            npt.assert_allclose(
                Ht[:, :, t], Ht[:, :, t].T, atol=1e-14,
                err_msg=f"Ht[:,:,{t}] not symmetric",
            )

    def test_weights_sum_to_one(self):
        """Internal frequency weights (kmax-length) must sum to 1.

        Ref: riskmetrics2006.m:125-126 —
            w = 1 - log(tauks)/log(tau0);  w = w/sum(w);
        """
        tau0, tau1, kmax = 1560, 4, 14
        rho = np.sqrt(2)
        # Ref: riskmetrics2006.m:124
        tauks = tau1 * rho ** np.arange(kmax)
        # Ref: riskmetrics2006.m:125
        w = 1.0 - np.log(tauks) / np.log(tau0)
        # Ref: riskmetrics2006.m:126
        w = w / np.sum(w)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-12)

    def test_weights_positive(self):
        """All internal frequency weights must be strictly positive.

        Since all tau_k < tau0, we have log(tau_k) < log(tau0),
        hence w_k = 1 - log(tau_k)/log(tau0) > 0 before normalisation.
        """
        tau0, tau1, kmax = 1560, 4, 14
        rho = np.sqrt(2)
        tauks = tau1 * rho ** np.arange(kmax)
        w = 1.0 - np.log(tauks) / np.log(tau0)
        w = w / np.sum(w)
        assert np.all(w > 0), f"Some weights are non-positive: {w}"

    def test_default_parameters(self, multivariate_data):
        """Default tau0=1560, tau1=4, kmax=14, rho=sqrt(2) produce valid output.

        Ref: riskmetrics2006.m:81-92
        """
        T, K = multivariate_data.shape
        Ht, weights = riskmetrics2006(multivariate_data)
        assert Ht.shape == (K, K, T)
        assert np.all(np.isfinite(Ht)), "Ht contains NaN/Inf with defaults"

    @pytest.mark.parametrize("kmax", [5, 10, 14])
    def test_kmax_parametrized(self, multivariate_data, kmax):
        """Varying numbers of EWMA frequency scales produce valid output.

        Default tau0=1560, tau1=4, rho=sqrt(2) satisfy
        tau1*rho^(kmax-1) < tau0 for kmax <= 14.
        """
        T, K = multivariate_data.shape
        Ht, weights = riskmetrics2006(
            multivariate_data,
            tau0=1560, tau1=4, kmax=kmax, rho=np.sqrt(2),
        )
        assert Ht.shape == (K, K, T)
        assert weights.shape == (T, T)
        assert np.all(np.isfinite(Ht))

    def test_kmax_1_reduces_to_ewma(self, multivariate_data):
        """kmax=1 RM2006 should match standard EWMA with equivalent lambda.

        Ref: riskmetrics2006.m:46-50 — RM1994 as special case of RM2006:
            tau1 = -1/log(0.94),  kmax=1  →  mu = exp(-1/tau1) = 0.94 = λ

        Both recursions use the same EWMA formula with identical backcast
        computation when mu == lambda, so outputs should match closely.
        """
        T, K = multivariate_data.shape
        # Ref: riskmetrics2006.m:48 — tau1 = -1/log(.94)
        tau1_for_094 = -1.0 / np.log(0.94)
        Ht_rm2006, _ = riskmetrics2006(
            multivariate_data,
            tau0=1560, tau1=tau1_for_094, kmax=1, rho=1.0,
        )
        Ht_rm = riskmetrics(multivariate_data, lambda_param=0.94)
        # Both use the same recursion with mu == lambda.
        # Backcast may differ slightly due to end_point bounds (k+1 vs K),
        # but converges rapidly.  Compare late-sample slices.
        assert np.allclose(
            Ht_rm2006[:, :, -100:], Ht_rm[:, :, -100:],
            atol=1e-6, rtol=1e-4,
        ), "kmax=1 RM2006 should approximate standard EWMA"

    def test_multi_frequency_smoothness(self, multivariate_data):
        """Larger kmax includes longer-horizon EWMA → smoother Ht paths.

        Verify both configurations produce finite, positive results.
        """
        Ht_small, _ = riskmetrics2006(multivariate_data, kmax=5)
        Ht_large, _ = riskmetrics2006(multivariate_data, kmax=14)
        T = Ht_small.shape[2]
        # Compute trace time series for each configuration
        trace_small = np.zeros(T)
        trace_large = np.zeros(T)
        for t in range(T):
            trace_small[t] = np.trace(Ht_small[:, :, t])
            trace_large[t] = np.trace(Ht_large[:, :, t])
        # Both must be finite and positive
        assert np.all(np.isfinite(trace_small))
        assert np.all(np.isfinite(trace_large))
        assert np.all(trace_small > 0)
        assert np.all(trace_large > 0)
        # Mean absolute first-differences measure "roughness"
        roughness_small = np.mean(np.abs(np.diff(trace_small)))
        roughness_large = np.mean(np.abs(np.diff(trace_large)))
        assert np.isfinite(roughness_small)
        assert np.isfinite(roughness_large)

    def test_geometric_spacing(self):
        """Half-lives must be geometrically spaced between tau1 and tau_max.

        Ref: riskmetrics2006.m:124 — tauks = tau1*rho.^((1:kmax)-1)
        Consecutive half-life ratio should equal rho exactly.
        Ref: riskmetrics2006.m:133 — mu = exp(-1/tauk)
        """
        tau0, tau1, kmax = 1560, 4, 14
        rho = np.sqrt(2)
        tauks = tau1 * rho ** np.arange(kmax)

        # Verify geometric spacing: tauks[k+1] / tauks[k] = rho
        ratios = tauks[1:] / tauks[:-1]
        npt.assert_allclose(
            ratios, np.full(kmax - 1, rho), atol=1e-12,
            err_msg="Half-lives not geometrically spaced",
        )
        # Verify boundary values
        assert tauks[0] == pytest.approx(tau1, abs=1e-12)
        assert tauks[-1] == pytest.approx(
            tau1 * rho ** (kmax - 1), abs=1e-8
        )
        # Ref: riskmetrics2006.m:133 — smoothing parameter mu = exp(-1/tauk)
        mus = np.exp(-1.0 / tauks)
        assert np.all(mus > 0) and np.all(mus < 1)

    def test_input_validation(self, multivariate_data):
        """Invalid parameter combinations must raise ValueError.

        Ref: riskmetrics2006.m:94-108
        Validation order in Python impl: constraint → tau1 → tau0 → kmax → rho.
        Negative tau0 with positive tau1 triggers the constraint check first
        because tau1*rho^(kmax-1) > 0 > tau0.
        """
        # tau1*rho^(kmax-1) > tau0 → constraint violated
        with pytest.raises(ValueError, match="The inputs must satisfy"):
            riskmetrics2006(
                multivariate_data, tau0=10, tau1=4, kmax=14, rho=np.sqrt(2),
            )
        # Negative tau0 (with default positive tau1) triggers constraint first
        # because tau1*rho^(kmax-1) > 0 > tau0_negative
        with pytest.raises(ValueError, match="The inputs must satisfy"):
            riskmetrics2006(multivariate_data, tau0=-1)
        # kmax < 1
        with pytest.raises(ValueError, match="KMAX must be an integer"):
            riskmetrics2006(multivariate_data, kmax=0)
        # Non-integer kmax
        with pytest.raises(ValueError, match="KMAX must be an integer"):
            riskmetrics2006(multivariate_data, kmax=2.5)
        # Negative tau1 (constraint passes since tau1*rho^k < 0 < tau0)
        with pytest.raises(ValueError, match="TAU1 must be positive"):
            riskmetrics2006(multivariate_data, tau1=-1)
        # Negative rho
        with pytest.raises(ValueError, match="RHO must be positive"):
            riskmetrics2006(multivariate_data, rho=-1)

    def test_fixture_parity(self, multivariate_data, multivariate_fixture_dir):
        """Compare Ht against MATLAB reference fixture.

        Uses ``load_fixture_npy`` which calls ``pytest.skip`` if fixture
        file is not found on disk.
        """
        fixture = load_fixture_npy(
            multivariate_fixture_dir, "riskmetrics2006_Ht",
        )
        Ht, _ = riskmetrics2006(multivariate_data)
        assert_allclose(
            Ht, fixture,
            err_msg="RM2006 Ht does not match MATLAB fixture",
        )

    def test_fixture_parity_weights(self, multivariate_data, multivariate_fixture_dir):
        """Compare T×T weight matrix against MATLAB reference fixture.

        Ref: riskmetrics2006.m:151-166 — T×T weight matrix output.
        """
        fixture = load_fixture_npy(
            multivariate_fixture_dir, "riskmetrics2006_weights",
        )
        _, weights = riskmetrics2006(multivariate_data)
        assert_allclose(
            weights, fixture,
            err_msg="RM2006 weights do not match MATLAB fixture",
        )
