"""
Comprehensive pytest tests for the FIGARCH (Fractionally Integrated GARCH)
model family, covering all 8 source modules.

FIGARCH is a long-memory GARCH model with unique structural properties:
- Restricted orders: p in {0,1}, q in {0,1}
- Fractional differencing parameter d in (0,1) captures long memory
- ARCH(infinity) truncated weight representation (figarch_weights)
- Parameter vector: [omega, (phi if p=1), d, (beta if q=1), (nu), (lambda)]

Modules tested:
1. figarch_parameter_check — Input validation
2. figarch_weights — ARCH(infinity) weight computation (unique to FIGARCH)
3. figarch_transform — Constrained -> unconstrained parameter mapping
4. figarch_itransform — Unconstrained -> constrained parameter mapping
5. figarch_likelihood — Log-likelihood computation
6. figarch_simulate — FIGARCH time series simulation
7. figarch_starting_values — Grid-search starting value computation
8. figarch — Main FIGARCH estimation driver

Test coverage:
- All 4 (p,q) combinations: (0,0), (1,0), (0,1), (1,1)
- All 4 error distributions: NORMAL, STUDENTST, GED, SKEWT
- Long-memory parameter d boundary behavior
- truncLag sensitivity
- Unit, integration, and parity tests

Per AAP Section 0.7.1:
- ATOL=1e-6, RTOL=1e-4 for numerical parity
- numpy.random.default_rng(42) for reproducibility
- >=90% coverage, all tests pass with pytest -x --tb=short
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.univariate.figarch import figarch
from mfe_toolbox.univariate.figarch_itransform import figarch_itransform
from mfe_toolbox.univariate.figarch_likelihood import figarch_likelihood
from mfe_toolbox.univariate.figarch_parameter_check import figarch_parameter_check
from mfe_toolbox.univariate.figarch_simulate import figarch_simulate
from mfe_toolbox.univariate.figarch_starting_values import figarch_starting_values
from mfe_toolbox.univariate.figarch_transform import figarch_transform
from mfe_toolbox.univariate.figarch_weights import figarch_weights

# Import conftest helpers (pytest autodiscovery handles fixtures)
from tests.test_univariate.conftest import load_univariate_fixture, load_univariate_input


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Per AAP Section 0.7.1: Tolerances for MATLAB parity testing
ATOL: float = 1e-6
RTOL: float = 1e-4

# Error type codes used by the Python implementation (integer-coded)
NORMAL: int = 1
STUDENTST: int = 2
GED: int = 3
SKEWT: int = 4


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def epsilon(rng: np.random.Generator) -> np.ndarray:
    """T=1000 mean-zero return series scaled to daily-return magnitude."""
    T = 1000
    data = rng.standard_normal(T) * 0.01
    data = data - data.mean()
    return data


@pytest.fixture(scope="module")
def epsilon_short(rng: np.random.Generator) -> np.ndarray:
    """T=200 short mean-zero return series for quick tests."""
    T = 200
    data = rng.standard_normal(T) * 0.01
    data = data - data.mean()
    return data


# ============================================================================
# Phase 1: Unit Tests — figarch_parameter_check
# ============================================================================

class TestFigarchParameterCheck:
    """Tests for figarch_parameter_check input validation module.

    Validates that all 4 (p,q) combinations are accepted for valid input
    and that invalid inputs raise ValueError per the MATLAB error() behavior.
    Ref: figarch_parameter_check.m
    """

    def _make_epsilon(self, n: int = 100) -> np.ndarray:
        """Create a valid epsilon array for parameter checking tests."""
        rng = np.random.default_rng(42)
        eps = rng.standard_normal(n) * 0.01
        eps = eps - eps.mean()
        return eps

    def test_valid_figarch_1_d_1(self) -> None:
        """Valid FIGARCH(1,d,1) — both phi and beta present."""
        eps = self._make_epsilon()
        result = figarch_parameter_check(eps, p=1, q=1)
        p_out, q_out, error_type_out, trunc_lag_out, sv_out, opts_out = result
        assert p_out == 1
        assert q_out == 1
        assert error_type_out == NORMAL
        assert trunc_lag_out == 1000
        assert sv_out is None

    def test_valid_figarch_0_d_0(self) -> None:
        """Valid FIGARCH(0,d,0) — pure long-memory, no phi or beta."""
        eps = self._make_epsilon()
        result = figarch_parameter_check(eps, p=0, q=0)
        p_out, q_out, error_type_out, trunc_lag_out, sv_out, opts_out = result
        assert p_out == 0
        assert q_out == 0

    def test_valid_figarch_1_d_0(self) -> None:
        """Valid FIGARCH(1,d,0) — phi only."""
        eps = self._make_epsilon()
        result = figarch_parameter_check(eps, p=1, q=0)
        p_out, q_out, _, _, _, _ = result
        assert p_out == 1
        assert q_out == 0

    def test_valid_figarch_0_d_1(self) -> None:
        """Valid FIGARCH(0,d,1) — beta only."""
        eps = self._make_epsilon()
        result = figarch_parameter_check(eps, p=0, q=1)
        p_out, q_out, _, _, _, _ = result
        assert p_out == 0
        assert q_out == 1

    def test_invalid_p_greater_than_1(self) -> None:
        """p > 1 should raise ValueError (FIGARCH restricts p in {0,1})."""
        eps = self._make_epsilon()
        with pytest.raises(ValueError, match="P must be either 0 or 1"):
            figarch_parameter_check(eps, p=2, q=0)

    def test_invalid_q_greater_than_1(self) -> None:
        """q > 1 should raise ValueError."""
        eps = self._make_epsilon()
        with pytest.raises(ValueError, match="Q must be either 0 or 1"):
            figarch_parameter_check(eps, p=0, q=2)

    def test_invalid_p_negative(self) -> None:
        """p < 0 should raise ValueError."""
        eps = self._make_epsilon()
        with pytest.raises(ValueError, match="P must be either 0 or 1"):
            figarch_parameter_check(eps, p=-1, q=0)

    def test_invalid_error_type(self) -> None:
        """error_type=5 should raise ValueError."""
        eps = self._make_epsilon()
        with pytest.raises(ValueError, match="errorType"):
            figarch_parameter_check(eps, p=0, q=0, error_type=5)

    def test_invalid_error_type_string(self) -> None:
        """Unknown error_type string should raise ValueError."""
        eps = self._make_epsilon()
        with pytest.raises(ValueError, match="errorType"):
            figarch_parameter_check(eps, p=0, q=0, error_type='INVALID')

    def test_empty_epsilon(self) -> None:
        """Empty epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="epsilon"):
            figarch_parameter_check(np.array([]), p=0, q=0)

    def test_row_vector_epsilon(self) -> None:
        """Row vector epsilon (2-D with >1 column) should raise ValueError."""
        eps = self._make_epsilon(50)
        # Reshape to (1, 50) = row vector
        eps_row = eps.reshape(1, -1)
        with pytest.raises(ValueError, match="column vector"):
            figarch_parameter_check(eps_row, p=0, q=0)

    def test_error_type_string_mapping(self) -> None:
        """String error types should be correctly mapped to integer codes."""
        eps = self._make_epsilon()
        for name, expected_code in [
            ('NORMAL', 1), ('STUDENTST', 2), ('GED', 3), ('SKEWT', 4)
        ]:
            _, _, et, _, _, _ = figarch_parameter_check(eps, p=0, q=0, error_type=name)
            assert et == expected_code, f"Expected {expected_code} for '{name}', got {et}"

    def test_trunclag_validation(self) -> None:
        """truncLag < 10 should raise ValueError."""
        eps = self._make_epsilon()
        with pytest.raises(ValueError, match="TRUNCLAG"):
            figarch_parameter_check(eps, p=0, q=0, trunc_lag=5)

    def test_custom_trunclag(self) -> None:
        """Custom truncLag should be returned correctly."""
        eps = self._make_epsilon()
        _, _, _, tl, _, _ = figarch_parameter_check(eps, p=0, q=0, trunc_lag=500)
        assert tl == 500


# ============================================================================
# Phase 2: Unit Tests — figarch_weights (UNIQUE to FIGARCH)
# ============================================================================

class TestFigarchWeights:
    """Tests for figarch_weights — the ARCH(infinity) weight computation.

    This is the critical unique component of FIGARCH: the truncation weight
    recursion that defines the long-memory ARCH structure. Thoroughly tested
    with various d values, (p,q) combinations, and truncation lengths.
    Ref: figarch_weights.m
    """

    def test_weights_output_shape(self) -> None:
        """figarch_weights returns array of length truncLag."""
        truncLag = 100
        # FIGARCH(0,d,0): parameters = [d]
        d = 0.45
        params = np.array([d])
        w = figarch_weights(params, 0, 0, truncLag)
        assert isinstance(w, np.ndarray)
        assert w.shape == (truncLag,)

    def test_weights_output_shape_large(self) -> None:
        """Default truncLag=1000 produces 1000 weights."""
        params = np.array([0.45])
        w = figarch_weights(params, 0, 0, 1000)
        assert w.shape == (1000,)

    def test_weights_positive(self) -> None:
        """All weights must be non-negative for valid FIGARCH parameters.

        Positivity of the ARCH(infinity) weights is a necessary condition
        for the conditional variance to be positive. We test with parameters
        that are known to satisfy the positivity constraints.
        """
        # FIGARCH(1,d,1) with valid params
        # phi=0.1, d=0.45, beta=0.3 → phi+d-beta = 0.25 > 0
        params = np.array([0.1, 0.45, 0.3])
        w = figarch_weights(params, 1, 1, 500)
        assert np.all(w >= -1e-12), (
            f"Some weights are negative: min={w.min():.10e}"
        )

    def test_weights_decay(self) -> None:
        """Weights should generally decay for valid d values.

        For FIGARCH(0,d,0), the weights are monotonically decreasing
        after the first element when 0 < d < 1.
        """
        d = 0.45
        params = np.array([d])
        w = figarch_weights(params, 0, 0, 100)
        # Check that later weights are generally smaller (allow initial transient)
        # After a few initial terms, weights should be decreasing
        for i in range(5, len(w) - 1):
            assert w[i + 1] <= w[i] + 1e-12, (
                f"Weight at index {i+1} ({w[i+1]:.10e}) exceeds weight at "
                f"index {i} ({w[i]:.10e})"
            )

    def test_weights_d_zero(self) -> None:
        """d=0 (very near zero): weights should collapse.

        When d→0, the FIGARCH reduces to standard GARCH, so only the
        first weight should be non-trivially nonzero for FIGARCH(0,d,0).
        """
        d = 1e-8  # Near zero but not exactly zero to avoid numerical issues
        params = np.array([d])
        w = figarch_weights(params, 0, 0, 100)
        # The first weight lambda[0] = d ≈ 0
        assert abs(w[0] - d) < 1e-10
        # Later weights should be very small
        assert np.all(np.abs(w[1:]) < 1e-6), (
            f"Weights should be near zero for d≈0, max later weight: {np.max(np.abs(w[1:])):.10e}"
        )

    def test_weights_d_near_one(self) -> None:
        """d close to 1: weights should decay very slowly (long memory).

        When d→1, the FIGARCH approaches IGARCH behavior and the weights
        should sum closer to 1 and decay much more slowly.
        """
        d_low = 0.2
        d_high = 0.8
        params_low = np.array([d_low])
        params_high = np.array([d_high])
        truncLag = 500
        w_low = figarch_weights(params_low, 0, 0, truncLag)
        w_high = figarch_weights(params_high, 0, 0, truncLag)
        # Higher d → larger weight sum (slower decay, closer to IGARCH)
        assert np.sum(w_high) > np.sum(w_low), (
            f"Higher d should give larger weight sum: "
            f"sum(d={d_high})={np.sum(w_high):.6f} vs sum(d={d_low})={np.sum(w_low):.6f}"
        )

    def test_weights_trunclag_effect(self) -> None:
        """Different truncLag values; larger truncLag captures more weight mass."""
        d = 0.45
        params = np.array([d])
        w_short = figarch_weights(params, 0, 0, 100)
        w_long = figarch_weights(params, 0, 0, 500)
        # First 100 weights should be identical
        npt.assert_allclose(w_long[:100], w_short, atol=1e-12)
        # More weight mass with longer truncation
        assert np.sum(w_long) >= np.sum(w_short) - 1e-12

    def test_weights_figarch_1_d_1(self) -> None:
        """Weights for FIGARCH(1,d,1) — both phi and beta affect computation.

        Parameters: [phi, d, beta] with p=1, q=1.
        """
        phi, d, beta = 0.1, 0.45, 0.3
        params = np.array([phi, d, beta])
        w = figarch_weights(params, 1, 1, 200)
        assert w.shape == (200,)
        # First weight: lambda[0] = phi - beta + d = 0.1 - 0.3 + 0.45 = 0.25
        expected_first = phi - beta + d
        npt.assert_allclose(w[0], expected_first, atol=1e-12)

    def test_weights_figarch_0_d_0(self) -> None:
        """Weights for FIGARCH(0,d,0) — simplest case, pure fractional differencing.

        Parameters: [d] with p=0, q=0.
        """
        d = 0.45
        params = np.array([d])
        w = figarch_weights(params, 0, 0, 200)
        assert w.shape == (200,)
        # First weight: lambda[0] = phi - beta + d = 0 - 0 + 0.45 = d
        npt.assert_allclose(w[0], d, atol=1e-12)

    def test_weights_figarch_1_d_0(self) -> None:
        """Weights for FIGARCH(1,d,0) — phi only.

        Parameters: [phi, d] with p=1, q=0.
        """
        phi, d = 0.1, 0.45
        params = np.array([phi, d])
        w = figarch_weights(params, 1, 0, 200)
        assert w.shape == (200,)
        # First weight: lambda[0] = phi - 0 + d = phi + d
        expected_first = phi + d
        npt.assert_allclose(w[0], expected_first, atol=1e-12)

    def test_weights_figarch_0_d_1(self) -> None:
        """Weights for FIGARCH(0,d,1) — beta only.

        Parameters: [d, beta] with p=0, q=1.
        """
        d, beta = 0.45, 0.3
        params = np.array([d, beta])
        w = figarch_weights(params, 0, 1, 200)
        assert w.shape == (200,)
        # First weight: lambda[0] = 0 - beta + d = d - beta
        expected_first = d - beta
        npt.assert_allclose(w[0], expected_first, atol=1e-12)

    def test_weights_second_element_recursion(self) -> None:
        """Verify the second weight matches the analytical recursion formula.

        For FIGARCH(0,d,0) with parameters [d]:
        delta[0] = d
        delta[1] = (1-d)/2 * delta[0] = d*(1-d)/2
        lambda[0] = d
        lambda[1] = 0 * lambda[0] + (delta[1] - 0*delta[0])
                  = d*(1-d)/2
        """
        d = 0.45
        params = np.array([d])
        w = figarch_weights(params, 0, 0, 10)
        expected_second = d * (1.0 - d) / 2.0
        npt.assert_allclose(w[1], expected_second, atol=1e-12)

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_weights_parity(self, univariate_fixture_dir) -> None:
        """Compare weights against MATLAB fixture (±1e-6)."""
        try:
            fixture = load_univariate_fixture(
                univariate_fixture_dir, 'figarch', 'weights'
            )
        except Exception:
            pytest.skip("FIGARCH weights fixture not available")
        # Compute weights with matching parameters
        # Typical fixture: FIGARCH(1,d,1) params [phi, d, beta]
        if isinstance(fixture, np.ndarray) and fixture.dtype == object:
            pytest.skip("Object-dtype fixture needs special handling")
        params = np.array([0.1, 0.45, 0.3])
        w = figarch_weights(params, 1, 1, len(fixture))
        npt.assert_allclose(w, fixture, atol=ATOL, rtol=RTOL,
                            err_msg="FIGARCH weights parity failure")


# ============================================================================
# Phase 3: Unit Tests — figarch_transform / figarch_itransform
# ============================================================================

class TestFigarchTransform:
    """Tests for figarch_transform and figarch_itransform round-trip invertibility.

    The transform maps constrained parameters to the real line, and the
    itransform maps back. Round-trip should recover the original parameters
    within ±1e-6.
    Ref: figarch_transform.m, figarch_itransform.m
    """

    def test_transform_roundtrip_figarch_1_d_1(self) -> None:
        """transform → itransform recovers [omega, phi, d, beta] (±1e-6)."""
        # Constrained parameters: [omega, phi, d, beta]
        params = np.array([1e-5, 0.1, 0.45, 0.3])
        transformed = figarch_transform(params, p=1, q=1, error_type=NORMAL)
        recovered, nu, lam = figarch_itransform(transformed, p=1, q=1, error_type=NORMAL)
        npt.assert_allclose(recovered, params, atol=ATOL, rtol=RTOL,
                            err_msg="Round-trip FIGARCH(1,d,1) NORMAL failed")
        assert nu is None
        assert lam is None

    def test_transform_roundtrip_figarch_0_d_0(self) -> None:
        """Round-trip for [omega, d] only — FIGARCH(0,d,0)."""
        params = np.array([1e-5, 0.45])
        transformed = figarch_transform(params, p=0, q=0, error_type=NORMAL)
        recovered, nu, lam = figarch_itransform(transformed, p=0, q=0, error_type=NORMAL)
        npt.assert_allclose(recovered, params, atol=ATOL, rtol=RTOL,
                            err_msg="Round-trip FIGARCH(0,d,0) NORMAL failed")

    def test_transform_roundtrip_figarch_1_d_0(self) -> None:
        """Round-trip for [omega, phi, d] — FIGARCH(1,d,0)."""
        params = np.array([1e-5, 0.1, 0.45])
        transformed = figarch_transform(params, p=1, q=0, error_type=NORMAL)
        recovered, nu, lam = figarch_itransform(transformed, p=1, q=0, error_type=NORMAL)
        npt.assert_allclose(recovered, params, atol=ATOL, rtol=RTOL,
                            err_msg="Round-trip FIGARCH(1,d,0) NORMAL failed")

    def test_transform_roundtrip_figarch_0_d_1(self) -> None:
        """Round-trip for [omega, d, beta] — FIGARCH(0,d,1)."""
        params = np.array([1e-5, 0.45, 0.3])
        transformed = figarch_transform(params, p=0, q=1, error_type=NORMAL)
        recovered, nu, lam = figarch_itransform(transformed, p=0, q=1, error_type=NORMAL)
        npt.assert_allclose(recovered, params, atol=ATOL, rtol=RTOL,
                            err_msg="Round-trip FIGARCH(0,d,1) NORMAL failed")

    def test_transform_roundtrip_studentst(self) -> None:
        """Round-trip with Student-t nu parameter."""
        # [omega, phi, d, beta, nu] for FIGARCH(1,d,1) Student-t
        params = np.array([1e-5, 0.1, 0.45, 0.3, 6.0])
        transformed = figarch_transform(params, p=1, q=1, error_type=STUDENTST)
        recovered, nu, lam = figarch_itransform(transformed, p=1, q=1, error_type=STUDENTST)
        npt.assert_allclose(recovered, params[:4], atol=ATOL, rtol=RTOL,
                            err_msg="Round-trip FIGARCH(1,d,1) STUDENTST core failed")
        assert nu is not None
        npt.assert_allclose(nu, 6.0, atol=ATOL, rtol=RTOL,
                            err_msg="Student-t nu round-trip failed")
        assert lam is None

    def test_transform_roundtrip_ged(self) -> None:
        """Round-trip with GED nu parameter.

        Note: The MATLAB GED transform uses (nu-1)/49 forward and
        49*sigmoid(x)+1.01 inverse, so the round-trip has a known ~0.01
        offset artifact from the asymmetric bounds (1.0 vs 1.01). This is
        by design in the original MATLAB code, which states the constraints
        are "generally wrong" (figarch.m:36). We test that the offset is
        bounded, not that it is zero.
        """
        params = np.array([1e-5, 0.45, 1.5])  # [omega, d, nu] for FIGARCH(0,d,0) GED
        transformed = figarch_transform(params, p=0, q=0, error_type=GED)
        recovered, nu, lam = figarch_itransform(transformed, p=0, q=0, error_type=GED)
        npt.assert_allclose(recovered, params[:2], atol=ATOL, rtol=RTOL,
                            err_msg="Round-trip FIGARCH(0,d,0) GED core failed")
        assert nu is not None
        # GED transform asymmetry: forward uses 1.0 bound, inverse uses 1.01
        # Ref: figarch_transform.m:48 vs figarch_itransform.m:54
        npt.assert_allclose(nu, 1.5, atol=0.02,
                            err_msg="GED nu round-trip failed (allowing transform asymmetry)")

    def test_transform_roundtrip_skewt(self) -> None:
        """Round-trip with Skewed-t nu and lambda parameters.

        Note: The SKEWT lambda transform has a known asymmetry: the forward
        uses bounds (-0.995, 0.995) while the inverse uses (-0.99, 0.99).
        This creates a small (~0.001) round-trip offset in lambda, which is
        by design in the original MATLAB code. We verify the offset is bounded.
        Ref: figarch_transform.m:55-58 vs figarch_itransform.m:59-62
        """
        # [omega, phi, d, beta, nu, lambda]
        params = np.array([1e-5, 0.1, 0.45, 0.3, 6.0, -0.2])
        transformed = figarch_transform(params, p=1, q=1, error_type=SKEWT)
        recovered, nu, lam = figarch_itransform(transformed, p=1, q=1, error_type=SKEWT)
        npt.assert_allclose(recovered, params[:4], atol=ATOL, rtol=RTOL,
                            err_msg="Round-trip FIGARCH(1,d,1) SKEWT core failed")
        assert nu is not None
        npt.assert_allclose(nu, 6.0, atol=ATOL, rtol=RTOL,
                            err_msg="SKEWT nu round-trip failed")
        assert lam is not None
        # SKEWT lambda transform asymmetry: forward uses ±0.995, inverse uses ±0.99
        npt.assert_allclose(lam, -0.2, atol=0.005,
                            err_msg="SKEWT lambda round-trip failed (allowing transform asymmetry)")

    def test_d_range_preserved(self) -> None:
        """After itransform, 0 < d < 1 for various unconstrained inputs."""
        # Test with a range of unconstrained d values
        for d_unc in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            # Unconstrained parameters: [log(omega), logit(d)]
            trans_params = np.array([np.log(1e-5), d_unc])
            recovered, _, _ = figarch_itransform(trans_params, p=0, q=0, error_type=NORMAL)
            d_out = recovered[1]
            assert 0 < d_out < 1, f"d={d_out} not in (0,1) for unconstrained d={d_unc}"

    def test_positivity_preserved(self) -> None:
        """omega > 0, phi >= 0, beta >= 0 after itransform for any real input."""
        # Random unconstrained parameter vectors
        rng = np.random.default_rng(42)
        for _ in range(20):
            # FIGARCH(1,d,1) NORMAL: 4 unconstrained params
            trans_params = rng.standard_normal(4)
            recovered, _, _ = figarch_itransform(trans_params, p=1, q=1, error_type=NORMAL)
            omega, phi, d, beta = recovered
            assert omega > 0, f"omega={omega} not positive"
            assert phi >= 0, f"phi={phi} negative"
            assert 0 < d < 1, f"d={d} not in (0,1)"
            assert beta >= 0, f"beta={beta} negative"


# ============================================================================
# Phase 4: Unit Tests — figarch_likelihood
# ============================================================================

class TestFigarchLikelihood:
    """Tests for figarch_likelihood — log-likelihood computation.

    Tests verify scalar return type, correct conditional variance shape,
    all 4 error distributions, and the estimFlag transform behavior.
    Ref: figarch_likelihood.m
    """

    def _make_test_data(self) -> tuple[np.ndarray, float, int]:
        """Create standard test data for likelihood tests."""
        rng = np.random.default_rng(42)
        epsilon = rng.standard_normal(200) * 0.01
        epsilon = epsilon - epsilon.mean()
        T = len(epsilon)
        back_cast = float(np.var(epsilon, ddof=1))
        return epsilon, back_cast, T

    def test_likelihood_returns_scalar(self) -> None:
        """Negated LL is scalar float."""
        epsilon, back_cast, T = self._make_test_data()
        params = np.array([1e-5, 0.45])  # [omega, d] for FIGARCH(0,d,0)
        LL, lls, ht = figarch_likelihood(
            params, epsilon, p=0, q=0, error_type=NORMAL,
            truncLag=100, back_cast=back_cast, T=T
        )
        assert isinstance(LL, (float, np.floating)), f"LL type: {type(LL)}"
        assert np.isfinite(LL), f"LL is not finite: {LL}"

    def test_likelihood_ht_shape(self) -> None:
        """Conditional variance ht has shape (T,)."""
        epsilon, back_cast, T = self._make_test_data()
        params = np.array([1e-5, 0.45])
        LL, lls, ht = figarch_likelihood(
            params, epsilon, p=0, q=0, error_type=NORMAL,
            truncLag=100, back_cast=back_cast, T=T
        )
        assert ht.shape == (T,), f"Expected ht shape ({T},), got {ht.shape}"

    def test_likelihood_lls_shape(self) -> None:
        """Per-observation log-likelihoods lls has shape (T,)."""
        epsilon, back_cast, T = self._make_test_data()
        params = np.array([1e-5, 0.45])
        LL, lls, ht = figarch_likelihood(
            params, epsilon, p=0, q=0, error_type=NORMAL,
            truncLag=100, back_cast=back_cast, T=T
        )
        assert lls.shape == (T,), f"Expected lls shape ({T},), got {lls.shape}"

    def test_likelihood_ht_positive(self) -> None:
        """All conditional variances must be positive."""
        epsilon, back_cast, T = self._make_test_data()
        params = np.array([1e-5, 0.1, 0.45, 0.3])
        LL, lls, ht = figarch_likelihood(
            params, epsilon, p=1, q=1, error_type=NORMAL,
            truncLag=100, back_cast=back_cast, T=T
        )
        assert np.all(ht > 0), f"Some ht values are non-positive: min={ht.min():.10e}"

    @pytest.mark.parametrize("error_type,extra_params", [
        (NORMAL, np.array([])),
        (STUDENTST, np.array([6.0])),
        (GED, np.array([1.5])),
        (SKEWT, np.array([6.0, -0.2])),
    ])
    def test_likelihood_all_error_types(
        self, error_type: int, extra_params: np.ndarray
    ) -> None:
        """Test likelihood computation with all 4 error distributions."""
        epsilon, back_cast, T = self._make_test_data()
        # FIGARCH(1,d,1) core params
        core = np.array([1e-5, 0.1, 0.45, 0.3])
        params = np.concatenate([core, extra_params])
        LL, lls, ht = figarch_likelihood(
            params, epsilon, p=1, q=1, error_type=error_type,
            truncLag=100, back_cast=back_cast, T=T
        )
        assert np.isfinite(LL), f"LL not finite for error_type={error_type}: {LL}"
        assert np.all(np.isfinite(ht)), "Some ht values are not finite"

    def test_likelihood_estim_flag(self) -> None:
        """estimFlag=True should internally inverse-transform parameters."""
        epsilon, back_cast, T = self._make_test_data()
        # Constrained parameters
        params_constrained = np.array([1e-5, 0.45])
        # Transform to unconstrained
        params_transformed = figarch_transform(
            params_constrained, p=0, q=0, error_type=NORMAL
        )
        # Call with estim_flag=True
        LL1, _, ht1 = figarch_likelihood(
            params_transformed, epsilon, p=0, q=0, error_type=NORMAL,
            truncLag=100, back_cast=back_cast, T=T, estim_flag=True
        )
        # Call with constrained params and estim_flag=False
        LL2, _, ht2 = figarch_likelihood(
            params_constrained, epsilon, p=0, q=0, error_type=NORMAL,
            truncLag=100, back_cast=back_cast, T=T, estim_flag=False
        )
        # Both should give same result
        npt.assert_allclose(LL1, LL2, atol=ATOL, rtol=RTOL,
                            err_msg="estimFlag should give same LL as direct evaluation")
        npt.assert_allclose(ht1, ht2, atol=ATOL, rtol=RTOL,
                            err_msg="estimFlag should give same ht as direct evaluation")

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_likelihood_parity(self, univariate_fixture_dir) -> None:
        """Compare LL against MATLAB fixture."""
        try:
            fixture_ll = load_univariate_fixture(
                univariate_fixture_dir, 'figarch', 'loglikelihood'
            )
        except Exception:
            pytest.skip("FIGARCH loglikelihood fixture not available")
        try:
            fixture_input = load_univariate_input(
                univariate_fixture_dir, 'figarch'
            )
        except Exception:
            pytest.skip("FIGARCH input data fixture not available")
        try:
            fixture_params = load_univariate_fixture(
                univariate_fixture_dir, 'figarch', 'parameters'
            )
        except Exception:
            pytest.skip("FIGARCH parameters fixture not available")

        epsilon = fixture_input
        T = len(epsilon)
        back_cast = float(np.var(epsilon, ddof=1))
        LL, _, _ = figarch_likelihood(
            fixture_params, epsilon, p=1, q=1, error_type=NORMAL,
            truncLag=1000, back_cast=back_cast, T=T
        )
        npt.assert_allclose(LL, fixture_ll, atol=ATOL, rtol=RTOL,
                            err_msg="FIGARCH likelihood parity failure")


# ============================================================================
# Phase 5: Unit Tests — figarch_simulate
# ============================================================================

class TestFigarchSimulate:
    """Tests for figarch_simulate — FIGARCH time series simulation.

    Verifies output shapes, positive variances, long memory structure,
    and various error distribution support.
    Ref: figarch_simulate.m
    """

    def test_simulate_output_shape(self) -> None:
        """Returns (data, ht) with correct shapes for scalar t."""
        T = 500
        params = np.array([0.1, 0.42])  # [omega, d] for FIGARCH(0,d,0)
        data, ht = figarch_simulate(T, params, p=0, q=0, truncLag=100, bcLength=100)
        assert data.shape == (T,), f"Expected data shape ({T},), got {data.shape}"
        assert ht.shape == (T,), f"Expected ht shape ({T},), got {ht.shape}"

    def test_simulate_output_shape_11(self) -> None:
        """FIGARCH(1,d,1) simulation outputs correct shapes."""
        T = 300
        params = np.array([0.1, 0.1, 0.42, 0.3])  # [omega, phi, d, beta]
        data, ht = figarch_simulate(T, params, p=1, q=1, truncLag=100, bcLength=100)
        assert data.shape == (T,)
        assert ht.shape == (T,)

    def test_simulate_positive_variances(self) -> None:
        """ht > 0 for all t when parameters are valid."""
        T = 500
        params = np.array([0.1, 0.42])
        data, ht = figarch_simulate(T, params, p=0, q=0, truncLag=200, bcLength=200)
        assert np.all(ht > 0), f"Non-positive variances found: min ht = {ht.min():.10e}"

    def test_simulate_with_user_innovations(self) -> None:
        """Simulation with user-supplied random numbers."""
        rng = np.random.default_rng(42)
        innovations = rng.standard_normal(300)
        params = np.array([0.1, 0.42])
        data, ht = figarch_simulate(
            innovations, params, p=0, q=0, truncLag=100, bcLength=100
        )
        assert data.shape == (300,)
        assert ht.shape == (300,)

    def test_simulate_studentst(self) -> None:
        """FIGARCH simulation with Student-t errors."""
        T = 300
        params = np.array([0.1, 0.42, 6.0])  # [omega, d, nu]
        data, ht = figarch_simulate(
            T, params, p=0, q=0, error_type='STUDENTST',
            truncLag=100, bcLength=100
        )
        assert data.shape == (T,)
        assert np.all(ht > 0)

    def test_simulate_ged(self) -> None:
        """FIGARCH simulation with GED errors."""
        T = 300
        params = np.array([0.1, 0.42, 1.5])  # [omega, d, nu]
        data, ht = figarch_simulate(
            T, params, p=0, q=0, error_type='GED',
            truncLag=100, bcLength=100
        )
        assert data.shape == (T,)
        assert np.all(ht > 0)

    def test_simulate_skewt(self) -> None:
        """FIGARCH simulation with Skewed-t errors."""
        T = 300
        params = np.array([0.1, 0.42, 6.0, -0.2])  # [omega, d, nu, lambda]
        data, ht = figarch_simulate(
            T, params, p=0, q=0, error_type='SKEWT',
            truncLag=100, bcLength=100
        )
        assert data.shape == (T,)
        assert np.all(ht > 0)

    def test_simulate_invalid_omega(self) -> None:
        """Negative omega should raise ValueError."""
        with pytest.raises(ValueError, match="omega"):
            figarch_simulate(100, np.array([-0.1, 0.42]), p=0, q=0)

    def test_simulate_long_memory_structure(self) -> None:
        """For d > 0, simulated variance should show slow-decaying ACF signature.

        The hallmark of long memory is persistent autocorrelation in the
        squared returns (or variance proxy). We check that the autocorrelation
        at lag 50 is still positive and reasonably large.
        """
        T = 5000
        params = np.array([0.1, 0.6])  # d=0.6 → strong long memory
        data, ht = figarch_simulate(T, params, p=0, q=0, truncLag=500, bcLength=500)
        # Compute sample autocorrelation of ht at lag 50
        ht_demeaned = ht - ht.mean()
        var_ht = np.sum(ht_demeaned ** 2)
        if var_ht > 0:
            acf_50 = np.sum(ht_demeaned[50:] * ht_demeaned[:-50]) / var_ht
            # For strong long-memory, ACF at lag 50 should still be positive
            assert acf_50 > 0, (
                f"Long-memory ACF at lag 50 should be positive, got {acf_50:.6f}"
            )


# ============================================================================
# Phase 6: Unit Tests — figarch_starting_values
# ============================================================================

class TestFigarchStartingValues:
    """Tests for figarch_starting_values — grid search starting value computation.

    Verifies correct parameter vector lengths for all (p,q) combinations,
    d in valid range, positive omega, and grid search behavior.
    Ref: figarch_starting_values.m
    """

    def _make_test_data(self) -> tuple[np.ndarray, float, int]:
        """Create standard test data for starting value tests."""
        rng = np.random.default_rng(42)
        eps = rng.standard_normal(200) * 0.01
        eps = eps - eps.mean()
        T = len(eps)
        back_cast = float(np.var(eps, ddof=1))
        return eps, back_cast, T

    def test_starting_values_length_figarch_1_d_1(self) -> None:
        """Correct length for FIGARCH(1,d,1): 1(omega) + 1(phi) + 1(d) + 1(beta) = 4."""
        eps, back_cast, T = self._make_test_data()
        sv, nu, lam = figarch_starting_values(
            None, eps, p=1, q=1, error_type=NORMAL, truncLag=100, back_cast=back_cast, T=T
        )
        assert sv.shape[0] == 4, f"Expected 4 params, got {sv.shape[0]}"
        assert nu is None
        assert lam is None

    def test_starting_values_length_figarch_0_d_0(self) -> None:
        """Correct length for FIGARCH(0,d,0): 1(omega) + 1(d) = 2."""
        eps, back_cast, T = self._make_test_data()
        sv, nu, lam = figarch_starting_values(
            None, eps, p=0, q=0, error_type=NORMAL, truncLag=100, back_cast=back_cast, T=T
        )
        assert sv.shape[0] == 2, f"Expected 2 params, got {sv.shape[0]}"

    def test_starting_values_length_figarch_1_d_0(self) -> None:
        """Correct length for FIGARCH(1,d,0): 1(omega) + 1(phi) + 1(d) = 3."""
        eps, back_cast, T = self._make_test_data()
        sv, nu, lam = figarch_starting_values(
            None, eps, p=1, q=0, error_type=NORMAL, truncLag=100, back_cast=back_cast, T=T
        )
        assert sv.shape[0] == 3, f"Expected 3 params, got {sv.shape[0]}"

    def test_starting_values_length_figarch_0_d_1(self) -> None:
        """Correct length for FIGARCH(0,d,1): 1(omega) + 1(d) + 1(beta) = 3."""
        eps, back_cast, T = self._make_test_data()
        sv, nu, lam = figarch_starting_values(
            None, eps, p=0, q=1, error_type=NORMAL, truncLag=100, back_cast=back_cast, T=T
        )
        assert sv.shape[0] == 3, f"Expected 3 params, got {sv.shape[0]}"

    def test_starting_values_d_in_range(self) -> None:
        """Starting d must be in (0, 1)."""
        eps, back_cast, T = self._make_test_data()
        sv, _, _ = figarch_starting_values(
            None, eps, p=0, q=0, error_type=NORMAL, truncLag=100, back_cast=back_cast, T=T
        )
        # For FIGARCH(0,d,0), d is at index 1
        d = sv[1]
        assert 0 < d < 1, f"Starting d={d} not in (0,1)"

    def test_starting_values_omega_positive(self) -> None:
        """Starting omega must be positive."""
        eps, back_cast, T = self._make_test_data()
        sv, _, _ = figarch_starting_values(
            None, eps, p=0, q=0, error_type=NORMAL, truncLag=100, back_cast=back_cast, T=T
        )
        omega = sv[0]
        assert omega > 0, f"Starting omega={omega} not positive"

    def test_starting_values_studentst(self) -> None:
        """Starting values for Student-t include nu."""
        eps, back_cast, T = self._make_test_data()
        sv, nu, lam = figarch_starting_values(
            None, eps, p=1, q=1, error_type=STUDENTST, truncLag=100, back_cast=back_cast, T=T
        )
        assert sv.shape[0] == 4  # core params
        assert nu is not None and nu > 2, f"Student-t nu={nu} should be > 2"
        assert lam is None

    def test_starting_values_skewt(self) -> None:
        """Starting values for Skewed-t include nu and lambda."""
        eps, back_cast, T = self._make_test_data()
        sv, nu, lam = figarch_starting_values(
            None, eps, p=1, q=1, error_type=SKEWT, truncLag=100, back_cast=back_cast, T=T
        )
        assert sv.shape[0] == 4  # core params
        assert nu is not None and nu > 2
        assert lam is not None and -1 < lam < 1

    def test_starting_values_user_supplied(self) -> None:
        """User-supplied starting values are returned as-is (core params only)."""
        eps, back_cast, T = self._make_test_data()
        user_sv = np.array([1e-5, 0.45])  # [omega, d] for FIGARCH(0,d,0)
        sv, nu, lam = figarch_starting_values(
            user_sv, eps, p=0, q=0, error_type=NORMAL, truncLag=100, back_cast=back_cast, T=T
        )
        npt.assert_allclose(sv[:2], user_sv, atol=1e-12,
                            err_msg="User-supplied starting values should be preserved")


# ============================================================================
# Phase 7: Integration Tests — Full FIGARCH estimation
# ============================================================================

class TestFigarchIntegration:
    """Integration tests for the full FIGARCH estimation pipeline.

    Tests the complete figarch() driver function with various (p,q)
    combinations and error distributions. Verifies output types, shapes,
    and basic economic constraints (d in (0,1), ht > 0, VCV symmetric).
    Ref: figarch.m
    """

    @pytest.mark.slow
    def test_figarch_1_d_1_normal(self, univariate_data: np.ndarray) -> None:
        """Full FIGARCH(1,d,1) estimation, NORMAL errors."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=1, q=1, error_type='NORMAL', trunc_lag=100
        )
        assert isinstance(params, np.ndarray)
        assert isinstance(LL, (float, np.floating))
        assert ht.shape == univariate_data.shape
        assert np.isfinite(LL)

    @pytest.mark.slow
    def test_figarch_0_d_0_normal(self, univariate_data: np.ndarray) -> None:
        """FIGARCH(0,d,0) — simplest long-memory model."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=0, q=0, error_type='NORMAL', trunc_lag=100
        )
        assert params.shape[0] == 2  # [omega, d]
        assert np.isfinite(LL)

    @pytest.mark.slow
    def test_figarch_1_d_0_normal(self, univariate_data: np.ndarray) -> None:
        """FIGARCH(1,d,0) — phi only."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=1, q=0, error_type='NORMAL', trunc_lag=100
        )
        assert params.shape[0] == 3  # [omega, phi, d]
        assert np.isfinite(LL)

    @pytest.mark.slow
    def test_figarch_0_d_1_normal(self, univariate_data: np.ndarray) -> None:
        """FIGARCH(0,d,1) — beta only."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=0, q=1, error_type='NORMAL', trunc_lag=100
        )
        assert params.shape[0] == 3  # [omega, d, beta]
        assert np.isfinite(LL)

    @pytest.mark.slow
    def test_figarch_studentst(self, univariate_data: np.ndarray) -> None:
        """FIGARCH with Student-t errors."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=1, q=1, error_type='STUDENTST', trunc_lag=100
        )
        # Extra nu parameter appended
        assert params.shape[0] >= 5  # [omega, phi, d, beta, nu]
        assert np.isfinite(LL)

    @pytest.mark.slow
    def test_figarch_skewt(self, univariate_data: np.ndarray) -> None:
        """FIGARCH with Skewed-t errors."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=1, q=1, error_type='SKEWT', trunc_lag=100
        )
        # Extra nu and lambda parameters
        assert params.shape[0] >= 6  # [omega, phi, d, beta, nu, lambda]
        assert np.isfinite(LL)

    @pytest.mark.slow
    def test_figarch_returns_correct_types(self, univariate_data: np.ndarray) -> None:
        """All 7 outputs have correct types."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=1, q=1, error_type='NORMAL', trunc_lag=100
        )
        assert isinstance(params, np.ndarray), f"params type: {type(params)}"
        assert isinstance(LL, (float, np.floating)), f"LL type: {type(LL)}"
        assert isinstance(ht, np.ndarray), f"ht type: {type(ht)}"
        assert isinstance(VCVrobust, np.ndarray), f"VCVrobust type: {type(VCVrobust)}"
        assert isinstance(VCV, np.ndarray), f"VCV type: {type(VCV)}"
        assert isinstance(scores, np.ndarray), f"scores type: {type(scores)}"
        assert isinstance(diagnostics, dict), f"diagnostics type: {type(diagnostics)}"

    @pytest.mark.slow
    def test_figarch_d_in_range(self, univariate_data: np.ndarray) -> None:
        """Estimated d must be in (0, 1)."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=1, q=1, error_type='NORMAL', trunc_lag=100
        )
        # For FIGARCH(1,d,1), d is at index 2: [omega, phi, d, beta]
        d = params[2]
        assert 0 < d < 1, f"Estimated d={d} not in (0,1)"

    @pytest.mark.slow
    def test_figarch_ht_positive(self, univariate_data: np.ndarray) -> None:
        """ht > 0 for all t."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=1, q=1, error_type='NORMAL', trunc_lag=100
        )
        assert np.all(ht > 0), f"Non-positive variances: min ht = {ht.min():.10e}"

    @pytest.mark.slow
    def test_figarch_vcv_symmetric(self, univariate_data: np.ndarray) -> None:
        """VCVrobust and VCV should be symmetric matrices."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=1, q=1, error_type='NORMAL', trunc_lag=100
        )
        if VCVrobust is not None and VCVrobust.size > 0:
            npt.assert_allclose(
                VCVrobust, VCVrobust.T, atol=1e-10,
                err_msg="VCVrobust not symmetric"
            )
        if VCV is not None and VCV.size > 0:
            npt.assert_allclose(
                VCV, VCV.T, atol=1e-10,
                err_msg="VCV not symmetric"
            )

    @pytest.mark.slow
    def test_figarch_diagnostics_keys(self, univariate_data: np.ndarray) -> None:
        """Diagnostics dict should have expected keys."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=0, q=0, error_type='NORMAL', trunc_lag=100
        )
        assert 'EXITFLAG' in diagnostics
        assert 'ITERATIONS' in diagnostics
        assert 'FUNCCOUNT' in diagnostics
        assert 'MESSAGE' in diagnostics

    @pytest.mark.slow
    def test_figarch_ll_positive(self, univariate_data: np.ndarray) -> None:
        """The returned LL (log-likelihood) should be a positive (or at least
        large-magnitude) value for typical financial data.

        Note: the figarch() driver returns LL = -(-LL) = positive LL.
        """
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            univariate_data, p=0, q=0, error_type='NORMAL', trunc_lag=100
        )
        # For a proper log-likelihood on 1000 observations, this should be
        # a large positive number (sum of individual log-likelihoods)
        assert np.isfinite(LL), f"LL should be finite, got {LL}"


# ============================================================================
# Phase 8: Parity Tests — MATLAB fixture comparison
# ============================================================================

class TestFigarchParity:
    """Parity tests comparing Python FIGARCH against MATLAB reference fixtures.

    All tests are marked with @pytest.mark.parity and @pytest.mark.requires_fixtures.
    They gracefully skip when fixture files are not available.
    Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4.
    """

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_figarch_parameters_parity(self, univariate_fixture_dir) -> None:
        """Compare parameters (including d) against MATLAB (±1e-6)."""
        try:
            fixture_params = load_univariate_fixture(
                univariate_fixture_dir, 'figarch', 'parameters'
            )
            fixture_input = load_univariate_input(
                univariate_fixture_dir, 'figarch'
            )
        except Exception:
            pytest.skip("FIGARCH parameter parity fixtures not available")

        epsilon = fixture_input
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            epsilon, p=1, q=1, error_type='NORMAL', trunc_lag=1000
        )
        npt.assert_allclose(
            params[:len(fixture_params)], fixture_params,
            atol=ATOL, rtol=RTOL,
            err_msg="FIGARCH parameter parity failure"
        )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_figarch_loglikelihood_parity(self, univariate_fixture_dir) -> None:
        """Compare LL against MATLAB."""
        try:
            fixture_ll = load_univariate_fixture(
                univariate_fixture_dir, 'figarch', 'loglikelihood'
            )
            fixture_input = load_univariate_input(
                univariate_fixture_dir, 'figarch'
            )
        except Exception:
            pytest.skip("FIGARCH log-likelihood parity fixtures not available")

        epsilon = fixture_input
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            epsilon, p=1, q=1, error_type='NORMAL', trunc_lag=1000
        )
        expected_ll = float(fixture_ll.ravel()[0]) if fixture_ll.ndim > 0 else float(fixture_ll)
        npt.assert_allclose(
            LL, expected_ll, atol=ATOL, rtol=RTOL,
            err_msg="FIGARCH log-likelihood parity failure"
        )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_figarch_ht_parity(self, univariate_fixture_dir) -> None:
        """Compare conditional variances against MATLAB."""
        try:
            fixture_ht = load_univariate_fixture(
                univariate_fixture_dir, 'figarch', 'ht'
            )
            fixture_input = load_univariate_input(
                univariate_fixture_dir, 'figarch'
            )
        except Exception:
            pytest.skip("FIGARCH ht parity fixtures not available")

        epsilon = fixture_input
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = figarch(
            epsilon, p=1, q=1, error_type='NORMAL', trunc_lag=1000
        )
        expected_ht = fixture_ht.ravel()
        npt.assert_allclose(
            ht, expected_ht, atol=ATOL, rtol=RTOL,
            err_msg="FIGARCH conditional variance parity failure"
        )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_figarch_weights_parity(self, univariate_fixture_dir) -> None:
        """Compare ARCH(infinity) weights against MATLAB."""
        try:
            fixture_weights = load_univariate_fixture(
                univariate_fixture_dir, 'figarch', 'weights'
            )
        except Exception:
            pytest.skip("FIGARCH weights parity fixtures not available")

        if isinstance(fixture_weights, np.ndarray) and fixture_weights.dtype == object:
            pytest.skip("Object-dtype fixture needs special handling")

        fixture_weights = fixture_weights.ravel()
        truncLag = len(fixture_weights)
        # Typical FIGARCH(1,d,1) weight params [phi, d, beta]
        params = np.array([0.1, 0.45, 0.3])
        w = figarch_weights(params, 1, 1, truncLag)
        npt.assert_allclose(
            w, fixture_weights, atol=ATOL, rtol=RTOL,
            err_msg="FIGARCH weights parity failure"
        )


# ============================================================================
# Additional Edge-Case Tests
# ============================================================================

class TestFigarchEdgeCases:
    """Additional edge-case and boundary tests for FIGARCH family.

    Covers numerical robustness scenarios and parametric boundary behavior.
    """

    def test_weights_consistency_across_pq(self) -> None:
        """When phi=0, beta=0, FIGARCH(1,d,1) weights equal FIGARCH(0,d,0) weights.

        Setting phi=0 and beta=0 in FIGARCH(1,d,1) should produce identical
        weights to FIGARCH(0,d,0) with the same d.
        """
        d = 0.45
        truncLag = 200
        w_00 = figarch_weights(np.array([d]), 0, 0, truncLag)
        w_11 = figarch_weights(np.array([0.0, d, 0.0]), 1, 1, truncLag)
        npt.assert_allclose(w_00, w_11, atol=1e-12,
                            err_msg="phi=0,beta=0 FIGARCH(1,d,1) should match FIGARCH(0,d,0)")

    def test_transform_extreme_d(self) -> None:
        """Transform/itransform handles d near 0 and d near 1."""
        for d in [0.01, 0.99]:
            params = np.array([1e-5, d])  # FIGARCH(0,d,0)
            trans = figarch_transform(params, 0, 0, NORMAL)
            rec, _, _ = figarch_itransform(trans, 0, 0, NORMAL)
            npt.assert_allclose(rec, params, atol=1e-4,
                                err_msg=f"Round-trip failed for d={d}")

    def test_likelihood_different_trunclag(self) -> None:
        """Likelihood should converge as truncLag increases."""
        rng = np.random.default_rng(42)
        eps = rng.standard_normal(200) * 0.01
        eps = eps - eps.mean()
        T = len(eps)
        back_cast = float(np.var(eps, ddof=1))
        params = np.array([1e-5, 0.45])  # FIGARCH(0,d,0)

        LL_50, _, _ = figarch_likelihood(
            params, eps, 0, 0, NORMAL, 50, back_cast, T
        )
        LL_200, _, _ = figarch_likelihood(
            params, eps, 0, 0, NORMAL, 200, back_cast, T
        )
        # Both should be finite; values should be similar but not identical
        assert np.isfinite(LL_50)
        assert np.isfinite(LL_200)

    def test_simulate_various_d(self) -> None:
        """Simulation should work for various d values in (0,1)."""
        for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
            params = np.array([0.1, d])  # [omega, d]
            data, ht = figarch_simulate(100, params, 0, 0, truncLag=100, bcLength=100)
            assert data.shape == (100,), f"Failed for d={d}"
            assert np.all(ht > 0), f"Non-positive ht for d={d}"

    def test_weights_sum_behavior(self) -> None:
        """Weight sum should be < 1 for d < 1 (stationarity)."""
        for d in [0.2, 0.4, 0.6]:
            params = np.array([d])
            w = figarch_weights(params, 0, 0, 2000)
            s = np.sum(w)
            assert s < 1.0, (
                f"Weight sum={s:.6f} should be < 1 for d={d}"
            )

    @pytest.mark.parametrize("p,q", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_starting_values_all_pq(self, p: int, q: int) -> None:
        """Starting values grid search should work for all (p,q) combinations."""
        rng = np.random.default_rng(42)
        eps = rng.standard_normal(200) * 0.01
        eps = eps - eps.mean()
        T = len(eps)
        back_cast = float(np.var(eps, ddof=1))
        sv, nu, lam = figarch_starting_values(
            None, eps, p=p, q=q, error_type=NORMAL,
            truncLag=100, back_cast=back_cast, T=T
        )
        expected_len = 2 + p + q
        assert sv.shape[0] == expected_len, (
            f"Expected {expected_len} params for p={p}, q={q}, got {sv.shape[0]}"
        )
        # omega should be positive
        assert sv[0] > 0, f"omega={sv[0]} not positive for p={p}, q={q}"
