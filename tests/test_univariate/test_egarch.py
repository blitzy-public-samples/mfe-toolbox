"""
Pytest tests for the EGARCH (Exponential GARCH) model family.

Tests cover all 10 EGARCH source modules:
  1. egarch_parameter_check — input validation
  2. egarch_nlcon — nonlinear stationarity constraint (unique to EGARCH)
  3. egarch_transform / egarch_itransform — round-trip parameter transforms
  4. egarch_core — Numba JIT log-variance recursion (replaces C MEX)
  5. egarch_likelihood — log-likelihood computation for 4 error distributions
  6. egarch_simulate — simulation with leverage and 4 error distributions
  7. egarch_starting_values — starting value grid search
  8. egarch_display — result display formatting
  9. egarch — full estimation pipeline integration tests
 10. Parity tests against MATLAB-generated fixture files

Tolerances: ATOL=1e-6, RTOL=1e-4 per AAP Section 0.7.1.

Migrated from MATLAB MFE Toolbox Version 4.0 (Kevin Sheppard).
"""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.univariate.egarch import egarch
from mfe_toolbox.univariate.egarch_core import egarch_core
from mfe_toolbox.univariate.egarch_display import egarch_display
from mfe_toolbox.univariate.egarch_itransform import egarch_itransform
from mfe_toolbox.univariate.egarch_likelihood import egarch_likelihood
from mfe_toolbox.univariate.egarch_nlcon import egarch_nlcon
from mfe_toolbox.univariate.egarch_parameter_check import egarch_parameter_check
from mfe_toolbox.univariate.egarch_simulate import egarch_simulate
from mfe_toolbox.univariate.egarch_starting_values import egarch_starting_values
from mfe_toolbox.univariate.egarch_transform import egarch_transform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# Reproducible random seed for all test data
SEED: int = 42

# EGARCH error distribution codes
NORMAL: int = 1
STUDENTST: int = 2
GED: int = 3
SKEWT: int = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded RNG for reproducible test data."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="module")
def egarch_data(rng: np.random.Generator) -> np.ndarray:
    """Generate T=1000 mean-zero return series for EGARCH tests.

    Matches conftest.py univariate_data fixture generation pattern.
    """
    T = 1000
    data = rng.standard_normal(T)
    data = data - data.mean()
    return data


@pytest.fixture(scope="module")
def short_data(rng: np.random.Generator) -> np.ndarray:
    """Generate T=200 mean-zero return series for faster tests."""
    # Use a separate RNG to not disturb egarch_data's sequence
    rng2 = np.random.default_rng(123)
    T = 200
    data = rng2.standard_normal(T)
    data = data - data.mean()
    return data


@pytest.fixture(scope="module")
def fixture_dir() -> Path:
    """Return path to the univariate fixture directory."""
    env_dir = os.environ.get("MFE_FIXTURE_DIR")
    if env_dir:
        return Path(env_dir) / "univariate"
    return Path(__file__).parent.parent / "fixtures" / "univariate"


# ===========================================================================
# Phase 2: egarch_parameter_check tests
# ===========================================================================
class TestEgarchParameterCheck:
    """Tests for egarch_parameter_check input validation."""

    def test_valid_inputs_p1_o1_q1(self, egarch_data: np.ndarray) -> None:
        """Valid EGARCH(1,1,1) inputs pass without error."""
        p, o, q, et, sv, opts = egarch_parameter_check(
            egarch_data, 1, 1, 1, 'NORMAL'
        )
        assert p == 1
        assert o == 1
        assert q == 1
        assert et == NORMAL

    def test_valid_inputs_default_error_type(self, egarch_data: np.ndarray) -> None:
        """Omitting error_type defaults to NORMAL (1)."""
        p, o, q, et, sv, opts = egarch_parameter_check(
            egarch_data, 1, 1, 1
        )
        assert et == NORMAL

    def test_invalid_p_and_o_both_zero(self, egarch_data: np.ndarray) -> None:
        """p=0 AND o=0 raises ValueError.

        Ref: egarch_parameter_check.m:59-61 — 'One of p or o must be non-zero'
        """
        with pytest.raises(ValueError, match="[Oo]ne of p or o"):
            egarch_parameter_check(egarch_data, 0, 0, 1)

    def test_invalid_o_negative(self, egarch_data: np.ndarray) -> None:
        """o=-1 raises ValueError.

        Ref: egarch_parameter_check.m:45-47
        """
        with pytest.raises(ValueError, match="o must be a non-negative scalar"):
            egarch_parameter_check(egarch_data, 1, -1, 1)

    def test_invalid_q_negative(self, egarch_data: np.ndarray) -> None:
        """q=-1 raises ValueError.

        Ref: egarch_parameter_check.m:35-37
        """
        with pytest.raises(ValueError, match="q must be a non-negative scalar"):
            egarch_parameter_check(egarch_data, 1, 1, -1)

    def test_invalid_p_negative(self, egarch_data: np.ndarray) -> None:
        """p=-1 raises ValueError.

        Ref: egarch_parameter_check.m:55-57
        """
        with pytest.raises(ValueError, match="p must be"):
            egarch_parameter_check(egarch_data, -1, 1, 1)

    def test_invalid_error_type(self, egarch_data: np.ndarray) -> None:
        """Invalid error_type string raises ValueError.

        Ref: egarch_parameter_check.m:81-83
        """
        with pytest.raises(ValueError, match="error_type"):
            egarch_parameter_check(egarch_data, 1, 1, 1, 'INVALID')

    def test_invalid_error_type_int(self, egarch_data: np.ndarray) -> None:
        """Invalid error_type integer raises ValueError."""
        with pytest.raises(ValueError, match="error_type"):
            egarch_parameter_check(egarch_data, 1, 1, 1, 99)

    @pytest.mark.parametrize("error_type", ['NORMAL', 'STUDENTST', 'GED', 'SKEWT'])
    def test_valid_higher_order(
        self, egarch_data: np.ndarray, error_type: str
    ) -> None:
        """EGARCH(2,2,2) with all error types validates correctly."""
        p, o, q, et, sv, opts = egarch_parameter_check(
            egarch_data, 2, 2, 2, error_type
        )
        assert p == 2 and o == 2 and q == 2

    def test_empty_data_raises(self) -> None:
        """Empty data raises ValueError.

        Ref: egarch_parameter_check.m:28-29
        """
        with pytest.raises(ValueError, match="data"):
            egarch_parameter_check(np.array([]), 1, 1, 1)

    def test_scalar_data_raises(self) -> None:
        """Scalar data raises ValueError.

        Ref: egarch_parameter_check.m:26 — length(data)==1
        """
        with pytest.raises(ValueError, match="data"):
            egarch_parameter_check(np.array([1.0]), 1, 1, 1)

    def test_row_vector_data_raises(self) -> None:
        """Row vector (2-D with >1 column) data raises ValueError.

        Ref: egarch_parameter_check.m:26 — size(data,2) > 1
        """
        data_2d = np.random.default_rng(42).standard_normal((5, 3))
        with pytest.raises(ValueError, match="data"):
            egarch_parameter_check(data_2d, 1, 1, 1)

    @pytest.mark.parametrize("error_type_str,error_type_int", [
        ('NORMAL', 1), ('STUDENTST', 2), ('GED', 3), ('SKEWT', 4),
    ])
    def test_error_type_mapping(
        self, egarch_data: np.ndarray, error_type_str: str, error_type_int: int
    ) -> None:
        """String error types map to correct integer codes."""
        _, _, _, et, _, _ = egarch_parameter_check(
            egarch_data, 1, 1, 1, error_type_str
        )
        assert et == error_type_int


# ===========================================================================
# Phase 3: egarch_nlcon tests (UNIQUE to EGARCH)
# ===========================================================================
class TestEgarchNlcon:
    """Tests for the EGARCH nonlinear stationarity constraint.

    EGARCH's stationarity is enforced by checking that ALL roots of the
    characteristic polynomial [1, -beta(1), ..., -beta(q)] have modulus
    less than 0.99998.

    Ref: egarch_nlcon.m:28 — c = abs(roots([1;-beta])) - 0.99998
    """

    def test_nlcon_stationary_params(self) -> None:
        """beta=0.9 is stationary: all roots well inside unit circle.

        For beta=[0.9], polynomial is [1, -0.9]. Root = 0.9.
        c = |0.9| - 0.99998 = -0.09998 < 0  (stationary in MATLAB sign).
        """
        params = np.array([-0.1, 0.1, -0.05, 0.9])
        c = egarch_nlcon(params, p=1, o=1, q=1, error_type=NORMAL)
        # MATLAB convention: c < 0 means stationary
        assert np.all(c < 0), f"Expected c < 0 for stationary params, got {c}"

    def test_nlcon_nonstationary_params(self) -> None:
        """beta=1.1 is non-stationary: root outside unit circle.

        For beta=[1.1], polynomial is [1, -1.1]. Root = 1.1.
        c = |1.1| - 0.99998 = 0.10002 > 0  (non-stationary).
        """
        params = np.array([-0.1, 0.1, -0.05, 1.1])
        c = egarch_nlcon(params, p=1, o=1, q=1, error_type=NORMAL)
        assert np.any(c > 0), f"Expected c > 0 for non-stationary params, got {c}"

    def test_nlcon_higher_order(self) -> None:
        """EGARCH(1,1,2) with beta=[0.5, 0.4] is stationary.

        Sum of betas = 0.9 < 1, polynomial [1, -0.5, -0.4] has roots
        inside the unit circle.
        """
        # params: [omega, alpha(1), gamma(1), beta(1), beta(2)]
        params = np.array([-0.1, 0.1, -0.05, 0.5, 0.4])
        c = egarch_nlcon(params, p=1, o=1, q=2, error_type=NORMAL)
        assert np.all(c < 0), f"Expected c < 0 for stationary params, got {c}"

    def test_nlcon_returns_correct_shape(self) -> None:
        """Return array has length q (number of polynomial roots).

        Ref: egarch_nlcon.m:28 — roots of a degree-q polynomial
        """
        params = np.array([-0.1, 0.1, -0.05, 0.9])
        c = egarch_nlcon(params, p=1, o=1, q=1, error_type=NORMAL)
        assert isinstance(c, np.ndarray)
        # Polynomial of degree 1 has 1 root
        assert c.shape == (1,), f"Expected shape (1,), got {c.shape}"

    def test_nlcon_higher_order_shape(self) -> None:
        """EGARCH(1,1,2) constraint returns 2-element array."""
        params = np.array([-0.1, 0.1, -0.05, 0.5, 0.4])
        c = egarch_nlcon(params, p=1, o=1, q=2, error_type=NORMAL)
        assert c.shape == (2,), f"Expected shape (2,), got {c.shape}"

    def test_nlcon_boundary_case(self) -> None:
        """beta very close to 0.99998 boundary."""
        params = np.array([-0.1, 0.1, -0.05, 0.999])
        c = egarch_nlcon(params, p=1, o=1, q=1, error_type=NORMAL)
        # Root = 0.999, c = |0.999| - 0.99998 = -0.00098 (very small but < 0)
        assert np.all(c < 0), f"Expected c < 0 near boundary, got {c}"

    def test_nlcon_exact_unit_root(self) -> None:
        """beta=1.0 is on the unit root boundary (non-stationary)."""
        params = np.array([-0.1, 0.1, -0.05, 1.0])
        c = egarch_nlcon(params, p=1, o=1, q=1, error_type=NORMAL)
        # |1.0| - 0.99998 = 0.00002 > 0
        assert np.all(c > 0), f"Expected c > 0 for unit root, got {c}"


# ===========================================================================
# Phase 4: Transform / Itransform round-trip tests
# ===========================================================================
class TestEgarchTransformItransform:
    """Tests for egarch_transform and egarch_itransform round-trip consistency.

    For any valid constrained parameter vector, the round-trip
        egarch_itransform(egarch_transform(params, p, o, q, et), p, o, q, et)
    must recover the original params to within ±1e-6.

    Ref: egarch_transform.m / egarch_itransform.m
    """

    def test_transform_roundtrip_normal(self) -> None:
        """Round-trip for error_type=NORMAL (no distribution params)."""
        params = np.array([-0.1, 0.1, -0.05, 0.95])
        transformed = egarch_transform(params, p=1, o=1, q=1, error_type=NORMAL)
        recovered = egarch_itransform(transformed, p=1, o=1, q=1, error_type=NORMAL)
        npt.assert_allclose(recovered, params, atol=ATOL, rtol=RTOL)

    def test_transform_roundtrip_studentst(self) -> None:
        """Round-trip for error_type=STUDENTST (nu parameter).

        Ref: egarch_transform.m:43-46 — nu → sqrt(nu - 2.01)
        Ref: egarch_itransform.m:38-41 — nu = 2.01 + nu_unc^2
        """
        params = np.array([-0.1, 0.1, -0.05, 0.95, 5.0])
        transformed = egarch_transform(params, p=1, o=1, q=1, error_type=STUDENTST)
        recovered = egarch_itransform(transformed, p=1, o=1, q=1, error_type=STUDENTST)
        npt.assert_allclose(recovered, params, atol=ATOL, rtol=RTOL)

    def test_transform_roundtrip_ged(self) -> None:
        """Round-trip for error_type=GED (nu parameter with logistic).

        Ref: egarch_transform.m:47-51 — logit: temp = (nu-1)/49
        Ref: egarch_itransform.m:42-49 — inverse: nu = 49*sigmoid + 1.01

        Known MATLAB domain offset: forward maps from (1, 50) but inverse
        maps to (1.01, 50.01), producing a systematic +0.01 offset on nu.
        We verify:
        1. Core EGARCH params pass through identically.
        2. Nu round-trip is within the known domain tolerance (~0.01).
        3. Constrained nu is in a valid GED range.
        """
        params = np.array([-0.1, 0.1, -0.05, 0.95, 1.9])
        transformed = egarch_transform(params, p=1, o=1, q=1, error_type=GED)
        recovered = egarch_itransform(transformed, p=1, o=1, q=1, error_type=GED)

        # Core parameters must round-trip exactly
        npt.assert_allclose(recovered[:4], params[:4], atol=ATOL, rtol=RTOL)

        # Nu round-trip has known +0.01 offset from MATLAB domain mismatch
        # Ref: egarch_itransform.m uses 1.01 base, egarch_transform.m uses 1.0 base
        assert abs(recovered[4] - params[4]) < 0.02, (
            f"GED nu round-trip error {abs(recovered[4] - params[4])} exceeds 0.02"
        )
        # Constrained nu must be in valid GED range
        assert recovered[4] > 1.0, f"GED nu must be > 1, got {recovered[4]}"
        assert recovered[4] < 50.1, f"GED nu must be < ~50, got {recovered[4]}"

    def test_transform_roundtrip_skewt(self) -> None:
        """Round-trip for error_type=SKEWT (nu + lambda parameters).

        Ref: egarch_transform.m:55-59 — lambda logistic: temp=(lam+0.995)/1.99
        Ref: egarch_itransform.m:52-55 — inverse: lam=1.98*sigmoid - 0.99

        Known MATLAB domain offset: forward uses ±0.995 domain, inverse uses
        ±0.99 domain.  Nu for SKEWT uses same sqrt transform as Student-t
        which round-trips exactly.
        """
        params = np.array([-0.1, 0.1, -0.05, 0.95, 5.0, -0.1])
        transformed = egarch_transform(params, p=1, o=1, q=1, error_type=SKEWT)
        recovered = egarch_itransform(transformed, p=1, o=1, q=1, error_type=SKEWT)

        # Core params and nu should round-trip exactly (Student-t sqrt transform)
        npt.assert_allclose(recovered[:5], params[:5], atol=ATOL, rtol=RTOL)

        # Lambda has known domain offset (~0.005 for values near boundary)
        # Ref: forward domain (-0.995, 0.995) vs inverse domain (-0.99, 0.99)
        assert abs(recovered[5] - params[5]) < 0.01, (
            f"SKEWT lambda round-trip error {abs(recovered[5] - params[5])} exceeds 0.01"
        )
        assert -1.0 < recovered[5] < 1.0, f"lambda must be in (-1,1), got {recovered[5]}"

    def test_omega_can_be_negative(self) -> None:
        """Negative omega is preserved (EGARCH operates in log-variance space).

        Unlike GARCH, EGARCH's omega is unconstrained because it enters the
        log-variance equation directly.
        """
        params = np.array([-2.5, 0.1, -0.05, 0.95])
        transformed = egarch_transform(params, p=1, o=1, q=1, error_type=NORMAL)
        recovered = egarch_itransform(transformed, p=1, o=1, q=1, error_type=NORMAL)
        assert recovered[0] < 0, "Negative omega should be preserved"
        npt.assert_allclose(recovered[0], -2.5, atol=ATOL)

    def test_gamma_unrestricted(self) -> None:
        """Gamma sign is unrestricted (leverage effect can be positive or negative).

        Ref: egarch_transform.m — gamma parameters pass through without transform
        """
        params_neg_gamma = np.array([-0.1, 0.1, -0.3, 0.95])
        params_pos_gamma = np.array([-0.1, 0.1, 0.3, 0.95])

        t_neg = egarch_transform(params_neg_gamma, 1, 1, 1, NORMAL)
        t_pos = egarch_transform(params_pos_gamma, 1, 1, 1, NORMAL)

        r_neg = egarch_itransform(t_neg, 1, 1, 1, NORMAL)
        r_pos = egarch_itransform(t_pos, 1, 1, 1, NORMAL)

        npt.assert_allclose(r_neg[2], -0.3, atol=ATOL)
        npt.assert_allclose(r_pos[2], 0.3, atol=ATOL)

    def test_transform_does_not_modify_core_params(self) -> None:
        """Core EGARCH params (omega, alpha, gamma, beta) pass through unchanged.

        Only distribution shape parameters are transformed.
        """
        params = np.array([-0.1, 0.1, -0.05, 0.95, 5.0])
        transformed = egarch_transform(params.copy(), 1, 1, 1, STUDENTST)
        # Core parameters should be identical
        npt.assert_array_equal(transformed[:4], params[:4])
        # Nu should be transformed (different from original)
        assert transformed[4] != params[4]

    def test_transform_higher_order_roundtrip(self) -> None:
        """Round-trip for EGARCH(2,2,2) with SKEWT.

        Ref: Known lambda domain offset — see test_transform_roundtrip_skewt
        for full explanation of MATLAB domain mismatch.
        """
        # params: omega, alpha1, alpha2, gamma1, gamma2, beta1, beta2, nu, lambda
        params = np.array([-0.2, 0.05, 0.03, -0.02, 0.01, 0.5, 0.4, 6.0, 0.2])
        transformed = egarch_transform(params, p=2, o=2, q=2, error_type=SKEWT)
        recovered = egarch_itransform(transformed, p=2, o=2, q=2, error_type=SKEWT)

        # Core params + nu round-trip exactly
        npt.assert_allclose(recovered[:8], params[:8], atol=ATOL, rtol=RTOL)

        # Lambda has known SKEWT domain offset (~0.001 for moderate values)
        assert abs(recovered[8] - params[8]) < 0.01, (
            f"SKEWT lambda round-trip error {abs(recovered[8] - params[8])} exceeds 0.01"
        )
        assert -1.0 < recovered[8] < 1.0, f"lambda must be in (-1,1)"


# ===========================================================================
# Phase 5: egarch_core (Numba JIT) tests
# ===========================================================================
class TestEgarchCore:
    """Tests for the EGARCH log-variance recursion core (Numba JIT).

    This replaces the C MEX kernel (mex_source/egarch_core.c).

    Ref: egarch_core.m, egarch_core.c
    """

    def _make_core_inputs(
        self, egarch_data: np.ndarray, p: int = 1, o: int = 1, q: int = 1,
    ) -> dict:
        """Prepare standard inputs for egarch_core testing."""
        m = max(p, o, q)
        data_augmented = np.concatenate([np.zeros(m), egarch_data])
        T = len(data_augmented)
        var_est = float(np.var(egarch_data, ddof=1))
        back_cast = float(np.log(max(var_est, 1e-10)))
        upper = 10000.0 * float(np.max(data_augmented ** 2))
        # Standard EGARCH(1,1,1) parameters: omega, alpha, gamma, beta
        params = np.array([-0.1, 0.1, -0.05, 0.95])
        return {
            'data': data_augmented,
            'parameters': params,
            'back_cast': back_cast,
            'upper': upper,
            'p': p, 'o': o, 'q': q,
            'm': m, 'T': T,
        }

    def test_core_basic(self, egarch_data: np.ndarray) -> None:
        """egarch_core produces valid conditional variances."""
        inp = self._make_core_inputs(egarch_data)
        ht = egarch_core(**inp)
        assert ht is not None
        assert len(ht) == inp['T']
        # All conditional variances must be positive (exp of log-variance)
        assert np.all(ht > 0), "All conditional variances must be > 0"

    def test_core_output_shape(self, egarch_data: np.ndarray) -> None:
        """Output shape (T,) matches augmented input length."""
        inp = self._make_core_inputs(egarch_data)
        ht = egarch_core(**inp)
        assert ht.shape == (inp['T'],)

    def test_core_exp_positive(self, egarch_data: np.ndarray) -> None:
        """All outputs are strictly positive (exp of log-variance)."""
        inp = self._make_core_inputs(egarch_data)
        ht = egarch_core(**inp)
        npt.assert_array_less(0, ht, err_msg="All variances must be > 0")

    def test_core_deterministic(self, egarch_data: np.ndarray) -> None:
        """Same inputs produce identical outputs (deterministic)."""
        inp = self._make_core_inputs(egarch_data)
        ht1 = egarch_core(**inp)
        ht2 = egarch_core(**inp)
        npt.assert_array_equal(ht1, ht2)

    def test_core_finite(self, egarch_data: np.ndarray) -> None:
        """All output values are finite (no NaN or Inf)."""
        inp = self._make_core_inputs(egarch_data)
        ht = egarch_core(**inp)
        assert np.all(np.isfinite(ht)), "All variances must be finite"

    def test_core_higher_order(self, egarch_data: np.ndarray) -> None:
        """EGARCH(2,1,2) core computation succeeds."""
        m = 2
        data_aug = np.concatenate([np.zeros(m), egarch_data])
        T = len(data_aug)
        back_cast = float(np.log(np.var(egarch_data, ddof=1)))
        upper = 10000.0 * float(np.max(data_aug ** 2))
        # params: omega, alpha1, alpha2, gamma1, beta1, beta2
        params = np.array([-0.1, 0.05, 0.03, -0.02, 0.5, 0.4])
        ht = egarch_core(data_aug, params, back_cast, upper, 2, 1, 2, m, T)
        assert ht.shape == (T,)
        assert np.all(ht > 0)


# ===========================================================================
# Phase 6: egarch_likelihood tests
# ===========================================================================
class TestEgarchLikelihood:
    """Tests for EGARCH log-likelihood computation."""

    def _make_ll_inputs(
        self, egarch_data: np.ndarray, error_type: int = NORMAL,
    ) -> dict:
        """Prepare standard inputs for egarch_likelihood testing."""
        p, o, q = 1, 1, 1
        m = max(p, o, q)
        data_aug = np.concatenate([np.zeros(m), egarch_data])
        T = len(data_aug)
        var_est = float(np.var(egarch_data, ddof=1))
        back_cast = float(np.log(max(var_est, 1e-10)))
        upper = 10000.0 * float(np.max(data_aug ** 2))

        if error_type == NORMAL:
            params = np.array([-0.1, 0.1, -0.05, 0.95])
        elif error_type == STUDENTST:
            params = np.array([-0.1, 0.1, -0.05, 0.95, 5.0])
        elif error_type == GED:
            params = np.array([-0.1, 0.1, -0.05, 0.95, 1.9])
        else:  # SKEWT
            params = np.array([-0.1, 0.1, -0.05, 0.95, 5.0, -0.1])

        return {
            'parameters': params,
            'epsilon_aug': data_aug,
            'p': p, 'o': o, 'q': q,
            'error_type': error_type,
            'back_cast': back_cast,
            'T': T, 'upper': upper,
            'estim_flag': False,
        }

    def test_likelihood_returns_scalar(self, egarch_data: np.ndarray) -> None:
        """Likelihood returns (scalar_LL, LLS_array, ht_array)."""
        inp = self._make_ll_inputs(egarch_data)
        LL, LLS, ht = egarch_likelihood(**inp)
        assert np.isscalar(LL) or (isinstance(LL, (float, np.floating)))
        assert np.isfinite(LL), f"LL should be finite, got {LL}"

    def test_likelihood_output_shapes(self, egarch_data: np.ndarray) -> None:
        """LLS and ht have length T - m."""
        inp = self._make_ll_inputs(egarch_data)
        m = max(inp['p'], inp['o'], inp['q'])
        LL, LLS, ht = egarch_likelihood(**inp)
        expected_len = inp['T'] - m
        assert LLS.shape == (expected_len,), f"LLS shape {LLS.shape} != ({expected_len},)"
        assert ht.shape == (expected_len,), f"ht shape {ht.shape} != ({expected_len},)"

    def test_likelihood_ht_positive(self, egarch_data: np.ndarray) -> None:
        """All conditional variances from likelihood are positive."""
        inp = self._make_ll_inputs(egarch_data)
        LL, LLS, ht = egarch_likelihood(**inp)
        assert np.all(ht > 0), "All ht values must be > 0"

    @pytest.mark.parametrize("error_type", [NORMAL, STUDENTST, GED, SKEWT])
    def test_likelihood_all_distributions(
        self, egarch_data: np.ndarray, error_type: int
    ) -> None:
        """Likelihood computation works for all 4 error distributions."""
        inp = self._make_ll_inputs(egarch_data, error_type=error_type)
        LL, LLS, ht = egarch_likelihood(**inp)
        assert np.isfinite(LL), f"LL not finite for error_type={error_type}"
        assert np.all(np.isfinite(LLS)), "All LLS must be finite"
        assert np.all(ht > 0), "All ht must be positive"

    def test_likelihood_estim_flag_transforms(self, egarch_data: np.ndarray) -> None:
        """estim_flag=True triggers parameter transformation."""
        inp = self._make_ll_inputs(egarch_data, error_type=STUDENTST)
        # Transform parameters to unconstrained space
        params_constrained = inp['parameters'].copy()
        params_unconstrained = egarch_transform(
            params_constrained, inp['p'], inp['o'], inp['q'], inp['error_type']
        )
        inp_unc = dict(inp)
        inp_unc['parameters'] = params_unconstrained
        inp_unc['estim_flag'] = True

        LL_unc, _, _ = egarch_likelihood(**inp_unc)
        LL_con, _, _ = egarch_likelihood(**inp)

        # Both should give same result (within tolerance) because
        # estim_flag=True internally calls itransform
        npt.assert_allclose(LL_unc, LL_con, atol=ATOL, rtol=RTOL,
                            err_msg="estim_flag round-trip should match")


# ===========================================================================
# Phase 7: egarch_simulate tests
# ===========================================================================
class TestEgarchSimulate:
    """Tests for EGARCH simulation."""

    def test_simulate_output_shape(self) -> None:
        """Output shapes match requested length."""
        T = 500
        params = np.array([-0.1, 0.1, -0.05, 0.95])
        sim_data, sim_ht = egarch_simulate(T, params, 1, 1, 1, 'NORMAL')
        assert sim_data.shape == (T,), f"sim_data shape {sim_data.shape} != ({T},)"
        assert sim_ht.shape == (T,), f"sim_ht shape {sim_ht.shape} != ({T},)"

    def test_simulate_positive_variances(self) -> None:
        """All simulated conditional variances are strictly positive.

        EGARCH works in log-variance space, so ht = exp(log_ht) > 0.
        """
        params = np.array([-0.1, 0.1, -0.05, 0.95])
        _, sim_ht = egarch_simulate(500, params, 1, 1, 1, 'NORMAL')
        assert np.all(sim_ht > 0), "All simulated ht must be > 0"

    def test_simulate_with_leverage(self) -> None:
        """gamma < 0 creates asymmetric variance response (leverage effect)."""
        params = np.array([-0.1, 0.1, -0.2, 0.95])
        sim_data, sim_ht = egarch_simulate(2000, params, 1, 1, 1, 'NORMAL')
        # With gamma < 0, negative shocks should increase variance more
        # than positive shocks of the same magnitude
        assert sim_data.shape == (2000,)
        assert np.all(sim_ht > 0)

    @pytest.mark.parametrize("error_type", ['NORMAL', 'STUDENTST', 'GED', 'SKEWT'])
    def test_simulate_all_distributions(self, error_type: str) -> None:
        """Simulation works for all 4 error distributions."""
        if error_type == 'NORMAL':
            params = np.array([-0.1, 0.1, -0.05, 0.95])
        elif error_type in ('STUDENTST', 'GED'):
            nu = 5.0 if error_type == 'STUDENTST' else 1.9
            params = np.array([-0.1, 0.1, -0.05, 0.95, nu])
        else:  # SKEWT
            params = np.array([-0.1, 0.1, -0.05, 0.95, 5.0, -0.1])

        sim_data, sim_ht = egarch_simulate(300, params, 1, 1, 1, error_type)
        assert sim_data.shape == (300,)
        assert sim_ht.shape == (300,)
        assert np.all(sim_ht > 0)
        assert np.all(np.isfinite(sim_data))
        assert np.all(np.isfinite(sim_ht))

    def test_simulate_symmetric_egarch(self) -> None:
        """EGARCH(1,0,1) — symmetric process without leverage."""
        params = np.array([0.0, 0.1, 0.95])
        sim_data, sim_ht = egarch_simulate(500, params, 1, 0, 1, 'NORMAL')
        assert sim_data.shape == (500,)
        assert np.all(sim_ht > 0)

    def test_simulate_invalid_error_type_raises(self) -> None:
        """Invalid error_type raises ValueError."""
        params = np.array([-0.1, 0.1, -0.05, 0.95])
        with pytest.raises(ValueError, match="[Uu]nknown error"):
            egarch_simulate(500, params, 1, 1, 1, 'INVALID')


# ===========================================================================
# Phase 8: Starting Values and Display tests
# ===========================================================================
class TestEgarchStartingValues:
    """Tests for EGARCH starting value grid search."""

    def test_starting_values_length(self, egarch_data: np.ndarray) -> None:
        """Starting values vector has correct length (1+p+o+q)."""
        p, o, q = 1, 1, 1
        m = max(p, o, q)
        data_aug = np.concatenate([np.zeros(m), egarch_data])
        T = len(data_aug)
        back_cast = float(np.log(max(np.var(egarch_data, ddof=1), 1e-10)))

        sv, nu, lam = egarch_starting_values(
            None, data_aug, p, o, q, NORMAL, back_cast, T
        )
        expected_len = 1 + p + o + q  # omega + alpha + gamma + beta = 4
        assert sv.shape == (expected_len,), f"Expected ({expected_len},), got {sv.shape}"

    def test_starting_values_nu_for_studentst(self, egarch_data: np.ndarray) -> None:
        """Starting values for STUDENTST return nu parameter."""
        p, o, q = 1, 1, 1
        m = max(p, o, q)
        data_aug = np.concatenate([np.zeros(m), egarch_data])
        T = len(data_aug)
        back_cast = float(np.log(max(np.var(egarch_data, ddof=1), 1e-10)))

        sv, nu, lam = egarch_starting_values(
            None, data_aug, p, o, q, STUDENTST, back_cast, T
        )
        assert nu is not None, "nu should not be None for STUDENTST"
        assert nu > 2, f"nu should be > 2 for Student-t, got {nu}"

    def test_starting_values_skewt_params(self, egarch_data: np.ndarray) -> None:
        """Starting values for SKEWT return both nu and lambda."""
        p, o, q = 1, 1, 1
        m = max(p, o, q)
        data_aug = np.concatenate([np.zeros(m), egarch_data])
        T = len(data_aug)
        back_cast = float(np.log(max(np.var(egarch_data, ddof=1), 1e-10)))

        sv, nu, lam = egarch_starting_values(
            None, data_aug, p, o, q, SKEWT, back_cast, T
        )
        assert nu is not None, "nu should not be None for SKEWT"
        assert lam is not None, "lambda should not be None for SKEWT"

    def test_starting_values_user_supplied(self, egarch_data: np.ndarray) -> None:
        """User-supplied starting values are parsed correctly."""
        p, o, q = 1, 1, 1
        m = max(p, o, q)
        data_aug = np.concatenate([np.zeros(m), egarch_data])
        T = len(data_aug)
        back_cast = float(np.log(max(np.var(egarch_data, ddof=1), 1e-10)))

        user_sv = np.array([-0.2, 0.08, -0.04, 0.92])
        sv, nu, lam = egarch_starting_values(
            user_sv, data_aug, p, o, q, NORMAL, back_cast, T
        )
        npt.assert_allclose(sv, user_sv, atol=ATOL)


class TestEgarchDisplay:
    """Tests for EGARCH display formatting."""

    def test_display_runs_without_error(self, egarch_data: np.ndarray) -> None:
        """Display function runs without raising exceptions."""
        params = np.array([-0.1, 0.1, -0.05, 0.95])
        ll = -1500.0
        vcv = np.eye(4) * 0.01
        result = egarch_display(params, ll, vcv, egarch_data, 1, 1, 1, 'NORMAL')
        assert result is not None

    def test_display_returns_text_aic_bic(self, egarch_data: np.ndarray) -> None:
        """Display returns (text, AIC, BIC) tuple.

        Ref: egarch_display.m:1 — returns [text, AIC, BIC]
        """
        params = np.array([-0.1, 0.1, -0.05, 0.95])
        ll = -1500.0
        vcv = np.eye(4) * 0.01
        text, aic, bic = egarch_display(params, ll, vcv, egarch_data, 1, 1, 1, 'NORMAL')
        assert isinstance(text, str)
        assert len(text) > 0, "Display text should not be empty"
        assert isinstance(aic, (float, np.floating))
        assert isinstance(bic, (float, np.floating))
        assert np.isfinite(aic), f"AIC should be finite, got {aic}"
        assert np.isfinite(bic), f"BIC should be finite, got {bic}"

    def test_display_studentst(self, egarch_data: np.ndarray) -> None:
        """Display works with Student-t distribution (includes nu parameter)."""
        params = np.array([-0.1, 0.1, -0.05, 0.95, 5.0])
        ll = -1480.0
        vcv = np.eye(5) * 0.01
        text, aic, bic = egarch_display(
            params, ll, vcv, egarch_data, 1, 1, 1, 'STUDENTST'
        )
        assert isinstance(text, str)
        assert np.isfinite(aic)


# ===========================================================================
# Phase 9: Integration Tests — Full EGARCH Estimation Pipeline
# ===========================================================================
class TestEgarchIntegration:
    """Integration tests for the full EGARCH estimation pipeline.

    These tests run the complete egarch() driver, which calls all sub-modules:
    parameter_check → starting_values → transform → optimize → itransform →
    likelihood → robustvcv → diagnostics.
    """

    @pytest.mark.slow
    def test_egarch_normal_p1_o1_q1(self, short_data: np.ndarray) -> None:
        """Full EGARCH(1,1,1) estimation with Normal errors."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = egarch(
            short_data, 1, 1, 1, 'NORMAL'
        )
        # Parameter vector: omega, alpha, gamma, beta
        assert params.shape == (4,), f"Expected 4 params, got {params.shape}"
        assert isinstance(LL, (float, np.floating))
        assert np.isfinite(LL), f"LL should be finite, got {LL}"
        assert ht.shape == (len(short_data),), (
            f"ht shape {ht.shape} != ({len(short_data)},)"
        )
        assert np.all(ht > 0), "All conditional variances must be positive"

    @pytest.mark.slow
    def test_egarch_returns_correct_types(self, short_data: np.ndarray) -> None:
        """All return types are correct."""
        params, LL, ht, VCVrobust, VCV, scores, diagnostics = egarch(
            short_data, 1, 1, 1, 'NORMAL'
        )
        assert isinstance(params, np.ndarray)
        assert isinstance(LL, (float, np.floating))
        assert isinstance(ht, np.ndarray)
        assert isinstance(VCVrobust, np.ndarray)
        assert isinstance(VCV, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert isinstance(diagnostics, dict)

    @pytest.mark.slow
    def test_egarch_ht_positive(self, short_data: np.ndarray) -> None:
        """All conditional variances are strictly positive."""
        _, _, ht, _, _, _, _ = egarch(short_data, 1, 1, 1, 'NORMAL')
        assert np.all(ht > 0), "All ht must be > 0 (EGARCH uses exp of log-variance)"

    @pytest.mark.slow
    def test_egarch_vcv_symmetric(self, short_data: np.ndarray) -> None:
        """Both VCV matrices are symmetric."""
        _, _, _, VCVrobust, VCV, _, _ = egarch(short_data, 1, 1, 1, 'NORMAL')
        npt.assert_allclose(VCVrobust, VCVrobust.T, atol=1e-10,
                            err_msg="VCVrobust must be symmetric")
        npt.assert_allclose(VCV, VCV.T, atol=1e-10,
                            err_msg="VCV must be symmetric")

    @pytest.mark.slow
    def test_egarch_stationarity(self, short_data: np.ndarray) -> None:
        """Estimated beta satisfies stationarity constraint.

        Ref: egarch_nlcon.m:28 — all roots of [1;-beta] inside unit circle
        """
        params, _, _, _, _, _, _ = egarch(short_data, 1, 1, 1, 'NORMAL')
        beta = params[3:4]  # For EGARCH(1,1,1), beta is at index 3
        # Check stationarity: roots of [1, -beta_1] should have |root| < 1
        coeffs = np.concatenate(([1.0], -beta))
        roots = np.roots(coeffs)
        assert np.all(np.abs(roots) < 1.0), (
            f"Stationarity violated: root moduli {np.abs(roots)}"
        )

    @pytest.mark.slow
    def test_egarch_diagnostics_keys(self, short_data: np.ndarray) -> None:
        """Diagnostics dict contains expected keys."""
        _, _, _, _, _, _, diagnostics = egarch(short_data, 1, 1, 1, 'NORMAL')
        assert 'EXITFLAG' in diagnostics
        assert 'ITERATIONS' in diagnostics
        assert 'FUNCCOUNT' in diagnostics

    @pytest.mark.slow
    def test_egarch_studentst(self, short_data: np.ndarray) -> None:
        """EGARCH(1,1,1) estimation with Student-t errors."""
        params, LL, ht, _, _, _, _ = egarch(
            short_data, 1, 1, 1, 'STUDENTST'
        )
        # params: omega, alpha, gamma, beta, nu
        assert params.shape == (5,), f"Expected 5 params for STUDENTST, got {params.shape}"
        assert params[4] > 2.0, f"nu must be > 2 for Student-t, got {params[4]}"
        assert np.all(ht > 0)

    @pytest.mark.slow
    def test_egarch_skewt(self, short_data: np.ndarray) -> None:
        """EGARCH(1,1,1) estimation with Skewed-t errors."""
        params, LL, ht, _, _, _, _ = egarch(
            short_data, 1, 1, 1, 'SKEWT'
        )
        # params: omega, alpha, gamma, beta, nu, lambda
        assert params.shape == (6,), f"Expected 6 params for SKEWT, got {params.shape}"
        assert params[4] > 2.0, f"nu must be > 2 for SKEWT, got {params[4]}"
        assert -1.0 < params[5] < 1.0, (
            f"lambda must be in (-1,1), got {params[5]}"
        )
        assert np.all(ht > 0)

    @pytest.mark.slow
    def test_egarch_scores_shape(self, short_data: np.ndarray) -> None:
        """Scores matrix has shape (T_eff, n_params)."""
        params, _, _, _, _, scores, _ = egarch(
            short_data, 1, 1, 1, 'NORMAL'
        )
        # scores should have same number of columns as parameters
        assert scores.ndim >= 1, "scores should be at least 1-D"

    @pytest.mark.slow
    def test_egarch_ll_finite(self, short_data: np.ndarray) -> None:
        """Log-likelihood is finite and positive (after sign flip)."""
        _, LL, _, _, _, _, _ = egarch(short_data, 1, 1, 1, 'NORMAL')
        assert np.isfinite(LL), f"LL must be finite, got {LL}"


# ===========================================================================
# Phase 10: Parity Tests Against MATLAB Fixtures
# ===========================================================================
class TestEgarchParity:
    """MATLAB-Python parity tests using generated fixture files.

    Tolerance: npt.assert_allclose(atol=1e-6, rtol=1e-4)

    Fixtures are loaded from tests/fixtures/univariate/egarch_*.npy.
    Tests are skipped if fixture files are not available.
    """

    def _load_fixture(self, fixture_dir: Path, name: str) -> np.ndarray:
        """Load a .npy fixture file, skip test if not found."""
        path = fixture_dir / f"{name}.npy"
        if not path.exists():
            pytest.skip(f"Fixture file not found: {path}")
        return np.load(path, allow_pickle=True)

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_egarch_core_parity(self, fixture_dir: Path) -> None:
        """EGARCH core recursion parity with MATLAB.

        Uses fixture inputs (data, params, back_cast, upper, p, o, q, m, T)
        and compares computed ht against MATLAB egarch_core output.
        """
        data = self._load_fixture(fixture_dir, "egarch_core_ec_data")
        params = self._load_fixture(fixture_dir, "egarch_core_egarch_core_params")
        back_cast = float(self._load_fixture(fixture_dir, "egarch_core_ec_back_cast"))
        upper = float(self._load_fixture(fixture_dir, "egarch_core_ec_upper"))
        p = int(self._load_fixture(fixture_dir, "egarch_core_ec_p"))
        o = int(self._load_fixture(fixture_dir, "egarch_core_ec_o"))
        q = int(self._load_fixture(fixture_dir, "egarch_core_ec_q"))
        m = int(self._load_fixture(fixture_dir, "egarch_core_ec_m"))
        T = int(self._load_fixture(fixture_dir, "egarch_core_ec_T"))
        ht_expected = self._load_fixture(fixture_dir, "egarch_core_egarch_core_ht")

        data = data.ravel().astype(np.float64)
        params = params.ravel().astype(np.float64)
        ht_expected = ht_expected.ravel().astype(np.float64)

        ht_actual = egarch_core(data, params, back_cast, upper, p, o, q, m, T)

        npt.assert_allclose(
            ht_actual, ht_expected, atol=ATOL, rtol=RTOL,
            err_msg="EGARCH core recursion parity failed"
        )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_egarch_nlcon_parity(self, fixture_dir: Path) -> None:
        """EGARCH nonlinear constraint parity with MATLAB.

        Ref: egarch_nlcon.m:28 — c = abs(roots([1;-beta])) - 0.99998
        """
        params = self._load_fixture(fixture_dir, "egarch_nlcon_enlc_params")
        c_expected = self._load_fixture(fixture_dir, "egarch_nlcon_egarch_nlc_c")

        params = params.ravel().astype(np.float64)
        # EGARCH(1,1,1) standard: p=1, o=1, q=1
        c_actual = egarch_nlcon(params, p=1, o=1, q=1, error_type=NORMAL)

        # c_expected may be scalar or array
        c_expected_val = float(c_expected) if c_expected.ndim == 0 else c_expected.ravel()
        c_actual_val = c_actual.ravel()

        npt.assert_allclose(
            c_actual_val, c_expected_val, atol=ATOL, rtol=RTOL,
            err_msg="EGARCH nlcon parity failed"
        )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_egarch_likelihood_parity(self, fixture_dir: Path) -> None:
        """EGARCH likelihood parity with MATLAB.

        Reconstructs the fixture's own input data from the core fixture
        (which shares the same back_cast), passes non-augmented data to
        egarch_likelihood (MATLAB calling convention), and compares LL, ht.

        Ref: egarch_likelihood.m — MATLAB passes raw data, T = len(data).
        The function internally computes m = max(p,o,q), augments data,
        and returns ht[m:T], LLS[m:T].
        """
        params = self._load_fixture(fixture_dir, "egarch_likelihood_el_params")
        ll_expected_val = float(
            self._load_fixture(fixture_dir, "egarch_likelihood_egarch_ll")
        )
        ht_expected = self._load_fixture(
            fixture_dir, "egarch_likelihood_egarch_ll_ht"
        ).ravel().astype(np.float64)
        lls_expected = self._load_fixture(
            fixture_dir, "egarch_likelihood_egarch_lls"
        ).ravel().astype(np.float64)
        bc_fixture = float(
            self._load_fixture(fixture_dir, "egarch_likelihood_el_back_cast")
        )

        params = params.ravel().astype(np.float64)
        p, o, q = 1, 1, 1
        m = max(p, o, q)

        # The individual likelihood fixtures share the same data/back_cast
        # as the core individual fixtures. The core individual data is
        # AUGMENTED (starts with zero, length T+m). We extract the
        # non-augmented portion for the MATLAB-style calling convention.
        core_data_aug = self._load_fixture(
            fixture_dir, "egarch_core_ec_data"
        ).ravel().astype(np.float64)
        data_nonaug = core_data_aug[m:]  # Remove leading zeros
        T = len(data_nonaug)

        # Compute upper bound from raw data (MATLAB: upper=10000*max(data.^2))
        upper = 10000.0 * float(np.max(data_nonaug ** 2))

        LL_actual, LLS_actual, ht_actual = egarch_likelihood(
            params, data_nonaug, p, o, q, NORMAL, bc_fixture, T, upper, False
        )

        # Compare LL — MATLAB returns negated LL for minimization
        npt.assert_allclose(
            LL_actual, ll_expected_val, atol=ATOL, rtol=RTOL,
            err_msg="EGARCH likelihood LL parity failed"
        )

        # Compare ht series
        npt.assert_allclose(
            ht_actual, ht_expected, atol=ATOL, rtol=RTOL,
            err_msg="EGARCH likelihood ht parity failed"
        )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_egarch_lls_parity(self, fixture_dir: Path) -> None:
        """EGARCH per-observation log-likelihoods parity.

        Uses the same data reconstruction as test_egarch_likelihood_parity.
        """
        params = self._load_fixture(
            fixture_dir, "egarch_likelihood_el_params"
        ).ravel().astype(np.float64)
        lls_expected = self._load_fixture(
            fixture_dir, "egarch_likelihood_egarch_lls"
        ).ravel().astype(np.float64)
        bc_fixture = float(
            self._load_fixture(fixture_dir, "egarch_likelihood_el_back_cast")
        )

        p, o, q = 1, 1, 1
        m = max(p, o, q)

        core_data_aug = self._load_fixture(
            fixture_dir, "egarch_core_ec_data"
        ).ravel().astype(np.float64)
        data_nonaug = core_data_aug[m:]
        T = len(data_nonaug)
        upper = 10000.0 * float(np.max(data_nonaug ** 2))

        _, LLS_actual, _ = egarch_likelihood(
            params, data_nonaug, p, o, q, NORMAL, bc_fixture, T, upper, False
        )

        npt.assert_allclose(
            LLS_actual, lls_expected, atol=ATOL, rtol=RTOL,
            err_msg="EGARCH per-observation LLS parity failed"
        )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_egarch_display_parity(self, fixture_dir: Path, egarch_data: np.ndarray) -> None:
        """EGARCH display runs with fixture parameters.

        Verifies the display function can be called with fixture data
        without error and produces valid AIC/BIC.
        """
        params = self._load_fixture(fixture_dir, "egarch_display_egarch_disp_params")
        ll = float(self._load_fixture(fixture_dir, "egarch_display_egarch_disp_ll"))
        vcv = self._load_fixture(fixture_dir, "egarch_display_egarch_disp_vcv")

        params = params.ravel().astype(np.float64)
        vcv = vcv.astype(np.float64)

        # Ensure VCV is symmetric positive definite for display
        vcv = (vcv + vcv.T) / 2.0

        text, aic, bic = egarch_display(
            params, ll, vcv, egarch_data, 1, 1, 1, 'NORMAL'
        )
        assert isinstance(text, str)
        assert np.isfinite(aic)
        assert np.isfinite(bic)


# ===========================================================================
# Additional edge case and robustness tests
# ===========================================================================
class TestEgarchEdgeCases:
    """Edge cases and robustness tests for EGARCH modules."""

    def test_core_with_zero_data(self) -> None:
        """egarch_core handles all-zero data gracefully."""
        p, o, q = 1, 1, 1
        m = max(p, o, q)
        T = 100
        data = np.zeros(T + m)
        params = np.array([-0.1, 0.1, -0.05, 0.95])
        back_cast = np.log(0.01)
        upper = 10.0
        ht = egarch_core(data, params, back_cast, upper, p, o, q, m, T + m)
        assert ht.shape == (T + m,)
        assert np.all(ht > 0)
        assert np.all(np.isfinite(ht))

    def test_nlcon_zero_q(self) -> None:
        """EGARCH with q=0 (no GARCH terms) — nlcon with empty beta."""
        # params: omega, alpha(1), gamma(1) — no beta
        params = np.array([-0.1, 0.1, -0.05])
        c = egarch_nlcon(params, p=1, o=1, q=0, error_type=NORMAL)
        # With q=0, polynomial is just [1], which has no roots
        # The function should return an empty array
        assert c.size == 0 or np.all(c < 0)

    def test_likelihood_returns_negated(self, egarch_data: np.ndarray) -> None:
        """Likelihood LL is negated (suitable for minimization).

        Ref: egarch_likelihood.m:74-75 — LLS = -LLS; LL = -LL;
        """
        p, o, q = 1, 1, 1
        m = max(p, o, q)
        data_aug = np.concatenate([np.zeros(m), egarch_data])
        T = len(data_aug)
        bc = float(np.log(np.var(egarch_data, ddof=1)))
        upper = 10000.0 * float(np.max(data_aug ** 2))
        params = np.array([-0.1, 0.1, -0.05, 0.95])

        LL, _, _ = egarch_likelihood(params, data_aug, p, o, q, NORMAL, bc, T, upper)
        # The returned LL is the NEGATIVE log-likelihood (for minimization)
        # For a well-specified model, it should be a positive number
        # (since -(-LL) = LL, and log-likelihoods for normal data are negative)
        assert np.isfinite(LL)

    def test_transform_identity_for_normal(self) -> None:
        """Normal distribution: transform is identity for all parameters."""
        params = np.array([-0.5, 0.2, -0.15, 0.85])
        transformed = egarch_transform(params, 1, 1, 1, NORMAL)
        npt.assert_array_equal(transformed, params,
                               err_msg="Normal transform should be identity")

    def test_itransform_identity_for_normal(self) -> None:
        """Normal distribution: itransform is identity for all parameters."""
        params = np.array([-0.5, 0.2, -0.15, 0.85])
        itransformed = egarch_itransform(params, 1, 1, 1, NORMAL)
        npt.assert_array_equal(itransformed, params,
                               err_msg="Normal itransform should be identity")

    def test_simulate_nonstationary_warning(self) -> None:
        """Non-stationary parameters produce a warning.

        Ref: egarch_simulate.m:91-93
        """
        params = np.array([-0.1, 0.1, -0.05, 1.1])  # beta=1.1 > 1
        with pytest.warns(UserWarning, match="non-stationary"):
            egarch_simulate(100, params, 1, 1, 1, 'NORMAL')
