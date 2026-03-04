"""
Pytest tests for ``mfe_toolbox.timeseries.armaxfilter_likelihood``.

Tests validate the ARMAX Gaussian log-likelihood computation migrated from
``timeseries/armaxfilter_likelihood.m`` (Kevin Sheppard, Revision 3, 4/1/2004).

The function under test computes the negative Gaussian log-likelihood for
ARMAX(p, q) models and is consumed by ``scipy.optimize.minimize`` during
maximum-likelihood estimation.

Parity contract per AAP Section 0.7.1:
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)

All test inputs use numpy arrays with deterministic random seeds
(``np.random.default_rng`` with fixed seed) for reproducibility.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.armaxfilter_likelihood import armaxfilter_likelihood

# ---------------------------------------------------------------------------
# Tolerance constants — matches tests/conftest.py (AAP Section 0.7.1)
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Helper: build a simple AR(1) dataset with known coefficient
# ---------------------------------------------------------------------------
def _make_ar1_data(rng, T, phi=0.5):
    """Generate AR(1) data: y[t] = phi * y[t-1] + eps[t]."""
    y = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


# ---------------------------------------------------------------------------
# Helper: load fixture dict from .npy file, skipping if absent
# ---------------------------------------------------------------------------
def _load_fixture_dict(fixture_dir, name):
    """Load a pickled dict-of-dicts fixture, skip test if file missing."""
    path = fixture_dir / f"{name}.npy"
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")
    return np.load(path, allow_pickle=True).item()


# ---------------------------------------------------------------------------
# Helper: infer number of exogenous regressors from parameter vector layout
# ---------------------------------------------------------------------------
def _infer_num_exog(parameters, p, q, constant):
    """Return k = number of exogenous regressors from the parameter count."""
    n_params = len(np.asarray(parameters).ravel())
    n_ar = len(np.asarray(p).ravel())
    n_ma = len(np.asarray(q).ravel())
    return n_params - int(constant) - n_ar - n_ma


# ---------------------------------------------------------------------------
# Helper: ensure x has correct 2-D shape for the number of exogenous vars
# ---------------------------------------------------------------------------
def _prepare_x(x, T, num_exog):
    """Reshape x to (T, k) if it is 1-D and k > 0 so armaxerrors sees it."""
    x = np.asarray(x, dtype=np.float64)
    if num_exog > 0 and x.ndim == 1:
        # Fixture stores MATLAB column vector as 1-D; reshape for Python
        x = x.reshape(T, num_exog)
    return x


# ===================================================================
# Test 1: Return type structure — three-element tuple
# ===================================================================
def test_armaxfilter_likelihood_returns_three():
    """Verify armaxfilter_likelihood returns a tuple of exactly 3 elements.

    Uses a simple AR(1) model with deterministic seed to confirm the
    function interface: (LLF, likelihoods, errors).
    Ref: armaxfilter_likelihood.m returns [LLF, likelihoods, errors].
    """
    rng = np.random.default_rng(42)
    T = 200
    y = _make_ar1_data(rng, T, phi=0.5)
    sigma = np.ones(T, dtype=np.float64)
    # AR(1) with constant: params = [constant_val, AR_coeff]
    params = np.array([0.0, 0.5])

    result = armaxfilter_likelihood(
        params, np.array([1]), np.array([]), 1, y,
        np.empty((T, 0)), 1, sigma,
    )

    assert isinstance(result, tuple), "Return value must be a tuple"
    assert len(result) == 3, f"Expected 3 elements, got {len(result)}"


# ===================================================================
# Test 2: LLF is a scalar
# ===================================================================
def test_armaxfilter_likelihood_llf_scalar():
    """Verify that the first return value (LLF) is a scalar float.

    The negated log-likelihood must be a single number suitable for
    scipy.optimize.minimize consumption.
    """
    rng = np.random.default_rng(42)
    T = 200
    y = _make_ar1_data(rng, T, phi=0.5)
    sigma = np.ones(T, dtype=np.float64)
    params = np.array([0.0, 0.5])

    llf, _, _ = armaxfilter_likelihood(
        params, np.array([1]), np.array([]), 1, y,
        np.empty((T, 0)), 1, sigma,
    )

    assert np.isscalar(llf) or isinstance(llf, (float, np.floating)), (
        f"LLF must be scalar, got type {type(llf)}"
    )


# ===================================================================
# Test 3: Likelihoods array shape
# ===================================================================
def test_armaxfilter_likelihood_likelihoods_shape():
    """Verify likelihoods array length matches effective observation count.

    For AR(1) with p=[1], q=[] (p_max=1, q_max=0), since q_max is NOT
    greater than p_max, all T observations are included.
    Ref: armaxfilter_likelihood.m lines 41-45.
    """
    rng = np.random.default_rng(42)
    T = 200
    y = _make_ar1_data(rng, T, phi=0.5)
    sigma = np.ones(T, dtype=np.float64)
    params = np.array([0.0, 0.5])

    _, likelihoods, _ = armaxfilter_likelihood(
        params, np.array([1]), np.array([]), 1, y,
        np.empty((T, 0)), 1, sigma,
    )

    # q_max=0 <= p_max=1 → t_slice = slice(0, T), no trimming
    assert likelihoods.shape == (T,), (
        f"Expected likelihoods length {T}, got {likelihoods.shape}"
    )


# ===================================================================
# Test 4: Errors array shape matches likelihoods
# ===================================================================
def test_armaxfilter_likelihood_errors_shape():
    """Verify that errors and likelihoods arrays have identical length.

    Both are trimmed by the same t_slice in the implementation.
    Ref: armaxfilter_likelihood.m lines 53-54.
    """
    rng = np.random.default_rng(42)
    T = 200
    y = _make_ar1_data(rng, T, phi=0.5)
    sigma = np.ones(T, dtype=np.float64)
    params = np.array([0.0, 0.5])

    _, likelihoods, errors = armaxfilter_likelihood(
        params, np.array([1]), np.array([]), 1, y,
        np.empty((T, 0)), 1, sigma,
    )

    assert errors.shape == likelihoods.shape, (
        f"Errors shape {errors.shape} != likelihoods shape {likelihoods.shape}"
    )


# ===================================================================
# Test 5: LLF is positive (negated log-likelihood for minimization)
# ===================================================================
def test_armaxfilter_likelihood_positive_llf():
    """Verify that LLF (negated log-likelihood) is positive.

    For normally distributed data with reasonable model parameters, the
    sum of per-observation negative log-likelihoods should be positive.
    """
    rng = np.random.default_rng(42)
    T = 200
    y = _make_ar1_data(rng, T, phi=0.5)
    sigma = np.ones(T, dtype=np.float64)
    params = np.array([0.0, 0.5])

    llf, _, _ = armaxfilter_likelihood(
        params, np.array([1]), np.array([]), 1, y,
        np.empty((T, 0)), 1, sigma,
    )

    assert llf > 0, f"Negated log-likelihood should be positive, got {llf}"


# ===================================================================
# Test 6: LLF equals sum of per-observation likelihoods
# ===================================================================
def test_armaxfilter_likelihood_llf_is_sum_of_likelihoods():
    """Verify LLF ≈ sum(likelihoods) per MATLAB line 55.

    The fundamental identity LLF = sum(likelihoods) must hold to within
    numerical tolerance.  This validates the aggregation step of
    armaxfilter_likelihood.m:55 — ``LLF = sum(likelihoods)``.
    """
    rng = np.random.default_rng(42)
    T = 200
    y = _make_ar1_data(rng, T, phi=0.5)
    sigma = np.ones(T, dtype=np.float64)
    params = np.array([0.0, 0.5])

    llf, likelihoods, _ = armaxfilter_likelihood(
        params, np.array([1]), np.array([]), 1, y,
        np.empty((T, 0)), 1, sigma,
    )

    npt.assert_allclose(
        llf, np.sum(likelihoods), atol=ATOL, rtol=RTOL,
        err_msg="LLF must equal sum(likelihoods)",
    )


# ===================================================================
# Test 7: Better parameters yield lower LLF
# ===================================================================
def test_armaxfilter_likelihood_better_params_lower_llf():
    """Verify that true AR(1) parameters yield lower LLF than zero params.

    Data is generated from y[t] = 0.5 * y[t-1] + eps[t].  Fitting with
    the true coefficient (0.5) should yield a lower (better) negative
    log-likelihood than fitting with AR coefficient = 0.0.
    """
    rng = np.random.default_rng(42)
    T = 300
    y = _make_ar1_data(rng, T, phi=0.5)
    sigma = np.ones(T, dtype=np.float64)

    # True parameters: constant=0.0, AR(1)=0.5  (constant flag = 1)
    llf_true, _, _ = armaxfilter_likelihood(
        np.array([0.0, 0.5]), np.array([1]), np.array([]), 1, y,
        np.empty((T, 0)), 2, sigma,
    )

    # Bad parameters: constant=0.0, AR(1)=0.0  (no AR structure)
    llf_bad, _, _ = armaxfilter_likelihood(
        np.array([0.0, 0.0]), np.array([1]), np.array([]), 1, y,
        np.empty((T, 0)), 2, sigma,
    )

    assert llf_true < llf_bad, (
        f"True params LLF ({llf_true:.4f}) should be < bad params LLF ({llf_bad:.4f})"
    )


# ===================================================================
# Test 8: Unit sigma — 2*log(sigma) term vanishes
# ===================================================================
def test_armaxfilter_likelihood_unit_sigma():
    """Verify that with sigma=ones the 2*log(sigma) term vanishes.

    When sigma = 1 everywhere, log(sigma) = 0.  The per-observation
    likelihood reduces to:
        0.5 * (log(sigma2) + stde^2/sigma2 + log(2*pi))
    where stde = errors (since sigma = 1) and sigma2 is the sample
    variance of the standardised errors over the effective sample.
    """
    rng = np.random.default_rng(42)
    T = 200
    y = rng.standard_normal(T)
    sigma = np.ones(T, dtype=np.float64)
    # Constant-only model: no AR, no MA
    params = np.array([0.0])
    p = np.array([], dtype=np.float64)
    q = np.array([], dtype=np.float64)

    llf, likelihoods, errors = armaxfilter_likelihood(
        params, p, q, 1, y, np.empty((T, 0)), 0, sigma,
    )

    # With sigma=ones, stde = errors and sigma2 = mean(errors^2)
    # Ref: armaxfilter_likelihood.m:47-48
    sigma2_hat = np.dot(errors, errors) / len(errors)
    expected_lls = 0.5 * (
        np.log(sigma2_hat)
        + errors ** 2 / sigma2_hat
        + np.log(2.0 * np.pi)
    )

    npt.assert_allclose(
        likelihoods, expected_lls, atol=ATOL, rtol=RTOL,
        err_msg="With unit sigma, likelihoods should have no log-sigma term",
    )


# ===================================================================
# Test 9: Heteroskedastic (non-constant varying) sigma changes LLF
# ===================================================================
def test_armaxfilter_likelihood_heteroskedastic():
    """Verify LLF changes when using genuinely heteroskedastic sigma.

    With non-constant conditional standard deviations the standardisation
    and variance estimation produce different per-observation likelihoods
    compared to unit sigma.

    Note: a *constant* sigma scaling (e.g. 2*ones) is algebraically
    absorbed by the sigma2 estimation and yields the same LLF as unit
    sigma.  Only genuinely *varying* sigma produces different results.
    """
    rng = np.random.default_rng(42)
    T = 200
    y = rng.standard_normal(T)
    sigma_unit = np.ones(T, dtype=np.float64)
    # Create genuinely varying sigma (not constant across observations)
    sigma_hetero = 0.5 + np.abs(0.3 * rng.standard_normal(T))

    params = np.array([0.0])
    p = np.array([], dtype=np.float64)
    q = np.array([], dtype=np.float64)

    llf_unit, _, _ = armaxfilter_likelihood(
        params, p, q, 1, y, np.empty((T, 0)), 0, sigma_unit,
    )
    llf_hetero, _, _ = armaxfilter_likelihood(
        params, p, q, 1, y, np.empty((T, 0)), 0, sigma_hetero,
    )

    # With varying sigma, LLFs must differ
    assert llf_unit != llf_hetero, (
        "LLF should change when sigma varies across observations"
    )

    # Additional: verify the 2*log(sigma) contribution is non-zero for
    # non-unit sigma.  For hetero sigma, some elements are > 1 (adding to
    # LLF) and some < 1 (subtracting).  The net effect depends on the
    # interplay with sigma2 estimation but the LLFs must not be equal.
    assert not np.isclose(llf_unit, llf_hetero, atol=1e-10), (
        "LLFs should differ significantly for heteroskedastic sigma"
    )


# ===================================================================
# Test 10: NaN protection — LLF returns 1e7 when NaN arises
# ===================================================================
def test_armaxfilter_likelihood_nan_protection():
    """Verify NaN in the likelihood computation yields LLF = 1e7.

    Per armaxfilter_likelihood.m lines 57-59:
        if isnan(LLF), LLF = 1e7; end

    This safeguards the optimizer from receiving NaN objective values.
    Sigma containing zeros triggers division by zero → inf → NaN chain.
    """
    T = 100
    y = np.ones(T, dtype=np.float64)
    # Sigma with zeros causes division by zero → NaN in likelihood
    sigma = np.zeros(T, dtype=np.float64)
    params = np.array([0.0])
    p = np.array([], dtype=np.float64)
    q = np.array([], dtype=np.float64)

    # Suppress runtime warnings for the expected div-by-zero / invalid ops
    with np.errstate(divide="ignore", invalid="ignore"):
        llf, _, _ = armaxfilter_likelihood(
            params, p, q, 1, y, np.empty((T, 0)), 0, sigma,
        )

    assert llf == 1e7, (
        f"NaN-protected LLF should be 1e7, got {llf}"
    )


# ===================================================================
# Test 11: MA burn-in trimming when max(q) > max(p)
# ===================================================================
def test_armaxfilter_likelihood_ma_burnin_trimming():
    """Verify observation trimming when MA order exceeds AR order.

    When max(q) > max(p) the implementation restricts the effective
    sample to observations starting at (max(q) - max(p)).
    Ref: armaxfilter_likelihood.m lines 41-45.

    For MA(2) with no AR: q_max=2, p_max=0 → trim first 2 observations.
    """
    rng = np.random.default_rng(42)
    T = 200
    y = rng.standard_normal(T)
    sigma = np.ones(T, dtype=np.float64)

    # MA(2) model with constant: params = [constant, MA_1, MA_2]
    params = np.array([0.0, 0.1, 0.05])
    p = np.array([], dtype=np.float64)  # no AR terms
    q = np.array([1, 2], dtype=np.float64)  # MA lags at positions 1 and 2
    m = 2  # burn-in for MA(2) recursion safety

    llf, likelihoods, errors = armaxfilter_likelihood(
        params, p, q, 1, y, np.empty((T, 0)), m, sigma,
    )

    # q_max=2, p_max=0 → trim = 2, effective length = T - 2
    expected_len = T - 2
    assert len(likelihoods) == expected_len, (
        f"Expected {expected_len} likelihoods after MA trimming, got {len(likelihoods)}"
    )
    assert len(errors) == expected_len, (
        f"Expected {expected_len} errors after MA trimming, got {len(errors)}"
    )

    # LLF = sum(likelihoods) must still hold
    npt.assert_allclose(
        llf, np.sum(likelihoods), atol=ATOL, rtol=RTOL,
        err_msg="LLF must equal sum(likelihoods) even with MA trimming",
    )


# ===================================================================
# Test 12: Constant-only (white noise) model — analytical verification
# ===================================================================
def test_armaxfilter_likelihood_white_noise_analytical():
    """Verify LLF for a constant-only model matches analytical Gaussian NLL.

    A constant-only model with params=[mu] and sigma=ones produces:
        errors = y - mu
        sigma2 = mean(errors^2)
        LLF = (T/2) * (log(sigma2) + 1 + log(2*pi))

    because sum(errors^2/sigma2) = T and 2*log(sigma) = 0 for unit sigma.
    """
    rng = np.random.default_rng(42)
    T = 500
    y = rng.standard_normal(T) + 1.0  # Mean ~ 1.0
    sigma = np.ones(T, dtype=np.float64)
    mu = 1.0
    params = np.array([mu])

    llf, likelihoods, errors = armaxfilter_likelihood(
        params, np.array([]), np.array([]), 1, y,
        np.empty((T, 0)), 0, sigma,
    )

    # Analytical: errors = y - mu, sigma2 = var(errors, ddof=0)
    expected_errors = y - mu
    sigma2_ana = np.mean(expected_errors ** 2)

    # LLF_analytical = (T/2) * (log(sigma2) + 1 + log(2*pi))
    # because sum(e^2/sigma2) / T = 1, so each obs contributes (1 + log(sigma2) + log(2pi))/2
    llf_analytical = 0.5 * T * (np.log(sigma2_ana) + 1.0 + np.log(2.0 * np.pi))

    npt.assert_allclose(
        llf, llf_analytical, atol=ATOL, rtol=RTOL,
        err_msg="White-noise LLF should match analytical Gaussian NLL",
    )

    # Verify individual errors match y - mu
    npt.assert_allclose(
        errors, expected_errors, atol=ATOL, rtol=RTOL,
        err_msg="Constant-only errors should equal y - mu",
    )


# ===================================================================
# Test 13: Fixture parity — parametrized over MATLAB scenarios
# ===================================================================
@pytest.mark.parity
@pytest.mark.parametrize(
    "scenario_key",
    ["arma11", "ar2", "armax11", "gls_sigma"],
)
def test_armaxfilter_likelihood_parity(timeseries_fixture_dir, scenario_key):
    """Parity test: compare Python output against MATLAB reference fixtures.

    Each scenario was generated in MATLAB/Octave with rng(42, 'twister')
    and saved to tests/fixtures/timeseries/armaxfilter_likelihood.npy.

    Scenarios:
        arma11   — ARMA(1,1) with constant, unit sigma
        ar2      — AR(2) with constant, unit sigma
        armax11  — ARMAX(1,1) with 1 exogenous variable, unit sigma
        gls_sigma — ARMA(1,1) with constant, heteroskedastic sigma

    Per AAP Section 0.7.1:
        atol=1e-6, rtol=1e-4
    """
    fixture = _load_fixture_dict(timeseries_fixture_dir, "armaxfilter_likelihood")
    scenario = fixture[scenario_key]

    # Extract inputs
    parameters = np.asarray(scenario["parameters"], dtype=np.float64)
    p = np.asarray(scenario["p"], dtype=np.float64)
    q = np.asarray(scenario["q"], dtype=np.float64)
    constant = int(scenario["constant"])
    y = np.asarray(scenario["y"], dtype=np.float64)
    sigma = np.asarray(scenario["sigma"], dtype=np.float64)
    m = int(scenario["m"])
    T = len(y)

    # Handle x: fixture may store MATLAB column vector as 1-D numpy array
    # Ref: armaxerrors.py treats 1-D x as k=0 (no exog); reshape if needed
    num_exog = _infer_num_exog(parameters, p, q, constant)
    x = _prepare_x(scenario["x"], T, num_exog)

    # Expected outputs from MATLAB
    expected_llf = float(scenario["LLF"])
    expected_likelihoods = np.asarray(scenario["likelihoods"], dtype=np.float64)
    expected_errors = np.asarray(scenario["errors"], dtype=np.float64)

    # Run Python implementation
    llf, likelihoods, errors = armaxfilter_likelihood(
        parameters, p, q, constant, y, x, m, sigma,
    )

    # Assert parity
    npt.assert_allclose(
        llf, expected_llf, atol=ATOL, rtol=RTOL,
        err_msg=f"[{scenario_key}] LLF mismatch",
    )
    npt.assert_allclose(
        likelihoods, expected_likelihoods, atol=ATOL, rtol=RTOL,
        err_msg=f"[{scenario_key}] likelihoods mismatch",
    )
    npt.assert_allclose(
        errors, expected_errors, atol=ATOL, rtol=RTOL,
        err_msg=f"[{scenario_key}] errors mismatch",
    )
