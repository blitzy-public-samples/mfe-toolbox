"""
Pytest tests for the Beveridge-Nelson decomposition.

Tests ``mfe_toolbox.timeseries.beveridgenelson.beveridgenelson`` which
decomposes an I(1) time series into permanent (trend) and transitory
(cyclic) components following the Beveridge-Nelson (1981) method.

Key properties validated:
- Additive decomposition identity: trend + cyclic == y  (always holds)
- Pure random walk ⇒ entire series is trend, cyclic ≈ 0
- ARMA structure in differences ⇒ non-trivial cyclical component
- Trend is integrated / non-stationary
- Cyclic component is bounded-variance / stationary

Ref: beveridgenelson.m — MATLAB MFE Toolbox Version 4.0
"""
import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.beveridgenelson import beveridgenelson

# ---------------------------------------------------------------------------
# Module-level tolerance constants matching conftest.py
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Helper: reproducible RNG
# ---------------------------------------------------------------------------
def _make_rng(seed: int = 42) -> np.random.Generator:
    """Return a seeded random generator for reproducible test data."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Helper: create synthetic I(1) series
# ---------------------------------------------------------------------------
def _make_random_walk(T: int = 500, seed: int = 42) -> np.ndarray:
    """Generate a T-length random walk (cumulative sum of N(0,1)).

    Parameters
    ----------
    T : int
        Length of the series.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` random walk series.
    """
    rng = _make_rng(seed)
    eps = rng.standard_normal(T)
    return np.cumsum(eps)


# ---------------------------------------------------------------------------
# Test 1: Return type — tuple of two elements
# ---------------------------------------------------------------------------
def test_beveridgenelson_returns_two() -> None:
    """``beveridgenelson`` must return a tuple of exactly two arrays.

    Ref: beveridgenelson.m:1-5 — function [trend, cyclic] = beveridgenelson(…)
    """
    y = _make_random_walk(T=200, seed=42)
    # Simple AR(1) model on differences with constant
    # Ref: beveridgenelson.m:48 — parameters = ARMA params for Δy
    params = np.array([0.0, 0.2])  # constant=0.0, AR(1) coeff=0.2
    result = beveridgenelson(y, params, constant=1, p=np.array([1]), q=None)

    assert isinstance(result, tuple), "Return value must be a tuple"
    assert len(result) == 2, "Tuple must contain exactly two elements"
    assert isinstance(result[0], np.ndarray), "trend must be np.ndarray"
    assert isinstance(result[1], np.ndarray), "cyclic must be np.ndarray"


# ---------------------------------------------------------------------------
# Test 2: Output shapes — both T-length 1D arrays
# ---------------------------------------------------------------------------
def test_beveridgenelson_output_shapes() -> None:
    """Both trend and cyclic must be 1D arrays with the same length as y.

    Ref: beveridgenelson.m:135-136 — trend(t) and cyclic(t) are T×1 vectors
    """
    T = 300
    y = _make_random_walk(T=T, seed=99)
    params = np.array([0.01, 0.25])  # [constant, AR(1)]
    trend, cyclic = beveridgenelson(y, params, constant=1, p=np.array([1]), q=None)

    assert trend.ndim == 1, f"trend must be 1D, got {trend.ndim}D"
    assert cyclic.ndim == 1, f"cyclic must be 1D, got {cyclic.ndim}D"
    assert trend.shape[0] == T, f"trend length {trend.shape[0]} != T={T}"
    assert cyclic.shape[0] == T, f"cyclic length {cyclic.shape[0]} != T={T}"


# ---------------------------------------------------------------------------
# Test 3: Additive decomposition identity — trend + cyclic = y
# ---------------------------------------------------------------------------
def test_beveridgenelson_additive_decomposition() -> None:
    """The fundamental BN identity: trend + cyclic == y must hold to tolerance.

    Ref: beveridgenelson.m:140 — cyclic(t) = y(t) - trend(t)
    This means trend(t) + cyclic(t) = y(t) for all t.
    """
    y = _make_random_walk(T=500, seed=42)
    # AR(1) on differences
    params = np.array([0.05, 0.3])
    trend, cyclic = beveridgenelson(y, params, constant=1, p=np.array([1]), q=None)

    # Ref: beveridgenelson.m:156 — cyclic(t) = y(t) - trend(t) always
    npt.assert_allclose(
        trend + cyclic,
        y,
        atol=ATOL,
        rtol=RTOL,
        err_msg="Additive decomposition identity trend + cyclic = y failed",
    )


# ---------------------------------------------------------------------------
# Test 4: Pure random walk — trend equals y, cyclic equals zero
# ---------------------------------------------------------------------------
def test_beveridgenelson_random_walk() -> None:
    """For a pure random walk (no ARMA structure in Δy), the entire
    series should be the trend component with cyclic identically zero.

    Ref: beveridgenelson.m:130 — When A is empty/zero, long-run multiplier
    is zero, so trend = y and cyclic = 0.
    """
    y = _make_random_walk(T=400, seed=42)
    # Pure random walk: constant only, drift = 0, no AR/MA
    # Ref: beveridgenelson.m:94 — maxp = max(p), when p is empty maxp=0
    params = np.array([0.0])  # just the constant (drift = 0)
    trend, cyclic = beveridgenelson(y, params, constant=1, p=None, q=None)

    # Trend should be exactly y
    npt.assert_allclose(
        trend,
        y,
        atol=ATOL,
        rtol=RTOL,
        err_msg="Pure random walk: trend should equal y",
    )
    # Cyclic should be identically zero
    npt.assert_allclose(
        cyclic,
        np.zeros(len(y)),
        atol=ATOL,
        rtol=RTOL,
        err_msg="Pure random walk: cyclic should be zero",
    )


# ---------------------------------------------------------------------------
# Test 5: Non-trivial cyclic with AR component
# ---------------------------------------------------------------------------
def test_beveridgenelson_with_ar_component() -> None:
    """When Δy has ARMA structure, the cyclic component should be non-trivial.

    An AR(1) coefficient of 0.5 on differences will produce meaningful
    transitory dynamics. The cyclic component should have non-zero values
    after the initial warm-up period.

    Ref: beveridgenelson.m:130 — long-run multiplier (I - A)^{-1} A ≠ 0
    """
    y = _make_random_walk(T=500, seed=42)
    # AR(1) with phi=0.5 — substantial persistence in differences
    params = np.array([0.01, 0.5])  # [constant, AR(1)=0.5]
    trend, cyclic = beveridgenelson(y, params, constant=1, p=np.array([1]), q=None)

    # Additive decomposition must still hold
    npt.assert_allclose(
        trend + cyclic,
        y,
        atol=ATOL,
        rtol=RTOL,
        err_msg="Additive decomposition failed for AR(1) case",
    )

    # Cyclic should be non-trivially non-zero (not all zeros)
    # Ref: beveridgenelson.m:130 — when A has non-zero entries, cyclic ≠ 0
    # After the first maxp+1 warm-up observations, cyclic should be non-zero
    maxp = 1  # AR(1)
    cyclic_active = cyclic[maxp + 1:]
    assert np.max(np.abs(cyclic_active)) > 1e-3, (
        "Cyclic component should be non-trivially non-zero for AR(1) model"
    )


# ---------------------------------------------------------------------------
# Test 6: Trend is integrated (non-stationary)
# ---------------------------------------------------------------------------
def test_beveridgenelson_trend_is_integrated() -> None:
    """The trend component should behave like a random walk — its variance
    should grow with the sample size (non-stationary).

    We check that the variance of the trend for the second half of the
    sample exceeds the variance for the first quarter, consistent with
    an integrated process.

    Ref: beveridgenelson.m:1 — trend is the permanent (random walk) component
    """
    T = 1000
    y = _make_random_walk(T=T, seed=42)
    params = np.array([0.05, 0.3])  # AR(1) model on differences
    trend, _cyclic = beveridgenelson(y, params, constant=1, p=np.array([1]), q=None)

    # Variance of trend in second half should exceed first quarter
    # because an I(1) process has growing variance
    quarter = T // 4
    half = T // 2
    var_first_quarter = np.var(trend[:quarter])
    var_second_half = np.var(trend[half:])

    assert var_second_half > var_first_quarter, (
        f"Trend variance should grow: var(second_half)={var_second_half:.4f} "
        f"should exceed var(first_quarter)={var_first_quarter:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: Cyclic component is stationary (bounded variance)
# ---------------------------------------------------------------------------
def test_beveridgenelson_cyclic_is_stationary() -> None:
    """The cyclic (transitory) component should be stationary with
    bounded variance — its variance should not grow systematically
    with sample size.

    Ref: beveridgenelson.m:1 — cyclic is the transitory (stationary) component
    """
    T = 1000
    y = _make_random_walk(T=T, seed=42)
    params = np.array([0.05, 0.3])  # AR(1) on differences
    trend, cyclic = beveridgenelson(y, params, constant=1, p=np.array([1]), q=None)

    # For a stationary cyclic component, the standard deviation should be
    # bounded and the ratio of std(second_half) / std(first_half) should not
    # blow up. For a truly stationary process these should be similar.
    half = T // 2
    # Skip initial warm-up where cyclic = 0
    maxp = 1
    cyclic_active = cyclic[maxp + 1:]
    mid = len(cyclic_active) // 2

    std_first = np.std(cyclic_active[:mid])
    std_second = np.std(cyclic_active[mid:])

    # Ratio should be bounded — for a stationary process, typically < 3
    # (allowing generous headroom for finite-sample variation)
    if std_first > 0:
        ratio = std_second / std_first
        assert ratio < 5.0, (
            f"Cyclic std ratio {ratio:.2f} too large; "
            f"std_first={std_first:.4f}, std_second={std_second:.4f}"
        )
    # Also verify cyclic std is much smaller than trend std
    trend_std = np.std(trend)
    cyclic_std = np.std(cyclic)
    assert cyclic_std < trend_std, (
        f"Cyclic std ({cyclic_std:.4f}) should be smaller than "
        f"trend std ({trend_std:.4f}) for an I(1) series"
    )


# ---------------------------------------------------------------------------
# Test 8: Parity with MATLAB fixtures
# ---------------------------------------------------------------------------
def test_beveridgenelson_parity() -> None:
    """Compare Python output against MATLAB-generated reference fixtures.

    The fixture file ``beveridgenelson.npy`` contains three scenarios:
    1. AR(1) on Δy — params=[0.05, 0.3], p=[1], q=[]
    2. ARMA(1,1) on Δy — params=[0.05, 0.25, 0.15], p=[1], q=[1]
    3. AR(2) on Δy — params=[0.05, 0.25, -0.1], p=[1,2], q=[]

    Ref: beveridgenelson.m — full decomposition for all three scenarios
    """
    # Determine fixture path
    fixture_dir = os.environ.get(
        "MFE_FIXTURE_DIR",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures"),
    )
    fixture_path = os.path.join(fixture_dir, "timeseries", "beveridgenelson.npy")

    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture file not found: {fixture_path}")

    data = np.load(fixture_path, allow_pickle=True).item()
    y = data["input_y"]

    # ------------------------------------------------------------------
    # Scenario 1: AR(1) on differences
    # Ref: beveridgenelson.m — ARMA(1,0) with constant
    # ------------------------------------------------------------------
    s1_params = data["scenario1_params"]
    s1_const = int(data["scenario1_constant"])
    s1_p = data["scenario1_p"]
    s1_q = data["scenario1_q"]

    s1_trend, s1_cyclic = beveridgenelson(
        y,
        s1_params,
        s1_const,
        p=s1_p if len(s1_p) > 0 else None,
        q=s1_q if len(s1_q) > 0 else None,
    )

    npt.assert_allclose(
        s1_trend,
        data["scenario1_trend"],
        atol=ATOL,
        rtol=RTOL,
        err_msg="Scenario 1 (AR(1)): trend mismatch vs MATLAB fixture",
    )
    npt.assert_allclose(
        s1_cyclic,
        data["scenario1_cyclic"],
        atol=ATOL,
        rtol=RTOL,
        err_msg="Scenario 1 (AR(1)): cyclic mismatch vs MATLAB fixture",
    )

    # ------------------------------------------------------------------
    # Scenario 2: ARMA(1,1) on differences
    # Ref: beveridgenelson.m — ARMA(1,1) with constant
    # ------------------------------------------------------------------
    s2_params = data["scenario2_params"]
    s2_const = int(data["scenario2_constant"])
    s2_p = data["scenario2_p"]
    s2_q = data["scenario2_q"]

    s2_trend, s2_cyclic = beveridgenelson(
        y,
        s2_params,
        s2_const,
        p=s2_p if len(s2_p) > 0 else None,
        q=s2_q if len(s2_q) > 0 else None,
    )

    npt.assert_allclose(
        s2_trend,
        data["scenario2_trend"],
        atol=ATOL,
        rtol=RTOL,
        err_msg="Scenario 2 (ARMA(1,1)): trend mismatch vs MATLAB fixture",
    )
    npt.assert_allclose(
        s2_cyclic,
        data["scenario2_cyclic"],
        atol=ATOL,
        rtol=RTOL,
        err_msg="Scenario 2 (ARMA(1,1)): cyclic mismatch vs MATLAB fixture",
    )

    # ------------------------------------------------------------------
    # Scenario 3: AR(2) on differences
    # Ref: beveridgenelson.m — ARMA(2,0) with constant, p=[1,2]
    # ------------------------------------------------------------------
    s3_params = data["scenario3_params"]
    s3_const = int(data["scenario3_constant"])
    s3_p = data["scenario3_p"]
    s3_q = data["scenario3_q"]

    s3_trend, s3_cyclic = beveridgenelson(
        y,
        s3_params,
        s3_const,
        p=s3_p if len(s3_p) > 0 else None,
        q=s3_q if len(s3_q) > 0 else None,
    )

    npt.assert_allclose(
        s3_trend,
        data["scenario3_trend"],
        atol=ATOL,
        rtol=RTOL,
        err_msg="Scenario 3 (AR(2)): trend mismatch vs MATLAB fixture",
    )
    npt.assert_allclose(
        s3_cyclic,
        data["scenario3_cyclic"],
        atol=ATOL,
        rtol=RTOL,
        err_msg="Scenario 3 (AR(2)): cyclic mismatch vs MATLAB fixture",
    )

    # Verify additive decomposition for all scenarios
    # Ref: beveridgenelson.m:156 — cyclic = y - trend
    for label, t, c in [
        ("Scenario 1", s1_trend, s1_cyclic),
        ("Scenario 2", s2_trend, s2_cyclic),
        ("Scenario 3", s3_trend, s3_cyclic),
    ]:
        npt.assert_allclose(
            t + c,
            y,
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"{label}: additive decomposition failed on fixture data",
        )


# ---------------------------------------------------------------------------
# Test 9: Invalid parameter length raises ValueError
# ---------------------------------------------------------------------------
def test_beveridgenelson_invalid_parameters_length() -> None:
    """Mismatched parameter vector length must raise ``ValueError``.

    For constant=1, p=[1], q=None: expected parameter count = 1 + 1 + 0 = 2.
    Passing 3 parameters should raise ValueError.

    Ref: beveridgenelson.m:72-82 — parameter count validation
    """
    y = _make_random_walk(T=200, seed=42)

    # Too many parameters: expect 2 (constant + AR(1)), provide 3
    with pytest.raises(ValueError):
        beveridgenelson(y, np.array([0.05, 0.3, 0.2]), constant=1,
                        p=np.array([1]), q=None)

    # Too few parameters: expect 2, provide 1
    with pytest.raises(ValueError):
        beveridgenelson(y, np.array([0.05]), constant=1,
                        p=np.array([1]), q=None)


# ---------------------------------------------------------------------------
# Test 10: Additional edge case — ARMA(1,1) additive decomposition
# ---------------------------------------------------------------------------
def test_beveridgenelson_arma11_additive_decomposition() -> None:
    """Additive decomposition must hold for ARMA(1,1) on differences.

    Ref: beveridgenelson.m:119-120 — ARMA residual computation
    Ref: beveridgenelson.m:130 — long-run multiplier with both AR and MA terms
    """
    y = _make_random_walk(T=600, seed=77)
    # ARMA(1,1): [constant, AR(1), MA(1)]
    params = np.array([0.02, 0.4, 0.2])
    trend, cyclic = beveridgenelson(
        y, params, constant=1, p=np.array([1]), q=np.array([1])
    )

    npt.assert_allclose(
        trend + cyclic,
        y,
        atol=ATOL,
        rtol=RTOL,
        err_msg="ARMA(1,1) additive decomposition failed",
    )

    # Cyclic should be non-trivial for an ARMA model
    maxp = 1
    assert np.max(np.abs(cyclic[maxp + 1:])) > 1e-3, (
        "Cyclic component should be non-zero for ARMA(1,1) model"
    )


# ---------------------------------------------------------------------------
# Test 11: Invalid constant value raises ValueError
# ---------------------------------------------------------------------------
def test_beveridgenelson_invalid_constant() -> None:
    """Passing constant not in {0, 1} should raise ValueError.

    Ref: beveridgenelson.m:56-58 — constant validation
    """
    y = _make_random_walk(T=200, seed=42)
    params = np.array([0.05, 0.3])

    with pytest.raises(ValueError):
        beveridgenelson(y, params, constant=2, p=np.array([1]), q=None)

    with pytest.raises(ValueError):
        beveridgenelson(y, params, constant=-1, p=np.array([1]), q=None)


# ---------------------------------------------------------------------------
# Test 12: Non-vector input raises ValueError
# ---------------------------------------------------------------------------
def test_beveridgenelson_non_vector_input() -> None:
    """A 2D input (matrix) should raise ValueError — only 1D vectors allowed.

    Ref: beveridgenelson.m:49-52 — input must be T×1 vector
    """
    rng = _make_rng(42)
    y_2d = rng.standard_normal((100, 2))  # 2D — invalid
    params = np.array([0.05, 0.3])

    with pytest.raises(ValueError):
        beveridgenelson(y_2d, params, constant=1, p=np.array([1]), q=None)
