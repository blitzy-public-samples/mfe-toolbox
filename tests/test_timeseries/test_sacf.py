"""
Pytest tests for mfe_toolbox.timeseries.sacf — Sample Autocorrelation Function.

Tests cover:
  - Return type and tuple structure (ac, acstd, fig)
  - Output shape validation for various lag configurations
  - Standard deviation positivity
  - Autocorrelation boundedness (|ac| <= 1)
  - Statistical properties (white noise within 95% CI, AR(1) first-lag approximation)
  - Robust vs classic standard error distinction
  - Graph suppression (graph=0 → fig is None)
  - Input validation (lags >= T → ValueError)
  - Numerical parity against MATLAB-generated fixtures (ATOL=1e-6, RTOL=1e-4)

Migrated from timeseries/sacf.m — MFE Toolbox v4.0 (Kevin Sheppard)
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless testing

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.sacf import sacf

# ---------------------------------------------------------------------------
# Tolerance constants for MATLAB parity tests
# Per AAP Section 0.7.1: atol=1e-6, rtol=1e-4
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Helper: Load MATLAB fixture data for sacf parity tests
# ---------------------------------------------------------------------------
def _load_sacf_fixture(timeseries_fixture_dir):
    """Load the sacf fixture .npy file and return its dict contents.

    Parameters
    ----------
    timeseries_fixture_dir : pathlib.Path
        Path to tests/fixtures/timeseries/.

    Returns
    -------
    dict
        Dictionary with keys: data, scenario_N_ac, scenario_N_acstd,
        scenario_N_lags, scenario_N_robust for N in {1,2,3,4}.
    """
    path = timeseries_fixture_dir / "sacf.npy"
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")
    return np.load(path, allow_pickle=True).item()


# ===========================================================================
# Test: Return structure
# ===========================================================================

class TestSacfReturnStructure:
    """Verify sacf returns the expected tuple (ac, acstd, fig)."""

    def test_sacf_returns_three(self):
        """sacf must return exactly 3 values: (ac, acstd, fig)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        result = sacf(data, 10, robust=True, graph=False)
        assert isinstance(result, tuple), "sacf must return a tuple"
        assert len(result) == 3, "sacf must return exactly 3 values"

    def test_sacf_return_types(self):
        """ac and acstd must be numpy arrays; fig must be None when graph=False."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        ac, acstd, fig = sacf(data, 10, robust=True, graph=False)
        assert isinstance(ac, np.ndarray), "ac must be a numpy ndarray"
        assert isinstance(acstd, np.ndarray), "acstd must be a numpy ndarray"
        assert fig is None, "fig must be None when graph=False"


# ===========================================================================
# Test: Output shapes
# ===========================================================================

class TestSacfShapes:
    """Verify output shape correctness for various lag configurations."""

    @pytest.mark.parametrize("lags", [1, 5, 10, 20, 50])
    def test_sacf_ac_shape(self, lags):
        """ac must have exactly `lags` elements."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        ac, acstd, _ = sacf(data, lags, graph=False)
        assert ac.shape == (lags,), f"Expected ac shape ({lags},), got {ac.shape}"

    @pytest.mark.parametrize("lags", [1, 5, 10, 20, 50])
    def test_sacf_acstd_shape(self, lags):
        """acstd must have exactly `lags` elements."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        ac, acstd, _ = sacf(data, lags, graph=False)
        assert acstd.shape == (lags,), f"Expected acstd shape ({lags},), got {acstd.shape}"

    def test_sacf_single_lag(self):
        """With lags=1, ac and acstd should each have exactly 1 element."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        ac, acstd, _ = sacf(data, 1, graph=False)
        assert ac.shape == (1,)
        assert acstd.shape == (1,)


# ===========================================================================
# Test: Standard deviation positivity
# ===========================================================================

class TestSacfStdPositivity:
    """All standard deviations must be strictly positive."""

    def test_sacf_acstd_positive_robust(self):
        """Robust standard deviations must all be > 0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        _, acstd, _ = sacf(data, 20, robust=True, graph=False)
        assert np.all(acstd > 0), "All robust standard deviations must be positive"

    def test_sacf_acstd_positive_classic(self):
        """Classic (non-robust) standard deviations must all be > 0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        _, acstd, _ = sacf(data, 20, robust=False, graph=False)
        assert np.all(acstd > 0), "All classic standard deviations must be positive"


# ===========================================================================
# Test: Autocorrelation boundedness
# ===========================================================================

def test_sacf_ac_bounded():
    """All sample autocorrelations must satisfy |ac| <= 1.

    The OLS-based sample autocorrelation coefficient for any finite sample
    must lie within [-1, 1] (correlation bound).
    """
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    ac, _, _ = sacf(data, 50, graph=False)
    assert np.all(np.abs(ac) <= 1.0 + 1e-10), "All |ac| must be <= 1"


# ===========================================================================
# Test: White noise statistical property
# ===========================================================================

def test_sacf_white_noise():
    """For white noise, most ACFs should fall within ±2/sqrt(T) (95% CI).

    Under H0 (iid white noise), each sample autocorrelation is approximately
    N(0, 1/T), so approximately 95% should lie within ±2/sqrt(T). We allow
    up to 3 exceedances out of 20 lags.

    Ref: sacf.m uses this property for confidence band visualization.
    """
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    ac, acstd, _ = sacf(data, 20, robust=False, graph=False)
    bound = 2.0 / np.sqrt(1000)
    # Most ACFs should be within bounds (allow 1-3 exceedances)
    n_exceedances = np.sum(np.abs(ac) > bound)
    assert n_exceedances <= 3, (
        f"Expected at most 3 exceedances for white noise, got {n_exceedances}"
    )


# ===========================================================================
# Test: AR(1) first-lag approximation
# ===========================================================================

def test_sacf_ar1_first_lag():
    """For AR(1) with phi=0.7 and T=5000, first sample ACF should be ≈ 0.7.

    Generate a long AR(1) process y_t = 0.7*y_{t-1} + e_t, then check
    the first sample autocorrelation is close to the true autoregressive
    coefficient. With T=5000, the sample estimate is typically within 0.05
    of the population value.

    Ref: sacf.m:77-81 — OLS regression y_t on y_{t-1} recovers the AR coeff.
    """
    rng = np.random.default_rng(42)
    T = 5000
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.7 * y[t - 1] + rng.standard_normal()
    ac, _, _ = sacf(y, 5, graph=False)
    # The first sample ACF should be close to the true AR(1) coefficient
    npt.assert_allclose(ac[0], 0.7, atol=0.05)


def test_sacf_ar1_decay_pattern():
    """For AR(1) with phi=0.7, ACFs should decay geometrically.

    The theoretical ACF for an AR(1) process is rho_k = phi^k. We verify
    that the second ACF is approximately phi^2 = 0.49 and that ACFs
    decrease monotonically in absolute value for the first few lags.
    """
    rng = np.random.default_rng(42)
    T = 5000
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.7 * y[t - 1] + rng.standard_normal()
    ac, _, _ = sacf(y, 5, graph=False)
    # Second ACF should be approximately phi^2 = 0.49
    npt.assert_allclose(ac[1], 0.49, atol=0.06)
    # ACFs should decrease in absolute value for first few lags
    assert np.abs(ac[0]) > np.abs(ac[1]), "ACF should decay: |ac[0]| > |ac[1]|"
    assert np.abs(ac[1]) > np.abs(ac[2]), "ACF should decay: |ac[1]| > |ac[2]|"


# ===========================================================================
# Test: Robust vs Classic standard errors
# ===========================================================================

def test_sacf_robust_vs_classic():
    """Robust and classic standard errors must differ for heteroskedastic data.

    The robust (White sandwich) standard errors account for
    heteroskedasticity, while the classic SE uses 1/sqrt(T-L). For data
    with time-varying variance, these should produce different SE values.

    Ref: sacf.m:82-90 — robust vs non-robust branch.
    """
    rng = np.random.default_rng(42)
    # Create data with some heteroskedasticity (ARCH-like)
    T = 1000
    data = rng.standard_normal(T)
    # Introduce heteroskedasticity: multiply by a time-varying scale
    scale = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(T) / 100.0)
    data = data * scale

    ac_robust, acstd_robust, _ = sacf(data, 20, robust=True, graph=False)
    ac_classic, acstd_classic, _ = sacf(data, 20, robust=False, graph=False)

    # Autocorrelations should be identical regardless of SE method
    # (the AC computation doesn't depend on the robust flag)
    npt.assert_allclose(
        ac_robust, ac_classic, atol=ATOL, rtol=RTOL,
        err_msg="Autocorrelations should be identical for robust and classic"
    )

    # Standard errors must differ (robust accounts for heteroskedasticity)
    assert not np.allclose(acstd_robust, acstd_classic, atol=1e-10), (
        "Robust and classic standard errors should differ for "
        "heteroskedastic data"
    )


def test_sacf_classic_se_formula():
    """Classic (non-robust) SE should equal 1/sqrt(T-L) for each lag L.

    Ref: sacf.m:89 — acstd(L) = sqrt(1/t) where t = T - L.
    """
    rng = np.random.default_rng(42)
    T = 500
    data = rng.standard_normal(T)
    lags = 10
    _, acstd, _ = sacf(data, lags, robust=False, graph=False)

    for lag_idx in range(lags):
        L = lag_idx + 1  # 1-based lag
        t = T - L
        expected_se = np.sqrt(1.0 / t)
        npt.assert_allclose(
            acstd[lag_idx], expected_se, atol=ATOL, rtol=RTOL,
            err_msg=f"Classic SE at lag {L} should be 1/sqrt({t})"
        )


# ===========================================================================
# Test: Graph suppression
# ===========================================================================

def test_sacf_no_graph():
    """graph=False must return fig=None (no figure created).

    Ref: sacf.m:119-120 — 'else hfig = [];' when graph is false.
    """
    rng = np.random.default_rng(42)
    data = rng.standard_normal(200)
    _, _, fig = sacf(data, 10, graph=False)
    assert fig is None, "fig must be None when graph=False"


def test_sacf_no_graph_int_arg():
    """graph=0 (integer) must also suppress graphing."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(200)
    _, _, fig = sacf(data, 10, graph=0)
    assert fig is None, "fig must be None when graph=0"


def test_sacf_with_graph():
    """graph=True must return a matplotlib Figure object.

    Ref: sacf.m:94-118 — creates bar plot with confidence bands.
    """
    import matplotlib.figure

    rng = np.random.default_rng(42)
    data = rng.standard_normal(200)
    _, _, fig = sacf(data, 10, robust=True, graph=True)
    assert isinstance(fig, matplotlib.figure.Figure), (
        "fig must be a matplotlib Figure when graph=True"
    )
    # Cleanup: close the figure to free memory
    import matplotlib.pyplot as plt
    plt.close(fig)


# ===========================================================================
# Test: Input validation
# ===========================================================================

class TestSacfInputValidation:
    """Verify that sacf raises ValueError for invalid inputs."""

    def test_sacf_lags_exceeds_data(self):
        """lags >= T must raise ValueError.

        Ref: sacf.m:47-49 — 'if T<=lags error(...)'
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal(20)
        with pytest.raises(ValueError, match="Length of data must be larger than LAGS"):
            sacf(data, 20, graph=False)

    def test_sacf_lags_equals_data(self):
        """lags == T must raise ValueError (equal case is also invalid)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(10)
        with pytest.raises(ValueError):
            sacf(data, 10, graph=False)

    def test_sacf_lags_exceeds_data_large(self):
        """lags much larger than T must raise ValueError."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(10)
        with pytest.raises(ValueError):
            sacf(data, 100, graph=False)

    def test_sacf_lags_zero(self):
        """lags=0 must raise ValueError.

        Ref: sacf.m:56 — lags<=0 check.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        with pytest.raises(ValueError, match="LAGS must be a positive integer"):
            sacf(data, 0, graph=False)

    def test_sacf_lags_negative(self):
        """Negative lags must raise ValueError."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        with pytest.raises(ValueError, match="LAGS must be a positive integer"):
            sacf(data, -5, graph=False)

    def test_sacf_lags_float(self):
        """Non-integer lags (e.g. 5.5) must raise ValueError.

        Ref: sacf.m:56 — floor(lags)~=lags check.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        with pytest.raises(ValueError, match="LAGS must be a positive integer"):
            sacf(data, 5.5, graph=False)

    def test_sacf_matrix_input(self):
        """Matrix input (T x K with K > 1) must raise ValueError.

        Ref: sacf.m:53-55 — 'if size(data,2)~=1 error(...)'
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 3))
        with pytest.raises(ValueError, match="DATA must be a column vector"):
            sacf(data, 10, graph=False)

    def test_sacf_row_vector_accepted(self):
        """Row vector (1 x T) must be accepted by converting to 1-D.

        Ref: sacf.m:50-52 — transposing row vectors.
        """
        rng = np.random.default_rng(42)
        data_1d = rng.standard_normal(200)
        data_row = data_1d.reshape(1, -1)
        ac_1d, acstd_1d, _ = sacf(data_1d, 10, graph=False)
        ac_row, acstd_row, _ = sacf(data_row, 10, graph=False)
        npt.assert_allclose(ac_1d, ac_row, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(acstd_1d, acstd_row, atol=ATOL, rtol=RTOL)

    def test_sacf_column_vector_accepted(self):
        """Column vector (T x 1) must be accepted by converting to 1-D.

        Ref: sacf.m:50-52
        """
        rng = np.random.default_rng(42)
        data_1d = rng.standard_normal(200)
        data_col = data_1d.reshape(-1, 1)
        ac_1d, acstd_1d, _ = sacf(data_1d, 10, graph=False)
        ac_col, acstd_col, _ = sacf(data_col, 10, graph=False)
        npt.assert_allclose(ac_1d, ac_col, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(acstd_1d, acstd_col, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Test: MATLAB Parity — Fixture-Based Tests
# ===========================================================================

@pytest.mark.parity
class TestSacfParity:
    """Verify numerical parity against MATLAB-generated fixtures.

    Per AAP Section 0.7.1: All migrated functions MUST pass
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
    against MATLAB-generated fixtures.

    Fixtures are stored in tests/fixtures/timeseries/sacf.npy and contain
    4 scenarios:
      - Scenario 1: lags=20, robust=1 (robust standard errors)
      - Scenario 2: lags=20, robust=0 (classic standard errors)
      - Scenario 3: lags=5,  robust=1
      - Scenario 4: lags=50, robust=1
    """

    def test_sacf_parity_scenario1_ac(self, timeseries_fixture_dir):
        """Scenario 1 (lags=20, robust=1): autocorrelations match MATLAB."""
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]
        lags = int(fixture["scenario_1_lags"])
        robust = bool(fixture["scenario_1_robust"])
        expected_ac = fixture["scenario_1_ac"]

        ac, _, _ = sacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            ac, expected_ac, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 1 AC values do not match MATLAB reference"
        )

    def test_sacf_parity_scenario1_acstd(self, timeseries_fixture_dir):
        """Scenario 1 (lags=20, robust=1): standard deviations match MATLAB."""
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]
        lags = int(fixture["scenario_1_lags"])
        robust = bool(fixture["scenario_1_robust"])
        expected_acstd = fixture["scenario_1_acstd"]

        _, acstd, _ = sacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            acstd, expected_acstd, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 1 ACSTD values do not match MATLAB reference"
        )

    def test_sacf_parity_scenario2_ac(self, timeseries_fixture_dir):
        """Scenario 2 (lags=20, robust=0): autocorrelations match MATLAB."""
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]
        lags = int(fixture["scenario_2_lags"])
        robust = bool(fixture["scenario_2_robust"])
        expected_ac = fixture["scenario_2_ac"]

        ac, _, _ = sacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            ac, expected_ac, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 2 AC values do not match MATLAB reference"
        )

    def test_sacf_parity_scenario2_acstd(self, timeseries_fixture_dir):
        """Scenario 2 (lags=20, robust=0): classic SEs match MATLAB."""
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]
        lags = int(fixture["scenario_2_lags"])
        robust = bool(fixture["scenario_2_robust"])
        expected_acstd = fixture["scenario_2_acstd"]

        _, acstd, _ = sacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            acstd, expected_acstd, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 2 ACSTD values do not match MATLAB reference"
        )

    def test_sacf_parity_scenario3_ac(self, timeseries_fixture_dir):
        """Scenario 3 (lags=5, robust=1): short-lag ACFs match MATLAB."""
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]
        lags = int(fixture["scenario_3_lags"])
        robust = bool(fixture["scenario_3_robust"])
        expected_ac = fixture["scenario_3_ac"]

        ac, _, _ = sacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            ac, expected_ac, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 3 AC values do not match MATLAB reference"
        )

    def test_sacf_parity_scenario3_acstd(self, timeseries_fixture_dir):
        """Scenario 3 (lags=5, robust=1): short-lag SEs match MATLAB."""
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]
        lags = int(fixture["scenario_3_lags"])
        robust = bool(fixture["scenario_3_robust"])
        expected_acstd = fixture["scenario_3_acstd"]

        _, acstd, _ = sacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            acstd, expected_acstd, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 3 ACSTD values do not match MATLAB reference"
        )

    def test_sacf_parity_scenario4_ac(self, timeseries_fixture_dir):
        """Scenario 4 (lags=50, robust=1): long-lag ACFs match MATLAB."""
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]
        lags = int(fixture["scenario_4_lags"])
        robust = bool(fixture["scenario_4_robust"])
        expected_ac = fixture["scenario_4_ac"]

        ac, _, _ = sacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            ac, expected_ac, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 4 AC values do not match MATLAB reference"
        )

    def test_sacf_parity_scenario4_acstd(self, timeseries_fixture_dir):
        """Scenario 4 (lags=50, robust=1): long-lag SEs match MATLAB."""
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]
        lags = int(fixture["scenario_4_lags"])
        robust = bool(fixture["scenario_4_robust"])
        expected_acstd = fixture["scenario_4_acstd"]

        _, acstd, _ = sacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            acstd, expected_acstd, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 4 ACSTD values do not match MATLAB reference"
        )

    def test_sacf_parity_sacf_out(self, timeseries_fixture_dir):
        """sacf_sacf_out.npy fixture: standalone AC reference vector matches.

        This fixture file contains the autocorrelation output from the
        default scenario (lags=20, robust=1).
        """
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]

        path = timeseries_fixture_dir / "sacf_sacf_out.npy"
        if not path.exists():
            pytest.skip(f"Fixture file not found: {path}")
        expected_ac = np.load(path, allow_pickle=True)

        ac, _, _ = sacf(data, len(expected_ac), robust=True, graph=False)
        npt.assert_allclose(
            ac, expected_ac, atol=ATOL, rtol=RTOL,
            err_msg="sacf_sacf_out AC vector does not match MATLAB reference"
        )

    def test_sacf_parity_sacf_bounds(self, timeseries_fixture_dir):
        """sacf_sacf_bounds.npy fixture: standalone SE reference vector matches.

        This fixture file contains the standard deviation output from the
        default scenario (lags=20, robust=1).
        """
        fixture = _load_sacf_fixture(timeseries_fixture_dir)
        data = fixture["data"]

        path = timeseries_fixture_dir / "sacf_sacf_bounds.npy"
        if not path.exists():
            pytest.skip(f"Fixture file not found: {path}")
        expected_acstd = np.load(path, allow_pickle=True)

        _, acstd, _ = sacf(data, len(expected_acstd), robust=True, graph=False)
        npt.assert_allclose(
            acstd, expected_acstd, atol=ATOL, rtol=RTOL,
            err_msg="sacf_sacf_bounds SE vector does not match MATLAB reference"
        )


# ===========================================================================
# Test: Consistency and edge cases
# ===========================================================================

class TestSacfConsistency:
    """Verify internal consistency and edge-case behaviour."""

    def test_sacf_deterministic(self):
        """Calling sacf twice with the same data must return identical results."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(300)
        ac1, acstd1, _ = sacf(data, 15, robust=True, graph=False)
        ac2, acstd2, _ = sacf(data, 15, robust=True, graph=False)
        npt.assert_array_equal(ac1, ac2, err_msg="sacf must be deterministic")
        npt.assert_array_equal(acstd1, acstd2, err_msg="sacf must be deterministic")

    def test_sacf_small_data(self):
        """sacf must work correctly with very small datasets (T=5, lags=2)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(5)
        ac, acstd, _ = sacf(data, 2, graph=False)
        assert ac.shape == (2,)
        assert acstd.shape == (2,)
        assert np.all(np.isfinite(ac)), "All ac values must be finite"
        assert np.all(np.isfinite(acstd)), "All acstd values must be finite"

    def test_sacf_lags_one_less_than_t(self):
        """sacf must handle the boundary case lags = T - 1 with classic SEs.

        Note: With robust=True and lags very close to T, the last few lags
        have too few observations (e.g. 1 obs → singular x'x matrix). Using
        robust=False avoids this since classic SE = 1/sqrt(t) does not
        require matrix inversion. This matches MATLAB behaviour where
        the same singularity would occur with robust SEs at the boundary.

        Ref: sacf.m:89 — acstd(L) = sqrt(1/t) for non-robust case.
        """
        rng = np.random.default_rng(42)
        T = 20
        data = rng.standard_normal(T)
        ac, acstd, _ = sacf(data, T - 1, robust=False, graph=False)
        assert ac.shape == (T - 1,)
        assert acstd.shape == (T - 1,)
        assert np.all(np.isfinite(ac)), "All ac values must be finite"

    def test_sacf_robust_default(self):
        """Default robust parameter should be True (robust standard errors).

        Ref: sacf.m:41 — 'robust = true;' as default.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal(300)
        # Call with default robust (should be True)
        ac_default, acstd_default, _ = sacf(data, 10, graph=False)
        ac_robust, acstd_robust, _ = sacf(data, 10, robust=True, graph=False)
        npt.assert_allclose(
            acstd_default, acstd_robust, atol=ATOL, rtol=RTOL,
            err_msg="Default robust parameter should produce robust SEs"
        )

    def test_sacf_near_zero_mean_data(self):
        """Near-zero-mean white noise should have near-zero ACFs.

        Generates a large white noise sample (T=2000), demeaned, and
        verifies that all sample ACFs are close to zero (within sampling
        variation bounds).
        """
        rng = np.random.default_rng(123)
        T = 2000
        data = rng.standard_normal(T)
        data = data - data.mean()  # exact demean
        ac, acstd, _ = sacf(data, 10, robust=False, graph=False)
        # For T=2000 iid data, each sample ACF has SE ≈ 1/sqrt(2000) ≈ 0.022
        # All ACFs should be within 3*SE ≈ 0.067 of zero
        bound = 3.0 / np.sqrt(T)
        assert np.all(np.abs(ac) < bound), (
            f"All ACFs for demeaned white noise should be < {bound:.4f}"
        )
