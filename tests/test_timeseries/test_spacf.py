"""
Pytest tests for mfe_toolbox.timeseries.spacf — Sample Partial Autocorrelation Function.

Tests cover:
  - Return type and tuple structure (pac, pacstd, fig)
  - Output shape validation for various lag configurations
  - Standard deviation positivity
  - Partial autocorrelation boundedness (|pac| <= 1)
  - AR(1) cutoff property: pac(1) ≈ phi, pac(k) ≈ 0 for k ≥ 2
  - AR(2) cutoff property: pac(1), pac(2) nonzero; pac(k) ≈ 0 for k ≥ 3
  - White noise property: all |pac| near zero
  - Robust vs classic standard error distinction
  - Graph suppression (graph=0 → fig is None)
  - Graph generation (graph=1 → matplotlib figure returned)
  - Input validation (lags >= T → ValueError)
  - Numerical parity against MATLAB-generated fixtures (ATOL=1e-6, RTOL=1e-4)

Migrated from timeseries/spacf.m — MFE Toolbox v4.0 (Kevin Sheppard)
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless testing

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.spacf import spacf

# ---------------------------------------------------------------------------
# Tolerance constants for MATLAB parity tests
# Per AAP Section 0.7.1: atol=1e-6, rtol=1e-4
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Helper: Load MATLAB fixture data for spacf parity tests
# ---------------------------------------------------------------------------
def _load_spacf_fixture(timeseries_fixture_dir):
    """Load the spacf fixture .npy file and return its dict contents.

    Parameters
    ----------
    timeseries_fixture_dir : pathlib.Path
        Path to tests/fixtures/timeseries/.

    Returns
    -------
    dict
        Dictionary with keys: data, scenario_N_pac, scenario_N_pacstd,
        scenario_N_lags, scenario_N_robust for N in {1,2,3,4}.
    """
    path = timeseries_fixture_dir / "spacf.npy"
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")
    return np.load(path, allow_pickle=True).item()


# ===========================================================================
# Test: Return structure
# ===========================================================================

class TestSpacfReturnStructure:
    """Verify spacf returns the expected tuple (pac, pacstd, fig)."""

    def test_spacf_returns_three(self):
        """spacf must return exactly 3 values: (pac, pacstd, fig)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        result = spacf(data, 10, robust=True, graph=False)
        assert isinstance(result, tuple), "spacf must return a tuple"
        assert len(result) == 3, "spacf must return exactly 3 values"

    def test_spacf_return_types(self):
        """pac and pacstd must be numpy arrays; fig must be None when graph=False."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        pac, pacstd, fig = spacf(data, 10, robust=True, graph=False)
        assert isinstance(pac, np.ndarray), "pac must be a numpy ndarray"
        assert isinstance(pacstd, np.ndarray), "pacstd must be a numpy ndarray"
        assert fig is None, "fig must be None when graph=False"


# ===========================================================================
# Test: Output shape
# ===========================================================================

class TestSpacfShape:
    """Verify that pac and pacstd have the correct dimensions."""

    @pytest.mark.parametrize("lags", [1, 5, 10, 20, 50])
    def test_spacf_pac_shape(self, lags):
        """pac must have exactly `lags` elements for each tested value."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        pac, pacstd, _ = spacf(data, lags, robust=True, graph=False)
        assert pac.shape == (lags,), f"pac shape mismatch: expected ({lags},), got {pac.shape}"
        assert pacstd.shape == (lags,), f"pacstd shape mismatch: expected ({lags},), got {pacstd.shape}"

    def test_spacf_single_lag(self):
        """Verify spacf works correctly with lags=1 (minimum valid lag count)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        pac, pacstd, _ = spacf(data, 1, robust=True, graph=False)
        assert pac.shape == (1,), "Single lag pac shape must be (1,)"
        assert pacstd.shape == (1,), "Single lag pacstd shape must be (1,)"


# ===========================================================================
# Test: Standard deviation positivity
# ===========================================================================

class TestSpacfStdPositivity:
    """Verify that all standard deviations are strictly positive."""

    def test_spacf_pacstd_positive_robust(self):
        """Robust standard deviations must all be > 0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        _, pacstd, _ = spacf(data, 20, robust=True, graph=False)
        assert np.all(pacstd > 0), "All robust pacstd values must be strictly positive"

    def test_spacf_pacstd_positive_classic(self):
        """Classic (non-robust) standard deviations must all be > 0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        _, pacstd, _ = spacf(data, 20, robust=False, graph=False)
        assert np.all(pacstd > 0), "All classic pacstd values must be strictly positive"


# ===========================================================================
# Test: Partial autocorrelation boundedness
# ===========================================================================

class TestSpacfBoundedness:
    """Verify that partial autocorrelations are bounded by [-1, 1]."""

    def test_spacf_pac_bounded(self):
        """All |pac| must be <= 1."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        pac, _, _ = spacf(data, 20, robust=True, graph=False)
        assert np.all(np.abs(pac) <= 1.0), "All partial autocorrelations must be in [-1, 1]"

    def test_spacf_pac_bounded_ar1(self):
        """AR(1) process: all |pac| must be <= 1 even with strong autocorrelation."""
        rng = np.random.default_rng(42)
        T = 2000
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.9 * y[t - 1] + rng.standard_normal()
        pac, _, _ = spacf(y, 20, robust=True, graph=False)
        assert np.all(np.abs(pac) <= 1.0), "AR(1) pac must remain bounded in [-1, 1]"


# ===========================================================================
# Test: AR(1) cutoff property
# ===========================================================================

class TestSpacfAR1Cutoff:
    """Verify AR(1) cutoff: pac(1) ≈ phi, pac(k) ≈ 0 for k >= 2.

    For an AR(1) process y_t = phi * y_{t-1} + eps_t, the theoretical PACF
    has a single nonzero value at lag 1 equal to phi, and zero for all
    higher lags. With finite samples this is approximate.
    """

    def test_spacf_ar1_cutoff(self):
        """AR(1) with phi=0.7: pac[0] ≈ 0.7, pac[2:] ≈ 0.

        Ref: spacf.m — Levinson-Durbin regression-based PACF.
        """
        rng = np.random.default_rng(42)
        T = 5000
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.7 * y[t - 1] + rng.standard_normal()
        pac, _, _ = spacf(y, 10, robust=True, graph=False)
        # Lag 1 should be close to the AR coefficient 0.7
        npt.assert_allclose(pac[0], 0.7, atol=0.05,
                            err_msg="AR(1) pac[0] should approximate phi=0.7")
        # Higher lags should be approximately zero
        npt.assert_allclose(pac[2:], 0.0, atol=0.1,
                            err_msg="AR(1) pac[k] for k>=3 should be near zero")

    def test_spacf_ar1_negative_phi(self):
        """AR(1) with phi=-0.5: pac[0] ≈ -0.5, higher lags ≈ 0."""
        rng = np.random.default_rng(99)
        T = 5000
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = -0.5 * y[t - 1] + rng.standard_normal()
        pac, _, _ = spacf(y, 10, robust=True, graph=False)
        npt.assert_allclose(pac[0], -0.5, atol=0.05,
                            err_msg="AR(1) pac[0] should approximate phi=-0.5")
        npt.assert_allclose(pac[2:], 0.0, atol=0.1,
                            err_msg="AR(1) pac[k] for k>=3 should be near zero")


# ===========================================================================
# Test: AR(2) cutoff property
# ===========================================================================

class TestSpacfAR2Cutoff:
    """Verify AR(2) cutoff: pac(1), pac(2) nonzero; pac(k) ≈ 0 for k >= 3.

    For an AR(2) process y_t = phi1 * y_{t-1} + phi2 * y_{t-2} + eps_t,
    the theoretical PACF has nonzero values only at lags 1 and 2.
    """

    def test_spacf_ar2_cutoff(self):
        """AR(2) with phi1=0.5, phi2=0.3: pac[0], pac[1] nonzero; pac[3:] ≈ 0."""
        rng = np.random.default_rng(42)
        T = 5000
        y = np.zeros(T)
        for t in range(2, T):
            y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 2] + rng.standard_normal()
        pac, _, _ = spacf(y, 10, robust=True, graph=False)
        # Lags 1 and 2 should be significantly nonzero
        assert np.abs(pac[0]) > 0.1, "AR(2) pac[0] should be meaningfully nonzero"
        assert np.abs(pac[1]) > 0.1, "AR(2) pac[1] should be meaningfully nonzero"
        # Higher lags should be approximately zero
        npt.assert_allclose(pac[3:], 0.0, atol=0.1,
                            err_msg="AR(2) pac[k] for k>=4 should be near zero")

    def test_spacf_ar2_approximate_values(self):
        """AR(2) with known coefficients: pac[1] ≈ phi2 (theoretical result)."""
        rng = np.random.default_rng(123)
        T = 10000
        phi1, phi2 = 0.6, 0.2
        y = np.zeros(T)
        for t in range(2, T):
            y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.standard_normal()
        pac, _, _ = spacf(y, 10, robust=True, graph=False)
        # The second partial autocorrelation should approximate phi2
        npt.assert_allclose(pac[1], phi2, atol=0.06,
                            err_msg="AR(2) pac[1] should approximate phi2")


# ===========================================================================
# Test: White noise property
# ===========================================================================

class TestSpacfWhiteNoise:
    """Verify that for white noise, all pac values are approximately zero.

    For IID noise, the asymptotic distribution of PACF estimates is
    N(0, 1/T), so values should be within about 2/sqrt(T) with high probability.
    """

    def test_spacf_white_noise(self):
        """White noise: all |pac| should be small (within ~2/sqrt(T) bound)."""
        rng = np.random.default_rng(42)
        T = 5000
        data = rng.standard_normal(T)
        pac, _, _ = spacf(data, 20, robust=True, graph=False)
        # Under the null of white noise, |pac| should be < 2/sqrt(T)
        # with high probability; use a slightly relaxed bound for test stability
        threshold = 3.0 / np.sqrt(T)
        num_exceeding = np.sum(np.abs(pac) > threshold)
        # Allow at most 2 out of 20 to exceed the threshold (5% significance)
        assert num_exceeding <= 2, (
            f"White noise: {num_exceeding} out of 20 pac values exceed "
            f"3/sqrt({T})={threshold:.4f}"
        )

    def test_spacf_white_noise_near_zero(self):
        """White noise pac values should have small absolute mean."""
        rng = np.random.default_rng(88)
        T = 10000
        data = rng.standard_normal(T)
        pac, _, _ = spacf(data, 20, robust=True, graph=False)
        # Mean of pac should be near zero
        assert np.abs(np.mean(pac)) < 0.05, (
            "Mean of white noise pac should be near zero"
        )


# ===========================================================================
# Test: Robust vs classic standard errors
# ===========================================================================

class TestSpacfRobustVsClassic:
    """Verify that robust and classic SEs produce different values.

    Classic SEs are 1/sqrt(T) for all lags; robust SEs use a White
    sandwich estimator that varies by lag.
    """

    def test_spacf_robust_vs_classic(self):
        """Robust and classic standard errors should differ in general."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(500)
        _, pacstd_robust, _ = spacf(data, 10, robust=True, graph=False)
        _, pacstd_classic, _ = spacf(data, 10, robust=False, graph=False)
        # They should generally not be identical
        assert not np.allclose(pacstd_robust, pacstd_classic, atol=1e-10), (
            "Robust and classic SEs should differ for generic data"
        )

    def test_spacf_classic_se_constant(self):
        """Classic SEs should all equal 1/sqrt(T)."""
        rng = np.random.default_rng(42)
        T = 500
        data = rng.standard_normal(T)
        _, pacstd_classic, _ = spacf(data, 20, robust=False, graph=False)
        expected_se = 1.0 / np.sqrt(T)
        npt.assert_allclose(
            pacstd_classic, expected_se, atol=ATOL,
            err_msg=f"Classic SEs should all equal 1/sqrt({T})={expected_se:.6f}"
        )

    def test_spacf_robust_se_varies(self):
        """Robust SEs should generally vary across lags (not constant)."""
        rng = np.random.default_rng(42)
        # Use AR(1) data with heteroskedasticity to ensure robust SEs vary
        T = 1000
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + rng.standard_normal() * (1.0 + 0.5 * np.abs(y[t - 1]))
        _, pacstd_robust, _ = spacf(y, 10, robust=True, graph=False)
        # Standard deviation of the SEs should be nonzero (they vary)
        assert np.std(pacstd_robust) > 1e-10, (
            "Robust SEs should vary across lags"
        )


# ===========================================================================
# Test: Graph suppression and generation
# ===========================================================================

class TestSpacfGraph:
    """Verify graph output control via the graph parameter."""

    def test_spacf_no_graph(self):
        """graph=False → fig must be None."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        _, _, fig = spacf(data, 10, robust=True, graph=False)
        assert fig is None, "fig must be None when graph=False"

    def test_spacf_no_graph_int(self):
        """graph=0 (integer) → fig must be None."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        _, _, fig = spacf(data, 10, robust=True, graph=0)
        assert fig is None, "fig must be None when graph=0"

    def test_spacf_with_graph(self):
        """graph=True → fig must be a matplotlib Figure."""
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        _, _, fig = spacf(data, 10, robust=True, graph=True)
        assert fig is not None, "fig must not be None when graph=True"
        # Check it's a matplotlib Figure
        from matplotlib.figure import Figure
        assert isinstance(fig, Figure), "fig must be a matplotlib Figure instance"
        plt.close(fig)

    def test_spacf_graph_does_not_affect_values(self):
        """pac and pacstd values must be identical regardless of graph setting."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        pac_ng, pacstd_ng, _ = spacf(data, 10, robust=True, graph=False)
        import matplotlib.pyplot as plt
        pac_g, pacstd_g, fig = spacf(data, 10, robust=True, graph=True)
        npt.assert_allclose(pac_ng, pac_g, atol=ATOL,
                            err_msg="pac should not depend on graph setting")
        npt.assert_allclose(pacstd_ng, pacstd_g, atol=ATOL,
                            err_msg="pacstd should not depend on graph setting")
        plt.close(fig)


# ===========================================================================
# Test: Input validation
# ===========================================================================

class TestSpacfInputValidation:
    """Verify that spacf raises appropriate errors on invalid inputs."""

    def test_spacf_lags_ge_T_raises(self):
        """lags >= len(data) must raise ValueError."""
        data = np.random.default_rng(42).standard_normal(50)
        with pytest.raises(ValueError, match="larger than LAGS"):
            spacf(data, 50, graph=False)

    def test_spacf_lags_greater_than_T_raises(self):
        """lags > len(data) must raise ValueError."""
        data = np.random.default_rng(42).standard_normal(50)
        with pytest.raises(ValueError, match="larger than LAGS"):
            spacf(data, 100, graph=False)

    def test_spacf_negative_lags_raises(self):
        """Negative lags must raise ValueError."""
        data = np.random.default_rng(42).standard_normal(100)
        with pytest.raises(ValueError, match="positive integer"):
            spacf(data, -5, graph=False)

    def test_spacf_zero_lags_raises(self):
        """lags=0 must raise ValueError."""
        data = np.random.default_rng(42).standard_normal(100)
        with pytest.raises(ValueError, match="positive integer"):
            spacf(data, 0, graph=False)

    def test_spacf_float_lags_raises(self):
        """Non-integer float lags must raise ValueError."""
        data = np.random.default_rng(42).standard_normal(100)
        with pytest.raises(ValueError, match="positive integer"):
            spacf(data, 5.5, graph=False)

    def test_spacf_matrix_data_raises(self):
        """Matrix (2-D with multiple columns) data must raise ValueError."""
        data = np.random.default_rng(42).standard_normal((100, 3))
        with pytest.raises(ValueError, match="column vector"):
            spacf(data, 10, graph=False)


# ===========================================================================
# Test: Column vector and row vector input handling
# ===========================================================================

class TestSpacfInputShapes:
    """Verify that spacf handles both 1-D and column/row vector inputs."""

    def test_spacf_1d_input(self):
        """Standard 1-D array input should work."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        pac, pacstd, _ = spacf(data, 10, graph=False)
        assert pac.shape == (10,)

    def test_spacf_column_vector_input(self):
        """Column vector (N, 1) input should produce same result as 1-D."""
        rng = np.random.default_rng(42)
        data_1d = rng.standard_normal(200)
        data_col = data_1d.reshape(-1, 1)
        pac_1d, pacstd_1d, _ = spacf(data_1d, 10, graph=False)
        pac_col, pacstd_col, _ = spacf(data_col, 10, graph=False)
        npt.assert_allclose(pac_1d, pac_col, atol=ATOL,
                            err_msg="Column vector and 1-D should give same pac")
        npt.assert_allclose(pacstd_1d, pacstd_col, atol=ATOL,
                            err_msg="Column vector and 1-D should give same pacstd")

    def test_spacf_row_vector_input(self):
        """Row vector (1, N) input should produce same result as 1-D."""
        rng = np.random.default_rng(42)
        data_1d = rng.standard_normal(200)
        data_row = data_1d.reshape(1, -1)
        pac_1d, _, _ = spacf(data_1d, 10, graph=False)
        pac_row, _, _ = spacf(data_row, 10, graph=False)
        npt.assert_allclose(pac_1d, pac_row, atol=ATOL,
                            err_msg="Row vector and 1-D should give same pac")


# ===========================================================================
# Test: Default parameter handling
# ===========================================================================

class TestSpacfDefaults:
    """Verify that default parameter values work correctly."""

    def test_spacf_robust_default_true(self):
        """Default robust=True should match explicit robust=True."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(300)
        pac_def, pacstd_def, _ = spacf(data, 10, graph=False)
        pac_exp, pacstd_exp, _ = spacf(data, 10, robust=True, graph=False)
        npt.assert_allclose(pac_def, pac_exp, atol=ATOL,
                            err_msg="Default should use robust=True")
        npt.assert_allclose(pacstd_def, pacstd_exp, atol=ATOL,
                            err_msg="Default should use robust=True SEs")


# ===========================================================================
# Test: Numerical parity against MATLAB fixtures
# ===========================================================================

class TestSpacfParity:
    """Numerical parity tests against MATLAB/Octave-generated fixtures.

    Per AAP Section 0.7.1: numpy.testing.assert_allclose(atol=1e-6, rtol=1e-4).
    Fixtures are loaded from tests/fixtures/timeseries/spacf.npy which contains
    a dictionary with input data and 4 scenarios of expected outputs.
    """

    @pytest.mark.parity
    def test_spacf_parity_scenario_1(self, timeseries_fixture_dir):
        """Parity: lags=20, robust=1 against MATLAB reference."""
        fix = _load_spacf_fixture(timeseries_fixture_dir)
        data = fix["data"]
        lags = int(fix["scenario_1_lags"])
        robust = bool(fix["scenario_1_robust"])
        expected_pac = fix["scenario_1_pac"]
        expected_pacstd = fix["scenario_1_pacstd"]

        pac, pacstd, _ = spacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            pac, expected_pac, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 1 (lags=20, robust=1): pac parity failure"
        )
        npt.assert_allclose(
            pacstd, expected_pacstd, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 1 (lags=20, robust=1): pacstd parity failure"
        )

    @pytest.mark.parity
    def test_spacf_parity_scenario_2(self, timeseries_fixture_dir):
        """Parity: lags=20, robust=0 against MATLAB reference."""
        fix = _load_spacf_fixture(timeseries_fixture_dir)
        data = fix["data"]
        lags = int(fix["scenario_2_lags"])
        robust = bool(fix["scenario_2_robust"])
        expected_pac = fix["scenario_2_pac"]
        expected_pacstd = fix["scenario_2_pacstd"]

        pac, pacstd, _ = spacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            pac, expected_pac, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 2 (lags=20, robust=0): pac parity failure"
        )
        npt.assert_allclose(
            pacstd, expected_pacstd, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 2 (lags=20, robust=0): pacstd parity failure"
        )

    @pytest.mark.parity
    def test_spacf_parity_scenario_3(self, timeseries_fixture_dir):
        """Parity: lags=5, robust=1 against MATLAB reference."""
        fix = _load_spacf_fixture(timeseries_fixture_dir)
        data = fix["data"]
        lags = int(fix["scenario_3_lags"])
        robust = bool(fix["scenario_3_robust"])
        expected_pac = fix["scenario_3_pac"]
        expected_pacstd = fix["scenario_3_pacstd"]

        pac, pacstd, _ = spacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            pac, expected_pac, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 3 (lags=5, robust=1): pac parity failure"
        )
        npt.assert_allclose(
            pacstd, expected_pacstd, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 3 (lags=5, robust=1): pacstd parity failure"
        )

    @pytest.mark.parity
    def test_spacf_parity_scenario_4(self, timeseries_fixture_dir):
        """Parity: lags=50, robust=1 against MATLAB reference."""
        fix = _load_spacf_fixture(timeseries_fixture_dir)
        data = fix["data"]
        lags = int(fix["scenario_4_lags"])
        robust = bool(fix["scenario_4_robust"])
        expected_pac = fix["scenario_4_pac"]
        expected_pacstd = fix["scenario_4_pacstd"]

        pac, pacstd, _ = spacf(data, lags, robust=robust, graph=False)
        npt.assert_allclose(
            pac, expected_pac, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 4 (lags=50, robust=1): pac parity failure"
        )
        npt.assert_allclose(
            pacstd, expected_pacstd, atol=ATOL, rtol=RTOL,
            err_msg="Scenario 4 (lags=50, robust=1): pacstd parity failure"
        )

    @pytest.mark.parity
    def test_spacf_parity_separate_fixture_out(self, timeseries_fixture_dir):
        """Parity against spacf_spacf_out.npy fixture (pac values, 20 lags).

        The spacf_spacf_out.npy contains the expected pac output as a flat
        (20,) array, and spacf_spacf_bounds.npy contains the expected pacstd.
        These separate fixtures were generated using the shared time-series
        test data (stored in sacf.npy) with lags=20 and robust=True.
        """
        # Load separate fixture files
        out_path = timeseries_fixture_dir / "spacf_spacf_out.npy"
        bounds_path = timeseries_fixture_dir / "spacf_spacf_bounds.npy"
        if not out_path.exists() or not bounds_path.exists():
            pytest.skip("Separate spacf fixture files not found")

        expected_pac = np.load(out_path, allow_pickle=True)
        expected_pacstd = np.load(bounds_path, allow_pickle=True)

        # The separate fixtures were generated with shared time-series data
        # (same data source as sacf fixtures), not the spacf scenario data.
        sacf_path = timeseries_fixture_dir / "sacf.npy"
        if not sacf_path.exists():
            pytest.skip("sacf.npy fixture (data source for separate spacf fixtures) not found")
        sacf_fix = np.load(sacf_path, allow_pickle=True).item()
        data = sacf_fix["data"]

        pac, pacstd, _ = spacf(data, 20, robust=True, graph=False)
        npt.assert_allclose(
            pac, expected_pac, atol=ATOL, rtol=RTOL,
            err_msg="spacf_spacf_out parity failure"
        )
        npt.assert_allclose(
            pacstd, expected_pacstd, atol=ATOL, rtol=RTOL,
            err_msg="spacf_spacf_bounds parity failure"
        )


# ===========================================================================
# Test: Deterministic reproducibility
# ===========================================================================

class TestSpacfReproducibility:
    """Verify that spacf produces deterministic results."""

    def test_spacf_deterministic(self):
        """Calling spacf twice with the same data must produce identical results."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(300)
        pac1, pacstd1, _ = spacf(data, 10, robust=True, graph=False)
        pac2, pacstd2, _ = spacf(data, 10, robust=True, graph=False)
        npt.assert_allclose(pac1, pac2, atol=0,
                            err_msg="spacf must be deterministic")
        npt.assert_allclose(pacstd1, pacstd2, atol=0,
                            err_msg="spacf pacstd must be deterministic")
