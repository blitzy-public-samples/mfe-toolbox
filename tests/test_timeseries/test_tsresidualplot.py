"""
Pytest tests for mfe_toolbox.timeseries.tsresidualplot.

Tests the matplotlib-based time series residual diagnostic plot function
migrated from MATLAB ``timeseries/tsresidualplot.m`` (Kevin Sheppard,
MFE Toolbox Version 4.0).

The MATLAB function produces a 2-panel figure:
  - Top panel: observed data *y* overlaid with fitted values (y − errors)
  - Bottom panel: residuals (*errors*)

The Python version returns ``(axes_list, figure)`` where *axes_list* is a
two-element list of ``matplotlib.axes.Axes`` and *figure* is a
``matplotlib.figure.Figure``.

Test categories:
  - Return type / shape validation
  - Basic invocation (no-error smoke tests)
  - Optional *dates* parameter
  - Input validation (mismatched lengths, non-vector, non-numeric dates)
  - Subplot structure verification
  - Numerical parity of plotted data (yhat = y − errors)
  - Figure cleanup to prevent memory leaks
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

# Configure non-interactive backend BEFORE importing pyplot.
# This is required for headless test execution in CI environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mfe_toolbox.timeseries.tsresidualplot import tsresidualplot

# Numerical parity tolerance constants (per AAP Section 0.7.1)
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible (y, errors) data for testing.

    Uses a seeded RNG (seed=42) to create a T=100 random walk *y* and
    small-amplitude Gaussian *errors*, matching the pattern recommended
    in the AAP agent_prompt example.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y, errors)`` where each has shape ``(100,)``.
    """
    rng = np.random.default_rng(42)
    T = 100
    y = np.cumsum(rng.standard_normal(T))
    errors = rng.standard_normal(T) * 0.1
    return y, errors


@pytest.fixture
def sample_data_with_dates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate reproducible (y, errors, dates) data for testing with dates.

    Uses a seeded RNG (seed=42) to create T=100 data plus a MATLAB-style
    datenum array starting at 733043 (approx 01-Jan-2007).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(y, errors, dates)`` where each has shape ``(100,)``.
    """
    rng = np.random.default_rng(42)
    T = 100
    y = np.cumsum(rng.standard_normal(T))
    errors = rng.standard_normal(T) * 0.1
    # Ref: tsresidualplot.m:38 — MATLAB datenum example
    dates = np.arange(733043, 733043 + T, dtype=np.float64)
    return y, errors, dates


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Autouse fixture that closes all matplotlib figures after each test.

    Prevents memory leaks from accumulated figures during the test session.
    Runs automatically for every test function in this module.
    """
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Test: Return Type and Structure
# ---------------------------------------------------------------------------

class TestTsresidualplotReturnTypes:
    """Tests verifying the return types and structure of tsresidualplot."""

    def test_tsresidualplot_returns_two(self, sample_data: tuple) -> None:
        """tsresidualplot returns a 2-tuple of (axes_list, figure).

        Ref: tsresidualplot.m:133-142 — MATLAB varargout returns [haxis, hfig].
        """
        y, errors = sample_data
        result = tsresidualplot(y, errors)

        # Must return a tuple/sequence with exactly 2 elements
        assert isinstance(result, tuple), "Return value must be a tuple"
        assert len(result) == 2, "Return tuple must have exactly 2 elements"

    def test_tsresidualplot_axes_list_type(self, sample_data: tuple) -> None:
        """First return element is a list of matplotlib Axes objects.

        Ref: tsresidualplot.m:136 — varargout{1} = [s1; s2] (axis handles).
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)

        assert isinstance(axes_list, list), "First element must be a list"
        assert len(axes_list) == 2, "Axes list must have exactly 2 elements"
        for i, ax in enumerate(axes_list):
            assert isinstance(ax, Axes), (
                f"axes_list[{i}] must be a matplotlib Axes instance, "
                f"got {type(ax).__name__}"
            )

    def test_tsresidualplot_figure_type(self, sample_data: tuple) -> None:
        """Second return element is a matplotlib Figure object.

        Ref: tsresidualplot.m:139 — varargout{2} = fig.
        """
        y, errors = sample_data
        _, fig = tsresidualplot(y, errors)

        assert isinstance(fig, Figure), (
            f"Second element must be a matplotlib Figure, got {type(fig).__name__}"
        )


# ---------------------------------------------------------------------------
# Test: Basic Invocation
# ---------------------------------------------------------------------------

class TestTsresidualplotBasic:
    """Tests verifying basic invocation completes without error."""

    def test_tsresidualplot_no_error(self) -> None:
        """Basic call with minimal valid inputs completes without error.

        Directly implements the example from the agent_prompt specification.
        """
        rng = np.random.default_rng(42)
        T = 100
        y = np.cumsum(rng.standard_normal(T))
        errors = rng.standard_normal(T) * 0.1
        result = tsresidualplot(y, errors)
        assert result is not None

    def test_tsresidualplot_small_input(self) -> None:
        """Function works with very small inputs (T=3)."""
        y = np.array([1.0, 2.0, 3.0])
        errors = np.array([0.1, -0.1, 0.05])
        axes_list, fig = tsresidualplot(y, errors)
        assert len(axes_list) == 2
        assert isinstance(fig, Figure)

    def test_tsresidualplot_single_element(self) -> None:
        """Function works with T=1 (single observation)."""
        y = np.array([5.0])
        errors = np.array([0.1])
        axes_list, fig = tsresidualplot(y, errors)
        assert len(axes_list) == 2
        assert isinstance(fig, Figure)

    def test_tsresidualplot_large_input(self) -> None:
        """Function works with large T=5000 array without error."""
        rng = np.random.default_rng(123)
        T = 5000
        y = np.cumsum(rng.standard_normal(T))
        errors = rng.standard_normal(T) * 0.5
        axes_list, fig = tsresidualplot(y, errors)
        assert len(axes_list) == 2
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Test: Dates Parameter
# ---------------------------------------------------------------------------

class TestTsresidualplotDates:
    """Tests verifying the optional dates parameter."""

    def test_tsresidualplot_with_dates(
        self, sample_data_with_dates: tuple
    ) -> None:
        """Function accepts and uses the dates parameter without error.

        Ref: tsresidualplot.m:36-39 — dates example with datenum.
        """
        y, errors, dates = sample_data_with_dates
        axes_list, fig = tsresidualplot(y, errors, dates=dates)

        assert isinstance(axes_list, list)
        assert len(axes_list) == 2
        assert isinstance(fig, Figure)

    def test_tsresidualplot_none_dates(self, sample_data: tuple) -> None:
        """Explicitly passing dates=None uses observation index (default).

        Ref: tsresidualplot.m:73-75 — dates = (1:T)' when not provided.
        """
        y, errors = sample_data
        axes_list, fig = tsresidualplot(y, errors, dates=None)
        assert len(axes_list) == 2
        assert isinstance(fig, Figure)

    def test_tsresidualplot_integer_dates(self) -> None:
        """Integer date arrays are accepted (auto-converted to float64)."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        errors = np.array([0.1, -0.1, 0.05, -0.05, 0.02])
        dates = np.arange(1, 6)  # integer array
        axes_list, fig = tsresidualplot(y, errors, dates=dates)
        assert len(axes_list) == 2

    def test_tsresidualplot_float_dates(self) -> None:
        """Float dates (e.g. MATLAB datenum style) are accepted."""
        y = np.array([10.0, 11.0, 12.0])
        errors = np.array([0.5, -0.3, 0.1])
        dates = np.array([733043.0, 733044.0, 733045.0])
        axes_list, fig = tsresidualplot(y, errors, dates=dates)
        assert len(axes_list) == 2


# ---------------------------------------------------------------------------
# Test: Input Validation — Error Conditions
# ---------------------------------------------------------------------------

class TestTsresidualplotInputValidation:
    """Tests verifying input validation and error raising.

    These tests correspond to MATLAB ``error(...)`` calls in
    tsresidualplot.m:51-88 which must raise Python ``ValueError``
    per AAP Section 0.7.1.
    """

    def test_tsresidualplot_mismatched_lengths(self) -> None:
        """Mismatched y and errors lengths raise ValueError.

        Ref: tsresidualplot.m:69-71 — 'Y and ERRORS must have the same
        dimensions'.
        """
        y = np.array([1.0, 2.0, 3.0, 4.0])
        errors = np.array([0.1, -0.1])
        with pytest.raises(ValueError):
            tsresidualplot(y, errors)

    def test_tsresidualplot_mismatched_lengths_reversed(self) -> None:
        """Errors longer than y also raises ValueError."""
        y = np.array([1.0, 2.0])
        errors = np.array([0.1, -0.1, 0.05, -0.05])
        with pytest.raises(ValueError):
            tsresidualplot(y, errors)

    def test_tsresidualplot_y_matrix_raises(self) -> None:
        """Non-vector (2-D matrix) y raises ValueError.

        Ref: tsresidualplot.m:62-64 — 'Y must be a column vector of data'.
        """
        y = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2 matrix
        errors = np.array([0.1, -0.1])
        with pytest.raises(ValueError):
            tsresidualplot(y, errors)

    def test_tsresidualplot_errors_matrix_raises(self) -> None:
        """Non-vector (2-D matrix) errors raises ValueError.

        Ref: tsresidualplot.m:66-68 — 'Y must be a column vector of data'
        (MATLAB has a bug in the error message; Python corrects to ERRORS).
        """
        y = np.array([1.0, 2.0])
        errors = np.array([[0.1, -0.1], [0.05, -0.05]])  # 2x2 matrix
        with pytest.raises(ValueError):
            tsresidualplot(y, errors)

    def test_tsresidualplot_string_dates_raises(self) -> None:
        """Non-numeric (string) dates raise ValueError.

        Ref: tsresidualplot.m:86-88 — 'DATES must contain numeric MATLAB
        dates, not strings.'
        """
        y = np.array([1.0, 2.0, 3.0])
        errors = np.array([0.1, -0.1, 0.05])
        with pytest.raises(ValueError):
            tsresidualplot(y, errors, dates=["2007-01-01", "2007-01-02", "2007-01-03"])

    def test_tsresidualplot_dates_matrix_raises(self) -> None:
        """Non-vector (matrix) dates raise ValueError.

        Ref: tsresidualplot.m:80-84 — 'DATES must be a column vector with
        the same dimensions as Y'.
        """
        y = np.array([1.0, 2.0])
        errors = np.array([0.1, -0.1])
        dates = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2 matrix
        with pytest.raises(ValueError):
            tsresidualplot(y, errors, dates=dates)


# ---------------------------------------------------------------------------
# Test: Subplot Structure
# ---------------------------------------------------------------------------

class TestTsresidualplotSubplots:
    """Tests verifying the figure contains exactly 2 subplots with correct
    titles and legend content."""

    def test_tsresidualplot_figure_has_two_subplots(
        self, sample_data: tuple
    ) -> None:
        """Figure must contain exactly 2 axes (top and bottom subplots).

        Ref: tsresidualplot.m:99,116 — subplot(2,1,1) and subplot(2,1,2).
        """
        y, errors = sample_data
        axes_list, fig = tsresidualplot(y, errors)

        # The figure should have exactly 2 axes objects
        assert len(fig.axes) == 2, (
            f"Expected 2 axes in figure, got {len(fig.axes)}"
        )
        # The returned axes list should match the figure's axes
        assert axes_list[0] in fig.axes
        assert axes_list[1] in fig.axes

    def test_tsresidualplot_top_subplot_title(
        self, sample_data: tuple
    ) -> None:
        """Top subplot has title 'Data and Fit'.

        Ref: tsresidualplot.m:111 — title('Data and Fit').
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)
        assert axes_list[0].get_title() == "Data and Fit"

    def test_tsresidualplot_bottom_subplot_title(
        self, sample_data: tuple
    ) -> None:
        """Bottom subplot has title 'Residual'.

        Ref: tsresidualplot.m:128 — title('Residual').
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)
        assert axes_list[1].get_title() == "Residual"

    def test_tsresidualplot_top_subplot_has_two_lines(
        self, sample_data: tuple
    ) -> None:
        """Top subplot has 2 lines: data and fitted values.

        Ref: tsresidualplot.m:100 — plot(dates, [y yhat]) creates 2 lines.
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)
        lines = axes_list[0].get_lines()
        assert len(lines) == 2, (
            f"Top subplot should have 2 lines (data + fit), got {len(lines)}"
        )

    def test_tsresidualplot_bottom_subplot_has_one_line(
        self, sample_data: tuple
    ) -> None:
        """Bottom subplot has 1 line: residuals.

        Ref: tsresidualplot.m:117 — plot(dates, errors) creates 1 line.
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)
        lines = axes_list[1].get_lines()
        assert len(lines) == 1, (
            f"Bottom subplot should have 1 line (residual), got {len(lines)}"
        )

    def test_tsresidualplot_legend_labels(
        self, sample_data: tuple
    ) -> None:
        """Subplots have correct legend labels.

        Ref: tsresidualplot.m:107 — legend('Data','Fit') for top.
        Ref: tsresidualplot.m:124 — legend('Residual') for bottom.
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)

        # Top subplot legend
        top_legend = axes_list[0].get_legend()
        assert top_legend is not None, "Top subplot must have a legend"
        top_labels = [t.get_text() for t in top_legend.get_texts()]
        assert "Data" in top_labels, "Top legend must contain 'Data'"
        assert "Fit" in top_labels, "Top legend must contain 'Fit'"

        # Bottom subplot legend
        bot_legend = axes_list[1].get_legend()
        assert bot_legend is not None, "Bottom subplot must have a legend"
        bot_labels = [t.get_text() for t in bot_legend.get_texts()]
        assert "Residual" in bot_labels, "Bottom legend must contain 'Residual'"


# ---------------------------------------------------------------------------
# Test: Numerical Parity of Plotted Data
# ---------------------------------------------------------------------------

class TestTsresidualplotParity:
    """Tests verifying the plotted data matches expected numerical values.

    The fitted values line in the top subplot must equal y − errors
    (yhat = y − errors), and the bottom subplot must plot the raw errors.
    This verifies the core computation from tsresidualplot.m:96.
    """

    def test_tsresidualplot_parity_yhat(self, sample_data: tuple) -> None:
        """Fitted values plotted in top subplot equal y − errors.

        Ref: tsresidualplot.m:96 — yhat = y - errors.
        Uses ATOL=1e-6, RTOL=1e-4 per AAP Section 0.7.1.
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)

        # The top subplot has 2 lines: line 0 = y, line 1 = yhat
        lines = axes_list[0].get_lines()
        plotted_y = lines[0].get_ydata()
        plotted_yhat = lines[1].get_ydata()

        # Verify the y data matches input
        npt.assert_allclose(plotted_y, y, atol=ATOL, rtol=RTOL,
                            err_msg="Top subplot line 0 should be y")

        # Verify yhat = y - errors
        expected_yhat = y - errors
        npt.assert_allclose(plotted_yhat, expected_yhat, atol=ATOL, rtol=RTOL,
                            err_msg="Top subplot line 1 should be y - errors")

    def test_tsresidualplot_parity_errors(self, sample_data: tuple) -> None:
        """Residuals plotted in bottom subplot equal input errors.

        Ref: tsresidualplot.m:117 — plot(dates, errors).
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)

        # The bottom subplot has 1 line: the residuals
        lines = axes_list[1].get_lines()
        plotted_errors = lines[0].get_ydata()

        npt.assert_allclose(plotted_errors, errors, atol=ATOL, rtol=RTOL,
                            err_msg="Bottom subplot should plot errors")

    def test_tsresidualplot_parity_xdata_default(self) -> None:
        """When dates is None, x-axis data is 1-based observation index.

        Ref: tsresidualplot.m:73-75 — dates = (1:T)' (MATLAB 1-based).
        """
        T = 50
        rng = np.random.default_rng(99)
        y = np.cumsum(rng.standard_normal(T))
        errors = rng.standard_normal(T) * 0.2

        axes_list, _ = tsresidualplot(y, errors)

        # Check x-data of the first line in the top subplot
        lines = axes_list[0].get_lines()
        plotted_x = lines[0].get_xdata()
        expected_x = np.arange(1, T + 1, dtype=np.float64)
        npt.assert_allclose(plotted_x, expected_x, atol=ATOL, rtol=RTOL,
                            err_msg="Default x-axis should be 1-based index")

    def test_tsresidualplot_parity_xdata_dates(self) -> None:
        """When dates are provided, x-axis data matches the dates array.

        Ref: tsresidualplot.m:100 — plot(dates, [y yhat]).
        """
        T = 30
        y = np.arange(T, dtype=np.float64) * 1.5
        errors = np.ones(T) * 0.1
        dates = np.arange(733043, 733043 + T, dtype=np.float64)

        axes_list, _ = tsresidualplot(y, errors, dates=dates)

        # Both subplots should use the provided dates as x-data
        top_x = axes_list[0].get_lines()[0].get_xdata()
        bot_x = axes_list[1].get_lines()[0].get_xdata()

        npt.assert_allclose(top_x, dates, atol=ATOL, rtol=RTOL,
                            err_msg="Top subplot x-axis should match dates")
        npt.assert_allclose(bot_x, dates, atol=ATOL, rtol=RTOL,
                            err_msg="Bottom subplot x-axis should match dates")


# ---------------------------------------------------------------------------
# Test: Auto-Transpose Behavior
# ---------------------------------------------------------------------------

class TestTsresidualplotTranspose:
    """Tests verifying auto-transpose of row vectors to column vectors.

    Ref: tsresidualplot.m:55-60 — if size(y,2)>size(y,1), y=y'.
    """

    def test_tsresidualplot_row_vector_both(self) -> None:
        """Both y and errors as row vectors (1, T) are auto-transposed.

        Ref: tsresidualplot.m:55-60 — both get transposed to (T, 1)
        so their shapes match after transpose.
        """
        y = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # shape (1, 5)
        errors = np.array([[0.1, -0.1, 0.05, -0.05, 0.02]])  # shape (1, 5)
        axes_list, fig = tsresidualplot(y, errors)
        assert len(axes_list) == 2
        assert isinstance(fig, Figure)

    def test_tsresidualplot_mixed_shapes_raises(self) -> None:
        """Mixing 2D row vector (1, T) with 1D (T,) raises ValueError.

        After auto-transpose the 2D becomes (T, 1) while the 1D stays (T,),
        causing a shape mismatch per tsresidualplot.m:69-71.
        """
        y = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # shape (1, 5) -> (5, 1)
        errors = np.array([0.1, -0.1, 0.05, -0.05, 0.02])  # shape (5,)
        with pytest.raises(ValueError):
            tsresidualplot(y, errors)

    def test_tsresidualplot_column_vector_2d(self) -> None:
        """Column vectors with shape (T, 1) are accepted."""
        y = np.array([[1.0], [2.0], [3.0]])  # shape (3, 1)
        errors = np.array([[0.1], [-0.1], [0.05]])  # shape (3, 1)
        axes_list, fig = tsresidualplot(y, errors)
        assert len(axes_list) == 2
        assert isinstance(fig, Figure)

    def test_tsresidualplot_row_vector_dates(self) -> None:
        """Row vector dates (1, T) is auto-transposed and accepted.

        Ref: tsresidualplot.m:77-79 — if size(dates,2)>size(dates,1), dates=dates'.
        """
        y = np.array([1.0, 2.0, 3.0])
        errors = np.array([0.1, -0.1, 0.05])
        dates = np.array([[733043.0, 733044.0, 733045.0]])  # shape (1, 3)
        axes_list, fig = tsresidualplot(y, errors, dates=dates)
        assert len(axes_list) == 2


# ---------------------------------------------------------------------------
# Test: Figure Styling
# ---------------------------------------------------------------------------

class TestTsresidualplotStyling:
    """Tests verifying visual styling properties match MATLAB specifications."""

    def test_tsresidualplot_line_widths(self, sample_data: tuple) -> None:
        """Plot lines have linewidth=2 as specified.

        Ref: tsresidualplot.m:112,129 — set(h1/h2, 'LineWidth', 2).
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)

        for line in axes_list[0].get_lines():
            assert line.get_linewidth() == 2.0, (
                "Top subplot lines must have linewidth=2"
            )
        for line in axes_list[1].get_lines():
            assert line.get_linewidth() == 2.0, (
                "Bottom subplot lines must have linewidth=2"
            )

    def test_tsresidualplot_title_font_size(self, sample_data: tuple) -> None:
        """Subplot titles have fontsize=12.

        Ref: tsresidualplot.m:114,131 — set(t1/t2, 'FontSize', 12).
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)

        assert axes_list[0].title.get_fontsize() == 12
        assert axes_list[1].title.get_fontsize() == 12

    def test_tsresidualplot_spine_widths(self, sample_data: tuple) -> None:
        """Subplot spines have linewidth=2.

        Ref: tsresidualplot.m:113,130 — set(s1/s2, 'LineWidth', 2).
        """
        y, errors = sample_data
        axes_list, _ = tsresidualplot(y, errors)

        for ax in axes_list:
            for spine in ax.spines.values():
                assert spine.get_linewidth() == 2.0, (
                    f"Spine linewidth should be 2.0, got {spine.get_linewidth()}"
                )


# ---------------------------------------------------------------------------
# Test: Cleanup (Explicit)
# ---------------------------------------------------------------------------

class TestTsresidualplotCleanup:
    """Tests verifying proper figure cleanup behavior."""

    def test_tsresidualplot_cleanup_close_all(self) -> None:
        """Closing all figures after tsresidualplot leaves no open figures.

        Ensures the autouse cleanup_figures fixture correctly cleans up.
        """
        rng = np.random.default_rng(42)
        T = 50
        y = np.cumsum(rng.standard_normal(T))
        errors = rng.standard_normal(T) * 0.1

        # Create a figure
        _, fig = tsresidualplot(y, errors)
        assert fig is not None

        # Explicitly close all figures
        plt.close("all")

        # Verify no figures remain open
        assert len(plt.get_fignums()) == 0, "All figures should be closed"

    def test_tsresidualplot_multiple_calls(self) -> None:
        """Multiple sequential calls produce independent figures."""
        rng = np.random.default_rng(42)
        T = 30
        y1 = np.cumsum(rng.standard_normal(T))
        e1 = rng.standard_normal(T) * 0.1
        y2 = np.cumsum(rng.standard_normal(T))
        e2 = rng.standard_normal(T) * 0.2

        _, fig1 = tsresidualplot(y1, e1)
        _, fig2 = tsresidualplot(y2, e2)

        # Both figures should exist and be distinct
        assert fig1 is not fig2, "Each call should create a new figure"
        assert fig1.number != fig2.number, "Figures should have different numbers"
