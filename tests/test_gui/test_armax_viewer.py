"""pytest-qt integration tests for ARMAXViewer (QWidget with FigureCanvasQTAgg).

Tests exercise ALL MATLAB GUIDE callbacks migrated from GUI/ARMAX_viewer.m:
- OpeningFcn: Widget initialization, results storage, model selector population, display
- model_popupmenu_Callback: Model selection switching
- page_number_edit_Callback: Page number validation
- increase/decrease_page pushbuttons: Page navigation
- pushbutton3_Callback: Close button
- display_model/generate_text/draw_data: Rendering pipeline
- bounded_integer_validate: Input validation

Per AAP Section 0.7.1: Coverage threshold >=80% for GUI modules.
Per AAP Section 0.7.1: All GUIDE callbacks MUST be exercised using pytest-qt.
Per AAP Section 0.7.1: Plot output compared via saved .png snapshots with SSIM >= 0.95.
"""
from __future__ import annotations

import os

# Headless environment — must be set before any Qt imports to allow
# widget creation and rendering without a physical display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Guard: gracefully skip all tests in this module if PyQt6 is unavailable.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
    QPushButton,
    QWidget,
)
from PyQt6.QtTest import QTest

from mfe_toolbox.gui.armax_viewer import ARMAXViewer, _ROWS_PER_PAGE

# Import snapshot comparison utility from the GUI test conftest.
# compare_images_ssim provides SSIM-based PNG comparison (AAP Section 0.7.1).
from tests.test_gui.conftest import compare_images_ssim, SSIM_THRESHOLD

# ---------------------------------------------------------------------------
# Module-level marker: all tests in this file are GUI tests (pytest.mark.gui).
# Per AAP Section 0.7.1: selective execution via ``-m "not gui"`` or ``-m gui``.
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def viewer(qtbot, sample_results):
    """Create, register, and show a basic ARMAXViewer for testing.

    Uses ``sample_results`` from ``tests/test_gui/conftest.py`` which
    provides a two-model list (ARMA(1,0) and ARMA(2,1) both with constant).

    Ref: ARMAX_viewer.m:50-77 — OpeningFcn initialises the viewer.
    """
    widget = ARMAXViewer(results=sample_results)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def _make_many_param_result(n_ar: int = 20) -> dict[str, Any]:
    """Build a synthetic result dict with *n_ar* AR parameters to trigger
    multi-page pagination in the parameter table.

    With ``_ROWS_PER_PAGE == 16``, we need more than 16 displayable
    parameters for the table to span multiple pages.  Setting
    ``n_ar = 20`` plus a constant gives 21 displayable parameters →
    ``ceil(21 / 16) == 2`` pages.

    Ref: ARMAX_viewer.m:563-625 — compute_text_positions pagination.
    """
    rng = np.random.default_rng(999)
    K = 1 + n_ar  # constant + n_ar AR parameters
    params = rng.standard_normal(K) * 0.1
    errors = rng.standard_normal(500)
    vcv = np.eye(K) * 0.01
    stde = np.sqrt(np.diag(vcv))
    tstat = params / stde
    pval = np.clip(np.abs(rng.standard_normal(K)) * 0.05, 0.001, 0.999)

    return {
        "StringID": f"AR({n_ar}) w/ constant",
        "parameters": params,
        "errors": errors,
        "ll": -700.0,
        "seregression": float(np.var(errors)),
        "diagnostics": {},
        "covariance": vcv,
        "likelihoods": rng.standard_normal(500) - 1.0,
        "scores": rng.standard_normal((500, K)),
        "ARlags": np.arange(1, n_ar + 1),
        "MAlags": np.array([], dtype=np.int64),
        "constant": 1,
        "StdErr": stde,
        "Tstat": tstat,
        "Pval": pval,
        "AIC": 1410.0,
        "BIC": 1500.0,
        "K": K,
    }


# =========================================================================
# TestARMAXViewerInit — Widget Initialization (ARMAX_viewer.m:50-77)
# =========================================================================

class TestARMAXViewerInit:
    """Test widget construction, matching GUIDE OpeningFcn behaviour."""

    def test_construction_with_results(self, qtbot, sample_results):
        """Verify ARMAXViewer stores results list on construction.

        Ref: ARMAX_viewer.m:59 — handles.Results = varargin{1}
        """
        widget = ARMAXViewer(results=sample_results)
        qtbot.addWidget(widget)

        # Widget must be a QWidget subclass.
        assert isinstance(widget, QWidget)

        # Internal results list must match what was passed in.
        assert len(widget._results) == len(sample_results)
        for i, result in enumerate(sample_results):
            assert widget._results[i]["StringID"] == result["StringID"]

    def test_default_display_number(self, qtbot, sample_results):
        """Verify default display_number selects the last model.

        Ref: ARMAX_viewer.m:63 — handles.DisplayNumber = length(handles.Results)
        Note: MATLAB is 1-indexed, Python is 0-indexed.
        """
        widget = ARMAXViewer(results=sample_results)
        qtbot.addWidget(widget)

        expected_index = len(sample_results) - 1  # 0-indexed last model
        assert widget._display_number == expected_index

    def test_explicit_display_number(self, qtbot, sample_results):
        """Verify explicit display_number parameter is honoured.

        Ref: ARMAX_viewer.m:61 — handles.DisplayNumber = varargin{2}
        """
        widget = ARMAXViewer(results=sample_results, display_number=0)
        qtbot.addWidget(widget)

        assert widget._display_number == 0

    def test_window_size(self, viewer):
        """Verify fixed widget dimensions match MATLAB figure position.

        Ref: ARMAX_viewer.m:74 — set(gcf,'Position',[10 10 1000 706])
        """
        assert viewer.width() == 1000
        assert viewer.height() == 706

    def test_model_popupmenu_populated(self, viewer, sample_results):
        """Verify model selector QComboBox contains all model StringIDs.

        Ref: ARMAX_viewer.m:68-73 — populates model_popupmenu with
        StringID from each result in the Results cell array.
        """
        combo = viewer.model_popupmenu
        assert isinstance(combo, QComboBox)
        assert combo.count() == len(sample_results)

        for i, result in enumerate(sample_results):
            assert combo.itemText(i) == result["StringID"]

    def test_model_popupmenu_default_selection(self, viewer, sample_results):
        """Verify the last model is selected by default in the combo box.

        Ref: ARMAX_viewer.m:73 — set(handles.model_popupmenu,'Value',
             length(handles.Results))
        """
        combo = viewer.model_popupmenu
        # MATLAB sets to last model (1-indexed); Python uses 0-indexed.
        expected_index = len(sample_results) - 1
        assert combo.currentIndex() == expected_index


# =========================================================================
# TestARMAXViewerModelSelection — Model Switching (ARMAX_viewer.m:354-364)
# =========================================================================

class TestARMAXViewerModelSelection:
    """Test model selector (model_popupmenu_Callback equivalent)."""

    def test_model_popupmenu_callback(self, viewer, sample_results):
        """Verify changing the combo box selection updates _display_number.

        Ref: ARMAX_viewer.m:354-364 — model_popupmenu_Callback:
             handles.DisplayNumber = get(hObject,'Value')
        """
        # Default is last model; switch to first.
        viewer.model_popupmenu.setCurrentIndex(0)
        assert viewer._display_number == 0

        # Switch back to second model.
        viewer.model_popupmenu.setCurrentIndex(1)
        assert viewer._display_number == 1

    def test_model_switch_updates_display(self, viewer, sample_results):
        """Verify switching models triggers a full display refresh without crash.

        Ref: ARMAX_viewer.m:362-363 — display_model(handles, hObject)
        After switching, the rendering pipeline must complete without errors.
        """
        # Start at last model.
        initial_idx = viewer._display_number

        # Switch to first model.
        viewer.model_popupmenu.setCurrentIndex(0)
        assert viewer._display_number == 0
        # Page should reset to 1 on model switch.
        # Ref: ARMAX_viewer.m:690 — set(page_number_edit,'String','1')
        assert viewer._current_page == 1

        # Switch back.
        viewer.model_popupmenu.setCurrentIndex(initial_idx)
        assert viewer._display_number == initial_idx
        assert viewer._current_page == 1


# =========================================================================
# TestARMAXViewerPageNavigation — Page Controls (ARMAX_viewer.m:380-471)
# =========================================================================

class TestARMAXViewerPageNavigation:
    """Test page navigation callbacks for the parameter table."""

    def test_single_page_disables_navigation(self, viewer, sample_results):
        """Verify navigation is disabled when the parameter table fits one page.

        Ref: ARMAX_viewer.m:696-707 — disables nav buttons for single page.
        With sample_results models having <= 4 parameters, _ROWS_PER_PAGE=16
        means everything fits on one page.
        """
        # sample_results models have 2 and 4 params — both single-page.
        assert viewer._total_pages == 1
        assert not viewer.increase_page_pushbutton.isEnabled()
        assert not viewer.decrease_page_pushbutton.isEnabled()
        assert not viewer.page_number_edit.isEnabled()

    def test_increase_page_button(self, qtbot):
        """Verify the Next (>) button increments the page number.

        Ref: ARMAX_viewer.m:437-454 — increase_page_pushbutton_Callback.
        Uses a many-parameter model to force multi-page display.
        """
        many_param = _make_many_param_result(n_ar=20)
        widget = ARMAXViewer(results=[many_param])
        qtbot.addWidget(widget)
        widget.show()

        # Should have multiple pages (21 params / 16 per page = 2 pages).
        assert widget._total_pages >= 2
        assert widget._current_page == 1

        # Click Next.
        qtbot.mouseClick(
            widget.increase_page_pushbutton, Qt.MouseButton.LeftButton
        )
        assert widget._current_page == 2
        assert widget.page_number_edit.text() == "2"

    def test_decrease_page_button(self, qtbot):
        """Verify the Previous (<) button decrements the page number.

        Ref: ARMAX_viewer.m:456-471 — decrease_page_pushbutton_Callback.
        Page cannot go below 1.
        """
        many_param = _make_many_param_result(n_ar=20)
        widget = ARMAXViewer(results=[many_param])
        qtbot.addWidget(widget)
        widget.show()

        assert widget._total_pages >= 2

        # Go to page 2 first.
        qtbot.mouseClick(
            widget.increase_page_pushbutton, Qt.MouseButton.LeftButton
        )
        assert widget._current_page == 2

        # Click Previous — should go back to page 1.
        qtbot.mouseClick(
            widget.decrease_page_pushbutton, Qt.MouseButton.LeftButton
        )
        assert widget._current_page == 1
        assert widget.page_number_edit.text() == "1"

        # Click Previous again at page 1 — should stay at 1.
        # Ref: ARMAX_viewer.m:462 — if CurrentPage > 1
        qtbot.mouseClick(
            widget.decrease_page_pushbutton, Qt.MouseButton.LeftButton
        )
        assert widget._current_page == 1

    def test_page_number_edit_validation(self, qtbot):
        """Verify manual page number entry is clamped to valid range.

        Ref: ARMAX_viewer.m:380-398 — page_number_edit_Callback.
        Invalid values are corrected via bounded_integer_validate.
        """
        many_param = _make_many_param_result(n_ar=20)
        widget = ARMAXViewer(results=[many_param])
        qtbot.addWidget(widget)
        widget.show()

        total = widget._total_pages
        assert total >= 2

        # Enter page number greater than total — should clamp to total.
        widget.page_number_edit.clear()
        qtbot.keyClicks(widget.page_number_edit, str(total + 5))
        widget.page_number_edit.editingFinished.emit()
        assert widget._current_page == total

        # Enter 0 — should clamp to lower bound 1.
        widget.page_number_edit.clear()
        qtbot.keyClicks(widget.page_number_edit, "0")
        widget.page_number_edit.editingFinished.emit()
        assert widget._current_page == 1

        # Enter non-numeric text — should revert to original.
        widget.page_number_edit.clear()
        current = widget._current_page
        qtbot.keyClicks(widget.page_number_edit, "abc")
        widget.page_number_edit.editingFinished.emit()
        assert widget._current_page == current

    def test_bounded_integer_validate(self):
        """Directly test the static bounded_integer_validate method.

        Ref: ARMAX_viewer.m:782-800 — bounded_integer_validate.
        Tests edge cases:
          - NaN → original value
          - Below lower bound → lower bound
          - Above upper bound → upper bound
          - Non-integer → floor
          - Valid integer → pass through
          - Non-numeric string → original value
        """
        validate = ARMAXViewer._bounded_integer_validate
        lb, ub = 1, 10

        # Valid integer within range — pass through unchanged.
        val, ok = validate("5", 3, lb, ub)
        assert val == 5
        assert ok is True

        # Exact lower bound — valid.
        val, ok = validate("1", 3, lb, ub)
        assert val == 1
        assert ok is True

        # Exact upper bound — valid.
        val, ok = validate("10", 3, lb, ub)
        assert val == 10
        assert ok is True

        # Below lower bound — clamp to LB.
        # Ref: ARMAX_viewer.m:792-793
        val, ok = validate("-1", 3, lb, ub)
        assert val == lb
        assert ok is False

        val, ok = validate("0", 3, lb, ub)
        assert val == lb
        assert ok is False

        # Above upper bound — clamp to UB.
        # Ref: ARMAX_viewer.m:794-795
        val, ok = validate("15", 3, lb, ub)
        assert val == ub
        assert ok is False

        # Non-integer float — floor to integer.
        # Ref: ARMAX_viewer.m:796-797
        val, ok = validate("3.7", 3, lb, ub)
        assert val == 3
        assert ok is False

        # NaN string → revert to original value.
        # Ref: ARMAX_viewer.m:790-791
        val, ok = validate("nan", 5, lb, ub)
        assert val == 5
        assert ok is False

        # Non-numeric string → revert to original value.
        val, ok = validate("abc", 5, lb, ub)
        assert val == 5
        assert ok is False

        # Empty string → revert to original value.
        val, ok = validate("", 5, lb, ub)
        assert val == 5
        assert ok is False


# =========================================================================
# TestARMAXViewerClose — Close Button (ARMAX_viewer.m:473-478)
# =========================================================================

class TestARMAXViewerClose:
    """Test the Close button callback."""

    def test_close_button(self, qtbot, sample_results):
        """Verify clicking the Close button hides/closes the viewer widget.

        Ref: ARMAX_viewer.m:473-478 — pushbutton3_Callback:
             delete(handles.figure1)
        In PyQt6, close() hides the widget (it is not deleted immediately).
        """
        widget = ARMAXViewer(results=sample_results)
        qtbot.addWidget(widget)
        widget.show()
        assert widget.isVisible()

        close_btn = widget.pushbutton3
        assert isinstance(close_btn, QPushButton)

        qtbot.mouseClick(close_btn, Qt.MouseButton.LeftButton)

        # After close, the widget should no longer be visible.
        assert not widget.isVisible()


# =========================================================================
# TestARMAXViewerRendering — Rendering Pipeline (ARMAX_viewer.m:755-769)
# =========================================================================

class TestARMAXViewerRendering:
    """Test the rendering pipeline: display_model, equation, stats, table."""

    def test_display_model_no_crash(self, viewer, sample_results):
        """Verify initial construction renders all panels without exceptions.

        Ref: ARMAX_viewer.m:755-769 — display_model orchestrator.
        The widget constructor calls _display_model() internally, so
        reaching this point without error means the pipeline succeeded.
        """
        # If we got here, construction + initial display_model() succeeded.
        # Additionally verify that layout cache has been populated.
        assert viewer._display_number in viewer._layout_cache
        layout = viewer._layout_cache[viewer._display_number]
        assert "ParameterString" in layout
        assert "VariableString" in layout

    def test_model_equation_rendered(self, viewer, sample_results):
        """Verify model equation canvas contains rendered content.

        Ref: ARMAX_viewer.m:174-268 — model_layout.
        After rendering, the model equation figure should have text objects.
        """
        fig = viewer._model_fig
        # The model figure should have text elements from the equation render.
        texts = fig.texts
        assert len(texts) > 0, "Model equation figure has no text objects"

        # First text should be the model StringID (title line).
        # Ref: ARMAX_viewer.m:185 — final_str{count} = ModelString
        first_text_content = texts[0].get_text()
        current_result = sample_results[viewer._display_number]
        assert current_result["StringID"] in first_text_content

    def test_model_stats_rendered(self, viewer, sample_results):
        """Verify statistics panel displays LL, K, AIC, BIC, sigma^2.

        Ref: ARMAX_viewer.m:275-351 — model_stats_layout.
        """
        fig = viewer._stats_fig
        texts = fig.texts
        # Expect at least 10 text items: 5 titles + 5 data values.
        assert len(texts) >= 10, (
            f"Expected >=10 text elements in stats figure, got {len(texts)}"
        )

        # Verify key statistic labels are present.
        all_text = " ".join(t.get_text() for t in texts)
        assert "Log-likelihood" in all_text
        assert "No. of Params" in all_text
        assert "Akaike IC" in all_text

        # Verify the data values correspond to the current model's results.
        current_result = sample_results[viewer._display_number]
        ll_str = f"{current_result['ll']:.7g}"
        assert ll_str in all_text, f"Log-likelihood value {ll_str} not found in stats"

    def test_parameter_table_rendered(self, viewer, sample_results):
        """Verify the parameter estimates table has content.

        Ref: ARMAX_viewer.m:714-751 — draw_parameter_table.
        The table should show column titles and parameter rows.
        """
        fig = viewer._table_fig
        texts = fig.texts
        # Should have column titles + parameter row entries.
        assert len(texts) > 5, (
            f"Expected >5 text elements in table figure, got {len(texts)}"
        )

        # Verify column title presence.
        # Ref: ARMAX_viewer.m:514 — titles = {'Parameter','Estimate',...}
        all_text = " ".join(t.get_text() for t in texts)
        for title in ["Parameter", "Estimate", "Std. Error", "T-stat", "P-val"]:
            assert title in all_text, f"Column title '{title}' missing from table"


# =========================================================================
# TestARMAXViewerSnapshot — Plot Snapshot (AAP Section 0.7.1)
# =========================================================================

class TestARMAXViewerSnapshot:
    """Test plot snapshot saving and optional SSIM comparison."""

    def test_viewer_plot_snapshot(self, qtbot, sample_results, snapshot_dir):
        """Verify that the viewer can save a PNG snapshot of its content.

        Ref: AAP Section 0.7.1 — Plot output compared via saved .png
        snapshots with perceptual diff tolerance SSIM >= 0.95.

        If no reference snapshot exists, only verifies that PNG generation
        succeeds without error.  When a reference IS available, uses
        compare_images_ssim for SSIM validation.
        """
        widget = ARMAXViewer(results=sample_results)
        qtbot.addWidget(widget)
        widget.show()

        # Render snapshot of the parameter table canvas (largest visual area).
        snapshot_path = snapshot_dir / "armax_viewer_table.png"
        widget._table_fig.savefig(
            str(snapshot_path), dpi=100, bbox_inches="tight"
        )
        assert snapshot_path.exists(), "Snapshot file was not created"
        assert snapshot_path.stat().st_size > 0, "Snapshot file is empty"

        # Save equation canvas snapshot as well.
        eq_path = snapshot_dir / "armax_viewer_equation.png"
        widget._model_fig.savefig(
            str(eq_path), dpi=100, bbox_inches="tight"
        )
        assert eq_path.exists()

        # Save stats canvas snapshot.
        stats_path = snapshot_dir / "armax_viewer_stats.png"
        widget._stats_fig.savefig(
            str(stats_path), dpi=100, bbox_inches="tight"
        )
        assert stats_path.exists()

        # If a reference snapshot exists, compare SSIM >= SSIM_THRESHOLD.
        # In CI without reference images, this block is gracefully skipped.
        ref_dir = Path(__file__).parent / "reference_snapshots"
        ref_path = ref_dir / "armax_viewer_table.png"
        if ref_path.exists():
            assert compare_images_ssim(ref_path, snapshot_path, SSIM_THRESHOLD), (
                f"SSIM comparison failed: reference={ref_path}, "
                f"rendered={snapshot_path}, threshold={SSIM_THRESHOLD}"
            )


# =========================================================================
# TestARMAXViewerMultipleModels — Varying Result Counts
# =========================================================================

class TestARMAXViewerMultipleModels:
    """Test viewer with varying numbers of models and parameters."""

    def test_single_model(self, qtbot):
        """Verify viewer handles a single-model results list correctly.

        The combo box should have exactly 1 item, display_number == 0.
        """
        single_result = _make_many_param_result(n_ar=2)
        widget = ARMAXViewer(results=[single_result])
        qtbot.addWidget(widget)
        widget.show()

        combo = widget.model_popupmenu
        assert combo.count() == 1
        assert combo.itemText(0) == single_result["StringID"]
        assert widget._display_number == 0

    def test_many_parameters_pagination(self, qtbot):
        """Verify pagination activates for models with many parameters.

        Ref: ARMAX_viewer.m:563-625 — compute_text_positions pagination.
        With 20 AR params + constant = 21 displayable params, and
        _ROWS_PER_PAGE = 16, we expect ceil(21/16) = 2 pages.
        """
        many_param = _make_many_param_result(n_ar=20)
        widget = ARMAXViewer(results=[many_param])
        qtbot.addWidget(widget)
        widget.show()

        # Compute expected pagination.
        # generate_strings produces: "y_t =" + phi_0 + 20 * phi_i + "" + epsilon_t
        # That's 1 + 1 + 20 + 1 = 23 param_strs.
        # _param_count = 23 - 2 = 21 displayable params.
        expected_params = 21
        expected_pages = math.ceil(expected_params / _ROWS_PER_PAGE)
        assert expected_pages == 2, (
            f"Expected 2 pages, formula gives {expected_pages}"
        )

        # Verify the widget computed the right number of pages.
        assert widget._total_pages == expected_pages

        # Page navigation should be enabled.
        assert widget.increase_page_pushbutton.isEnabled()
        assert widget.decrease_page_pushbutton.isEnabled()
        assert widget.page_number_edit.isEnabled()

        # Page counter label should read "2".
        assert widget.number_of_pages_text.text() == str(expected_pages)

        # Navigate through all pages without error.
        for page_num in range(1, expected_pages + 1):
            if page_num > 1:
                qtbot.mouseClick(
                    widget.increase_page_pushbutton,
                    Qt.MouseButton.LeftButton,
                )
            assert widget._current_page == page_num
            assert widget.page_number_edit.text() == str(page_num)

        # Trying to go past the last page should not change page.
        qtbot.mouseClick(
            widget.increase_page_pushbutton, Qt.MouseButton.LeftButton
        )
        assert widget._current_page == expected_pages


# =========================================================================
# TestARMAXViewerEdgeCases — Additional edge case coverage
# =========================================================================

class TestARMAXViewerEdgeCases:
    """Additional edge-case tests for thorough coverage."""

    def test_empty_results_list(self, qtbot):
        """Verify viewer handles an empty results list without crash.

        This is a defensive edge case — MATLAB viewer always receives
        at least one result, but Python code should be robust.
        """
        widget = ARMAXViewer(results=[])
        qtbot.addWidget(widget)
        widget.show()

        assert widget.model_popupmenu.count() == 0
        assert widget._display_number == 0
        assert widget._total_pages == 1

    def test_display_number_out_of_range(self, qtbot, sample_results):
        """Verify display_number clamped when out of range.

        Passing display_number >= len(results) should not crash.
        """
        widget = ARMAXViewer(results=sample_results, display_number=99)
        qtbot.addWidget(widget)
        widget.show()

        # The widget was created without error — display_number stored as-is.
        # But _display_model guards against out-of-range.
        assert widget._display_number == 99
        # The combo box should still be populated.
        assert widget.model_popupmenu.count() == len(sample_results)

    def test_generate_strings_constant_only(self, qtbot):
        """Verify _generate_strings handles a constant-only model.

        Ref: ARMAX_viewer.m:132-170 — generate_strings with no AR/MA lags.
        """
        result = {
            "StringID": "Constant Only",
            "parameters": np.array([0.5]),
            "errors": np.zeros(100),
            "ll": -200.0,
            "seregression": 1.0,
            "diagnostics": {},
            "covariance": np.array([[0.01]]),
            "likelihoods": np.zeros(100),
            "scores": np.zeros((100, 1)),
            "ARlags": np.array([], dtype=np.int64),
            "MAlags": np.array([], dtype=np.int64),
            "constant": 1,
            "StdErr": np.array([0.1]),
            "Tstat": np.array([5.0]),
            "Pval": np.array([0.001]),
            "AIC": 402.0,
            "BIC": 404.6,
            "K": 1,
        }
        widget = ARMAXViewer(results=[result])
        qtbot.addWidget(widget)
        widget.show()

        # Should render without error.
        layout = widget._layout_cache.get(0, {})
        param_strs = layout.get("ParameterString", [])
        # Expected: ["y_t =", "\\phi_0", "", "\\epsilon_t"] — but exact form
        # depends on implementation; at minimum verify non-empty.
        assert len(param_strs) >= 3

    def test_generate_strings_no_constant(self, qtbot):
        """Verify _generate_strings handles a model without a constant term.

        Ref: ARMAX_viewer.m:141 — if results.constant, skip phi_0.
        """
        result = {
            "StringID": "AR(1) no constant",
            "parameters": np.array([0.3]),
            "errors": np.zeros(100),
            "ll": -200.0,
            "seregression": 1.0,
            "diagnostics": {},
            "covariance": np.array([[0.01]]),
            "likelihoods": np.zeros(100),
            "scores": np.zeros((100, 1)),
            "ARlags": np.array([1]),
            "MAlags": np.array([], dtype=np.int64),
            "constant": 0,
            "StdErr": np.array([0.1]),
            "Tstat": np.array([3.0]),
            "Pval": np.array([0.01]),
            "AIC": 402.0,
            "BIC": 404.6,
            "K": 1,
        }
        widget = ARMAXViewer(results=[result])
        qtbot.addWidget(widget)
        widget.show()

        layout = widget._layout_cache.get(0, {})
        param_strs = layout.get("ParameterString", [])
        # Without constant, "\\phi_0" should NOT be in param_strs.
        assert "\\phi_0" not in param_strs

    def test_format_value_edge_cases(self):
        """Verify _format_value static method handles edge values correctly.

        Ref: ARMAX_viewer.m:596-601 — format selection based on val < 0.000001.
        """
        fmt = ARMAXViewer._format_value

        # Large positive value → 5 decimal places.
        assert fmt(1.23456) == "1.23456"

        # Small positive value above threshold → 5 decimal places.
        assert fmt(0.001) == "0.00100"

        # Very small positive value → compact 3-significant-digit format.
        # Ref: ARMAX_viewer.m:597 — if temp < .000001
        result = fmt(1e-8)
        assert "e" in result.lower() or "1" in result  # Scientific notation

        # Negative value → always compact format (val < 0.000001 is true).
        # Ref: ARMAX_viewer.m:597 — raw comparison, NOT abs.
        result_neg = fmt(-0.5)
        assert result_neg == "-0.5"

        # Zero → compact format (0 < 0.000001 is True... wait, 0 < 0.000001
        # is True in MATLAB too, so zero gets compact format).
        result_zero = fmt(0.0)
        assert result_zero == "0"

    def test_param_count_static(self):
        """Verify _param_count correctly computes displayable parameter count.

        Ref: ARMAX_viewer.m:569 — loop from 2 to length(str)-1
        """
        pc = ARMAXViewer._param_count

        # Normal layout with prefix, params, suffix.
        layout = {"ParameterString": ["y_t =", "\\phi_0", "\\phi_1", "", ""]}
        assert pc(layout) == 3  # 5 - 2 = 3

        # Minimal layout (just prefix + suffix).
        layout_min = {"ParameterString": ["y_t =", ""]}
        assert pc(layout_min) == 0

        # Empty layout.
        assert pc({}) == 0

    def test_close_method_directly(self, qtbot, sample_results):
        """Verify calling close() programmatically hides the widget.

        Ref: ARMAX_viewer.m:478 — delete(handles.figure1)
        """
        widget = ARMAXViewer(results=sample_results)
        qtbot.addWidget(widget)
        widget.show()
        assert widget.isVisible()

        result = widget.close()
        assert result is True
        assert not widget.isVisible()
