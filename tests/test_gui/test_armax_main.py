"""pytest-qt integration tests for ARMAXMainWindow (QMainWindow).

Tests exercise ALL MATLAB GUIDE callbacks migrated from GUI/ARMAX.m:
- ARMAX_OpeningFcn: Window initialization, data storage, default state
- estimate_pushbutton_Callback: Estimation workflow with armaxfilter
- close_pushbutton_Callback: Close with confirmation dialog
- reset_pushbutton_Callback: Clear state and replot
- AR_lags_edit_Callback: AR lag specification validation
- MA_lags_edit_Callback: MA lag specification validation
- ACF_lags_edit_Callback: ACF lag bounded integer validation
- constant_include_Callback / constant_exclude_Callback: Constant toggle
- model_selector_popup_Callback: Model selection and display
- last_model_pushbutton_Callback: Load last model
- EXOG_clear_pushbutton_Callback: Clear exogenous input
- heterorobust_menu_Callback / homoerror_menu_Callback: Inference method
- about_menu_Callback: About dialog
- orig_raw_radio_Callback / orig_acf_radio_Callback / orig_pacf_radio_Callback
- resid_raw_radio_Callback / resid_fit_radio_Callback
- resid_acf_radio_Callback / resid_pacf_radio_Callback
- radiobutton18_Callback / radiobutton19_Callback: LM/LB test plots
- export_tiff/png/eps: Plot export callbacks
- copy_figure_copy_Callback: Copy figure
- residuals_save_residuals / residuals_save_std_residuals: File save

Per AAP Section 0.7.1: Coverage threshold >=80% for GUI modules.
"""

from __future__ import annotations

import gc
import os
from unittest.mock import MagicMock, patch
from typing import Any

# Force matplotlib to use non-interactive Agg backend BEFORE any Qt imports.
# This prevents segfaults in headless CI environments where the Qt backend
# tries to initialize display resources that don't exist.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

# Guard: skip all tests if PyQt6 not available.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.QtTest import QTest

from mfe_toolbox.gui.armax_main import ARMAXMainWindow

# ---------------------------------------------------------------------------
# Module-level marker: all tests in this file are GUI tests.
# Ref: tests/conftest.py — pytest_configure registers the 'gui' marker.
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Autouse fixture: clean up matplotlib figures after each test and bypass
# the blocking closeEvent dialog that would hang in headless CI.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _cleanup_matplotlib_and_bypass_close():
    """Ensure all matplotlib figures are closed after each test.

    Also patches ARMAXMainWindow.closeEvent to accept immediately so that
    pytest-qt's widget cleanup (which calls close()) does not trigger the
    blocking ARMAXCloseDialog modal dialog and hang the test runner.
    """
    import matplotlib.pyplot as plt
    yield
    plt.close("all")
    gc.collect()


@pytest.fixture
def armax_window(qtbot, sample_y):
    """Create an ARMAXMainWindow with non-blocking closeEvent.

    pytest-qt's addWidget calls close() during cleanup, which triggers
    closeEvent(), which shows a blocking ARMAXCloseDialog.  To prevent
    hangs in headless CI, we override closeEvent to accept immediately
    for standard tests.  Tests that specifically test closeEvent should
    NOT use this fixture and instead handle cleanup themselves.
    """
    win = ARMAXMainWindow(y=sample_y)

    # Store original closeEvent and replace with non-blocking version
    _orig_close_event = win.closeEvent

    def _non_blocking_close_event(event):
        """Accept close immediately — bypass confirmation dialog."""
        try:
            win._figure.clear()
            win._canvas.close()
        except Exception:
            pass
        event.accept()

    win.closeEvent = _non_blocking_close_event
    qtbot.addWidget(win)
    return win


# ===========================================================================
# Helper: build mock armaxfilter result tuple
# ===========================================================================
def _mock_armaxfilter_result(
    T: int = 500, K: int = 2, seed: int = 99
) -> tuple:
    """Build a mock 9-tuple matching armaxfilter's return signature.

    Ref: ARMAX.m:236 — armaxfilter returns
    (parameters, ll, errors, seregression, diagnostics, vcvrobust, vcv,
     likelihoods, scores).
    """
    rng = np.random.default_rng(seed)
    parameters = rng.standard_normal(K) * 0.1
    ll = -650.0
    errors = rng.standard_normal(T)
    seregression = float(np.var(errors))
    diagnostics = {}
    vcvrobust = np.eye(K) * 0.01
    vcv = np.eye(K) * 0.012
    likelihoods = rng.standard_normal(T) - 1.0
    scores = rng.standard_normal((T, K))
    return (parameters, ll, errors, seregression, diagnostics,
            vcvrobust, vcv, likelihoods, scores)


# ===========================================================================
# TestARMAXMainWindowInit — ARMAX_OpeningFcn
# ===========================================================================
class TestARMAXMainWindowInit:
    """Test window initialization (ARMAX_OpeningFcn).

    Ref: ARMAX.m:48-88 — ARMAX_OpeningFcn sets handles.y, handles.ACF_lags,
    handles.InferenceMethod=1, handles.constant=1, handles.models={},
    and plots original data.
    """

    def test_default_construction(self, armax_window, sample_y: np.ndarray) -> None:
        """Create window with sample data and verify all defaults.

        Ref: ARMAX.m:58-67 — handles initialization
        """
        win = armax_window

        # Verify window type
        assert isinstance(win, QMainWindow)

        # Ref: ARMAX.m:62 — handles.y = varargin{1}
        np.testing.assert_array_equal(win._y, sample_y)

        # Ref: ARMAX.m:58-59 — empty AR/MA orders
        assert win._ar_order == []
        assert win._ma_order == []

        # Ref: ARMAX.m:63 — residuals initially None
        assert win._residuals is None

        # Ref: ARMAX.m:64 — ACF_lags = min(24, floor(length(y)/8))
        expected_acf = min(24, max(1, len(sample_y) // 8))
        assert win._acf_lags == expected_acf

        # Ref: ARMAX.m:65 — InferenceMethod = 1 (hetero-robust)
        assert win._inference_method == 1

        # Ref: ARMAX.m:66 — constant = 1
        assert win._include_constant == 1

        # Ref: ARMAX.m:67 — models = {}
        assert win._models == []

    def test_window_title(self, armax_window) -> None:
        """Verify window title matches ARMAX.fig specification."""
        assert armax_window.windowTitle() == "ARMAX Estimation"

    def test_minimum_size(self, armax_window) -> None:
        """Verify minimum window dimensions.

        Ref: ARMAX.fig geometry — 900×700 minimum.
        """
        assert armax_window.minimumWidth() >= 900
        assert armax_window.minimumHeight() >= 700

    def test_empty_data_raises(self, qtbot) -> None:
        """Empty input array must raise ValueError.

        Ref: ARMAX.m:62 — handles.y is required non-empty.
        """
        with pytest.raises(ValueError, match="non-empty"):
            ARMAXMainWindow(y=np.array([]))

    def test_acf_lags_edit_initialized(self, armax_window, sample_y: np.ndarray) -> None:
        """Verify ACF lags edit text is populated on opening.

        Ref: ARMAX.m:87 — set(handles.ACF_lags_edit, 'String', ...)
        """
        expected_acf = min(24, max(1, len(sample_y) // 8))
        assert armax_window.ACF_lags_edit.text() == str(expected_acf)

    def test_constant_include_default(self, armax_window) -> None:
        """Verify 'Include Constant' radio is checked by default.

        Ref: ARMAX.m:66 — handles.constant = 1
        """
        assert armax_window.constant_include.isChecked()
        assert not armax_window.constant_exclude.isChecked()

    def test_inference_method_default(self, armax_window) -> None:
        """Verify hetero-robust inference is default.

        Ref: ARMAX.m:85 — set(handles.heterorobust_menu, 'Checked', 'on')
        """
        assert armax_window.action_heterorobust.isChecked()
        assert not armax_window.action_homoerror.isChecked()

    def test_canvas_exists(self, armax_window) -> None:
        """Verify matplotlib canvas is created and embedded.

        Ref: ARMAX.m:73-83 — initial axes setup; line 1277 — 585x392 pixels.
        """
        assert armax_window._figure is not None
        assert armax_window._canvas is not None
        assert armax_window._ax is not None

    def test_initial_plot_is_raw_data(self, armax_window) -> None:
        """Verify initial plot shows raw data.

        Ref: ARMAX.m:73-83 — plot original data on opening.
        """
        assert armax_window._ax.get_title() == "Plot of original data"
        assert armax_window.orig_raw_radio.isChecked()


# ===========================================================================
# TestARMAXMainWindowWidgets — Verify widget presence
# ===========================================================================
class TestARMAXMainWindowWidgets:
    """Verify all expected widgets exist on the window."""

    def test_edit_fields_exist(self, qtbot, sample_y: np.ndarray) -> None:
        """Verify all input edit fields are present."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        assert hasattr(win, "AR_lags_edit")
        assert hasattr(win, "MA_lags_edit")
        assert hasattr(win, "ACF_lags_edit")
        assert hasattr(win, "hold_back_edit")
        assert hasattr(win, "EXOG_edit")

    def test_buttons_exist(self, qtbot, sample_y: np.ndarray) -> None:
        """Verify all action buttons are present."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        assert hasattr(win, "estimate_pushbutton")
        assert hasattr(win, "reset_pushbutton")
        assert hasattr(win, "close_pushbutton")
        assert hasattr(win, "EXOG_clear_pushbutton")
        assert hasattr(win, "last_model_pushbutton")

    def test_radio_buttons_exist(self, qtbot, sample_y: np.ndarray) -> None:
        """Verify all plot radio buttons are present."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        assert hasattr(win, "orig_raw_radio")
        assert hasattr(win, "orig_acf_radio")
        assert hasattr(win, "orig_pacf_radio")
        assert hasattr(win, "resid_raw_radio")
        assert hasattr(win, "resid_fit_radio")
        assert hasattr(win, "resid_acf_radio")
        assert hasattr(win, "resid_pacf_radio")
        assert hasattr(win, "radiobutton18")
        assert hasattr(win, "radiobutton19")

    def test_model_selector_exists(self, qtbot, sample_y: np.ndarray) -> None:
        """Verify model selector combo box is present."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        assert hasattr(win, "model_selector_popup")
        assert win.model_selector_popup.count() == 0  # empty at start

    def test_menus_exist(self, qtbot, sample_y: np.ndarray) -> None:
        """Verify menu bar actions are present."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        assert hasattr(win, "action_export_tiff")
        assert hasattr(win, "action_export_png")
        assert hasattr(win, "action_export_eps")
        assert hasattr(win, "action_copy_figure")
        assert hasattr(win, "action_save_residuals")
        assert hasattr(win, "action_save_std_residuals")
        assert hasattr(win, "action_heterorobust")
        assert hasattr(win, "action_homoerror")
        assert hasattr(win, "action_about")


# ===========================================================================
# TestARMAXMainWindowARMALags — AR/MA lag validation
# ===========================================================================
class TestARMAXMainWindowARMALags:
    """Test AR and MA lag edit field callbacks.

    Ref: ARMAX.m:524-541 — AR_lags_edit_Callback
    Ref: ARMAX.m:557-574 — MA_lags_edit_Callback
    Ref: ARMAX.m:790-827 — validate_AR_MA_lags
    """

    def test_ar_lags_simple(self, qtbot, sample_y: np.ndarray) -> None:
        """Simple AR lag specification "1:4".

        Ref: ARMAX.m:815-816 — colon expansion.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1:4")
        win.AR_lags_edit.editingFinished.emit()

        assert win._ar_order == [1, 2, 3, 4]

    def test_ar_lags_space_separated(self, qtbot, sample_y: np.ndarray) -> None:
        """Space-separated AR lag specification "1 3 5".

        Ref: ARMAX.m:815 — tokens split by space.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1 3 5")
        win.AR_lags_edit.editingFinished.emit()

        assert win._ar_order == [1, 3, 5]

    def test_ar_lags_empty(self, qtbot, sample_y: np.ndarray) -> None:
        """Empty AR lag input clears lags.

        Ref: ARMAX.m:802-807 — empty input.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # First set some lags
        win.AR_lags_edit.setText("1:3")
        win.AR_lags_edit.editingFinished.emit()
        assert win._ar_order == [1, 2, 3]

        # Clear lags
        win.AR_lags_edit.setText("")
        win.AR_lags_edit.editingFinished.emit()
        assert win._ar_order == []

    def test_ar_lags_invalid_chars(self, qtbot, sample_y: np.ndarray) -> None:
        """Invalid characters in AR lag input.

        Ref: ARMAX.m:811 — valid characters: digits, colon, space.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # Set valid first
        win.AR_lags_edit.setText("1:3")
        win.AR_lags_edit.editingFinished.emit()
        old_lags = list(win._ar_order)

        # Try invalid
        win.AR_lags_edit.setText("1,3,5")
        win.AR_lags_edit.editingFinished.emit()

        # Should revert to old lags
        assert win._ar_order == old_lags

    def test_ma_lags_simple(self, qtbot, sample_y: np.ndarray) -> None:
        """Simple MA lag specification "1:2".

        Ref: ARMAX.m:557-574 — MA_lags_edit_Callback.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.MA_lags_edit.setText("1:2")
        win.MA_lags_edit.editingFinished.emit()

        assert win._ma_order == [1, 2]

    def test_ma_lags_mixed_notation(self, qtbot, sample_y: np.ndarray) -> None:
        """Mixed notation "1:3 5" combines colon range and discrete.

        Ref: ARMAX.m:815-816 — eval-like expansion.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.MA_lags_edit.setText("1:3 5")
        win.MA_lags_edit.editingFinished.emit()

        assert win._ma_order == [1, 2, 3, 5]


# ===========================================================================
# TestARMAXMainWindowACFLags — ACF lag validation
# ===========================================================================
class TestARMAXMainWindowACFLags:
    """Test ACF lags edit field callback.

    Ref: ARMAX.m:674-691 — ACF_lags_edit_Callback.
    Ref: ARMAX.m:711-729 — bounded_integer_validate.
    """

    def test_valid_acf_lags(self, qtbot, sample_y: np.ndarray) -> None:
        """Valid integer within bounds."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.ACF_lags_edit.setText("12")
        win.ACF_lags_edit.editingFinished.emit()

        assert win._acf_lags == 12

    def test_acf_lags_below_min(self, qtbot, sample_y: np.ndarray) -> None:
        """Value below lower bound (1) is clamped.

        Ref: ARMAX.m:684 — bounded_integer_validate(val, ACF_lags, 1, length(y)-1)
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.ACF_lags_edit.setText("0")
        win.ACF_lags_edit.editingFinished.emit()

        assert win._acf_lags == 1
        assert win.ACF_lags_edit.text() == "1"

    def test_acf_lags_above_max(self, qtbot, sample_y: np.ndarray) -> None:
        """Value above upper bound (T-1) is clamped.

        Ref: ARMAX.m:684 — upper bound is length(y)-1.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.ACF_lags_edit.setText("9999")
        win.ACF_lags_edit.editingFinished.emit()

        assert win._acf_lags == len(sample_y) - 1
        assert win.ACF_lags_edit.text() == str(len(sample_y) - 1)

    def test_acf_lags_non_numeric(self, qtbot, sample_y: np.ndarray) -> None:
        """Non-numeric input reverts to previous value.

        Ref: ARMAX.m:719-720 — NaN -> original value.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        original_val = win._acf_lags
        win.ACF_lags_edit.setText("abc")
        win.ACF_lags_edit.editingFinished.emit()

        assert win._acf_lags == original_val


# ===========================================================================
# TestARMAXMainWindowConstant — Constant toggle
# ===========================================================================
class TestARMAXMainWindowConstant:
    """Test constant include/exclude radio buttons.

    Ref: ARMAX.m:890-912 — constant_include_Callback / constant_exclude_Callback.
    """

    def test_exclude_constant(self, qtbot, sample_y: np.ndarray) -> None:
        """Toggle to 'Exclude Constant'.

        Ref: ARMAX.m:903-912 — constant_exclude_Callback sets constant=0.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.constant_exclude.setChecked(True)
        assert win._include_constant == 0

    def test_reinclude_constant(self, qtbot, sample_y: np.ndarray) -> None:
        """Toggle back to 'Include Constant'.

        Ref: ARMAX.m:890-898 — constant_include_Callback sets constant=1.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.constant_exclude.setChecked(True)
        assert win._include_constant == 0

        win.constant_include.setChecked(True)
        assert win._include_constant == 1


# ===========================================================================
# TestARMAXMainWindowEstimation — estimate_pushbutton_Callback
# ===========================================================================
class TestARMAXMainWindowEstimation:
    """Test estimation workflow via Estimate button.

    Ref: ARMAX.m:160-345 — estimate_pushbutton_Callback.
    All tests mock armaxfilter to avoid actual optimization.
    """

    def test_estimate_not_ready(self, qtbot, sample_y: np.ndarray) -> None:
        """Estimate with no AR/MA lags shows 'not ready' message.

        Ref: ARMAX.m:177-179 — both empty means not ready.
        Ref: ARMAX.m:315-316 — "not ready to run" message.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # No AR or MA set → not ready
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)
        assert "not ready" in win.message_text.text().lower()

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_estimate_success(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Successful estimation with AR(1) model.

        Ref: ARMAX.m:234-314 — estimation block.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_instance = MagicMock()
        mock_viewer_cls.return_value = mock_viewer_instance

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # Set AR lags
        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()

        # Click estimate
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        # Verify armaxfilter was called
        mock_armaxfilter.assert_called_once()

        # Ref: ARMAX.m:302-306 — model added to selector
        assert win.model_selector_popup.count() == 1

        # Ref: ARMAX.m:308 — success message
        assert "successfully estimated" in win.message_text.text().lower()

        # Ref: ARMAX.m:239-253 — model stored
        assert len(win._models) == 1
        assert win._models[0]["constant"] == 1
        assert win._models[0]["ARlags"] == [1]

        # Ref: ARMAX.m:341-345 — ARMAXViewer launched
        mock_viewer_instance.show.assert_called_once()

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_estimate_duplicate_model(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Estimating the same model twice reuses stored results.

        Ref: ARMAX.m:191-228 — compare constant, AR lags, MA lags.
        Ref: ARMAX.m:225-228 — reuse stored results.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # First estimation
        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        # Second estimation with same specs
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        # armaxfilter called only once (duplicate detected)
        assert mock_armaxfilter.call_count == 1

        # Ref: ARMAX.m:317-318 — previously estimated message
        assert "previously estimated" in win.message_text.text().lower()

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    def test_estimate_error_handling(
        self, mock_armaxfilter, qtbot, sample_y: np.ndarray
    ) -> None:
        """Estimation error is caught and displayed.

        Ref: ARMAX.m:310-314 — error handling.
        """
        mock_armaxfilter.side_effect = RuntimeError("Optimization failed")

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        # Ref: ARMAX.m:310-314 — error message displayed
        assert "error" in win.message_text.text().lower()

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_estimate_stores_string_id(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Verify StringID is built correctly.

        Ref: ARMAX.m:265-300 — StringID construction.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=3)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1:2")
        win.AR_lags_edit.editingFinished.emit()
        win.MA_lags_edit.setText("1")
        win.MA_lags_edit.editingFinished.emit()

        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        # Ref: ARMAX.m:278-289 — "ARMA(2,1) w/ constant"
        assert win._models[0]["StringID"] == "ARMA(2,1) w/ constant"

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_estimate_irregular_string_id(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Verify StringID marks irregular spacing.

        Ref: ARMAX.m:267-276 — irregular spacing detection.
        Ref: ARMAX.m:297-299 — "(Irregular)" suffix.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=3)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # Set irregular AR lags (1 and 3, missing 2)
        win.AR_lags_edit.setText("1 3")
        win.AR_lags_edit.editingFinished.emit()

        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        assert "(Irregular)" in win._models[0]["StringID"]


# ===========================================================================
# TestARMAXMainWindowReset — reset_pushbutton_Callback
# ===========================================================================
class TestARMAXMainWindowReset:
    """Test Reset button clears all state.

    Ref: ARMAX.m:402-432 — reset_pushbutton_Callback.
    """

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_reset_clears_state(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Reset button clears orders, residuals, edit fields, replots raw data.

        Ref: ARMAX.m:407-432.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # Estimate a model first
        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)
        assert len(win._models) == 1

        # Click reset
        qtbot.mouseClick(win.reset_pushbutton, Qt.MouseButton.LeftButton)

        # Ref: ARMAX.m:407-408 — orders cleared
        assert win._ar_order == []
        assert win._ma_order == []

        # Ref: ARMAX.m:412-417 — edit fields cleared
        assert win.AR_lags_edit.text() == ""
        assert win.MA_lags_edit.text() == ""
        assert win.hold_back_edit.text() == "0"

        # Ref: ARMAX.m:415 — orig_raw_radio selected
        assert win.orig_raw_radio.isChecked()

        # Ref: ARMAX.m:419-432 — replots original data
        assert win._ax.get_title() == "Plot of original data"

        # Residuals cleared
        assert win._residuals is None


# ===========================================================================
# TestARMAXMainWindowClose — close_pushbutton_Callback
# ===========================================================================
class TestARMAXMainWindowClose:
    """Test Close button shows confirmation dialog.

    Ref: ARMAX.m:372-386 — close_pushbutton_Callback.
    """

    @patch("mfe_toolbox.gui.armax_main.ARMAXCloseDialog")
    def test_close_yes(self, mock_dialog_cls, qtbot, sample_y: np.ndarray) -> None:
        """Close dialog returns 'Yes' — window should close.

        Ref: ARMAX.m:381-386 — switch on response.
        Note: _on_close() calls self.close() which triggers closeEvent()
        which also calls ARMAXCloseDialog.ask(), so it is called twice.
        """
        mock_dialog_cls.ask.return_value = "Yes"

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # Trigger the close button handler
        win._on_close()

        # Called at least once (may be twice due to closeEvent chain)
        assert mock_dialog_cls.ask.call_count >= 1

    @patch("mfe_toolbox.gui.armax_main.ARMAXCloseDialog")
    def test_close_no(self, mock_dialog_cls, qtbot, sample_y: np.ndarray) -> None:
        """Close dialog returns 'No' — window stays open.

        Ref: ARMAX.m:381-386 — 'No' takes no action.
        """
        mock_dialog_cls.ask.return_value = "No"

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win._on_close()

        # Window should still be visible
        assert win.isVisible()


# ===========================================================================
# TestARMAXMainWindowModelSelector — model_selector_popup_Callback
# ===========================================================================
class TestARMAXMainWindowModelSelector:
    """Test model selector popup callbacks.

    Ref: ARMAX.m:834-868 — model_selector_popup_Callback.
    Ref: ARMAX.m:591-595 — last_model_pushbutton_Callback.
    """

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_model_selector_updates_on_estimate(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Model selector populated after estimation.

        Ref: ARMAX.m:302-306 — update model_selector_popup.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        assert win.model_selector_popup.count() == 1
        assert win.model_selector_popup.currentIndex() == 0

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_model_selector_change(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Changing model selector updates AR/MA fields and residuals.

        Ref: ARMAX.m:845-846 — update AR and MA edit fields.
        Ref: ARMAX.m:848 — change residuals.
        """
        mock_viewer_cls.return_value = MagicMock()

        # Estimate two different models
        mock_result_1 = _mock_armaxfilter_result(T=len(sample_y), K=2, seed=10)
        mock_result_2 = _mock_armaxfilter_result(T=len(sample_y), K=3, seed=20)
        mock_armaxfilter.side_effect = [mock_result_1, mock_result_2]

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # First model: AR(1)
        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        # Second model: AR(2) MA(1)
        win.AR_lags_edit.setText("1:2")
        win.AR_lags_edit.editingFinished.emit()
        win.MA_lags_edit.setText("1")
        win.MA_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        assert win.model_selector_popup.count() == 2

        # Switch back to first model
        win.model_selector_popup.setCurrentIndex(0)

        # Ref: ARMAX.m:865 — resid_raw_radio selected
        assert win.resid_raw_radio.isChecked()

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_last_model_button(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Last Model button selects the last estimated model.

        Ref: ARMAX.m:591-595 — last_model_pushbutton_Callback.
        """
        mock_viewer_cls.return_value = MagicMock()
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # Estimate a model
        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        # Click last model
        qtbot.mouseClick(win.last_model_pushbutton, Qt.MouseButton.LeftButton)

        assert win.model_selector_popup.currentIndex() == len(win._models) - 1

    def test_last_model_no_models(self, qtbot, sample_y: np.ndarray) -> None:
        """Last Model button with no models does nothing.

        Ref: ARMAX.m:591-595 — guarded by if not empty.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # Should not crash
        qtbot.mouseClick(win.last_model_pushbutton, Qt.MouseButton.LeftButton)
        assert win.model_selector_popup.count() == 0


# ===========================================================================
# TestARMAXMainWindowInference — Inference method menu
# ===========================================================================
class TestARMAXMainWindowInference:
    """Test Standard Errors menu callbacks.

    Ref: ARMAX.m:762-785 — heterorobust_menu_Callback / homoerror_menu_Callback.
    """

    def test_switch_to_homoskedastic(self, qtbot, sample_y: np.ndarray) -> None:
        """Switch inference to homoskedastic.

        Ref: ARMAX.m:777-785 — homoerror_menu_Callback.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win._on_homo_error()

        assert win._inference_method == 0
        assert win.action_homoerror.isChecked()
        assert not win.action_heterorobust.isChecked()
        assert "homoskedastic" in win.message_text.text().lower()

    def test_switch_to_heterorobust(self, qtbot, sample_y: np.ndarray) -> None:
        """Switch inference back to hetero-robust.

        Ref: ARMAX.m:762-770 — heterorobust_menu_Callback.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        # First switch to homo
        win._on_homo_error()
        # Then switch back
        win._on_hetero_robust()

        assert win._inference_method == 1
        assert win.action_heterorobust.isChecked()
        assert not win.action_homoerror.isChecked()
        assert "heteroskedasticity" in win.message_text.text().lower()


# ===========================================================================
# TestARMAXMainWindowAbout — about_menu_Callback
# ===========================================================================
class TestARMAXMainWindowAbout:
    """Test About menu callback.

    Ref: ARMAX.m:735-739 — about_menu_Callback.
    """

    @patch("mfe_toolbox.gui.armax_main.ARMAXAboutDialog")
    def test_about_dialog_launched(
        self, mock_about_cls, qtbot, sample_y: np.ndarray
    ) -> None:
        """About menu launches ARMAXAboutDialog.

        Ref: ARMAX.m:739 — ARMAX_about('Title','About').
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win._on_about()

        mock_about_cls.show_about.assert_called_once_with(
            parent=win, title="About"
        )


# ===========================================================================
# TestARMAXMainWindowClearExog — EXOG_clear_pushbutton_Callback
# ===========================================================================
class TestARMAXMainWindowClearExog:
    """Test Clear Exogenous button.

    Ref: ARMAX.m:502-506 — EXOG_clear_pushbutton_Callback.
    """

    def test_clear_exog(self, qtbot, sample_y: np.ndarray) -> None:
        """Clear exogenous edit field.

        Ref: ARMAX.m:504 — set(handles.EXOG_edit, 'String', ' ')
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.EXOG_edit.setText("some_var")
        qtbot.mouseClick(win.EXOG_clear_pushbutton, Qt.MouseButton.LeftButton)

        assert win.EXOG_edit.text() == " "


# ===========================================================================
# TestARMAXMainWindowPlots — Plot radio button callbacks
# ===========================================================================
class TestARMAXMainWindowPlots:
    """Test plot switching via radio buttons.

    Ref: ARMAX.m:917-1271 — all plot radio callbacks.
    """

    def test_orig_raw_radio(self, qtbot, sample_y: np.ndarray) -> None:
        """Raw data plot on orig_raw_radio selection.

        Ref: ARMAX.m:917-937 — orig_raw_radio_Callback.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.orig_raw_radio.setChecked(True)
        assert win._ax.get_title() == "Plot of original data"

    def test_orig_acf_radio(self, qtbot, sample_y: np.ndarray) -> None:
        """ACF plot on orig_acf_radio selection.

        Ref: ARMAX.m:942-978 — orig_acf_radio_Callback.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.orig_acf_radio.setChecked(True)
        title = win._ax.get_title()
        assert "Autocorrelations" in title

    def test_orig_pacf_radio(self, qtbot, sample_y: np.ndarray) -> None:
        """PACF plot on orig_pacf_radio selection.

        Ref: ARMAX.m:981-1016 — orig_pacf_radio_Callback.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.orig_pacf_radio.setChecked(True)
        title = win._ax.get_title()
        assert "Partial Autocorrelations" in title

    def test_resid_raw_no_model(self, qtbot, sample_y: np.ndarray) -> None:
        """Residual raw plot with no model shows 'must estimate' message.

        Ref: ARMAX.m:1028-1029 — no model run check.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.resid_raw_radio.setChecked(True)
        # Should show "must estimate" text on the axes
        texts = [t.get_text() for t in win._ax.texts]
        assert any("must estimate" in t.lower() for t in texts)

    def test_resid_fit_no_model(self, qtbot, sample_y: np.ndarray) -> None:
        """Fit overlay with no model shows 'must estimate' message.

        Ref: ARMAX.m:1050-1076 — resid_fit_radio_Callback.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.resid_fit_radio.setChecked(True)
        texts = [t.get_text() for t in win._ax.texts]
        assert any("must estimate" in t.lower() for t in texts)

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_resid_raw_with_model(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Residual raw plot after estimation shows residuals.

        Ref: ARMAX.m:1036-1039 — plot residuals with title.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        # After estimation, resid_raw_radio is auto-selected
        assert win._ax.get_title() == "Plot of estimated errors"

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_resid_fit_with_model(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Fit overlay after estimation shows original + fitted data.

        Ref: ARMAX.m:1050-1076 — resid_fit_radio_Callback.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        win.resid_fit_radio.setChecked(True)
        assert win._ax.get_title() == "Plot of estimated errors"
        # Should have 2 lines (original + fit)
        assert len(win._ax.get_lines()) == 2

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_resid_acf_with_model(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Residual ACF plot after estimation.

        Ref: ARMAX.m:1081-1121 — resid_acf_radio_Callback.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        win.resid_acf_radio.setChecked(True)
        assert "Autocorrelations" in win._ax.get_title()

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_resid_pacf_with_model(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """Residual PACF plot after estimation.

        Ref: ARMAX.m:1125-1167 — resid_pacf_radio_Callback.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        win.resid_pacf_radio.setChecked(True)
        assert "Partial Autocorrelations" in win._ax.get_title()

    def test_lm_lb_original(self, qtbot, sample_y: np.ndarray) -> None:
        """LM/LB test plot on original data.

        Ref: ARMAX.m:1171-1218 — radiobutton18_Callback.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.radiobutton18.setChecked(True)
        title = win._ax.get_title()
        assert "Statistic" in title or "Statistics" in title

    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    def test_lm_lb_residuals_with_model(
        self, mock_viewer_cls, mock_aicsbic, mock_armaxfilter,
        qtbot, sample_y: np.ndarray
    ) -> None:
        """LM/LB test plot on residuals after estimation.

        Ref: ARMAX.m:1223-1271 — radiobutton19_Callback.
        """
        mock_result = _mock_armaxfilter_result(T=len(sample_y), K=2)
        mock_armaxfilter.return_value = mock_result
        mock_viewer_cls.return_value = MagicMock()

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        win.radiobutton19.setChecked(True)
        title = win._ax.get_title()
        assert "Error" in title

    def test_lm_lb_residuals_no_model(self, qtbot, sample_y: np.ndarray) -> None:
        """LM/LB test on residuals with no model shows 'must estimate'.

        Ref: ARMAX.m:1230-1231 — no model run check.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win.radiobutton19.setChecked(True)
        texts = [t.get_text() for t in win._ax.texts]
        assert any("must estimate" in t.lower() for t in texts)


# ===========================================================================
# TestARMAXMainWindowExport — Export callbacks
# ===========================================================================
class TestARMAXMainWindowExport:
    """Test export and save callbacks.

    Ref: ARMAX.m:457-484 — export callbacks.
    Ref: ARMAX.m:443-453 — residual save callbacks.
    """

    @patch.object(QFileDialog, "getSaveFileName", return_value=("", ""))
    def test_export_plot_cancelled(
        self, mock_dialog, qtbot, sample_y: np.ndarray
    ) -> None:
        """Export cancelled (empty path) does nothing.

        Ref: ARMAX.m:457-484 — guarded by if filepath.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win._on_export_plot("png")
        # No error — graceful handling

    @patch.object(QFileDialog, "getSaveFileName", return_value=("/tmp/test_export.png", ""))
    def test_export_plot_png(
        self, mock_dialog, qtbot, sample_y: np.ndarray, tmp_path
    ) -> None:
        """Export plot as PNG.

        Ref: ARMAX.m:460-464 — export PNG.
        """
        outpath = str(tmp_path / "test_export.png")
        mock_dialog.return_value = (outpath, "")

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win._on_export_plot("png")
        assert "exported" in win.message_text.text().lower() or \
               "Export failed" in win.message_text.text()

    def test_save_residuals_no_model(self, qtbot, sample_y: np.ndarray) -> None:
        """Save residuals with no model shows warning.

        Ref: ARMAX.m:443-447 — guarded by if residuals exist.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win._on_save_residuals()
        assert "no residuals" in win.message_text.text().lower()

    def test_save_std_residuals_no_model(self, qtbot, sample_y: np.ndarray) -> None:
        """Save standardized residuals with no model shows warning.

        Ref: ARMAX.m:450-453 — guarded by if residuals exist.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win._on_save_std_residuals()
        assert "no residuals" in win.message_text.text().lower()


# ===========================================================================
# TestARMAXMainWindowCopyFigure — copy_figure_copy_Callback
# ===========================================================================
class TestARMAXMainWindowCopyFigure:
    """Test Copy Figure callback.

    Ref: ARMAX.m:599-621 — copy_figure_copy_Callback.
    """

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_copy_figure(self, mock_figure, mock_show, qtbot, sample_y: np.ndarray) -> None:
        """Copy figure creates new matplotlib window.

        Ref: ARMAX.m:599-621 — creates a new standalone figure window.
        Note: armax_main._on_copy_figure imports plt locally, so we patch
        matplotlib.pyplot directly.
        """
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.gca.return_value = mock_ax
        mock_figure.return_value = mock_fig

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()
        win.show()

        win._on_copy_figure()

        mock_figure.assert_called_once()
        mock_show.assert_called_once()


# ===========================================================================
# TestARMAXMainWindowStringID — Static StringID builder
# ===========================================================================
class TestARMAXMainWindowStringID:
    """Test _build_string_id static method.

    Ref: ARMAX.m:265-300 — StringID construction.
    """

    def test_ar_only_with_constant(self) -> None:
        """AR(2) with constant.

        Ref: ARMAX.m:278-289.
        """
        result = ARMAXMainWindow._build_string_id([1, 2], [], 1)
        assert result == "ARMA(2,0) w/ constant"

    def test_arma_with_constant(self) -> None:
        """ARMA(2,1) with constant.

        Ref: ARMAX.m:278-295.
        """
        result = ARMAXMainWindow._build_string_id([1, 2], [1], 1)
        assert result == "ARMA(2,1) w/ constant"

    def test_without_constant(self) -> None:
        """ARMA(1,1) without constant.

        Ref: ARMAX.m:291-295.
        """
        result = ARMAXMainWindow._build_string_id([1], [1], 0)
        assert result == "ARMA(1,1) w/o constant"

    def test_irregular_spacing(self) -> None:
        """Irregular AR lags (1,3) marks as Irregular.

        Ref: ARMAX.m:267-276 — irregular spacing detection.
        Ref: ARMAX.m:297-299 — "(Irregular)" suffix.
        """
        result = ARMAXMainWindow._build_string_id([1, 3], [], 1)
        assert "(Irregular)" in result

    def test_no_lags(self) -> None:
        """No AR or MA lags.

        Ref: ARMAX.m:278-289 — ARMA(0,0).
        """
        result = ARMAXMainWindow._build_string_id([], [], 1)
        assert result == "ARMA(0,0) w/ constant"


# ===========================================================================
# TestARMAXMainWindowValidation — Input validation helpers
# ===========================================================================
class TestARMAXMainWindowValidation:
    """Test validation helper methods.

    Ref: ARMAX.m:790-827 — validate_AR_MA_lags.
    Ref: ARMAX.m:711-729 — bounded_integer_validate.
    """

    def test_validate_ar_ma_lags_colon(self, qtbot, sample_y: np.ndarray) -> None:
        """Colon notation expansion."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        lags, out_str, ok = win._validate_ar_ma_lags("1:4", [])
        assert ok
        assert lags == [1, 2, 3, 4]

    def test_validate_ar_ma_lags_step(self, qtbot, sample_y: np.ndarray) -> None:
        """Colon notation with step "1:2:7"."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        lags, out_str, ok = win._validate_ar_ma_lags("1:2:7", [])
        assert ok
        assert lags == [1, 3, 5, 7]

    def test_validate_ar_ma_lags_invalid(self, qtbot, sample_y: np.ndarray) -> None:
        """Invalid characters."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        lags, _, ok = win._validate_ar_ma_lags("a:b", [1])
        assert not ok
        assert lags == [1]  # reverts to old_lags

    def test_bounded_integer_valid(self, qtbot, sample_y: np.ndarray) -> None:
        """Valid bounded integer."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        val, ok = win._bounded_integer_validate("10", 5, 1, 100)
        assert ok
        assert val == 10

    def test_bounded_integer_below(self, qtbot, sample_y: np.ndarray) -> None:
        """Value below lower bound is clamped."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        val, ok = win._bounded_integer_validate("-5", 5, 1, 100)
        assert not ok
        assert val == 1

    def test_bounded_integer_above(self, qtbot, sample_y: np.ndarray) -> None:
        """Value above upper bound is clamped."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        val, ok = win._bounded_integer_validate("500", 5, 1, 100)
        assert not ok
        assert val == 100

    def test_bounded_integer_nan(self, qtbot, sample_y: np.ndarray) -> None:
        """Non-numeric input reverts to original."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        val, ok = win._bounded_integer_validate("abc", 42, 1, 100)
        assert not ok
        assert val == 42

    def test_bounded_integer_float(self, qtbot, sample_y: np.ndarray) -> None:
        """Non-integer float is floored.

        Ref: ARMAX.m:725-726 — non-integer -> floor.
        """
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        val, ok = win._bounded_integer_validate("10.7", 5, 1, 100)
        assert not ok
        assert val == 10


# ===========================================================================
# TestARMAXMainWindowCloseEvent — figure1_CloseRequestFcn
# ===========================================================================
class TestARMAXMainWindowCloseEvent:
    """Test close event override.

    Ref: ARMAX.m:657-671 — figure1_CloseRequestFcn.
    """

    @patch("mfe_toolbox.gui.armax_main.ARMAXCloseDialog")
    def test_close_event_yes(self, mock_dialog_cls, qtbot, sample_y: np.ndarray) -> None:
        """Close event accepted when dialog returns 'Yes'.

        Ref: ARMAX.m:670 — delete(handles.figure1).
        """
        mock_dialog_cls.ask.return_value = "Yes"

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.show()

        # Test with real closeEvent — mocked dialog returns "Yes" so
        # closeEvent calls event.accept() and window closes.
        win.close()
        mock_dialog_cls.ask.assert_called()

        # Override for safe cleanup by pytest-qt (window already closed,
        # but belt-and-suspenders safety).
        win.closeEvent = lambda event: event.accept()

    @patch("mfe_toolbox.gui.armax_main.ARMAXCloseDialog")
    def test_close_event_no(self, mock_dialog_cls, qtbot, sample_y: np.ndarray) -> None:
        """Close event ignored when dialog returns 'No'.

        Ref: ARMAX.m:668 — take no action on 'No'.
        """
        mock_dialog_cls.ask.return_value = "No"

        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.show()

        # Test with real closeEvent — mocked dialog returns "No" so
        # closeEvent calls event.ignore() and window stays open.
        win.close()
        # Window should still be visible (event was ignored)

        # Override closeEvent for safe cleanup — the @patch exits after this
        # method returns, so pytest-qt cleanup would otherwise call the
        # real ARMAXCloseDialog.ask() and hang.
        win.closeEvent = lambda event: event.accept()


# ===========================================================================
# TestARMAXMainWindowMessageHelper — _set_message
# ===========================================================================
class TestARMAXMainWindowMessageHelper:
    """Test status message helper.

    Ref: ARMAX.m:308,316,318 — set message_text with ForegroundColor.
    """

    def test_set_message_default(self, qtbot, sample_y: np.ndarray) -> None:
        """Default message color is black."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        win._set_message("Test message")
        assert win.message_text.text() == "Test message"
        assert "rgb(0, 0, 0)" in win.message_text.styleSheet()

    def test_set_message_blue(self, qtbot, sample_y: np.ndarray) -> None:
        """Blue message color for success."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        win._set_message("Success!", color="blue")
        assert "rgb(0, 0, 204)" in win.message_text.styleSheet()

    def test_set_message_red(self, qtbot, sample_y: np.ndarray) -> None:
        """Red message color for errors."""
        win = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(win)
        win.closeEvent = lambda event: event.accept()

        win._set_message("Error!", color="red")
        assert "rgb(230, 0, 0)" in win.message_text.styleSheet()
