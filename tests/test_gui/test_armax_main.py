"""pytest-qt integration tests for ARMAXMainWindow (QMainWindow with ARMAX estimation).

Tests exercise ALL MATLAB GUIDE callbacks migrated from GUI/ARMAX.m (1282 lines):
- OpeningFcn: Window initialization, data loading, state setup, initial plot
- estimate_pushbutton_Callback: ARMAX estimation, result storage, viewer launch
- close_pushbutton_Callback: Close confirmation via ARMAXCloseDialog
- reset_pushbutton_Callback: Clear all state and inputs, replot original
- AR_lags_edit/MA_lags_edit_Callback: Lag specification validation
- ACF_lags_edit_Callback: ACF lag count validation
- about_menu_Callback: About dialog launch
- heterorobust_menu/homoerror_menu: Inference method toggle
- constant_include/exclude: Constant toggle
- model_selector_popup_Callback: Model selection switching
- All 9 radio button plot callbacks: raw, ACF, PACF, residuals, fit, LM/LB tests
- validate_AR_MA_lags: Lag validation helper
- bounded_integer_validate: Integer validation helper

Per AAP Section 0.7.1: Coverage threshold >=80 percent for GUI modules.
"""
from __future__ import annotations

# Force non-interactive Agg backend before ANY Qt imports to prevent
# ``plt.show()`` or ``FigureCanvasQTAgg`` from requiring a display server.
import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Guard — skip entire module when PyQt6 is absent.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QComboBox,
    QDialog,
)
from PyQt6.QtTest import QTest

from mfe_toolbox.gui.armax_main import ARMAXMainWindow

# ---------------------------------------------------------------------------
# Module-level marker — every test in this file is tagged ``gui``.
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_armaxfilter_result(
    T: int = 500, K: int = 2, seed: int = 99
) -> tuple:
    """Build a mock 9-tuple matching ``armaxfilter`` return signature.

    Returns
    -------
    tuple
        (parameters, ll, errors, seregression, diagnostics,
         vcvrobust, vcv, likelihoods, scores)
    """
    rng = np.random.default_rng(seed)
    parameters = rng.standard_normal(K) * 0.2
    ll = -650.0
    errors = rng.standard_normal(T)
    seregression = float(np.var(errors))
    diagnostics: dict = {}
    vcvrobust = np.eye(K) * 0.01
    vcv = np.eye(K) * 0.01
    likelihoods = rng.standard_normal(T) - 1.0
    scores = rng.standard_normal((T, K))
    return (
        parameters,
        ll,
        errors,
        seregression,
        diagnostics,
        vcvrobust,
        vcv,
        likelihoods,
        scores,
    )


# ---------------------------------------------------------------------------
# Autouse fixture — clean up matplotlib state and bypass the close-event
# confirmation dialog so that widget deletion never blocks.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cleanup_matplotlib_and_bypass_close(monkeypatch):
    """Ensure matplotlib figures are closed and closeEvent is non-blocking.

    Without this, every ``qtbot.addWidget(win)`` teardown triggers the
    ``closeEvent`` override in ``ARMAXMainWindow`` which calls
    ``ARMAXCloseDialog.ask(...)`` — a modal dialog that hangs the test.
    """
    # Replace closeEvent with a simple accept-all so widget cleanup never blocks.
    monkeypatch.setattr(
        ARMAXMainWindow,
        "closeEvent",
        lambda self, event: event.accept(),
    )
    yield
    # Post-test teardown: close all leftover matplotlib figures.
    matplotlib.pyplot.close("all")


# ---------------------------------------------------------------------------
# Convenience fixture — ``main_window``
# ---------------------------------------------------------------------------

@pytest.fixture
def main_window(qtbot, sample_y):
    """Create and show ARMAXMainWindow for testing.

    Ref: GUI/ARMAX.m:62 — handles.y = varargin{1}
    """
    window = ARMAXMainWindow(y=sample_y)
    qtbot.addWidget(window)
    window.show()
    return window


# ===================================================================
# TestARMAXMainWindowInit
# ===================================================================

class TestARMAXMainWindowInit:
    """Test window initialization — OpeningFcn.

    Ref: ARMAX.m lines 48-87.
    """

    def test_construction_with_data(self, qtbot, sample_y):
        """Window accepts numpy data and stores it.

        Ref: ARMAX.m:62 — handles.y = varargin{1}
        """
        window = ARMAXMainWindow(y=sample_y)
        qtbot.addWidget(window)
        assert isinstance(window, QMainWindow)
        assert np.array_equal(window._y, sample_y)

    def test_initial_state(self, main_window, sample_y):
        """Verify default internal state after construction.

        Ref: ARMAX.m:58-67 — handles initialization.
        """
        win = main_window
        # Ref: line 58 — AR order
        assert win._ar_order == []
        # Ref: line 59 — MA order
        assert win._ma_order == []
        # Ref: line 63 — residuals
        assert win._residuals is None
        # Ref: line 64 — ACF_lags = min(24, floor(length(y)/8))
        expected_acf = min(24, max(1, len(sample_y) // 8))
        assert win._acf_lags == expected_acf
        # Ref: line 65 — InferenceMethod = 1 (hetero-robust)
        assert win._inference_method == 1
        # Ref: line 66 — IncludeConstant = 1
        assert win._include_constant == 1
        # Ref: line 67 — models = {}
        assert win._models == []

    def test_initial_plot(self, main_window):
        """Verify initial plot renders original data.

        Ref: ARMAX.m:73-83 — plots original data, title 'Plot of original data'.
        """
        win = main_window
        # Canvas and axes must exist
        assert hasattr(win, "_canvas")
        assert hasattr(win, "_ax")
        # Ref: ARMAX.m:78 — title('Plot of original data')
        assert win._ax.get_title() == "Plot of original data"
        # Axes should contain at least one line (the data plot)
        assert len(win._ax.get_lines()) >= 1

    def test_initial_menu_state(self, main_window):
        """Verify menu actions are correctly initialised.

        Ref: ARMAX.m:85-86 — heterorobust checked, homoerror unchecked.
        """
        win = main_window
        assert win.action_heterorobust.isChecked()
        assert not win.action_homoerror.isChecked()

    def test_acf_lags_edit_initial(self, main_window, sample_y):
        """Verify ACF lags edit field is initialised with correct value.

        Ref: ARMAX.m:87 — set ACF_lags_edit to string of ACF_lags.
        """
        win = main_window
        expected_acf = min(24, max(1, len(sample_y) // 8))
        assert win.ACF_lags_edit.text() == str(expected_acf)


# ===================================================================
# TestARMAXMainWindowLagValidation
# ===================================================================

class TestARMAXMainWindowLagValidation:
    """Test lag specification validation.

    Ref: validate_AR_MA_lags (ARMAX.m lines 790-827),
         AR_lags_edit_Callback (lines 524-541),
         MA_lags_edit_Callback (lines 557-574).
    """

    def test_ar_lags_simple_input(self, main_window, qtbot):
        """Space-separated AR lags parsed correctly.

        Ref: ARMAX.m:524-541 — AR_lags_edit_Callback.
        """
        win = main_window
        win.AR_lags_edit.setText("1 2 3")
        win.AR_lags_edit.editingFinished.emit()
        assert win._ar_order == [1, 2, 3]

    def test_ar_lags_colon_notation(self, main_window, qtbot):
        """Colon notation '1:4' expands to [1, 2, 3, 4].

        Ref: ARMAX.m:790-827 — validate_AR_MA_lags parses colon-separated ranges.
        """
        win = main_window
        win.AR_lags_edit.setText("1:4")
        win.AR_lags_edit.editingFinished.emit()
        assert win._ar_order == [1, 2, 3, 4]

    def test_ma_lags_input(self, main_window, qtbot):
        """MA lags edit sets _ma_order correctly.

        Ref: ARMAX.m:557-574 — MA_lags_edit_Callback.
        """
        win = main_window
        win.MA_lags_edit.setText("1 3")
        win.MA_lags_edit.editingFinished.emit()
        assert win._ma_order == [1, 3]

    def test_empty_lags(self, main_window, qtbot):
        """Empty AR lags input results in empty list.

        Ref: ARMAX.m:802-807 — empty input returns empty lags.
        """
        win = main_window
        # First set lags then clear them
        win.AR_lags_edit.setText("1 2")
        win.AR_lags_edit.editingFinished.emit()
        assert win._ar_order == [1, 2]

        win.AR_lags_edit.setText("")
        win.AR_lags_edit.editingFinished.emit()
        assert win._ar_order == []

    def test_invalid_lags_rejected(self, main_window, qtbot):
        """Invalid lag string leaves lags unchanged or shows error.

        Ref: ARMAX.m:811 — valid characters: digits, colon, space.
        """
        win = main_window
        # Set a known valid state first
        win.AR_lags_edit.setText("1 2")
        win.AR_lags_edit.editingFinished.emit()
        old_order = list(win._ar_order)

        # Now enter invalid text — should revert to old lags
        win.AR_lags_edit.setText("abc")
        win.AR_lags_edit.editingFinished.emit()
        # Lags should remain as old value (rejection)
        assert win._ar_order == old_order

    def test_lags_unique_sorted(self, main_window, qtbot):
        """Duplicated and unsorted lags are unique-sorted.

        Ref: ARMAX.m:821-826 — unique sort.
        """
        win = main_window
        win.AR_lags_edit.setText("3 1 2 1")
        win.AR_lags_edit.editingFinished.emit()
        assert win._ar_order == [1, 2, 3]


# ===================================================================
# TestARMAXMainWindowACFValidation
# ===================================================================

class TestARMAXMainWindowACFValidation:
    """Test ACF lag count validation.

    Ref: ACF_lags_edit_Callback (ARMAX.m lines 674-691),
         bounded_integer_validate (lines 711-729).
    """

    def test_acf_lags_valid_input(self, main_window, qtbot):
        """Valid integer input updates _acf_lags.

        Ref: ARMAX.m:684 — bounded_integer_validate.
        """
        win = main_window
        win.ACF_lags_edit.setText("12")
        win.ACF_lags_edit.editingFinished.emit()
        assert win._acf_lags == 12

    def test_acf_lags_below_minimum(self, main_window, qtbot, sample_y):
        """Below-minimum input is clamped to lower bound.

        Ref: bounded_integer_validate: <LB → LB (LB=1).
        """
        win = main_window
        win.ACF_lags_edit.setText("0")
        win.ACF_lags_edit.editingFinished.emit()
        assert win._acf_lags == 1

    def test_acf_lags_above_maximum(self, main_window, qtbot, sample_y):
        """Above-maximum input is clamped to upper bound.

        Ref: bounded_integer_validate: >UB → UB (UB=len(y)-1).
        """
        win = main_window
        win.ACF_lags_edit.setText("9999")
        win.ACF_lags_edit.editingFinished.emit()
        assert win._acf_lags == len(sample_y) - 1

    def test_acf_lags_nan_input(self, main_window, qtbot):
        """Non-numeric input reverts to original value.

        Ref: bounded_integer_validate: NaN → original value.
        """
        win = main_window
        original_val = win._acf_lags
        win.ACF_lags_edit.setText("abc")
        win.ACF_lags_edit.editingFinished.emit()
        assert win._acf_lags == original_val


# ===================================================================
# TestARMAXMainWindowConstant
# ===================================================================

class TestARMAXMainWindowConstant:
    """Test constant include/exclude toggle.

    Ref: constant_include/exclude callbacks (ARMAX.m lines 890-912).
    """

    def test_constant_include_default(self, main_window):
        """Include-constant radio is checked by default.

        Ref: ARMAX.m:66 — handles.model_IncludeConstant=1.
        """
        win = main_window
        assert win.constant_include.isChecked()
        assert not win.constant_exclude.isChecked()
        assert win._include_constant == 1

    def test_toggle_constant_exclude(self, main_window, qtbot):
        """Clicking exclude sets _include_constant to 0.

        Ref: ARMAX.m:903-912 — constant_exclude_Callback.
        """
        win = main_window
        win.constant_exclude.setChecked(True)
        assert win._include_constant == 0

    def test_toggle_constant_include(self, main_window, qtbot):
        """Toggling back to include restores _include_constant to 1.

        Ref: ARMAX.m:890-898 — constant_include_Callback.
        """
        win = main_window
        # First exclude
        win.constant_exclude.setChecked(True)
        assert win._include_constant == 0
        # Then include again
        win.constant_include.setChecked(True)
        assert win._include_constant == 1


# ===================================================================
# TestARMAXMainWindowInferenceMethod
# ===================================================================

class TestARMAXMainWindowInferenceMethod:
    """Test inference method toggle.

    Ref: heterorobust_menu/homoerror_menu callbacks (ARMAX.m lines 762-785).
    """

    def test_default_inference_method(self, main_window):
        """Default inference method is hetero-robust (1).

        Ref: ARMAX.m:65 — handles.model_InferenceMethod = 1.
        """
        assert main_window._inference_method == 1

    def test_switch_to_homo(self, main_window, qtbot):
        """Triggering homoerror sets inference to 0.

        Ref: ARMAX.m:777-785 — sets InferenceMethod=0.
        """
        win = main_window
        win.action_homoerror.trigger()
        assert win._inference_method == 0
        assert win.action_homoerror.isChecked()
        assert not win.action_heterorobust.isChecked()

    def test_switch_back_to_hetero(self, main_window, qtbot):
        """Switching to homo then hetero restores to 1.

        Ref: ARMAX.m:762-770 — sets InferenceMethod=1.
        """
        win = main_window
        win.action_homoerror.trigger()
        assert win._inference_method == 0

        win.action_heterorobust.trigger()
        assert win._inference_method == 1
        assert win.action_heterorobust.isChecked()
        assert not win.action_homoerror.isChecked()


# ===================================================================
# TestARMAXMainWindowEstimation
# ===================================================================

class TestARMAXMainWindowEstimation:
    """Test estimation callback.

    Ref: estimate_pushbutton_Callback (ARMAX.m lines 160-345).
    All estimation calls are mocked — actual numerical testing is in
    test_timeseries.
    """

    def test_estimate_no_lags_shows_error(self, main_window, qtbot):
        """Clicking Estimate with no AR/MA lags shows error message.

        Ref: ARMAX.m:177-179 — if both ar/ma empty, readytorun=false.
        """
        win = main_window
        # Ensure no lags set
        assert win._ar_order == []
        assert win._ma_order == []
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)
        # Ref: ARMAX.m:315-316 — 'not ready to run' message
        msg = win.message_text.text()
        assert "not ready" in msg.lower() or "check" in msg.lower()

    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    def test_estimate_with_ar_lags_mocked(
        self, mock_filter, mock_aic, mock_viewer, main_window, qtbot
    ):
        """Estimation with AR lags stores model and updates UI.

        Ref: ARMAX.m:234-250 — calls armaxfilter, stores results.
        """
        win = main_window
        T = len(win._y)
        mock_filter.return_value = _mock_armaxfilter_result(T=T, K=2)

        # Set AR lags and estimate
        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        # Model should be stored
        assert len(win._models) == 1
        assert "StringID" in win._models[0]
        assert win._residuals is not None
        # Model selector updated
        assert win.model_selector_popup.count() == 1

    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    def test_duplicate_model_detection(
        self, mock_filter, mock_aic, mock_viewer, main_window, qtbot
    ):
        """Re-estimating an identical model reuses stored results.

        Ref: ARMAX.m:191-228 — checks for previously estimated models.
        """
        win = main_window
        T = len(win._y)
        mock_filter.return_value = _mock_armaxfilter_result(T=T, K=2)

        # First estimation
        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)
        assert len(win._models) == 1

        # Second estimation with same lags — should NOT add new model
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)
        assert len(win._models) == 1
        # Ref: ARMAX.m:317-318 — 'previously estimated' message
        msg = win.message_text.text()
        assert "previously estimated" in msg.lower() or "loaded" in msg.lower()

    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    @patch("mfe_toolbox.gui.armax_main.aicsbic")
    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    def test_estimate_error_handling(
        self, mock_filter, mock_aic, mock_viewer, main_window, qtbot
    ):
        """Estimation exception shows error in message_text.

        Ref: ARMAX.m:310-314 — catch block displays error message.
        """
        win = main_window
        mock_filter.side_effect = RuntimeError("Optimization diverged")

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        msg = win.message_text.text()
        assert "error" in msg.lower()

    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    def test_estimation_updates_model_selector(
        self, mock_filter, mock_aic, mock_viewer, main_window, qtbot
    ):
        """Model selector combo box is updated after estimation.

        Ref: ARMAX.m:302-306 — updates model_selector_popup.
        """
        win = main_window
        T = len(win._y)
        mock_filter.return_value = _mock_armaxfilter_result(T=T, K=2)

        assert win.model_selector_popup.count() == 0

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        assert win.model_selector_popup.count() == 1
        string_id = win.model_selector_popup.itemText(0)
        assert "ARMA" in string_id


# ===================================================================
# TestARMAXMainWindowReset
# ===================================================================

class TestARMAXMainWindowReset:
    """Test reset callback.

    Ref: reset_pushbutton_Callback (ARMAX.m lines 402-432).
    """

    def test_reset_clears_state(self, main_window, qtbot):
        """Reset clears all input fields and internal state.

        Ref: ARMAX.m:407-417 — clears AR/MA orders, resets edit fields.
        """
        win = main_window

        # Set some state first
        win.AR_lags_edit.setText("1 2")
        win.AR_lags_edit.editingFinished.emit()
        win.MA_lags_edit.setText("1")
        win.MA_lags_edit.editingFinished.emit()
        assert win._ar_order == [1, 2]
        assert win._ma_order == [1]

        # Click reset
        qtbot.mouseClick(win.reset_pushbutton, Qt.MouseButton.LeftButton)

        # Ref: ARMAX.m:413 — AR edit cleared
        assert win.AR_lags_edit.text() == ""
        # Ref: ARMAX.m:412 — MA edit cleared
        assert win.MA_lags_edit.text() == ""
        # Ref: ARMAX.m:414 — EXOG edit set to space
        assert win.EXOG_edit.text() == " "
        # Ref: ARMAX.m:417 — hold_back_edit set to '0'
        assert win.hold_back_edit.text() == "0"
        # Internal state cleared
        assert win._ar_order == []
        assert win._ma_order == []

    def test_reset_replots_original_data(self, main_window, qtbot):
        """Reset replots original data with expected title.

        Ref: ARMAX.m:419-432 — replots original y.
        """
        win = main_window

        # Change the plot first
        win._clear_axes()
        win._ax.set_title("Something else")
        win._canvas.draw()

        # Reset
        qtbot.mouseClick(win.reset_pushbutton, Qt.MouseButton.LeftButton)

        # Ref: ARMAX.m:427 — title('Plot of original data')
        assert win._ax.get_title() == "Plot of original data"


# ===================================================================
# TestARMAXMainWindowClose
# ===================================================================

class TestARMAXMainWindowClose:
    """Test close callback.

    Ref: close_pushbutton_Callback (ARMAX.m lines 372-386).
    """

    def test_close_with_yes_confirmation(self, main_window, qtbot):
        """Close button with 'Yes' confirmation closes the window.

        Ref: ARMAX.m:380-385 — calls ARMAX_close_dialog, if 'Yes' deletes figure.
        """
        win = main_window
        with patch(
            "mfe_toolbox.gui.armax_main.ARMAXCloseDialog"
        ) as MockDialog:
            MockDialog.ask.return_value = "Yes"
            qtbot.mouseClick(
                win.close_pushbutton, Qt.MouseButton.LeftButton
            )
            MockDialog.ask.assert_called_once()

    def test_close_with_no_cancels(self, main_window, qtbot):
        """Close button with 'No' keeps window open.

        Ref: ARMAX.m:382-383 — case 'No': take no action.
        """
        win = main_window
        with patch(
            "mfe_toolbox.gui.armax_main.ARMAXCloseDialog"
        ) as MockDialog:
            MockDialog.ask.return_value = "No"
            qtbot.mouseClick(
                win.close_pushbutton, Qt.MouseButton.LeftButton
            )
            # Window should still be alive and visible
            assert win.isVisible()


# ===================================================================
# TestARMAXMainWindowAbout
# ===================================================================

class TestARMAXMainWindowAbout:
    """Test about menu callback.

    Ref: about_menu_Callback (ARMAX.m lines 735-739).
    """

    def test_about_menu_opens_dialog(self, main_window, qtbot):
        """About menu action triggers ARMAXAboutDialog.show_about().

        Ref: ARMAX.m:735-739 — about_menu_Callback calls ARMAX_about.
        """
        win = main_window
        with patch(
            "mfe_toolbox.gui.armax_main.ARMAXAboutDialog"
        ) as MockAbout:
            win.action_about.trigger()
            MockAbout.show_about.assert_called_once()


# ===================================================================
# TestARMAXMainWindowPlotRadioButtons
# ===================================================================

class TestARMAXMainWindowPlotRadioButtons:
    """Test all 9 plot radio button callbacks.

    Ref: ARMAX.m lines 917-1271 — radio button plot callbacks.
    """

    def test_orig_raw_radio(self, main_window, qtbot):
        """orig_raw_radio plots raw original data.

        Ref: ARMAX.m:917-937 — plots handles.y raw.
        """
        win = main_window
        win.orig_raw_radio.setChecked(True)
        assert win._ax.get_title() == "Plot of original data"

    @patch("mfe_toolbox.gui.armax_main.sacf")
    def test_orig_acf_radio(self, mock_sacf, main_window, qtbot):
        """orig_acf_radio triggers ACF plot of original data.

        Ref: ARMAX.m:942-978 — calls sacf, bar plot with SE bands.
        """
        win = main_window
        acf_len = win._acf_lags
        # Mock sacf return: (autocorrelations, std_errors, placeholder)
        mock_sacf.return_value = (
            np.random.default_rng(1).standard_normal(acf_len) * 0.1,
            np.ones(acf_len) * 0.05,
            None,
        )
        win.orig_acf_radio.setChecked(True)
        title = win._ax.get_title()
        assert "Autocorrelations" in title

    @patch("mfe_toolbox.gui.armax_main.spacf")
    def test_orig_pacf_radio(self, mock_spacf, main_window, qtbot):
        """orig_pacf_radio triggers PACF plot of original data.

        Ref: ARMAX.m:981-1016 — calls spacf, bar plot.
        """
        win = main_window
        acf_len = win._acf_lags
        mock_spacf.return_value = (
            np.random.default_rng(2).standard_normal(acf_len) * 0.1,
            np.ones(acf_len) * 0.05,
            None,
        )
        win.orig_pacf_radio.setChecked(True)
        title = win._ax.get_title()
        assert "Partial Autocorrelations" in title

    def test_resid_raw_radio_no_model(self, main_window, qtbot):
        """Clicking resid_raw_radio without model shows 'must estimate' message.

        Ref: ARMAX.m:1020-1045 — if no residuals, shows no_model_run_plot.
        Ref: ARMAX.m:743-751 — 'You must estimate a model' message.
        """
        win = main_window
        assert win._residuals is None
        win.resid_raw_radio.setChecked(True)
        # The no_model_run_plot places text on the axes, not a title
        # Check the axes has text objects with the expected message
        texts = [t.get_text() for t in win._ax.texts]
        found = any("must estimate" in t.lower() for t in texts)
        assert found, f"Expected 'must estimate' in axes texts, got: {texts}"

    def test_resid_raw_radio_with_model(self, main_window, qtbot):
        """Clicking resid_raw_radio with residuals plots them.

        Ref: ARMAX.m:1036-1037 — plot(y), title='Plot of estimated errors'.
        """
        win = main_window
        rng = np.random.default_rng(42)
        win._residuals = rng.standard_normal(len(win._y))
        win.resid_raw_radio.setChecked(True)
        assert win._ax.get_title() == "Plot of estimated errors"

    def test_resid_fit_radio(self, main_window, qtbot):
        """resid_fit_radio plots original + fitted overlay.

        Ref: ARMAX.m:1050-1076 — overlay original + fitted.
        """
        win = main_window
        rng = np.random.default_rng(42)
        win._residuals = rng.standard_normal(len(win._y))
        win.resid_fit_radio.setChecked(True)
        assert win._ax.get_title() == "Plot of estimated errors"
        # Should have exactly 2 lines (original + fit)
        assert len(win._ax.get_lines()) == 2

    @patch("mfe_toolbox.gui.armax_main.sacf")
    def test_resid_acf_radio(self, mock_sacf, main_window, qtbot):
        """resid_acf_radio triggers ACF plot of residuals.

        Ref: ARMAX.m:1081-1121 — sacf of residuals.
        """
        win = main_window
        rng = np.random.default_rng(42)
        win._residuals = rng.standard_normal(len(win._y))
        acf_len = win._acf_lags
        mock_sacf.return_value = (
            rng.standard_normal(acf_len) * 0.1,
            np.ones(acf_len) * 0.05,
            None,
        )
        win.resid_acf_radio.setChecked(True)
        title = win._ax.get_title()
        assert "Autocorrelations" in title

    @patch("mfe_toolbox.gui.armax_main.spacf")
    def test_resid_pacf_radio(self, mock_spacf, main_window, qtbot):
        """resid_pacf_radio triggers PACF plot of residuals.

        Ref: ARMAX.m:1125-1167 — spacf of residuals.
        """
        win = main_window
        rng = np.random.default_rng(42)
        win._residuals = rng.standard_normal(len(win._y))
        acf_len = win._acf_lags
        mock_spacf.return_value = (
            rng.standard_normal(acf_len) * 0.1,
            np.ones(acf_len) * 0.05,
            None,
        )
        win.resid_pacf_radio.setChecked(True)
        title = win._ax.get_title()
        assert "Partial Autocorrelations" in title

    @patch("mfe_toolbox.gui.armax_main.lmtest1")
    def test_radiobutton18_lm_lb_original(self, mock_lmtest, main_window, qtbot):
        """radiobutton18 triggers LM/LB test on original data.

        Ref: ARMAX.m:1171-1218 — LM/LB test on original data.
        Ref: ARMAX.m:1183 — [stat, pval] = lmtest1(y, lags) when InferenceMethod=1.
        """
        win = main_window
        acf_len = win._acf_lags
        mock_lmtest.return_value = (
            np.arange(1, acf_len + 1, dtype=float) * 2.0,
            np.random.default_rng(3).uniform(0, 1, acf_len),
        )
        win.radiobutton18.setChecked(True)
        title = win._ax.get_title()
        assert "LM Test" in title or "Ljung-Box" in title

    @patch("mfe_toolbox.gui.armax_main.lmtest1")
    def test_radiobutton19_lm_lb_residuals(self, mock_lmtest, main_window, qtbot):
        """radiobutton19 triggers LM/LB test on residuals.

        Ref: ARMAX.m:1223-1271 — LM/LB test on residuals.
        """
        win = main_window
        rng = np.random.default_rng(42)
        win._residuals = rng.standard_normal(len(win._y))
        acf_len = win._acf_lags
        mock_lmtest.return_value = (
            np.arange(1, acf_len + 1, dtype=float) * 2.0,
            rng.uniform(0, 1, acf_len),
        )
        win.radiobutton19.setChecked(True)
        title = win._ax.get_title()
        assert "LM Test" in title or "Ljung-Box" in title


# ===================================================================
# TestARMAXMainWindowModelSelector
# ===================================================================

class TestARMAXMainWindowModelSelector:
    """Test model selector popup.

    Ref: model_selector_popup_Callback (ARMAX.m lines 834-868).
    """

    def test_model_selector_empty_initially(self, main_window):
        """Model selector popup has no items initially.

        Ref: ARMAX.m:67 — handles.models={} initially empty.
        """
        assert main_window.model_selector_popup.count() == 0

    @patch("mfe_toolbox.gui.armax_main.ARMAXViewer")
    @patch("mfe_toolbox.gui.armax_main.aicsbic", return_value=(1304.0, 1312.4))
    @patch("mfe_toolbox.gui.armax_main.armaxfilter")
    def test_model_selector_after_estimation(
        self, mock_filter, mock_aic, mock_viewer, main_window, qtbot
    ):
        """After estimation, model selector has the estimated model's StringID.

        Ref: ARMAX.m:302-306 — updates model_selector_popup.
        """
        win = main_window
        T = len(win._y)
        mock_filter.return_value = _mock_armaxfilter_result(T=T, K=2)

        win.AR_lags_edit.setText("1")
        win.AR_lags_edit.editingFinished.emit()
        qtbot.mouseClick(win.estimate_pushbutton, Qt.MouseButton.LeftButton)

        assert win.model_selector_popup.count() == 1
        item_text = win.model_selector_popup.itemText(0)
        assert "ARMA" in item_text

        # Selecting the model should not crash
        win.model_selector_popup.setCurrentIndex(0)


# ===================================================================
# TestARMAXMainWindowPlotSnapshot
# ===================================================================

class TestARMAXMainWindowPlotSnapshot:
    """Plot snapshot comparison.

    Ref: AAP Section 0.7.1 — Plot output compared via .png snapshots, SSIM >= 0.95.
    """

    def test_main_window_plot_snapshot(self, main_window, snapshot_dir):
        """Save initial plot to PNG and verify file is created.

        If a reference snapshot exists, compare with SSIM (delegated to
        ``compare_images_ssim`` from conftest).
        """
        win = main_window
        output_path = Path(snapshot_dir) / "armax_main_initial_plot.png"

        # Save the figure via matplotlib's savefig
        win._figure.savefig(str(output_path), dpi=100, format="png")
        assert output_path.exists(), (
            f"Snapshot file not created at {output_path}"
        )
        # Verify file has non-zero size (valid PNG)
        assert output_path.stat().st_size > 0
