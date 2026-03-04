"""pytest-qt integration tests for ARMAXAboutDialog (QDialog with Oxford logo).

Tests exercise ALL MATLAB GUIDE callbacks migrated from GUI/ARMAX_about.m:
- OpeningFcn: Dialog initialization, title/string override, logo display, modality
- pushbutton1_Callback: OK button sets output
- pushbutton2_Callback: Secondary button sets output
- CloseRequestFcn: Window close behavior
- KeyPressFcn: Escape → 'No', Return → accept

Per AAP Section 0.7.1: Coverage threshold ≥80% for GUI modules.

Ref: GUI/ARMAX_about.m — 206 lines, complete callback implementation.
"""

import pytest
import numpy as np
from pathlib import Path

# Guard: skip ALL tests in this module if PyQt6 is not available.
# Per schema: pytest.importorskip('PyQt6') for conditional test skipping.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog
from PyQt6.QtTest import QTest

from mfe_toolbox.gui.armax_about import ARMAXAboutDialog

# Mark all tests as GUI tests for selective execution.
# Per AAP Section 0.7.1: gui marker registered in tests/conftest.py:pytest_configure.
pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# TestARMAXAboutDialogInit — OpeningFcn (ARMAX_about.m lines 47-126)
# ---------------------------------------------------------------------------

class TestARMAXAboutDialogInit:
    """Test dialog initialization — exercises OpeningFcn.

    Ref: ARMAX_about.m lines 47-126 — ARMAX_about_OpeningFcn
    Validates default output, title, message, modality, logo, and widget
    existence matching the original GUIDE figure initialisation sequence.
    """

    def test_default_construction(self, qtbot):
        """Verify default dialog construction sets expected initial state.

        Ref: ARMAX_about.m:55 — handles.output = 'Yes'
        Ref: ARMAX_about.m:69 — default window title 'About'
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Dialog is a QDialog subclass.
        assert isinstance(dialog, QDialog)
        # Ref: ARMAX_about.m:55 — default output is 'Yes'.
        assert dialog.output == "Yes"
        # Ref: armax_about.py _DEFAULT_TITLE = "About".
        assert dialog.windowTitle() == "About"
        # The output property must be accessible.
        assert hasattr(dialog, "output")

        dialog.close()

    def test_custom_title(self, qtbot):
        """Verify custom title parameter override.

        Ref: ARMAX_about.m:68-69 — case 'title': set(hObject, 'Name', varargin{index+1})
        """
        custom_title = "Custom About Title"
        dialog = ARMAXAboutDialog(title=custom_title)
        qtbot.addWidget(dialog)
        dialog.show()

        assert dialog.windowTitle() == custom_title

        dialog.close()

    def test_custom_message(self, qtbot):
        """Verify custom message/string parameter override.

        Ref: ARMAX_about.m:70-71 — case 'string': set(handles.text1, 'String', varargin{index+1})
        """
        custom_msg = "This is a custom about message for testing"
        dialog = ARMAXAboutDialog(message=custom_msg)
        qtbot.addWidget(dialog)
        dialog.show()

        # text1 is the QLabel displaying the body message.
        assert dialog.text1.text() == custom_msg

        dialog.close()

    def test_modality(self, qtbot):
        """Verify dialog is application-modal.

        Ref: ARMAX_about.m:123 — set(handles.figure1,'WindowStyle','modal')
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # The dialog must be modal — matching MATLAB 'modal' WindowStyle.
        assert dialog.windowModality() == Qt.WindowModality.ApplicationModal
        assert dialog.isModal()

        dialog.close()

    def test_logo_display(self, qtbot):
        """Verify Oxford logo display or graceful degradation when missing.

        Ref: ARMAX_about.m:106-120 — loads OxLogo.mat, displays in axes1.
        The Python equivalent uses icon_label (QLabel) with a QPixmap or
        placeholder text when the asset file is absent.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # icon_label widget must exist (replaces MATLAB axes1).
        assert hasattr(dialog, "icon_label")
        assert dialog.icon_label is not None

        # Determine if the logo asset file exists on disk.
        logo_path = (
            Path(__file__).resolve().parent.parent.parent
            / "mfe_toolbox"
            / "gui"
            / "assets"
            / "ox_logo.png"
        )
        if logo_path.exists():
            # When asset exists, pixmap should be loaded and non-null.
            pixmap = dialog.icon_label.pixmap()
            assert pixmap is not None and not pixmap.isNull()
        else:
            # Graceful degradation — placeholder text displayed, no crash.
            text = dialog.icon_label.text()
            assert text is not None
            assert len(text) > 0

        dialog.close()

    def test_widgets_exist(self, qtbot):
        """Verify all expected child widgets are present.

        Ref: ARMAX_about.m — handles.pushbutton1, handles.pushbutton2,
        handles.text1, handles.axes1 (→ icon_label).
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # All widgets from _build_ui must be accessible as attributes.
        assert hasattr(dialog, "pushbutton1")
        assert hasattr(dialog, "pushbutton2")
        assert hasattr(dialog, "text1")
        assert hasattr(dialog, "icon_label")

        # Verify widget types.
        from PyQt6.QtWidgets import QPushButton, QLabel

        assert isinstance(dialog.pushbutton1, QPushButton)
        assert isinstance(dialog.pushbutton2, QPushButton)
        assert isinstance(dialog.text1, QLabel)
        assert isinstance(dialog.icon_label, QLabel)

        dialog.close()

    def test_button_labels(self, qtbot):
        """Verify default button labels match the .ui file definition.

        Ref: armax_about.ui — pushbutton1 text is 'OK'.
        Ref: armax_about.ui — pushbutton2 text is 'Cancel'.
        Note: The .ui file takes precedence over the _build_ui() fallback
        (which would use 'Close'). At runtime the .ui file is loaded.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        assert dialog.pushbutton1.text() == "OK"
        # pushbutton2 label is "Cancel" per the .ui file definition.
        assert dialog.pushbutton2.text() == "Cancel"

        dialog.close()

    def test_default_output_before_interaction(self, qtbot):
        """Verify output property returns 'Yes' before any user interaction.

        Ref: ARMAX_about.m:55 — handles.output = 'Yes' (set in OpeningFcn
        before any callback fires).
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        # Do NOT show or interact — just check the initial state.
        assert dialog.output == "Yes"

    def test_parent_widget(self, qtbot):
        """Verify dialog can be created with a parent widget.

        Ref: ARMAX_about.m:78-102 — dialog centering logic uses parent geometry.
        """
        from PyQt6.QtWidgets import QWidget

        parent = QWidget()
        qtbot.addWidget(parent)
        parent.resize(800, 600)
        parent.show()

        dialog = ARMAXAboutDialog(parent=parent)
        qtbot.addWidget(dialog)
        dialog.show()

        assert dialog.parentWidget() is parent
        assert dialog.windowTitle() == "About"

        dialog.close()
        parent.close()


# ---------------------------------------------------------------------------
# TestARMAXAboutDialogButtons — pushbutton callbacks (lines 141-169)
# ---------------------------------------------------------------------------

class TestARMAXAboutDialogButtons:
    """Test button callbacks — exercises pushbutton1_Callback and pushbutton2_Callback.

    Ref: ARMAX_about.m lines 141-169
    Both callbacks read the button's String property and store it as output,
    then resume the figure event loop (uiresume).
    """

    def test_pushbutton1_click(self, qtbot):
        """Click pushbutton1 (OK) and verify output set to button text.

        Ref: ARMAX_about.m:147 — handles.output = get(hObject,'String')
        Ref: ARMAX_about.m:154 — uiresume(handles.figure1) → accept()
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Simulate mouse click on the OK button.
        qtbot.mouseClick(dialog.pushbutton1, Qt.MouseButton.LeftButton)

        # The output should be the button's text label ("OK").
        assert dialog.output == "OK"
        # The dialog should have been accepted (uiresume equivalent).
        assert dialog.result() == QDialog.DialogCode.Accepted

    def test_pushbutton2_click(self, qtbot):
        """Click pushbutton2 (Cancel) and verify output set to button text.

        Ref: ARMAX_about.m:162 — handles.output = get(hObject,'String')
        Ref: ARMAX_about.m:169 — uiresume(handles.figure1) → accept()
        Note: The .ui file labels pushbutton2 as 'Cancel'.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Simulate mouse click on the Cancel button.
        qtbot.mouseClick(dialog.pushbutton2, Qt.MouseButton.LeftButton)

        # The output should be the button's text label ("Cancel").
        assert dialog.output == "Cancel"
        # Both buttons accept the dialog (MATLAB behaviour: uiresume).
        assert dialog.result() == QDialog.DialogCode.Accepted

    def test_pushbutton1_changes_default_output(self, qtbot):
        """Verify pushbutton1 click changes output from the default 'Yes' to 'OK'.

        Ref: ARMAX_about.m:55 — initial output 'Yes'
        Ref: ARMAX_about.m:147 — overwritten to button String on click.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Before clicking, output should be the default 'Yes'.
        assert dialog.output == "Yes"

        qtbot.mouseClick(dialog.pushbutton1, Qt.MouseButton.LeftButton)

        # After clicking, output should be 'OK' (the button label).
        assert dialog.output == "OK"

    def test_pushbutton2_changes_default_output(self, qtbot):
        """Verify pushbutton2 click changes output from the default 'Yes' to 'Cancel'.

        Ref: ARMAX_about.m:55 — initial output 'Yes'
        Ref: ARMAX_about.m:162 — overwritten to button String on click.
        Note: The .ui file labels pushbutton2 as 'Cancel'.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        assert dialog.output == "Yes"

        qtbot.mouseClick(dialog.pushbutton2, Qt.MouseButton.LeftButton)

        assert dialog.output == "Cancel"


# ---------------------------------------------------------------------------
# TestARMAXAboutDialogKeyPress — KeyPressFcn (lines 188-206)
# ---------------------------------------------------------------------------

class TestARMAXAboutDialogKeyPress:
    """Test keyboard interactions — exercises KeyPressFcn.

    Ref: ARMAX_about.m lines 188-206 — figure1_KeyPressFcn
    Escape sets output to 'No' and resumes; Return resumes without changing
    output.
    """

    def test_escape_key_sets_no(self, qtbot):
        """Press Escape and verify output is set to 'No'.

        Ref: ARMAX_about.m:194-201 — if 'escape': handles.output = 'No', uiresume
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Send Escape key press directly to the dialog.
        QTest.keyPress(dialog, Qt.Key.Key_Escape)

        # Output must be 'No' — per MATLAB behaviour.
        assert dialog.output == "No"
        # Dialog should be rejected (not accepted).
        assert dialog.result() == QDialog.DialogCode.Rejected

    def test_return_key_accepts(self, qtbot):
        """Press Return/Enter and verify dialog is accepted.

        Ref: ARMAX_about.m:204-206 — if 'return': uiresume
        Note: In the MATLAB original, pressing Return does NOT change output;
        it just resumes the event loop. The Python keyPressEvent calls
        accept() without modifying _output.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Send Return key press to the dialog.
        QTest.keyPress(dialog, Qt.Key.Key_Return)

        # The dialog should be accepted (uiresume equivalent).
        assert dialog.result() == QDialog.DialogCode.Accepted

    def test_escape_key_changes_default_output(self, qtbot):
        """Verify Escape changes the default 'Yes' output to 'No'.

        Ref: ARMAX_about.m:55 — initial output 'Yes'
        Ref: ARMAX_about.m:196 — handles.output = 'No'
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Before pressing Escape, output is the default.
        assert dialog.output == "Yes"

        QTest.keyPress(dialog, Qt.Key.Key_Escape)

        # After Escape, output must be 'No'.
        assert dialog.output == "No"

    def test_other_key_does_not_close(self, qtbot):
        """Verify that pressing a non-special key does not close the dialog.

        Ref: ARMAX_about.m:188-206 — only 'escape' and 'return' are handled;
        all other keys are propagated to the base class without side effects.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Press 'A' key — should NOT close the dialog.
        QTest.keyPress(dialog, Qt.Key.Key_A)

        # Dialog should still be visible.
        assert dialog.isVisible()
        # Output should remain at the default.
        assert dialog.output == "Yes"

        dialog.close()


# ---------------------------------------------------------------------------
# TestARMAXAboutDialogClose — CloseRequestFcn (lines 172-184)
# ---------------------------------------------------------------------------

class TestARMAXAboutDialogClose:
    """Test window close behavior — exercises CloseRequestFcn.

    Ref: ARMAX_about.m lines 172-184 — figure1_CloseRequestFcn
    The MATLAB version checks waitstatus and calls uiresume or delete.
    The Python version calls accept() in closeEvent.
    """

    def test_close_event(self, qtbot):
        """Trigger close event and verify dialog closes cleanly.

        Ref: ARMAX_about.m:172-184 — CloseRequestFcn
        Ref: armax_about.py:196-204 — closeEvent calls accept()
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Verify dialog is visible before close.
        assert dialog.isVisible()

        # Trigger close event (equivalent to clicking X button).
        dialog.close()

        # Verify dialog is no longer visible.
        assert not dialog.isVisible()

    def test_close_event_result_is_accepted(self, qtbot):
        """Verify close event sets result to Accepted.

        Ref: armax_about.py:204 — closeEvent calls self.accept()
        The MATLAB CloseRequestFcn calls uiresume; the Python equivalent
        calls accept() so OutputFcn can read the output property.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        dialog.close()

        # closeEvent calls accept(), so result should be Accepted.
        assert dialog.result() == QDialog.DialogCode.Accepted

    def test_close_preserves_default_output(self, qtbot):
        """Verify closing without interaction preserves the default output.

        Ref: ARMAX_about.m:55 — handles.output = 'Yes' (never modified by close)
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        dialog.close()

        # Output should still be the default 'Yes'.
        assert dialog.output == "Yes"


# ---------------------------------------------------------------------------
# TestARMAXAboutDialogStaticMethod — Convenience API
# ---------------------------------------------------------------------------

class TestARMAXAboutDialogStaticMethod:
    """Test the static convenience method show_about().

    Ref: armax_about.py:140-171 — show_about creates dialog, calls exec(),
    and returns the output string. Since exec() blocks the event loop, we
    test the method's existence and the dialog creation path non-blockingly.
    """

    def test_show_about_exists_and_callable(self, qtbot):
        """Verify show_about is a callable static method.

        Ref: armax_about.py:140 — @staticmethod def show_about(...)
        """
        assert hasattr(ARMAXAboutDialog, "show_about")
        assert callable(ARMAXAboutDialog.show_about)

    def test_show_about_dialog_path(self, qtbot):
        """Verify the dialog creation path used by show_about.

        Creates a dialog with the same parameters show_about would use and
        verifies the resulting state — without calling exec() (which blocks).
        """
        custom_title = "Static Test Title"
        custom_msg = "Static test message body"

        dialog = ARMAXAboutDialog(
            title=custom_title, message=custom_msg
        )
        qtbot.addWidget(dialog)
        dialog.show()

        assert dialog.windowTitle() == custom_title
        assert dialog.text1.text() == custom_msg
        assert dialog.output == "Yes"

        dialog.close()

    def test_show_about_nonblocking(self, qtbot):
        """End-to-end test of show_about using QTimer to accept the dialog.

        Schedules a 50 ms timer that finds the About dialog and clicks OK
        before the blocking exec() call, then verifies the returned output.
        """
        from PyQt6.QtCore import QTimer

        def _accept_dialog():
            """Find and accept the ARMAXAboutDialog instance."""
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, ARMAXAboutDialog):
                    widget.accept()
                    return

        # Schedule acceptance before the blocking exec() call.
        QTimer.singleShot(50, _accept_dialog)

        output = ARMAXAboutDialog.show_about(
            title="Non-Blocking Test"
        )
        # Default output 'Yes' should be returned since we accepted
        # without changing the output.
        assert output == "Yes"


# ---------------------------------------------------------------------------
# TestARMAXAboutDialogAcceptReject — Direct accept/reject API
# ---------------------------------------------------------------------------

class TestARMAXAboutDialogAcceptReject:
    """Test direct accept() and reject() calls.

    Ref: armax_about.py:176-191 — accept() and reject() overrides.
    These are called by button clicks, key presses, and close events.
    """

    def test_accept_sets_accepted_result(self, qtbot):
        """Verify accept() sets result to Accepted.

        Ref: armax_about.py:176-181 — accept calls super().accept()
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        dialog.accept()

        assert dialog.result() == QDialog.DialogCode.Accepted

    def test_reject_sets_output_no(self, qtbot):
        """Verify reject() sets output to 'No' and result to Rejected.

        Ref: armax_about.py:183-191 — reject sets _output='No', then super().reject()
        Ref: ARMAX_about.m:196 — handles.output = 'No' on Escape
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        dialog.reject()

        assert dialog.output == "No"
        assert dialog.result() == QDialog.DialogCode.Rejected

    def test_accept_preserves_output(self, qtbot):
        """Verify accept() does not modify the current output value.

        Ref: armax_about.py:176-181 — accept() calls super().accept() only,
        it does not alter _output.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Output starts at 'Yes'.
        assert dialog.output == "Yes"

        dialog.accept()

        # accept() should not change the output.
        assert dialog.output == "Yes"

    def test_reject_overrides_any_prior_output(self, qtbot):
        """Verify reject() always sets output to 'No' regardless of prior state.

        If the output was changed by a button click and then reject() is
        called, the output must be reset to 'No'.
        """
        dialog = ARMAXAboutDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Simulate pushbutton1 click setting output to 'OK'.
        # We manipulate _output directly to simulate prior interaction
        # without triggering accept().
        dialog._output = "OK"
        assert dialog.output == "OK"

        dialog.reject()

        # reject() must override to 'No'.
        assert dialog.output == "No"
        assert dialog.result() == QDialog.DialogCode.Rejected
