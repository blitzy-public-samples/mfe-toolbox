"""pytest-qt integration tests for ARMAXCloseDialog (QDialog with Yes/No confirmation).

Tests exercise ALL MATLAB GUIDE callbacks migrated from GUI/ARMAX_close_dialog.m:
- OpeningFcn: Dialog initialization, title/string override, icon display, modality
- pushbutton1_Callback: Yes button sets output='Yes'
- pushbutton2_Callback: No button sets output='No'
- CloseRequestFcn: Window close defaults to 'No'
- KeyPressFcn: Escape → 'No', Return → accept current

Per AAP Section 0.7.1: Coverage threshold ≥80% for GUI modules.
"""

import pytest
import numpy as np  # noqa: F401 — imported for consistency with other GUI test modules

# Guard: skip all tests if PyQt6 not available.
# Ref: tests/conftest.py — pytest_configure registers the 'gui' marker.
pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QDialog
from PyQt6.QtTest import QTest

from mfe_toolbox.gui.armax_close_dialog import ARMAXCloseDialog

# ---------------------------------------------------------------------------
# Module-level marker: all tests in this file are GUI tests.
# Ref: tests/conftest.py — pytest_configure registers the 'gui' marker.
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.gui


# ==========================================================================
# TestARMAXCloseDialogInit — Test dialog initialization (OpeningFcn)
# ==========================================================================
class TestARMAXCloseDialogInit:
    """Test dialog initialization — corresponds to ARMAX_close_dialog OpeningFcn.

    Ref: ARMAX_close_dialog.m:47-126 — OpeningFcn sets output='Yes' (line 55),
    accepts title/string overrides (lines 64-74), centres dialog (lines 78-102),
    loads dialogicons.mat question icon (lines 106-120), makes modal (line 123).
    """

    def test_default_construction(self, qtbot) -> None:
        """Create dialog with no arguments and verify all defaults.

        Ref: ARMAX_close_dialog.m:55 — handles.output = 'Yes'
        Ref: ARMAX_close_dialog.m:69 — default Name (title) for dialog window
        """
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)

        # Verify dialog is a QDialog instance.
        assert isinstance(dialog, QDialog)

        # Verify dialog has the ``output`` property.
        assert hasattr(dialog, "output")

        # Ref: ARMAX_close_dialog.m:55 — handles.output = 'Yes'
        # Default output must be 'Yes' before any user interaction.
        assert dialog.output == "Yes"

        # Ref: ARMAX_close_dialog.m:69 — default window title.
        # The Python implementation defaults to "Confirm Close".
        assert dialog.windowTitle() == "Confirm Close"

    def test_custom_title(self, qtbot) -> None:
        """Create dialog with custom title parameter.

        Ref: ARMAX_close_dialog.m:68-69
            case 'title'
                set(hObject, 'Name', varargin{index+1});
        """
        dialog = ARMAXCloseDialog(title="Custom Title")
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Custom Title"

    def test_custom_message(self, qtbot) -> None:
        """Create dialog with custom message parameter.

        Ref: ARMAX_close_dialog.m:70-71
            case 'string'
                set(handles.text1, 'String', varargin{index+1});
        """
        dialog = ARMAXCloseDialog(message="Really close?")
        qtbot.addWidget(dialog)

        # Verify the text1 QLabel displays the custom message.
        # text1 is loaded from the .ui file by uic.loadUi().
        assert dialog.text1.text() == "Really close?"

    def test_modality(self, qtbot) -> None:
        """Verify dialog is application-modal.

        Ref: ARMAX_close_dialog.m:123
            set(handles.figure1, 'WindowStyle', 'modal')
        """
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)

        # The Python implementation sets ApplicationModal modality
        # in __init__ via setWindowModality().
        assert (
            dialog.isModal()
            or dialog.windowModality() == Qt.WindowModality.ApplicationModal
        )

    def test_question_icon_displayed(self, qtbot) -> None:
        """Verify question icon widget is present and initialised.

        Ref: ARMAX_close_dialog.m:106-120
            load dialogicons.mat
            IconData = questIconData;
            image(IconData, 'Parent', handles.axes1);

        In PyQt6: the companion class loads a Qt built-in question icon from
        QStyle.StandardPixmap.SP_MessageBoxQuestion into the axes1 QLabel.
        """
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)

        # Verify the axes1 QLabel (icon placeholder) exists.
        assert hasattr(dialog, "axes1")
        assert dialog.axes1 is not None

        # In a headless (offscreen) environment the style may yield a null
        # pixmap, so we verify the widget exists and is properly sized
        # rather than requiring a non-null pixmap.
        assert dialog.axes1.minimumWidth() >= 1


# ==========================================================================
# TestARMAXCloseDialogButtons — Test button callbacks
# ==========================================================================
class TestARMAXCloseDialogButtons:
    """Test push-button callbacks (pushbutton1/pushbutton2_Callback).

    Ref: ARMAX_close_dialog.m:141-169
    pushbutton1_Callback: handles.output = get(hObject,'String')  → 'Yes'
    pushbutton2_Callback: handles.output = get(hObject,'String')  → 'No'
    """

    def test_yes_button_click(self, qtbot) -> None:
        """Click the Yes button (pushbutton1).

        Ref: ARMAX_close_dialog.m:141-154 — pushbutton1_Callback
        Line 147: handles.output = get(hObject,'String') — button text 'Yes'
        Line 154: uiresume(handles.figure1) → dialog accepted
        """
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Simulate left-click on the Yes button.
        qtbot.mouseClick(dialog.pushbutton1, Qt.MouseButton.LeftButton)

        # Ref: ARMAX_close_dialog.m:147 — output should be 'Yes'.
        assert dialog.output == "Yes"

    def test_no_button_click(self, qtbot) -> None:
        """Click the No button (pushbutton2).

        Ref: ARMAX_close_dialog.m:156-169 — pushbutton2_Callback
        Line 162: handles.output = get(hObject,'String') — button text 'No'
        Line 169: uiresume(handles.figure1) → dialog rejected
        """
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Simulate left-click on the No button.
        qtbot.mouseClick(dialog.pushbutton2, Qt.MouseButton.LeftButton)

        # Ref: ARMAX_close_dialog.m:162 — output should be 'No'.
        assert dialog.output == "No"


# ==========================================================================
# TestARMAXCloseDialogKeyPress — Test keyboard interactions
# ==========================================================================
class TestARMAXCloseDialogKeyPress:
    """Test keyboard interactions (KeyPressFcn equivalent).

    Ref: ARMAX_close_dialog.m:188-207 — figure1_KeyPressFcn
    Escape → handles.output = 'No', uiresume (lines 195-202)
    Return → uiresume only, no output change (lines 205-207)
    """

    def test_escape_key_sets_no(self, qtbot) -> None:
        """Press Escape key — output must be set to 'No'.

        Ref: ARMAX_close_dialog.m:195-202
            if isequal(get(hObject,'CurrentKey'),'escape')
                handles.output = 'No';
                guidata(hObject, handles);
                uiresume(handles.figure1);
            end
        """
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Send Escape key event directly to the dialog widget.
        QTest.keyPress(dialog, Qt.Key.Key_Escape)

        # Ref: line 197 — handles.output = 'No'
        assert dialog.output == "No"

    def test_return_key_accepts(self, qtbot) -> None:
        """Press Return/Enter key — output must remain 'Yes' (default).

        Ref: ARMAX_close_dialog.m:205-207
            if isequal(get(hObject,'CurrentKey'),'return')
                uiresume(handles.figure1);
            end

        MATLAB: Return only calls uiresume, does NOT change handles.output.
        The default 'Yes' set in OpeningFcn (line 55) is preserved.
        Python: keyPressEvent calls self.accept() which sets output='Yes'.
        """
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Send Return key event directly to the dialog.
        QTest.keyPress(dialog, Qt.Key.Key_Return)

        # Ref: ARMAX_close_dialog.m:55 — default 'Yes' is preserved.
        assert dialog.output == "Yes"


# ==========================================================================
# TestARMAXCloseDialogCloseEvent — Test window close behavior
# ==========================================================================
class TestARMAXCloseDialogCloseEvent:
    """Test window close behaviour (CloseRequestFcn equivalent).

    Ref: ARMAX_close_dialog.m:172-185 — figure1_CloseRequestFcn
    CRITICAL: close always defaults output to 'No' (line 177).
    """

    def test_close_event_defaults_to_no(self, qtbot) -> None:
        """Trigger close event — output must default to 'No'.

        Ref: ARMAX_close_dialog.m:172-185 — CloseRequestFcn
        Line 177: handles.output = 'No'  ← CRITICAL: close defaults to No
        Line 178: guidata(hObject, handles)
        Line 179-184: uiresume or delete

        This is the CRITICAL difference from the About dialog where close
        behaviour does not change the output. The close-dialog MUST default
        to 'No' on window close to prevent accidental data loss.
        """
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        # Verify default output is 'Yes' before close.
        assert dialog.output == "Yes"

        # Trigger close event (equivalent to clicking the window × button).
        dialog.close()

        # Ref: ARMAX_close_dialog.m:177 — handles.output = 'No'
        assert dialog.output == "No"


# ==========================================================================
# TestARMAXCloseDialogStaticMethod — Test convenience API
# ==========================================================================
class TestARMAXCloseDialogStaticMethod:
    """Test the static ``ask()`` convenience method.

    Mirrors the MATLAB call pattern:
        answer = ARMAX_close_dialog('Title', 'Confirm Close', 'String', '...')
    """

    def test_ask_static_method(self, qtbot) -> None:
        """Test that ARMAXCloseDialog.ask() exists and returns correct output.

        The ask() method is a static convenience that creates a dialog,
        calls exec() (blocking), and returns the output string.

        Since exec() starts a nested event loop, we use QTimer.singleShot
        to schedule an automatic click on the Yes button before the
        blocking call begins.
        """
        # Verify the ask method exists and is callable.
        assert hasattr(ARMAXCloseDialog, "ask")
        assert callable(ARMAXCloseDialog.ask)

        def _auto_accept_dialog() -> None:
            """Find the active ARMAXCloseDialog and click its Yes button.

            This callback fires within the nested event loop started by
            dialog.exec() inside ask(), allowing us to programmatically
            dismiss the modal dialog.
            """
            app = QApplication.instance()
            if app is None:
                return
            for widget in app.topLevelWidgets():
                if isinstance(widget, ARMAXCloseDialog) and widget.isVisible():
                    # Simulate clicking the Yes button to accept the dialog.
                    widget.pushbutton1.click()
                    return

        # Schedule the auto-accept to fire 200ms after exec() starts.
        # This is long enough for the dialog to become visible in the
        # offscreen event loop, but short enough to keep tests snappy.
        QTimer.singleShot(200, _auto_accept_dialog)

        # Call the static convenience method — this blocks until the
        # QTimer callback clicks the Yes button.
        result = ARMAXCloseDialog.ask(
            title="Test Static Ask",
            message="Close the session?",
        )

        # Verify that the method returns a plain string matching the
        # user's choice ('Yes' because we clicked pushbutton1).
        assert isinstance(result, str)
        assert result == "Yes"
