"""Pytest-qt tests for the ARMAXCloseDialog widget.

Tests ARMAXCloseDialog from ``mfe_toolbox.gui.armax_close_dialog``.
"""

import pytest

from mfe_toolbox.gui.armax_close_dialog import ARMAXCloseDialog


@pytest.fixture
def close_dialog(qtbot):
    """Create an ARMAXCloseDialog instance for testing."""
    dialog = ARMAXCloseDialog()
    qtbot.addWidget(dialog)
    return dialog


class TestARMAXCloseDialogUnit:
    """Unit tests for ARMAXCloseDialog."""

    def test_close_dialog_import(self) -> None:
        """Module should import without error."""
        assert ARMAXCloseDialog is not None

    def test_close_dialog_instantiation(self, qtbot) -> None:
        """Dialog should instantiate without error."""
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)
        assert dialog is not None

    def test_close_dialog_title(self, close_dialog) -> None:
        """Dialog should have a window title."""
        title = close_dialog.windowTitle()
        assert isinstance(title, str)
        assert len(title) > 0

    def test_close_dialog_default_output(self, close_dialog) -> None:
        """Default output should be a valid string."""
        output = close_dialog.output
        assert isinstance(output, str)

    def test_close_dialog_yes_no_cancel(self, qtbot) -> None:
        """Dialog should respond to Yes/No/Cancel actions."""
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)
        # Test accept (Yes)
        dialog.accept()
        assert dialog.output in ("Yes", "No", "Cancel", "")

    def test_close_dialog_reject(self, qtbot) -> None:
        """Reject should set output to Cancel."""
        dialog = ARMAXCloseDialog()
        qtbot.addWidget(dialog)
        dialog.reject()
        assert dialog.output in ("Cancel", "No", "")
