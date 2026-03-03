"""
Save-before-close confirmation dialog for the ARMAX GUI.

Migrates MATLAB GUIDE ``GUI/ARMAX_close_dialog.m`` (207 lines) and its
companion ``GUI/ARMAX_close_dialog.fig`` layout to a PyQt6 QDialog class
with equivalent modal behaviour, signal/slot wiring, and keyboard handling.

The dialog presents a question icon, a configurable message, and Yes / No
push-buttons.  It returns the user's choice as a plain string (``'Yes'`` or
``'No'``) via the :pyattr:`output` property, exactly matching the MATLAB
``handles.output`` contract used by ``ARMAX_close_dialog_OutputFcn``.

Ref: GUI/ARMAX_close_dialog.m — full GUIDE structure migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)
from PyQt6 import uic


# ---------------------------------------------------------------------------
# Resolve the .ui companion file path once at module-load time.
# Ref: ARMAX_close_dialog.m — replaces MATLAB implicit GUIDE .fig path
# resolution with an explicit filesystem lookup relative to this module.
# ---------------------------------------------------------------------------
_UI_PATH: Path = Path(__file__).parent / "armax_close_dialog.ui"


class ARMAXCloseDialog(QDialog):
    """Save-before-close confirmation dialog.

    A modal dialog that asks the user to confirm whether they want to close
    the current ARMAX session.  The dialog displays a standard question icon
    (replacing the MATLAB ``dialogicons.mat`` asset), a text message, and
    *Yes* / *No* push-buttons.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget.  When provided the dialog is centred over this widget;
        otherwise it is centred on the primary screen.
        Ref: ARMAX_close_dialog.m:78-102 — centering over gcbf or screen.
    title : str, optional
        Window title.  Defaults to ``'Confirm Close'``.
        Ref: ARMAX_close_dialog.m:69 — ``set(hObject, 'Name', …)``.
    message : str, optional
        Body text displayed beside the question icon.  Defaults to
        ``'Do you want to close?'``.
        Ref: ARMAX_close_dialog.m:71 — ``set(handles.text1, 'String', …)``.

    Attributes
    ----------
    output : str
        Read-only property returning ``'Yes'`` or ``'No'`` after the dialog
        has been closed.
        Ref: ARMAX_close_dialog.m:136 — ``varargout{1} = handles.output``.

    Ref: GUI/ARMAX_close_dialog.m — GUIDE modal dialog replacement.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "Confirm Close",
        message: str = "Do you want to close?",
    ) -> None:
        """Initialise the close-confirmation dialog.

        Parameters
        ----------
        parent : QWidget or None
            Optional parent widget for centering.
        title : str
            Dialog window title.
            Ref: ARMAX_close_dialog.m:69 — switch/case 'title'.
        message : str
            Body text shown beside the question icon.
            Ref: ARMAX_close_dialog.m:71 — switch/case 'string'.
        """
        super().__init__(parent)

        # Ref: ARMAX_close_dialog.m:55 — handles.output = 'Yes'
        self._output: str = "Yes"

        # --- Load the Qt Designer layout ---------------------------------
        # Ref: ARMAX_close_dialog.m:28-34 — gui_State / gui_mainfcn replaced
        # by uic.loadUi which constructs the widget hierarchy from the .ui
        # XML file and assigns child widgets as instance attributes.
        uic.loadUi(str(_UI_PATH), self)

        # --- Apply caller-supplied title and message ---------------------
        # Ref: ARMAX_close_dialog.m:64-74 — varargin property/value pairs
        self.setWindowTitle(title)  # Ref: line 69
        # text1 is a QLabel defined in the .ui file (objectName="text1").
        # Ref: ARMAX_close_dialog.m:71 — set(handles.text1, 'String', …)
        self.text1: QLabel  # created by loadUi
        self.text1.setText(message)

        # --- Set modality ------------------------------------------------
        # Ref: ARMAX_close_dialog.m:123 — set(handles.figure1,'WindowStyle','modal')
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # --- Display a Qt built-in question icon -------------------------
        # Replaces MATLAB: load dialogicons.mat → questIconData/questIconMap
        # Ref: ARMAX_close_dialog.m:106-120
        self.axes1: QLabel  # created by loadUi (objectName="axes1")
        self._setup_question_icon()

        # --- Connect push-button signals to slots ------------------------
        # pushbutton1 → "Yes" button
        # Ref: ARMAX_close_dialog.m:141-154 — pushbutton1_Callback
        self.pushbutton1: QPushButton  # created by loadUi
        self.pushbutton1.clicked.connect(self._on_yes_clicked)

        # pushbutton2 → "No" button
        # Ref: ARMAX_close_dialog.m:156-169 — pushbutton2_Callback
        self.pushbutton2: QPushButton  # created by loadUi
        self.pushbutton2.clicked.connect(self._on_no_clicked)

        # --- Centre the dialog -------------------------------------------
        # Ref: ARMAX_close_dialog.m:78-102
        self._center_on_parent_or_screen()

    # ------------------------------------------------------------------
    # Public property
    # ------------------------------------------------------------------

    @property
    def output(self) -> str:
        """Return the user's choice as a string (``'Yes'`` or ``'No'``).

        Ref: ARMAX_close_dialog.m:136 — ``varargout{1} = handles.output``.
        """
        return self._output

    # ------------------------------------------------------------------
    # Static convenience method
    # ------------------------------------------------------------------

    @staticmethod
    def ask(
        parent: Optional[QWidget] = None,
        title: str = "Confirm Close",
        message: str = "Do you want to close?",
    ) -> str:
        """Show the dialog modally and return the user's choice.

        This mirrors the MATLAB call pattern::

            answer = ARMAX_close_dialog('Title','Confirm Close','String','…')

        Parameters
        ----------
        parent : QWidget or None
            Parent widget for centering.
        title : str
            Window title.
        message : str
            Body text.

        Returns
        -------
        str
            ``'Yes'`` if the user confirmed, ``'No'`` otherwise.
        """
        dialog = ARMAXCloseDialog(parent=parent, title=title, message=message)
        # Ref: ARMAX_close_dialog.m:126 — uiwait(handles.figure1)
        dialog.exec()
        return dialog.output

    # ------------------------------------------------------------------
    # Overridden QDialog slots: accept / reject
    # ------------------------------------------------------------------

    def accept(self) -> None:
        """Accept the dialog (user chose *Yes* or pressed Return).

        Sets :pyattr:`output` to ``'Yes'`` and closes the dialog.
        Ref: ARMAX_close_dialog.m:147-154 — pushbutton1_Callback sets
        handles.output to button String then calls uiresume.
        """
        self._output = "Yes"
        super().accept()

    def reject(self) -> None:
        """Reject the dialog (user chose *No* or pressed Escape).

        Sets :pyattr:`output` to ``'No'`` and closes the dialog.
        Ref: ARMAX_close_dialog.m:162-169 — pushbutton2_Callback sets
        handles.output to button String then calls uiresume.
        """
        self._output = "No"
        super().reject()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Handle the window-manager close event (e.g. title-bar × button).

        Ref: ARMAX_close_dialog.m:172-185 — ``figure1_CloseRequestFcn``
        sets ``handles.output = 'No'`` then calls ``uiresume`` or deletes.
        """
        # Ref: ARMAX_close_dialog.m:177 — handles.output = 'No'
        self._output = "No"
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        """Handle key-press events on the dialog.

        * **Escape** → set output to ``'No'`` and close (reject).
          Ref: ARMAX_close_dialog.m:195-202 — ``get(hObject,'CurrentKey')``
          equals ``'escape'`` → ``handles.output = 'No'``, ``uiresume``.
        * **Return / Enter** → accept (keep current output, default ``'Yes'``).
          Ref: ARMAX_close_dialog.m:205-206 — ``'return'`` → ``uiresume``.
        * All other keys → delegate to the base-class handler.
        """
        key = event.key()

        if key == Qt.Key.Key_Escape:
            # Ref: ARMAX_close_dialog.m:195-202 — Escape → output='No'
            self._output = "No"
            self.reject()
            return

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Ref: ARMAX_close_dialog.m:205-206 — Return → uiresume
            self.accept()
            return

        # Delegate all other keys to the default handler.
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_question_icon(self) -> None:
        """Load the Qt built-in question icon into *axes1*.

        Replaces the MATLAB code that loaded ``dialogicons.mat`` and
        displayed ``questIconData`` in an axes widget with a reversed Y
        direction and a custom colourmap.

        Ref: ARMAX_close_dialog.m:106-120
            ``load dialogicons.mat``
            ``IconData = questIconData;``
            ``image(IconData, 'Parent', handles.axes1);``
        """
        style = self.style()
        if style is not None:
            # Retrieve the platform question-mark icon at a suitable size.
            icon = style.standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxQuestion
            )
            # Render the icon to a 48×48 pixmap (matching the .ui label size).
            pixmap: QPixmap = icon.pixmap(48, 48)
            self.axes1.setPixmap(pixmap)

    def _center_on_parent_or_screen(self) -> None:
        """Centre the dialog over its parent widget or the primary screen.

        Ref: ARMAX_close_dialog.m:78-102
            If ``gcbf`` (calling figure) exists → centre over that figure.
            Otherwise → centre on screen.
        """
        parent = self.parentWidget()
        if parent is not None:
            # Ref: ARMAX_close_dialog.m:93-98 — centre over calling figure
            parent_geometry = parent.frameGeometry()
            center = parent_geometry.center()
            own_geometry = self.frameGeometry()
            own_geometry.moveCenter(center)
            self.move(own_geometry.topLeft())
        else:
            # Ref: ARMAX_close_dialog.m:85-91 — centre on screen
            screen = self.screen()
            if screen is not None:
                screen_geometry = screen.availableGeometry()
                center = screen_geometry.center()
                own_geometry = self.frameGeometry()
                own_geometry.moveCenter(center)
                self.move(own_geometry.topLeft())

    # ------------------------------------------------------------------
    # Private signal-slot callbacks
    # ------------------------------------------------------------------

    def _on_yes_clicked(self) -> None:
        """Slot connected to *pushbutton1* ("Yes") ``clicked`` signal.

        Ref: ARMAX_close_dialog.m:141-154 — ``pushbutton1_Callback``
            ``handles.output = get(hObject,'String');``   → 'Yes'
            ``uiresume(handles.figure1);``                → self.accept()
        """
        # Ref: ARMAX_close_dialog.m:147 — handles.output = get(hObject,'String')
        self._output = "Yes"
        self.accept()

    def _on_no_clicked(self) -> None:
        """Slot connected to *pushbutton2* ("No") ``clicked`` signal.

        Ref: ARMAX_close_dialog.m:156-169 — ``pushbutton2_Callback``
            ``handles.output = get(hObject,'String');``   → 'No'
            ``uiresume(handles.figure1);``                → self.reject()
        """
        # Ref: ARMAX_close_dialog.m:162 — handles.output = get(hObject,'String')
        self._output = "No"
        self.reject()
