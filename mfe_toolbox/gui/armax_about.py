"""
ARMAX About/Credits Dialog — PyQt6 QDialog with Oxford Logo display.

Migrated from GUI/ARMAX_about.m (206 lines) + GUI/ARMAX_about.fig.
This module provides the ARMAXAboutDialog class, a modal dialog that
displays the Oxford University logo and About information for the
MFE Toolbox ARMAX estimation GUI.

Ref: GUI/ARMAX_about.m — complete GUIDE About dialog replacement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Ref: ARMAX_about.m:55 — default output value
_DEFAULT_OUTPUT: str = "Yes"

# Ref: ARMAX_about.m:69 — default window title when no title argument supplied
_DEFAULT_TITLE: str = "About"

# Ref: ARMAX_about.m:71 — default message when no string argument supplied
_DEFAULT_MESSAGE: str = (
    "MFE Toolbox — ARMAX Estimation GUI\n\n"
    "Version 4.0\n"
    "Kevin Sheppard\n"
    "University of Oxford\n\n"
    "Python migration of the original MATLAB MFE Toolbox.\n"
    "© Oxford-Man Institute of Quantitative Finance"
)

# Resolve asset paths relative to this module file.
# Ref: ARMAX_about.m:106 — load OxLogo.mat path resolution
_MODULE_DIR: Path = Path(__file__).parent
_UI_FILE: Path = _MODULE_DIR / "armax_about.ui"
_LOGO_PATH: Path = _MODULE_DIR / "assets" / "ox_logo.png"


class ARMAXAboutDialog(QDialog):
    """About/credits dialog with Oxford logo.

    Provides a modal dialog that displays the Oxford University logo
    and About/credits information for the MFE Toolbox ARMAX application.

    The dialog supports optional ``title`` and ``message`` overrides
    mirroring the MATLAB property-value pair pattern:
    ``ARMAX_about('Title','About','String','Custom text')``.

    Ref: GUI/ARMAX_about.m — GUIDE About dialog replacement.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget for centering. ``None`` centres on screen.
        Ref: ARMAX_about.m:78-102 — centering logic.
    title : str, optional
        Window title. Defaults to ``"About"``.
        Ref: ARMAX_about.m:69 — ``set(hObject, 'Name', varargin{index+1})``.
    message : str, optional
        Body text displayed below the logo.
        Ref: ARMAX_about.m:71 — ``set(handles.text1, 'String', ...)``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = _DEFAULT_TITLE,
        message: str = _DEFAULT_MESSAGE,
    ) -> None:
        super().__init__(parent)

        # Ref: ARMAX_about.m:55 — handles.output = 'Yes'
        self._output: str = _DEFAULT_OUTPUT

        # Try to load .ui layout; fall back to programmatic build.
        # Ref: ARMAX_about.m:28-34 — GUIDE gui_State initialization
        ui_loaded = self._try_load_ui()
        if not ui_loaded:
            self._build_ui()

        # --- Apply title and message ---
        # Ref: ARMAX_about.m:64-74 — switch on 'title'/'string' varargin
        self.setWindowTitle(title)  # Ref: line 69
        self._set_message(message)  # Ref: line 71

        # --- Load and display Oxford logo ---
        # Ref: ARMAX_about.m:106-120 — load OxLogo.mat, image(IconData, ...)
        self._load_logo()

        # --- Modal window ---
        # Ref: ARMAX_about.m:123 — set(handles.figure1,'WindowStyle','modal')
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # --- Connect signals/slots ---
        # Ref: ARMAX_about.m:141-169 — pushbutton1_Callback, pushbutton2_Callback
        self._connect_signals()

        # --- Centre dialog over parent or screen ---
        # Ref: ARMAX_about.m:78-102 — dialog positioning logic
        self._center_on_parent_or_screen()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def output(self) -> str:
        """Return the dialog output string.

        After the dialog closes, this property holds:
        - The text of the clicked button (e.g. ``"OK"``), or
        - ``"Yes"`` (default), or
        - ``"No"`` if the user pressed Escape.

        Ref: ARMAX_about.m:136 — ``varargout{1} = handles.output``
        """
        return self._output

    @staticmethod
    def show_about(
        parent: Optional[QWidget] = None,
        title: str = _DEFAULT_TITLE,
        message: str = _DEFAULT_MESSAGE,
    ) -> str:
        """Show the About dialog modally and return the output string.

        This convenience method mirrors the MATLAB call pattern::

            ARMAX_about('Title', 'About')

        Parameters
        ----------
        parent : QWidget or None, optional
            Parent widget.
        title : str, optional
            Dialog window title.
        message : str, optional
            Body text.

        Returns
        -------
        str
            The dialog output string (button text or ``'No'``).

        Ref: ARMAX_about.m — invocation via ``ARMAX_about('Title','About')``
        """
        dialog = ARMAXAboutDialog(parent=parent, title=title, message=message)
        # Ref: ARMAX_about.m:126 — uiwait(handles.figure1)
        dialog.exec()
        return dialog.output

    # ------------------------------------------------------------------
    # accept / reject overrides (schema required)
    # ------------------------------------------------------------------
    def accept(self) -> None:
        """Accept the dialog (OK / Enter / button click).

        Ref: ARMAX_about.m:154 — uiresume(handles.figure1)
        """
        super().accept()

    def reject(self) -> None:
        """Reject the dialog (Escape key).

        Sets output to ``'No'`` before closing.
        Ref: ARMAX_about.m:194-201 — Escape → output='No', uiresume
        """
        # Ref: ARMAX_about.m:196 — handles.output = 'No'
        self._output = "No"
        super().reject()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def closeEvent(self, event: "QKeyEvent") -> None:  # type: ignore[override]
        """Handle the window close (X button) event.

        Ref: ARMAX_about.m:172-184 — figure1_CloseRequestFcn
        The MATLAB version calls uiresume; we call accept() so the
        event loop exits cleanly.
        """
        # Ref: ARMAX_about.m:180 — uiresume(handles.figure1)
        self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key-press events for Escape and Return/Enter.

        Ref: ARMAX_about.m:188-206 — figure1_KeyPressFcn

        - **Escape**: Sets output to ``'No'`` and rejects the dialog.
          Ref: ARMAX_about.m:194-201
        - **Return / Enter**: Accepts the dialog with the current output.
          Ref: ARMAX_about.m:204-206
        """
        key = event.key()

        # Ref: ARMAX_about.m:194 — isequal(get(hObject,'CurrentKey'),'escape')
        if key == Qt.Key.Key_Escape:
            # Ref: ARMAX_about.m:196 — handles.output = 'No'
            self._output = "No"
            self.reject()
            return

        # Ref: ARMAX_about.m:204 — isequal(get(hObject,'CurrentKey'),'return')
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
            return

        # Propagate all other key events to the base class.
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Private helpers — UI construction
    # ------------------------------------------------------------------
    def _try_load_ui(self) -> bool:
        """Attempt to load the ``armax_about.ui`` Qt Designer file.

        Returns ``True`` on success so the caller can skip the
        programmatic fallback.
        """
        if not _UI_FILE.is_file():
            return False
        try:
            from PyQt6 import uic  # Deferred import — only needed here

            uic.loadUi(str(_UI_FILE), self)
            return True
        except Exception:
            logger.debug("Failed to load %s; building UI programmatically.", _UI_FILE)
            return False

    def _build_ui(self) -> None:
        """Build the dialog UI programmatically.

        This fallback creates the same widget structure expected by
        ``armax_about.ui`` so the rest of the class works identically
        regardless of whether the ``.ui`` file was loaded.

        Widget objectNames mirror the MATLAB GUIDE names:
        - ``icon_label`` — QLabel replacing MATLAB axes1 (logo display)
        - ``text1`` — QLabel for About body text
        - ``pushbutton1`` — QPushButton ("OK")
        - ``pushbutton2`` — QPushButton ("Close")
        """
        self.setObjectName("ARMAXAboutDialog")
        self.setMinimumSize(360, 300)
        self.resize(380, 340)

        # --- Main vertical layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # --- Logo label (replacing MATLAB axes1) ---
        # Ref: ARMAX_about.m:108-120 — image display in axes1
        self.icon_label = QLabel(self)
        self.icon_label.setObjectName("icon_label")
        self.icon_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self.icon_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.icon_label.setMinimumHeight(80)
        main_layout.addWidget(self.icon_label)

        # --- Message label (text1) ---
        # Ref: ARMAX_about.m:71 — set(handles.text1, 'String', ...)
        self.text1 = QLabel(self)
        self.text1.setObjectName("text1")
        self.text1.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self.text1.setWordWrap(True)
        self.text1.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(self.text1)

        # --- Button row ---
        # Ref: ARMAX_about.m:141-169 — pushbutton1 and pushbutton2 callbacks
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        button_layout.addStretch(1)

        self.pushbutton1 = QPushButton("OK", self)
        self.pushbutton1.setObjectName("pushbutton1")
        self.pushbutton1.setMinimumWidth(80)
        self.pushbutton1.setDefault(True)
        button_layout.addWidget(self.pushbutton1)

        self.pushbutton2 = QPushButton("Close", self)
        self.pushbutton2.setObjectName("pushbutton2")
        self.pushbutton2.setMinimumWidth(80)
        button_layout.addWidget(self.pushbutton2)

        button_layout.addStretch(1)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    # ------------------------------------------------------------------
    # Private helpers — widget population
    # ------------------------------------------------------------------
    def _set_message(self, message: str) -> None:
        """Set the body text in the ``text1`` label.

        Ref: ARMAX_about.m:71 — ``set(handles.text1, 'String', varargin{index+1})``
        """
        text_label: Optional[QLabel] = self.findChild(QLabel, "text1")
        if text_label is not None:
            text_label.setText(message)

    def _load_logo(self) -> None:
        """Load and display the Oxford logo in ``icon_label``.

        Path resolution: ``Path(__file__).parent / 'assets' / 'ox_logo.png'``

        If the logo file is not found the label shows a placeholder text
        string so the dialog remains usable.

        Ref: ARMAX_about.m:106-120 — ``load OxLogo.mat``, ``image(IconData, ...)``
        """
        icon_label: Optional[QLabel] = self.findChild(QLabel, "icon_label")
        if icon_label is None:
            # If the .ui file used a different objectName, try axes1 as well
            icon_label = self.findChild(QLabel, "axes1")
        if icon_label is None:
            logger.debug("No icon_label or axes1 QLabel found in dialog.")
            return

        if _LOGO_PATH.is_file():
            # Ref: ARMAX_about.m:108 — IconData=OxLogo
            pixmap = QPixmap(str(_LOGO_PATH))
            if not pixmap.isNull():
                # Scale the pixmap to a reasonable size while keeping aspect ratio.
                # Ref: ARMAX_about.m:115-119 — axes limits match image size
                scaled = pixmap.scaledToHeight(
                    min(pixmap.height(), 120),
                    Qt.TransformationMode.SmoothTransformation,
                )
                icon_label.setPixmap(scaled)
                icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                icon_label.setText("[Oxford Logo]")
                logger.debug("QPixmap load returned null for %s", _LOGO_PATH)
        else:
            # Graceful degradation — show placeholder text if asset is missing.
            # Ref: ARMAX_about.m:106 — load OxLogo.mat could fail
            icon_label.setText("[Oxford Logo — asset not found]")
            logger.debug("Logo file not found: %s", _LOGO_PATH)

    # ------------------------------------------------------------------
    # Private helpers — signal/slot wiring
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        """Wire pushbutton clicked signals to the handler.

        Ref: ARMAX_about.m:141-154 — pushbutton1_Callback
        Ref: ARMAX_about.m:156-169 — pushbutton2_Callback
        """
        btn1: Optional[QPushButton] = self.findChild(QPushButton, "pushbutton1")
        btn2: Optional[QPushButton] = self.findChild(QPushButton, "pushbutton2")

        if btn1 is not None:
            btn1.clicked.connect(self._on_button_clicked)
        if btn2 is not None:
            btn2.clicked.connect(self._on_button_clicked)

    def _on_button_clicked(self) -> None:
        """Slot for pushbutton clicks.

        Reads the sender button's text and stores it as the dialog output,
        then accepts the dialog.

        Ref: ARMAX_about.m:147 — ``handles.output = get(hObject,'String')``
        Ref: ARMAX_about.m:162 — same pattern for pushbutton2
        Ref: ARMAX_about.m:154,169 — ``uiresume(handles.figure1)``
        """
        sender = self.sender()
        if isinstance(sender, QPushButton):
            self._output = sender.text()
        self.accept()

    # ------------------------------------------------------------------
    # Private helpers — positioning
    # ------------------------------------------------------------------
    def _center_on_parent_or_screen(self) -> None:
        """Centre the dialog over its parent widget (or the screen).

        Ref: ARMAX_about.m:78-102 — dialog centering over gcbf or screen.
        """
        parent = self.parentWidget()
        if parent is not None:
            # Ref: ARMAX_about.m:93-98 — centre over calling figure
            parent_geom = parent.geometry()
            self_geom = self.geometry()
            x = parent_geom.x() + (parent_geom.width() - self_geom.width()) // 2
            y = parent_geom.y() + (parent_geom.height() - self_geom.height()) // 2
            self.move(x, y)
        else:
            # Ref: ARMAX_about.m:84-91 — centre on screen
            screen = self.screen()
            if screen is not None:
                screen_geom = screen.availableGeometry()
                self_geom = self.geometry()
                x = (screen_geom.width() - self_geom.width()) // 2
                y = (screen_geom.height() - self_geom.height()) * 2 // 3
                self.move(x + screen_geom.x(), y + screen_geom.y())
