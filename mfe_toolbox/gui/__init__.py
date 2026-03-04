"""MFE Toolbox ARMAX Estimation GUI — PyQt6 application for ARMAX model
estimation and visualization.

This package provides the PyQt6-based graphical user interface for ARMAX
model estimation, replacing the original MATLAB GUIDE application
(``GUI/ARMAX.m``, ``GUI/ARMAX.fig``, and companion dialogs).

The primary entry point is :func:`main`, which is the target for both:

- ``python -m mfe_toolbox.gui`` invocation
- ``mfe-toolbox-gui`` CLI command declared in ``pyproject.toml``

Ref: AAP Section 0.4.1 — ``mfe_toolbox/gui/__init__.py`` is "NEW — main() entry point".
Ref: AAP Section 0.6.3 — ``[project.scripts] mfe-toolbox-gui = "mfe_toolbox.gui:main"``.

Public API
----------
ARMAXMainWindow
    Main ARMAX estimation window (QMainWindow).
ARMAXViewer
    ARMAX result display/plotting viewer (QWidget).
ARMAXAboutDialog
    About/credits dialog with Oxford logo (QDialog).
ARMAXCloseDialog
    Save-before-close confirmation dialog (QDialog).
main
    Launch the ARMAX GUI application.

All class imports are *lazy* — PyQt6 is not imported until one of the
exported names is actually accessed.  This avoids a heavy import penalty
when importing the ``mfe_toolbox`` package in headless or non-GUI contexts.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Lazy-import support via module-level __getattr__ (PEP 562).
# Ref: AAP instructions — lazy import of ARMAXMainWindow, ARMAXViewer,
# ARMAXAboutDialog, ARMAXCloseDialog from their respective submodules
# inside main() and via __getattr__ to avoid circular imports and the
# heavy PyQt6 import penalty at package load time.
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "ARMAXMainWindow",
    "ARMAXViewer",
    "ARMAXAboutDialog",
    "ARMAXCloseDialog",
    "main",
]

# Mapping from exported name to (relative module path, class name).
# Used by __getattr__ below to resolve lazy imports on first access.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ARMAXMainWindow": ("mfe_toolbox.gui.armax_main", "ARMAXMainWindow"),
    "ARMAXViewer": ("mfe_toolbox.gui.armax_viewer", "ARMAXViewer"),
    "ARMAXAboutDialog": ("mfe_toolbox.gui.armax_about", "ARMAXAboutDialog"),
    "ARMAXCloseDialog": ("mfe_toolbox.gui.armax_close_dialog", "ARMAXCloseDialog"),
}

if TYPE_CHECKING:
    # Provide static type-checker visibility without runtime import.
    from mfe_toolbox.gui.armax_about import ARMAXAboutDialog as ARMAXAboutDialog
    from mfe_toolbox.gui.armax_close_dialog import ARMAXCloseDialog as ARMAXCloseDialog
    from mfe_toolbox.gui.armax_main import ARMAXMainWindow as ARMAXMainWindow
    from mfe_toolbox.gui.armax_viewer import ARMAXViewer as ARMAXViewer


def __getattr__(name: str) -> object:
    """Lazy-import handler for heavy GUI class exports.

    When any of the four public class names (``ARMAXMainWindow``,
    ``ARMAXViewer``, ``ARMAXAboutDialog``, ``ARMAXCloseDialog``) is first
    accessed via attribute lookup on this module, the corresponding
    submodule is imported and the class is returned.  This avoids
    importing PyQt6 at package load time.

    Parameters
    ----------
    name : str
        The attribute name being looked up.

    Returns
    -------
    object
        The resolved class.

    Raises
    ------
    AttributeError
        If *name* is not a recognised lazy-import target.
    """
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        # Perform the actual import on first access.
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        # Cache on the module so __getattr__ is not called again.
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# main() — GUI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the ARMAX Estimation GUI application.

    This function serves as the console-script entry point declared in
    ``pyproject.toml`` (``mfe-toolbox-gui = "mfe_toolbox.gui:main"``).
    It can also be invoked via ``python -m mfe_toolbox.gui``.

    Command-line usage::

        # Launch with sample data (500 random observations)
        mfe-toolbox-gui

        # Launch with data from a NumPy .npy file
        mfe-toolbox-gui path/to/data.npy

        # Launch with data from a CSV file (single column)
        mfe-toolbox-gui path/to/data.csv

    The function:

    1. Creates a :class:`PyQt6.QtWidgets.QApplication` singleton,
       forwarding ``sys.argv`` for Qt command-line argument handling.
    2. Determines the data source:
       - If a file path is provided as the first CLI argument
         (``sys.argv[1]``), loads data from that file (``.npy`` via
         :func:`numpy.load`, ``.csv`` via :func:`numpy.loadtxt`).
       - Otherwise generates 500 standard-normal observations using
         :func:`numpy.random.default_rng` with seed 42 for
         reproducibility.
    3. Instantiates :class:`ARMAXMainWindow` with the data array.
    4. Shows the window and enters the Qt event loop.
    5. Exits cleanly via :func:`sys.exit`.

    Ref: GUI/ARMAX.m:1-43 — GUIDE initialization and gui_mainfcn entry.
    Ref: GUI/ARMAX.m:62 — ``handles.y = varargin{1}`` (data argument).
    Ref: AAP Section 0.7.1 — "PyQt6 application MUST launch without error
    via ``python -m mfe_toolbox.gui``".
    """
    # --- Lazy imports inside main() to avoid package-level PyQt6 cost ---
    import numpy as np  # noqa: F811 — intentional lazy import
    from PyQt6.QtWidgets import QApplication

    from mfe_toolbox.gui.armax_main import ARMAXMainWindow

    # --- Create QApplication singleton ---
    # Ref: Standard PyQt6 pattern — QApplication must be created before any
    # QWidget.  sys.argv is forwarded so Qt can consume its own CLI flags
    # (e.g. -style, -platform).
    app = QApplication(sys.argv)

    # --- Determine data source ---
    # Ref: ARMAX.m:62 — the MATLAB GUI requires data y as its first argument.
    # In the Python CLI, an optional file path may be supplied via sys.argv.
    y: np.ndarray

    # sys.argv[0] is the script/entry-point name; sys.argv[1] (if present)
    # is the user-supplied data file path.
    if len(sys.argv) > 1:
        data_path: str = sys.argv[1]
        if data_path.lower().endswith(".npy"):
            # NumPy binary format
            y = np.load(data_path)
        elif data_path.lower().endswith(".csv"):
            # Comma-separated values — single column or first column used
            y = np.loadtxt(data_path, delimiter=",")
        else:
            # Attempt generic numpy load for other extensions
            try:
                y = np.load(data_path)
            except Exception:
                y = np.loadtxt(data_path, delimiter=",")
    else:
        # Ref: ARMAX.m:62 — generate sample data for demonstration when no
        # file is provided.  Uses a fixed seed for reproducibility.
        rng = np.random.default_rng(42)
        y = rng.standard_normal(500)

    # Ensure 1-D float64 array (matching MATLAB column vector convention).
    y = np.asarray(y, dtype=np.float64).ravel()

    # --- Create and show main window ---
    # Ref: ARMAX.m:48-88 — ARMAX_OpeningFcn creates the window with data.
    window = ARMAXMainWindow(y)
    window.show()

    # --- Enter Qt event loop ---
    # Ref: Standard PyQt6 pattern — app.exec() blocks until the last window
    # is closed; sys.exit propagates the return code to the OS.
    sys.exit(app.exec())


# ---------------------------------------------------------------------------
# __main__ support — enables ``python -m mfe_toolbox.gui``
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
