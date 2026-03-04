"""
ARMAX Main Estimation Window — PyQt6 QMainWindow.

Migrated from GUI/ARMAX.m (1282 lines) + GUI/ARMAX.fig (MATLAB GUIDE).
Provides the main ARMAX estimation GUI with:
  - Model specification inputs (AR lags, MA lags, exogenous variables)
  - Constant inclusion toggle
  - Estimation, reset, and close actions
  - Plot panel for data visualization (original data, ACF, PACF, residuals)
  - Model selector for comparing multiple estimated models
  - Inference method selection (heteroskedastic-robust / homoskedastic)
  - Export and help menus

Ref: GUI/ARMAX.m — complete GUIDE main window replacement.
Ref: GUI/ARMAX.fig — geometry reconstructed into armax_main.ui.

Signal/slot wiring follows the MATLAB callback pattern:
  - estimate_pushbutton_Callback → _on_estimate_clicked
  - reset_pushbutton_Callback → _on_reset_clicked
  - close_pushbutton_Callback → _on_close_clicked
  - orig_raw_radio_Callback → _on_orig_plot_changed
  - resid_raw_radio_Callback → _on_resid_plot_changed
  - model_selector_popup_Callback → _on_model_selected
  - last_model_pushbutton_Callback → _on_last_model_clicked
  - export_menu callbacks → _on_export_*
  - help/about menu → _on_about_clicked
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Ref: ARMAX.m:68 — default ACF lag count
_DEFAULT_ACF_LAGS_FACTOR: int = 8
_DEFAULT_ACF_LAGS_MAX: int = 24

# Ref: ARMAX.m:69 — inference method codes
_INFERENCE_HETEROROBUST: int = 1
_INFERENCE_HOMOSKEDASTIC: int = 0

# UI file path (Qt Designer XML)
_MODULE_DIR: Path = Path(__file__).parent
_UI_FILE: Path = _MODULE_DIR / "armax_main.ui"


class ARMAXMainWindow(QMainWindow):
    """Main ARMAX estimation window.

    Provides the complete GUI front-end for ARMAX model estimation,
    including model specification controls, plot visualization, model
    comparison, and result export.

    Ref: GUI/ARMAX.m — 1282-line GUIDE main window.

    Parameters
    ----------
    y : numpy.ndarray or None, optional
        Initial time series data. If provided, the data is plotted
        immediately on opening. Ref: ARMAX.m:65 — ``handles.y = varargin{1}``
    parent : QWidget or None, optional
        Parent widget. ``None`` for standalone window.
    """

    # Custom signal emitted when estimation completes successfully
    estimation_complete = pyqtSignal(dict)

    def __init__(
        self,
        y: Optional[np.ndarray] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # --- Internal state (mirrors MATLAB handles structure) ---
        # Ref: ARMAX.m:60-72 — handles initialization
        self._y: Optional[np.ndarray] = None
        self._models: list[dict[str, Any]] = []
        self._current_model_index: int = -1
        self._ar_order: list[int] = []
        self._ma_order: list[int] = []
        self._ar_order_string: str = ""
        self._ma_order_string: str = ""
        self._residuals: Optional[np.ndarray] = None
        self._acf_lags: int = _DEFAULT_ACF_LAGS_MAX
        self._inference_method: int = _INFERENCE_HETEROROBUST
        self._include_constant: bool = True

        # --- Build UI ---
        self._build_ui()
        self._create_canvas()
        self._connect_signals()

        # --- Load data if provided ---
        if y is not None:
            self.set_data(y)

        logger.debug("ARMAXMainWindow initialized.")

    # ===================================================================
    # UI Construction
    # ===================================================================

    def _build_ui(self) -> None:
        """Construct the main window layout programmatically.

        Mirrors the layout defined in armax_main.ui. Widgets are created
        with objectNames matching the .ui file for consistency.
        Ref: GUI/ARMAX.fig — geometry and layout.
        """
        self.setWindowTitle("ARMAX Estimation")
        self.setMinimumSize(900, 700)
        self.resize(900, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Top area: left panel + right panel
        top_layout = QHBoxLayout()
        top_layout.setSpacing(6)
        main_layout.addLayout(top_layout, stretch=1)

        # --- LEFT PANEL ---
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)
        top_layout.addLayout(left_layout, stretch=0)

        # Model Specification Group
        model_spec_group = QGroupBox("Model Specification")
        model_spec_group.setObjectName("modelSpecGroupBox")
        model_spec_layout = QVBoxLayout(model_spec_group)
        model_spec_layout.setSpacing(4)

        # AR lags
        ar_layout = QHBoxLayout()
        ar_layout.addWidget(QLabel("AR Lags:"))
        self.AR_lags_edit = QLineEdit()
        self.AR_lags_edit.setObjectName("AR_lags_edit")
        self.AR_lags_edit.setPlaceholderText("e.g., 1 2 3")
        ar_layout.addWidget(self.AR_lags_edit)
        model_spec_layout.addLayout(ar_layout)

        # MA lags
        ma_layout = QHBoxLayout()
        ma_layout.addWidget(QLabel("MA Lags:"))
        self.MA_lags_edit = QLineEdit()
        self.MA_lags_edit.setObjectName("MA_lags_edit")
        self.MA_lags_edit.setPlaceholderText("e.g., 1 2")
        ma_layout.addWidget(self.MA_lags_edit)
        model_spec_layout.addLayout(ma_layout)

        # Exogenous variables
        exog_layout = QHBoxLayout()
        exog_layout.addWidget(QLabel("Exogenous:"))
        self.EXOG_edit = QLineEdit()
        self.EXOG_edit.setObjectName("EXOG_edit")
        exog_layout.addWidget(self.EXOG_edit)
        self.EXOG_clear_pushbutton = QPushButton("Clear")
        self.EXOG_clear_pushbutton.setObjectName("EXOG_clear_pushbutton")
        exog_layout.addWidget(self.EXOG_clear_pushbutton)
        model_spec_layout.addLayout(exog_layout)

        # Constant toggle — Ref: ARMAX.m:70 — IncludeConstant radio buttons
        const_layout = QHBoxLayout()
        self.constant_include = QRadioButton("Include Constant")
        self.constant_include.setObjectName("constant_include")
        self.constant_include.setChecked(True)
        self.constant_exclude = QRadioButton("Exclude Constant")
        self.constant_exclude.setObjectName("constant_exclude")
        const_layout.addWidget(self.constant_include)
        const_layout.addWidget(self.constant_exclude)
        model_spec_layout.addLayout(const_layout)

        # ACF lags
        acf_layout = QHBoxLayout()
        acf_layout.addWidget(QLabel("ACF Lags:"))
        self.ACF_lags_edit = QLineEdit()
        self.ACF_lags_edit.setObjectName("ACF_lags_edit")
        self.ACF_lags_edit.setText(str(self._acf_lags))
        acf_layout.addWidget(self.ACF_lags_edit)
        model_spec_layout.addLayout(acf_layout)

        # Hold-back
        holdback_layout = QHBoxLayout()
        holdback_layout.addWidget(QLabel("Hold Back:"))
        self.hold_back_edit = QLineEdit()
        self.hold_back_edit.setObjectName("hold_back_edit")
        self.hold_back_edit.setText("0")
        holdback_layout.addWidget(self.hold_back_edit)
        model_spec_layout.addLayout(holdback_layout)

        left_layout.addWidget(model_spec_group)

        # Actions Group — Ref: ARMAX.m:101-103 button callbacks
        actions_group = QGroupBox("Actions")
        actions_group.setObjectName("actionsGroupBox")
        actions_layout = QVBoxLayout(actions_group)

        self.estimate_pushbutton = QPushButton("Estimate")
        self.estimate_pushbutton.setObjectName("estimate_pushbutton")
        actions_layout.addWidget(self.estimate_pushbutton)

        self.reset_pushbutton = QPushButton("Reset")
        self.reset_pushbutton.setObjectName("reset_pushbutton")
        actions_layout.addWidget(self.reset_pushbutton)

        self.close_pushbutton = QPushButton("Close")
        self.close_pushbutton.setObjectName("close_pushbutton")
        actions_layout.addWidget(self.close_pushbutton)

        left_layout.addWidget(actions_group)

        # Model selector — Ref: ARMAX.m:425-455 model_selector_popup_Callback
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Model:"))
        self.model_selector_popup = QComboBox()
        self.model_selector_popup.setObjectName("model_selector_popup")
        selector_layout.addWidget(self.model_selector_popup)
        left_layout.addLayout(selector_layout)

        self.last_model_pushbutton = QPushButton("Show Last Model")
        self.last_model_pushbutton.setObjectName("last_model_pushbutton")
        left_layout.addWidget(self.last_model_pushbutton)

        # Message text — Ref: ARMAX.m:97 status messages
        self.message_text = QLabel("")
        self.message_text.setObjectName("message_text")
        self.message_text.setWordWrap(True)
        left_layout.addWidget(self.message_text)

        left_layout.addStretch(1)

        # --- RIGHT PANEL: Plot container ---
        self.plot_container = QWidget()
        self.plot_container.setObjectName("plot_container")
        self.plot_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        top_layout.addWidget(self.plot_container, stretch=1)

        # --- BOTTOM: Plot type selectors ---
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout, stretch=0)

        # Original Data plot options — Ref: ARMAX.m:493-559 radio callbacks
        orig_group = QGroupBox("Original Data")
        orig_group.setObjectName("originalDataGroupBox")
        orig_layout = QHBoxLayout(orig_group)
        self.orig_raw_radio = QRadioButton("Raw")
        self.orig_raw_radio.setObjectName("orig_raw_radio")
        self.orig_raw_radio.setChecked(True)
        self.orig_acf_radio = QRadioButton("ACF")
        self.orig_acf_radio.setObjectName("orig_acf_radio")
        self.orig_pacf_radio = QRadioButton("PACF")
        self.orig_pacf_radio.setObjectName("orig_pacf_radio")
        orig_layout.addWidget(self.orig_raw_radio)
        orig_layout.addWidget(self.orig_acf_radio)
        orig_layout.addWidget(self.orig_pacf_radio)
        bottom_layout.addWidget(orig_group)

        # Residuals plot options — Ref: ARMAX.m:561-647 residual radio callbacks
        resid_group = QGroupBox("Residuals")
        resid_group.setObjectName("residualsGroupBox")
        resid_layout = QHBoxLayout(resid_group)
        self.resid_raw_radio = QRadioButton("Raw")
        self.resid_raw_radio.setObjectName("resid_raw_radio")
        self.resid_fit_radio = QRadioButton("Fitted")
        self.resid_fit_radio.setObjectName("resid_fit_radio")
        self.resid_acf_radio = QRadioButton("ACF")
        self.resid_acf_radio.setObjectName("resid_acf_radio")
        self.resid_pacf_radio = QRadioButton("PACF")
        self.resid_pacf_radio.setObjectName("resid_pacf_radio")
        resid_layout.addWidget(self.resid_raw_radio)
        resid_layout.addWidget(self.resid_fit_radio)
        resid_layout.addWidget(self.resid_acf_radio)
        resid_layout.addWidget(self.resid_pacf_radio)
        bottom_layout.addWidget(resid_group)

        # Tests group — Ref: ARMAX.m:649-700 test radio buttons
        tests_group = QGroupBox("Tests")
        tests_group.setObjectName("testsGroupBox")
        tests_layout = QHBoxLayout(tests_group)
        self.radiobutton18 = QRadioButton("LB Q-Stat")
        self.radiobutton18.setObjectName("radiobutton18")
        self.radiobutton19 = QRadioButton("LM ARCH")
        self.radiobutton19.setObjectName("radiobutton19")
        tests_layout.addWidget(self.radiobutton18)
        tests_layout.addWidget(self.radiobutton19)
        bottom_layout.addWidget(tests_group)

        # --- Menu bar ---
        self._build_menu_bar()

    def _build_menu_bar(self) -> None:
        """Build the menu bar with Export and Help menus.

        Ref: ARMAX.m:702-739 — export and help menu callbacks.
        """
        menubar = self.menuBar()

        # Export menu
        self.export_menu = QMenu("Export", self)
        self.export_menu.setObjectName("export_menu")
        self.action_export_workspace = self.export_menu.addAction("To Workspace")
        self.action_export_workspace.setObjectName("action_export_workspace")
        self.action_export_csv = self.export_menu.addAction("To CSV File")
        self.action_export_csv.setObjectName("action_export_csv")
        menubar.addMenu(self.export_menu)

        # Inference menu — Ref: ARMAX.m:720-733 inference method toggling
        self.inference_menu = QMenu("Inference", self)
        self.action_heterorobust = self.inference_menu.addAction("Heteroskedastic-Robust")
        self.action_heterorobust.setObjectName("action_heterorobust")
        self.action_heterorobust.setCheckable(True)
        self.action_heterorobust.setChecked(True)
        self.action_homoskedastic = self.inference_menu.addAction("Homoskedastic")
        self.action_homoskedastic.setObjectName("action_homoskedastic")
        self.action_homoskedastic.setCheckable(True)
        self.action_homoskedastic.setChecked(False)
        menubar.addMenu(self.inference_menu)

        # Help menu
        self.help_menu = QMenu("Help", self)
        self.action_about = self.help_menu.addAction("About...")
        self.action_about.setObjectName("action_about")
        menubar.addMenu(self.help_menu)

    def _create_canvas(self) -> None:
        """Create the matplotlib FigureCanvasQTAgg for data visualization.

        Ref: ARMAX.m:75-88 — initial plot of original data.
        """
        self._figure = Figure(figsize=(6, 4), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)

        # Embed canvas into the plot container
        layout = QVBoxLayout(self.plot_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def _connect_signals(self) -> None:
        """Wire button/menu signals to slot methods.

        Ref: ARMAX.m — each callback function maps to a slot here.
        """
        # Action buttons
        self.estimate_pushbutton.clicked.connect(self._on_estimate_clicked)
        self.reset_pushbutton.clicked.connect(self._on_reset_clicked)
        self.close_pushbutton.clicked.connect(self._on_close_clicked)
        self.EXOG_clear_pushbutton.clicked.connect(self._on_exog_clear_clicked)

        # Model selector
        self.model_selector_popup.currentIndexChanged.connect(self._on_model_selected)
        self.last_model_pushbutton.clicked.connect(self._on_last_model_clicked)

        # Plot type radios — original data
        self.orig_raw_radio.toggled.connect(self._on_orig_plot_changed)
        self.orig_acf_radio.toggled.connect(self._on_orig_plot_changed)
        self.orig_pacf_radio.toggled.connect(self._on_orig_plot_changed)

        # Plot type radios — residuals
        self.resid_raw_radio.toggled.connect(self._on_resid_plot_changed)
        self.resid_fit_radio.toggled.connect(self._on_resid_plot_changed)
        self.resid_acf_radio.toggled.connect(self._on_resid_plot_changed)
        self.resid_pacf_radio.toggled.connect(self._on_resid_plot_changed)

        # Tests radios
        self.radiobutton18.toggled.connect(self._on_test_plot_changed)
        self.radiobutton19.toggled.connect(self._on_test_plot_changed)

        # Constant toggle
        self.constant_include.toggled.connect(self._on_constant_toggled)

        # Menu actions
        self.action_export_workspace.triggered.connect(self._on_export_workspace)
        self.action_export_csv.triggered.connect(self._on_export_csv)
        self.action_heterorobust.triggered.connect(
            lambda: self._on_inference_changed(_INFERENCE_HETEROROBUST)
        )
        self.action_homoskedastic.triggered.connect(
            lambda: self._on_inference_changed(_INFERENCE_HOMOSKEDASTIC)
        )
        self.action_about.triggered.connect(self._on_about_clicked)

    # ===================================================================
    # Public API
    # ===================================================================

    def set_data(self, y: np.ndarray) -> None:
        """Load a time series dataset into the GUI.

        Ref: ARMAX.m:65 — ``handles.y = varargin{1}``

        Parameters
        ----------
        y : numpy.ndarray
            1-D time series data array.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        self._y = y
        self._acf_lags = min(_DEFAULT_ACF_LAGS_MAX, len(y) // _DEFAULT_ACF_LAGS_FACTOR)
        self.ACF_lags_edit.setText(str(self._acf_lags))
        self._plot_original_raw()
        self._set_message(f"Data loaded: {len(y)} observations.")
        logger.info("Data loaded: T=%d", len(y))

    # ===================================================================
    # Slot Methods — Button Callbacks
    # ===================================================================

    def _on_estimate_clicked(self) -> None:
        """Handle Estimate button click.

        Parses model specification inputs, runs ARMAX estimation, stores
        the result, and updates the plot and model selector.

        Ref: ARMAX.m:101-424 — estimate_pushbutton_Callback.
        """
        if self._y is None:
            self._set_message("Error: No data loaded.")
            return

        # Parse AR lags
        ar_text = self.AR_lags_edit.text().strip()
        ar_lags = self._parse_lag_string(ar_text)

        # Parse MA lags
        ma_text = self.MA_lags_edit.text().strip()
        ma_lags = self._parse_lag_string(ma_text)

        # Parse ACF lags
        try:
            acf_lags = int(self.ACF_lags_edit.text().strip())
            if acf_lags < 1:
                raise ValueError
        except (ValueError, TypeError):
            acf_lags = self._acf_lags

        # Parse hold-back
        try:
            hold_back = int(self.hold_back_edit.text().strip())
            if hold_back < 0:
                hold_back = 0
        except (ValueError, TypeError):
            hold_back = 0

        # Include constant
        include_constant = self.constant_include.isChecked()

        # Store current specification
        self._ar_order = ar_lags
        self._ma_order = ma_lags
        self._ar_order_string = ar_text
        self._ma_order_string = ma_text
        self._acf_lags = acf_lags
        self._include_constant = include_constant

        self._set_message("Estimating...")

        # Attempt estimation using armaxfilter
        # Ref: ARMAX.m:159-300 — estimation block
        try:
            from mfe_toolbox.timeseries.armaxfilter import armaxfilter

            # Determine max AR and MA order for armaxfilter call
            p = max(ar_lags) if ar_lags else 0
            q = max(ma_lags) if ma_lags else 0

            # Call armaxfilter
            # Ref: ARMAX.m:186 — armaxfilter(y, constant, p, q, ...)
            result = armaxfilter(
                self._y,
                int(include_constant),
                p,
                q,
            )

            # Package result
            model_result: dict[str, Any] = {
                "ar_order": ar_lags,
                "ma_order": ma_lags,
                "ar_order_string": ar_text,
                "ma_order_string": ma_text,
                "include_constant": include_constant,
                "acf_lags": acf_lags,
                "hold_back": hold_back,
                "result": result,
            }

            # Extract residuals from result
            # armaxfilter returns (parameters, LL, residuals, ...)
            if isinstance(result, tuple) and len(result) >= 3:
                self._residuals = np.asarray(result[2], dtype=np.float64).ravel()
                model_result["residuals"] = self._residuals
                model_result["parameters"] = result[0]
                model_result["LL"] = result[1]

            self._models.append(model_result)
            self._current_model_index = len(self._models) - 1

            # Update model selector combo
            label = f"Model {len(self._models)}: AR({ar_text}) MA({ma_text})"
            self.model_selector_popup.addItem(label)
            self.model_selector_popup.setCurrentIndex(self._current_model_index)

            self._set_message(f"Estimation complete. Model {len(self._models)} stored.")
            self.estimation_complete.emit(model_result)
            self._update_plot()

        except Exception as exc:
            self._set_message(f"Estimation failed: {exc}")
            logger.exception("ARMAX estimation failed")

    def _on_reset_clicked(self) -> None:
        """Handle Reset button — clear inputs and plots.

        Ref: ARMAX.m:456-490 — reset_pushbutton_Callback.
        """
        self.AR_lags_edit.clear()
        self.MA_lags_edit.clear()
        self.EXOG_edit.clear()
        self.hold_back_edit.setText("0")
        self.constant_include.setChecked(True)
        self._ar_order = []
        self._ma_order = []
        self._residuals = None
        self.orig_raw_radio.setChecked(True)

        if self._y is not None:
            self._plot_original_raw()
        else:
            self._clear_plot()

        self._set_message("Reset complete.")
        logger.debug("Reset clicked.")

    def _on_close_clicked(self) -> None:
        """Handle Close button — show save-before-close dialog.

        Ref: ARMAX.m:491-492 — close_pushbutton_Callback invokes
        ARMAX_close_dialog.
        """
        from mfe_toolbox.gui.armax_close_dialog import ARMAXCloseDialog

        result = ARMAXCloseDialog.ask(parent=self)
        if result == "Yes":
            self.close()
        elif result == "No":
            self.close()
        # "Cancel" — do nothing, stay open

    def _on_exog_clear_clicked(self) -> None:
        """Clear exogenous variable input.

        Ref: ARMAX.m:142 — EXOG_clear_pushbutton_Callback.
        """
        self.EXOG_edit.clear()

    def _on_model_selected(self, index: int) -> None:
        """Handle model selector combo box change.

        Ref: ARMAX.m:425-455 — model_selector_popup_Callback.
        """
        if 0 <= index < len(self._models):
            self._current_model_index = index
            model = self._models[index]
            self._residuals = model.get("residuals")
            self._update_plot()
            self._set_message(f"Displaying Model {index + 1}.")

    def _on_last_model_clicked(self) -> None:
        """Show the most recently estimated model.

        Ref: ARMAX.m:456 — last_model_pushbutton_Callback.
        """
        if self._models:
            last_idx = len(self._models) - 1
            self.model_selector_popup.setCurrentIndex(last_idx)

    # ===================================================================
    # Slot Methods — Plot Radio Callbacks
    # ===================================================================

    def _on_orig_plot_changed(self, checked: bool) -> None:
        """Handle original data plot type radio change.

        Ref: ARMAX.m:493-559 — orig_raw/acf/pacf radio callbacks.
        """
        if not checked:
            return
        self._update_plot()

    def _on_resid_plot_changed(self, checked: bool) -> None:
        """Handle residuals plot type radio change.

        Ref: ARMAX.m:561-647 — resid_raw/fit/acf/pacf radio callbacks.
        """
        if not checked:
            return
        self._update_plot()

    def _on_test_plot_changed(self, checked: bool) -> None:
        """Handle tests plot type radio change.

        Ref: ARMAX.m:649-700 — test radio callbacks.
        """
        if not checked:
            return
        self._update_plot()

    def _on_constant_toggled(self, checked: bool) -> None:
        """Handle constant include/exclude toggle.

        Ref: ARMAX.m:70 — IncludeConstant flag.
        """
        self._include_constant = checked

    # ===================================================================
    # Slot Methods — Menu Callbacks
    # ===================================================================

    def _on_inference_changed(self, method: int) -> None:
        """Handle inference method menu change.

        Ref: ARMAX.m:720-733 — inference method toggling.
        """
        self._inference_method = method
        self.action_heterorobust.setChecked(method == _INFERENCE_HETEROROBUST)
        self.action_homoskedastic.setChecked(method == _INFERENCE_HOMOSKEDASTIC)
        logger.debug("Inference method set to %d", method)

    def _on_export_workspace(self) -> None:
        """Export current model results (placeholder for workspace export).

        Ref: ARMAX.m:702-710 — export to workspace callback.
        In Python context, emits model data via signal or prints to console.
        """
        if self._current_model_index < 0 or not self._models:
            self._set_message("No model to export.")
            return
        model = self._models[self._current_model_index]
        # Print model summary to console as Python-side "workspace" export
        params = model.get("parameters")
        ll = model.get("LL")
        print(f"--- Exported Model {self._current_model_index + 1} ---")
        print(f"AR order: {model.get('ar_order_string', '')}")
        print(f"MA order: {model.get('ma_order_string', '')}")
        if params is not None:
            print(f"Parameters: {params}")
        if ll is not None:
            print(f"Log-likelihood: {ll}")
        self._set_message("Model exported to console.")

    def _on_export_csv(self) -> None:
        """Export current model results to CSV file.

        Ref: ARMAX.m:712-718 — export to file callback.
        """
        if self._current_model_index < 0 or not self._models:
            self._set_message("No model to export.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Model to CSV", "", "CSV Files (*.csv)"
        )
        if not filepath:
            return

        model = self._models[self._current_model_index]
        try:
            import csv

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Field", "Value"])
                writer.writerow(["AR_order", model.get("ar_order_string", "")])
                writer.writerow(["MA_order", model.get("ma_order_string", "")])
                writer.writerow(["Include_constant", model.get("include_constant", True)])
                params = model.get("parameters")
                if params is not None:
                    for i, p in enumerate(np.atleast_1d(params)):
                        writer.writerow([f"param_{i}", p])
                ll = model.get("LL")
                if ll is not None:
                    writer.writerow(["Log_likelihood", ll])
            self._set_message(f"Model exported to {filepath}.")
        except Exception as exc:
            self._set_message(f"Export failed: {exc}")
            logger.exception("CSV export failed")

    def _on_about_clicked(self) -> None:
        """Show the About dialog.

        Ref: ARMAX.m:735-739 — about callback.
        """
        from mfe_toolbox.gui.armax_about import ARMAXAboutDialog

        dlg = ARMAXAboutDialog(parent=self)
        dlg.exec()

    # ===================================================================
    # Plot Methods
    # ===================================================================

    def _update_plot(self) -> None:
        """Refresh the plot based on current radio button selection.

        Dispatches to the appropriate plot method based on which radio
        button is currently checked.
        """
        if self._y is None:
            self._clear_plot()
            return

        if self.orig_raw_radio.isChecked():
            self._plot_original_raw()
        elif self.orig_acf_radio.isChecked():
            self._plot_original_acf()
        elif self.orig_pacf_radio.isChecked():
            self._plot_original_pacf()
        elif self.resid_raw_radio.isChecked():
            self._plot_residuals_raw()
        elif self.resid_fit_radio.isChecked():
            self._plot_residuals_fitted()
        elif self.resid_acf_radio.isChecked():
            self._plot_residuals_acf()
        elif self.resid_pacf_radio.isChecked():
            self._plot_residuals_pacf()
        elif self.radiobutton18.isChecked():
            self._plot_lb_qstat()
        elif self.radiobutton19.isChecked():
            self._plot_lm_arch()
        else:
            self._plot_original_raw()

    def _plot_original_raw(self) -> None:
        """Plot the raw original data.

        Ref: ARMAX.m:75-88 — initial data plot.
        """
        self._ax.clear()
        if self._y is not None:
            self._ax.plot(self._y, linewidth=2, color=(0, 0, 0.6))
            self._ax.set_title("Plot of original data")
            self._ax.autoscale(tight=True)
        self._canvas.draw()

    def _plot_original_acf(self) -> None:
        """Plot the ACF of the original data.

        Ref: ARMAX.m:530-540 — ACF plot for original data.
        """
        self._ax.clear()
        if self._y is not None:
            try:
                from mfe_toolbox.timeseries.sacf import sacf

                acf_vals = sacf(self._y, self._acf_lags)
                lags = np.arange(1, len(acf_vals) + 1)
                self._ax.bar(lags, acf_vals.ravel(), color=(0, 0, 0.6))
                self._ax.set_title("ACF of original data")
                self._ax.set_xlabel("Lag")
                # Significance bounds
                n = len(self._y)
                bound = 1.96 / np.sqrt(n)
                self._ax.axhline(bound, color="r", linestyle="--", linewidth=0.8)
                self._ax.axhline(-bound, color="r", linestyle="--", linewidth=0.8)
                self._ax.axhline(0, color="k", linewidth=0.5)
            except Exception as exc:
                self._ax.set_title(f"ACF error: {exc}")
        self._canvas.draw()

    def _plot_original_pacf(self) -> None:
        """Plot the PACF of the original data.

        Ref: ARMAX.m:545-559 — PACF plot for original data.
        """
        self._ax.clear()
        if self._y is not None:
            try:
                from mfe_toolbox.timeseries.spacf import spacf

                pacf_vals = spacf(self._y, self._acf_lags)
                lags = np.arange(1, len(pacf_vals) + 1)
                self._ax.bar(lags, pacf_vals.ravel(), color=(0, 0, 0.6))
                self._ax.set_title("PACF of original data")
                self._ax.set_xlabel("Lag")
                n = len(self._y)
                bound = 1.96 / np.sqrt(n)
                self._ax.axhline(bound, color="r", linestyle="--", linewidth=0.8)
                self._ax.axhline(-bound, color="r", linestyle="--", linewidth=0.8)
                self._ax.axhline(0, color="k", linewidth=0.5)
            except Exception as exc:
                self._ax.set_title(f"PACF error: {exc}")
        self._canvas.draw()

    def _plot_residuals_raw(self) -> None:
        """Plot raw residuals.

        Ref: ARMAX.m:561-580 — residual raw plot.
        """
        self._ax.clear()
        if self._residuals is not None:
            self._ax.plot(self._residuals, linewidth=1.5, color=(0.6, 0, 0))
            self._ax.set_title("Residuals")
            self._ax.autoscale(tight=True)
        else:
            self._ax.set_title("No residuals available — estimate a model first")
        self._canvas.draw()

    def _plot_residuals_fitted(self) -> None:
        """Plot fitted values (y - residuals).

        Ref: ARMAX.m:585-600 — fitted values overlay.
        """
        self._ax.clear()
        if self._residuals is not None and self._y is not None:
            n_resid = len(self._residuals)
            n_y = len(self._y)
            fitted = self._y[-n_resid:] - self._residuals
            self._ax.plot(self._y[-n_resid:], linewidth=1.5, color=(0, 0, 0.6),
                          label="Original")
            self._ax.plot(fitted, linewidth=1.5, color=(0.6, 0, 0),
                          label="Fitted", linestyle="--")
            self._ax.set_title("Original vs Fitted")
            self._ax.legend()
            self._ax.autoscale(tight=True)
        else:
            self._ax.set_title("No fitted values — estimate a model first")
        self._canvas.draw()

    def _plot_residuals_acf(self) -> None:
        """Plot ACF of residuals.

        Ref: ARMAX.m:605-625 — residual ACF.
        """
        self._ax.clear()
        if self._residuals is not None:
            try:
                from mfe_toolbox.timeseries.sacf import sacf

                acf_vals = sacf(self._residuals, self._acf_lags)
                lags = np.arange(1, len(acf_vals) + 1)
                self._ax.bar(lags, acf_vals.ravel(), color=(0.6, 0, 0))
                self._ax.set_title("ACF of residuals")
                self._ax.set_xlabel("Lag")
                n = len(self._residuals)
                bound = 1.96 / np.sqrt(n)
                self._ax.axhline(bound, color="r", linestyle="--", linewidth=0.8)
                self._ax.axhline(-bound, color="r", linestyle="--", linewidth=0.8)
                self._ax.axhline(0, color="k", linewidth=0.5)
            except Exception as exc:
                self._ax.set_title(f"ACF error: {exc}")
        else:
            self._ax.set_title("No residuals — estimate a model first")
        self._canvas.draw()

    def _plot_residuals_pacf(self) -> None:
        """Plot PACF of residuals.

        Ref: ARMAX.m:630-647 — residual PACF.
        """
        self._ax.clear()
        if self._residuals is not None:
            try:
                from mfe_toolbox.timeseries.spacf import spacf

                pacf_vals = spacf(self._residuals, self._acf_lags)
                lags = np.arange(1, len(pacf_vals) + 1)
                self._ax.bar(lags, pacf_vals.ravel(), color=(0.6, 0, 0))
                self._ax.set_title("PACF of residuals")
                self._ax.set_xlabel("Lag")
                n = len(self._residuals)
                bound = 1.96 / np.sqrt(n)
                self._ax.axhline(bound, color="r", linestyle="--", linewidth=0.8)
                self._ax.axhline(-bound, color="r", linestyle="--", linewidth=0.8)
                self._ax.axhline(0, color="k", linewidth=0.5)
            except Exception as exc:
                self._ax.set_title(f"PACF error: {exc}")
        else:
            self._ax.set_title("No residuals — estimate a model first")
        self._canvas.draw()

    def _plot_lb_qstat(self) -> None:
        """Plot Ljung-Box Q-statistics for residuals.

        Ref: ARMAX.m:649-680 — LB Q-stat test plot.
        """
        self._ax.clear()
        if self._residuals is not None:
            try:
                from mfe_toolbox.tests.ljungbox import ljungbox

                q_stats, p_vals = ljungbox(self._residuals, self._acf_lags)
                lags = np.arange(1, len(q_stats) + 1)
                self._ax.bar(lags, p_vals.ravel(), color=(0, 0.4, 0))
                self._ax.axhline(0.05, color="r", linestyle="--", linewidth=1.0,
                                 label="α=0.05")
                self._ax.set_title("Ljung-Box p-values")
                self._ax.set_xlabel("Lag")
                self._ax.set_ylabel("p-value")
                self._ax.legend()
            except Exception as exc:
                self._ax.set_title(f"LB test error: {exc}")
        else:
            self._ax.set_title("No residuals — estimate a model first")
        self._canvas.draw()

    def _plot_lm_arch(self) -> None:
        """Plot LM ARCH test statistics for residuals.

        Ref: ARMAX.m:685-700 — LM ARCH test plot.
        """
        self._ax.clear()
        if self._residuals is not None:
            try:
                from mfe_toolbox.tests.lmtest1 import lmtest1

                stat, pval = lmtest1(self._residuals, self._acf_lags)
                lags = np.arange(1, len(pval) + 1)
                self._ax.bar(lags, pval.ravel(), color=(0.5, 0, 0.5))
                self._ax.axhline(0.05, color="r", linestyle="--", linewidth=1.0,
                                 label="α=0.05")
                self._ax.set_title("LM ARCH p-values")
                self._ax.set_xlabel("Lag")
                self._ax.set_ylabel("p-value")
                self._ax.legend()
            except Exception as exc:
                self._ax.set_title(f"LM test error: {exc}")
        else:
            self._ax.set_title("No residuals — estimate a model first")
        self._canvas.draw()

    def _clear_plot(self) -> None:
        """Clear the plot axes."""
        self._ax.clear()
        self._canvas.draw()

    # ===================================================================
    # Helper Methods
    # ===================================================================

    def _set_message(self, msg: str) -> None:
        """Update the status message label.

        Ref: ARMAX.m:97 — set(handles.message_text, 'String', msg).
        """
        self.message_text.setText(msg)

    @staticmethod
    def _parse_lag_string(text: str) -> list[int]:
        """Parse a space-separated lag specification string.

        Parameters
        ----------
        text : str
            Space-separated integers, e.g. "1 2 3" or "1 3 5".

        Returns
        -------
        list[int]
            Sorted list of lag integers. Empty list if text is empty.
        """
        if not text:
            return []
        try:
            lags = sorted(set(int(x) for x in text.split() if x.strip()))
            return [lag for lag in lags if lag > 0]
        except ValueError:
            return []

    # ===================================================================
    # Overrides
    # ===================================================================

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Handle window close event — cleanup matplotlib resources.

        Ref: ARMAX.m:491-492 — close callback.
        """
        try:
            self._figure.clear()
            self._canvas.close()
        except Exception:
            pass
        super().closeEvent(event)
