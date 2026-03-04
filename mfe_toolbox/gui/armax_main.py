"""ARMAX Main Estimation Window — PyQt6 QMainWindow.

Migrated from GUI/ARMAX.m (1282 lines) + GUI/ARMAX.fig (MATLAB GUIDE).
Provides the main ARMAX estimation GUI with:
  - Model specification inputs (AR lags, MA lags, exogenous variables)
  - Constant inclusion toggle
  - Estimation, reset, and close actions
  - Plot panel for data visualization (original data, ACF, PACF, residuals,
    fit overlay, LM test, Ljung-Box test)
  - Model selector for comparing multiple estimated models
  - Inference method selection (heteroskedastic-robust / homoskedastic)
  - Export and help menus

Ref: GUI/ARMAX.m — complete GUIDE main window replacement.
Ref: GUI/ARMAX.fig — geometry reconstructed into armax_main.ui.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.stats import norm

# Internal imports — Ref: AAP Section 0.5.2
from mfe_toolbox.timeseries.armaxfilter import armaxfilter
from mfe_toolbox.timeseries.aicsbic import aicsbic
from mfe_toolbox.timeseries.sacf import sacf
from mfe_toolbox.timeseries.spacf import spacf
from mfe_toolbox.tests.lmtest1 import lmtest1
from mfe_toolbox.tests.ljungbox import ljungbox
from mfe_toolbox.gui.armax_viewer import ARMAXViewer
from mfe_toolbox.gui.armax_about import ARMAXAboutDialog
from mfe_toolbox.gui.armax_close_dialog import ARMAXCloseDialog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Ref: ARMAX.m:64 — ACF lag defaults
_DEFAULT_ACF_LAGS_FACTOR: int = 8
_DEFAULT_ACF_LAGS_MAX: int = 24

# Ref: ARMAX.m:65 — inference method codes
_INFERENCE_HETEROROBUST: int = 1
_INFERENCE_HOMOSKEDASTIC: int = 0

# Ref: ARMAX.m:76 — plot line color for data (dark blue)
_DATA_LINE_COLOR: tuple[float, float, float] = (0.0, 0.0, 0.6)

# Ref: ARMAX.m:1066 — fit overlay color (dark green)
_FIT_LINE_COLOR: tuple[float, float, float] = (0.16, 0.38, 0.27)

# Ref: ARMAX.m:959,1101 — ACF bar color
_ACF_BAR_COLOR: tuple[float, float, float] = (0.5, 0.5, 1.0)

# Ref: ARMAX.m:1209 — LM/LB bar color
_TEST_BAR_COLOR: tuple[float, float, float] = (0.5, 0.5, 1.0)

# UI file path (Qt Designer XML)
_MODULE_DIR: Path = Path(__file__).parent
_UI_FILE: Path = _MODULE_DIR / "armax_main.ui"


class ARMAXMainWindow(QMainWindow):
    """Main ARMAX estimation interface.

    Ref: GUI/ARMAX.m — GUIDE main window replacement.
    Provides ARMAX model specification, estimation, plotting, and result viewing.

    Parameters
    ----------
    y : numpy.ndarray
        1-D time series data array.
        Ref: ARMAX.m:62 — ``handles.y = varargin{1}``
    parent : QWidget or None, optional
        Parent widget. ``None`` for standalone window.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        y: np.ndarray,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Initialise the ARMAX main window.

        Ref: ARMAX.m:48-88 — ARMAX_OpeningFcn.
        """
        super().__init__(parent)

        # --- Validate and store data ---
        # Ref: ARMAX.m:62 — handles.y = varargin{1}
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size == 0:
            raise ValueError("Input time series y must be non-empty.")

        # --- Internal state (mirrors MATLAB handles structure) ---
        # Ref: ARMAX.m:58-67 — handles initialization
        self._y: np.ndarray = y
        self._ar_order: list[int] = []             # Ref: line 58
        self._ma_order: list[int] = []             # Ref: line 59
        self._ar_order_string: str = ""             # Ref: line 60
        self._ma_order_string: str = ""             # Ref: line 61
        self._residuals: Optional[np.ndarray] = None  # Ref: line 63
        # Ref: line 64 — ACF_lags = min(24, floor(length(y)/8))
        self._acf_lags: int = min(
            _DEFAULT_ACF_LAGS_MAX,
            max(1, len(y) // _DEFAULT_ACF_LAGS_FACTOR),
        )
        self._inference_method: int = _INFERENCE_HETEROROBUST  # Ref: line 65
        self._include_constant: int = 1                        # Ref: line 66
        self._models: list[dict[str, Any]] = []                # Ref: line 67
        self._current_model_number: int = -1

        # --- Build UI ---
        self._build_ui()
        self._create_canvas()
        self._connect_signals()

        # --- Initial state ---
        # Ref: ARMAX.m:87 — set ACF lags edit text
        self.ACF_lags_edit.setText(str(self._acf_lags))

        # --- Plot initial data ---
        # Ref: ARMAX.m:73-83 — plot original data on opening
        self._plot_raw_data()

        logger.debug("ARMAXMainWindow initialized with T=%d observations.", len(y))

    # ==================================================================
    # UI Construction
    # ==================================================================
    def _build_ui(self) -> None:
        """Construct the main window layout programmatically.

        Mirrors the layout defined in armax_main.ui. Widget objectNames
        match the .ui file for consistency.
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
        model_spec_layout.setContentsMargins(8, 8, 8, 8)

        # AR lags — Ref: ARMAX.m:524 AR_lags_edit
        ar_layout = QHBoxLayout()
        ar_label = QLabel("AR Lags:")
        ar_label.setMinimumWidth(80)
        ar_layout.addWidget(ar_label)
        self.AR_lags_edit = QLineEdit()
        self.AR_lags_edit.setObjectName("AR_lags_edit")
        self.AR_lags_edit.setPlaceholderText("e.g. 1:4")
        self.AR_lags_edit.setToolTip(
            'Enter AR lag specification (e.g. "1:4" or "1 3 5")'
        )
        ar_layout.addWidget(self.AR_lags_edit)
        model_spec_layout.addLayout(ar_layout)

        # MA lags — Ref: ARMAX.m:557 MA_lags_edit
        ma_layout = QHBoxLayout()
        ma_label = QLabel("MA Lags:")
        ma_label.setMinimumWidth(80)
        ma_layout.addWidget(ma_label)
        self.MA_lags_edit = QLineEdit()
        self.MA_lags_edit.setObjectName("MA_lags_edit")
        self.MA_lags_edit.setPlaceholderText("e.g. 1:2")
        self.MA_lags_edit.setToolTip(
            'Enter MA lag specification (e.g. "1:4" or "1 3 5")'
        )
        ma_layout.addWidget(self.MA_lags_edit)
        model_spec_layout.addLayout(ma_layout)

        # Exogenous variables — Ref: ARMAX.m:108,502
        exog_layout = QHBoxLayout()
        exog_label = QLabel("Exogenous:")
        exog_label.setMinimumWidth(80)
        exog_layout.addWidget(exog_label)
        self.EXOG_edit = QLineEdit()
        self.EXOG_edit.setObjectName("EXOG_edit")
        self.EXOG_edit.setToolTip("Enter exogenous variable specification")
        exog_layout.addWidget(self.EXOG_edit)
        self.EXOG_clear_pushbutton = QPushButton("Clear")
        self.EXOG_clear_pushbutton.setObjectName("EXOG_clear_pushbutton")
        self.EXOG_clear_pushbutton.setMaximumWidth(60)
        self.EXOG_clear_pushbutton.setToolTip("Clear exogenous variable specification")
        exog_layout.addWidget(self.EXOG_clear_pushbutton)
        model_spec_layout.addLayout(exog_layout)

        # Constant toggle — Ref: ARMAX.m:890-912
        const_layout = QHBoxLayout()
        self.constant_include = QRadioButton("Include Constant")
        self.constant_include.setObjectName("constant_include")
        self.constant_include.setChecked(True)
        self.constant_include.setToolTip("Include a constant term in the model (default)")
        self.constant_exclude = QRadioButton("Exclude Constant")
        self.constant_exclude.setObjectName("constant_exclude")
        self.constant_exclude.setToolTip("Exclude the constant term from the model")
        const_layout.addWidget(self.constant_include)
        const_layout.addWidget(self.constant_exclude)
        model_spec_layout.addLayout(const_layout)

        # ACF lags — Ref: ARMAX.m:674
        acf_layout = QHBoxLayout()
        acf_label = QLabel("ACF Lags:")
        acf_label.setMinimumWidth(80)
        acf_layout.addWidget(acf_label)
        self.ACF_lags_edit = QLineEdit()
        self.ACF_lags_edit.setObjectName("ACF_lags_edit")
        self.ACF_lags_edit.setToolTip("Number of autocorrelation lags to display")
        acf_layout.addWidget(self.ACF_lags_edit)
        model_spec_layout.addLayout(acf_layout)

        # Hold-back — Ref: ARMAX.m:625
        holdback_layout = QHBoxLayout()
        holdback_label = QLabel("Hold Back:")
        holdback_label.setMinimumWidth(80)
        holdback_layout.addWidget(holdback_label)
        self.hold_back_edit = QLineEdit()
        self.hold_back_edit.setObjectName("hold_back_edit")
        self.hold_back_edit.setText("0")
        self.hold_back_edit.setToolTip("Number of initial observations to hold back")
        holdback_layout.addWidget(self.hold_back_edit)
        model_spec_layout.addLayout(holdback_layout)

        left_layout.addWidget(model_spec_group)

        # Actions Group — Ref: ARMAX.m buttons
        actions_group = QGroupBox("Actions")
        actions_group.setObjectName("actionsGroupBox")
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setSpacing(6)
        actions_layout.setContentsMargins(8, 8, 8, 8)

        self.estimate_pushbutton = QPushButton("Estimate")
        self.estimate_pushbutton.setObjectName("estimate_pushbutton")
        self.estimate_pushbutton.setToolTip("Run ARMAX estimation with current specification")
        actions_layout.addWidget(self.estimate_pushbutton)

        self.reset_pushbutton = QPushButton("Reset")
        self.reset_pushbutton.setObjectName("reset_pushbutton")
        self.reset_pushbutton.setToolTip("Clear all inputs and reset to original data")
        actions_layout.addWidget(self.reset_pushbutton)

        self.close_pushbutton = QPushButton("Close")
        self.close_pushbutton.setObjectName("close_pushbutton")
        self.close_pushbutton.setToolTip("Close the ARMAX estimation window")
        actions_layout.addWidget(self.close_pushbutton)

        left_layout.addWidget(actions_group)

        # Model selector — Ref: ARMAX.m:834-868
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Model:"))
        self.model_selector_popup = QComboBox()
        self.model_selector_popup.setObjectName("model_selector_popup")
        self.model_selector_popup.setToolTip("Select a previously estimated model")
        self.model_selector_popup.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        selector_layout.addWidget(self.model_selector_popup)
        left_layout.addLayout(selector_layout)

        # Last Model button — Ref: ARMAX.m:591
        self.last_model_pushbutton = QPushButton("Last Model")
        self.last_model_pushbutton.setObjectName("last_model_pushbutton")
        self.last_model_pushbutton.setToolTip("Load the last estimated model")
        left_layout.addWidget(self.last_model_pushbutton)

        # Message text — Ref: ARMAX.m:308 status messages
        self.message_text = QLabel("")
        self.message_text.setObjectName("message_text")
        self.message_text.setWordWrap(True)
        self.message_text.setMinimumHeight(40)
        self.message_text.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        self.message_text.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
        )
        left_layout.addWidget(self.message_text, stretch=1)

        # --- RIGHT PANEL: Plot container ---
        self.plot_container = QWidget()
        self.plot_container.setObjectName("plot_container")
        self.plot_container.setMinimumSize(585, 392)
        self.plot_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        top_layout.addWidget(self.plot_container, stretch=1)

        # --- BOTTOM: Plot type selectors ---
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(6)
        main_layout.addLayout(bottom_layout, stretch=0)

        # All radio buttons across groups share one QButtonGroup so only
        # one can be selected at a time.
        self._plot_button_group = QButtonGroup(self)
        self._plot_button_group.setExclusive(True)

        # Original Data plot options — Ref: ARMAX.m:917-978
        orig_group = QGroupBox("Original Data")
        orig_group.setObjectName("originalDataGroupBox")
        orig_layout = QHBoxLayout(orig_group)
        orig_layout.setSpacing(4)
        orig_layout.setContentsMargins(8, 4, 8, 4)

        self.orig_raw_radio = QRadioButton("Raw Data")
        self.orig_raw_radio.setObjectName("orig_raw_radio")
        self.orig_raw_radio.setChecked(True)
        self.orig_raw_radio.setToolTip("Plot raw original data")
        self.orig_acf_radio = QRadioButton("ACF")
        self.orig_acf_radio.setObjectName("orig_acf_radio")
        self.orig_acf_radio.setToolTip("Plot autocorrelation function of original data")
        self.orig_pacf_radio = QRadioButton("PACF")
        self.orig_pacf_radio.setObjectName("orig_pacf_radio")
        self.orig_pacf_radio.setToolTip(
            "Plot partial autocorrelation function of original data"
        )
        orig_layout.addWidget(self.orig_raw_radio)
        orig_layout.addWidget(self.orig_acf_radio)
        orig_layout.addWidget(self.orig_pacf_radio)
        self._plot_button_group.addButton(self.orig_raw_radio)
        self._plot_button_group.addButton(self.orig_acf_radio)
        self._plot_button_group.addButton(self.orig_pacf_radio)
        bottom_layout.addWidget(orig_group)

        # Residuals plot options — Ref: ARMAX.m:1020-1167
        resid_group = QGroupBox("Residuals")
        resid_group.setObjectName("residualsGroupBox")
        resid_layout = QHBoxLayout(resid_group)
        resid_layout.setSpacing(4)
        resid_layout.setContentsMargins(8, 4, 8, 4)

        self.resid_raw_radio = QRadioButton("Raw Residuals")
        self.resid_raw_radio.setObjectName("resid_raw_radio")
        self.resid_raw_radio.setToolTip("Plot raw model residuals")
        self.resid_fit_radio = QRadioButton("Fit vs Original")
        self.resid_fit_radio.setObjectName("resid_fit_radio")
        self.resid_fit_radio.setToolTip("Overlay original data and fitted values")
        self.resid_acf_radio = QRadioButton("Residual ACF")
        self.resid_acf_radio.setObjectName("resid_acf_radio")
        self.resid_acf_radio.setToolTip("Plot autocorrelation function of residuals")
        self.resid_pacf_radio = QRadioButton("Residual PACF")
        self.resid_pacf_radio.setObjectName("resid_pacf_radio")
        self.resid_pacf_radio.setToolTip(
            "Plot partial autocorrelation function of residuals"
        )
        resid_layout.addWidget(self.resid_raw_radio)
        resid_layout.addWidget(self.resid_fit_radio)
        resid_layout.addWidget(self.resid_acf_radio)
        resid_layout.addWidget(self.resid_pacf_radio)
        self._plot_button_group.addButton(self.resid_raw_radio)
        self._plot_button_group.addButton(self.resid_fit_radio)
        self._plot_button_group.addButton(self.resid_acf_radio)
        self._plot_button_group.addButton(self.resid_pacf_radio)
        bottom_layout.addWidget(resid_group)

        # Tests group — Ref: ARMAX.m:1171-1271
        tests_group = QGroupBox("Tests")
        tests_group.setObjectName("testsGroupBox")
        tests_layout = QVBoxLayout(tests_group)
        tests_layout.setSpacing(2)
        tests_layout.setContentsMargins(8, 4, 8, 4)

        self.radiobutton18 = QRadioButton("Original: LM/LB Tests")
        self.radiobutton18.setObjectName("radiobutton18")
        self.radiobutton18.setToolTip(
            "LM test or Ljung-Box statistics for original data"
        )
        self.radiobutton19 = QRadioButton("Residual: LM/LB Tests")
        self.radiobutton19.setObjectName("radiobutton19")
        self.radiobutton19.setToolTip(
            "LM test or Ljung-Box statistics for model residuals"
        )
        tests_layout.addWidget(self.radiobutton18)
        tests_layout.addWidget(self.radiobutton19)
        self._plot_button_group.addButton(self.radiobutton18)
        self._plot_button_group.addButton(self.radiobutton19)
        bottom_layout.addWidget(tests_group)

        # --- Menu bar ---
        self._build_menu_bar()

    def _build_menu_bar(self) -> None:
        """Build the menu bar matching ARMAX.m menus.

        Ref: ARMAX.m — export_menu, residual_menu, stderror_menu, help menu.
        """
        menubar = self.menuBar()

        # File / Export menu — Ref: ARMAX.m:457-484, 599-621
        self.export_menu = menubar.addMenu("File")
        self.export_menu.setObjectName("export_menu")
        self.action_export_tiff = QAction("Export as TIFF...", self)
        self.action_export_tiff.setObjectName("action_export_tiff")
        self.export_menu.addAction(self.action_export_tiff)
        self.action_export_png = QAction("Export as PNG...", self)
        self.action_export_png.setObjectName("action_export_png")
        self.export_menu.addAction(self.action_export_png)
        self.action_export_eps = QAction("Export as EPS...", self)
        self.action_export_eps.setObjectName("action_export_eps")
        self.export_menu.addAction(self.action_export_eps)
        self.export_menu.addSeparator()
        self.action_copy_figure = QAction("Copy Figure", self)
        self.action_copy_figure.setObjectName("action_copy_figure")
        self.export_menu.addAction(self.action_copy_figure)

        # Residuals menu — Ref: ARMAX.m:443-453
        self.residual_menu = menubar.addMenu("Residuals")
        self.residual_menu.setObjectName("residual_menu")
        self.action_save_residuals = QAction("Save Residuals...", self)
        self.action_save_residuals.setObjectName("action_save_residuals")
        self.residual_menu.addAction(self.action_save_residuals)
        self.action_save_std_residuals = QAction(
            "Save Standardized Residuals...", self
        )
        self.action_save_std_residuals.setObjectName("action_save_std_residuals")
        self.residual_menu.addAction(self.action_save_std_residuals)

        # Standard Errors menu — Ref: ARMAX.m:762-785
        self.stderror_menu = menubar.addMenu("Standard Errors")
        self.stderror_menu.setObjectName("stderror_menu")

        self.action_heterorobust = QAction("Heteroskedasticity-Robust", self)
        self.action_heterorobust.setObjectName("action_heterorobust")
        self.action_heterorobust.setCheckable(True)
        self.action_heterorobust.setChecked(True)  # Ref: ARMAX.m:85
        self.stderror_menu.addAction(self.action_heterorobust)

        self.action_homoerror = QAction("Homoskedastic", self)
        self.action_homoerror.setObjectName("action_homoerror")
        self.action_homoerror.setCheckable(True)
        self.action_homoerror.setChecked(False)  # Ref: ARMAX.m:86
        self.stderror_menu.addAction(self.action_homoerror)

        # Help menu — Ref: ARMAX.m:735-739
        self.help_menu = menubar.addMenu("Help")
        self.help_menu.setObjectName("menu_help")
        self.action_about = QAction("About...", self)
        self.action_about.setObjectName("action_about")
        self.help_menu.addAction(self.action_about)

    def _create_canvas(self) -> None:
        """Create the matplotlib FigureCanvasQTAgg for data visualization.

        Ref: ARMAX.m:73-83 — initial axes setup; line 1277 — 585×392 pixels.
        """
        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)

        # Embed canvas into the plot container
        layout = QVBoxLayout(self.plot_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def _connect_signals(self) -> None:
        """Wire button/menu signals to slot methods.

        Ref: ARMAX.m — each GUIDE callback function maps to a slot here.
        """
        # Action buttons
        self.estimate_pushbutton.clicked.connect(self._on_estimate)
        self.close_pushbutton.clicked.connect(self._on_close)
        self.reset_pushbutton.clicked.connect(self._on_reset)
        self.EXOG_clear_pushbutton.clicked.connect(self._on_clear_exog)

        # Edit field validation — Ref: ARMAX.m:524,557,674
        self.AR_lags_edit.editingFinished.connect(self._on_ar_lags_changed)
        self.MA_lags_edit.editingFinished.connect(self._on_ma_lags_changed)
        self.ACF_lags_edit.editingFinished.connect(self._on_acf_lags_changed)

        # Constant toggle — Ref: ARMAX.m:890-912
        self.constant_include.toggled.connect(self._on_constant_include)
        self.constant_exclude.toggled.connect(self._on_constant_exclude)

        # Model selector — Ref: ARMAX.m:834-868
        self.model_selector_popup.currentIndexChanged.connect(
            self._on_model_selected
        )
        self.last_model_pushbutton.clicked.connect(self._on_last_model)

        # Plot radio buttons — connected via QButtonGroup
        self._plot_button_group.buttonToggled.connect(self._on_plot_radio_toggled)

        # Menu actions
        self.action_heterorobust.triggered.connect(self._on_hetero_robust)
        self.action_homoerror.triggered.connect(self._on_homo_error)
        self.action_about.triggered.connect(self._on_about)

        # Export actions
        self.action_export_tiff.triggered.connect(
            lambda: self._on_export_plot("tiff")
        )
        self.action_export_png.triggered.connect(
            lambda: self._on_export_plot("png")
        )
        self.action_export_eps.triggered.connect(
            lambda: self._on_export_plot("eps")
        )
        self.action_copy_figure.triggered.connect(self._on_copy_figure)
        self.action_save_residuals.triggered.connect(self._on_save_residuals)
        self.action_save_std_residuals.triggered.connect(
            self._on_save_std_residuals
        )

    # ==================================================================
    # Public API — show() and close() from schema
    # ==================================================================
    def show(self) -> None:
        """Show the main window.

        Overrides QMainWindow.show() for schema compliance.
        """
        super().show()

    def close(self) -> bool:
        """Close the main window.

        Overrides QMainWindow.close() for schema compliance.
        """
        return super().close()

    # ==================================================================
    # Core Estimation — _on_estimate()
    # ==================================================================
    def _on_estimate(self) -> None:
        """Handle Estimate button click — run ARMAX estimation.

        Ref: ARMAX.m:160-345 — estimate_pushbutton_Callback.
        Retrieves model specification, checks for duplicate models,
        calls armaxfilter, computes AIC/BIC and p-values, builds
        StringID, updates model selector, and launches ARMAXViewer.
        """
        y = self._y
        ar = list(self._ar_order)
        ma = list(self._ma_order)
        constant = self._include_constant

        # --- Readiness check ---
        # Ref: ARMAX.m:177-179 — both ar and ma empty/zero means not ready
        ready_to_run = True
        not_already_run = True

        if (not ar or all(a == 0 for a in ar)) and (
            not ma or all(m == 0 for m in ma)
        ):
            ready_to_run = False

        # --- Check for previously estimated duplicate model ---
        # Ref: ARMAX.m:191-228 — compare constant, AR lags, MA lags
        matched_model = -1
        errors: Optional[np.ndarray] = None
        for i, model in enumerate(self._models):
            match = True
            # Ref: ARMAX.m:203 — constant comparison
            if constant != model.get("constant", None):
                match = False
            # Ref: ARMAX.m:207-210 — AR lag comparison
            model_ar = model.get("ARlags", [])
            if len(ar) != len(model_ar):
                match = False
            elif ar and model_ar and sorted(set(ar)) != sorted(set(model_ar)):
                match = False
            # Ref: ARMAX.m:213-216 — MA lag comparison
            model_ma = model.get("MAlags", [])
            if len(ma) != len(model_ma):
                match = False
            elif ma and model_ma and sorted(set(ma)) != sorted(set(model_ma)):
                match = False
            if match:
                matched_model = i
                not_already_run = False

        if matched_model >= 0:
            # Ref: ARMAX.m:225-228 — reuse stored results
            self._current_model_number = matched_model
            errors = self._models[matched_model].get("errors")

        # --- Estimate if ready and not duplicate ---
        # Ref: ARMAX.m:234-314 — estimation block
        if ready_to_run and not_already_run:
            try:
                # Ref: ARMAX.m:236 — armaxfilter(y, constant, ar, ma)
                result = armaxfilter(y, constant, ar, ma)
                (
                    parameters,
                    ll,
                    est_errors,
                    seregression,
                    diagnostics,
                    vcvrobust,
                    vcv,
                    likelihoods,
                    scores,
                ) = result

                # Ref: ARMAX.m:237-253 — store model results
                model_no = len(self._models)
                model_dict: dict[str, Any] = {
                    "parameters": parameters,
                    "errors": est_errors,
                    "ll": ll,
                    "seregression": seregression,
                    "diagnostics": diagnostics,
                    "vcvrobust": vcvrobust,
                    "vcv": vcv,
                    "likelihoods": likelihoods,
                    "scores": scores,
                    "ARlags": ar,
                    "MAlags": ma,
                    "constant": constant,
                    "y": y,
                    "yhat": y - est_errors,
                }

                # Ref: ARMAX.m:244-248 — select covariance by inference method
                if self._inference_method:
                    model_dict["covariance"] = vcvrobust
                else:
                    model_dict["covariance"] = vcv

                # Ref: ARMAX.m:256 — compute AIC/BIC
                # aicsbic(errors, constant, ar, ma)
                aic_val, bic_val = aicsbic(est_errors, constant, ar, ma)
                model_dict["AIC"] = aic_val
                model_dict["BIC"] = bic_val

                # Ref: ARMAX.m:259-262 — StdErr, Tstat, Pval
                model_dict["K"] = len(parameters)
                cov_diag = np.diag(model_dict["covariance"])
                # Guard against negative diagonal entries
                cov_diag = np.maximum(cov_diag, 0.0)
                model_dict["StdErr"] = np.sqrt(cov_diag)
                model_dict["Tstat"] = parameters / model_dict["StdErr"]
                # Ref: ARMAX.m:262 — Pval = 2 - 2*normcdf(abs(Tstat))
                # AAP: normcdf(x) → norm.cdf(x)
                model_dict["Pval"] = 2.0 - 2.0 * norm.cdf(
                    np.abs(model_dict["Tstat"])
                )

                # --- Build StringID ---
                # Ref: ARMAX.m:265-300
                model_dict["StringID"] = self._build_string_id(
                    ar, ma, constant
                )

                self._models.append(model_dict)
                self._current_model_number = model_no
                self._residuals = est_errors
                errors = est_errors

                # --- Update model_selector_popup ---
                # Ref: ARMAX.m:302-306
                self.model_selector_popup.blockSignals(True)
                self.model_selector_popup.clear()
                for mdl in self._models:
                    self.model_selector_popup.addItem(mdl["StringID"])
                self.model_selector_popup.setCurrentIndex(
                    self.model_selector_popup.count() - 1
                )
                self.model_selector_popup.blockSignals(False)

                # Ref: ARMAX.m:308 — success message
                self._set_message(
                    "The model has been successfully estimated.",
                    color="blue",
                )

            except Exception as exc:
                # Ref: ARMAX.m:310-314 — error handling
                self._set_message(
                    f"There was an error: {exc}",
                    color="red",
                )
                logger.exception("ARMAX estimation failed")
                return

        elif not ready_to_run:
            # Ref: ARMAX.m:315-316 — not ready message
            self._set_message(
                "The model is not ready to run. Please check the inputs.",
                color="red",
            )
        elif not not_already_run:
            # Ref: ARMAX.m:317-318 — already estimated message
            self._set_message(
                "Model previously estimated; results loaded.",
                color="blue",
            )

        # --- Plot residuals ---
        # Ref: ARMAX.m:321-335 — plot errors on estimation or reuse
        if ready_to_run or not not_already_run:
            if errors is not None:
                self._residuals = errors
            self.resid_raw_radio.setChecked(True)
            self._plot_residuals()

        # --- Launch ARMAXViewer ---
        # Ref: ARMAX.m:341-345
        if ready_to_run and not_already_run:
            try:
                self._viewer = ARMAXViewer(
                    results=self._models, parent=self
                )
                self._viewer.show()
            except Exception as exc:
                logger.warning("Failed to launch ARMAXViewer: %s", exc)
        elif not not_already_run:
            try:
                self._viewer = ARMAXViewer(
                    results=self._models,
                    display_number=matched_model,
                    parent=self,
                )
                self._viewer.show()
            except Exception as exc:
                logger.warning("Failed to launch ARMAXViewer: %s", exc)

    # ==================================================================
    # Close / Reset Methods
    # ==================================================================
    def _on_close(self) -> None:
        """Handle Close button — show save-before-close dialog.

        Ref: ARMAX.m:372-386 — close_pushbutton_Callback.
        Calls ARMAXCloseDialog.ask() and closes window on 'Yes'.
        """
        # Ref: ARMAX.m:380 — ARMAX_close_dialog('Title','Confirm Close')
        user_response = ARMAXCloseDialog.ask(
            parent=self, title="Confirm Close"
        )
        # Ref: ARMAX.m:381-386 — switch on response
        if user_response == "Yes":
            self.close()
        # 'No' — take no action

    def _on_reset(self) -> None:
        """Handle Reset button — clear all state and inputs, replot original data.

        Ref: ARMAX.m:402-432 — reset_pushbutton_Callback.
        """
        # Ref: ARMAX.m:407-408 — clear orders
        self._ar_order = []
        self._ma_order = []
        self._ar_order_string = ""
        self._ma_order_string = ""
        self._residuals = None

        # Ref: ARMAX.m:412-417 — clear edit fields
        self.AR_lags_edit.setText("")
        self.MA_lags_edit.setText("")
        self.EXOG_edit.setText(" ")
        self.hold_back_edit.setText("0")
        self.message_text.setText("")

        # Ref: ARMAX.m:415 — set orig_raw_radio selected
        self.orig_raw_radio.setChecked(True)

        # Ref: ARMAX.m:419-432 — replot original data
        self._plot_raw_data()

    def closeEvent(self, event: Any) -> None:
        """Override close event to show confirmation dialog.

        Ref: ARMAX.m:657-671 — figure1_CloseRequestFcn.
        """
        # Ref: ARMAX.m:665 — ARMAX_close_dialog('Title','Confirm Close')
        user_response = ARMAXCloseDialog.ask(
            parent=self, title="Confirm Close"
        )
        if user_response == "Yes":
            # Ref: ARMAX.m:670 — delete(handles.figure1)
            try:
                self._figure.clear()
                self._canvas.close()
            except Exception:
                pass
            event.accept()
        else:
            # Ref: ARMAX.m:668 — take no action on 'No'
            event.ignore()

    # ==================================================================
    # Plot Methods
    # ==================================================================
    def _plot_raw_data(self) -> None:
        """Plot the raw original data.

        Ref: ARMAX.m:917-937 — orig_raw_radio_Callback.
        Also used for initial data display (ARMAX.m:73-83).
        """
        self._clear_axes()
        y = self._y
        # Ref: ARMAX.m:929-930 — plot(y), LineWidth=2, Color=[0 0 .6]
        self._ax.plot(y, linewidth=2, color=_DATA_LINE_COLOR)
        # Ref: ARMAX.m:932 — title('Plot of original data')
        self._ax.set_title("Plot of original data")
        # Ref: ARMAX.m:931,933-936 — axis tight + 10% padding
        self._ax.autoscale(tight=True)
        self._apply_padding()
        self._canvas.draw()

    def _plot_orig_acf(self) -> None:
        """Plot ACF of original data with standard error bands.

        Ref: ARMAX.m:942-978 — orig_acf_radio_Callback.
        Calls sacf(y, ACF_lags, robust, 0) and plots bar + SE lines.
        """
        self._clear_axes()
        y = self._y
        acf_lags = self._acf_lags
        robust = bool(self._inference_method)

        # Ref: ARMAX.m:956 — [ac, acstd] = sacf(y, ACF_lags, robust, 0)
        ac, acstd, _ = sacf(y, acf_lags, robust, 0)
        ac = np.asarray(ac).ravel()
        acstd = np.asarray(acstd).ravel()

        # Ref: ARMAX.m:958 — bar(ac)
        lags = np.arange(1, len(ac) + 1)
        self._ax.bar(lags, ac, color=_ACF_BAR_COLOR)

        # Ref: ARMAX.m:960-967 — axis tight + spread
        self._ax.autoscale(tight=True)
        ax_lim = list(self._ax.get_ylim())
        spread = 0.2 * (ax_lim[1] - ax_lim[0])
        self._ax.set_xlim(0, acf_lags + 1)
        self._ax.set_ylim(ax_lim[0] - spread, ax_lim[1] + spread)

        # Ref: ARMAX.m:969 — SE confidence bands
        # Build SE line: extend first and last values for plotting 0..ACF_lags+1
        se_x = np.arange(0, acf_lags + 2)
        se_upper = np.concatenate(
            [[acstd[0]], acstd, [acstd[acf_lags - 1]]]
        )
        se_lower = -se_upper
        # Ref: ARMAX.m:977-978 — LineWidth=2, LineStyle=':', Color=[0 0 0]
        self._ax.plot(
            se_x, se_upper, linewidth=2, linestyle=":", color="black"
        )
        self._ax.plot(
            se_x, se_lower, linewidth=2, linestyle=":", color="black"
        )

        # Ref: ARMAX.m:971-975 — title based on robust flag
        if robust:
            self._ax.set_title(
                "Sample Autocorrelations and Robust Standard Errors"
            )
        else:
            self._ax.set_title(
                "Sample Autocorrelations and Non-robust Standard Errors"
            )
        self._canvas.draw()

    def _plot_orig_pacf(self) -> None:
        """Plot PACF of original data with 2×SE confidence bands.

        Ref: ARMAX.m:981-1016 — orig_pacf_radio_Callback.
        Calls spacf(y, ACF_lags, robust, 0) and plots bar + 2×SE lines.
        """
        self._clear_axes()
        y = self._y
        acf_lags = self._acf_lags
        robust = bool(self._inference_method)

        # Ref: ARMAX.m:994 — [pac, pacstd] = spacf(y, ACF_lags, robust, 0)
        pac, pacstd, _ = spacf(y, acf_lags, robust, 0)
        pac = np.asarray(pac).ravel()
        pacstd = np.asarray(pacstd).ravel()

        # Ref: ARMAX.m:996 — bar(pac)
        lags = np.arange(1, len(pac) + 1)
        self._ax.bar(lags, pac, color=_ACF_BAR_COLOR)

        # Ref: ARMAX.m:998-1005 — axis tight + spread
        self._ax.autoscale(tight=True)
        ax_lim = list(self._ax.get_ylim())
        spread = 0.2 * (ax_lim[1] - ax_lim[0])
        self._ax.set_xlim(0, acf_lags + 1)
        self._ax.set_ylim(ax_lim[0] - spread, ax_lim[1] + spread)

        # Ref: ARMAX.m:1007 — 2×SE confidence bands (note PACF uses 2× multiplier)
        se_x = np.arange(0, acf_lags + 2)
        se_upper = 2.0 * np.concatenate(
            [[pacstd[0]], pacstd, [pacstd[acf_lags - 1]]]
        )
        se_lower = -se_upper
        self._ax.plot(
            se_x, se_upper, linewidth=2, linestyle=":", color="black"
        )
        self._ax.plot(
            se_x, se_lower, linewidth=2, linestyle=":", color="black"
        )

        # Ref: ARMAX.m:1009-1013 — title based on robust flag
        if robust:
            self._ax.set_title(
                "Sample Partial Autocorrelations and Robust Standard Errors"
            )
        else:
            self._ax.set_title(
                "Sample Partial Autocorrelations and Non-robust Standard Errors"
            )
        self._canvas.draw()

    def _plot_residuals(self) -> None:
        """Plot raw model residuals (errors).

        Ref: ARMAX.m:1020-1045 — resid_raw_radio_Callback.
        """
        self._clear_axes()
        if self._residuals is None or len(self._residuals) == 0:
            # Ref: ARMAX.m:1028-1029 — no model run
            self._no_model_run_plot()
            return

        y = self._residuals
        # Ref: ARMAX.m:1036-1037 — plot(y), LineWidth=2, Color=[0 0 .6]
        self._ax.plot(y, linewidth=2, color=_DATA_LINE_COLOR)
        # Ref: ARMAX.m:1039 — title('Plot of estimated errors')
        self._ax.set_title("Plot of estimated errors")
        # Ref: ARMAX.m:1038,1040-1044 — axis tight + 10% padding
        self._ax.autoscale(tight=True)
        self._apply_padding()
        self._canvas.draw()

    def _plot_fit_overlay(self) -> None:
        """Plot original data and fitted values overlay.

        Ref: ARMAX.m:1050-1076 — resid_fit_radio_Callback.
        """
        self._clear_axes()
        if self._residuals is None or len(self._residuals) == 0:
            self._no_model_run_plot()
            return

        # Ref: ARMAX.m:1059-1060 — yhat = y - errors
        yhat = self._y - self._residuals
        y = self._y
        # Ref: ARMAX.m:1065-1067 — plot original and fit
        self._ax.plot(
            y, linewidth=2, color=_DATA_LINE_COLOR, label="Original Data"
        )
        self._ax.plot(
            yhat, linewidth=2, color=_FIT_LINE_COLOR, label="Fit Data"
        )
        # Ref: ARMAX.m:1069 — title('Plot of estimated errors')
        self._ax.set_title("Plot of estimated errors")
        # Ref: ARMAX.m:1070 — legend
        self._ax.legend()
        # Ref: ARMAX.m:1068,1071-1075 — axis tight + padding
        self._ax.autoscale(tight=True)
        self._apply_padding()
        self._canvas.draw()

    def _plot_resid_acf(self) -> None:
        """Plot ACF of residuals with standard error bands.

        Ref: ARMAX.m:1081-1121 — resid_acf_radio_Callback.
        """
        self._clear_axes()
        if self._residuals is None or len(self._residuals) == 0:
            self._no_model_run_plot()
            return

        y = self._residuals
        acf_lags = self._acf_lags
        robust = bool(self._inference_method)

        # Ref: ARMAX.m:1098 — [ac, acstd] = sacf(y, ACF_lags, robust, 0)
        ac, acstd, _ = sacf(y, acf_lags, robust, 0)
        ac = np.asarray(ac).ravel()
        acstd = np.asarray(acstd).ravel()

        lags = np.arange(1, len(ac) + 1)
        self._ax.bar(lags, ac, color=_ACF_BAR_COLOR)

        self._ax.autoscale(tight=True)
        ax_lim = list(self._ax.get_ylim())
        spread = 0.2 * (ax_lim[1] - ax_lim[0])
        self._ax.set_xlim(0, acf_lags + 1)
        self._ax.set_ylim(ax_lim[0] - spread, ax_lim[1] + spread)

        # Ref: ARMAX.m:1111 — SE bands
        se_x = np.arange(0, acf_lags + 2)
        se_upper = np.concatenate(
            [[acstd[0]], acstd, [acstd[acf_lags - 1]]]
        )
        se_lower = -se_upper
        self._ax.plot(
            se_x, se_upper, linewidth=2, linestyle=":", color="black"
        )
        self._ax.plot(
            se_x, se_lower, linewidth=2, linestyle=":", color="black"
        )

        # Ref: ARMAX.m:1113-1117 — title
        if robust:
            self._ax.set_title(
                "Sample Autocorrelations and Robust Standard Errors"
            )
        else:
            self._ax.set_title(
                "Sample Autocorrelations and Non-robust Standard Errors"
            )
        self._canvas.draw()

    def _plot_resid_pacf(self) -> None:
        """Plot PACF of residuals with 2×SE confidence bands.

        Ref: ARMAX.m:1125-1167 — resid_pacf_radio_Callback.
        """
        self._clear_axes()
        if self._residuals is None or len(self._residuals) == 0:
            self._no_model_run_plot()
            return

        y = self._residuals
        acf_lags = self._acf_lags
        robust = bool(self._inference_method)

        # Ref: ARMAX.m:1143 — [pac, pacstd] = spacf(y, ACF_lags, robust, 0)
        pac, pacstd, _ = spacf(y, acf_lags, robust, 0)
        pac = np.asarray(pac).ravel()
        pacstd = np.asarray(pacstd).ravel()

        lags = np.arange(1, len(pac) + 1)
        self._ax.bar(lags, pac, color=_ACF_BAR_COLOR)

        self._ax.autoscale(tight=True)
        ax_lim = list(self._ax.get_ylim())
        spread = 0.2 * (ax_lim[1] - ax_lim[0])
        self._ax.set_xlim(0, acf_lags + 1)
        self._ax.set_ylim(ax_lim[0] - spread, ax_lim[1] + spread)

        # Ref: ARMAX.m:1156 — 2×SE bands (PACF uses 2× multiplier)
        se_x = np.arange(0, acf_lags + 2)
        se_upper = 2.0 * np.concatenate(
            [[pacstd[0]], pacstd, [pacstd[acf_lags - 1]]]
        )
        se_lower = -se_upper
        self._ax.plot(
            se_x, se_upper, linewidth=2, linestyle=":", color="black"
        )
        self._ax.plot(
            se_x, se_lower, linewidth=2, linestyle=":", color="black"
        )

        # Ref: ARMAX.m:1158-1162
        if robust:
            self._ax.set_title(
                "Sample Partial Autocorrelations and Robust Standard Errors"
            )
        else:
            self._ax.set_title(
                "Sample Partial Autocorrelations and Non-robust Standard Errors"
            )
        self._canvas.draw()

    def _plot_lm_lb_original(self) -> None:
        """Plot LM test or Ljung-Box test on original data y.

        Ref: ARMAX.m:1171-1218 — radiobutton18_Callback.
        Uses dual-axis: bar(stat) on left, scatter(pval) on right.
        """
        self._clear_axes()
        lags_count = self._acf_lags
        y = self._y

        # Ref: ARMAX.m:1182-1188 — choose test by inference method
        if self._inference_method:
            # Ref: ARMAX.m:1183 — [stat, pval] = lmtest1(y, lags)
            stat, pval = lmtest1(y, lags_count)
            title_string = (
                "LM Test Statistic Values (left) and P-values (right)"
            )
        else:
            # Ref: ARMAX.m:1186 — [stat, pval] = ljungbox(y, lags)
            stat, pval = ljungbox(y, lags_count)
            title_string = (
                "Ljung-Box Q Statistics (left) and P-values (right)"
            )

        stat = np.asarray(stat).ravel()
        pval = np.asarray(pval).ravel()

        self._draw_dual_axis_test_plot(stat, pval, title_string, lags_count)

    def _plot_lm_lb_residuals(self) -> None:
        """Plot LM test or Ljung-Box test on model residuals.

        Ref: ARMAX.m:1223-1271 — radiobutton19_Callback.
        Uses dual-axis: bar(stat) on left, scatter(pval) on right.
        """
        self._clear_axes()
        if self._residuals is None or len(self._residuals) == 0:
            # Ref: ARMAX.m:1230-1231
            self._no_model_run_plot()
            return

        lags_count = self._acf_lags
        y = self._residuals

        # Ref: ARMAX.m:1235-1241 — choose test by inference method
        if self._inference_method:
            stat, pval = lmtest1(y, lags_count)
            title_string = (
                "Error LM Test Statistic Values (left) and P-values (right)"
            )
        else:
            stat, pval = ljungbox(y, lags_count)
            title_string = (
                "Error Ljung-Box Q Statistics (left) and P-values (right)"
            )

        stat = np.asarray(stat).ravel()
        pval = np.asarray(pval).ravel()

        self._draw_dual_axis_test_plot(stat, pval, title_string, lags_count)

    # ==================================================================
    # Helper — Dual-Axis Test Plot
    # ==================================================================
    def _draw_dual_axis_test_plot(
        self,
        stat: np.ndarray,
        pval: np.ndarray,
        title_string: str,
        lags_count: int,
    ) -> None:
        """Draw a dual-axis bar+scatter plot for LM or LB test statistics.

        Ref: ARMAX.m:1190-1217 and 1243-1270 — shared dual-axis logic.
        Left axis: bar chart of test statistics.
        Right axis: scatter plot of p-values scaled to left axis range.

        Parameters
        ----------
        stat : numpy.ndarray
            Test statistic values, shape (lags,).
        pval : numpy.ndarray
            P-values, shape (lags,).
        title_string : str
            Title for the plot.
        lags_count : int
            Number of lags.
        """
        max_stat = float(np.max(stat)) if stat.size > 0 else 1.0
        if max_stat == 0.0:
            max_stat = 1.0

        # Ref: ARMAX.m:1192-1195 — axis limits
        bar_ul = max_stat * 1.1
        bar_ll = -max_stat * 0.1

        # Left axis: bar chart of statistics
        lags_x = np.arange(1, lags_count + 1)
        self._ax.set_xlim(0, lags_count + 1)
        self._ax.set_ylim(bar_ll, bar_ul)
        # Ref: ARMAX.m:1208-1209 — bar(stat)
        self._ax.bar(
            lags_x,
            stat,
            edgecolor="white",
            color=_TEST_BAR_COLOR,
        )

        # Right axis: scatter of scaled p-values
        # Ref: ARMAX.m:1212-1213 — plot(max(stat)*pval) on same axes
        # The MATLAB code multiplies pval by max_stat to scale onto stat axis,
        # then plots as filled square markers.
        scaled_pval = max_stat * pval
        self._ax.plot(
            lags_x,
            scaled_pval,
            linewidth=1,
            linestyle="none",
            color="black",
            marker="s",
            markersize=5,
            markerfacecolor="black",
            markeredgecolor="black",
        )

        # Ref: ARMAX.m:1214 — legend
        self._ax.legend(
            ["Statistic Value", "P-value"], loc="upper left"
        )

        # Create twinx for right-side p-value axis
        ax2 = self._ax.twinx()
        ax2.set_xlim(0, lags_count + 1)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_ylabel("P-value")

        # Ref: ARMAX.m:1217 — title
        self._ax.set_title(title_string)
        self._canvas.draw()

    # ==================================================================
    # Helper — Clear Axes and No Model Plot
    # ==================================================================
    def _clear_axes(self) -> None:
        """Clear both primary and any secondary axes.

        Ref: ARMAX.m:1274-1282 — clear_axes(handles).
        """
        # Remove any twinx axes that may exist
        for ax in self._figure.get_axes():
            if ax is not self._ax:
                ax.remove()
        self._ax.clear()

    def _no_model_run_plot(self) -> None:
        """Display 'Must estimate model first' message in empty axes.

        Ref: ARMAX.m:743-751 — no_model_run_plot.
        """
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        # Ref: ARMAX.m:749 — text message
        self._ax.text(
            0.1,
            0.5,
            "You must estimate a model before residual plots can be produced",
            fontweight="bold",
            fontsize=8,
            fontfamily="Tahoma",
            verticalalignment="center",
        )
        # Ref: ARMAX.m:751 — hide axes
        self._ax.spines["top"].set_visible(False)
        self._ax.spines["right"].set_visible(False)
        self._ax.spines["bottom"].set_visible(False)
        self._ax.spines["left"].set_visible(False)
        self._canvas.draw()

    def _apply_padding(self) -> None:
        """Apply 10% vertical padding to the current axes.

        Ref: ARMAX.m:79-82 — range = AX(4)-AX(3); AX(3)-.1*range; AX(4)+.1*range
        """
        y_lo, y_hi = self._ax.get_ylim()
        y_range = y_hi - y_lo
        if y_range > 0:
            self._ax.set_ylim(y_lo - 0.1 * y_range, y_hi + 0.1 * y_range)

    # ==================================================================
    # Validation Methods
    # ==================================================================
    def _validate_ar_ma_lags(
        self, text: str, old_lags: list[int]
    ) -> tuple[list[int], str, bool]:
        """Validate AR or MA lag specification text.

        Accepts space-separated and/or colon-separated integers (e.g.
        "1:4" → [1,2,3,4], "1 3 5" → [1,3,5], "1:3 5" → [1,2,3,5]).

        Ref: ARMAX.m:790-827 — validate_AR_MA_lags.

        Parameters
        ----------
        text : str
            The lag specification text from the edit field.
        old_lags : list[int]
            Previous lag values to revert to on failure.

        Returns
        -------
        lags : list[int]
            Sorted unique lag integers. Empty if input is empty/blank.
        out_str : str
            The validated string representation.
        ok : bool
            True if validation succeeded.
        """
        # Ref: ARMAX.m:802-807 — empty input
        stripped = text.strip()
        if not stripped:
            return [], "", True

        # Ref: ARMAX.m:811 — valid characters: digits, colon, space
        valid_chars = set("0123456789: ")
        if not all(c in valid_chars for c in stripped):
            return list(old_lags), text, False

        # Ref: ARMAX.m:815-816 — eval-like expansion of colon notation
        lags: list[int] = []
        try:
            parts = stripped.split()
            for part in parts:
                if ":" in part:
                    # Colon range expansion: "a:b" → range(a, b+1)
                    range_parts = part.split(":")
                    if len(range_parts) == 2:
                        start = int(range_parts[0])
                        end = int(range_parts[1])
                        lags.extend(range(start, end + 1))
                    elif len(range_parts) == 3:
                        # "a:step:b" syntax
                        start = int(range_parts[0])
                        step = int(range_parts[1])
                        end = int(range_parts[2])
                        lags.extend(range(start, end + 1, step))
                    else:
                        return list(old_lags), text, False
                else:
                    lags.append(int(part))
        except (ValueError, TypeError):
            return list(old_lags), text, False

        # Ref: ARMAX.m:821-826 — unique sort; empty means failure
        if lags:
            lags = sorted(set(lags))
            return lags, stripped, True
        else:
            return list(old_lags), text, False

    def _bounded_integer_validate(
        self, val_str: str, orig_val: int, lb: int, ub: int
    ) -> tuple[int, bool]:
        """Validate and clamp an integer string within bounds.

        Ref: ARMAX.m:711-729 — bounded_integer_validate.

        Parameters
        ----------
        val_str : str
            String value to validate.
        orig_val : int
            Original value to revert to on NaN.
        lb : int
            Lower bound (inclusive).
        ub : int
            Upper bound (inclusive).

        Returns
        -------
        val : int
            Validated integer value.
        ok : bool
            True if the value was valid without clamping.
        """
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            # Ref: ARMAX.m:719-720 — NaN → original value
            return orig_val, False

        if np.isnan(val):
            return orig_val, False
        elif val < lb:
            return lb, False
        elif val > ub:
            return ub, False
        elif int(val) != val:
            # Ref: ARMAX.m:725-726 — non-integer → floor
            return int(np.floor(val)), False
        else:
            return int(val), True

    # ==================================================================
    # Input Change Handlers
    # ==================================================================
    def _on_ar_lags_changed(self) -> None:
        """Handle AR lags edit field editing finished.

        Ref: ARMAX.m:524-541 — AR_lags_edit_Callback.
        """
        text = self.AR_lags_edit.text()
        lags, out_str, ok = self._validate_ar_ma_lags(text, self._ar_order)
        if ok and lags:
            self._ar_order = lags
            self._ar_order_string = out_str
            # Ref: ARMAX.m:536
            lag_str = " ".join(str(x) for x in lags)
            self._set_message(f"AR lags set successfully to {lag_str}.")
        elif ok and not lags:
            self._ar_order = []
            self._ar_order_string = ""
            # Ref: ARMAX.m:538
            self._set_message("No AR lags included.")
        else:
            # Ref: ARMAX.m:540
            self._set_message(
                "There was a problem setting the AR lags. Please try again.",
                color="red",
            )

    def _on_ma_lags_changed(self) -> None:
        """Handle MA lags edit field editing finished.

        Ref: ARMAX.m:557-574 — MA_lags_edit_Callback.
        """
        text = self.MA_lags_edit.text()
        lags, out_str, ok = self._validate_ar_ma_lags(text, self._ma_order)
        if ok and lags:
            self._ma_order = lags
            self._ma_order_string = out_str
            lag_str = " ".join(str(x) for x in lags)
            self._set_message(f"MA lags set successfully to {lag_str}.")
        elif ok and not lags:
            self._ma_order = []
            self._ma_order_string = ""
            self._set_message("No MA lags included.")
        else:
            self._set_message(
                "There was a problem setting the MA lags. Please try again.",
                color="red",
            )

    def _on_acf_lags_changed(self) -> None:
        """Handle ACF lags edit field editing finished.

        Ref: ARMAX.m:674-691 — ACF_lags_edit_Callback.
        """
        val_str = self.ACF_lags_edit.text()
        # Ref: ARMAX.m:684 — bounded_integer_validate(val, ACF_lags, 1, length(y)-1)
        val, ok = self._bounded_integer_validate(
            val_str, self._acf_lags, 1, len(self._y) - 1
        )
        if not ok:
            # Ref: ARMAX.m:686 — update text to corrected value
            self.ACF_lags_edit.setText(str(val))
        self._acf_lags = val

    def _on_constant_include(self, checked: bool) -> None:
        """Handle 'Include Constant' radio toggle.

        Ref: ARMAX.m:890-898 — constant_include_Callback.
        """
        if checked:
            self._include_constant = 1
            self._set_message("Model will include a constant.")

    def _on_constant_exclude(self, checked: bool) -> None:
        """Handle 'Exclude Constant' radio toggle.

        Ref: ARMAX.m:903-912 — constant_exclude_Callback.
        """
        if checked:
            self._include_constant = 0
            self._set_message("Model will not include a constant.")

    def _on_clear_exog(self) -> None:
        """Clear exogenous variable input.

        Ref: ARMAX.m:502-506 — EXOG_clear_pushbutton_Callback.
        """
        self.EXOG_edit.setText(" ")

    # ==================================================================
    # Model Selector
    # ==================================================================
    def _on_model_selected(self, index: int) -> None:
        """Handle model selector popup change.

        Ref: ARMAX.m:834-868 — model_selector_popup_Callback.
        Updates AR/MA lags, residuals, replots, updates message.
        """
        if index < 0 or index >= len(self._models):
            return

        model = self._models[index]
        self._current_model_number = index

        # Ref: ARMAX.m:845-846 — update AR and MA edit fields
        ar_lags = model.get("ARlags", [])
        ma_lags = model.get("MAlags", [])
        self.AR_lags_edit.setText(
            " ".join(str(x) for x in ar_lags) if ar_lags else ""
        )
        self.MA_lags_edit.setText(
            " ".join(str(x) for x in ma_lags) if ma_lags else ""
        )

        # Ref: ARMAX.m:848 — change residuals
        self._residuals = model.get("errors")

        # Ref: ARMAX.m:849-863 — plot raw residuals
        self._clear_axes()
        if self._residuals is not None:
            self._ax.plot(
                self._residuals, linewidth=2, color=_DATA_LINE_COLOR
            )
            self._ax.set_title("Plot of estimated errors")
            self._ax.autoscale(tight=True)
            self._apply_padding()
        self._canvas.draw()

        # Ref: ARMAX.m:865 — set resid_raw_radio selected
        self.resid_raw_radio.setChecked(True)

        # Ref: ARMAX.m:867 — update message
        string_id = model.get("StringID", f"Model {index + 1}")
        self._set_message(
            f"Results from model {string_id} loaded.", color="blue"
        )

    def _on_last_model(self) -> None:
        """Show the last estimated model.

        Ref: ARMAX.m:591-595 — last_model_pushbutton_Callback.
        """
        if self._models:
            self.model_selector_popup.setCurrentIndex(len(self._models) - 1)

    # ==================================================================
    # Menu Callbacks
    # ==================================================================
    def _on_hetero_robust(self) -> None:
        """Set inference method to heteroskedasticity-robust.

        Ref: ARMAX.m:762-770 — heterorobust_menu_Callback.
        """
        self._inference_method = _INFERENCE_HETEROROBUST
        self.action_heterorobust.setChecked(True)
        self.action_homoerror.setChecked(False)
        self._set_message(
            "Inference will be made using a heteroskedasticity robust "
            "covariance estimator."
        )

    def _on_homo_error(self) -> None:
        """Set inference method to homoskedastic.

        Ref: ARMAX.m:777-785 — homoerror_menu_Callback.
        """
        self._inference_method = _INFERENCE_HOMOSKEDASTIC
        self.action_homoerror.setChecked(True)
        self.action_heterorobust.setChecked(False)
        self._set_message(
            "Inference will be made assuming homoskedastic errors."
        )

    def _on_about(self) -> None:
        """Show the About dialog.

        Ref: ARMAX.m:735-739 — about_menu_Callback.
        """
        # Ref: ARMAX.m:739 — ARMAX_about('Title','About')
        ARMAXAboutDialog.show_about(parent=self, title="About")

    # ==================================================================
    # Export Callbacks
    # ==================================================================
    def _on_export_plot(self, fmt: str) -> None:
        """Export the current plot to a file.

        Ref: ARMAX.m:457-484 — export_tiff/png/eps callbacks.

        Parameters
        ----------
        fmt : str
            File format ('tiff', 'png', or 'eps').
        """
        ext_map = {"tiff": "TIFF Files (*.tiff)", "png": "PNG Files (*.png)",
                    "eps": "EPS Files (*.eps)"}
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export as {fmt.upper()}", "", ext_map.get(fmt, "")
        )
        if filepath:
            try:
                self._figure.savefig(filepath, format=fmt, dpi=150)
                self._set_message(f"Plot exported to {filepath}.")
            except Exception as exc:
                self._set_message(f"Export failed: {exc}")
                logger.exception("Plot export failed")

    def _on_copy_figure(self) -> None:
        """Copy the current figure (open in a new matplotlib window).

        Ref: ARMAX.m:599-621 — copy_figure_copy_Callback.
        Creates a new standalone figure window with the current plot.
        """
        import matplotlib.pyplot as plt

        fig_copy = plt.figure()
        fig_copy.set_size_inches(11, 8.5)
        # Copy plot data by re-rendering
        for line in self._ax.get_lines():
            fig_copy.gca().plot(
                line.get_xdata(),
                line.get_ydata(),
                linewidth=line.get_linewidth(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
            )
        fig_copy.gca().set_title(self._ax.get_title())
        plt.show()

    def _on_save_residuals(self) -> None:
        """Save residuals to a file.

        Ref: ARMAX.m:443-447 — residuals_save_residuals_Callback.
        """
        if self._residuals is None:
            self._set_message("No residuals to save — estimate a model first.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Residuals", "", "NumPy Files (*.npy);;CSV Files (*.csv)"
        )
        if filepath:
            try:
                if filepath.endswith(".csv"):
                    np.savetxt(filepath, self._residuals, delimiter=",")
                else:
                    np.save(filepath, self._residuals)
                self._set_message(f"Residuals saved to {filepath}.")
            except Exception as exc:
                self._set_message(f"Save failed: {exc}")

    def _on_save_std_residuals(self) -> None:
        """Save standardized residuals to a file.

        Ref: ARMAX.m:450-453 — residuals_save_std_residuals_Callback.
        """
        if self._residuals is None:
            self._set_message("No residuals to save — estimate a model first.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Standardized Residuals",
            "",
            "NumPy Files (*.npy);;CSV Files (*.csv)",
        )
        if filepath:
            try:
                std_resid = self._residuals / np.std(self._residuals, ddof=1)
                if filepath.endswith(".csv"):
                    np.savetxt(filepath, std_resid, delimiter=",")
                else:
                    np.save(filepath, std_resid)
                self._set_message(f"Standardized residuals saved to {filepath}.")
            except Exception as exc:
                self._set_message(f"Save failed: {exc}")

    # ==================================================================
    # Radio Button Dispatch
    # ==================================================================
    def _on_plot_radio_toggled(self, button: QRadioButton, checked: bool) -> None:
        """Dispatch plot updates when any radio button toggles.

        Routes to the correct plot method based on which button was activated.
        """
        if not checked:
            return

        dispatch = {
            self.orig_raw_radio: self._plot_raw_data,
            self.orig_acf_radio: self._plot_orig_acf,
            self.orig_pacf_radio: self._plot_orig_pacf,
            self.resid_raw_radio: self._plot_residuals,
            self.resid_fit_radio: self._plot_fit_overlay,
            self.resid_acf_radio: self._plot_resid_acf,
            self.resid_pacf_radio: self._plot_resid_pacf,
            self.radiobutton18: self._plot_lm_lb_original,
            self.radiobutton19: self._plot_lm_lb_residuals,
        }

        handler = dispatch.get(button)
        if handler is not None:
            try:
                handler()
            except Exception as exc:
                logger.exception("Plot failed: %s", exc)
                self._clear_axes()
                self._ax.set_title(f"Plot error: {exc}")
                self._canvas.draw()

    # ==================================================================
    # Private Helpers
    # ==================================================================
    def _set_message(self, msg: str, color: str = "black") -> None:
        """Update the status message label with colored text.

        Ref: ARMAX.m:308,316,318 — set message_text with ForegroundColor.

        Parameters
        ----------
        msg : str
            Message text.
        color : str
            Color name: 'black', 'blue', or 'red'.
        """
        color_map = {
            "black": "rgb(0, 0, 0)",
            "blue": "rgb(0, 0, 204)",
            "red": "rgb(230, 0, 0)",
        }
        css_color = color_map.get(color, "rgb(0, 0, 0)")
        self.message_text.setStyleSheet(f"color: {css_color};")
        self.message_text.setText(msg)

    @staticmethod
    def _build_string_id(
        ar: list[int], ma: list[int], constant: int
    ) -> str:
        """Build the ARMA model string identifier.

        Ref: ARMAX.m:265-300 — StringID construction.

        Parameters
        ----------
        ar : list[int]
            AR lag orders.
        ma : list[int]
            MA lag orders.
        constant : int
            1 if constant included, 0 otherwise.

        Returns
        -------
        str
            E.g. "ARMA(4,2) w/ constant (Irregular)".
        """
        # Ref: ARMAX.m:267-276 — check for irregular spacing
        irregular = False
        if ar:
            if len(ar) < max(ar):
                irregular = True
        if ma:
            if len(ma) < max(ma):
                irregular = True

        # Ref: ARMAX.m:278-289 — build ARMA(p,q) string
        this_str = "ARMA("
        if ar:
            this_str += str(max(ar)) + ","
        else:
            this_str += "0,"

        if ma:
            this_str += str(max(ma)) + ")"
        else:
            this_str += "0)"

        # Ref: ARMAX.m:291-295 — constant
        if constant:
            this_str += " w/ constant"
        else:
            this_str += " w/o constant"

        # Ref: ARMAX.m:297-299 — irregular
        if irregular:
            this_str += " (Irregular)"

        return this_str
