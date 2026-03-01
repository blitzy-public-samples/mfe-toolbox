"""ARMAX Result Display/Plotting Viewer — PyQt6 QWidget.

Migrated from GUI/ARMAX_viewer.m + GUI/ARMAX_viewer.fig (MATLAB GUIDE).
Provides a multi-panel viewer with:
  - Model equation display (LaTeX via matplotlib)
  - Model statistics display (LL, K, AIC, BIC, sigma^2)
  - Paginated parameter estimates table
  - Model selection combo box
  - Page navigation controls

Ref: GUI/ARMAX_viewer.m — 800 lines, Version 4.0 MFE Toolbox.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Attempt to import uic for .ui file loading (used if .ui file exists)
try:
    from PyQt6 import uic
except ImportError:  # pragma: no cover
    uic = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants — row‐height and default rows per page for the parameter table.
# These mirror the MATLAB layout computed in title_layout (lines 548‑557).
# ---------------------------------------------------------------------------
_ROW_HEIGHT: float = 0.055
_NUM_ROWS: int = max(1, math.floor((1.0 - 0.5 * _ROW_HEIGHT) / _ROW_HEIGHT))
_ROWS_PER_PAGE: int = max(1, _NUM_ROWS - 1)

# Column titles for the parameter‐estimates table.
# Ref: ARMAX_viewer.m:514
_TABLE_TITLES: list[str] = ["Parameter", "Estimate", "Std. Error", "T-stat", "P-val"]

# Normalised x‑positions for 5 table columns (matching MATLAB linspace).
_COL_X: list[float] = [0.02, 0.22, 0.42, 0.62, 0.82]


class ARMAXViewer(QWidget):
    """ARMAX result display viewer with model equation, statistics, and
    paginated parameter table.

    Ref: GUI/ARMAX_viewer.m — GUIDE viewer replacement.
    Embeds three ``matplotlib.backends.backend_qtagg.FigureCanvasQTAgg``
    instances for equation/stats/table rendering.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries, each containing at minimum:
        ``StringID``, ``parameters``, ``StdErr``, ``Tstat``, ``Pval``,
        ``ll``, ``K``, ``AIC``, ``BIC``, ``seregression``, ``ARlags``,
        ``MAlags``, ``constant``.
    display_number : int | None
        0‑based index of the model to display initially.  Defaults to the
        last model in *results*.
        Ref: ARMAX_viewer.m:60‑64 — MATLAB uses 1‑indexed DisplayNumber.
    parent : QWidget | None
        Optional parent widget.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        results: list[dict[str, Any]],
        display_number: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        # --- Store results (0‑indexed) ---
        # Ref: ARMAX_viewer.m:59 — handles.Results = varargin{1}
        self._results: list[dict[str, Any]] = list(results)

        # Ref: ARMAX_viewer.m:60‑64 — default to last model (MATLAB 1‑index)
        if display_number is not None:
            self._display_number: int = display_number
        else:
            self._display_number = max(0, len(self._results) - 1)

        # Current page number for the parameter table (1‑based, like MATLAB).
        # Ref: ARMAX_viewer.m:689 — handles.LastPage = 1
        self._current_page: int = 1
        self._total_pages: int = 1

        # Cache for computed layout info per model (keyed by index).
        # Ref: ARMAX_viewer.m:127 — Results{number}.TextFinished = true
        self._layout_cache: dict[int, dict[str, Any]] = {}

        # --- Build the widget hierarchy ---
        self._build_ui()

        # --- Create matplotlib canvases ---
        self._create_canvases()

        # --- Populate model selector ---
        # Ref: ARMAX_viewer.m:68‑73
        for result in self._results:
            string_id = result.get("StringID", "Model")
            self.model_popupmenu.addItem(str(string_id))

        # Select the initial model (Ref: line 73 — set to last model).
        if self._results:
            self.model_popupmenu.setCurrentIndex(self._display_number)

        # --- Connect signals → slots ---
        self._connect_signals()

        # --- Initial render ---
        # Ref: ARMAX_viewer.m:76 — display_model(handles, hObject)
        if self._results:
            self._display_model()

    # ------------------------------------------------------------------
    # UI construction (programmatic fallback when .ui is unavailable)
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Build the widget hierarchy programmatically.

        Attempts to load ``armax_viewer.ui`` first; falls back to code‑based
        construction if the file is missing.
        Ref: ARMAX_viewer.m:74 — set(gcf,'Position',[10 10 1000 706])
        """
        ui_path = Path(__file__).parent / "armax_viewer.ui"
        if ui_path.is_file() and uic is not None:
            uic.loadUi(str(ui_path), self)
            # Ensure fixed size even if .ui doesn't set it.
            self.setFixedSize(1000, 706)
            self.setWindowTitle("ARMAX Viewer")
            return

        # ---- Programmatic construction ----
        self.setObjectName("ARMAXViewer")
        self.setWindowTitle("ARMAX Viewer")
        self.setFixedSize(1000, 706)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(4)

        # 1. Model equation group box
        self.model_uipanel = QGroupBox("Model Equation", self)
        self.model_uipanel.setMinimumHeight(80)
        model_inner = QVBoxLayout(self.model_uipanel)
        model_inner.setContentsMargins(2, 2, 2, 2)
        self.model_display = QWidget(self.model_uipanel)
        model_inner.addWidget(self.model_display)
        main_layout.addWidget(self.model_uipanel)

        # 2. Model statistics group box
        self.model_stats_uipanel = QGroupBox("Model Statistics", self)
        self.model_stats_uipanel.setFixedHeight(100)
        stats_inner = QVBoxLayout(self.model_stats_uipanel)
        stats_inner.setContentsMargins(2, 2, 2, 2)
        self.stats_display = QWidget(self.model_stats_uipanel)
        stats_inner.addWidget(self.stats_display)
        main_layout.addWidget(self.model_stats_uipanel)

        # 3. Parameter estimates group box (takes remaining space)
        self.estimates_uipanel = QGroupBox("Parameter Estimates", self)
        estimates_inner = QVBoxLayout(self.estimates_uipanel)
        estimates_inner.setContentsMargins(2, 2, 2, 2)
        self.estimates_display = QWidget(self.estimates_uipanel)
        estimates_inner.addWidget(self.estimates_display)
        main_layout.addWidget(self.estimates_uipanel, stretch=1)

        # 4. Controls bar
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(6)

        # Model selector combo box
        # Ref: ARMAX_viewer.m:68‑73
        lbl_model = QLabel("Model:", self)
        self.model_popupmenu = QComboBox(self)
        self.model_popupmenu.setMinimumWidth(200)
        controls_layout.addWidget(lbl_model)
        controls_layout.addWidget(self.model_popupmenu)

        controls_layout.addStretch()

        # Page navigation
        # Ref: ARMAX_viewer.m:437‑471
        self.decrease_page_pushbutton = QPushButton("<", self)
        self.decrease_page_pushbutton.setFixedWidth(30)
        controls_layout.addWidget(self.decrease_page_pushbutton)

        lbl_page = QLabel("Page:", self)
        self.page_number_edit = QLineEdit("1", self)
        self.page_number_edit.setFixedWidth(40)
        self.page_number_edit.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        controls_layout.addWidget(lbl_page)
        controls_layout.addWidget(self.page_number_edit)

        lbl_of = QLabel("of", self)
        self.number_of_pages_text = QLabel("1", self)
        controls_layout.addWidget(lbl_of)
        controls_layout.addWidget(self.number_of_pages_text)

        self.increase_page_pushbutton = QPushButton(">", self)
        self.increase_page_pushbutton.setFixedWidth(30)
        controls_layout.addWidget(self.increase_page_pushbutton)

        controls_layout.addStretch()

        # Close button
        # Ref: ARMAX_viewer.m:473‑478
        self.pushbutton3 = QPushButton("Close", self)
        self.pushbutton3.setFixedWidth(80)
        controls_layout.addWidget(self.pushbutton3)

        main_layout.addLayout(controls_layout)

    # ------------------------------------------------------------------
    # Canvas creation
    # ------------------------------------------------------------------
    def _create_canvases(self) -> None:
        """Create three FigureCanvasQTAgg instances and embed them into
        their respective QGroupBox containers.

        Replaces MATLAB model_axes, stats_axes, estimates_axes.
        Ref: ARMAX_viewer.m — axes(handles.model_axes), etc.
        """
        # --- Model equation canvas ---
        self._model_fig = Figure(figsize=(9.5, 1.0), dpi=100)
        self._model_fig.patch.set_facecolor("white")
        self._model_canvas = FigureCanvasQTAgg(self._model_fig)
        _embed_canvas(self.model_display, self._model_canvas)

        # --- Statistics canvas ---
        self._stats_fig = Figure(figsize=(9.5, 0.8), dpi=100)
        self._stats_fig.patch.set_facecolor("white")
        self._stats_canvas = FigureCanvasQTAgg(self._stats_fig)
        _embed_canvas(self.stats_display, self._stats_canvas)

        # --- Parameter table canvas ---
        self._table_fig = Figure(figsize=(9.5, 4.0), dpi=100)
        self._table_fig.patch.set_facecolor("white")
        self._table_canvas = FigureCanvasQTAgg(self._table_fig)
        _embed_canvas(self.estimates_display, self._table_canvas)

    # ------------------------------------------------------------------
    # Signal → slot wiring
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        """Wire PyQt6 signals to their slot methods.

        Ref: ARMAX_viewer.m — callback assignments in gui_State and GUIDE.
        """
        # Model selector → Ref: ARMAX_viewer.m:354‑364
        self.model_popupmenu.currentIndexChanged.connect(self._on_model_selected)

        # Page number edit → Ref: ARMAX_viewer.m:380‑398
        self.page_number_edit.editingFinished.connect(self._on_page_changed)

        # Next / previous page → Ref: ARMAX_viewer.m:437‑471
        self.increase_page_pushbutton.clicked.connect(self._on_next_page)
        self.decrease_page_pushbutton.clicked.connect(self._on_prev_page)

        # Close button → Ref: ARMAX_viewer.m:473‑478
        self.pushbutton3.clicked.connect(self.close)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def close(self) -> bool:
        """Close the viewer widget.

        Ref: ARMAX_viewer.m:478 — delete(handles.figure1)
        """
        return super().close()

    # ------------------------------------------------------------------
    # Core orchestration
    # ------------------------------------------------------------------
    def _display_model(self) -> None:
        """Orchestrate the rendering of the currently selected model.

        If layout data has already been computed for this model, skip
        straight to rendering; otherwise compute layout first.
        Ref: ARMAX_viewer.m:755‑769 — display_model
        """
        idx = self._display_number
        if idx < 0 or idx >= len(self._results):
            return

        result = self._results[idx]

        # Compute layout data if not cached.
        # Ref: ARMAX_viewer.m:759 — isfield(results, 'TextFinished')
        if idx not in self._layout_cache:
            param_strings, var_strings = self._generate_strings(result)
            self._layout_cache[idx] = {
                "ParameterString": param_strings,
                "VariableString": var_strings,
            }

        layout = self._layout_cache[idx]

        # Reset page to 1 for newly selected model.
        self._current_page = 1

        # Render all three panels.
        self._render_model_equation(result)
        self._render_model_stats(result)

        # Compute total pages before rendering the table.
        num_params = self._param_count(layout)
        self._total_pages = max(1, math.ceil(num_params / _ROWS_PER_PAGE))

        self._render_parameter_table(result)
        self._update_pagination_ui()

    # ------------------------------------------------------------------
    # String generation
    # ------------------------------------------------------------------
    def _generate_strings(
        self, result: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Build LaTeX parameter and variable strings for the ARMA equation.

        Returns
        -------
        parameter_strings : list[str]
            LaTeX fragments for each term's coefficient.
        variable_strings : list[str]
            LaTeX fragments for each term's variable.

        Ref: ARMAX_viewer.m:132‑170 — generate_strings
        """
        param_strs: list[str] = []
        var_strs: list[str] = []

        # Index 0: equation prefix  (Ref: line 137)
        param_strs.append("y_t =")
        var_strs.append("")

        # Constant term  (Ref: lines 141‑145)
        if result.get("constant", False):
            param_strs.append("\\phi_0")
            var_strs.append("")

        # AR terms  (Ref: lines 147‑155)
        ar_lags = result.get("ARlags", np.array([]))
        if ar_lags is not None and len(ar_lags) > 0:
            for lag in ar_lags:
                lag_int = int(lag)
                # Ref: ARMAX_viewer.m:150
                param_strs.append(f"\\phi_{{{lag_int}}}")
                var_strs.append(f"y_{{t-{lag_int}}}")

        # MA terms  (Ref: lines 158‑165)
        ma_lags = result.get("MAlags", np.array([]))
        if ma_lags is not None and len(ma_lags) > 0:
            for lag in ma_lags:
                lag_int = int(lag)
                # Ref: ARMAX_viewer.m:161
                param_strs.append(f"\\theta_{{{lag_int}}}")
                var_strs.append(f"\\epsilon_{{t-{lag_int}}}")

        # Error term suffix  (Ref: lines 168‑169)
        param_strs.append("")
        var_strs.append("\\epsilon_t")

        return param_strs, var_strs

    # ------------------------------------------------------------------
    # Model equation rendering
    # ------------------------------------------------------------------
    def _render_model_equation(self, result: dict[str, Any]) -> None:
        """Render the model equation using matplotlib text with LaTeX.

        Ref: ARMAX_viewer.m:174‑268 — model_layout
        Ref: ARMAX_viewer.m:627‑653 — draw_data (equation portion)
        """
        fig = self._model_fig
        fig.clear()

        idx = self._display_number
        layout = self._layout_cache.get(idx, {})
        param_strs = layout.get("ParameterString", [])
        var_strs = layout.get("VariableString", [])

        if not param_strs:
            self._model_canvas.draw()
            return

        # Build display lines, wrapping if too long.
        # Ref: ARMAX_viewer.m:182‑210
        model_string = result.get("StringID", "Model")

        # First line: the equation prefix and initial terms.
        lines: list[str] = [str(model_string)]

        # Start building the equation string.
        # Ref: ARMAX_viewer.m:189‑193
        if result.get("constant", False) and len(param_strs) > 2:
            eq_line = f"{param_strs[0]} {param_strs[1]}"
        elif len(param_strs) > 2:
            eq_line = f"{param_strs[0]} {param_strs[1]}{var_strs[1]}"
        else:
            eq_line = param_strs[0] if param_strs else ""

        # Append remaining terms with "+" separators.
        # Ref: ARMAX_viewer.m:197‑210
        # In MATLAB, line wrapping is based on text extent.  Here we use a
        # character‑count heuristic (~90 chars before wrapping).
        max_chars = 90
        for k in range(2, len(param_strs)):
            term = f"+{param_strs[k]}{var_strs[k]}" if param_strs[k] else var_strs[k]
            if len(eq_line) + len(term) > max_chars:
                lines.append(eq_line)
                eq_line = term
            else:
                eq_line += term
        if eq_line:
            lines.append(eq_line)

        # Draw text lines on the figure.
        # Ref: ARMAX_viewer.m:214‑226
        n_lines = len(lines)
        positions = [(i + 1) / (n_lines + 1) for i in range(n_lines)]
        positions.reverse()  # top‑to‑bottom

        for i, (line_text, ypos) in enumerate(zip(lines, positions)):
            xpos = 0.02 if i < 2 else 0.07
            if i == 0:
                # Title line — plain text (Ref: line 223)
                fig.text(
                    xpos, ypos, line_text,
                    fontsize=12, fontfamily="sans-serif",
                    verticalalignment="center",
                )
            else:
                # Equation lines — LaTeX (Ref: line 221)
                fig.text(
                    xpos, ypos, f"${line_text}$",
                    fontsize=12, usetex=False,
                    verticalalignment="center",
                )

        self._model_canvas.draw()

    # ------------------------------------------------------------------
    # Statistics rendering
    # ------------------------------------------------------------------
    def _render_model_stats(self, result: dict[str, Any]) -> None:
        """Render the five key model statistics.

        Displays: Log‑likelihood, No. of Params, AIC, BIC, σ².
        Ref: ARMAX_viewer.m:275‑351 — model_stats_layout
        Ref: ARMAX_viewer.m:654‑685 — draw_data (stats portion)
        """
        fig = self._stats_fig
        fig.clear()

        # Titles and formatted data values.
        # Ref: ARMAX_viewer.m:284
        titles = [
            "Log-likelihood",
            "No. of Params",
            "Akaike IC",
            "Schwartz/Bayesian IC",
            r"$\hat{\sigma}^2$",
        ]

        # Ref: ARMAX_viewer.m:308‑312 — sprintf('%0.7g', ...)
        ll_val = result.get("ll", 0.0)
        k_val = result.get("K", 0)
        aic_val = result.get("AIC", 0.0)
        bic_val = result.get("BIC", 0.0)
        se_val = result.get("seregression", 0.0)

        data = [
            f"{ll_val:.7g}",
            str(int(k_val)),
            f"{aic_val:.7g}",
            f"{bic_val:.7g}",
            f"{se_val:.7g}",
        ]

        # Evenly spaced positions across the figure width.
        # Ref: ARMAX_viewer.m:285 — positions = linspace(.02,.9,5)
        x_positions = [0.02 + i * 0.22 for i in range(5)]

        for i, (title, datum, xp) in enumerate(
            zip(titles, data, x_positions)
        ):
            # Title (upper row) — Ref: lines 287‑297
            fig.text(
                xp, 0.72, title,
                fontsize=10, fontfamily="sans-serif",
                verticalalignment="top",
            )

            # Underline separator — Ref: lines 324‑331
            # Use a thin horizontal line via an axes trick.
            # We draw a short line below the title.
            ax = fig.add_axes([xp, 0.48, 0.18, 0.005], frameon=False)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color="black", linewidth=0.8)
            ax.set_xticks([])
            ax.set_yticks([])

            # Data value (lower row, right‑aligned under title).
            # Ref: ARMAX_viewer.m:316‑321 — right‑aligned data
            fig.text(
                xp + 0.18, 0.20, datum,
                fontsize=10, fontfamily="sans-serif",
                horizontalalignment="right",
                verticalalignment="bottom",
            )

        self._stats_canvas.draw()

    # ------------------------------------------------------------------
    # Parameter table rendering
    # ------------------------------------------------------------------
    def _render_parameter_table(
        self, result: dict[str, Any], page: int | None = None
    ) -> None:
        """Render the paginated parameter estimates table.

        Ref: ARMAX_viewer.m:714‑751 — draw_parameter_table
        Ref: ARMAX_viewer.m:513‑560 — title_layout
        Ref: ARMAX_viewer.m:563‑625 — compute_text_positions
        """
        if page is not None:
            self._current_page = page

        fig = self._table_fig
        fig.clear()

        idx = self._display_number
        layout = self._layout_cache.get(idx, {})
        param_strs = layout.get("ParameterString", [])

        # Extract parameter arrays from result.
        parameters = np.asarray(result.get("parameters", []))
        std_errs = np.asarray(result.get("StdErr", []))
        tstats = np.asarray(result.get("Tstat", []))
        pvals = np.asarray(result.get("Pval", []))

        # Number of displayable parameters (excluding "y_t =" prefix and
        # "\epsilon_t" suffix).
        # Ref: ARMAX_viewer.m:569 — for i=2:length(str)-1
        num_params = self._param_count(layout)
        if num_params == 0:
            self._table_canvas.draw()
            return

        rows_per_page = _ROWS_PER_PAGE
        total_pages = max(1, math.ceil(num_params / rows_per_page))
        self._total_pages = total_pages

        cur_page = max(1, min(self._current_page, total_pages))
        self._current_page = cur_page

        # --- Column titles ---
        # Ref: ARMAX_viewer.m:514 — titles = {'Parameter','Estimate',...}
        title_y = 0.95
        for col_idx, title in enumerate(_TABLE_TITLES):
            fig.text(
                _COL_X[col_idx], title_y, title,
                fontsize=10, fontfamily="sans-serif",
                verticalalignment="top", fontweight="bold",
            )

        # Horizontal separator line below titles.
        # Ref: ARMAX_viewer.m:534‑544 — plot underlines
        sep_ax = fig.add_axes([0.02, 0.90, 0.96, 0.003], frameon=False)
        sep_ax.set_xlim(0, 1)
        sep_ax.set_ylim(0, 1)
        sep_ax.axhline(y=0.5, color="black", linewidth=0.8)
        sep_ax.set_xticks([])
        sep_ax.set_yticks([])

        # --- Data rows for the current page ---
        # Ref: ARMAX_viewer.m:563‑621 — compute_text_positions
        start_idx = (cur_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, num_params)

        for row_offset in range(end_idx - start_idx):
            param_idx = start_idx + row_offset
            # Ref: ARMAX_viewer.m:569 — i goes from 2 to length(str)-1
            # param_strs[0] = "y_t =", param_strs[1..N-2] = params,
            # param_strs[N-1] = "" (error term).
            # param_idx 0 → param_strs[1], parameter values at index 0.
            str_idx = param_idx + 1  # offset into param_strs

            # y position for this row (top → bottom within data area).
            # Ref: ARMAX_viewer.m:611 — y_pos = 1 - actual_position * row_height
            actual_position = row_offset + 2  # +2 for title row offset
            y_pos = 0.88 - actual_position * _ROW_HEIGHT

            # Column 1: parameter name (LaTeX).
            # Ref: ARMAX_viewer.m:582 — ['$' str{i} '$']
            if str_idx < len(param_strs):
                latex_name = f"${param_strs[str_idx]}$"
            else:
                latex_name = ""

            fig.text(
                _COL_X[0], y_pos, latex_name,
                fontsize=11, usetex=False,
                verticalalignment="center",
            )

            # Columns 2‑5: numeric values.
            # Ref: ARMAX_viewer.m:583‑601 — format selection
            for col, arr in enumerate(
                [parameters, std_errs, tstats, pvals], start=1
            ):
                if param_idx < len(arr):
                    val = float(arr[param_idx])
                    formatted = self._format_value(val)
                else:
                    formatted = ""

                fig.text(
                    _COL_X[col], y_pos, formatted,
                    fontsize=10, fontfamily="sans-serif",
                    verticalalignment="center",
                )

        self._table_canvas.draw()

    # ------------------------------------------------------------------
    # Slot: model selection changed
    # ------------------------------------------------------------------
    def _on_model_selected(self, index: int) -> None:
        """Handle model selector combo box changes.

        Ref: ARMAX_viewer.m:354‑364 — model_popupmenu_Callback
        """
        if index < 0 or index >= len(self._results):
            return
        # Ref: ARMAX_viewer.m:362 — handles.DisplayNumber = get(hObject,'Value')
        self._display_number = index
        self._display_model()

    # ------------------------------------------------------------------
    # Slot: page number edited
    # ------------------------------------------------------------------
    def _on_page_changed(self) -> None:
        """Validate and apply manual page number input.

        Ref: ARMAX_viewer.m:380‑398 — page_number_edit_Callback
        """
        val_str = self.page_number_edit.text()
        new_page, ok = self._bounded_integer_validate(
            val_str, self._current_page, 1, self._total_pages
        )
        if not ok:
            # Ref: ARMAX_viewer.m:393 — set(hObject,'String',num2str(CurrentPage))
            self.page_number_edit.setText(str(new_page))

        self._current_page = new_page
        idx = self._display_number
        if 0 <= idx < len(self._results):
            self._render_parameter_table(self._results[idx])
        self._update_pagination_ui()

    # ------------------------------------------------------------------
    # Slot: next page
    # ------------------------------------------------------------------
    def _on_next_page(self) -> None:
        """Advance to the next parameter table page.

        Ref: ARMAX_viewer.m:437‑454 — increase_page_pushbutton_Callback
        """
        if self._current_page < self._total_pages:
            self._current_page += 1
            # Ref: ARMAX_viewer.m:447 — set(page_number_edit,'String',num2str(CurrentPage))
            self.page_number_edit.setText(str(self._current_page))
            idx = self._display_number
            if 0 <= idx < len(self._results):
                self._render_parameter_table(self._results[idx])
            self._update_pagination_ui()

    # ------------------------------------------------------------------
    # Slot: previous page
    # ------------------------------------------------------------------
    def _on_prev_page(self) -> None:
        """Go back to the previous parameter table page.

        Ref: ARMAX_viewer.m:456‑471 — decrease_page_pushbutton_Callback
        """
        if self._current_page > 1:
            self._current_page -= 1
            # Ref: ARMAX_viewer.m:465
            self.page_number_edit.setText(str(self._current_page))
            idx = self._display_number
            if 0 <= idx < len(self._results):
                self._render_parameter_table(self._results[idx])
            self._update_pagination_ui()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    @staticmethod
    def _bounded_integer_validate(
        val_str: str,
        orig_val: int,
        lb: int,
        ub: int,
    ) -> tuple[int, bool]:
        """Validate and clamp an integer input string.

        Returns
        -------
        val : int
            The validated integer value, clamped to [lb, ub].
        ok : bool
            ``True`` if the input required no correction.

        Ref: ARMAX_viewer.m:782‑800 — bounded_integer_validate
        """
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            # Ref: ARMAX_viewer.m:790‑791 — NaN → orig_val
            return orig_val, False

        if math.isnan(val):
            # Ref: ARMAX_viewer.m:790‑791
            return orig_val, False
        if val < lb:
            # Ref: ARMAX_viewer.m:792‑793
            return lb, False
        if val > ub:
            # Ref: ARMAX_viewer.m:794‑795
            return ub, False
        if math.floor(val) != val:
            # Ref: ARMAX_viewer.m:796‑797 — non‑integer → floor
            return int(math.floor(val)), False

        return int(val), True

    # ------------------------------------------------------------------
    # Pagination UI update
    # ------------------------------------------------------------------
    def _update_pagination_ui(self) -> None:
        """Update page navigation widgets based on current state.

        Ref: ARMAX_viewer.m:694‑707 — pagination control updates in draw_data
        """
        self.page_number_edit.setText(str(self._current_page))
        self.number_of_pages_text.setText(str(self._total_pages))

        if self._total_pages <= 1:
            # Single page — disable navigation.
            # Ref: ARMAX_viewer.m:696‑701
            self.page_number_edit.setEnabled(False)
            self.page_number_edit.setStyleSheet("background-color: #808080;")
            self.increase_page_pushbutton.setEnabled(False)
            self.decrease_page_pushbutton.setEnabled(False)
        else:
            # Multi‑page — enable navigation.
            # Ref: ARMAX_viewer.m:703‑706
            self.page_number_edit.setEnabled(True)
            self.page_number_edit.setStyleSheet("background-color: #FFFFFF;")
            self.increase_page_pushbutton.setEnabled(True)
            self.decrease_page_pushbutton.setEnabled(True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _param_count(layout: dict[str, Any]) -> int:
        """Return the number of displayable parameters in the layout.

        Excludes the equation prefix (``"y_t ="``) and the error‑term
        suffix (``"\\epsilon_t"``).
        Ref: ARMAX_viewer.m:569 — loop from 2 to length(str)-1
        """
        param_strs = layout.get("ParameterString", [])
        return max(0, len(param_strs) - 2)

    @staticmethod
    def _format_value(val: float) -> str:
        """Format a numeric value for table display.

        MATLAB uses ``temp < .000001`` (raw comparison, NOT abs), so all
        negative values and very‑small positives get the compact format.
        Ref: ARMAX_viewer.m:596‑601
        """
        if numpy_isnan_safe(val):
            return ""
        # Ref: ARMAX_viewer.m:597 — if temp<.000001
        if val < 0.000001:
            # Ref: ARMAX_viewer.m:598 — sprintf('%0.3g', temp)
            return f"{val:.3g}"
        # Ref: ARMAX_viewer.m:600 — sprintf('%0.5f', temp)
        return f"{val:.5f}"


# -----------------------------------------------------------------------
# Module‑level helper utilities
# -----------------------------------------------------------------------

def _embed_canvas(placeholder: QWidget, canvas: FigureCanvasQTAgg) -> None:
    """Replace the contents of *placeholder* with *canvas*.

    Creates a tight QVBoxLayout if one does not already exist.
    """
    layout = placeholder.layout()
    if layout is None:
        layout = QVBoxLayout(placeholder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
    layout.addWidget(canvas)


def numpy_isnan_safe(val: float) -> bool:
    """Check for NaN using both :func:`math.isnan` and :func:`numpy.isnan`.

    Falls back gracefully if *val* cannot be checked.
    """
    try:
        if math.isnan(val):
            return True
    except (TypeError, ValueError):
        pass
    try:
        if np.isnan(val):
            return True
    except (TypeError, ValueError):
        pass
    return False
