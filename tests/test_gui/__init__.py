"""PyQt6 GUI integration tests for MFE Toolbox ARMAX interface.

Uses pytest-qt for widget interaction and signal verification.
Tests cover all 4 PyQt6 GUI widget modules:
- test_armax_main: QMainWindow with embedded FigureCanvasQTAgg for ARMAX estimation
- test_armax_about: QDialog with Oxford logo display
- test_armax_close_dialog: QDialog with Yes/No/Cancel confirmation
- test_armax_viewer: QWidget with FigureCanvasQTAgg for result display

Coverage threshold for GUI modules is ≥80% (vs ≥90% for non-GUI).
Plot output compared via saved .png snapshots with perceptual diff tolerance SSIM ≥ 0.95.

Per AAP Section 0.7.1: All GUIDE callbacks MUST be exercised in integration tests using pytest-qt.
"""
