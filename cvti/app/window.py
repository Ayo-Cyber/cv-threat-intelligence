"""Main application window."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy,
    QSplitter, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

from cvti.app.widgets.alerts import AlertsPanel
from cvti.app.widgets.config import ConfigPanel
from cvti.app.widgets.feed import FeedWidget
from cvti.app.widgets.mapper import MapperPanel
from cvti.app.worker import DetectionWorker

DARK = """
QMainWindow, QWidget { background:#141414; color:#eee; }
QTabWidget::pane  { border:none; }
QTabBar::tab      { background:#1e1e1e; color:#aaa; padding:7px 18px; border-radius:4px 4px 0 0; }
QTabBar::tab:selected { background:#2a2a2a; color:#fff; }
QLineEdit, QComboBox { background:#1e1e1e; border:1px solid #333; border-radius:4px; padding:4px 8px; color:#eee; }
QPushButton { background:#1e1e1e; border:1px solid #333; border-radius:4px; padding:5px 14px; color:#eee; }
QPushButton:hover  { background:#2a2a2a; }
QPushButton:disabled { color:#555; }
QStatusBar { background:#0e0e0e; color:#666; font-size:11px; }
QSplitter::handle { background:#222; }
"""

START_STYLE = "background:#1a6b3c; color:white; font-weight:bold; padding:5px 18px; border-radius:4px; border:none;"
STOP_STYLE  = "background:#6b1a1a; color:white; font-weight:bold; padding:5px 18px; border-radius:4px; border:none;"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CV Threat Intelligence")
        self.setMinimumSize(1200, 720)
        self.setStyleSheet(DARK)

        self._worker: DetectionWorker | None = None
        self._scene_context: dict | None = None

        self._build_ui()
        self._status("Ready. Configure source and press Start.")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(8)

        root.addLayout(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        # Left — live feed
        self.feed = FeedWidget()
        self.feed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        splitter.addWidget(self.feed)

        # Right — tabbed panels
        tabs = QTabWidget()
        tabs.setMinimumWidth(380)
        tabs.setMaximumWidth(480)

        self.alerts_panel = AlertsPanel()
        self.mapper_panel = MapperPanel()
        self.config_panel = ConfigPanel()

        tabs.addTab(self.alerts_panel, "Alerts")
        tabs.addTab(self.mapper_panel, "Agent Map")
        tabs.addTab(self.config_panel, "Rules")

        splitter.addWidget(tabs)
        splitter.setSizes([780, 400])
        root.addWidget(splitter)

        # Wire mapper → worker scene context
        self.mapper_panel.scene_context_updated.connect(self._on_scene_context)
        self.config_panel.config_changed.connect(self._on_config_changed)

        self.setStatusBar(QStatusBar())

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        # Source
        bar.addWidget(QLabel("Source:"))
        self.source_input = QLineEdit("0")
        self.source_input.setPlaceholderText("0  /  rtsp://…  /  video.mp4")
        self.source_input.setFixedWidth(260)
        bar.addWidget(self.source_input)

        # Gate provider
        bar.addWidget(QLabel("Gate:"))
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["mock", "anthropic"])
        self.gate_combo.setFixedWidth(110)
        bar.addWidget(self.gate_combo)

        bar.addStretch()

        # FPS display
        self.fps_label = QLabel("FPS: —")
        self.fps_label.setStyleSheet("color:#888; min-width:70px;")
        bar.addWidget(self.fps_label)

        # Start / Stop
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setStyleSheet(START_STYLE)
        self.start_btn.clicked.connect(self._start)
        bar.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setStyleSheet(STOP_STYLE)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        bar.addWidget(self.stop_btn)

        return bar

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _start(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        source      = self.source_input.text().strip() or "0"
        config_path = self.config_panel.current_config_path()
        zones_path  = self.config_panel.current_zones_path()
        gate        = self.gate_combo.currentText()

        if not config_path:
            self._status("No rule config selected. Pick one in the Rules tab.")
            return

        self._worker = DetectionWorker(
            source=source,
            config_path=config_path,
            zones_path=zones_path,
            gate_provider=gate,
            scene_context=self._scene_context,
        )
        self._worker.frame_ready.connect(self.feed.update_frame)
        self._worker.alert_fired.connect(self.alerts_panel.add_alert)
        self._worker.status_update.connect(self._status)
        self._worker.fps_update.connect(lambda fps: self.fps_label.setText(f"FPS: {fps:.1f}"))
        self._worker.finished.connect(self._on_worker_done)

        self._worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._status("Starting…")

    def _stop(self) -> None:
        if self._worker:
            self._worker.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_worker_done(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_scene_context(self, context: dict) -> None:
        self._scene_context = context
        env = context.get("environment_type", "unknown")
        self._status(f"Scene context updated — environment: {env}")

    def _on_config_changed(self, path: str) -> None:
        if self._worker and self._worker.isRunning():
            self._status(f"Config saved. Restart to apply: {path}")
        else:
            self._status(f"Config ready: {path}")

    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def closeEvent(self, event) -> None:
        self._stop()
        super().closeEvent(event)
