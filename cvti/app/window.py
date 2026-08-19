"""Main application window."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy,
    QSplitter, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

from cvti.app.widgets.alerts import AlertsPanel
from cvti.app.widgets.cameras import CamerasPanel
from cvti.app.widgets.config import ConfigPanel
from cvti.app.widgets.feed import FeedWidget
from cvti.app.widgets.mapper import MapperPanel
from cvti.app.worker import DetectionWorker
from cvti.verification import ollama

from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Local VLM models offered for the offline gate. Gemma 3 first (client default);
# the QAT build keeps BF16-level quality at the same ~3.3 GB as plain Q4, so it leads.
# moondream is the low-RAM fallback (~2 GB, weaker quality); the rest are heavier.
LOCAL_MODELS = ["gemma3:4b-it-qat", "gemma3:4b", "moondream", "qwen2.5vl:7b", "llama3.2-vision"]


class ModelPullWorker(QThread):
    """Pulls an Ollama model in the background, streaming status lines."""
    progress = pyqtSignal(str)
    done     = pyqtSignal(bool, str)   # (success, message)

    def __init__(self, model: str, parent=None) -> None:
        super().__init__(parent)
        self.model = model

    def run(self) -> None:
        try:
            for line in ollama.pull_model(self.model, on_progress=None):
                self.progress.emit(f"Downloading {self.model}: {line}")
            self.done.emit(True, f"{self.model} ready.")
        except Exception as exc:  # noqa: BLE001 - surface any pull failure to the UI
            log.error("worker task failed", exc_info=True)
            self.done.emit(False, str(exc))

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

        self.cameras_panel = CamerasPanel()
        self.alerts_panel = AlertsPanel()
        self.mapper_panel = MapperPanel()
        self.config_panel = ConfigPanel()

        tabs.addTab(self.cameras_panel, "Cameras")
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
        self.gate_combo.addItems(["mock", "local", "anthropic"])
        self.gate_combo.setFixedWidth(100)
        self.gate_combo.currentTextChanged.connect(self._on_provider_changed)
        bar.addWidget(self.gate_combo)

        # VLM model (used by local / anthropic)
        bar.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setFixedWidth(150)
        bar.addWidget(self.model_combo)
        self._on_provider_changed(self.gate_combo.currentText())

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

    def _on_provider_changed(self, provider: str) -> None:
        """Repopulate the model list and enable it only when the provider uses one."""
        self.model_combo.clear()
        if provider == "local":
            self.model_combo.addItems(LOCAL_MODELS)
            self.model_combo.setEnabled(True)
        elif provider == "anthropic":
            self.model_combo.addItems(["claude-sonnet-4-6"])
            self.model_combo.setEnabled(True)
        else:  # mock
            self.model_combo.setEnabled(False)

    def _start(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        source      = self.source_input.text().strip() or "0"
        config_path = self.config_panel.current_config_path()
        zones_path  = self.config_panel.current_zones_path()
        gate        = self.gate_combo.currentText()
        model       = self.model_combo.currentText().strip() if self.model_combo.isEnabled() else ""

        if not config_path:
            self._status("No rule config selected. Pick one in the Rules tab.")
            return

        # Local gate needs an Ollama server with the chosen model pulled.
        if gate == "local" and not self._ensure_local_ready(model):
            return

        self._worker = DetectionWorker(
            source=source,
            config_path=config_path,
            zones_path=zones_path,
            gate_provider=gate,
            gate_model=model,
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

    def _ensure_local_ready(self, model: str) -> bool:
        """Ensure Ollama is up with `model` pulled. Returns True if ready to start now.

        If the model needs downloading, kicks off a background pull and returns False;
        the run auto-starts when the pull finishes.
        """
        import time

        if not model:
            self._status("Pick a local model first.")
            return False

        if not ollama.server_up():
            self._status("Starting local VLM server…")
            if ollama.start_server():
                for _ in range(10):
                    time.sleep(0.5)
                    if ollama.server_up():
                        break
        if not ollama.server_up():
            QMessageBox.warning(
                self, "Ollama not running",
                "The local VLM needs Ollama. Install it from https://ollama.com, "
                "then run `ollama serve` and try again.",
            )
            self._status("Local VLM unavailable — Ollama not running.")
            return False

        if ollama.has_model(model):
            return True

        # Model absent — offer to download it once.
        if QMessageBox.question(
            self, "Download model?",
            f"The model '{model}' isn't installed yet. Download it now?",
        ) != QMessageBox.StandardButton.Yes:
            self._status("Local model not downloaded — cannot start.")
            return False

        self.start_btn.setEnabled(False)
        self._pull_worker = ModelPullWorker(model)
        self._pull_worker.progress.connect(self._status)
        self._pull_worker.done.connect(self._on_pull_done)
        self._pull_worker.start()
        return False

    def _on_pull_done(self, ok: bool, message: str) -> None:
        self.start_btn.setEnabled(True)
        self._status(message)
        if ok:
            self._start()  # model is now present; this pass will proceed to run

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
