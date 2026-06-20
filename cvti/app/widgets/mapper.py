"""Agent Mapper panel — run scene context VLM and display results."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)


class MapperPanel(QWidget):
    scene_context_updated = pyqtSignal(dict)   # emitted when mapper finishes

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Agent Mapper — Scene Context")
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        # Source input row
        src_row = QHBoxLayout()
        src_label = QLabel("Source:")
        src_label.setFixedWidth(55)
        self.src_input = QComboBox()
        self.src_input.setEditable(True)
        self.src_input.addItems(["0", "data/test_clips/theft_shop_01.mp4"])
        self.run_btn = QPushButton("Run Mapper")
        self.run_btn.clicked.connect(self._run_mapper)
        src_row.addWidget(src_label)
        src_row.addWidget(self.src_input, 1)
        src_row.addWidget(self.run_btn)
        layout.addLayout(src_row)

        # Provider row
        prov_row = QHBoxLayout()
        prov_label = QLabel("Provider:")
        prov_label.setFixedWidth(55)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["mock", "anthropic", "ollama"])
        prov_row.addWidget(prov_label)
        prov_row.addWidget(self.provider_combo)
        prov_row.addStretch()
        layout.addLayout(prov_row)

        # Status
        self.status_label = QLabel("Not run yet.")
        self.status_label.setStyleSheet("color:#888; font-size:11px;")
        layout.addWidget(self.status_label)

        # Output display
        output_label = QLabel("Scene Context JSON")
        output_label.setStyleSheet("font-weight:bold; margin-top:6px;")
        layout.addWidget(output_label)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("background:#1a1a1a; color:#ccc; border:none; border-radius:4px; font-family:monospace; font-size:11px;")
        self.output.setPlaceholderText("Scene context will appear here after running the mapper…")
        layout.addWidget(self.output)

        # Environment summary row
        self.env_label = QLabel("")
        self.env_label.setStyleSheet("color:#44aaff; font-size:12px; font-weight:bold;")
        layout.addWidget(self.env_label)

        self._context: dict | None = None

    # ------------------------------------------------------------------

    def _run_mapper(self) -> None:
        self.run_btn.setEnabled(False)
        self.status_label.setText("Running agent mapper…")
        source   = self.src_input.currentText().strip()
        provider = self.provider_combo.currentText()
        threading.Thread(target=self._mapper_thread, args=(source, provider), daemon=True).start()

    def _mapper_thread(self, source: str, provider: str) -> None:
        try:
            from cvti.scene.agent_mapper import AgentMapper
            mapper  = AgentMapper(provider=provider)
            context = mapper.map(source)
            self._context = context
            pretty = json.dumps(context, indent=2)
            env    = context.get("environment_type", "unknown")
            self._set_output(pretty, env, None)
            self.scene_context_updated.emit(context)
        except Exception as exc:
            self._set_output("", "", str(exc))

    def _set_output(self, text: str, env: str, error: str | None) -> None:
        from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
        # Schedule on main thread
        if error:
            self.output.setPlainText(f"Error: {error}")
            self.status_label.setText(f"Failed: {error}")
            self.env_label.setText("")
        else:
            self.output.setPlainText(text)
            self.status_label.setText("Done.")
            self.env_label.setText(f"Environment: {env}")
        self.run_btn.setEnabled(True)

    def current_context(self) -> dict | None:
        return self._context
