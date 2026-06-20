"""Customization panel — load and edit rule configs."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from cvti.utils import writable_configs_dir


class ConfigPanel(QWidget):
    config_changed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Resolve writable dir once (seeds from bundle on first run)
        self._configs_dir = writable_configs_dir()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Customization — Rules")
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        # Config selector
        sel_row = QHBoxLayout()
        self.config_combo = QComboBox()
        self._populate_configs()
        self.config_combo.currentTextChanged.connect(self._load_config)
        reload_btn = QPushButton("Reload")
        reload_btn.setFixedWidth(70)
        reload_btn.clicked.connect(lambda: self._load_config(self.config_combo.currentText()))
        sel_row.addWidget(QLabel("Config:"))
        sel_row.addWidget(self.config_combo, 1)
        sel_row.addWidget(reload_btn)
        layout.addLayout(sel_row)

        # Zones selector
        zones_row = QHBoxLayout()
        self.zones_combo = QComboBox()
        self._populate_zones()
        zones_row.addWidget(QLabel("Zones: "))
        zones_row.addWidget(self.zones_combo, 1)
        layout.addLayout(zones_row)

        # JSON editor
        edit_label = QLabel("Rule JSON  (edit and Apply to activate)")
        edit_label.setStyleSheet("font-weight:bold; margin-top:6px;")
        layout.addWidget(edit_label)

        self.editor = QPlainTextEdit()
        self.editor.setStyleSheet(
            "background:#1a1a1a; color:#ccc; border:none; "
            "border-radius:4px; font-family:monospace; font-size:11px;"
        )
        layout.addWidget(self.editor)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply Config")
        apply_btn.setStyleSheet(
            "background:#1a6b3c; color:white; font-weight:bold; "
            "padding:6px 16px; border-radius:4px;"
        )
        apply_btn.clicked.connect(self._apply)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#888; font-size:11px;")
        btn_row.addStretch()
        btn_row.addWidget(self.status_label)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

        if self.config_combo.count() > 0:
            self._load_config(self.config_combo.currentText())

    # ------------------------------------------------------------------

    def _populate_configs(self) -> None:
        self.config_combo.clear()
        for p in sorted(self._configs_dir.glob("*_v*.json")):
            if "zones" not in p.name and "example" not in p.name:
                self.config_combo.addItem(p.name, str(p))

    def _populate_zones(self) -> None:
        self.zones_combo.clear()
        self.zones_combo.addItem("(none)", "")
        for p in sorted(self._configs_dir.glob("*zones*.json")):
            if "example" not in p.name and "rules" not in p.name:
                self.zones_combo.addItem(p.name, str(p))

    def _load_config(self, name: str) -> None:
        path = self.config_combo.currentData() or str(self._configs_dir / name)
        try:
            self.editor.setPlainText(Path(path).read_text())
            self.status_label.setText(f"Loaded {name}")
        except Exception as exc:
            self.status_label.setText(f"Error loading: {exc}")

    def _apply(self) -> None:
        try:
            json.loads(self.editor.toPlainText())
        except json.JSONDecodeError as exc:
            QMessageBox.warning(self, "Invalid JSON", str(exc))
            return
        path = self.config_combo.currentData()
        if not path:
            return
        try:
            Path(path).write_text(self.editor.toPlainText())
            self.status_label.setText("Saved.")
            self.config_changed.emit(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def current_config_path(self) -> str:
        return self.config_combo.currentData() or ""

    def current_zones_path(self) -> str:
        return self.zones_combo.currentData() or ""
