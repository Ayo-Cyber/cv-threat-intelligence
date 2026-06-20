"""Alert log panel."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

PRIORITY_COLORS = {
    "high":   "#ff4444",
    "medium": "#ff9900",
    "low":    "#44aaff",
}


class AlertsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        title = QLabel("Alert Log")
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._clear)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(clear_btn)
        layout.addLayout(header)

        self.list = QListWidget()
        self.list.setStyleSheet("""
            QListWidget { background:#1a1a1a; border:none; border-radius:4px; }
            QListWidget::item { padding:6px; border-bottom:1px solid #2a2a2a; color:#eee; }
            QListWidget::item:selected { background:#2a2a2a; }
        """)
        layout.addWidget(self.list)

        self.count_label = QLabel("0 alerts")
        self.count_label.setStyleSheet("color:#888; font-size:11px;")
        layout.addWidget(self.count_label)

    def add_alert(self, alert: dict) -> None:
        priority = alert.get("priority", "low")
        color    = PRIORITY_COLORS.get(priority, "#aaa")
        text = (
            f"[{alert['timestamp']}]  {alert['rule'].upper()}  "
            f"· Person #{alert.get('person_id', '?')}  "
            f"· conf {alert['confidence']:.0%}\n"
            f"{alert['reason']}"
        )
        item = QListWidgetItem(text)
        item.setForeground(QColor(color))
        self.list.insertItem(0, item)
        self.count_label.setText(f"{self.list.count()} alert(s)")

    def _clear(self) -> None:
        self.list.clear()
        self.count_label.setText("0 alerts")
