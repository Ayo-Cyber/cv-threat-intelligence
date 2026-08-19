"""Cameras tab — customer-facing camera onboarding.

Add a camera without touching JSON or the CLI: scan the network for cameras,
paste/pick an RTSP URL, Test it (see a snapshot + resolution), then Add — it's
saved into the site config the pipeline runs. Thin shell over
cvti.serving.onboarding (all logic + tests live there).
"""
from __future__ import annotations

import base64

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from cvti.serving import onboarding

from cvti.logging_setup import get_logger

log = get_logger(__name__)


class _TestWorker(QThread):
    done = pyqtSignal(dict)

    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url

    def run(self) -> None:
        try:
            self.done.emit(onboarding.test_url(self._url))
        except Exception as exc:  # noqa: BLE001
            log.warning("camera test failed", exc_info=True)
            self.done.emit({"ok": False, "error": str(exc)[:160]})


class _ScanWorker(QThread):
    done = pyqtSignal(list)

    def __init__(self, cidr: str) -> None:
        super().__init__()
        self._cidr = cidr

    def run(self) -> None:
        try:
            self.done.emit(onboarding.scan_subnet(self._cidr))
        except Exception as exc:  # noqa: BLE001
            log.warning("camera scan failed", exc_info=True)
            self.done.emit([])


class CamerasPanel(QWidget):
    """Site-config camera manager. `site_path` is the JSON the pipeline reads."""

    def __init__(self, site_path: str = "configs/site_live.json", parent=None) -> None:
        super().__init__(parent)
        self.site_path = site_path
        self._test: _TestWorker | None = None
        self._scan: _ScanWorker | None = None
        self._build()
        self._refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        title = QLabel("Cameras")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        root.addWidget(title)

        # discover
        scan_row = QHBoxLayout()
        self.cidr = QLineEdit()
        self.cidr.setPlaceholderText("192.168.1.0/24")
        self.scan_btn = QPushButton("Scan network")
        self.scan_btn.clicked.connect(self._on_scan)
        scan_row.addWidget(QLabel("Find cameras:"))
        scan_row.addWidget(self.cidr, 1)
        scan_row.addWidget(self.scan_btn)
        root.addLayout(scan_row)
        self.scan_result = QLabel("")
        self.scan_result.setWordWrap(True)
        self.scan_result.setStyleSheet("color:#8c97a7")
        root.addWidget(self.scan_result)

        # add form
        form = QFormLayout()
        self.url = QLineEdit()
        self.url.setPlaceholderText("rtsp://user:pass@192.168.1.10:554/Streaming/Channels/102")
        self.name = QLineEdit()
        self.name.setPlaceholderText("entrance")
        self.preset = QComboBox()
        for label in onboarding.RULE_PRESETS:
            self.preset.addItem(label, onboarding.RULE_PRESETS[label])
        toggles = QHBoxLayout()
        self.conceal = QCheckBox("concealment")
        self.conceal.setChecked(True)
        self.video = QCheckBox("video-action")
        self.video.setChecked(True)
        toggles.addWidget(self.conceal)
        toggles.addWidget(self.video)
        toggles.addStretch(1)
        form.addRow("Camera URL", self.url)
        form.addRow("Name", self.name)
        form.addRow("Rules", self.preset)
        form.addRow("Signals", self._wrap(toggles))
        root.addLayout(form)

        btns = QHBoxLayout()
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self._on_test)
        self.add_btn = QPushButton("Add camera")
        self.add_btn.clicked.connect(self._on_add)
        btns.addWidget(self.test_btn)
        btns.addWidget(self.add_btn)
        btns.addStretch(1)
        root.addLayout(btns)

        self.preview = QLabel("")
        self.preview.setMinimumHeight(140)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.preview)
        root.addWidget(self.status)

        root.addWidget(self._divider("Cameras on this site"))
        self.cam_list = QListWidget()
        root.addWidget(self.cam_list, 1)
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._on_remove)
        root.addWidget(self.remove_btn)

    # --- helpers ---
    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _divider(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        lbl.setStyleSheet("color:#8c97a7;margin-top:6px")
        return lbl

    def _refresh(self) -> None:
        self.cam_list.clear()
        for c in onboarding.list_cameras(self.site_path):
            sig = " · ".join(s for s in (("concealment" if c.get("concealment") else ""),
                                         ("video" if c.get("video_action") else "")) if s) or "basic"
            item = QListWidgetItem(f"{c.get('id')}   {str(c.get('source','')).split('@')[-1]}   [{sig}]")
            item.setData(256, c.get("id"))
            self.cam_list.addItem(item)

    # --- scan ---
    def _on_scan(self) -> None:
        cidr = self.cidr.text().strip()
        if not cidr:
            self.scan_result.setText("Enter a subnet like 192.168.1.0/24")
            return
        self.scan_result.setText("Scanning…")
        self.scan_btn.setEnabled(False)
        self._scan = _ScanWorker(cidr)
        self._scan.done.connect(self._on_scan_done)
        self._scan.start()

    def _on_scan_done(self, hosts: list) -> None:
        self.scan_btn.setEnabled(True)
        if not hosts:
            self.scan_result.setText("No cameras found on that subnet.")
            return
        self.scan_result.setText("Found: " + ", ".join(hosts) + "  — click a suggestion below to use it.")
        # offer the first as a URL template
        self.url.setText(f"rtsp://user:pass@{hosts[0]}:554/Streaming/Channels/102")

    # --- test ---
    def _on_test(self) -> None:
        url = self.url.text().strip()
        if not url:
            self.status.setText("Enter a camera URL first.")
            return
        self.status.setText("Testing…")
        self.test_btn.setEnabled(False)
        self._test = _TestWorker(url)
        self._test.done.connect(self._on_test_done)
        self._test.start()

    def _on_test_done(self, result: dict) -> None:
        self.test_btn.setEnabled(True)
        if not result.get("ok"):
            self.preview.clear()
            self.status.setText("❌ " + result.get("error", "test failed"))
            self.status.setStyleSheet("color:#e0655b")
            return
        pm = QPixmap()
        pm.loadFromData(base64.b64decode(result["jpeg_b64"]), "JPEG")
        self.preview.setPixmap(pm)
        self.status.setText(f"✓ OK — {result['w']}×{result['h']} @ {result['fps']} fps")
        self.status.setStyleSheet("color:#36c98a")

    # --- add / remove ---
    def _on_add(self) -> None:
        url = self.url.text().strip()
        if not url:
            self.status.setText("Enter a camera URL first.")
            return
        cam = {
            "id": self.name.text().strip() or None,
            "source": url,
            "config": self.preset.currentData(),
            "concealment": self.conceal.isChecked(),
            "video_action": self.video.isChecked(),
        }
        try:
            onboarding.add_camera(self.site_path, {k: v for k, v in cam.items() if v is not None})
        except Exception as exc:  # noqa: BLE001
            log.warning("adding a camera failed", exc_info=True)
            QMessageBox.warning(self, "Add camera", str(exc))
            return
        self.status.setText(f"✓ Added — saved to {self.site_path}")
        self.status.setStyleSheet("color:#36c98a")
        self.url.clear()
        self.name.clear()
        self._refresh()

    def _on_remove(self) -> None:
        item = self.cam_list.currentItem()
        if item is None:
            return
        onboarding.remove_camera(self.site_path, item.data(256))
        self._refresh()
