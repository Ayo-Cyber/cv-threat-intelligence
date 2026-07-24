"""QWebChannel bridge — exposes ConsoleBackend to the web frontend.

Thin QObject: each slot calls the tested ConsoleBackend and returns JSON. Slots
run on the Qt main thread, but QWebEngine renders the page in its own process,
so a slow call (scan/test ~1-3s) shows the frontend's spinner without freezing
the UI; the JS callback fires when it returns.
"""
from __future__ import annotations

import json

from PyQt6.QtCore import QObject, pyqtSlot

from cvti.app.console_backend import ConsoleBackend


def _j(obj) -> str:
    return json.dumps(obj)


class Backend(QObject):
    def __init__(self, core: ConsoleBackend, parent=None) -> None:
        super().__init__(parent)
        self._core = core

    def _safe(self, fn):
        try:
            return _j(fn())
        except Exception as exc:  # noqa: BLE001 - surface errors to the UI, don't crash
            return _j({"error": str(exc)[:200]})

    @pyqtSlot(result=str)
    def listCameras(self) -> str:
        return self._safe(self._core.list_cameras)

    @pyqtSlot(result=str)
    def counts(self) -> str:
        return self._safe(self._core.counts)

    @pyqtSlot(result=str)
    def presets(self) -> str:
        return self._safe(self._core.presets)

    @pyqtSlot(str, result=str)
    def scan(self, cidr: str) -> str:
        return self._safe(lambda: self._core.scan(cidr))

    @pyqtSlot(result=str)
    def detectSubnet(self) -> str:
        return self._safe(self._core.detect_subnet)

    @pyqtSlot(str, result=str)
    def testUrl(self, url: str) -> str:
        return self._safe(lambda: self._core.test(url))

    @pyqtSlot(str, result=str)
    def addCamera(self, camera_json: str) -> str:
        return self._safe(lambda: self._core.add_camera(json.loads(camera_json)))

    @pyqtSlot(str, result=str)
    def removeCamera(self, camera_id: str) -> str:
        return self._safe(lambda: self._core.remove_camera(camera_id))

    @pyqtSlot(int, result=str)
    def listEvents(self, limit: int) -> str:
        return self._safe(lambda: self._core.list_events(limit or 100))

    @pyqtSlot(str, str, result=str)
    def review(self, event_id: str, label: str) -> str:
        return self._safe(lambda: self._core.set_review(event_id, label))

    # --- first-run setup wizard ---
    @pyqtSlot(result=str)
    def setupState(self) -> str:
        return self._safe(self._core.setup_state)

    @pyqtSlot(result=str)
    def getSite(self) -> str:
        return self._safe(self._core.get_site)

    @pyqtSlot(str, str, result=str)
    def setSite(self, name: str, notify: str) -> str:
        return self._safe(lambda: self._core.set_site(name=name or None, notify=notify or None))

    @pyqtSlot(result=str)
    def markConfigured(self) -> str:
        return self._safe(self._core.mark_configured)

    @pyqtSlot(result=str)
    def gateStatus(self) -> str:
        return self._safe(self._core.gate_status)

    @pyqtSlot(result=str)
    def pullModel(self) -> str:
        return self._safe(self._core.pull_model)

    @pyqtSlot(result=str)
    def pullProgress(self) -> str:
        return self._safe(self._core.pull_progress)

    # --- live wall ---
    @pyqtSlot(int, result=str)
    def liveStart(self, count: int) -> str:
        return self._safe(lambda: self._core.live_start(count or 6))

    @pyqtSlot(result=str)
    def liveFrames(self) -> str:
        return self._safe(self._core.live_frames)

    @pyqtSlot(result=str)
    def liveStop(self) -> str:
        return self._safe(self._core.live_stop)
