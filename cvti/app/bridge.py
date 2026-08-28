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

from cvti.logging_setup import get_logger
from cvti.security.permissions import PermissionDenied

log = get_logger(__name__)


def _j(obj) -> str:
    return json.dumps(obj)


class Backend(QObject):
    def __init__(self, core: ConsoleBackend, parent=None) -> None:
        super().__init__(parent)
        self._core = core

    def _safe(self, fn):
        try:
            return _j(fn())
        except PermissionDenied as exc:
            # Expected, not exceptional: a role reaching for something that is
            # not theirs. Logged at warning by the permission check itself.
            return _j({"error": str(exc), "denied": True})
        except Exception as exc:  # noqa: BLE001 - surface errors to the UI, don't crash
            log.error("backend call failed; surfaced to the UI as an error", exc_info=True)
            return _j({"error": str(exc)[:200]})

    @pyqtSlot(result=str)
    def listCameras(self) -> str:
        return self._safe(self._core.list_cameras)

    @pyqtSlot(result=str)
    def counts(self) -> str:
        return self._safe(self._core.counts)

    @pyqtSlot(result=str)
    def learningStats(self) -> str:
        return self._safe(self._core.learning_stats)

    @pyqtSlot(result=str)
    def learningCalibrate(self) -> str:
        return self._safe(self._core.learning_calibrate)

    @pyqtSlot(result=str)
    def feedSources(self) -> str:
        return self._safe(self._core.feed_sources)

    @pyqtSlot(str, result=str)
    def switchFeed(self, key: str) -> str:
        # returns immediately; the switch runs on a background thread
        return self._safe(lambda: self._core.switch_feed(key))

    @pyqtSlot(result=str)
    def feedSwitchStatus(self) -> str:
        return self._safe(self._core.feed_switch_status)

    @pyqtSlot(result=str)
    def presets(self) -> str:
        return self._safe(self._core.presets)

    @pyqtSlot(result=str)
    def useCaseTemplates(self) -> str:
        return self._safe(self._core.use_case_templates)

    @pyqtSlot(str, result=str)
    def applyTemplate(self, key: str) -> str:
        return self._safe(lambda: self._core.apply_template(key))

    @pyqtSlot(result=str)
    def setupCheck(self) -> str:
        return self._safe(self._core.setup_check)

    @pyqtSlot(result=str)
    def detectorValidation(self) -> str:
        return self._safe(self._core.detector_validation)

    @pyqtSlot(result=str)
    def discoverCameras(self) -> str:
        return self._safe(self._core.discover_cameras)

    @pyqtSlot(str, str, float, result=str)
    def setCustomRule(self, camera_id: str, question: str, dwell: float) -> str:
        return self._safe(lambda: self._core.set_custom_rule(camera_id, question, dwell or 4.0))

    @pyqtSlot(str, str, float, result=str)
    def addCustomRule(self, camera_id: str, question: str, dwell: float) -> str:
        return self._safe(lambda: self._core.add_custom_rule(camera_id, question, dwell or 4.0))

    @pyqtSlot(str, str, result=str)
    def removeCustomRule(self, camera_id: str, question: str) -> str:
        return self._safe(lambda: self._core.remove_custom_rule(camera_id, question))

    @pyqtSlot(result=str)
    def backupNow(self) -> str:
        return self._safe(self._core.backup_now)

    @pyqtSlot(result=str)
    def listBackups(self) -> str:
        return self._safe(self._core.list_backups)

    @pyqtSlot(str, result=str)
    def restoreBackup(self, zip_path: str) -> str:
        return self._safe(lambda: self._core.restore_backup(zip_path))

    @pyqtSlot(str, result=str)
    def setBackupDir(self, path: str) -> str:
        return self._safe(lambda: self._core.set_backup_dir(path))

    @pyqtSlot(result=str)
    def weeklySummary(self) -> str:
        return self._safe(self._core.weekly_summary)

    @pyqtSlot(str, str, result=str)
    def setCameraRules(self, camera_id: str, rules_json: str) -> str:
        return self._safe(lambda: self._core.set_camera_rules(camera_id, json.loads(rules_json)))

    @pyqtSlot(str, result=str)
    def sceneContext(self, camera_id: str) -> str:
        return self._safe(lambda: self._core.scene_context(camera_id))

    # --- zones ---
    @pyqtSlot(str, result=str)
    def cameraSnapshot(self, camera_id: str) -> str:
        return self._safe(lambda: self._core.camera_snapshot(camera_id))

    @pyqtSlot(str, result=str)
    def listZones(self, camera_id: str) -> str:
        return self._safe(lambda: self._core.list_zones(camera_id))

    @pyqtSlot(str, str, str, float, result=str)
    def addZone(self, camera_id: str, name: str, points_json: str, dwell_seconds: float) -> str:
        return self._safe(lambda: self._core.add_zone(camera_id, name, json.loads(points_json), dwell_seconds))

    @pyqtSlot(str, str, result=str)
    def removeZone(self, camera_id: str, name: str) -> str:
        return self._safe(lambda: self._core.remove_zone(camera_id, name))

    @pyqtSlot(str, str, str, result=str)
    def addCustomThreat(self, camera_id: str, name: str, description: str) -> str:
        return self._safe(lambda: self._core.add_custom_threat(camera_id, name, description))

    @pyqtSlot(str, int, result=str)
    def removeCustomThreat(self, camera_id: str, index: int) -> str:
        return self._safe(lambda: self._core.remove_custom_threat(camera_id, index))

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

    @pyqtSlot(int, result=str)
    def listEventsLite(self, limit: int) -> str:
        # no base64 frames — cheap enough to poll every few seconds
        return self._safe(lambda: self._core.list_events(limit or 100, embed_frames=False))

    @pyqtSlot(str, str, result=str)
    def review(self, event_id: str, label: str) -> str:
        return self._safe(lambda: self._core.set_review(event_id, label))

    @pyqtSlot(result=str)
    def sendTestNotification(self) -> str:
        return self._safe(self._core.send_test_notification)

    @pyqtSlot(str, result=str)
    def searchEvents(self, query: str) -> str:
        return self._safe(lambda: self._core.search_events(query))

    @pyqtSlot(str, result=str)
    def eventClip(self, evidence_dir: str) -> str:
        # Lazily fetch the selected event's real-video clip.mp4 as a data URI
        # (embedding every event's video in listEvents would be far too heavy).
        return self._safe(lambda: self._core.event_clip(evidence_dir))

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

    @pyqtSlot(int, result=str)
    def valueSummary(self, days: int) -> str:
        return self._safe(lambda: self._core.value_summary(days or 30))

    @pyqtSlot(float, float, float, result=str)
    def setValueInputs(self, incident_value: float, guard_hourly_cost: float,
                       review_minutes: float) -> str:
        return self._safe(lambda: self._core.set_value_inputs(
            incident_value=incident_value, guard_hourly_cost=guard_hourly_cost,
            review_minutes=review_minutes))

    @pyqtSlot(result=str)
    def markConfigured(self) -> str:
        return self._safe(self._core.mark_configured)

    @pyqtSlot(int, bool, result=str)
    def setLegalHold(self, event_id: int, hold: bool) -> str:
        return self._safe(lambda: self._core.set_legal_hold(event_id, hold))

    @pyqtSlot(result=str)
    def retentionStatus(self) -> str:
        return self._safe(self._core.retention_status)

    @pyqtSlot(float, result=str)
    def setRetention(self, days: float) -> str:
        return self._safe(lambda: self._core.set_retention(days))

    @pyqtSlot(str, result=str)
    def exportEvidence(self, event_ids: str) -> str:
        return self._safe(lambda: self._core.export_evidence(event_ids))

    # --- identity, roles, audit (EP-03) ---
    @pyqtSlot(result=str)
    def authState(self) -> str:
        return self._safe(self._core.auth_state)

    @pyqtSlot(str, str, result=str)
    def createFirstOwner(self, username: str, password: str) -> str:
        return self._safe(lambda: self._core.create_first_owner(username, password))

    @pyqtSlot(str, str, result=str)
    def signIn(self, username: str, password: str) -> str:
        return self._safe(lambda: self._core.sign_in(username, password))

    @pyqtSlot(result=str)
    def authRecovery(self) -> str:
        return self._safe(self._core.auth_recovery)

    @pyqtSlot(result=str)
    def signOut(self) -> str:
        return self._safe(self._core.sign_out)

    @pyqtSlot(str, str, result=str)
    def changeOwnPassword(self, current: str, new: str) -> str:
        return self._safe(lambda: self._core.change_own_password(current, new))

    @pyqtSlot(result=str)
    def listUsers(self) -> str:
        return self._safe(self._core.list_users)

    @pyqtSlot(str, str, str, result=str)
    def addUser(self, username: str, password: str, role: str) -> str:
        return self._safe(lambda: self._core.add_user(username, password, role))

    @pyqtSlot(str, str, result=str)
    def setUserRole(self, username: str, role: str) -> str:
        return self._safe(lambda: self._core.set_user_role(username, role))

    @pyqtSlot(str, result=str)
    def removeUser(self, username: str) -> str:
        return self._safe(lambda: self._core.remove_user(username))

    @pyqtSlot(int, result=str)
    def auditEntries(self, limit: int) -> str:
        return self._safe(lambda: self._core.audit_entries(limit or 200))

    @pyqtSlot(result=str)
    def auditVerify(self) -> str:
        return self._safe(self._core.audit_verify)

    @pyqtSlot(result=str)
    def auditExport(self) -> str:
        return self._safe(self._core.audit_export)

    @pyqtSlot(str, result=str)
    def acknowledgeAlert(self, event_id: str) -> str:
        return self._safe(lambda: self._core.acknowledge_alert(event_id))

    @pyqtSlot(str, str, str, result=str)
    def resolveAlert(self, event_id: str, outcome: str, note: str) -> str:
        return self._safe(lambda: self._core.resolve_alert(event_id, outcome, note))

    @pyqtSlot(str, result=str)
    def exportIncidentPdf(self, event_id: str) -> str:
        return self._safe(lambda: self._core.export_incident_pdf(event_id))

    @pyqtSlot(float, result=str)
    def handover(self, hours: float) -> str:
        return self._safe(lambda: self._core.handover(hours or 8.0))

    @pyqtSlot(str, result=str)
    def needsAttention(self, min_priority: str) -> str:
        return self._safe(lambda: self._core.needs_attention(min_priority or "medium"))

    @pyqtSlot(result=str)
    def heartbeatStatus(self) -> str:
        return self._safe(self._core.heartbeat_status)

    @pyqtSlot(str, str, result=str)
    def setHeartbeat(self, url: str, key: str) -> str:
        return self._safe(lambda: self._core.set_heartbeat(url, key))

    @pyqtSlot(result=str)
    def diskEncryption(self) -> str:
        return self._safe(self._core.disk_encryption)

    @pyqtSlot(result=str)
    def roleTable(self) -> str:
        return self._safe(self._core.role_table)

    @pyqtSlot(result=str)
    def cameraLinks(self) -> str:
        return self._safe(self._core.camera_links)

    @pyqtSlot(result=str)
    def downloadDiagnostics(self) -> str:
        return self._safe(self._core.download_diagnostics)

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

    # --- monitoring engine ---
    @pyqtSlot(result=str)
    def startMonitoring(self) -> str:
        return self._safe(self._core.start_monitoring)

    @pyqtSlot(result=str)
    def stopMonitoring(self) -> str:
        return self._safe(self._core.stop_monitoring)

    @pyqtSlot(result=str)
    def monitoringStatus(self) -> str:
        return self._safe(self._core.monitoring_status)
