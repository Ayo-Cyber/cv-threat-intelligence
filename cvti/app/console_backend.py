"""Console backend core — the logic behind the desktop app's UI (Phase 9).

Qt-free and unit-tested. The QWebChannel bridge (cvti/app/bridge.py) is a thin
wrapper that JSON-serializes these methods for the web frontend; the desktop
shell hosts the frontend in a native window. Keeping the logic here means the
part that matters is verifiable without a display.

Covers the two core operator flows:
  • Cameras  — list / scan / test / add / remove (reuses cvti.serving.onboarding).
  • Alerts   — list confirmed events (with evidence as data URIs) + review them
               true / false / acknowledged (the label that feeds retraining).
"""
from __future__ import annotations

import base64
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from cvti.logging_setup import get_logger
from cvti.security import permissions as perms
from cvti.security.accounts import AccountStore, AuthError
from cvti.security.audit import AuditLog
from cvti.serving import onboarding, vlm
from cvti.utils import resource_path

log = get_logger(__name__)

_REVIEW_VALUES = {"ack", "true", "false", "new"}


class ConsoleBackend:
    def __init__(self, site_path: str = "configs/site_live.json",
                 db_path: str = "runs/site/events.db", enable_demo: bool = True) -> None:
        self.site_path = site_path
        self.db_path = db_path
        self._live = None       # LiveWall instance while the Live screen is open
        self._fs = None         # localhost FrameServer serving live JPEGs
        self._monitor = None    # detection-engine subprocess (Start monitoring)
        self._monitor_should_run = False  # watchdog: respawn engine if it dies
        self._restarts = 0
        # bundled playback demo (for machines w/o the engine); off in tests
        self._demo = self._locate_demo() if enable_demo else None

        # --- identity, roles, audit (EP-03) ---
        # Credentials and the audit trail sit beside the events database, never
        # inside it: one deletion must not take both the footage and the record
        # of who touched it.
        _sec_dir = Path(self.db_path).parent
        self.accounts = AccountStore(_sec_dir / "auth.db")
        self.audit = AuditLog(_sec_dir / "audit.db")
        # EP-08-T1: a corrupt events.db is quarantined LOUDLY at startup (a
        # fresh store follows), and the config is backed up daily, versioned.
        from cvti import backup as _backup
        try:
            self.db_check = _backup.check_events_db(self.db_path)
        except Exception:  # noqa: BLE001 - the check must not stop the app
            log.error("events.db startup check failed", exc_info=True)
            self.db_check = {"ok": True, "state": "unchecked"}
        try:
            self._maybe_auto_backup()
        except Exception:  # noqa: BLE001
            log.warning("auto-backup failed", exc_info=True)
        self._session: str = ""          # token of the signed-in operator

    # --- session ----------------------------------------------------------
    @property
    def current_user(self):
        return self.accounts.session_user(self._session) if self._session else None

    def _role(self):
        user = self.current_user
        return user.role if user else None

    def _require(self, permission: str) -> None:
        """Server-side authorisation. Hiding a button changes what is easy;
        this changes what is possible, which is the question that matters."""
        perms.require(self._role(), permission)

    def auth_state(self) -> dict:
        """What the UI needs to pick a screen. Never a permission source — the
        backend re-checks on every call."""
        user = self.current_user
        return {
            "configured": self.accounts.any_users(),
            "signed_in": user is not None,
            "username": user.username if user else "",
            "role": user.role if user else "",
            "must_change_password": bool(user.must_change) if user else False,
            "landing": perms.landing_for(user.role) if user else "",
            "permissions": sorted(perms.permissions_for(user.role)) if user else [],
        }

    def create_first_owner(self, username: str, password: str) -> dict:
        """First run. Nothing ships with a known credential because nothing
        ships with an account at all — the first one is created here."""
        if self.accounts.any_users():
            return {"ok": False, "error": "this site already has accounts"}
        try:
            self.accounts.create_user(username, password, role=perms.OWNER)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self.audit.record(username, "role_change", f"user:{username}",
                          {"created": True, "role": perms.OWNER, "first_run": True})
        return self.sign_in(username, password)

    def sign_in(self, username: str, password: str) -> dict:
        try:
            user = self.accounts.authenticate(username, password)
        except AuthError as exc:
            self.audit.record(username or "<unknown>", "login",
                              detail={"outcome": "refused", "reason": str(exc)})
            return {"ok": False, "error": str(exc)}
        self._session = self.accounts.open_session(user.username)
        self.audit.record(user.username, "login",
                          detail={"outcome": "success", "role": user.role})
        return {"ok": True, **self.auth_state()}

    def sign_out(self) -> dict:
        user = self.current_user
        if user:
            self.audit.record(user.username, "login", detail={"outcome": "signed out"})
        self.accounts.close_session(self._session)
        self._session = ""
        return {"ok": True}

    def change_own_password(self, current: str, new: str) -> dict:
        user = self.current_user
        if user is None:
            return {"ok": False, "error": "not signed in"}
        try:
            self.accounts.authenticate(user.username, current)
        except AuthError:
            return {"ok": False, "error": "current password is incorrect"}
        try:
            self.accounts.set_password(user.username, new)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self.audit.record(user.username, "role_change", f"user:{user.username}",
                          {"password_changed": True})
        self._session = self.accounts.open_session(user.username)   # set_password revoked it
        return {"ok": True}

    # --- user administration (owner only) ---------------------------------
    def list_users(self) -> list:
        self._require(perms.MANAGE_USERS)
        return [{"username": u.username, "role": u.role, "must_change": u.must_change,
                 "last_login": u.last_login} for u in self.accounts.list_users()]

    def add_user(self, username: str, password: str, role: str = "operator") -> dict:
        self._require(perms.MANAGE_USERS)
        try:
            self.accounts.create_user(username, password, role=role)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self.audit.record(self.current_user.username, "role_change", f"user:{username}",
                          {"created": True, "role": role})
        return {"ok": True}

    def set_user_role(self, username: str, role: str) -> dict:
        self._require(perms.MANAGE_USERS)
        before = self.accounts.user(username)
        try:
            self.accounts.set_role(username, role)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self.audit.record(self.current_user.username, "role_change", f"user:{username}",
                          {"from": before.role if before else None, "to": role})
        return {"ok": True}

    def remove_user(self, username: str) -> dict:
        self._require(perms.MANAGE_USERS)
        me = self.current_user
        if me and me.username == username:
            return {"ok": False, "error": "you cannot remove your own account"}
        owners = [u for u in self.accounts.list_users() if u.role == perms.OWNER]
        if len(owners) <= 1 and any(u.username == username for u in owners):
            # A site with no owner has nobody who can grant access to anyone again.
            return {"ok": False, "error": "the last owner cannot be removed"}
        self.accounts.delete_user(username)
        self.audit.record(me.username if me else "?", "role_change", f"user:{username}",
                          {"removed": True})
        return {"ok": True}

    # --- audit trail (owner only) -----------------------------------------
    def audit_entries(self, limit: int = 200) -> list:
        self._require(perms.VIEW_AUDIT)
        return [e.to_dict() for e in self.audit.entries(limit=limit)]

    def audit_verify(self) -> dict:
        self._require(perms.VIEW_AUDIT)
        return self.audit.verify()

    def audit_export(self) -> dict:
        self._require(perms.VIEW_AUDIT)
        path = self.audit.export(Path(self.db_path).parent / f"argus-audit-{int(time.time())}.json")
        self.audit.record(self.current_user.username, "evidence_export", "audit_log",
                          {"path": str(path)})
        return {"ok": True, "path": str(path)}

    def heartbeat_status(self) -> dict:
        """Config + exactly what was last transmitted, so 'what leaves my
        machine?' is answered by looking, not by trusting the docs."""
        meta = self.get_site()
        out = {"enabled": bool(meta.get("heartbeat_url")),
               "url": meta.get("heartbeat_url", ""),
               "has_key": bool(meta.get("heartbeat_key"))}
        engine = self._gate_health() or {}
        out["live"] = engine.get("heartbeat") or {}
        try:
            out["last_payload"] = json.loads(
                (Path(self.db_path).parent / "heartbeat_last.json").read_text())
        except (OSError, ValueError):
            out["last_payload"] = None
        return out

    def set_heartbeat(self, url: str = "", key: str = "") -> dict:
        self._require(perms.CONFIGURE_SITE)
        onboarding.set_site_meta(self.site_path, heartbeat_url=(url or "").strip(),
                                 heartbeat_key=(key or "").strip())
        self.audit.record(self.current_user.username, "config_change", "heartbeat",
                          {"enabled": bool((url or "").strip())})
        return self.heartbeat_status()

    def disk_encryption(self) -> dict:
        """Is the evidence on this machine readable if the machine is taken?"""
        from cvti.security.disk import encryption_status, requirement_message
        status = encryption_status()
        status["message"] = requirement_message(status)
        return status

    def role_table(self) -> dict:
        """The whole permission table — for the UI, and for procurement."""
        return perms.describe()

    @staticmethod
    def _locate_demo():
        """Self-contained playback demo (clips + recorded alerts) shipped in the
        bundle, so the app shows the live wall + alert system on any machine
        without the engine/Ollama/clips installed."""
        cands = []
        if getattr(sys, "frozen", False):
            cands.append(Path(getattr(sys, "_MEIPASS", ".")) / "demo_data")
        cands.append(Path(__file__).resolve().parents[2] / "packaging" / "demo_data")
        for c in cands:
            if (c / "events.db").exists():
                return c
        return None

    # --- cameras (delegate to onboarding) ---
    def list_cameras(self) -> list[dict]:
        return onboarding.list_cameras(self.site_path)

    def scan(self, cidr: str) -> dict:
        return {"hosts": onboarding.scan_subnet(cidr)}

    def detect_subnet(self) -> dict:
        return {"cidr": onboarding.detect_subnet()}

    def test(self, url: str) -> dict:
        """Stream test with a NAME for every failure (EP-05-T4).

        For RTSP, a raw protocol probe first: it distinguishes unreachable /
        wrong credentials / wrong path / unsupported codec — cv2 collapses all
        four into "could not open". Only a passing probe pays for the cv2 open
        that produces the preview snapshot."""
        if str(url).startswith("rtsp"):
            from cvti.serving import discovery
            probe = discovery.probe_rtsp(url)
            if not probe["ok"]:
                return {"error": probe["message"], "kind": probe["kind"]}
            out = onboarding.test_url(url)
            if out.get("error"):
                out.setdefault("kind", "open-failed")
            else:
                out["codec"] = probe.get("codec")
            return out
        return onboarding.test_url(url)

    def discover_cameras(self) -> dict:
        """ONVIF WS-Discovery sweep of the local segment (EP-05-T4)."""
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.serving import discovery
        cams = discovery.discover()
        return {"cameras": cams, "count": len(cams)}

    def add_camera(self, camera: dict) -> list[dict]:
        self._require(perms.CONFIGURE_CAMERAS)
        return onboarding.add_camera(self.site_path, camera)

    def remove_camera(self, camera_id: str) -> list[dict]:
        self._require(perms.CONFIGURE_CAMERAS)
        return onboarding.remove_camera(self.site_path, camera_id)

    def presets(self) -> dict:
        return onboarding.RULE_PRESETS

    # Detectors the operator can toggle per camera (drive which models run).
    # Must stay in sync with PerCameraState's flags in cvti/serving/camera.py.
    RULE_FLAGS = ("concealment", "video_action", "violence", "weapons", "theft", "tamper",
                  "fire_smoke", "running", "crowd_formation", "fall")

    # Tuning params a detector needs to behave sensibly. Applied when it is first
    # switched on so a toggle "just works" without hand-editing the site config;
    # never overwrite a value the operator has already set.
    DETECTOR_DEFAULTS = {
        "running": {"running_min_speed_ratio": 0.08, "running_min_frames": 3},
        "crowd_formation": {"crowd_min_people": 5, "crowd_min_frames": 3,
                            "crowd_max_cluster_ratio": 0.32},
    }

    # --- first-run use-case templates (EP-05-T3) ---------------------------
    # Where the CustomizationEngine's flags finally meet the interface: a
    # non-technical installer picks what the site IS, not which of ten
    # detectors to enable. Keys are RULE_FLAGS; anything absent is OFF.
    USE_CASE_TEMPLATES = {
        "retail": {
            "label": "Retail / Store",
            "blurb": "Shoplifting, concealment, weapons and fire — the shop-floor set.",
            "detectors": {"concealment": True, "video_action": True, "theft": True,
                          "weapons": True, "violence": True, "tamper": True,
                          "fire_smoke": True},
            "config": "configs/all_threats_video_v1.json",
        },
        "warehouse": {
            "label": "Warehouse / HSE",
            "blurb": "People down, fire, panic and crowding — safety first, theft off.",
            "detectors": {"fall": True, "fire_smoke": True, "running": True,
                          "crowd_formation": True, "tamper": True, "weapons": True},
            "config": "configs/all_threats_v1.json",
        },
        "office": {
            "label": "Office",
            "blurb": "After-hours intrusion essentials: violence, weapons, fire, tamper.",
            "detectors": {"violence": True, "weapons": True, "fire_smoke": True,
                          "tamper": True},
            "config": "configs/all_threats_v1.json",
        },
    }

    def use_case_templates(self) -> dict:
        """The three templates, detector labels resolved — read-only, any role."""
        return {k: {"label": t["label"], "blurb": t["blurb"],
                    "detectors": {f: bool(t["detectors"].get(f)) for f in self.RULE_FLAGS}}
                for k, t in self.USE_CASE_TEMPLATES.items()}

    def apply_template(self, key: str) -> dict:
        """Apply a use-case template to EVERY configured camera.

        Sets each RULE_FLAG explicitly (on or off) so switching templates is
        deterministic, then lets set_camera_rules seed detector defaults for
        anything newly enabled. Per-camera fine-tuning stays possible after —
        the template is a starting point, not a lock.
        """
        self._require(perms.CONFIGURE_DETECTORS)
        tpl = self.USE_CASE_TEMPLATES.get(key)
        if tpl is None:
            return {"error": f"unknown template '{key}'"}
        rules = {f: bool(tpl["detectors"].get(f)) for f in self.RULE_FLAGS}
        rules["config"] = tpl["config"]
        cams = onboarding.list_cameras(self.site_path)
        for cam in cams:
            self.set_camera_rules(cam["id"], rules)
        self.audit.record(self.current_user.username, "config_change",
                          "site:template", detail={"template": key, "cameras": len(cams)})
        return {"ok": True, "template": key, "cameras": len(cams),
                "detectors_on": sorted(k for k, v in tpl["detectors"].items() if v)}

    # --- setup self-test (EP-05-T3) -----------------------------------------
    @staticmethod
    def _probe_stream(source) -> tuple:
        """(ok, detail) — fast reachability, never a blocking cv2 open.

        A wrong-credentials RTSP URL still connects at the TCP layer; the full
        stream test is the wizard's per-camera Test button. This check answers
        the self-test's question — is anything answering at all — in <1s.
        """
        import socket
        from urllib.parse import urlparse
        src = str(source)
        if src.isdigit():
            return True, "built-in webcam"
        if "://" not in src:
            return (True, "video file on disk") if Path(src).exists()                 else (False, f"file not found: {src}")
        u = urlparse(src)
        port = u.port or {"rtsp": 554, "http": 80, "https": 443}.get(u.scheme, 554)
        try:
            with socket.create_connection((u.hostname, port), timeout=1.5):
                return True, f"{u.hostname}:{port} answers"
        except OSError as exc:
            return False, f"{u.hostname}:{port} unreachable ({exc})"

    def setup_check(self) -> list:
        """Every component, verified, in plain English (EP-05-T3 acceptance:
        'names exactly what is missing'). Each item: ok True/False/None(warn),
        what was checked, what was found, and the one action that fixes it."""
        checks = []

        cams = [c for c in onboarding.list_cameras(self.site_path) if c.get("source")]
        checks.append({"id": "cameras", "ok": bool(cams),
                       "label": "Cameras configured",
                       "detail": f"{len(cams)} camera(s) in the site config" if cams
                       else "no cameras yet",
                       "fix": None if cams else "Add a camera in the Cameras step."})

        for c in cams[:8]:
            ok, detail = self._probe_stream(c.get("source"))
            checks.append({"id": f"stream:{c['id']}", "ok": ok,
                           "label": f"Camera '{c['id']}' reachable",
                           "detail": detail,
                           "fix": None if ok else "Check the camera's power, network "
                           "cable and IP address, then use Test on the Cameras step."})

        g = self.gate_status()
        if g.get("mode") == "live":
            checks.append({"id": "verifier", "ok": True, "label": "AI verifier (TrueSight)",
                           "detail": "running, model installed", "fix": None})
        elif g.get("mode") == "no-model":
            checks.append({"id": "verifier", "ok": False, "label": "AI verifier (TrueSight)",
                           "detail": "runtime is up but the vision model is not downloaded",
                           "fix": "Download the model in the Verification step (~3.3 GB, resumes)."})
        else:
            checks.append({"id": "verifier", "ok": False, "label": "AI verifier (TrueSight)",
                           "detail": "not running",
                           "fix": "Click Start verifier in the Verification step."
                           if g.get("runtime_bundled")
                           else "Install Ollama from ollama.com, then recheck."})

        det = resource_path("models/yolov8n.pt")
        checks.append({"id": "detector", "ok": det.exists(), "label": "Detection models",
                       "detail": "YOLO weights present" if det.exists()
                       else "models/yolov8n.pt is missing from this install",
                       "fix": None if det.exists() else "Reinstall Argus — the bundle is incomplete."})
        vm = resource_path("runs/video_finetune/videomae")
        if not vm.exists():
            checks.append({"id": "video_model", "ok": None, "label": "Video action model",
                           "detail": "VideoMAE fine-tune not bundled — video theft "
                           "detection will be off; everything else still runs",
                           "fix": None})

        meta = onboarding.get_site_meta(self.site_path)
        notify = (meta.get("notify") or "console").strip()
        if notify and notify != "console":
            checks.append({"id": "notify", "ok": True, "label": "Notifications",
                           "detail": f"alerts go to {notify}",
                           "fix": "Confirm delivery with the test alert step."})
        else:
            checks.append({"id": "notify", "ok": None, "label": "Notifications",
                           "detail": "alerts only appear inside the app (console)",
                           "fix": "Pick WhatsApp/Telegram/webhook in Site settings "
                           "to be reached when you're not at this screen."})

        import shutil as _shutil
        try:
            du = _shutil.disk_usage(str(Path(self.db_path).parent))
            pct = du.used * 100.0 / max(1, du.total)
            free_gb = du.free / 1e9
            low = free_gb < 5
            checks.append({"id": "disk", "ok": (None if low else True) if free_gb >= 1 else False,
                           "label": "Disk space for evidence",
                           "detail": f"{free_gb:.0f} GB free ({pct:.0f}% used)",
                           "fix": "Free some space — evidence recording needs room."
                           if low else None})
        except OSError:
            checks.append({"id": "disk", "ok": None, "label": "Disk space for evidence",
                           "detail": "could not be measured", "fix": None})
        return checks

    def set_camera_rules(self, camera_id: str, rules: dict) -> dict:
        """Update which threat detectors run on a camera (+ optional rule preset).
        Takes effect on the next Start monitoring."""
        self._require(perms.CONFIGURE_DETECTORS)
        cams = onboarding.list_cameras(self.site_path)
        cam = next((c for c in cams if c.get("id") == camera_id), None)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        for k in self.RULE_FLAGS:
            if k in rules:
                turning_on = bool(rules[k]) and not cam.get(k)
                cam[k] = bool(rules[k])
                if turning_on:      # seed this detector's tuning params (don't clobber)
                    for pk, pv in self.DETECTOR_DEFAULTS.get(k, {}).items():
                        cam.setdefault(pk, pv)
        if rules.get("config"):
            # A preset choice is a new BASE, not a raw config pointer: cameras
            # with zones or an English rule keep them — regen layers the base
            # under the generated rules instead of clobbering the pointer.
            # (Audit 23 Aug, #9: applying a template used to silently drop the
            # camera's loitering + custom-English rules, and the next zone edit
            # reverted the camera to the pre-template preset.)
            cam["_base_config"] = rules["config"]
            zones = self.list_zones(camera_id)
            if zones or cam.get("custom_rule"):
                self._regen_zone_rules(camera_id, cam, zones)
            else:
                cam["config"] = rules["config"]
        onboarding.add_camera(self.site_path, cam)   # upsert by id
        return {"ok": True, "camera": cam}

    # --- zones (draw in-app -> geometry + a loitering rule the engine runs) ---
    def camera_snapshot(self, camera_id: str) -> dict:
        """A still from the camera to draw zones on — plus its ORIGINAL pixel
        size so the UI can map canvas coords back to real zone coordinates."""
        import cv2
        cam = next((c for c in self.list_cameras() if c.get("id") == camera_id), None)
        if cam is None or not cam.get("source"):
            return {"error": "camera not found"}
        src = cam["source"]
        cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        try:
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if n > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.3))
            ok, fr = cap.read()
        finally:
            cap.release()
        if not ok:
            return {"error": "could not read a frame from this camera"}
        h, w = fr.shape[:2]
        ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok2:
            return {"error": "encode failed"}
        return {"uri": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                "w": int(w), "h": int(h)}

    def _zones_file(self, camera_id: str) -> Path:
        return Path("configs/zones") / f"{camera_id}.json"

    def list_zones(self, camera_id: str) -> list[dict]:
        f = self._zones_file(camera_id)
        if f.exists():
            try:
                return json.loads(f.read_text()).get("zones", [])
            except (ValueError, OSError):
                return []
        return []

    def _regen_zone_rules(self, camera_id: str, cam: dict, zones: list[dict]) -> None:
        """Per-camera rule config = the camera's base preset + one loitering rule
        per zone. Keeps the shared preset untouched."""
        base_cfg = cam.get("_base_config") or cam.get("config") or "configs/all_threats_v1.json"
        cam["_base_config"] = base_cfg
        try:
            rules = list(json.loads(Path(base_cfg).read_text()).get("rules", []))
        except (ValueError, OSError):
            rules = []
        for z in zones:
            dw = z.get("dwell_alert_seconds", 5)
            rules.append({"name": f"loitering_{z['name']}", "trigger": {"detector": "presence"},
                          "context_filter": f"zone == '{z['name']}' and dwell_seconds >= {dw}",
                          "priority": "medium"})
        custom = cam.get("custom_rule") or {}
        if custom.get("question"):
            # The camera's plain-English rule (EP-05 follow-through, demo-day
            # gap): the sentence goes to TrueSight verbatim whenever a person
            # lingers anywhere on this camera. Regenerated here so drawing or
            # deleting zones never silently drops it.
            rules.append({"name": "custom_english",
                          "title": "CUSTOM RULE MATCHED",
                          "trigger": {"detector": "presence"},
                          "context_filter": f"dwell_seconds >= {float(custom.get('dwell', 4))}",
                          "priority": custom.get("priority", "high"),
                          "gate_question": custom["question"]})
        rdir = Path("configs/rules")
        rdir.mkdir(parents=True, exist_ok=True)
        rfile = rdir / f"{camera_id}.json"
        rfile.write_text(json.dumps({"use_case_id": f"{camera_id}_zones", "rules": rules}, indent=2))
        cam["config"] = str(rfile)

    def set_custom_rule(self, camera_id: str, question: str, dwell: float = 4.0) -> dict:
        """The plain-English rule, from the app (demo-day gap: the sentence had
        no home in the UI). Empty question removes the rule. A person lingering
        `dwell` seconds anywhere on this camera triggers it; the sentence is
        what TrueSight judges. Applies on the next Start monitoring.

        If the camera has no zones yet, an everywhere-zone is created — without
        one the presence detector never runs and the sentence would sit there
        looking configured while detecting nothing.
        """
        self._require(perms.CONFIGURE_DETECTORS)
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        question = (question or "").strip()
        if question:
            cam["custom_rule"] = {"question": question, "dwell": float(dwell)}
        else:
            cam.pop("custom_rule", None)
        f = self._zones_file(camera_id)
        data = json.loads(f.read_text()) if f.exists() else {"zones": []}
        if question and not data["zones"]:
            f.parent.mkdir(parents=True, exist_ok=True)
            data["zones"] = [{"name": "everywhere", "kind": "restricted",
                              "anchors": ["BOTTOM_CENTER"], "dwell_alert_seconds": 999999,
                              "polygon": [[0, 0], [99999, 0], [99999, 99999], [0, 99999]]}]
            f.write_text(json.dumps(data, indent=2))
            cam["zones"] = str(f)
        self._regen_zone_rules(camera_id, cam, data["zones"])
        onboarding.add_camera(self.site_path, cam)
        self.audit.record(self.current_user.username, "config_change",
                          f"camera:{camera_id}",
                          detail={"custom_rule": question[:120] or "(removed)"})
        return {"ok": True, "question": question,
                "note": "applies the next time monitoring starts"}

    def add_zone(self, camera_id: str, name: str, points: list, dwell_seconds: float = 5.0) -> dict:
        """Save a drawn zone (>=3 [x,y] points in ORIGINAL pixels) + wire a
        loitering rule for it. Takes effect on the next Start monitoring."""
        self._require(perms.CONFIGURE_CAMERAS)
        pts = [[int(p[0]), int(p[1])] for p in (points or []) if len(p) == 2]
        if len(pts) < 3:
            return {"error": "a zone needs at least 3 points"}
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        f = self._zones_file(camera_id)
        f.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(f.read_text()) if f.exists() else {"zones": []}
        data["zones"] = [z for z in data.get("zones", []) if z.get("name") != name]
        data["zones"].append({"name": name or "zone", "kind": "restricted",
                              "anchors": ["BOTTOM_CENTER"], "dwell_alert_seconds": float(dwell_seconds),
                              "polygon": pts})
        f.write_text(json.dumps(data, indent=2))
        cam["zones"] = str(f)
        self._regen_zone_rules(camera_id, cam, data["zones"])
        onboarding.add_camera(self.site_path, cam)
        return {"ok": True, "zones": data["zones"]}

    def remove_zone(self, camera_id: str, name: str) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        f = self._zones_file(camera_id)
        data = json.loads(f.read_text()) if f.exists() else {"zones": []}
        data["zones"] = [z for z in data.get("zones", []) if z.get("name") != name]
        f.write_text(json.dumps(data, indent=2))
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam:
            if data["zones"]:
                self._regen_zone_rules(camera_id, cam, data["zones"])
            else:                                   # no zones left -> restore base preset
                cam["config"] = cam.get("_base_config", cam.get("config"))
                cam.pop("zones", None)
            onboarding.add_camera(self.site_path, cam)
        return {"ok": True, "zones": data["zones"]}

    # --- scene context + custom (customer-defined) threats ---
    def scene_context(self, camera_id: str) -> dict | None:
        """What this camera watches — the 'place'. From live agent-mapping output
        (runs/context/<cam>/scene_context.json) or static config fields."""
        p = Path("runs/context") / camera_id / "scene_context.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (ValueError, OSError):
                pass
        cam = next((c for c in self.list_cameras() if c.get("id") == camera_id), None)
        if cam and cam.get("scene_description"):
            return {"environment_type": cam.get("environment_type", "unknown"),
                    "scene_description": cam["scene_description"]}
        return None

    def _cam(self, cams: list, camera_id: str):
        return next((c for c in cams if c.get("id") == camera_id), None)

    def add_custom_threat(self, camera_id: str, name: str, description: str) -> dict:
        """A customer-defined threat in plain English — the VLM gate evaluates it
        in this camera's scene context. Native detectors are kept alongside."""
        if not (description or "").strip():
            return {"error": "describe what to watch for"}
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        threats = cam.get("custom_threats") or []
        threats.append({"name": (name or "custom").strip(), "description": description.strip()})
        cam["custom_threats"] = threats
        onboarding.add_camera(self.site_path, cam)
        return {"ok": True, "custom_threats": threats}

    def remove_custom_threat(self, camera_id: str, index: int) -> dict:
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        threats = cam.get("custom_threats") or []
        if 0 <= index < len(threats):
            threats.pop(index)
        cam["custom_threats"] = threats
        onboarding.add_camera(self.site_path, cam)
        return {"ok": True, "custom_threats": threats}

    # --- first-run setup wizard ---
    def get_site(self) -> dict:
        meta = onboarding.get_site_meta(self.site_path)
        # Playback demo with no real cameras: skip the wizard, land on the dash.
        if meta["camera_count"] == 0 and self._demo:
            meta["configured"] = True
            meta["name"] = "Demo Store"
            meta["camera_count"] = len(self._live_sources(99))
        return meta

    def set_site(self, name: str | None = None, notify: str | None = None) -> dict:
        self._require(perms.CONFIGURE_SITE)
        return onboarding.set_site_meta(self.site_path, name=name, notify=notify)

    def mark_configured(self) -> dict:
        return onboarding.set_site_meta(self.site_path, configured=True)

    def send_test_notification(self) -> dict:
        """Fire a synthetic alert through the site's configured notifier so the
        operator can confirm Telegram/WhatsApp/webhook actually reaches their phone."""
        from cvti.serving.alert_sink import build_notifier
        meta = onboarding.get_site_meta(self.site_path)
        notify = (meta.get("notify") or "console").strip()
        event = {
            "ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "camera_id": "test_camera", "rule": "test_alert", "priority": "high",
            "confidence": 0.99, "zone": None, "track_id": None, "object_label": None,
            "reason": "✅ Test alert from Argus — your notifications are working.",
            "evidence_dir": None,
        }
        try:
            build_notifier(notify).notify(event)
            return {"ok": True, "via": notify}
        except Exception as exc:  # noqa: BLE001
            log.warning("test notification failed", exc_info=True)
            return {"ok": False, "via": notify, "error": str(exc)[:200]}

    # --- diagnostics ---
    # --- retention / legal hold -------------------------------------------
    def set_legal_hold(self, event_id: int, hold: bool = True) -> dict:
        """Exempt an event's evidence from retention purge, or release it."""
        self._require(perms.MANAGE_LEGAL_HOLD)
        db, _ = self._effective_db()
        try:
            con = self._connect(db)
            con.execute("UPDATE events SET legal_hold = ? WHERE id = ?",
                        (1 if hold else 0, int(event_id)))
            con.commit()
            con.close()
        except sqlite3.OperationalError as exc:
            log.warning("legal hold update failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:200]}
        log.info("legal hold %s for event %s", "set" if hold else "released", event_id)
        return {"ok": True, "id": int(event_id), "legal_hold": bool(hold)}

    def retention_status(self) -> dict:
        """Policy, disk, and what is being retained past expiry and why."""
        from cvti.serving.retention import RetentionManager, RetentionPolicy
        meta = self.get_site()
        policy = RetentionPolicy.from_site(meta)
        out_dir = Path(self.db_path).parent
        db, _ = self._effective_db()
        status = RetentionManager(out_dir, policy, db_path=db).status()
        # The engine's own view wins when it is running — it is the process that
        # actually purges.
        engine = (self._gate_health() or {}).get("retention")
        if engine:
            status["last_run"] = engine.get("last_run") or status["last_run"]
        return status

    def set_retention(self, days: float = None) -> dict:
        self._require(perms.CONFIGURE_SITE)
        return onboarding.set_site_meta(self.site_path, retention_days=days)

    def export_evidence(self, event_ids: str = "", dest: str = "") -> dict:
        """Zip an event's evidence so a customer can keep it past expiry.

        Deliberately the opposite of the diagnostics bundle: that one excludes
        evidence because it is going to us; this one IS the evidence, and is
        going to the person who owns it.
        """
        self._require(perms.EXPORT_EVIDENCE)
        import zipfile
        db, base = self._effective_db()
        ids = [int(x) for x in str(event_ids).split(",") if str(x).strip().isdigit()]
        out_dir = Path(self.db_path).parent
        target = Path(dest) if dest else (out_dir / f"argus-evidence-{int(time.time())}.zip")
        try:
            con = self._connect(db)
            sql = "SELECT * FROM events"
            if ids:
                sql += f" WHERE id IN ({','.join('?' * len(ids))})"
            rows = con.execute(sql, ids).fetchall()
            con.close()
        except sqlite3.OperationalError as exc:
            log.warning("evidence export query failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:200]}

        exported = 0
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for row in rows:
                ev = row["evidence_dir"]
                if not ev:
                    continue
                path = Path(ev)
                if not path.is_absolute() and base:
                    path = base / path
                if not path.exists():
                    continue
                for f in sorted(path.rglob("*")):
                    if f.is_file():
                        zf.write(f, f"event_{row['id']}/{f.relative_to(path)}")
                exported += 1
            zf.writestr("MANIFEST.txt",
                        "Argus evidence export\n"
                        f"events: {exported}\n\n"
                        "CONTAINS camera images and video of identifiable people.\n"
                        "Handle under the same data-protection terms as the system itself.\n")
        log.info("exported evidence for %d event(s) -> %s", exported, target)
        return {"ok": True, "events": exported, "path": str(target),
                "size_kb": round(target.stat().st_size / 1024, 1)}

    def camera_links(self) -> list[dict]:
        """Per-camera link state from the running engine.

        Deliberately reports "unknown" when the engine is not running rather
        than "connected": claiming coverage we cannot observe is the exact
        failure this exists to prevent.
        """
        health = self._gate_health()
        if not health:
            return [{"camera_id": c["id"], "state": "unknown", "time_in_state": 0,
                     "reconnects": 0} for c in self.list_cameras()]
        return health.get("cameras") or []

    def download_diagnostics(self) -> dict:
        """Zip logs + a health snapshot for support. Never includes evidence.

        Returns the path rather than the bytes: the operator sends us a file,
        and keeping it on disk means they can inspect it before they do.
        """
        self._require(perms.VIEW_DIAGNOSTICS)
        from cvti.diagnostics import build_bundle
        out_dir = Path(self.db_path).parent
        try:
            path = build_bundle(out_dir)
        except Exception as exc:  # noqa: BLE001 - support tooling must not crash the app
            log.exception("diagnostics bundle failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:200]}
        return {"ok": True, "path": str(path),
                "size_kb": round(path.stat().st_size / 1024, 1)}

    # --- config backup / restore (EP-08-T1) --------------------------------
    def _backup_dir(self):
        meta = onboarding.get_site_meta(self.site_path)
        d = (meta.get("backup_dir") or "").strip() if isinstance(meta, dict) else ""
        return d or None

    def _maybe_auto_backup(self) -> None:
        """One versioned backup per calendar day, on app start."""
        from cvti import backup as _backup
        dest = Path(self._backup_dir() or _backup._default_dir())
        today = time.strftime("%Y%m%d")
        if any(dest.glob(f"argus-config-{today}_*.zip")):
            return
        _backup.backup_config(self.site_path, dest)

    def backup_now(self) -> dict:
        self._require(perms.CONFIGURE_SITE)
        from cvti import backup as _backup
        out = _backup.backup_config(self.site_path, self._backup_dir())
        self.audit.record(self.current_user.username, "config_change", "backup",
                          detail={"path": out.get("path")})
        return out

    def list_backups(self) -> list:
        self._require(perms.CONFIGURE_SITE)
        from cvti import backup as _backup
        return _backup.list_backups(self._backup_dir())

    def restore_backup(self, zip_path: str) -> dict:
        self._require(perms.CONFIGURE_SITE)
        from cvti import backup as _backup
        out = _backup.restore_config(zip_path, self.site_path)
        self.audit.record(self.current_user.username, "config_change", "restore",
                          detail={"path": zip_path, "ok": out.get("ok")})
        return out

    def set_backup_dir(self, path: str) -> dict:
        self._require(perms.CONFIGURE_SITE)
        onboarding.set_site_meta(self.site_path, backup_dir=(path or "").strip())
        return {"ok": True, "backup_dir": (path or "").strip()}

    def gate_status(self, model: str = vlm.DEFAULT_MODEL) -> dict:
        """Ollama reachability + the running engine's own view of the gate.

        Two different failures look identical from the operator's chair: Ollama
        being down, and the gate erroring on every alert while Ollama is up. The
        first comes from probing localhost, the second only the engine knows —
        it publishes it to gate_health.json.
        """
        status = vlm.gate_status(model)
        status["engine"] = self._gate_health()
        # Whether an Ollama binary ships inside this app: the UI's "offline"
        # advice differs — "click Download" beats "go install ollama.com".
        try:
            from cvti.verification import ollama as _ollama
            status["runtime_bundled"] = _ollama.ollama_binary() is not None
        except Exception:  # noqa: BLE001
            log.debug("could not resolve a local ollama binary", exc_info=True)
            status["runtime_bundled"] = False
        return status

    def _gate_health(self) -> dict:
        """Gate stats published by the engine subprocess. Empty when it isn't running."""
        path = Path(self.db_path).parent / "gate_health.json"
        try:
            health = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        # Stale file from a previous run is worse than no file — it would show a
        # green gate for an engine that exited an hour ago.
        if time.time() - float(health.get("updated_at") or 0) > 30:
            return {}
        return health

    def pull_model(self, model: str = vlm.DEFAULT_MODEL) -> dict:
        # The user clicked Download: if no server is answering, start the
        # bundled/installed one now so the pull has somewhere to go. Ollama's
        # pull resumes partial downloads natively, so a killed 3 GB download
        # continues instead of restarting.
        try:
            from cvti.verification import ollama as _ollama
            _ollama.ensure_server()
        except Exception:  # noqa: BLE001
            log.warning("could not start the local VLM server", exc_info=True)
        return vlm.start_pull(model)

    def pull_progress(self, model: str = vlm.DEFAULT_MODEL) -> dict:
        return vlm.pull_progress(model)

    # --- live wall (multi-camera video grid) ---
    def _live_sources(self, count: int) -> list[dict]:
        """Sources for the live grid: the site's file/RTSP cameras if configured,
        otherwise fall back to demo clips in data/test_clips/."""
        cams = [c for c in self.list_cameras() if c.get("source")]
        if cams:
            return [{"id": c["id"], "source": c["source"]} for c in cams[:count]]
        if self._demo and (self._demo / "clips").exists():
            clips = sorted((self._demo / "clips").glob("*.mp4"))[:count]
            return [{"id": p.stem, "source": str(p)} for p in clips]
        clips = sorted(Path("data/test_clips").glob("*.mp4"))[:count]
        return [{"id": p.stem, "source": str(p)} for p in clips]

    def _engine_frame_port(self) -> tuple:
        """The engine's frame-publisher port, if it's running and serving.

        When the engine is up it has already decoded every stream and knows where
        everyone is, so we display ITS frames (with boxes) rather than opening the
        same videos again — decode is the dominant per-camera cost."""
        try:
            info = json.loads((Path(self.db_path).parent / "frames.json").read_text())
            port = int(info.get("port") or 0)
            token = str(info.get("token") or "")
        except Exception as exc:  # noqa: BLE001
            log.debug("engine frame port unreadable", exc_info=True)
            return 0, ""
        if not port:
            return 0, ""
        try:      # only trust it if it actually answers — WITH the token: the
            # publisher 401s tokenless probes, which made this check always
            # fail and the app silently re-decode every stream itself.
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{port}/cameras",
                                         headers={"X-Argus-Token": token})
            with urllib.request.urlopen(req, timeout=1.5) as r:
                cams = json.loads(r.read().decode()).get("cameras") or []
            return (port, token) if cams else (0, "")
        except Exception as exc:  # noqa: BLE001
            log.debug("engine frame port unreachable", exc_info=True)
            return 0, ""

    def live_start(self, count: int = 6) -> dict:
        self._require(perms.VIEW_LIVE)
        from cvti.app.live_wall import FrameServer, LiveWall
        self.live_stop()
        # Prefer the engine's already-decoded frames (no second decode, live boxes).
        port, token = self._engine_frame_port()
        if port:
            try:
                import urllib.request
                req = urllib.request.Request(f"http://127.0.0.1:{port}/cameras",
                                             headers={"X-Argus-Token": token})
                with urllib.request.urlopen(req, timeout=1.5) as r:
                    cams = json.loads(r.read().decode()).get("cameras") or []
                # The token MUST travel with the port: the UI's <img> tags are
                # refused without it, which is exactly the broken-tile bug.
                return {"cameras": [{"id": c} for c in cams], "port": port,
                        "token": token, "source": "engine"}
            except Exception as exc:  # noqa: BLE001
                log.debug("engine frames unavailable; decoding locally", exc_info=True)
                pass          # fall through to decoding ourselves
        sources = self._live_sources(count)
        if not sources:
            return {"cameras": [], "port": 0}
        self._live = LiveWall(sources, fps=10).start()
        self._fs = FrameServer(self._live)
        port = self._fs.start()
        # cameras + the localhost port the UI fetches JPEG frames from
        return {"cameras": [{"id": s["id"]} for s in sources], "port": port,
                "token": self._fs.token, "source": "app"}

    def live_frames(self) -> dict:
        return self._live.frames() if self._live else {}

    def live_stop(self) -> dict:
        if self._fs:
            self._fs.stop()
            self._fs = None
        if self._live:
            self._live.stop()
            self._live = None
        return {"stopped": True}

    # --- monitoring engine (Start/Stop) ---
    # Launches the full detection pipeline (YOLO + VideoMAE + Gemma gate) as a
    # subprocess pointed at this site, writing confirmed alerts into events.db.
    # In dev that is `python -m cvti.serving.pipeline`; in the installed app it
    # is the argus-engine executable shipped INSIDE the bundle (EP-05-T1) — the
    # installer that "cannot detect anything on its own" is exactly what the
    # audit said we must stop shipping.
    @staticmethod
    def _bundled_engine() -> "Path | None":
        """The engine executable next to this app's own, if we're a bundle."""
        if not getattr(sys, "frozen", False):
            return None
        exe = "argus-engine.exe" if sys.platform == "win32" else "argus-engine"
        candidate = Path(sys.executable).parent / exe
        return candidate if candidate.exists() else None

    def _engine_command(self) -> list:
        engine = self._bundled_engine()
        if engine is not None:
            return [str(engine)]
        return [sys.executable, "-m", "cvti.serving.pipeline"]

    def _spawn_engine(self) -> "subprocess.Popen":
        out_dir = Path(self.db_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        notify = self.get_site().get("notify") or "console"
        log_file = open(out_dir / "monitor.log", "a")  # noqa: SIM115 - lives with the subprocess
        # Lean defaults keep the box cool: lower fps + image size cut compute a lot
        # with negligible quality loss at demo scale.
        # The gate needs the local Ollama server; in the bundled app nobody has
        # run `ollama serve` in a terminal — that is the point — so bring up the
        # bundled runtime if nothing is answering. Best-effort: if it still is
        # not up, the gate stays fail-visible and alerts arrive UNVERIFIED.
        try:
            from cvti.verification import ollama as _ollama
            _ollama.ensure_server()
        except Exception:  # noqa: BLE001 - engine start must not die on this
            log.warning("could not ensure the local VLM server", exc_info=True)
        cmd = self._engine_command() + [
               "--site-config", self.site_path,
               "--gate-provider", "ollama", "--gate-model", "gemma3:4b",
               "--notify", notify, "--output-dir", str(out_dir),
               "--target-fps", "4", "--imgsz", "512",
               "--seconds", "100000", "--gate-drain", "60"]
        kwargs = {}
        if sys.platform == "win32":
            # The engine is a console-mode exe; from the windowed app that would
            # flash a terminal at the user. Same flag is a no-op run from a shell.
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, **kwargs)

    def start_monitoring(self) -> dict:
        # A packaged app has no engine (torch/Ollama) inside it — it's a playback
        # demo. Don't try to spawn; the recorded alerts are already shown.
        self._require(perms.CONTROL_ENGINE)
        if getattr(sys, "frozen", False) and self._bundled_engine() is None:
            # A lean viewer-only build (no engine inside). The full installer
            # ships argus-engine and never takes this branch.
            return {"running": False, "demo": True,
                    "note": "Playback demo — alerts are pre-recorded. This build has no detection engine inside."}
        if self._monitor and self._monitor.poll() is None:
            return {"running": True, "pid": self._monitor.pid, "already": True}
        self._monitor_should_run = True
        self._restarts = 0
        self._monitor = self._spawn_engine()
        self._start_watchdog()
        return {"running": True, "pid": self._monitor.pid}

    def _start_watchdog(self, max_restarts: int = 5) -> None:
        """Respawn the engine if it dies unexpectedly (crash / OOM), up to a cap
        so a genuinely broken config can't loop forever."""
        import threading
        if getattr(self, "_watchdog", None) and self._watchdog.is_alive():
            return

        def loop():
            started = time.time()
            while getattr(self, "_monitor_should_run", False):
                time.sleep(3)
                if not getattr(self, "_monitor_should_run", False):
                    break
                if self._monitor and self._monitor.poll() is not None:   # died
                    # An engine that ran for an hour+ did not crash-loop: the
                    # spawn caps --seconds (~28h), so long-lived exits are
                    # SCHEDULED. Without this reset, a 24/7 site burned one
                    # restart per day and the watchdog gave up inside a week —
                    # silently, on a monitoring product. (Audit 23 Aug, #1.)
                    if time.time() - started > 3600:
                        self._restarts = 0
                    started = time.time()
                    if self._restarts < max_restarts:
                        self._restarts += 1
                        log.info(f"[watchdog] engine exited unexpectedly — restarting "
                              f"({self._restarts}/{max_restarts})")
                        self._monitor = self._spawn_engine()
                    else:
                        log.info("[watchdog] engine died too many times — giving up")
                        self._monitor_should_run = False

        self._watchdog = threading.Thread(target=loop, name="engine-watchdog", daemon=True)
        self._watchdog.start()

    def stop_monitoring(self) -> dict:
        self._require(perms.CONTROL_ENGINE)
        self._monitor_should_run = False   # tell the watchdog this is intentional
        if self._monitor and self._monitor.poll() is None:
            self._monitor.terminate()
            try:
                self._monitor.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._monitor.kill()
        self._monitor = None
        return {"running": False}

    def monitoring_status(self) -> dict:
        running = bool(self._monitor and self._monitor.poll() is None)
        return {"running": running, "pid": (self._monitor.pid if running else None)}

    # --- feed source switcher: flip between demo videos and live cameras ---
    def _feeds_registry(self) -> dict:
        p = Path("configs/feeds.json")
        if not p.exists():
            return {"sources": []}
        try:
            return json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            log.warning("feeds registry unreadable; no feeds offered", exc_info=True)
            return {"sources": []}

    def feed_sources(self) -> dict:
        """The switchable feed sources + which one is active (matched by config path)."""
        reg = self._feeds_registry()
        active = None
        srcs = []
        for s in reg.get("sources", []):
            is_active = str(Path(s.get("config", "")).resolve()) == str(Path(self.site_path).resolve())
            if is_active:
                active = s["key"]
            srcs.append({"key": s["key"], "label": s["label"], "kind": s.get("kind", "demo")})
        return {"sources": srcs, "active": active}

    def switch_feed(self, key: str) -> dict:
        """Start switching the app (and, if running, the engine) to another feed.

        Returns IMMEDIATELY and does the work on a background thread — resolving
        live stream URLs takes seconds per feed and restarting the engine takes
        more, and doing that inline would freeze the Qt UI thread. Poll
        feed_switch_status() for progress."""
        self._require(perms.CONFIGURE_SITE)
        import threading
        st = getattr(self, "_switch_state", None)
        if st and st.get("busy"):
            return {"ok": False, "busy": True, "status": st.get("status", "switching…")}
        reg = self._feeds_registry()
        src = next((s for s in reg.get("sources", []) if s["key"] == key), None)
        if not src:
            return {"ok": False, "error": f"unknown feed source '{key}'"}
        self._switch_state = {"busy": True, "status": "starting…", "error": None,
                              "active": None, "done": False}
        threading.Thread(target=self._do_switch, args=(src, key),
                         name="feed-switch", daemon=True).start()
        return {"ok": True, "busy": True, "status": "switching…"}

    def feed_switch_status(self) -> dict:
        """Progress of an in-flight switch_feed (the UI polls this)."""
        return dict(getattr(self, "_switch_state", {"busy": False, "done": True}))

    def _do_switch(self, src: dict, key: str) -> None:
        st = self._switch_state
        try:
            if src.get("kind") == "live":
                st["status"] = "resolving live streams…"
                res = self._resolve_live_config(src)
                if not res.get("ok"):
                    st.update(busy=False, done=True, error=res.get("error", "resolve failed"))
                    return
            was_running = bool(self._monitor and self._monitor.poll() is None)
            if was_running:
                st["status"] = "stopping engine…"
                self.stop_monitoring()
            self.site_path = src["config"]
            restarted = False
            # The frozen guard predated EP-05: the installed bundle SHIPS an
            # engine now, so a feed switch there used to stop monitoring and
            # report "done" — permanently dark. start_monitoring itself knows
            # how to refuse on a truly engine-less lean build.
            # (Audit 23 Aug, #3.)
            if was_running:
                st["status"] = "restarting engine…"
                out = self.start_monitoring()
                restarted = bool(out.get("running"))
            st.update(busy=False, done=True, active=key, error=None,
                      kind=src.get("kind", "demo"), config=src["config"],
                      engine_restarted=restarted, status="done")
        except Exception as exc:  # noqa: BLE001 - a failed switch must not wedge the UI
            log.error("feed switch failed", exc_info=True)
            st.update(busy=False, done=True, error=str(exc)[:200], status="failed")

    def _resolve_live_config(self, src: dict) -> dict:
        """Resolve each YouTube id to a fresh HLS URL (yt-dlp) and write the live
        config with running + the shared loitering watch-zone."""
        import shutil
        if not shutil.which("yt-dlp"):
            return {"ok": False, "error": "yt-dlp not installed — needed for live feeds. pip install yt-dlp"}
        from concurrent.futures import ThreadPoolExecutor

        def _resolve_one(entry):
            name, vid = entry
            try:
                out = subprocess.run(
                    ["yt-dlp", "-g", "--extractor-args", "youtube:player_client=android",
                     "-f", "best[height<=720]/best", f"https://www.youtube.com/watch?v={vid}"],
                    capture_output=True, text=True, timeout=25)
                return name, ((out.stdout or "").strip().splitlines() or [""])[0]
            except Exception as exc:  # noqa: BLE001 - a dead feed shouldn't sink the others
                log.debug("camera place lookup failed", exc_info=True)
                return name, ""

        feeds = list(src.get("youtube", []))
        cams = []
        # Resolve every feed CONCURRENTLY — sequential resolves made the switch
        # take ~4x longer than it needed to.
        with ThreadPoolExecutor(max_workers=min(6, max(1, len(feeds)))) as pool:
            for name, url in pool.map(_resolve_one, feeds):
                if url:
                    cams.append({
                        "id": name, "source": url,
                        "config": "configs/rules/live_watch.json",
                        "zones": "configs/zones/live_watch.json",
                        "environment_type": "public area",
                        "scene_description": "A public street/plaza/terminal; a person lingering (loitering) or running may signal an incident.",
                        "running": True, "running_min_speed_ratio": 0.08,
                        "running_min_frames": 3})
        if not cams:
            return {"ok": False, "error": "could not resolve any live feed (try `yt-dlp -U`)"}
        Path(src["config"]).write_text(json.dumps(
            {"name": "Live Dashboard", "notify": "console", "configured": True, "cameras": cams}, indent=2))
        return {"ok": True, "resolved": len(cams)}

    def setup_state(self) -> dict:
        """Everything the wizard needs to decide whether to show + where to resume."""
        meta = self.get_site()
        return {
            "configured": meta["configured"],
            "site_name": meta["name"],
            "notify": meta["notify"],
            "cameras": meta["camera_count"],
            "gate": self.gate_status(),
            # playback demo (bundled, no real cameras) -> app opens on the Live wall
            "demo": bool(self._demo and not self.list_cameras()),
        }

    # --- events / review ---
    def _connect(self, path: str | None = None) -> sqlite3.Connection:
        con = sqlite3.connect(path or self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _effective_db(self) -> tuple[str, "Path | None"]:
        """The DB to read + a base dir to resolve evidence frames against.

        Real DB always wins. The bundled playback demo is used ONLY when there
        are no real cameras configured — so a live site opens empty and fills as
        detection happens (never shows pre-recorded alerts before monitoring).
        """
        if Path(self.db_path).exists():
            return self.db_path, None
        if self._demo and not self.list_cameras():
            return str(self._demo / "events.db"), self._demo
        return self.db_path, None

    def list_events(self, limit: int = 100, embed_frames: bool = True) -> list[dict]:
        self._require(perms.VIEW_ALERTS)
        db, frame_base = self._effective_db()
        try:
            con = self._connect(db)
        except sqlite3.OperationalError as exc:
            # "No alerts" and "the events database is unreadable" must never
            # look the same on an operator's screen. (Audit 23 Aug, #12.)
            log.error("events db unreadable", exc_info=True)
            return {"error": f"events database unavailable: {str(exc)[:120]}"}
        try:
            rows = con.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError as exc:
            con.close()
            if "no such table" in str(exc):
                return []          # a fresh site with no events yet IS quiet
            log.error("events query failed", exc_info=True)
            return {"error": f"events database unavailable: {str(exc)[:120]}"}
        con.close()
        out = []
        for r in rows:
            e = dict(r)
            e["review"] = e.get("review") or "new"
            if embed_frames:
                e["frames"] = self._frames_as_data_uris(e.get("evidence_dir"), frame_base)
                e["subject"] = self._subject_uri(e.get("evidence_dir"), frame_base)
            out.append(e)
        return out

    def event_clip(self, evidence_dir: str | None) -> dict:
        """Return the event's real-video clip.mp4 as a data URI (lazy, per-selection).

        Works for both a live run (absolute evidence_dir) and the bundled playback
        demo (evidence_dir relative to the demo bundle)."""
        self._require(perms.VIEW_ALERTS)
        _, frame_base = self._effective_db()
        d = Path(evidence_dir or "")
        if not d.exists() and frame_base and evidence_dir:
            d = frame_base / evidence_dir
        # Return ALL the event's frames (for the smooth image cine-loop the app plays)
        # plus the mp4 as a data URI (archival / download).
        frames = self._frames_as_data_uris(evidence_dir, frame_base, cap=120)
        clip = d / "clip.mp4"
        uri = None
        if clip.exists():
            uri = "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode()
        return {"uri": uri, "frames": frames}

    def search_events(self, query: str, limit: int = 200) -> dict:
        """Ask-your-cameras: natural-language search over past events.

        TrueSight reads a compact catalogue of events and returns the ones that
        match the plain-English query (e.g. 'anyone near the till after 6pm').
        Falls back to keyword matching if the local model isn't reachable."""
        query = (query or "").strip()
        if not query:
            return {"query": "", "matches": [], "answer": "", "engine": "none"}
        events = self.list_events(limit, embed_frames=False)
        if not events:
            return {"query": query, "matches": [], "answer": "No events recorded yet.", "engine": "none"}
        catalogue = "\n".join(
            f"[{e['id']}] {e.get('iso','')} cam={e.get('camera_id')} rule={e.get('rule')} "
            f"zone={e.get('zone') or '-'} :: {e.get('reason','')}" for e in events)
        ids, answer, engine = self._vlm_search(query, catalogue)
        if ids is None:                     # local model unavailable -> keyword fallback
            terms = [t for t in query.lower().split() if len(t) > 2]
            ids = [e["id"] for e in events if any(
                t in (str(e.get("reason", "")) + " " + str(e.get("rule", "")) + " "
                      + str(e.get("camera_id", "")) + " " + str(e.get("zone", ""))).lower()
                for t in terms)]
            answer, engine = "", "keyword"
        idset = set(ids)
        matches = [e for e in events if e["id"] in idset]
        _, frame_base = self._effective_db()
        for e in matches[:24]:              # attach evidence only to the matches shown
            e["frames"] = self._frames_as_data_uris(e.get("evidence_dir"), frame_base)
        return {"query": query, "matches": matches, "answer": answer, "engine": engine}

    def _vlm_search(self, query: str, catalogue: str):
        """Ask the local model which event IDs match. Returns (ids, answer, engine)
        or (None, '', '') if the model is unreachable (caller does keyword fallback)."""
        import urllib.error
        import urllib.request
        prompt = (
            "You are a security-footage search assistant. Below is a catalogue of past "
            "CCTV events (one per line, prefixed with [id]). Return ONLY the events that "
            "match the user's query. Reason over the description, camera, zone, and time.\n"
            "Respond with a single JSON object and nothing else:\n"
            '{"ids": [matching ids as integers], "answer": "one short sentence summarising what you found"}\n\n'
            f"EVENTS:\n{catalogue}\n\nQUERY: {query}\n")
        payload = {"model": "gemma3:4b", "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}
        try:
            req = urllib.request.Request(
                "http://localhost:11434/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=40) as r:
                body = json.loads(r.read().decode("utf-8"))
            txt = body["choices"][0]["message"]["content"]
            data = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            ids = [int(i) for i in data.get("ids", []) if str(i).strip().lstrip("-").isdigit()]
            return ids, str(data.get("answer", "")).strip(), "TrueSight"
        except Exception as exc:  # noqa: BLE001 - unreachable/parse error -> keyword fallback
            log.debug("evidence lookup failed", exc_info=True)
            return None, "", ""

    def _frames_as_data_uris(self, evidence_dir: str | None,
                             frame_base: "Path | None" = None, cap: int = 5) -> list[str]:
        d = Path(evidence_dir or "")
        if not d.exists() and frame_base and evidence_dir:
            d = frame_base / evidence_dir       # bundled demo: paths are relative
        if not d.exists():
            return []
        uris = []
        for p in sorted(d.iterdir()):
            # subject.jpg is the annotated "who" shot, shown on its own — including
            # it here would make a box flash at the end of every cine-loop.
            if p.name == "subject.jpg":
                continue
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and len(uris) < cap:
                b64 = base64.b64encode(p.read_bytes()).decode()
                uris.append(f"data:image/jpeg;base64,{b64}")
        return uris

    def _subject_uri(self, evidence_dir: str | None,
                     frame_base: "Path | None" = None) -> str | None:
        """The annotated frame with the subject boxed, if one was saved."""
        d = Path(evidence_dir or "")
        if not d.exists() and frame_base and evidence_dir:
            d = frame_base / evidence_dir
        p = d / "subject.jpg"
        if not p.exists():
            return None
        return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()

    def set_review(self, event_id: int | str, label: str) -> dict:
        """Legacy label entry point — routed through the state machine so every
        transition carries an owner and lands in the audit trail."""
        self._require(perms.REVIEW_ALERTS)
        if label not in _REVIEW_VALUES:
            raise ValueError(f"review must be one of {_REVIEW_VALUES}")
        mapping = {"true": ("resolve", "real"), "false": ("resolve", "false_alarm"),
                   "ack": ("acknowledge", None)}
        action, outcome = mapping[label]
        if action == "acknowledge":
            return self.acknowledge_alert(event_id)
        return self.resolve_alert(event_id, outcome)

    def _triage_connect(self):
        """Write to the SAME db we read from. The bundled read-only demo can't
        be written — callers degrade gracefully rather than showing a modal."""
        db, _ = self._effective_db()
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        con = self._connect(db)
        from cvti import triage
        triage.ensure_columns(con)
        return con

    def _actor(self) -> str:
        user = self.current_user
        return user.username if user else "<unknown>"

    def acknowledge_alert(self, event_id: int | str) -> dict:
        """Claim it. Everyone else sees your name against it."""
        self._require(perms.REVIEW_ALERTS)
        from cvti import triage
        try:
            con = self._triage_connect()
        except (sqlite3.OperationalError, OSError):
            return {"ok": False, "persisted": False}
        try:
            result = triage.acknowledge(con, int(event_id), self._actor())
        except triage.TriageError as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            con.close()
        self.audit.record(self._actor(), "alert_resolution", f"event:{event_id}",
                          {"transition": "acknowledged"})
        return {"id": event_id, "persisted": True, **result}

    def resolve_alert(self, event_id: int | str, outcome: str, note: str = "") -> dict:
        """Conclude it. The outcome feeds the model's feedback loop; the note is
        for the humans on the next shift."""
        self._require(perms.REVIEW_ALERTS)
        from cvti import triage
        try:
            con = self._triage_connect()
        except (sqlite3.OperationalError, OSError):
            return {"ok": False, "persisted": False}
        try:
            result = triage.resolve(con, int(event_id), self._actor(), outcome, note)
        except triage.TriageError as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            con.close()
        self.audit.record(self._actor(), "alert_resolution", f"event:{event_id}",
                          {"transition": "resolved", "outcome": outcome,
                           "note": bool(note)})
        return {"id": event_id, "persisted": True, **result}

    def export_incident_pdf(self, event_id: int | str) -> dict:
        """The incident record as a file that leaves the building intact —
        what a manager reviews, what goes to an insurer or the police."""
        self._require(perms.EXPORT_EVIDENCE)
        db, frame_base = self._effective_db()
        try:
            con = self._connect(db)
            row = con.execute("SELECT * FROM events WHERE id = ?",
                              (int(event_id),)).fetchone()
            con.close()
        except sqlite3.OperationalError as exc:
            return {"ok": False, "error": str(exc)[:200]}
        if row is None:
            return {"ok": False, "error": f"no incident #{event_id}"}
        event = dict(row)

        frames = []
        ev_dir = event.get("evidence_dir")
        if ev_dir:
            path = Path(ev_dir)
            if not path.is_absolute() and frame_base:
                path = frame_base / path
            if path.exists():
                for f in sorted(path.glob("frame_*.jpg"))[:8]:
                    try:
                        frames.append((f.name, f.read_bytes()))
                    except OSError:
                        log.debug("unreadable evidence frame %s", f, exc_info=True)
                subject = path / "subject.jpg"
                if subject.exists():
                    frames.insert(0, ("subject — boxed at the moment it fired",
                                      subject.read_bytes()))

        from cvti.incident_pdf import build_incident_pdf
        dest = Path(self.db_path).parent / f"argus-incident-{event['id']}.pdf"
        try:
            build_incident_pdf(event, frames, dest)
        except Exception as exc:  # noqa: BLE001 - export must not crash the app
            log.error("incident PDF failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:200]}
        self.audit.record(self._actor(), "evidence_export", f"event:{event_id}",
                          {"format": "pdf", "frames": len(frames)})
        return {"ok": True, "path": str(dest),
                "size_kb": round(dest.stat().st_size / 1024, 1)}

    def handover(self, hours: float = 8.0) -> dict:
        """What the incoming shift needs: what fired, what was concluded and by
        whom, and — flagged loudest — what is still open. Context must not
        reset at shift change."""
        self._require(perms.VIEW_ALERTS)
        db, _ = self._effective_db()
        since = time.time() - max(1.0, float(hours)) * 3600
        try:
            con = self._connect(db)
        except sqlite3.OperationalError:
            return {"hours": hours, "fired": [], "resolved": [], "open": [],
                    "counts": {}}
        state_expr = ("COALESCE(state, CASE WHEN review IN ('true','false') "
                      "THEN 'resolved' WHEN review='ack' THEN 'acknowledged' "
                      "ELSE 'new' END)")
        try:
            fired = [dict(r) for r in con.execute(
                "SELECT id, ts, iso, camera_id, rule, priority FROM events "
                "WHERE ts >= ? ORDER BY ts DESC", (since,))]
            resolved = [dict(r) for r in con.execute(
                f"SELECT id, iso, camera_id, rule, owner, outcome, note, resolved_at "
                f"FROM events WHERE {state_expr} = 'resolved' AND resolved_at >= ? "
                f"ORDER BY resolved_at DESC", (since,))]
            # Open items are NOT windowed: an incident from three shifts ago
            # that nobody concluded is precisely what must not be forgotten.
            open_items = [dict(r) for r in con.execute(
                f"SELECT id, ts, iso, camera_id, rule, priority, owner, "
                f"{state_expr} AS state FROM events "
                f"WHERE {state_expr} != 'resolved' ORDER BY ts ASC")]
        except sqlite3.OperationalError as exc:
            con.close()
            return {"hours": hours, "error": str(exc)[:200]}
        con.close()
        now = time.time()
        for item in open_items:
            item["age_h"] = round((now - (item.get("ts") or now)) / 3600, 1)
            item["carried_over"] = (item.get("ts") or now) < since
        outcomes = {}
        for r in resolved:
            outcomes[r.get("outcome") or "?"] = outcomes.get(r.get("outcome") or "?", 0) + 1
        return {"hours": float(hours), "since": since,
                "fired": fired[:100], "resolved": resolved[:50],
                "open": open_items[:50],
                "counts": {"fired": len(fired), "resolved": len(resolved),
                           "open": len(open_items), "outcomes": outcomes}}

    def needs_attention(self, min_priority: str = "medium") -> dict:
        """The Now view: one alert first, the queue behind it, who holds what."""
        self._require(perms.VIEW_ALERTS)
        from cvti import triage
        try:
            con = self._triage_connect()
        except (sqlite3.OperationalError, OSError):
            return {"now": None, "then": [], "waiting": 0, "held": []}
        try:
            out = triage.needs_attention(con, min_priority=min_priority)
        finally:
            con.close()
        db, frame_base = self._effective_db()
        if out["now"]:
            out["now"]["review"] = out["now"].get("review") or "new"
            out["now"]["frames"] = self._frames_as_data_uris(
                out["now"].get("evidence_dir"), frame_base)
            out["now"]["subject"] = self._subject_uri(
                out["now"].get("evidence_dir"), frame_base)
        return out

    def learning_stats(self) -> dict:
        """Feedback / reinforcement-training status for the Learning screen."""
        from cvti.feedback.manager import FeedbackManager
        db, _ = self._effective_db()
        return FeedbackManager(db).status()

    def learning_calibrate(self) -> dict:
        """Re-run online calibration from the operator's labels (writes calibration.json;
        the running engine hot-reloads it and stops paging on chronically-wrong rules)."""
        from cvti.feedback.manager import FeedbackManager
        db, _ = self._effective_db()
        return FeedbackManager(db).calibrate()

    # --- value surface ----------------------------------------------------
    def set_value_inputs(self, incident_value: "float | None" = None,
                         guard_hourly_cost: "float | None" = None,
                         review_minutes: "float | None" = None) -> dict:
        """The site's own money figures. Blank stays blank — see value_summary."""
        self._require(perms.CONFIGURE_SITE)
        return onboarding.set_site_meta(
            self.site_path, incident_value=incident_value,
            guard_hourly_cost=guard_hourly_cost, review_minutes=review_minutes)

    def value_summary(self, days: int = 30) -> dict:
        """What the system was worth over `days`, in the buyer's terms.

        Suppression percentage is an engineering metric. What a buyer is actually
        deciding about is: how many incidents did I get told about, how many
        false alarms did I not have to look at, and how much of my guards' shift
        did that give back.

        Every figure here is a count of real rows — `incidents` from the events
        table, the rest from the suppression ledger — so any number on the screen
        can be walked back to the events behind it. Nothing is modelled or
        extrapolated; the money figures are simply those counts multiplied by
        rates the site typed in, and are omitted entirely when it hasn't.
        """
        db, _ = self._effective_db()
        since = time.time() - max(1, int(days)) * 86400
        since_day = time.strftime("%Y-%m-%d", time.localtime(since))

        incidents = reviewed_true = reviewed_false = unverified = 0
        shown = rejected = deduped = errors = 0
        try:
            con = self._connect(db)
            try:
                row = con.execute(
                    "SELECT COUNT(*), "
                    "SUM(CASE WHEN review='true' THEN 1 ELSE 0 END), "
                    "SUM(CASE WHEN review='false' THEN 1 ELSE 0 END) "
                    "FROM events WHERE ts >= ? AND COALESCE(retracted,0)=0 "
                    "AND COALESCE(provisional,0)=0", (since,)).fetchone()
                incidents = row[0] or 0
                reviewed_true = row[1] or 0
                reviewed_false = row[2] or 0
            except sqlite3.OperationalError:
                pass
            try:
                # An unverified alert reached the operator because the gate could
                # NOT decide. Counting it as a detection would credit the product
                # for work it did not do.
                unverified = con.execute(
                    "SELECT COUNT(*) FROM events WHERE ts >= ? AND unverified = 1",
                    (since,)).fetchone()[0] or 0
            except sqlite3.OperationalError:
                unverified = 0
            try:
                row = con.execute(
                    "SELECT SUM(shown), SUM(rejected), SUM(deduped), SUM(errors) "
                    "FROM suppression_daily WHERE day >= ?", (since_day,)).fetchone()
                shown, rejected, deduped, errors = (v or 0 for v in row)
            except sqlite3.OperationalError:
                pass       # ledger only exists once an engine has run
            con.close()
        except sqlite3.OperationalError:
            pass

        # The gate only ever sees candidates the detectors produced, so raw is
        # exactly what an operator would have faced without verification.
        raw_alerts = shown + rejected + deduped
        # Kept apart on purpose. A rejected alert is one the AI looked at and
        # judged wrong; a deduped one is a repeat of an event already queued.
        # Both cost an operator attention, but only the first is a false alarm,
        # and rolling them together would overstate the claim that matters most.
        noise_removed = rejected + deduped
        meta = self.get_site()
        review_minutes = float(meta.get("review_minutes") or 0.0)
        hours_saved = noise_removed * review_minutes / 60.0

        money = {}
        guard_rate = float(meta.get("guard_hourly_cost") or 0.0)
        incident_value = float(meta.get("incident_value") or 0.0)
        if guard_rate > 0:
            money["attention_saved"] = round(hours_saved * guard_rate, 2)
        if incident_value > 0:
            money["incidents_value"] = round(incidents * incident_value, 2)

        return {
            "days": int(days),
            "incidents": max(0, incidents - unverified),   # gate actually confirmed these
            "unverified": unverified,               # surfaced because the gate could not decide
            "false_alarms_prevented": rejected,     # AI looked and said no
            "duplicates_collapsed": deduped,        # repeat of an event already queued
            "noise_removed": noise_removed,
            "raw_alerts": raw_alerts,               # what you would have seen
            "shown": shown,                         # what you were actually shown
            "suppression_pct": round(noise_removed / raw_alerts, 4) if raw_alerts else None,
            "attention_hours_saved": round(hours_saved, 2),
            "gate_errors": errors,                  # alerts the gate could not verify
            "operator_labels": {"true": reviewed_true, "false": reviewed_false},
            "inputs": {"review_minutes": review_minutes, "guard_hourly_cost": guard_rate,
                       "incident_value": incident_value},
            "money": money,
            # Three states, not two. A database with incidents but no ledger —
            # the bundled playback demo, or any run from before suppression was
            # recorded — would otherwise render "0 false alarms prevented, 0.0
            # hours saved", which reads as the product doing nothing rather than
            # as a measurement we never took.
            "has_data": bool(raw_alerts or incidents),
            "has_verification_history": bool(raw_alerts),
        }

    def counts(self) -> dict:
        """Header/nav summary numbers."""
        cams = self.list_cameras()
        n_cams = len(cams) if cams else len(self._live_sources(99))
        db, _ = self._effective_db()
        pending = 0
        try:
            con = self._connect(db)
            try:
                # "to review" = not yet handled. Ack/True/False all clear it.
                pending = con.execute(
                    "SELECT COUNT(*) FROM events WHERE review IS NULL "
                    "AND COALESCE(retracted, 0) = 0").fetchone()[0]
            except sqlite3.OperationalError:
                pending = 0
            con.close()
        except sqlite3.OperationalError:
            pending = 0
        return {"cameras": n_cams, "pending_alerts": pending}
