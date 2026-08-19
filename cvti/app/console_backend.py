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
from cvti.serving import onboarding, vlm

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
        return onboarding.test_url(url)

    def add_camera(self, camera: dict) -> list[dict]:
        return onboarding.add_camera(self.site_path, camera)

    def remove_camera(self, camera_id: str) -> list[dict]:
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

    def set_camera_rules(self, camera_id: str, rules: dict) -> dict:
        """Update which threat detectors run on a camera (+ optional rule preset).
        Takes effect on the next Start monitoring."""
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
        rdir = Path("configs/rules")
        rdir.mkdir(parents=True, exist_ok=True)
        rfile = rdir / f"{camera_id}.json"
        rfile.write_text(json.dumps({"use_case_id": f"{camera_id}_zones", "rules": rules}, indent=2))
        cam["config"] = str(rfile)

    def add_zone(self, camera_id: str, name: str, points: list, dwell_seconds: float = 5.0) -> dict:
        """Save a drawn zone (>=3 [x,y] points in ORIGINAL pixels) + wire a
        loitering rule for it. Takes effect on the next Start monitoring."""
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
        return onboarding.set_site_meta(self.site_path, retention_days=days)

    def export_evidence(self, event_ids: str = "", dest: str = "") -> dict:
        """Zip an event's evidence so a customer can keep it past expiry.

        Deliberately the opposite of the diagnostics bundle: that one excludes
        evidence because it is going to us; this one IS the evidence, and is
        going to the person who owns it.
        """
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
        from cvti.diagnostics import build_bundle
        out_dir = Path(self.db_path).parent
        try:
            path = build_bundle(out_dir)
        except Exception as exc:  # noqa: BLE001 - support tooling must not crash the app
            log.exception("diagnostics bundle failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:200]}
        return {"ok": True, "path": str(path),
                "size_kb": round(path.stat().st_size / 1024, 1)}

    def gate_status(self, model: str = vlm.DEFAULT_MODEL) -> dict:
        """Ollama reachability + the running engine's own view of the gate.

        Two different failures look identical from the operator's chair: Ollama
        being down, and the gate erroring on every alert while Ollama is up. The
        first comes from probing localhost, the second only the engine knows —
        it publishes it to gate_health.json.
        """
        status = vlm.gate_status(model)
        status["engine"] = self._gate_health()
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

    def _engine_frame_port(self) -> int:
        """The engine's frame-publisher port, if it's running and serving.

        When the engine is up it has already decoded every stream and knows where
        everyone is, so we display ITS frames (with boxes) rather than opening the
        same videos again — decode is the dominant per-camera cost."""
        try:
            info = json.loads((Path(self.db_path).parent / "frames.json").read_text())
            port = int(info.get("port") or 0)
        except Exception as exc:  # noqa: BLE001
            log.debug("engine frame port unreadable", exc_info=True)
            return 0
        if not port:
            return 0
        try:      # only trust it if it actually answers
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/cameras", timeout=1.5) as r:
                cams = json.loads(r.read().decode()).get("cameras") or []
            return port if cams else 0
        except Exception as exc:  # noqa: BLE001
            log.debug("engine frame port unreachable", exc_info=True)
            return 0

    def live_start(self, count: int = 6) -> dict:
        from cvti.app.live_wall import FrameServer, LiveWall
        self.live_stop()
        # Prefer the engine's already-decoded frames (no second decode, live boxes).
        port = self._engine_frame_port()
        if port:
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/cameras", timeout=1.5) as r:
                    cams = json.loads(r.read().decode()).get("cameras") or []
                return {"cameras": [{"id": c} for c in cams], "port": port, "source": "engine"}
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
        return {"cameras": [{"id": s["id"]} for s in sources], "port": port, "source": "app"}

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
    # Runs from-source / dev env (needs torch etc.); not from the lean app bundle.
    def _spawn_engine(self) -> "subprocess.Popen":
        out_dir = Path(self.db_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        notify = self.get_site().get("notify") or "console"
        log_file = open(out_dir / "monitor.log", "a")  # noqa: SIM115 - lives with the subprocess
        # Lean defaults keep the box cool: lower fps + image size cut compute a lot
        # with negligible quality loss at demo scale.
        cmd = [sys.executable, "-m", "cvti.serving.pipeline",
               "--site-config", self.site_path,
               "--gate-provider", "ollama", "--gate-model", "gemma3:4b",
               "--notify", notify, "--output-dir", str(out_dir),
               "--target-fps", "4", "--imgsz", "512",
               "--seconds", "100000", "--gate-drain", "60"]
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    def start_monitoring(self) -> dict:
        # A packaged app has no engine (torch/Ollama) inside it — it's a playback
        # demo. Don't try to spawn; the recorded alerts are already shown.
        if getattr(sys, "frozen", False):
            return {"running": False, "demo": True,
                    "note": "Playback demo — alerts are pre-recorded. Run from source for live monitoring."}
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
            while getattr(self, "_monitor_should_run", False):
                time.sleep(3)
                if not getattr(self, "_monitor_should_run", False):
                    break
                if self._monitor and self._monitor.poll() is not None:   # died
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
            if was_running and not getattr(sys, "frozen", False):
                st["status"] = "restarting engine…"
                self.start_monitoring()
                restarted = True
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
        db, frame_base = self._effective_db()
        try:
            con = self._connect(db)
        except sqlite3.OperationalError:
            return []
        try:
            rows = con.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError:
            con.close()
            return []
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
        if label not in _REVIEW_VALUES:
            raise ValueError(f"review must be one of {_REVIEW_VALUES}")
        # Write to the SAME db we read from, and make sure its folder exists.
        # A read-only bundled demo can't be written — degrade gracefully, no modal.
        db, _ = self._effective_db()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            Path(db).parent.mkdir(parents=True, exist_ok=True)
            con = self._connect(db)
        except (sqlite3.OperationalError, OSError):
            return {"id": event_id, "review": label, "reviewed_at": iso, "persisted": False}
        try:
            cols = {c[1] for c in con.execute("PRAGMA table_info(events)")}
            if "review" not in cols:
                con.execute("ALTER TABLE events ADD COLUMN review TEXT")
            if "reviewed_at" not in cols:
                con.execute("ALTER TABLE events ADD COLUMN reviewed_at TEXT")
            con.execute("UPDATE events SET review=?, reviewed_at=? WHERE id=?", (label, iso, event_id))
            con.commit()
        except sqlite3.OperationalError:
            con.close()
            return {"id": event_id, "review": label, "reviewed_at": iso, "persisted": False}
        con.close()
        return {"id": event_id, "review": label, "reviewed_at": iso, "persisted": True}

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
                    "FROM events WHERE ts >= ?", (since,)).fetchone()
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
                    "SELECT COUNT(*) FROM events WHERE review IS NULL").fetchone()[0]
            except sqlite3.OperationalError:
                pending = 0
            con.close()
        except sqlite3.OperationalError:
            pending = 0
        return {"cameras": n_cams, "pending_alerts": pending}
