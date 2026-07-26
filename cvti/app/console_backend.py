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

from cvti.serving import onboarding, vlm

_REVIEW_VALUES = {"ack", "true", "false", "new"}


class ConsoleBackend:
    def __init__(self, site_path: str = "configs/site_live.json",
                 db_path: str = "runs/site/events.db", enable_demo: bool = True) -> None:
        self.site_path = site_path
        self.db_path = db_path
        self._live = None       # LiveWall instance while the Live screen is open
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

    # detectors the operator can toggle per camera (drive which models run)
    RULE_FLAGS = ("concealment", "video_action", "violence", "weapons", "theft", "tamper")

    def set_camera_rules(self, camera_id: str, rules: dict) -> dict:
        """Update which threat detectors run on a camera (+ optional rule preset).
        Takes effect on the next Start monitoring."""
        cams = onboarding.list_cameras(self.site_path)
        cam = next((c for c in cams if c.get("id") == camera_id), None)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        for k in self.RULE_FLAGS:
            if k in rules:
                cam[k] = bool(rules[k])
        if rules.get("config"):
            cam["config"] = rules["config"]
        onboarding.add_camera(self.site_path, cam)   # upsert by id
        return {"ok": True, "camera": cam}

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

    def gate_status(self, model: str = vlm.DEFAULT_MODEL) -> dict:
        return vlm.gate_status(model)

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

    def live_start(self, count: int = 6) -> list[dict]:
        from cvti.app.live_wall import LiveWall
        self.live_stop()
        sources = self._live_sources(count)
        if not sources:
            return []
        self._live = LiveWall(sources).start()
        return [{"id": s["id"]} for s in sources]

    def live_frames(self) -> dict:
        return self._live.frames() if self._live else {}

    def live_stop(self) -> dict:
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
        log = open(out_dir / "monitor.log", "a")  # noqa: SIM115 - lives with the subprocess
        # Lean defaults keep the box cool: lower fps + image size cut compute a lot
        # with negligible quality loss at demo scale.
        cmd = [sys.executable, "-m", "cvti.serving.pipeline",
               "--site-config", self.site_path,
               "--gate-provider", "ollama", "--gate-model", "gemma3:4b",
               "--notify", notify, "--output-dir", str(out_dir),
               "--target-fps", "4", "--imgsz", "512",
               "--seconds", "100000", "--gate-drain", "60"]
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

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
                        print(f"[watchdog] engine exited unexpectedly — restarting "
                              f"({self._restarts}/{max_restarts})")
                        self._monitor = self._spawn_engine()
                    else:
                        print("[watchdog] engine died too many times — giving up")
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
            out.append(e)
        return out

    def _frames_as_data_uris(self, evidence_dir: str | None,
                             frame_base: "Path | None" = None, cap: int = 5) -> list[str]:
        d = Path(evidence_dir or "")
        if not d.exists() and frame_base and evidence_dir:
            d = frame_base / evidence_dir       # bundled demo: paths are relative
        if not d.exists():
            return []
        uris = []
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and len(uris) < cap:
                b64 = base64.b64encode(p.read_bytes()).decode()
                uris.append(f"data:image/jpeg;base64,{b64}")
        return uris

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
