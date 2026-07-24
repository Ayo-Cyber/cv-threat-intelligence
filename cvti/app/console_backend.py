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
import sqlite3
import time
from pathlib import Path

from cvti.serving import onboarding, vlm

_REVIEW_VALUES = {"ack", "true", "false", "new"}


class ConsoleBackend:
    def __init__(self, site_path: str = "configs/site_live.json",
                 db_path: str = "runs/site/events.db") -> None:
        self.site_path = site_path
        self.db_path = db_path
        self._live = None  # LiveWall instance while the Live screen is open

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

    # --- first-run setup wizard ---
    def get_site(self) -> dict:
        return onboarding.get_site_meta(self.site_path)

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

    def setup_state(self) -> dict:
        """Everything the wizard needs to decide whether to show + where to resume."""
        meta = self.get_site()
        return {
            "configured": meta["configured"],
            "site_name": meta["name"],
            "notify": meta["notify"],
            "cameras": meta["camera_count"],
            "gate": self.gate_status(),
        }

    # --- events / review ---
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def list_events(self, limit: int = 100, embed_frames: bool = True) -> list[dict]:
        try:
            con = self._connect()
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
                e["frames"] = self._frames_as_data_uris(e.get("evidence_dir"))
            out.append(e)
        return out

    def _frames_as_data_uris(self, evidence_dir: str | None, cap: int = 5) -> list[str]:
        d = Path(evidence_dir or "")
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
        con = self._connect()
        # defensive: older DBs may predate the review columns
        cols = {c[1] for c in con.execute("PRAGMA table_info(events)")}
        if "review" not in cols:
            con.execute("ALTER TABLE events ADD COLUMN review TEXT")
        if "reviewed_at" not in cols:
            con.execute("ALTER TABLE events ADD COLUMN reviewed_at TEXT")
        iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        con.execute("UPDATE events SET review=?, reviewed_at=? WHERE id=?", (label, iso, event_id))
        con.commit()
        con.close()
        return {"id": event_id, "review": label, "reviewed_at": iso}

    def counts(self) -> dict:
        """Header/nav summary numbers."""
        cams = self.list_cameras()
        pending = 0
        try:
            con = self._connect()
            try:
                pending = con.execute(
                    "SELECT COUNT(*) FROM events WHERE review IS NULL OR review='ack'").fetchone()[0]
            except sqlite3.OperationalError:
                pending = 0
            con.close()
        except sqlite3.OperationalError:
            pending = 0
        return {"cameras": len(cams), "pending_alerts": pending}
