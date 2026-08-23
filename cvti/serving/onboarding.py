"""Camera onboarding logic — Qt-free so it's unit-testable and reusable by the
desktop app (and anything else). The PyQt widgets are thin shells over this.

Covers the customer-facing parts of adding a camera: find it on the network,
confirm a URL works (with a snapshot), and save it into a site config — no CLI,
no hand-edited JSON.
"""
from __future__ import annotations

import ipaddress
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Common vendor sub-stream (low-res) URL patterns — offered as suggestions.
VENDOR_PATHS = {
    "Hikvision": "/Streaming/Channels/102",
    "Dahua/Amcrest": "/cam/realmonitor?channel=1&subtype=1",
    "Reolink": "/h264Preview_01_sub",
    "Generic": "/stream1",
}

RULE_PRESETS = {
    "All threats + video": "configs/all_threats_video_v1.json",
    "All threats": "configs/all_threats_v1.json",
    "Loitering / zones": "configs/shelf_zones_demo.json",
}


def test_url(url: str, snapshot_size: int = 320) -> dict:
    """Open a stream and report whether it works (+ a base64 snapshot to preview)."""
    import base64
    import cv2
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return {"ok": False, "error": "Could not open — check the IP, credentials, path, and that the PC is on the same network."}
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    if not ok or frame is None:
        return {"ok": False, "error": "Connected but no video — the stream path is probably wrong, or the codec isn't supported."}
    h, w = frame.shape[:2]
    tw = snapshot_size
    th = int(tw * h / w) if w else 240
    b64 = base64.b64encode(cv2.imencode(".jpg", cv2.resize(frame, (tw, th)))[1]).decode()
    return {"ok": True, "w": w, "h": h, "fps": round(fps, 1),
            "snapshot": f"data:image/jpeg;base64,{b64}", "jpeg_b64": b64}


def detect_subnet() -> str | None:
    """Best-effort local /24 the machine is on — so the operator never types a
    subnet. Uses the primary route's source IP (no packets sent) and assumes a
    /24, which is what virtually every camera LAN uses.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # picks the default-route interface; no traffic
            ip = s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None
    if not ip or ip.startswith("127."):
        return None
    net = ipaddress.ip_network(f"{ip}/24", strict=False)
    return str(net)


def scan_subnet(cidr: str, port: int = 554, timeout: float = 0.4, max_hosts: int = 512) -> list[str]:
    """Return hosts on the subnet with the RTSP port open (likely cameras)."""
    hosts = [str(h) for h in ipaddress.ip_network(cidr, strict=False).hosts()][:max_hosts]

    def check(ip: str):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            return ip if s.connect_ex((ip, port)) == 0 else None
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=64) as ex:
        return [ip for ip in ex.map(check, hosts) if ip]


def load_site(site_path: str | Path) -> dict:
    p = Path(site_path)
    return json.loads(p.read_text()) if p.exists() else {"cameras": []}


def list_cameras(site_path: str | Path) -> list[dict]:
    return load_site(site_path).get("cameras", [])


def add_camera(site_path: str | Path, camera: dict) -> list[dict]:
    """Upsert a camera (by id) into the site config and persist. Returns cameras."""
    if not camera.get("source"):
        raise ValueError("camera needs a source (RTSP/HTTP URL, webcam index, or file)")
    data = load_site(site_path)
    cam_id = camera.get("id") or f"cam{len(data.get('cameras', [])) + 1}"
    camera = {**camera, "id": cam_id}
    cams = [c for c in data.get("cameras", []) if c.get("id") != cam_id]
    cams.append(camera)
    data["cameras"] = cams
    p = Path(site_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return cams


def remove_camera(site_path: str | Path, camera_id: str) -> list[dict]:
    data = load_site(site_path)
    cams = [c for c in data.get("cameras", []) if c.get("id") != camera_id]
    data["cameras"] = cams
    Path(site_path).write_text(json.dumps(data, indent=2))
    return cams


# --- site metadata (name, notifier, first-run flag) ---------------------------
# Stored alongside "cameras" in the same site JSON so one file fully describes a
# deployment. The app's setup wizard reads/writes these.

# The last three feed the Value screen. They are the site's own numbers, not
# generic benchmarks — an ROI figure computed from someone else's assumptions is
# worth nothing to the person signing the renewal.
_META_KEYS = ("name", "notify", "gate", "configured",
              "incident_value", "guard_hourly_cost", "review_minutes",
              "retention_days", "disk_warn_pct", "disk_critical_pct",
              "daily_normal",   # the daily "all systems normal" message; opt-OUT
              "heartbeat_url", "heartbeat_key",  # remote monitoring; opt-IN (empty = off)
              "backup_dir")                      # config backups; empty = per-user default

# Deliberately conservative: a low review time and a modest guard rate make the
# saving harder to argue with than a flattering one.
VALUE_DEFAULTS = {
    "incident_value": 0.0,      # 0 until the site says otherwise — no invented number
    "guard_hourly_cost": 0.0,
    "review_minutes": 2.0,      # minutes of attention one alert costs to triage
    # Storage limitation is not optional under GDPR/NDPR, so this has a real
    # default rather than "off" — a site that never configures it still deletes.
    "retention_days": 30.0,
    "disk_warn_pct": 85.0,
    "disk_critical_pct": 95.0,
}


def get_site_meta(site_path: str | Path) -> dict:
    data = load_site(site_path)
    meta = {k: data.get(k) for k in _META_KEYS}
    meta["name"] = meta.get("name") or "My Site"
    meta["notify"] = meta.get("notify") or "console"
    meta["configured"] = bool(meta.get("configured"))
    meta["camera_count"] = len(data.get("cameras", []))
    # On by default: a system that only speaks when something is wrong cannot
    # be trusted when it is silent. Only an explicit False opts out.
    meta["daily_normal"] = meta.get("daily_normal") is not False
    # Heartbeat is opt-IN: nothing is sent anywhere until a URL is configured.
    meta["heartbeat_url"] = (meta.get("heartbeat_url") or "").strip()
    meta["heartbeat_key"] = (meta.get("heartbeat_key") or "").strip()
    for k, default in VALUE_DEFAULTS.items():
        try:
            meta[k] = float(meta.get(k)) if meta.get(k) is not None else default
        except (TypeError, ValueError):
            meta[k] = default
    return meta


def set_site_meta(site_path: str | Path, **fields: Any) -> dict:
    """Update site-level fields (name/notify/gate/configured) and persist."""
    data = load_site(site_path)
    for k, v in fields.items():
        if k in _META_KEYS and v is not None:
            data[k] = v
    p = Path(site_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return get_site_meta(site_path)
