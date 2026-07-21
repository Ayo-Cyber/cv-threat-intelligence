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
