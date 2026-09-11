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

from cvti.scene.context_store import _directory_name_for

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


DEFAULT_ORGANIZATION_ID = "organization--default"
DEFAULT_BRANCH_ID = "branch--default"


class HierarchyConflict(ValueError):
    pass


_AREA_KEYS = {
    "id", "name", "branch_id", "site_type", "area_type", "expected_actors", "note",
}


def _write_site(site_path: str | Path, data: dict) -> None:
    path = Path(site_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def _validated_area(area: dict) -> dict:
    if not isinstance(area, dict):
        raise ValueError("area must be an object")
    extras = sorted(set(area) - _AREA_KEYS)
    if extras:
        raise ValueError(f"area contains unsupported field(s): {', '.join(extras)}")
    area_id = str(area.get("id", "")).strip()
    name = str(area.get("name", "")).strip()
    if not area_id or not name:
        raise ValueError("area id and name must not be empty")
    result = {"id": area_id, "name": name}
    if "branch_id" in area:
        branch_id = str(area["branch_id"]).strip()
        if not branch_id:
            raise ValueError("area branch_id must not be empty")
        result["branch_id"] = branch_id
    for key in ("site_type", "area_type", "note"):
        if key in area:
            result[key] = str(area[key]).strip()
    if "expected_actors" in area:
        actors = area["expected_actors"]
        if not isinstance(actors, list):
            raise ValueError("expected_actors must be an array")
        result["expected_actors"] = list(dict.fromkeys(
            text for value in actors if (text := str(value).strip())
        ))
    return result


def camera_area_id(camera: dict) -> str:
    explicit = str(camera.get("area_id", "")).strip()
    if explicit:
        return explicit
    camera_id = str(camera.get("id", "camera")).strip() or "camera"
    return f"camera--{_directory_name_for(camera_id)}"


def normalized_areas(site_path: str | Path) -> list[dict]:
    """Return explicit and derived single-camera areas without mutating config."""
    data = load_site(site_path)
    explicit = [_validated_area(area) for area in data.get("areas", [])]
    by_id = {area["id"]: {**area, "implicit": False, "camera_ids": []}
             for area in explicit}
    for camera in data.get("cameras", []):
        area_id = camera_area_id(camera)
        if area_id not in by_id:
            by_id[area_id] = {
                "id": area_id,
                "name": str(camera.get("id", "Camera")),
                "implicit": True,
                "camera_ids": [],
            }
        by_id[area_id]["camera_ids"].append(str(camera.get("id", "")))
    return list(by_id.values())


def _validated_named_item(value: dict, kind: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must be an object")
    item_id = str(value.get("id", "")).strip()
    name = str(value.get("name", "")).strip()
    if not item_id or not name:
        raise ValueError(f"{kind} id and name must not be empty")
    return {"id": item_id, "name": name}


def _normalized_organization(data: dict) -> dict:
    organization = data.get("organization")
    if organization is not None:
        return _validated_named_item(organization, "organization")
    return {
        "id": DEFAULT_ORGANIZATION_ID,
        "name": str(data.get("name", "")).strip() or "My Site",
    }


def normalized_organization(site_path: str | Path) -> dict:
    """Return the persisted organization or a stable legacy default."""
    return _normalized_organization(load_site(site_path))


def _normalized_branches(data: dict) -> list[dict]:
    if "branches" not in data:
        return [{"id": DEFAULT_BRANCH_ID, "name": "Main branch"}]
    branches = []
    seen = set()
    for value in data.get("branches", []):
        branch = _validated_named_item(value, "branch")
        if branch["id"] in seen:
            raise ValueError(f"duplicate branch id: {branch['id']}")
        seen.add(branch["id"])
        branches.append(branch)
    return branches


def normalized_branches(site_path: str | Path) -> list[dict]:
    """Return validated branches or the stable legacy default branch."""
    return _normalized_branches(load_site(site_path))


def normalized_hierarchy(site_path: str | Path) -> dict:
    """Build the location tree without changing the persisted site config."""
    data = load_site(site_path)
    legacy = "branches" not in data
    branches = _normalized_branches(data)
    branch_ids = {branch["id"] for branch in branches}
    branch_nodes = [{**branch, "areas": []} for branch in branches]
    branch_by_id = {branch["id"]: branch for branch in branch_nodes}

    areas = normalized_areas(site_path)
    area_by_id = {area["id"]: area for area in areas}
    explicit_area_ids = {
        str(area.get("id", "")).strip() for area in data.get("areas", [])
    }
    cameras_by_area = {area["id"]: [] for area in areas}
    unresolved_area_ids = set()
    unassigned_cameras = []
    for value in data.get("cameras", []):
        camera = dict(value)
        explicit_area_id = str(camera.get("area_id", "")).strip()
        if explicit_area_id and explicit_area_id not in explicit_area_ids:
            unresolved_area_ids.add(explicit_area_id)
            camera.pop("branch_id", None)
            unassigned_cameras.append(camera)
            continue
        area_id = camera_area_id(camera)
        area = area_by_id.get(area_id)
        if area is None:
            unassigned_cameras.append(camera)
            continue
        branch_id = area.get("branch_id")
        if not branch_id and legacy:
            branch_id = DEFAULT_BRANCH_ID
        camera["area_id"] = area_id
        if branch_id in branch_ids:
            camera["branch_id"] = branch_id
        else:
            camera.pop("branch_id", None)
        cameras_by_area[area_id].append(camera)

    unassigned_areas = []
    for area in areas:
        if (
            area.get("implicit")
            and area["id"] in unresolved_area_ids
            and not cameras_by_area[area["id"]]
        ):
            continue
        branch_id = area.get("branch_id")
        if not branch_id and legacy:
            branch_id = DEFAULT_BRANCH_ID
        node = {**area, "cameras": cameras_by_area[area["id"]]}
        node.pop("camera_ids", None)
        if branch_id in branch_by_id:
            node["branch_id"] = branch_id
            branch_by_id[branch_id]["areas"].append(node)
        else:
            unassigned_areas.append(node)

    return {
        "organization": _normalized_organization(data),
        "branches": branch_nodes,
        "unassigned_areas": unassigned_areas,
        "unassigned_cameras": unassigned_cameras,
    }


def _materialize_hierarchy(data: dict) -> None:
    legacy = "branches" not in data
    data["organization"] = _normalized_organization(data)
    branches = _normalized_branches(data)
    data["branches"] = branches
    branch_ids = {branch["id"] for branch in branches}
    areas = [_validated_area(area) for area in data.get("areas", [])]
    if legacy:
        for area in areas:
            area.setdefault("branch_id", DEFAULT_BRANCH_ID)
    for area in areas:
        branch_id = area.get("branch_id")
        if branch_id and branch_id not in branch_ids:
            raise ValueError(f"unknown branch: {branch_id}")
    data["areas"] = areas


def set_organization(site_path: str | Path, organization: dict) -> dict:
    prepared = _validated_named_item(organization, "organization")
    data = load_site(site_path)
    _materialize_hierarchy(data)
    data["organization"] = prepared
    _write_site(site_path, data)
    return prepared


def upsert_branch(site_path: str | Path, branch: dict) -> list[dict]:
    prepared = _validated_named_item(branch, "branch")
    data = load_site(site_path)
    _materialize_hierarchy(data)
    branches = [item for item in data["branches"] if item["id"] != prepared["id"]]
    branches.append(prepared)
    data["branches"] = branches
    _write_site(site_path, data)
    return branches


def remove_branch(site_path: str | Path, branch_id: str) -> list[dict]:
    branch_id = str(branch_id).strip()
    if not branch_id:
        raise ValueError("branch id must not be empty")
    data = load_site(site_path)
    _materialize_hierarchy(data)
    if branch_id not in {branch["id"] for branch in data["branches"]}:
        raise ValueError(f"unknown branch: {branch_id}")
    if any(area.get("branch_id") == branch_id for area in data["areas"]):
        raise HierarchyConflict("branch contains areas")
    data["branches"] = [
        branch for branch in data["branches"] if branch["id"] != branch_id
    ]
    _write_site(site_path, data)
    return data["branches"]


def upsert_area(site_path: str | Path, area: dict) -> list[dict]:
    data = load_site(site_path)
    _materialize_hierarchy(data)
    prepared = _validated_area(area)
    prepared.setdefault("branch_id", DEFAULT_BRANCH_ID)
    known_branches = {branch["id"] for branch in data["branches"]}
    if prepared["branch_id"] not in known_branches:
        raise ValueError(f"unknown branch: {prepared['branch_id']}")
    areas = [item for item in data.get("areas", [])
             if str(item.get("id", "")) != prepared["id"]]
    areas.append(prepared)
    data["areas"] = areas
    _write_site(site_path, data)
    return normalized_areas(site_path)


def remove_area(site_path: str | Path, area_id: str) -> list[dict]:
    area_id = str(area_id).strip()
    data = load_site(site_path)
    data["areas"] = [area for area in data.get("areas", [])
                     if str(area.get("id", "")) != area_id]
    for camera in data.get("cameras", []):
        if str(camera.get("area_id", "")) == area_id:
            camera.pop("area_id", None)
    _write_site(site_path, data)
    return normalized_areas(site_path)


def assign_camera_area(
    site_path: str | Path, camera_id: str, area_id: str
) -> dict:
    data = load_site(site_path)
    area_id = str(area_id).strip()
    known = {str(area.get("id", "")) for area in data.get("areas", [])}
    if area_id not in known:
        raise ValueError(f"unknown area: {area_id}")
    for camera in data.get("cameras", []):
        if str(camera.get("id", "")) == str(camera_id):
            camera["area_id"] = area_id
            _write_site(site_path, data)
            return dict(camera)
    raise ValueError(f"unknown camera: {camera_id}")


def add_camera(site_path: str | Path, camera: dict) -> list[dict]:
    """Upsert a camera (by id) into the site config and persist. Returns cameras."""
    if not camera.get("source"):
        raise ValueError("camera needs a source (RTSP/HTTP URL, webcam index, or file)")
    data = load_site(site_path)
    area_id = str(camera.get("area_id", "")).strip()
    if area_id:
        known = {str(area.get("id", "")) for area in data.get("areas", [])}
        if area_id not in known:
            raise ValueError(f"unknown area: {area_id}")
    cam_id = camera.get("id") or f"cam{len(data.get('cameras', [])) + 1}"
    camera = {**camera, "id": cam_id}
    cams = [c for c in data.get("cameras", []) if c.get("id") != cam_id]
    cams.append(camera)
    data["cameras"] = cams
    _write_site(site_path, data)
    return cams


def remove_camera(site_path: str | Path, camera_id: str) -> list[dict]:
    data = load_site(site_path)
    cams = [c for c in data.get("cameras", []) if c.get("id") != camera_id]
    data["cameras"] = cams
    _write_site(site_path, data)
    return cams


# --- site metadata (name, notifier, first-run flag) ---------------------------
# Stored alongside "cameras" in the same site JSON so one file fully describes a
# deployment. The app's setup wizard reads/writes these.

# The last three feed the Value screen. They are the site's own numbers, not
# generic benchmarks — an ROI figure computed from someone else's assumptions is
# worth nothing to the person signing the renewal.
_META_KEYS = ("name", "notify", "gate", "configured", "scene_context_policy",
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


def complete_first_run(site_path: str | Path) -> dict:
    """Finish setup. Never stamps a scene-context policy.

    This used to stamp fresh sites 'require_reviewed' — which blocked every
    camera until a human approved its scene context, and a pilot's first-run
    install spent two days as 'the engine never starts' (watchdog restart
    loop, 1 Sep): mapping timed out on his hardware, the strict policy
    blocked all cameras, the engine exited, repeat. New sites run the 'auto'
    default — mapping failure means the camera runs generic AND LOUD, never
    that monitoring silently refuses to exist. Strict policies remain an
    explicit operator choice (scene_context_policy in the site config)."""
    data = load_site(site_path)
    data["configured"] = True
    path = Path(site_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return get_site_meta(site_path)
