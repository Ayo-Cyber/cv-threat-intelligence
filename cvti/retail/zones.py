"""retail_zones.py — Supervision-based shelf-zone + tracking scaffolding for retail theft.

This is the SPATIAL foundation the action-recognition layer plugs into. It answers
"WHO is WHERE, and for HOW LONG" — it does NOT decide theft. The chain is:

    YOLO detect + ByteTrack (Ultralytics)  ->  who, with a stable id
    Supervision PolygonZone                ->  is that person in the shelf zone
    dwell accounting (here)                ->  how long they have lingered
        |
        v
    [later] when a person interacts with a shelf, grab the rolling clip ->
            action-recognition model -> Verification Gate (VLM) -> alert

Design notes:
- Tracking is taken from Ultralytics `model.track(...)` (ByteTrack via bytetrack.yaml),
  NOT `sv.ByteTrack`, which is deprecated in supervision 0.28 and removed in 0.30.
- The monitor is class-agnostic: feed it any tracked `sv.Detections`. The CLI demo
  filters to persons (COCO class 0); you could just as well watch tracked merchandise.
- Polygon zones are CAMERA-SPECIFIC (pixel coordinates at a given resolution). Define
  one config per physical camera. See configs/retail_zones.example.json.

CLI demo (needs `ultralytics` installed; tracking model auto-downloads):
    python retail_zones.py --source data/test_clips/theft_shop_01.mp4 \
        --zones configs/retail_zones.example.json --show

The pure zone/dwell logic has no torch dependency and is covered by
tests/test_retail_zones.py, which runs on synthetic detections.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import supervision as sv
from cvti.logging_setup import get_logger

log = get_logger(__name__)

# COCO person class id, used by the CLI demo to keep only people.
PERSON_CLASS_ID = 0

# Map the anchor names accepted in the config to supervision Position enum members.
_ANCHOR_BY_NAME: dict[str, sv.Position] = {p.name: p for p in sv.Position}


@dataclass
class ZoneSpec:
    """One named polygon zone parsed from the config."""

    name: str
    polygon: np.ndarray                       # (N, 2) int array of pixel points
    anchors: tuple[sv.Position, ...] = (sv.Position.BOTTOM_CENTER,)
    kind: str = "shelf"                       # free-form tag: shelf | exit | aisle | ...
    dwell_alert_seconds: float | None = None  # optional loiter threshold for this zone
    # A person must be inside this long before "entered" fires; a one-frame
    # false detection never becomes a HIGH alert. 0 = instant (library default;
    # load_zone_config defaults configs to 0.5s).
    entry_confirm_seconds: float = 0.0
    # How long a track that VANISHED in the zone's interior stays present
    # before it counts as gone. None -> RetailZoneMonitor.LOST_GRACE_DEFAULT.
    lost_grace_seconds: float | None = None
    # Polygon given in 0..1 frame fractions: scaled to the first frame seen.
    normalized: bool = False
    polygon_raw: np.ndarray | None = None     # the 0..1 points, when normalized


@dataclass
class PersonZoneState:
    """Per-detection snapshot returned by RetailZoneMonitor.update()."""

    tracker_id: int | None
    bbox: tuple[int, int, int, int]
    zones: list[str] = field(default_factory=list)        # zones currently occupied
    dwell_seconds: dict[str, float] = field(default_factory=dict)
    loitering: bool = False                               # crossed any zone's dwell threshold
    entered_zones: list[str] = field(default_factory=list)  # zones this track crossed INTO this frame

    def label(self) -> str:
        tag = f"#{self.tracker_id}" if self.tracker_id is not None else "#?"
        if not self.zones:
            return tag
        z = self.zones[0]
        dwell = self.dwell_seconds.get(z, 0.0)
        flag = " LOITER" if self.loitering else ""
        return f"{tag} {z} {dwell:.1f}s{flag}"


@dataclass(eq=False)
class ZoneExit:
    """One debounced departure from RetailZoneMonitor.drain_exits().

    Unpacks and compares as the historical (tracker_id, zone) tuple, and adds
    how long the person was inside and HOW they went: `walked_out` (last seen
    at a frame edge / the zone boundary, or seen outside the zone) versus
    `lost` (vanished in the interior and never came back within the lost
    window — an occlusion that outlasted patience, not a doorway)."""

    tracker_id: int
    zone: str
    dwell_seconds: float = 0.0
    how: str = "walked_out"

    def __iter__(self):
        yield self.tracker_id
        yield self.zone

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ZoneExit):
            return (self.tracker_id, self.zone, self.dwell_seconds, self.how) == \
                   (other.tracker_id, other.zone, other.dwell_seconds, other.how)
        if isinstance(other, tuple):
            return (self.tracker_id, self.zone) == other
        return NotImplemented


def _point_to_polygon_distance(point: tuple[float, float], polygon: np.ndarray) -> float:
    """Shortest distance from a point to the polygon's outline."""
    px, py = point
    best = float("inf")
    n = len(polygon)
    for i in range(n):
        ax, ay = (float(v) for v in polygon[i])
        bx, by = (float(v) for v in polygon[(i + 1) % n])
        dx, dy = bx - ax, by - ay
        seg = dx * dx + dy * dy
        t = 0.0 if seg == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)
    return best


def filter_person_detections(
    detections: sv.Detections,
    frame_hw: tuple[int, int],
    min_area_ratio: float = 0.012,
    min_aspect: float = 1.10,
) -> sv.Detections:
    """Drop implausible 'person' boxes (mannequin heads, wig displays, reflections).

    A standing shopper is reasonably large and taller-than-wide; mannequin heads on a
    shelf are small and roughly square. Filters by box-area-as-fraction-of-frame and by
    height/width aspect ratio.
    """
    if len(detections) == 0:
        return detections
    h, w = frame_hw
    frame_area = float(max(1, h * w))
    keep = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        area_ratio = (bw * bh) / frame_area
        aspect = bh / bw
        keep.append(area_ratio >= min_area_ratio and aspect >= min_aspect)
    return detections[np.array(keep, dtype=bool)]


def parse_anchors(raw: Iterable[str] | None) -> tuple[sv.Position, ...]:
    if not raw:
        return (sv.Position.BOTTOM_CENTER,)
    anchors: list[sv.Position] = []
    for name in raw:
        key = str(name).strip().upper()
        if key not in _ANCHOR_BY_NAME:
            raise ValueError(
                f"Unknown zone anchor '{name}'. Allowed: {sorted(_ANCHOR_BY_NAME)}"
            )
        anchors.append(_ANCHOR_BY_NAME[key])
    return tuple(anchors)


def load_zone_config(path: str | Path) -> list[ZoneSpec]:
    """Load named polygon zones from a JSON config. See configs/retail_zones.example.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    specs: list[ZoneSpec] = []
    for entry in data.get("zones", []):
        pts = np.array(entry["polygon"], dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError(
                f"Zone '{entry.get('name')}' polygon must be a list of >=3 [x, y] points."
            )
        # Points all within 0..1 are frame FRACTIONS: a doorway drawn once is
        # right on a 720p and a 1080p feed alike. The monitor scales them to
        # the first frame it sees; until then they sit on a nominal 1080p
        # canvas so a monitor that never learns the frame still has a shape.
        normalized = bool(pts.min() >= 0.0 and pts.max() <= 1.0)
        polygon = (np.rint(pts * np.array([1920.0, 1080.0])) if normalized else pts).astype(np.int64)
        specs.append(
            ZoneSpec(
                name=str(entry["name"]),
                polygon=polygon,
                anchors=parse_anchors(entry.get("anchors")),
                kind=str(entry.get("kind", "shelf")),
                dwell_alert_seconds=entry.get("dwell_alert_seconds"),
                # Configs default to a confirmed entry: at ~4fps that is two
                # consecutive sightings, which a single false box cannot fake.
                entry_confirm_seconds=float(entry.get("entry_confirm_seconds", 0.5)),
                lost_grace_seconds=entry.get("lost_grace_seconds"),
                normalized=normalized,
                polygon_raw=pts if normalized else None,
            )
        )
    if not specs:
        raise ValueError(f"No zones found in {path}.")
    return specs


class RetailZoneMonitor:
    """Tracks which person ids occupy which polygon zones, and for how long.

    Stateless about detection: call update() once per frame with the frame's tracked
    detections and a monotonic timestamp (seconds). Dwell is measured from the moment a
    track *enters* a zone and resets the moment it leaves.
    """

    # Production default (10 Sep 2026): every engine construction site ran at
    # 0.0 — one dropped frame reset a 60s loiter timer — while only the CLI
    # demo passed 1.5. At the detect cadence (~4fps) 2.5s tolerates ~10 absent
    # frames; against a 60s threshold it cannot fabricate a loiterer, it only
    # stops real ones being forgotten mid-dwell.
    DWELL_GRACE_DEFAULT = 2.5
    # A track that vanishes in the INTERIOR of a zone did not leave — nobody
    # walks out through the middle of the floor. A turn away from a close
    # webcam, an occlusion, a detector dropout all look like that, and each
    # used to read as "left" then "entered" again (12 Sep). Such a track stays
    # present for this long; a fresh id appearing at the spot inherits it. A
    # track last seen at a frame edge or the zone boundary, or seen OUTSIDE the
    # zone, did walk out: that exit fires after the ordinary grace.
    LOST_GRACE_DEFAULT = 8.0
    # "At the edge" = within this fraction of the longer frame side (~115px at 1080p).
    EDGE_MARGIN_RATIO = 0.06

    def __init__(self, zones: list[ZoneSpec],
                 dwell_grace_seconds: float | None = None) -> None:
        self.zones = zones
        self.dwell_grace_seconds = (self.DWELL_GRACE_DEFAULT
                                    if dwell_grace_seconds is None
                                    else dwell_grace_seconds)
        # (tracker_id, zone) pairs whose ENTRY has fired (or was inherited);
        # only these ever produce an exit — a flicker that never confirmed
        # leaves no trace either way.
        self._confirmed: set[tuple[int, str]] = set()
        self._frame_hw: tuple[int, int] | None = None
        self._pending_scale = any(z.normalized for z in zones)
        self._sv_zones: dict[str, sv.PolygonZone] = {
            z.name: sv.PolygonZone(polygon=z.polygon, triggering_anchors=z.anchors)
            for z in zones
        }
        self._spec_by_name: dict[str, ZoneSpec] = {z.name: z for z in zones}
        # (tracker_id, zone_name) -> timestamp the track entered that zone
        self._entered_at: dict[tuple[int, str], float] = {}
        # (tracker_id, zone_name) -> last timestamp the track was actually in the zone.
        # Lets dwell survive brief gaps (boundary jitter, 1-frame track loss) up to grace.
        self._last_in_zone: dict[tuple[int, str], float] = {}
        # (tracker_id, zone_name) -> last box centre while in the zone. When the
        # tracker loses a person behind an occlusion it hands back a NEW id —
        # new id meant new timer, and the loiterer was forgotten mid-dwell. A
        # new track appearing in the same zone within the grace window, at the
        # spot a track just vanished from, INHERITS that track's entry time.
        self._last_pos: dict[tuple[int, str], tuple[float, float, float]] = {}
        # (tracker_id, zone_name) -> last full box while in the zone, for the
        # edge test. The CENTRE is the wrong point for that: on a close-range
        # webcam a body half out of frame still has its centre well inside,
        # so a real walk-out read as "lost from view" (12 Sep, first minute of
        # the hardened build). A box that touches the frame border is at the edge.
        self._last_box: dict[tuple[int, str], tuple[int, int, int, int]] = {}
        # Zone exits detected on the last update() — a (tracker_id, zone) pair
        # forgotten after the grace window (a debounced departure, so a
        # one-frame occlusion is NOT read as leaving-and-re-entering). Drained
        # by the caller each frame via drain_exits().
        self._exits: list[tuple[int, str]] = []

    def drain_exits(self) -> list[ZoneExit]:
        """The departures since the last call — each a ZoneExit that still
        unpacks as the historical (tracker_id, zone).

        Exit is debounced twice over. A track SEEN leaving (outside the zone,
        or last at a frame edge / the zone boundary) is gone once absent past
        dwell_grace_seconds. A track that vanished in the interior waits the
        zone's lost window instead — an occlusion mid-floor is not a doorway.
        Only a confirmed entry can ever produce an exit."""
        out = self._exits
        self._exits = []
        return out

    def _fit_to_frame(self, frame_hw: tuple[int, int]) -> None:
        """Scale normalized (0..1) polygons to the frame the first time one is
        seen; remember the frame either way for the edge test."""
        self._frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
        if not self._pending_scale:
            return
        h, w = self._frame_hw
        for z in self.zones:
            if z.normalized and z.polygon_raw is not None:
                z.polygon = np.rint(z.polygon_raw * np.array([w, h], dtype=float)).astype(np.int64)
                self._sv_zones[z.name] = sv.PolygonZone(polygon=z.polygon,
                                                        triggering_anchors=z.anchors)
        self._pending_scale = False

    def _frame_extent(self) -> tuple[int, int]:
        """(h, w) to judge 'at the edge' against: the real frame when known,
        else the extent of the zones themselves (the CLI/replay callers)."""
        if self._frame_hw is not None:
            return self._frame_hw
        pts = np.concatenate([z.polygon for z in self.zones])
        return int(pts[:, 1].max()), int(pts[:, 0].max())

    def _lost_grace(self, zone: str) -> float:
        v = self._spec_by_name[zone].lost_grace_seconds
        return self.LOST_GRACE_DEFAULT if v is None else float(v)

    def _near_boundary(self, key: tuple[int, str]) -> bool:
        """Was this track last seen where a person can actually leave from —
        the frame edge, or the zone's own outline?"""
        box = self._last_box.get(key)
        if box is None:
            return True                 # nothing known: the old, prompt behaviour
        x1, y1, x2, y2 = box
        h, w = self._frame_extent()
        margin = self.EDGE_MARGIN_RATIO * max(h, w)
        # Any side of the box within the margin of the frame border: the person
        # is (partly) out of frame — they are leaving through the edge.
        if x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin:
            return True
        # Otherwise: were their feet at the zone's own outline (a drawn doorway)?
        feet = ((x1 + x2) / 2.0, float(y2))
        return _point_to_polygon_distance(feet, self._spec_by_name[key[1]].polygon) <= margin

    def _forget(self, key: tuple[int, str]) -> None:
        for store in (self._entered_at, self._last_in_zone, self._last_pos, self._last_box):
            store.pop(key, None)
        self._confirmed.discard(key)

    def update(self, detections: sv.Detections, timestamp: float,
               frame_hw: tuple[int, int] | None = None) -> list[PersonZoneState]:
        if frame_hw is not None and (self._frame_hw is None or self._pending_scale):
            self._fit_to_frame(frame_hw)
        n = len(detections)
        # Boolean membership mask per zone, aligned to detection order.
        masks = {name: zone.trigger(detections) for name, zone in self._sv_zones.items()}

        current_keys: set[tuple[int, str]] = set()
        present_tids: set[int] = set()      # every tracked person in frame, in a zone or not
        states: list[PersonZoneState] = []

        for i in range(n):
            tid = _tracker_id_at(detections, i)
            bbox = tuple(int(v) for v in detections.xyxy[i])
            state = PersonZoneState(tracker_id=tid, bbox=bbox)  # type: ignore[arg-type]
            if tid is not None:
                present_tids.add(tid)

            for name in self._sv_zones:
                if not bool(masks[name][i]):
                    continue
                state.zones.append(name)
                if tid is None:
                    # Untracked detection: report presence but no dwell (no identity to time).
                    state.dwell_seconds[name] = 0.0
                    continue
                key = (tid, name)
                current_keys.add(key)
                if key not in self._entered_at:
                    inherited = self._inherit_entry(key, bbox, timestamp, current_keys)
                    if inherited is None:
                        self._entered_at[key] = timestamp
                    else:
                        # The same person handed back under a fresh track id
                        # after an occlusion — not a new crossing. They keep
                        # the clock, and if their entry already fired, that too.
                        self._entered_at[key], donor_confirmed = inherited
                        if donor_confirmed:
                            self._confirmed.add(key)
                entered = self._entered_at[key]
                self._last_in_zone[key] = timestamp
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                self._last_pos[key] = (cx, cy, float(bbox[2] - bbox[0]))
                self._last_box[key] = bbox
                dwell = max(0.0, timestamp - entered)
                # ENTRY fires once the person has been inside for the zone's
                # confirmation window (dwell counts from first sight, so the
                # loiter clock is unchanged). A one-frame false box never
                # becomes "PERSON ENTERED"; 0 keeps the instant behaviour.
                if (key not in self._confirmed
                        and dwell >= self._spec_by_name[name].entry_confirm_seconds):
                    self._confirmed.add(key)
                    state.entered_zones.append(name)
                state.dwell_seconds[name] = dwell
                threshold = self._spec_by_name[name].dwell_alert_seconds
                if threshold is not None and dwell >= threshold:
                    state.loitering = True

            states.append(state)

        # Forget a (track, zone) pair only once it has been gone long enough —
        # the ordinary grace when we SAW it leave (outside the zone, or last at
        # an edge), the longer lost window when it vanished mid-zone. Brief
        # boundary jitter / 1-frame track loss never resets dwell either way.
        for key in list(self._entered_at):
            if key in current_keys:
                continue
            entered = self._entered_at[key]
            last = self._last_in_zone.get(key, entered)
            absent = timestamp - last
            if absent <= self.dwell_grace_seconds:
                continue
            walked_out = key[0] in present_tids or self._near_boundary(key)
            if not walked_out and absent <= self._lost_grace(key[1]):
                continue
            confirmed = key in self._confirmed
            self._forget(key)
            if confirmed:                 # an unconfirmed flicker leaves silently
                self._exits.append(ZoneExit(key[0], key[1], max(0.0, last - entered),
                                            "walked_out" if walked_out else "lost"))

        return states

    def _inherit_entry(self, key: tuple[int, str], bbox: tuple,
                       timestamp: float, current_keys: set) -> tuple[float, bool] | None:
        """A vanished track's (entry time, entry-already-fired), if this NEW
        track is standing where it stood.

        ByteTrack hands an occluded person back under a fresh id; without this
        the loiter timer restarted from zero every time. The donor must be the
        SAME zone, absent from this frame, gone for at most its patience window
        (the grace if it was at an edge, the lost window if it vanished
        mid-zone), and its last centre within ~one body-width of the new box —
        i.e. the person visibly never left the spot. The donor is consumed so
        one vanished track cannot seed two heirs."""
        _, zone = key
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        width = max(1.0, float(bbox[2] - bbox[0]))
        best_key, best_dist = None, None
        for old_key, (ox, oy, ow) in self._last_pos.items():
            if old_key == key or old_key[1] != zone or old_key in current_keys:
                continue
            patience = (self.dwell_grace_seconds if self._near_boundary(old_key)
                        else self._lost_grace(zone))
            if timestamp - self._last_in_zone.get(old_key, 0.0) > patience:
                continue
            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if dist <= 1.2 * max(width, ow) and (best_dist is None or dist < best_dist):
                best_key, best_dist = old_key, dist
        if best_key is None:
            return None
        entered = self._entered_at.get(best_key, timestamp)
        confirmed = best_key in self._confirmed
        self._forget(best_key)
        return entered, confirmed

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        states: list[PersonZoneState],
    ) -> np.ndarray:
        """Draw zones, tracked boxes, and per-person id/zone/dwell labels."""
        out = frame
        for name, zone in self._sv_zones.items():
            color = sv.Color.RED if self._spec_by_name[name].kind == "shelf" else sv.Color.BLUE
            annot = sv.PolygonZoneAnnotator(zone=zone, color=color, thickness=2)
            out = annot.annotate(scene=out)
        out = sv.BoxAnnotator().annotate(scene=out, detections=detections)
        labels = [s.label() for s in states]
        out = sv.LabelAnnotator().annotate(scene=out, detections=detections, labels=labels)
        return out


def _tracker_id_at(detections: sv.Detections, i: int) -> int | None:
    if detections.tracker_id is None:
        return None
    value = detections.tracker_id[i]
    return None if value is None else int(value)


# ---------------------------------------------------------------------------
# CLI demo — needs ultralytics installed (imported lazily so the module + tests
# work without torch).
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervision shelf-zone + tracking demo.")
    p.add_argument("--source", required=True, help="Video file path, RTSP URL, or webcam index (e.g. 0).")
    p.add_argument("--zones", required=True, help="Path to a retail_zones JSON config.")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO detection weights. Default: yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold.")
    p.add_argument("--tracker", default="configs/bytetrack_retail.yaml",
                   help="Ultralytics tracker config. Default: the retail-tuned ByteTrack.")
    p.add_argument("--dwell-grace", type=float, default=1.5,
                   help="Seconds a track may be absent from a zone before its dwell resets.")
    p.add_argument("--min-box-area", type=float, default=0.012,
                   help="Min person box area as a fraction of the frame (rejects mannequins).")
    p.add_argument("--min-aspect", type=float, default=1.10,
                   help="Min person box height/width (standing people are taller than wide).")
    p.add_argument("--no-person-filter", action="store_true", help="Disable the person-plausibility filter.")
    p.add_argument("--rules", default="",
                   help="Optional user_config rules JSON (e.g. configs/banking_zones_v1.json). "
                        "If set, runs the Customization Engine on zone presence and prints fired alerts.")
    p.add_argument("--simulate-time", default="",
                   help="Override the clock as HH:MM for time-filtered rules (e.g. 21:00 to test "
                        "'after 8pm' rules during the day). Demo only.")
    p.add_argument("--show", action="store_true", help="Display the annotated video window.")
    p.add_argument("--no-track", action="store_true", help="Disable ByteTrack (dwell will be unavailable).")
    return p.parse_args()


def _normalize_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def main() -> None:
    # Entrypoint: configure logging before anything can fail.
    from cvti.logging_setup import setup_logging
    setup_logging(component="argus-zones")
    args = _parse_args()
    zones = load_zone_config(args.zones)
    monitor = RetailZoneMonitor(zones, dwell_grace_seconds=args.dwell_grace)
    log.info(f"[retail_zones] Loaded {len(zones)} zone(s): {', '.join(z.name for z in zones)}")

    # Optional: run the Customization Engine on zone presence so zone+time+dwell RULES fire.
    engine = None
    sim_now = None
    if args.rules:
        from cvti.event_adapters import zone_states_to_events  # noqa: F401
        from cvti.rules.customization import CustomizationEngine
        engine = CustomizationEngine(args.rules)
        if args.simulate_time:
            from datetime import datetime
            hh, mm = (int(x) for x in args.simulate_time.split(":"))
            base = datetime.now()
            sim_now = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
            log.info(f"[retail_zones] Simulating clock at {args.simulate_time} for time-filtered rules.")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on heavy optional deps
        raise SystemExit(
            f"This demo needs ultralytics + opencv installed ({exc}). "
            "Run: pip install -r requirements.txt"
        )

    model = YOLO(args.weights)
    source = _normalize_source(args.source)
    frame_index = 0
    # Approximate timestamps from frame index when the stream has no real clock.
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    dt = 1.0 / fps if fps and fps > 1e-3 else 1.0 / 25.0

    stream = (
        model.track(source=source, stream=True, persist=True, tracker=args.tracker,
                    conf=args.conf, classes=[PERSON_CLASS_ID], verbose=False)
        if not args.no_track
        else model.predict(source=source, stream=True, conf=args.conf,
                           classes=[PERSON_CLASS_ID], verbose=False)
    )

    last_report: dict[int, str] = {}
    for result in stream:
        detections = sv.Detections.from_ultralytics(result)
        if not args.no_person_filter:
            detections = filter_person_detections(
                detections, result.orig_img.shape[:2],
                min_area_ratio=args.min_box_area, min_aspect=args.min_aspect,
            )
        timestamp = frame_index * dt
        states = monitor.update(detections, timestamp)

        for s in states:
            if not s.zones or s.tracker_id is None:
                continue
            line = s.label()
            if last_report.get(s.tracker_id) != line:
                event = "LOITER" if s.loitering else "in-zone"
                log.info(f"[{event}] {line}")
                last_report[s.tracker_id] = line

        if engine is not None:
            from cvti.event_adapters import zone_states_to_events
            events = zone_states_to_events(states, timestamp=timestamp)
            for alert in engine.evaluate(events, now=sim_now):
                sig = f"{alert.rule_name}:{alert.person_id}"
                if last_report.get(sig) != alert.title:
                    log.info(f"[RULE FIRED] {alert.rule_name} ({alert.priority.upper()}) "
                          f"person #{alert.person_id} — {alert.title}")
                    last_report[sig] = alert.title

        if args.show:
            annotated = monitor.annotate(result.orig_img.copy(), detections, states)
            cv2.imshow("retail_zones", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
