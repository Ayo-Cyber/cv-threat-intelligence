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


@dataclass
class PersonZoneState:
    """Per-detection snapshot returned by RetailZoneMonitor.update()."""

    tracker_id: int | None
    bbox: tuple[int, int, int, int]
    zones: list[str] = field(default_factory=list)        # zones currently occupied
    dwell_seconds: dict[str, float] = field(default_factory=dict)
    loitering: bool = False                               # crossed any zone's dwell threshold

    def label(self) -> str:
        tag = f"#{self.tracker_id}" if self.tracker_id is not None else "#?"
        if not self.zones:
            return tag
        z = self.zones[0]
        dwell = self.dwell_seconds.get(z, 0.0)
        flag = " LOITER" if self.loitering else ""
        return f"{tag} {z} {dwell:.1f}s{flag}"


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
        polygon = np.array(entry["polygon"], dtype=np.int64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError(
                f"Zone '{entry.get('name')}' polygon must be a list of >=3 [x, y] points."
            )
        specs.append(
            ZoneSpec(
                name=str(entry["name"]),
                polygon=polygon,
                anchors=parse_anchors(entry.get("anchors")),
                kind=str(entry.get("kind", "shelf")),
                dwell_alert_seconds=entry.get("dwell_alert_seconds"),
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

    def __init__(self, zones: list[ZoneSpec], dwell_grace_seconds: float = 0.0) -> None:
        self.zones = zones
        self.dwell_grace_seconds = dwell_grace_seconds
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

    def update(self, detections: sv.Detections, timestamp: float) -> list[PersonZoneState]:
        n = len(detections)
        # Boolean membership mask per zone, aligned to detection order.
        masks = {name: zone.trigger(detections) for name, zone in self._sv_zones.items()}

        current_keys: set[tuple[int, str]] = set()
        states: list[PersonZoneState] = []

        for i in range(n):
            tid = _tracker_id_at(detections, i)
            bbox = tuple(int(v) for v in detections.xyxy[i])
            state = PersonZoneState(tracker_id=tid, bbox=bbox)  # type: ignore[arg-type]

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
                entered = self._entered_at.setdefault(key, timestamp)
                self._last_in_zone[key] = timestamp
                dwell = max(0.0, timestamp - entered)
                state.dwell_seconds[name] = dwell
                threshold = self._spec_by_name[name].dwell_alert_seconds
                if threshold is not None and dwell >= threshold:
                    state.loitering = True

            states.append(state)

        # Forget a (track, zone) pair only after it has been absent longer than the grace
        # window, so brief boundary jitter / 1-frame track loss does not reset dwell.
        for key in list(self._entered_at):
            if key in current_keys:
                continue
            last = self._last_in_zone.get(key, self._entered_at[key])
            if timestamp - last > self.dwell_grace_seconds:
                del self._entered_at[key]
                self._last_in_zone.pop(key, None)

        return states

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
    args = _parse_args()
    zones = load_zone_config(args.zones)
    monitor = RetailZoneMonitor(zones, dwell_grace_seconds=args.dwell_grace)
    print(f"[retail_zones] Loaded {len(zones)} zone(s): {', '.join(z.name for z in zones)}")

    # Optional: run the Customization Engine on zone presence so zone+time+dwell RULES fire.
    engine = None
    sim_now = None
    if args.rules:
        from customization import CustomizationEngine, zone_states_to_events  # noqa: F401
        engine = CustomizationEngine(args.rules)
        if args.simulate_time:
            from datetime import datetime
            hh, mm = (int(x) for x in args.simulate_time.split(":"))
            base = datetime.now()
            sim_now = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
            print(f"[retail_zones] Simulating clock at {args.simulate_time} for time-filtered rules.")

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
                print(f"[{event}] {line}")
                last_report[s.tracker_id] = line

        if engine is not None:
            from customization import zone_states_to_events
            events = zone_states_to_events(states, timestamp=timestamp)
            for alert in engine.evaluate(events, now=sim_now):
                sig = f"{alert.rule_name}:{alert.person_id}"
                if last_report.get(sig) != alert.title:
                    print(f"[RULE FIRED] {alert.rule_name} ({alert.priority.upper()}) "
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
