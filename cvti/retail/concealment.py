"""concealment.py — pose-based concealment-motion detector (the action layer, V1).

WHAT THIS IS
------------
The "gesture/action recognition" layer, built the way Veesion effectively works:
recognise the *motion* of concealment rather than tracking a product into a bag. V1 is
a transparent, tunable HEURISTIC over a per-person skeleton sequence — NOT a trained
model — which is consistent with the locked decision to anchor V1 on VLM verification and
defer model training to Phase 2. It is a recall-oriented CANDIDATE generator: it answers
"did this person's hands just do something concealment-shaped?" and hands the candidate to
the Verification Gate (VLM) for the real confirm/reject.

THE SIGNAL
----------
Concealment = reach out to merchandise, then bring the hand IN to the body — to the
waist / pocket / waistband / under clothing — often pulling the arm in and lingering. So
the temporal features are:
  * hand moved close to the hip/waist line          (f_waist)
  * arm reached out then retracted toward the body  (f_retract)
  * a hand lingered at the waist for several frames  (f_dwell)
These are normalised by body scale (shoulder<->hip distance) so they are invariant to how
far the person is from the camera.

THE SEAM (Phase 2)
------------------
`score_window()` is the swappable head. Today it is a weighted heuristic. To upgrade,
replace its body with a trained pose-sequence classifier (LSTM / 1D-CNN / small
transformer) fed the same per-frame feature vectors — nothing else changes.

This core is torch-free and unit-tested on synthetic skeleton sequences
(tests/test_concealment.py). A YOLO-pose adapter + demo live at the bottom (needs
ultralytics; pose keypoints come from yolov8n-pose.pt).
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Any

# COCO-17 keypoint indices we use. Hips (11,12) are the addition over detector.py's
# violence pose features and are essential for the hand-to-waist signal.
COCO_KEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

Point = tuple[float, float]

# Personal-carry containers that count as a CONCEALMENT destination. A shopping
# trolley / basket / cart is deliberately NOT here: putting goods in a cart is normal
# (they'll pay at the counter), and carts aren't COCO classes, so "hand into trolley"
# never produces a bag bbox and never fires. Putting goods into your OWN bag does.
BAG_CLASSES = ("backpack", "handbag", "suitcase")
COCO_BAG_IDS = (24, 26, 28)  # backpack, handbag, suitcase in COCO

# ---- Tunables (all overridable via ConcealmentDetector.__init__) -------------
WINDOW_SECONDS = 1.2      # how much recent history each decision looks at
WAIST_NEAR = 0.60         # normalised hand<->hip distance considered "at the waist"
BAG_NEAR = 0.60           # normalised hand<->bag distance considered "reaching the bag"
BAG_AT = 0.20             # normalised hand<->bag distance considered "in the bag"
RETRACT_SCALE = 0.80      # normalised arm-extension drop that counts as a full retract
DWELL_FRAMES = 6          # frames a hand at a concealment destination for full dwell credit
SCORE_THRESHOLD = 0.72    # per-frame score above which the window looks concealment-like
MIN_CANDIDATE_FRAMES = 6  # consecutive frames over threshold before a candidate fires
                          # (0.60/4 fired on normal hand-near-waist posture — too noisy;
                          #  tightened so it needs stronger, more sustained concealment shape)
WEIGHTS = (0.40, 0.30, 0.30)  # (destination_reached, retract, dwell), must sum to 1.0


@dataclass
class PoseFrame:
    """One person's skeleton at one instant. Missing/low-confidence joints are None."""

    track_id: int
    timestamp: float
    keypoints: dict[str, Point | None]
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class _FrameFeatures:
    timestamp: float
    hand_to_hip: float | None      # normalised; None if hips or wrists unavailable
    hand_to_bag: float | None      # normalised; None if no bag detected or no wrists
    lateral_reach: float | None    # normalised |wrist.x - torso_axis.x|; None if unavailable
    hand_at_waist: bool
    hand_at_bag: bool
    has_hips: bool


@dataclass
class ConcealmentAssessment:
    track_id: int
    score: float
    candidate: bool
    destination: str | None = None  # "waist" | "bag" | None — where the item went
    reasons: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    limited: bool = False          # True when hips were never seen (occluded) -> degraded


def _dist(a: Point | None, b: Point | None) -> float | None:
    if a is None or b is None:
        return None
    return hypot(a[0] - b[0], a[1] - b[1])


def _mid(a: Point | None, b: Point | None) -> Point | None:
    if a is not None and b is not None:
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return a or b


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _point_to_bbox(p: Point, box: tuple[float, float, float, float]) -> float:
    """Distance from a point to an axis-aligned box; 0 if the point is inside."""
    x, y = p
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return hypot(dx, dy)


def _body_scale(kp: dict[str, Point | None], bbox: tuple[float, float, float, float] | None) -> float | None:
    shoulder_c = _mid(kp.get("left_shoulder"), kp.get("right_shoulder"))
    hip_c = _mid(kp.get("left_hip"), kp.get("right_hip"))
    s = _dist(shoulder_c, hip_c)
    if s and s > 1e-3:
        return s
    # Fallbacks when torso keypoints are partly missing.
    shoulder_w = _dist(kp.get("left_shoulder"), kp.get("right_shoulder"))
    if shoulder_w and shoulder_w > 1e-3:
        return shoulder_w * 1.5
    if bbox is not None:
        height = abs(bbox[3] - bbox[1])
        if height > 1e-3:
            return height * 0.5
    return None


def _frame_features(
    frame: PoseFrame,
    bag_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> _FrameFeatures:
    kp = frame.keypoints
    scale = _body_scale(kp, frame.bbox)
    shoulder_c = _mid(kp.get("left_shoulder"), kp.get("right_shoulder"))
    hip_c = _mid(kp.get("left_hip"), kp.get("right_hip"))
    wrists = [w for w in (kp.get("left_wrist"), kp.get("right_wrist")) if w is not None]
    has_hips = hip_c is not None

    # Vertical torso axis: x of the body centerline (shoulder/hip), for lateral reach.
    axis_pts = [p[0] for p in (shoulder_c, hip_c) if p is not None]
    torso_axis_x = sum(axis_pts) / len(axis_pts) if axis_pts else None

    hand_to_hip: float | None = None
    hand_to_bag: float | None = None
    lateral_reach: float | None = None
    hand_at_waist = False
    hand_at_bag = False

    if scale and scale > 1e-3:
        if hip_c is not None and wrists:
            dists = [d for d in (_dist(w, hip_c) for w in wrists) if d is not None]
            if dists:
                hand_to_hip = min(dists) / scale
        if torso_axis_x is not None and wrists:
            # Most-reaching hand: max horizontal offset from the torso centerline.
            lateral_reach = max(abs(w[0] - torso_axis_x) for w in wrists) / scale
        # "At the waist" = a wrist near the hip AND not up above the shoulders.
        if hip_c is not None and shoulder_c is not None and wrists:
            for w in wrists:
                d = _dist(w, hip_c)
                if d is not None and (d / scale) < WAIST_NEAR and w[1] >= shoulder_c[1]:
                    hand_at_waist = True
                    break
        # Hand reaching into a PERSONAL bag (concealment destination).
        if bag_bboxes and wrists:
            nearest = min(_point_to_bbox(w, b) for w in wrists for b in bag_bboxes) / scale
            hand_to_bag = nearest
            hand_at_bag = nearest < BAG_AT

    return _FrameFeatures(
        timestamp=frame.timestamp,
        hand_to_hip=hand_to_hip,
        hand_to_bag=hand_to_bag,
        lateral_reach=lateral_reach,
        hand_at_waist=hand_at_waist,
        hand_at_bag=hand_at_bag,
        has_hips=has_hips,
    )


class ConcealmentDetector:
    """Per-track concealment-motion scorer over a rolling skeleton window."""

    def __init__(
        self,
        window_seconds: float = WINDOW_SECONDS,
        score_threshold: float = SCORE_THRESHOLD,
        min_candidate_frames: int = MIN_CANDIDATE_FRAMES,
        waist_near: float = WAIST_NEAR,
        retract_scale: float = RETRACT_SCALE,
        dwell_frames: int = DWELL_FRAMES,
        weights: tuple[float, float, float] = WEIGHTS,
    ) -> None:
        self.window_seconds = window_seconds
        self.score_threshold = score_threshold
        self.min_candidate_frames = min_candidate_frames
        self.waist_near = waist_near
        self.retract_scale = retract_scale
        self.dwell_frames = dwell_frames
        self.weights = weights
        self._buffers: dict[int, deque[_FrameFeatures]] = {}
        self._over_threshold: dict[int, int] = {}

    def update(
        self,
        pose_frames: list[PoseFrame],
        timestamp: float,
        bag_bboxes: list[tuple[float, float, float, float]] | None = None,
    ) -> list[ConcealmentAssessment]:
        """bag_bboxes: detected PERSONAL-bag boxes this frame (backpack/handbag/suitcase).
        Trolleys/baskets are not personal bags, so pass nothing for them — they stay safe."""
        active_ids = {f.track_id for f in pose_frames}
        results: list[ConcealmentAssessment] = []

        for frame in pose_frames:
            buf = self._buffers.setdefault(frame.track_id, deque())
            buf.append(_frame_features(frame, bag_bboxes))
            cutoff = timestamp - self.window_seconds
            while buf and buf[0].timestamp < cutoff:
                buf.popleft()

            score, reasons, components, limited, destination = self.score_window(list(buf))

            if score >= self.score_threshold:
                self._over_threshold[frame.track_id] = self._over_threshold.get(frame.track_id, 0) + 1
            else:
                self._over_threshold[frame.track_id] = 0
            candidate = self._over_threshold[frame.track_id] >= self.min_candidate_frames

            results.append(ConcealmentAssessment(
                track_id=frame.track_id, score=score, candidate=candidate, destination=destination,
                reasons=reasons, components=components, limited=limited,
            ))

        # Drop state for tracks that have left the scene.
        for tid in list(self._buffers):
            if tid not in active_ids:
                del self._buffers[tid]
                self._over_threshold.pop(tid, None)
        return results

    def score_window(
        self, window: list[_FrameFeatures]
    ) -> tuple[float, list[str], dict[str, float], bool, str | None]:
        """Heuristic scorer — THE SEAM. Replace this body with a trained classifier later.

        Returns (score in [0,1], reasons, components, limited-flag, destination).
        destination is "waist" | "bag" | None — which concealment destination drove it.
        """
        if not window:
            return 0.0, [], {}, False, None

        recent = window[-1]
        hips_ever = any(f.has_hips for f in window)
        hand_to_hip_vals = [f.hand_to_hip for f in window if f.hand_to_hip is not None]
        hand_to_bag_vals = [f.hand_to_bag for f in window if f.hand_to_bag is not None]
        reach_vals = [f.lateral_reach for f in window if f.lateral_reach is not None]
        dwell_count = sum(1 for f in window if f.hand_at_waist or f.hand_at_bag)

        # f_waist: how close the nearest hand got to the waist over the window.
        f_waist = 0.0
        if hand_to_hip_vals:
            f_waist = _clamp01(1.0 - (min(hand_to_hip_vals) / self.waist_near))

        # f_bag: how close the nearest hand got to a personal bag over the window.
        f_bag = 0.0
        if hand_to_bag_vals:
            f_bag = _clamp01(1.0 - (min(hand_to_bag_vals) / BAG_NEAR))

        # The item's concealment destination: waist (pocket/waistband) or a personal bag.
        f_dest = max(f_waist, f_bag)
        destination: str | None = None
        if f_dest >= 0.5:
            destination = "bag" if f_bag >= f_waist else "waist"

        # f_retract: hand reached out laterally then pulled back in toward the torso/bag,
        # AND ended at a concealment destination. max(reach) early - recent(reach).
        f_retract = 0.0
        if reach_vals and recent.lateral_reach is not None:
            retract = max(reach_vals) - recent.lateral_reach
            ended_low = 1.0 if (recent.hand_at_waist or recent.hand_at_bag) else 0.3
            f_retract = _clamp01(retract / self.retract_scale) * ended_low

        # f_dwell: a hand lingered at a concealment destination (pocket/waistband/bag).
        f_dwell = _clamp01(dwell_count / float(self.dwell_frames))

        w_dest, w_retract, w_dwell = self.weights
        score = w_dest * f_dest + w_retract * f_retract + w_dwell * f_dwell

        reasons: list[str] = []
        if destination == "bag":
            reasons.append(f"hand reached a personal bag (f_bag={f_bag:.2f})")
        elif f_waist >= 0.5:
            reasons.append(f"hand reached the waist line (f_waist={f_waist:.2f})")
        if f_retract >= 0.5:
            reasons.append(f"arm reached out then retracted to body (f_retract={f_retract:.2f})")
        if f_dwell >= 0.5:
            dest_word = destination or "concealment point"
            reasons.append(f"hand lingered at the {dest_word} ({dwell_count} frames)")

        limited = not hips_ever and not hand_to_bag_vals
        if limited:
            reasons.append("LIMITED: hips never visible (occlusion) — score degraded")

        components = {"f_waist": round(f_waist, 3),
                      "f_bag": round(f_bag, 3),
                      "f_retract": round(f_retract, 3),
                      "f_dwell": round(f_dwell, 3)}
        return score, reasons, components, limited, destination


# ---------------------------------------------------------------------------
# YOLO-pose adapter + CLI demo (needs ultralytics; lazy import keeps core torch-free).
# ---------------------------------------------------------------------------

def keypoints_from_ultralytics(result: Any, person_index: int) -> dict[str, Point | None]:
    """Pull the joints we need from one person in a YOLO-pose result, or None each."""
    kp: dict[str, Point | None] = {name: None for name in COCO_KEYPOINTS}
    kdata = getattr(result, "keypoints", None)
    if kdata is None or kdata.xy is None:
        return kp
    xy = kdata.xy[person_index]
    conf = kdata.conf[person_index] if kdata.conf is not None else None
    for name, idx in COCO_KEYPOINTS.items():
        x, y = float(xy[idx][0]), float(xy[idx][1])
        c = float(conf[idx]) if conf is not None else 1.0
        kp[name] = (x, y) if (c >= 0.3 and (x > 0 or y > 0)) else None
    return kp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pose-based concealment-motion demo.")
    p.add_argument("--source", required=True, help="Video path / RTSP URL / webcam index.")
    p.add_argument("--pose-weights", default="yolov8n-pose.pt")
    p.add_argument("--tracker", default="configs/bytetrack_retail.yaml")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--object-weights", default="yolov8n.pt",
                   help="Object detector for personal-bag destinations (backpack/handbag/suitcase).")
    p.add_argument("--no-bags", action="store_true",
                   help="Disable bag-destination detection (waist/pocket concealment only).")
    p.add_argument("--debug", action="store_true", help="Print per-track score each frame it changes.")
    p.add_argument("--show", action="store_true", help="Open a live window with the score overlaid.")
    p.add_argument("--save", default="", help="Write an annotated .mp4 to this path.")
    return p.parse_args()


def _draw_overlay(cv2: Any, frame: Any, pose_frames: list[PoseFrame],
                  assess_by_id: dict[int, ConcealmentAssessment]) -> Any:
    out = frame.copy()
    for pf in pose_frames:
        a = assess_by_id.get(pf.track_id)
        if a is None or pf.bbox is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in pf.bbox)
        red = (0, 0, 255)
        green = (0, 180, 0)
        amber = (0, 170, 255)
        color = red if a.candidate else amber if a.score >= 0.4 else green
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"CONCEAL>{(a.destination or '?').upper()}" if a.candidate else f"{a.score:.2f}"
        label = f"#{pf.track_id} {tag}"
        cv2.rectangle(out, (x1, max(0, y1 - 22)), (x1 + 11 * len(label), y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


def main() -> None:
    args = _parse_args()
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"Demo needs ultralytics + opencv ({exc}). Run: pip install -r requirements.txt")

    detector = ConcealmentDetector()
    model = YOLO(args.pose_weights)
    object_model = None if args.no_bags else YOLO(args.object_weights)
    source = int(args.source) if str(args.source).isdigit() else args.source

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    dt = 1.0 / fps if fps > 1e-3 else 1.0 / 25.0

    last: dict[int, str] = {}
    n = 0
    peak: dict[int, float] = {}
    writer = None
    for result in model.track(source=source, stream=True, persist=True, tracker=args.tracker,
                              conf=args.conf, classes=[0], verbose=False):
        ids = result.boxes.id if result.boxes is not None else None
        pose_frames: list[PoseFrame] = []
        if ids is not None and result.keypoints is not None:
            for i in range(len(ids)):
                tid = int(ids[i])
                pose_frames.append(PoseFrame(
                    track_id=tid, timestamp=n * dt,
                    keypoints=keypoints_from_ultralytics(result, i),
                    bbox=tuple(float(v) for v in result.boxes.xyxy[i]),
                ))
        bag_bboxes = None
        if object_model is not None:
            obj = object_model(result.orig_img, classes=list(COCO_BAG_IDS), conf=0.35, verbose=False)[0]
            if obj.boxes is not None and len(obj.boxes) > 0:
                bag_bboxes = [tuple(float(v) for v in b) for b in obj.boxes.xyxy]
        assessments = detector.update(pose_frames, n * dt, bag_bboxes=bag_bboxes)
        assess_by_id = {a.track_id: a for a in assessments}
        for a in assessments:
            peak[a.track_id] = max(peak.get(a.track_id, 0.0), a.score)
            line = f"#{a.track_id} score={a.score:.2f} cand={a.candidate} dest={a.destination} {a.components}"
            if args.debug and last.get(a.track_id) != line and (a.score > 0.3 or a.candidate):
                tag = "CONCEAL-CANDIDATE" if a.candidate else "motion"
                print(f"[{tag}] {line} {'; '.join(a.reasons)}")
                last[a.track_id] = line

        if args.show or args.save:
            annotated = _draw_overlay(cv2, result.orig_img, pose_frames, assess_by_id)
            if args.save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (w, h))
                writer.write(annotated)
            if args.show:
                cv2.imshow("concealment", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        n += 1

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print("\n=== concealment summary ===")
    print(f"frames processed: {n}")
    for tid, pk in sorted(peak.items()):
        print(f"  track #{tid}: peak_score={pk:.2f}")
    if args.save:
        print(f"annotated video written to {args.save}")


if __name__ == "__main__":
    main()
