"""Validates concealment.py on synthetic skeleton sequences. No torch needed.

Run:  python tests/test_concealment.py

Geometry: torso with shoulders at y=100, hips at y=250 (body scale = 150px, torso axis
x=100). We drive a single active wrist through motions and assert the score behaves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.retail.concealment import ConcealmentDetector, PoseFrame  # noqa: E402

_SHOULDERS = {"left_shoulder": (80.0, 100.0), "right_shoulder": (120.0, 100.0)}
_HIPS = {"left_hip": (85.0, 250.0), "right_hip": (115.0, 250.0)}


def frame(ts: float, wrist: tuple[float, float], hips: bool = True) -> PoseFrame:
    kp: dict = {**_SHOULDERS, "left_wrist": wrist, "right_wrist": None}
    if hips:
        kp.update(_HIPS)
    return PoseFrame(track_id=1, timestamp=ts, keypoints=kp, bbox=(60, 90, 140, 260))


def _run(detector: ConcealmentDetector, frames: list[PoseFrame]):
    last = None
    for f in frames:
        last = detector.update([f], f.timestamp)[0]
    return last


def test_concealment_motion_fires() -> None:
    det = ConcealmentDetector()
    frames = []
    t = 0.0
    for _ in range(4):                       # reach OUT to the shelf
        frames.append(frame(t, (200.0, 110.0))); t += 0.1
    for _ in range(9):                       # pull hand IN to the waist and hold
        frames.append(frame(t, (105.0, 245.0))); t += 0.1
    result = _run(det, frames)
    assert result.score >= 0.6, result.score
    assert result.candidate, "a reach-then-conceal-to-waist motion should become a candidate"
    assert any("waist" in r for r in result.reasons), result.reasons
    assert not result.limited
    print(f"PASS concealment motion fires (score={result.score:.2f}, components={result.components})")


def test_normal_browsing_does_not_fire() -> None:
    det = ConcealmentDetector()
    # Hand stays up at the shelf the whole time, never goes to the waist.
    frames = [frame(i * 0.1, (180.0, 110.0)) for i in range(15)]
    result = _run(det, frames)
    assert result.score < 0.6, result.score
    assert not result.candidate, "browsing at a shelf must not fire a concealment candidate"
    print(f"PASS normal browsing stays quiet (score={result.score:.2f})")


def test_occluded_hips_degrades_gracefully() -> None:
    det = ConcealmentDetector()
    # No hips visible (caption banner / occlusion). Same motion, but waist features blind.
    frames = []
    t = 0.0
    for _ in range(4):
        frames.append(frame(t, (200.0, 110.0), hips=False)); t += 0.1
    for _ in range(9):
        frames.append(frame(t, (105.0, 245.0), hips=False)); t += 0.1
    result = _run(det, frames)
    assert result.limited, "no-hips case should set the limited flag"
    assert result.score < 0.6, "without hips the waist signal cannot confirm -> degraded"
    assert any("LIMITED" in r for r in result.reasons), result.reasons
    print(f"PASS occluded hips degrade gracefully (score={result.score:.2f}, limited={result.limited})")


def test_empty_window_is_zero() -> None:
    det = ConcealmentDetector()
    score, reasons, components, limited, destination = det.score_window([])
    assert score == 0.0 and not reasons and not limited and destination is None
    print("PASS empty window scores zero")


# A personal bag sitting at the person's side (not at the hip/waist line).
_BAG_BBOX = (180.0, 170.0, 240.0, 235.0)
_DROP_IN_BAG = (210.0, 200.0)   # inside the bag bbox, and far enough from the hip to not be "waist"


def _run_with_bags(detector, frames, bag_bboxes):
    last = None
    for f in frames:
        last = detector.update([f], f.timestamp, bag_bboxes=bag_bboxes)[0]
    return last


def test_bag_concealment_fires_with_destination_bag() -> None:
    det = ConcealmentDetector()
    frames = []
    t = 0.0
    for _ in range(4):                                   # reach OUT to the shelf
        frames.append(frame(t, (280.0, 150.0))); t += 0.1
    for _ in range(9):                                   # bring hand INTO the bag and hold
        frames.append(frame(t, _DROP_IN_BAG)); t += 0.1
    result = _run_with_bags(det, frames, [_BAG_BBOX])
    assert result.candidate, "reach-then-put-in-personal-bag should fire"
    assert result.destination == "bag", result.destination
    assert any("bag" in r for r in result.reasons), result.reasons
    print(f"PASS bag concealment fires (score={result.score:.2f}, dest={result.destination})")


def test_trolley_destination_is_safe() -> None:
    # IDENTICAL hand motion, but the destination is a trolley/basket — NOT a personal bag,
    # so no bag bbox is passed. Putting goods in a cart is normal shopping; must NOT fire.
    det = ConcealmentDetector()
    frames = []
    t = 0.0
    for _ in range(4):
        frames.append(frame(t, (280.0, 150.0))); t += 0.1
    for _ in range(9):
        frames.append(frame(t, _DROP_IN_BAG)); t += 0.1
    result = _run_with_bags(det, frames, None)           # no personal bag at the destination
    assert not result.candidate, "placing an item in a trolley must not fire a concealment candidate"
    assert result.destination is None, result.destination
    print(f"PASS trolley destination stays safe (score={result.score:.2f}, dest={result.destination})")


def test_state_cleared_when_track_leaves() -> None:
    det = ConcealmentDetector()
    det.update([frame(0.0, (105.0, 245.0))], 0.0)
    assert 1 in det._buffers
    det.update([], 0.5)                       # track gone this frame
    assert 1 not in det._buffers, "buffer should be dropped when the track disappears"
    print("PASS per-track state is cleared when the track leaves")


if __name__ == "__main__":
    test_concealment_motion_fires()
    test_normal_browsing_does_not_fire()
    test_occluded_hips_degrades_gracefully()
    test_empty_window_is_zero()
    test_bag_concealment_fires_with_destination_bag()
    test_trolley_destination_is_safe()
    test_state_cleared_when_track_leaves()
    print("\nAll concealment tests passed.")
