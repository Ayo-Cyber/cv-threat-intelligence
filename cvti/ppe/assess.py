"""Detect → associate with the right person → three states → evidence over
time → verdict.

The claims are kept apart on purpose:
  * "a hard hat is in the frame"          (detector)
  * "THIS person is wearing it"           (association: the box sits on their head)
  * "this person has NO hard hat"         (absent: head visible, checked N times, never there)
  * "cannot tell"                         (unknown: head cut off / two people overlap / detector down)
  * "this person violates the policy"     (a required item is confidently absent)

Every observation is timestamped and expires: a helmet seen 30 seconds ago is
not a helmet worn now. Track ids are the tracker's temporary camera-local ids —
no faces, no identities.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from cvti.ppe.policy import PPEItem, PPEPolicy

PRESENT, ABSENT, UNKNOWN = "present", "absent", "unknown"
VIOLATION, COMPLIANT, UNABLE, NOT_REQUIRED = "violation", "compliant", "unable", "not_required"

# Where on a standing person each region sits, as fractions of box height.
# Deliberately generous bands: a box is not a skeleton. Pose estimation can
# replace these later if it demonstrably earns its cost on hard cases.
REGION_BANDS: dict[str, tuple[float, float]] = {
    "head": (0.00, 0.22),
    "face": (0.04, 0.26),
    "torso": (0.18, 0.62),
    "hands": (0.38, 0.78),
    "feet": (0.84, 1.03),
}
MIN_REGION_PX = 16          # a region band shorter than this cannot carry a verdict (16 Sep: 70px-tall workers in hard hats were called bare-headed — their 15px heads were below what the detector resolves)
MIN_VISIBLE_FRACTION = 0.6  # of the region's area inside the frame
SIDE_PAD_FRACTION = 0.15    # items overhang the box (brim, elbows): widen the region


def region_box(person_box: tuple, region: str) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in person_box)
    h, w = y2 - y1, x2 - x1
    lo, hi = REGION_BANDS[region]
    pad = SIDE_PAD_FRACTION * w
    return (x1 - pad, y1 + lo * h, x2 + pad, y1 + hi * h)


def region_visible(person_box: tuple, region: str, frame_hw: tuple,
                   min_person_height_px: int) -> tuple[bool, str]:
    """Can this region be assessed at all? (False, why-not) when it cannot."""
    fh, fw = frame_hw[0], frame_hw[1]
    x1, y1, x2, y2 = (float(v) for v in person_box)
    if (y2 - y1) < min_person_height_px:
        return False, "person too small to assess"
    rx1, ry1, rx2, ry2 = region_box(person_box, region)
    if (ry2 - ry1) < MIN_REGION_PX:
        return False, f"{region} region too small"
    area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
    ix1, iy1 = max(rx1, 0.0), max(ry1, 0.0)
    ix2, iy2 = min(rx2, float(fw)), min(ry2, float(fh))
    inside = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inside / area < MIN_VISIBLE_FRACTION:
        return False, f"{region} outside the frame"
    if region == "feet" and y2 >= fh - 2:
        # The box is clipped by the bottom edge: the feet are most likely
        # below the frame, not "not wearing boots".
        return False, "feet cut off by frame edge"
    return True, ""


def _inside(pt: tuple[float, float], box: tuple) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


# How much of an item box (or of the region band, whichever is shorter) must
# overlap the band vertically for the item to count as worn THERE.
MIN_BAND_OVERLAP = 0.4


def item_on_region(item_box: tuple, region: tuple) -> bool:
    """Does this detection sit on this body region?

    Centre-in-band was the first rule and it failed on the one garment every
    lab requires (16 Sep): a lab coat runs shoulders-to-knees, so its box
    centre lands at ~65 % of body height — just under the torso band's edge —
    and coats worn in plain sight read as absent. The test is now the box's
    horizontal centre inside the band plus a real vertical overlap with it.
    A helmet carried at waist height still has no overlap with the head band;
    a coat, vest or apron overlaps the torso band however long it hangs.
    """
    cx = (float(item_box[0]) + float(item_box[2])) / 2.0
    if not (region[0] <= cx <= region[2]):
        return False
    iy1, iy2 = float(item_box[1]), float(item_box[3])
    overlap = min(iy2, region[3]) - max(iy1, region[1])
    if overlap <= 0:
        return False
    shorter = max(1.0, min(iy2 - iy1, region[3] - region[1]))
    return overlap / shorter >= MIN_BAND_OVERLAP


@dataclass
class ItemObservation:
    item: str
    status: str            # present | absent | unknown
    score: float = 0.0
    reason: str = ""


@dataclass
class PersonObservation:
    track_id: int
    box: tuple
    zones: tuple[str, ...]
    required: tuple[str, ...]
    items: dict[str, ItemObservation] = field(default_factory=dict)


def observe_people(people: list, detections: list[dict] | None, policy: PPEPolicy,
                   frame_hw: tuple, zones_by_track: dict | None = None, *,
                   zero_shot: bool = False) -> list[PersonObservation]:
    """One frame's per-person, per-required-item observations.

    `people`: [(track_id, x1, y1, x2, y2), ...] from the engine's tracker.
    `detections`: [{phrase, score, box}] from the item detector, or None when
    the detector could not answer this frame (everything becomes unknown —
    coverage degrades visibly instead of people being marked compliant).
    `zero_shot`: the detector is the open-vocab model — items whose absence
    it cannot reliably see are observed in SHADOW: absent becomes unknown.
    """
    zones_by_track = zones_by_track or {}
    shadow = set(policy.shadow_items(zero_shot))
    observations: list[PersonObservation] = []
    persons: list[tuple[int, tuple]] = []
    for row in people or []:
        tid, x1, y1, x2, y2 = row[0], row[1], row[2], row[3], row[4]
        if tid is None:
            continue
        persons.append((int(tid), (float(x1), float(y1), float(x2), float(y2))))

    # Association: an item belongs to the ONE person whose matching body region
    # it sits on (see item_on_region). Two candidates = ownership unclear =
    # unknown for both. Zero candidates (a helmet in a hand, boots on a
    # shelf) = irrelevant.
    assigned: dict[tuple[int, str], float] = {}
    ambiguous: set[tuple[int, str]] = set()
    for det in detections or []:
        item = policy.item_for_phrase(str(det.get("phrase", "")))
        if item is None:
            continue
        bx = det.get("box") or ()
        if len(bx) < 4:
            continue
        owners = [tid for tid, pbox in persons if item_on_region(bx, region_box(pbox, item.region))]
        if len(owners) == 1:
            key = (owners[0], item.key)
            assigned[key] = max(assigned.get(key, 0.0), float(det.get("score", 0.0)))
        elif len(owners) > 1:
            for tid in owners:
                ambiguous.add((tid, item.key))

    for tid, pbox in persons:
        zones = tuple(zones_by_track.get(tid, ()) or ())
        required = policy.required_for(zones)
        obs = PersonObservation(track_id=tid, box=pbox, zones=zones, required=required)
        for key in required:
            item: PPEItem = policy.items[key]
            visible, why = region_visible(pbox, item.region, frame_hw, policy.min_person_height_px)
            if not visible:
                obs.items[key] = ItemObservation(key, UNKNOWN, 0.0, why)
            elif detections is None:
                obs.items[key] = ItemObservation(key, UNKNOWN, 0.0, "detector unavailable")
            elif (tid, key) in ambiguous:
                obs.items[key] = ItemObservation(key, UNKNOWN, 0.0,
                                                 "ownership unclear (overlapping people)")
            elif assigned.get((tid, key), 0.0) >= item.min_score:
                obs.items[key] = ItemObservation(key, PRESENT, assigned[(tid, key)],
                                                 f"{item.label} on {item.region}")
            elif key in shadow:
                obs.items[key] = ItemObservation(
                    key, UNKNOWN, assigned.get((tid, key), 0.0),
                    f"{item.label} not seen — zero-shot detector cannot confirm absence (shadow)")
            else:
                obs.items[key] = ItemObservation(key, ABSENT, assigned.get((tid, key), 0.0),
                                                 f"no {item.label} seen on {item.region}")
        observations.append(obs)
    return observations


class TrackEvidence:
    """Timestamped observations per item for one track, with expiry.

    A verdict needs `confirm` DETERMINATE observations inside the window:
    absent only when they are unanimous (a violation is the costly call),
    present on a majority, anything else stays unknown.
    """

    def __init__(self, window_seconds: float = 8.0, confirm: int = 3) -> None:
        self.window = float(window_seconds)
        self.confirm = max(1, int(confirm))
        self._obs: dict[str, deque] = {}
        self.last_seen: float = 0.0
        self.first_seen: float | None = None

    def add(self, obs: PersonObservation, now: float) -> None:
        self.last_seen = now
        if self.first_seen is None:
            self.first_seen = now
        for key, io in obs.items.items():
            self._obs.setdefault(key, deque()).append((now, io.status, io.score, io.reason))
        self._prune(now)

    def _prune(self, now: float) -> None:
        for q in self._obs.values():
            while q and now - q[0][0] > self.window:
                q.popleft()

    def verdict(self, item: str, now: float) -> tuple[str, str]:
        """(status, detail) for one item as of `now`."""
        self._prune(now)
        seq = list(self._obs.get(item, ()))
        if not seq:
            return UNKNOWN, "no observations yet"
        determinate = [o for o in seq if o[1] != UNKNOWN]
        if len(determinate) < self.confirm:
            latest_unknown = next((o for o in reversed(seq) if o[1] == UNKNOWN), None)
            if latest_unknown is not None and len(determinate) == 0:
                return UNKNOWN, latest_unknown[3]
            return UNKNOWN, f"{len(determinate)} of {self.confirm} checks so far"
        last = determinate[-self.confirm:]
        n_present = sum(1 for o in last if o[1] == PRESENT)
        if n_present == 0:
            return ABSENT, f"absent in {self.confirm} consecutive checks"
        if n_present * 2 > self.confirm:
            best = max(o[2] for o in last if o[1] == PRESENT)
            return PRESENT, f"seen in {n_present}/{self.confirm} checks (score {best:.2f})"
        return UNKNOWN, "inconsistent observations"

    def determinate_count(self, item: str) -> int:
        return sum(1 for o in self._obs.get(item, ()) if o[1] != UNKNOWN)


@dataclass
class Compliance:
    status: str                                  # violation | compliant | unable | not_required
    required: tuple[str, ...]
    missing: tuple[str, ...] = ()
    present: tuple[str, ...] = ()
    unknown: dict[str, str] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def summary(self, items: dict[str, PPEItem] | None = None) -> str:
        def label(k: str) -> str:
            return items[k].label if items and k in items else k.replace("_", " ")
        parts: list[str] = []
        if self.missing:
            parts.append("missing " + ", ".join(label(k) for k in self.missing))
        if self.present:
            parts.append(", ".join(label(k) for k in self.present) + " worn")
        for k, why in self.unknown.items():
            parts.append(f"{label(k)}: {why}")
        return "; ".join(parts) or self.status

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "required": list(self.required),
                "missing": list(self.missing), "present": list(self.present),
                "unknown": dict(self.unknown), "details": dict(self.details),
                "observed": f"{len(self.present)} of {len(self.required)} items observed"}


def assess_compliance(required: tuple[str, ...], verdicts: dict[str, tuple[str, str]]) -> Compliance:
    """The policy decision. Any required item confidently absent = violation;
    every required item confidently present = compliant; otherwise unable.
    Two of three present is NOT 67% compliant — and it is not a violation
    either, until the third is confidently absent."""
    if not required:
        return Compliance(NOT_REQUIRED, ())
    missing = tuple(k for k in required if verdicts.get(k, (UNKNOWN, ""))[0] == ABSENT)
    present = tuple(k for k in required if verdicts.get(k, (UNKNOWN, ""))[0] == PRESENT)
    unknown = {k: verdicts.get(k, (UNKNOWN, "no observations yet"))[1]
               for k in required if verdicts.get(k, (UNKNOWN, ""))[0] == UNKNOWN}
    details = {k: v[1] for k, v in verdicts.items() if k in required}
    if missing:
        status = VIOLATION
    elif len(present) == len(required):
        status = COMPLIANT
    else:
        status = UNABLE
    return Compliance(status, tuple(required), missing, present, unknown, details)


def violation_confidence(evidence: TrackEvidence, missing: tuple[str, ...]) -> float:
    """How sure the ABSENT call is: grows with agreeing checks, capped well
    below certainty — this is a detector not seeing something, repeatedly."""
    if not missing:
        return 0.0
    n = min(evidence.determinate_count(k) for k in missing)
    return round(min(0.92, 0.55 + 0.08 * n), 2)


def ceil_half(n: int) -> int:
    return int(math.ceil(n / 2.0))
