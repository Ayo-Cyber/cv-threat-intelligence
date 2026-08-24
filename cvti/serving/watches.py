"""Watches — describe in plain English what to follow, and the system tracks it.

Custom NL threats already let an operator say "tell me if X happens", but each hit
is a one-off alert with no memory: the same person triggers it again a minute
later as a brand-new event, and nothing says where they went.

A *watch* is different. It names a subject to FOLLOW:

    "the man in the red jacket near the till"
    "anyone who keeps returning to the spirits aisle"

and the system keeps an open CASE for them — first seen, last seen, how many
sightings, which camera, and their current box.

The hard part is binding a sentence to a specific person. Asking a VLM for pixel
coordinates is unreliable, so we don't: the tracker already knows where everyone
is, so we draw NUMBERED boxes on the tracked people and ask the model which
number matches the description. Geometry comes from the tracker (good at it),
semantics from the VLM (good at that) — and the answer is a track id, which is
stable across frames, so the case can genuinely follow someone.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Watch:
    name: str
    description: str

    @classmethod
    def from_config(cls, raw: dict) -> "Watch":
        return cls(str(raw.get("name") or "watch").strip(),
                   str(raw.get("description") or "").strip())


@dataclass
class Case:
    """An open case: one watched subject, followed over time."""
    camera_id: str
    watch: str
    track_id: int
    first_seen: float
    last_seen: float
    sightings: int = 1
    bbox: tuple | None = None
    reason: str = ""
    closed: bool = False

    @property
    def key(self) -> tuple:
        return (self.camera_id, self.watch, self.track_id)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def to_dict(self) -> dict:
        return {"camera_id": self.camera_id, "watch": self.watch, "track_id": self.track_id,
                "first_seen": self.first_seen, "last_seen": self.last_seen,
                "sightings": self.sightings, "duration_seconds": round(self.duration, 1),
                "bbox": list(self.bbox) if self.bbox else None,
                "reason": self.reason, "closed": self.closed}


def annotate_candidates(frame: Any, boxes: list) -> Any:
    """Draw numbered boxes so the model can answer with a NUMBER, not coordinates.

    boxes: [(track_id, x1, y1, x2, y2), ...] -> returns (annotated_frame, {n: track_id})
    """
    import cv2
    out = frame.copy()
    mapping: dict[int, int] = {}
    for n, (tid, x1, y1, x2, y2) in enumerate(boxes, start=1):
        mapping[n] = tid
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        tag = str(n)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 12, y1), (0, 220, 255), -1)
        cv2.putText(out, tag, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return out, mapping


def build_prompt(watches: list, n_people: int, scene: str = "") -> str:
    lines = "\n".join(f'- {w.name}: {w.description}' for w in watches)
    return (
        "You are watching a security camera. The people in view are outlined with "
        f"NUMBERED boxes (1 to {n_people}).\n"
        f"{('Scene: ' + scene + chr(10)) if scene else ''}"
        f"Watch list:\n{lines}\n\n"
        "For each watch, say which numbered person matches its description — or 0 if "
        "nobody in this image matches. Judge only what you can see; do not guess.\n"
        'Reply ONLY compact JSON: {"matches": [{"watch": "<exact watch name>", '
        '"person": <number or 0>, "reason": "<short reason>"}]}'
    )


def parse_matches(raw: str, watches: list, mapping: dict) -> list[dict]:
    """Turn the model's reply into [{watch, track_id, reason}], ignoring anything
    that isn't a watch we defined or a person number we actually drew."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    by_name = {w.name.lower(): w for w in watches}
    out = []
    for item in data.get("matches") or []:
        if not isinstance(item, dict):
            continue
        w = by_name.get(str(item.get("watch", "")).strip().lower())
        if w is None:                       # never invent a watch the operator didn't define
            continue
        try:
            person = int(item.get("person", 0))
        except (TypeError, ValueError):
            continue
        tid = mapping.get(person)
        if not person or tid is None:       # 0 = nobody matched; unknown number = ignore
            continue
        out.append({"watch": w.name, "track_id": tid,
                    "reason": str(item.get("reason", ""))[:200]})
    return out


class CaseBook:
    """Open cases, keyed by (camera, watch, track). Reopening the same subject
    updates their case instead of raising a fresh alert every sighting."""

    def __init__(self, *, stale_after: float = 60.0) -> None:
        self.stale_after = stale_after
        self._cases: dict[tuple, Case] = {}

    def observe(self, camera_id: str, watch: str, track_id: int, *,
                bbox: tuple | None = None, reason: str = "",
                now: float | None = None) -> tuple[Case, bool]:
        """Record a sighting. Returns (case, is_new) — is_new drives alerting."""
        now = time.time() if now is None else now
        key = (camera_id, watch, track_id)
        case = self._cases.get(key)
        if case is None or case.closed:
            case = Case(camera_id, watch, track_id, now, now, 1, bbox, reason)
            self._cases[key] = case
            return case, True
        case.last_seen = now
        case.sightings += 1
        if bbox:
            case.bbox = bbox
        if reason:
            case.reason = reason
        return case, False

    def expire(self, now: float | None = None) -> list[Case]:
        """Close cases whose subject hasn't been seen for a while.

        Closed cases are DELETED after being returned once: track ids never
        recur, so a closed case can never reopen — it used to sit in the dict
        forever, one dead Case per person who ever matched a watch, plus a
        full-dict scan every cycle. (RAM audit 24 Aug, service #1.)
        """
        now = time.time() if now is None else now
        closed = []
        for key, case in list(self._cases.items()):
            if case.closed:
                self._cases.pop(key, None)        # reported last cycle; gone now
                continue
            if (now - case.last_seen) > self.stale_after:
                case.closed = True
                closed.append(case)
        return closed

    def active(self, now: float | None = None) -> list[Case]:
        now = time.time() if now is None else now
        return [c for c in self._cases.values()
                if not c.closed and (now - c.last_seen) <= self.stale_after]

    @property
    def all_cases(self) -> list[Case]:
        return list(self._cases.values())
