"""Feedback store — read operator-labeled events from the events DB.

Every confirmed alert carries a `review` label the operator set:
  'true'  -> a real threat (TrueSight was right to confirm)   [positive]
  'false' -> a false alarm (TrueSight was wrong to confirm)   [negative]
  'ack'   -> acknowledged, outcome not asserted               [neutral]
  NULL    -> not yet reviewed

The store is read-only over the DB the running pipeline writes to, so it always
reflects the latest human feedback. Each labeled row also points at its
`evidence_dir` (frames + clip.mp4), which the dataset exporter turns into
training data.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

LABELS_POSITIVE = "true"
LABELS_NEGATIVE = "false"
LABELS_NEUTRAL = "ack"
REVIEWED = (LABELS_POSITIVE, LABELS_NEGATIVE, LABELS_NEUTRAL)


@dataclass
class LabeledEvent:
    id: int
    camera_id: str
    rule: str
    priority: str
    confidence: float
    reason: str
    review: str          # 'true' | 'false' | 'ack'
    evidence_dir: str | None
    ts: float

    @property
    def key(self) -> str:
        return f"{self.camera_id}::{self.rule}"

    @property
    def is_positive(self) -> bool:
        return self.review == LABELS_POSITIVE

    @property
    def is_negative(self) -> bool:
        return self.review == LABELS_NEGATIVE


class FeedbackStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def labeled_events(self, limit: int = 5000) -> list[LabeledEvent]:
        """All events the operator has labeled (true/false/ack), newest first."""
        if not Path(self.db_path).exists():
            return []
        try:
            con = self._connect()
        except sqlite3.OperationalError:
            return []
        try:
            rows = con.execute(
                "SELECT id,camera_id,rule,priority,confidence,reason,review,evidence_dir,ts "
                "FROM events WHERE review IN ('true','false','ack') "
                "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError:
            con.close()
            return []
        con.close()
        return [LabeledEvent(
            id=r["id"], camera_id=r["camera_id"] or "", rule=r["rule"] or "",
            priority=r["priority"] or "", confidence=float(r["confidence"] or 0.0),
            reason=r["reason"] or "", review=r["review"], evidence_dir=r["evidence_dir"],
            ts=float(r["ts"] or 0.0)) for r in rows]

    def examples(self, camera_id: str, rule: str, k: int = 4) -> list[LabeledEvent]:
        """Recent decisive (true/false) labels for one (camera, rule) — the few-shot
        memory the gate uses to calibrate itself to THIS site."""
        out = [e for e in self.labeled_events()
               if e.camera_id == camera_id and e.rule == rule and e.review in
               (LABELS_POSITIVE, LABELS_NEGATIVE)]
        return out[:k]

    def counts(self) -> dict:
        """Quick totals: reviewed / positive / negative."""
        evs = self.labeled_events()
        pos = sum(1 for e in evs if e.is_positive)
        neg = sum(1 for e in evs if e.is_negative)
        return {"reviewed": len(evs), "positive": pos, "negative": neg,
                "neutral": len(evs) - pos - neg}
