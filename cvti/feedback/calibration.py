"""Online calibration — turn operator labels into per-(camera, rule) actions.

We can only measure gate PRECISION (of the alerts it confirmed, how many the
operator said were real). So the loop targets over-confirmation: a (camera, rule)
pair the operator keeps marking as a false alarm gets DEMOTED — it still fires and
is stored (so the operator can keep correcting it), but it stops paging them. Pairs
the operator keeps confirming become TRUSTED.

This is deterministic and transparent (no black-box), runs on the edge box with no
GPU, and takes effect the moment the pipeline reloads calibration.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from cvti.feedback.store import FeedbackStore, LabeledEvent

from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Tuning: how much evidence before we act, and the precision below which a rule is
# "noisy" enough to demote.
MIN_NEGATIVES = 3          # need at least this many false alarms before demoting
MIN_REVIEWED = 4           # ...and this many decisive labels total
DEMOTE_BELOW = 0.34        # precision below this (with enough evidence) -> demote
TRUST_ABOVE = 0.80         # precision at/above this (with >=3 positives) -> trusted

ACTION_DEMOTE = "demote"
ACTION_TRUSTED = "trusted"
ACTION_WATCH = "watch"


@dataclass
class RuleStat:
    camera_id: str
    rule: str
    positive: int = 0
    negative: int = 0
    neutral: int = 0

    @property
    def key(self) -> str:
        return f"{self.camera_id}::{self.rule}"

    @property
    def decisive(self) -> int:
        return self.positive + self.negative

    @property
    def precision(self) -> float | None:
        return self.positive / self.decisive if self.decisive else None

    def action(self) -> str:
        p = self.precision
        if p is None:
            return ACTION_WATCH
        if self.negative >= MIN_NEGATIVES and self.decisive >= MIN_REVIEWED and p < DEMOTE_BELOW:
            return ACTION_DEMOTE
        if self.positive >= 3 and p >= TRUST_ABOVE:
            return ACTION_TRUSTED
        return ACTION_WATCH

    def to_dict(self) -> dict:
        return {"camera_id": self.camera_id, "rule": self.rule,
                "positive": self.positive, "negative": self.negative,
                "neutral": self.neutral, "reviewed": self.decisive + self.neutral,
                "precision": round(self.precision, 3) if self.precision is not None else None,
                "action": self.action()}


@dataclass
class Calibration:
    """The computed calibration — what to demote/trust, plus the raw stats."""
    rules: dict = field(default_factory=dict)   # key -> RuleStat
    generated_at: float = 0.0

    @classmethod
    def compute(cls, events: list[LabeledEvent]) -> "Calibration":
        rules: dict = {}
        for e in events:
            st = rules.setdefault(e.key, RuleStat(e.camera_id, e.rule))
            if e.review == "true":
                st.positive += 1
            elif e.review == "false":
                st.negative += 1
            else:
                st.neutral += 1
        return cls(rules=rules, generated_at=time.time())

    @classmethod
    def from_store(cls, store: FeedbackStore) -> "Calibration":
        return cls.compute(store.labeled_events())

    # --- queries the pipeline uses ---
    def demoted(self, camera_id: str, rule: str) -> bool:
        st = self.rules.get(f"{camera_id}::{rule}")
        return bool(st and st.action() == ACTION_DEMOTE)

    def demoted_keys(self) -> list[str]:
        return sorted(k for k, st in self.rules.items() if st.action() == ACTION_DEMOTE)

    def overall_precision(self) -> float | None:
        pos = sum(st.positive for st in self.rules.values())
        neg = sum(st.negative for st in self.rules.values())
        return pos / (pos + neg) if (pos + neg) else None

    def to_dict(self) -> dict:
        return {"version": 1, "generated_at": self.generated_at,
                "overall_precision": (round(self.overall_precision(), 3)
                                      if self.overall_precision() is not None else None),
                "demoted": self.demoted_keys(),
                "rules": {k: st.to_dict() for k, st in sorted(self.rules.items())}}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # --- loading (the pipeline side reads only the demote list) ---
    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        """Load a saved calibration.json. Only reconstructs enough to answer
        demoted()/queries (rebuilds RuleStat from the persisted counts)."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            log.warning("calibration unreadable; running uncalibrated", exc_info=True)
            return cls()
        rules = {}
        for key, d in (data.get("rules") or {}).items():
            st = RuleStat(d.get("camera_id", ""), d.get("rule", ""),
                          int(d.get("positive", 0)), int(d.get("negative", 0)),
                          int(d.get("neutral", 0)))
            rules[key] = st
        return cls(rules=rules, generated_at=float(data.get("generated_at", 0.0)))


class NullCalibration(Calibration):
    """A calibration that never demotes — used when no calibration file exists."""
    def demoted(self, camera_id: str, rule: str) -> bool:  # noqa: ARG002
        return False
