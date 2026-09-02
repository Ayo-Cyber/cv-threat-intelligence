"""Shared data contracts for the threat-intelligence pipelines.

These dataclasses are intentionally dependency-light. They are the boundary objects
passed between detection, customization rules, and verification, so they should not
belong to any one pipeline implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The ONE local VLM tag the product runs on. Until 2 Sep this was two tags:
# the engine, the first-run pull, and the evals used "gemma3:4b" while the
# mapper's and the `local` provider's defaults said "gemma3:4b-it-qat" — so a
# default-configured mapper asked the server for a model nobody had pulled
# (a second ~3 GB download, or a failed mapping, depending on the machine).
# Every measured number in this repo (SENSITIVITY_MEASURED, runs/eval/*) is
# gemma3:4b; anything else here is a lie about what was measured.
LOCAL_VLM_MODEL = "gemma3:4b"


@dataclass
class RawEvent:
    detector: str
    active: bool
    title: str
    level: str
    state: str = ""
    person_id: int | None = None
    object_label: str | None = None
    timestamp: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class CandidateAlert:
    rule_name: str
    priority: str
    detector: str
    title: str
    person_id: int | None
    object_label: str | None
    timestamp: float
    reasons: list[str] = field(default_factory=list)
    # Optional rule-specific gate question (used by compound recipes); falls
    # back to the gate's per-rule default question when None.
    question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "priority": self.priority,
            "detector": self.detector,
            "title": self.title,
            "person_id": self.person_id,
            "object_label": self.object_label,
            "timestamp": self.timestamp,
            "reasons": self.reasons,
            "question": self.question,
        }


@dataclass
class VerificationResult:
    confirmed: bool
    confidence: float
    reason: str
    alert_priority: str
    timestamp: str
    raw_response: str = ""
    # Why `confirmed=False` — and they are not the same thing. A rejection is a
    # verdict: the model looked and said no. An error is the absence of a
    # verdict: the model was unreachable, or answered with something we could
    # not parse. Collapsing the two means a connection failure is indistinguishable
    # from TrueSight examining a fire and deciding it is safe.
    error: str = ""
    # Which prompt wording produced this verdict (short fingerprint, see
    # cvti/eval/prompt_fingerprint.py). Three revisions moved precision 26
    # points; an event that cannot say which wording judged it cannot be
    # compared honestly with one judged by another.
    prompt_version: str = ""

    @property
    def errored(self) -> bool:
        """True when no verdict was reached. Never equivalent to a rejection."""
        return bool(self.error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": self.confirmed,
            "confidence": self.confidence,
            "reason": self.reason,
            "alert_priority": self.alert_priority,
            "timestamp": self.timestamp,
            "error": self.error,
        }
