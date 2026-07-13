"""Shared data contracts for the threat-intelligence pipelines.

These dataclasses are intentionally dependency-light. They are the boundary objects
passed between detection, customization rules, and verification, so they should not
belong to any one pipeline implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": self.confirmed,
            "confidence": self.confidence,
            "reason": self.reason,
            "alert_priority": self.alert_priority,
            "timestamp": self.timestamp,
        }
