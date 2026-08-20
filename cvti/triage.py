"""Alert states and ownership (EP-06-T1).

Sorting is not triage; colour is not triage. The model cut alerts by 86%, and
the remaining 14% still landed in an undifferentiated list — so the product's
core claim, that it reduces alert fatigue, was delivered entirely by the model
and not at all by the interface.

The state machine:

    NEW ──acknowledge(user)──▶ ACKNOWLEDGED ──resolve(user, outcome, note)──▶ RESOLVED

**Acknowledging claims the alert.** The claimant is visible to everyone, which
is the whole point: with two guards on a shift, both respond or neither does —
ownership is what makes "someone is on it" a fact instead of an assumption.

Transitions are enforced here, in one place, not scattered across the UI:
- a claimed alert cannot be claimed again (the second guard is told who has it)
- a resolved alert is finished — no re-claiming, no re-resolving
- resolving an unclaimed alert claims it in the same breath, and both
  transitions land in the audit trail

The legacy `review` column ('ack'/'true'/'false') is maintained as a projection
of the new state, so the feedback loop, the Value screen and the pending badge
keep working unchanged — resolution outcomes feed `cvti/feedback/` through the
same column they always read.
"""

from __future__ import annotations

import sqlite3
import time

from cvti.logging_setup import get_logger

log = get_logger(__name__)

NEW = "new"
ACKNOWLEDGED = "acknowledged"
RESOLVED = "resolved"

# What a resolution can conclude. 'real' and 'false_alarm' feed the model's
# feedback loop; 'inconclusive' is an honest third option that feeds nothing —
# forcing a binary answer on a genuinely unclear clip poisons the training data.
OUTCOMES = ("real", "false_alarm", "inconclusive")

_OUTCOME_TO_REVIEW = {"real": "true", "false_alarm": "false", "inconclusive": "ack"}

_PRIORITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


class TriageError(Exception):
    """An impossible transition. The message names what actually happened."""


def _row(con: sqlite3.Connection, event_id: int) -> sqlite3.Row:
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT id, state, owner, priority, rule, camera_id, outcome, review "
        "FROM events WHERE id = ?", (int(event_id),)).fetchone()
    if row is None:
        raise TriageError(f"no alert with id {event_id}")
    return row


def state_of(row) -> str:
    """Effective state, tolerant of rows from before the state column."""
    state = row["state"] if "state" in row.keys() else None
    if state in (NEW, ACKNOWLEDGED, RESOLVED):
        return state
    # Legacy projection: infer from the old review column.
    review = row["review"] if "review" in row.keys() else None
    if review in ("true", "false"):
        return RESOLVED
    if review == "ack":
        return ACKNOWLEDGED
    return NEW


def acknowledge(con: sqlite3.Connection, event_id: int, user: str) -> dict:
    """Claim the alert. Fails loudly if someone already has it."""
    row = _row(con, event_id)
    state = state_of(row)
    if state == RESOLVED:
        raise TriageError(f"alert {event_id} is already resolved")
    if state == ACKNOWLEDGED:
        if row["owner"] == user:
            return {"ok": True, "state": ACKNOWLEDGED, "owner": user,
                    "already_yours": True}
        # The other guard finding out someone has it IS the feature.
        raise TriageError(f"{row['owner']} is already on alert {event_id}")
    now = time.time()
    con.execute(
        "UPDATE events SET state = ?, owner = ?, acknowledged_at = ?, "
        "review = 'ack', reviewed_at = ? WHERE id = ?",
        (ACKNOWLEDGED, user, now, time.strftime("%Y-%m-%dT%H:%M:%S"), int(event_id)))
    con.commit()
    log.info("alert %s claimed by %s", event_id, user)
    return {"ok": True, "state": ACKNOWLEDGED, "owner": user}


def resolve(con: sqlite3.Connection, event_id: int, user: str,
            outcome: str, note: str = "") -> dict:
    """Finish the alert with a conclusion and (optionally) a note.

    Resolving an unclaimed alert claims it first — both transitions are real
    and both are reported so the audit trail shows what actually happened.
    """
    if outcome not in OUTCOMES:
        raise TriageError(f"unknown outcome {outcome!r}; expected one of {OUTCOMES}")
    row = _row(con, event_id)
    state = state_of(row)
    if state == RESOLVED:
        raise TriageError(
            f"alert {event_id} was already resolved as {row['outcome'] or 'unknown'}")
    transitions = []
    if state == NEW:
        acknowledge(con, event_id, user)
        transitions.append(ACKNOWLEDGED)
    now = time.time()
    con.execute(
        "UPDATE events SET state = ?, resolved_at = ?, outcome = ?, note = ?, "
        "review = ?, reviewed_at = ? WHERE id = ?",
        (RESOLVED, now, outcome, note or None, _OUTCOME_TO_REVIEW[outcome],
         time.strftime("%Y-%m-%dT%H:%M:%S"), int(event_id)))
    con.commit()
    transitions.append(RESOLVED)
    log.info("alert %s resolved by %s as %s", event_id, user, outcome)
    return {"ok": True, "state": RESOLVED, "owner": row["owner"] or user,
            "outcome": outcome, "transitions": transitions}


def needs_attention(con: sqlite3.Connection, *, min_priority: str = "medium",
                    limit: int = 3) -> dict:
    """What needs a human right now.

    One item first — at 2am a guard needs a single next action, not a list —
    plus the short queue behind it and who is already holding what, so the
    second guard on shift can see the first one working.
    """
    con.row_factory = sqlite3.Row
    threshold = _PRIORITY_RANK.get(min_priority, 1)
    ranked = " ".join(f"WHEN '{p}' THEN {r}" for p, r in _PRIORITY_RANK.items())
    rows = con.execute(
        f"SELECT * FROM events WHERE COALESCE(retracted, 0) = 0 "
        f"AND COALESCE(state, CASE "
        f"  WHEN review IN ('true','false') THEN 'resolved' "
        f"  WHEN review = 'ack' THEN 'acknowledged' ELSE 'new' END) = 'new' "
        f"AND (CASE priority {ranked} ELSE 0 END) >= ? "
        f"ORDER BY (CASE priority {ranked} ELSE 0 END) DESC, ts ASC LIMIT ?",
        (threshold, int(limit))).fetchall()
    held = con.execute(
        "SELECT id, rule, camera_id, owner, acknowledged_at FROM events "
        "WHERE state = 'acknowledged' ORDER BY acknowledged_at DESC LIMIT 5").fetchall()
    waiting = con.execute(
        f"SELECT COUNT(*) FROM events WHERE COALESCE(retracted, 0) = 0 "
        f"AND COALESCE(state, CASE "
        f"  WHEN review IN ('true','false') THEN 'resolved' "
        f"  WHEN review = 'ack' THEN 'acknowledged' ELSE 'new' END) = 'new' "
        f"AND (CASE priority {ranked} ELSE 0 END) >= ?", (threshold,)).fetchone()[0]
    return {
        "now": dict(rows[0]) if rows else None,
        "then": [dict(r) for r in rows[1:]],
        "waiting": waiting,
        "held": [dict(h) for h in held],
    }


def ensure_columns(con: sqlite3.Connection) -> None:
    """Migrate an existing events table in place. Idempotent."""
    for col, decl in (("state", "TEXT"), ("owner", "TEXT"),
                      ("acknowledged_at", "REAL"), ("resolved_at", "REAL"),
                      ("outcome", "TEXT"), ("note", "TEXT")):
        try:
            con.execute(f"ALTER TABLE events ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass
    con.commit()
