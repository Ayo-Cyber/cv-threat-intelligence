"""Serving-only adapters and freshness guards for object-watch results."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cvti.event_adapters import object_observations_to_events
from cvti.object_watch.store import library_revision, load_targets


def object_watch_rule_signature(rule: Any) -> str:
    """Stable identity for the complete rendered policy, not just its label."""
    if not isinstance(rule, dict):
        return ""
    return json.dumps(rule, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def current_object_watch_rule(state: Any, candidate: Any, *, disk: bool = True) -> dict | None:
    metadata = getattr(candidate, "metadata", None) or {}
    rules = None
    path = getattr(state, "_object_watch_rule_path", None)
    if disk and path:
        try:
            loaded = json.loads(Path(path).read_text())
            rules = tuple(loaded.get("rules", ()))
        except (OSError, TypeError, ValueError):
            return None
    if rules is None:
        rules = tuple(getattr(getattr(state, "engine", None), "baseline_rules", ()) or ()) + \
            tuple(getattr(getattr(state, "engine", None), "rules", ()) or ())
    for rule in rules:
        trigger = rule.get("trigger", {}) if isinstance(rule, dict) else {}
        if (rule.get("name") == getattr(candidate, "rule_name", None)
                and trigger.get("detector") == "object_watch"
                and trigger.get("object_id") == metadata.get("object_id")
                and ("zone" not in trigger or trigger.get("zone") == metadata.get("zone"))):
            return rule
    return None


def object_watch_alert_snapshot_current(alert: Any, state: Any, runtime: Any,
                                        result: Any, now_monotonic: float) -> bool:
    """Camera-thread guard using worker/rule snapshots only; never touches disk."""
    payload = getattr(alert, "payload", None) or {}
    candidate = payload.get("candidate") if isinstance(payload, dict) else None
    if getattr(candidate, "detector", "") != "object_watch":
        return True
    if state is None or runtime is None or result is None \
            or not bool(getattr(state, "object_watch", False)):
        return False
    metadata = getattr(candidate, "metadata", None) or {}
    if int(metadata.get("source_generation", -1)) != int(
            getattr(state, "_object_watch_generation", -2)):
        return False
    if not runtime.result_current(result):
        return False
    ttl = float(getattr(runtime.config, "result_ttl_seconds", 0.0))
    if ttl <= 0 or float(now_monotonic) - float(result.observed_at_monotonic) > ttl:
        return False
    rule = current_object_watch_rule(state, candidate, disk=False)
    return bool(rule is not None and metadata.get("object_watch_rule_signature")
                == object_watch_rule_signature(rule))


def object_watch_result_events(result: Any) -> list:
    """Adapt only matches for which the runtime supplied reference evidence.

    Each result is one match plus evidence captured from that same sample and
    enrollment snapshot.
    """
    matches = tuple(getattr(result, "matches", ()) or ())
    evidence = tuple(getattr(result, "evidence", ()) or ())
    if len(matches) != 1 or len(evidence) < 2:
        return []
    match = matches[0]
    observation = SimpleNamespace(
        timestamp=float(getattr(result, "event_timestamp", None) or match.timestamp),
        state="object_seen",
        object_id=match.object_id,
        object_label=match.object_label,
        category=match.category,
        zone_id=match.zone_id,
        track_id=match.track_id,
        bbox=tuple(match.bbox),
        similarity=float(match.similarity),
        dwell_seconds=0.0,
        reasons=(f"reference similarity {float(match.similarity):.3f}",),
    )
    events = object_observations_to_events([observation], observation.timestamp)
    for event in events:
        event.extra.update({
            "camera_id": str(result.camera_id),
            "source_generation": int(result.source_generation),
            "sample_sequence": int(result.sample_sequence),
            "library_revision": int(result.library_revision),
            "target_revision": int(match.target_revision),
            "model_fingerprint": str(result.model_fingerprint),
        })
    return events


def object_watch_alert_current(alert: Any, state: Any, library: str | Path) -> bool:
    """Fail closed when an object alert no longer names an active revision."""
    payload = getattr(alert, "payload", None) or {}
    candidate = payload.get("candidate") if isinstance(payload, dict) else None
    if getattr(candidate, "detector", "") != "object_watch":
        return True
    if state is None or not bool(getattr(state, "object_watch", False)):
        return False
    metadata = getattr(candidate, "metadata", None) or {}
    object_id = metadata.get("object_id")
    try:
        config_path = Path(library) / "runtime.json"
        try:
            config_stamp = (config_path.stat().st_mtime_ns, config_path.stat().st_size)
        except OSError:
            config_stamp = ()
        recorded_stamp = metadata.get("runtime_config_stamp")
        if recorded_stamp is None or tuple(recorded_stamp) != config_stamp:
            return False
        if int(metadata.get("source_generation", -1)) != int(
                getattr(state, "_object_watch_generation", -2)):
            return False
        if int(metadata.get("library_revision", -1)) != library_revision(library):
            return False
        target = next(item for item in load_targets(library) if item.id == object_id)
    except (OSError, TypeError, ValueError, StopIteration):
        return False
    rule = current_object_watch_rule(state, candidate)
    if rule is None or metadata.get("object_watch_rule_signature") != object_watch_rule_signature(rule):
        return False
    model_fingerprint = str(metadata.get("model_fingerprint") or "")
    if model_fingerprint:
        try:
            from cvti.object_watch.runtime_config import (
                configured_backend_metadata, resolve_config,
            )
            current_model = configured_backend_metadata(
                resolve_config(Path(library).resolve().parent)
            ).fingerprint
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        if model_fingerprint != current_model:
            return False
    return (target.review_state == "active"
            and target.revision == int(metadata.get("target_revision", -1)))
