"""Deterministic rule-to-scene compatibility checks.

The mapper describes a scene; it does not decide policy. This module only
enforces restrictions explicitly attached to a rule and never treats mapper
zone suggestions as accepted configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompatibilityDecision:
    allowed: bool
    mode: str
    reason: str


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def evaluate_context_compatibility(
    rule: dict,
    context: dict | None,
    *,
    baseline: bool = False,
    active_zone_roles: set[str] | None = None,
) -> CompatibilityDecision:
    if baseline or bool(rule.get("critical_baseline")):
        return CompatibilityDecision(
            True,
            "critical_baseline",
            "always-on critical safety rules bypass scene restrictions",
        )

    requirements = rule.get("context_requirements")
    if not isinstance(requirements, dict):
        return CompatibilityDecision(
            True, "allowed", "rule has no scene-context restriction"
        )

    mode = str(requirements.get("mode", "enforce")).strip().lower()
    if mode == "override":
        return CompatibilityDecision(
            True,
            "explicit_override",
            "customer explicitly allowed this rule outside its default context",
        )

    allowed_environments = _string_set(requirements.get("environment_types"))
    allowed_zone_roles = _string_set(requirements.get("zone_roles_any"))
    environment = str((context or {}).get("environment_type", "unknown"))
    accepted_roles = set(active_zone_roles or set())

    if environment in allowed_environments:
        return CompatibilityDecision(
            True, "context_match", f"environment {environment} is allowed"
        )
    matched_roles = sorted(accepted_roles & allowed_zone_roles)
    if matched_roles:
        return CompatibilityDecision(
            True,
            "context_match",
            f"accepted zone role {matched_roles[0]} is allowed",
        )

    requirements_text: list[str] = []
    if allowed_environments:
        requirements_text.append(
            "environment " + ", ".join(sorted(allowed_environments))
        )
    if allowed_zone_roles:
        requirements_text.append(
            "accepted zone role " + ", ".join(sorted(allowed_zone_roles))
        )
    expected = " or ".join(requirements_text) or "a configured context match"
    return CompatibilityDecision(
        False,
        "context_incompatible",
        f"requires {expected}; observed environment {environment}",
    )
