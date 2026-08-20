"""Three roles, and what each may actually do.

The interface assumed installer, operator and owner were one person. In a real
deployment they are three, with near-disjoint needs: the installer commissions
cameras and leaves, the operator watches alerts on a shift, the owner answers
for the site. Showing all three every surface makes the product worse for each
— and lets an operator disable a detector by accident.

Enforcement lives here and is checked in the backend, not in the interface.
Hiding a button changes what is easy; it does not change what is possible, and
the question procurement asks is what is *possible*.
"""

from __future__ import annotations

from cvti.logging_setup import get_logger

log = get_logger(__name__)

OWNER = "owner"
OPERATOR = "operator"
INSTALLER = "installer"
ROLES = (OWNER, OPERATOR, INSTALLER)

# --- permissions ------------------------------------------------------------
VIEW_ALERTS = "view_alerts"           # the alert list and its evidence
REVIEW_ALERTS = "review_alerts"       # label an alert true/false/acknowledged
VIEW_LIVE = "view_live"               # live camera frames
CONFIGURE_CAMERAS = "configure_cameras"
CONFIGURE_DETECTORS = "configure_detectors"   # which threats each camera watches
CONFIGURE_SITE = "configure_site"     # notifications, retention, value inputs
CONTROL_ENGINE = "control_engine"     # start/stop monitoring
EXPORT_EVIDENCE = "export_evidence"
MANAGE_LEGAL_HOLD = "manage_legal_hold"
VIEW_AUDIT = "view_audit"             # the audit trail
MANAGE_USERS = "manage_users"
VIEW_DIAGNOSTICS = "view_diagnostics"

# Deliberately explicit rather than derived from a hierarchy. A hierarchy makes
# "can an operator do X?" a question about inheritance; a table makes it a
# question you can answer by reading one line, which is what a security review
# actually needs.
_GRANTS: dict[str, frozenset] = {
    OWNER: frozenset({
        VIEW_ALERTS, REVIEW_ALERTS, VIEW_LIVE, CONFIGURE_CAMERAS, CONFIGURE_DETECTORS,
        CONFIGURE_SITE, CONTROL_ENGINE, EXPORT_EVIDENCE, MANAGE_LEGAL_HOLD,
        VIEW_AUDIT, MANAGE_USERS, VIEW_DIAGNOSTICS,
    }),
    # The operator watches and judges. They do not decide what the system
    # watches for — that is the whole point of separating the roles, and it is
    # what stops a detector being switched off during a shift.
    OPERATOR: frozenset({
        VIEW_ALERTS, REVIEW_ALERTS, VIEW_LIVE, EXPORT_EVIDENCE, MANAGE_LEGAL_HOLD,
    }),
    # The installer commissions the site and leaves. They need cameras, zones
    # and detectors to work — and have no business reading recorded incidents.
    INSTALLER: frozenset({
        VIEW_LIVE, CONFIGURE_CAMERAS, CONFIGURE_DETECTORS, CONTROL_ENGINE,
        VIEW_DIAGNOSTICS,
    }),
}

# Where each role lands, because the first screen should be the one they came for.
LANDING: dict[str, str] = {
    OWNER: "alerts",
    OPERATOR: "alerts",
    INSTALLER: "cameras",
}


class PermissionDenied(Exception):
    """The role is real; the action is not theirs."""

    def __init__(self, role: str, permission: str) -> None:
        super().__init__(f"role {role!r} may not {permission}")
        self.role = role
        self.permission = permission


def permissions_for(role: str) -> frozenset:
    return _GRANTS.get(role, frozenset())


def allows(role: str, permission: str) -> bool:
    return permission in permissions_for(role)


def require(role: str | None, permission: str) -> None:
    """Raise unless `role` holds `permission`. The server-side check."""
    if role is None:
        raise PermissionDenied("<anonymous>", permission)
    if not allows(role, permission):
        log.warning("permission denied: role=%s action=%s", role, permission)
        raise PermissionDenied(role, permission)


def landing_for(role: str) -> str:
    return LANDING.get(role, "alerts")


def describe() -> dict:
    """The whole table, for the UI and for a procurement questionnaire."""
    return {role: sorted(perms) for role, perms in _GRANTS.items()}
