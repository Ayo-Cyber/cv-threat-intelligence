"""W4 write-side: every pending contract row, served through the REAL backend.

The contract's rule is the design: "each endpoint enforces exactly the
permission its backing ConsoleBackend method already enforces — the API adds
transport, never a second permission model." So there is no second
implementation here at all: one ConsoleBackend instance does the work, the
API's bearer principal is bound to it per request (its `current_user` property
is the seam), and PermissionDenied surfaces as the contract's 403 with the
missing permission named in `detail.permission`.

The route table below IS the write-side: one row per contract row, naming the
backing method and how HTTP shapes (path params, JSON body, query) map onto
its keyword arguments. Adding an endpoint means adding a row — and the
contract consistency tests hold docs/api-v1.md, this app, and openapi.json
equal, so a row can't ship undocumented.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Callable, Optional

from fastapi import Depends, Request

from cvti.logging_setup import get_logger

log = get_logger(__name__)


class _ApiBackend:
    """Lazy holder for a ConsoleBackend that acts as the API's principal.

    ConsoleBackend resolves identity through its `current_user` property; the
    subclass points that at whoever the bearer token says is calling, so
    `_require` and every audit record see the API user — the same authz and
    the same paper trail as the desktop console, over a different transport.
    """

    def __init__(self, site_path: str, db_path: str) -> None:
        self.site_path = site_path
        self.db_path = db_path
        self._backend = None
        self._lock = threading.Lock()   # ConsoleBackend is not re-entrant

    def _build(self):
        from cvti.app.console_backend import ConsoleBackend

        class Impersonating(ConsoleBackend):
            _acting = None

            @property
            def current_user(self):  # noqa: D401 - property seam
                if self._acting is not None:
                    return self._acting
                return ConsoleBackend.current_user.fget(self)

        # enable_demo=False: the API serves THIS site — the console's bundled
        # playback demo (a convenience for engine-less laptops) must never be
        # what a remote client reads or writes.
        return Impersonating(site_path=self.site_path, db_path=self.db_path,
                             enable_demo=False)

    def call(self, principal, method: str, /, **kwargs) -> Any:
        """One backend call as `principal` (None = anonymous/public)."""
        with self._lock:
            if self._backend is None:
                self._backend = self._build()
            b = self._backend
            b._acting = None if principal is None else SimpleNamespace(
                username=principal.username, role=principal.role,
                must_change=False)
            try:
                return getattr(b, method)(**kwargs)
            finally:
                b._acting = None


class R:
    """One contract row: verb+path served by `bridge`(**mapped kwargs).

    path_map: url-placeholder -> backend kwarg (values arrive as str).
    body:     JSON-body field -> backend kwarg; a field the body omits is
              simply not passed, so the method's own defaults apply.
    query:    query-param -> (backend kwarg, caster).
    public:   no bearer token required (first-run endpoints only).
    """

    def __init__(self, bridge: str, verb: str, path: str, *,
                 public: bool = False,
                 path_map: dict[str, str] | None = None,
                 body: dict[str, str] | None = None,
                 query: dict[str, tuple[str, Callable]] | None = None,
                 status: int = 200) -> None:
        self.bridge, self.verb, self.path = bridge, verb, path
        self.public = public
        self.path_map = path_map or {}
        self.body = body or {}
        self.query = query or {}
        self.status = status


ROUTES: list[R] = [
    # --- auth & first run ---
    R("auth_state", "GET", "/auth/state", public=True),
    R("create_first_owner", "POST", "/auth/first-owner", public=True,
      body={"username": "username", "password": "password"}),
    R("setup_state", "GET", "/setup/state"),
    R("setup_check", "GET", "/setup/check"),
    R("mark_configured", "POST", "/setup/configured"),
    # --- site & templates ---
    R("get_site", "GET", "/site"),
    R("set_site", "PUT", "/site", body={"name": "name", "notify": "notify"}),
    R("use_case_templates", "GET", "/site/templates"),
    R("apply_template", "POST", "/site/templates/{name}/apply",
      path_map={"name": "key"}),
    R("approve_site_context", "POST", "/site/context/approve",
      body={"context": "context"}),
    R("send_test_notification", "POST", "/site/notifications/test"),
    # --- organization hierarchy ---
    R("organization", "GET", "/organization"),
    R("update_organization", "PUT", "/organization",
      body={"organization": "organization"}),
    R("list_branches", "GET", "/branches"),
    R("create_branch", "POST", "/branches",
      body={"branch": "branch"}, status=201),
    R("update_branch", "PUT", "/branches/{branch_id}",
      path_map={"branch_id": "branch_id"}, body={"branch": "branch"}),
    R("remove_branch", "DELETE", "/branches/{branch_id}",
      path_map={"branch_id": "branch_id"}),
    R("hierarchy", "GET", "/hierarchy"),
    # --- cameras ---
    R("add_camera", "POST", "/cameras", body={"camera": "camera"}, status=201),
    R("remove_camera", "DELETE", "/cameras/{camera_id}",
      path_map={"camera_id": "camera_id"}),
    R("test", "POST", "/cameras/probe",
      body={"url": "url", "source": "url"}),
    R("discover_cameras", "GET", "/cameras/discovery"),
    R("scan", "POST", "/cameras/discovery/scan", body={"cidr": "cidr"}),
    R("detect_subnet", "GET", "/cameras/discovery/subnet"),
    R("presets", "GET", "/cameras/presets"),
    R("camera_snapshot", "GET", "/cameras/{camera_id}/snapshot",
      path_map={"camera_id": "camera_id"}),
    R("camera_links", "GET", "/cameras/{camera_id}/links",
      path_map={"camera_id": None}),   # backend method reads the whole site
    R("assign_camera_area", "PUT", "/cameras/{camera_id}/area",
      path_map={"camera_id": "camera_id"}, body={"area_id": "area_id"}),
    # --- zones ---
    R("list_zones", "GET", "/cameras/{camera_id}/zones",
      path_map={"camera_id": "camera_id"}),
    R("add_zone", "POST", "/cameras/{camera_id}/zones",
      path_map={"camera_id": "camera_id"},
      body={"name": "name", "points": "points",
            "dwell_seconds": "dwell_seconds"}, status=201),
    R("remove_zone", "DELETE", "/cameras/{camera_id}/zones/{name}",
      path_map={"camera_id": "camera_id", "name": "name"}),
    R("accept_suggested_zone", "POST",
      "/cameras/{camera_id}/zones/suggestions/{name}/accept",
      path_map={"camera_id": "camera_id", "name": "zone_id"},
      body={"dwell_seconds": "dwell_seconds"}),
    # --- rules ---
    R("set_camera_rules", "PUT", "/cameras/{camera_id}/rules",
      path_map={"camera_id": "camera_id"}, body={"rules": "rules"}),
    R("add_custom_rule", "POST", "/cameras/{camera_id}/rules/custom",
      path_map={"camera_id": "camera_id"},
      body={"question": "question", "dwell": "dwell"}, status=201),
    R("remove_custom_rule", "DELETE",
      "/cameras/{camera_id}/rules/custom/{name}",
      path_map={"camera_id": "camera_id", "name": "question"}),
    R("english_rules_status", "GET", "/rules/english/status"),
    # --- scene contexts ---
    R("scene_context", "GET", "/cameras/{camera_id}/scene",
      path_map={"camera_id": "camera_id"}),
    R("update_scene_context", "PUT", "/cameras/{camera_id}/scene",
      path_map={"camera_id": "camera_id"}, body={"context": "context"}),
    R("approve_scene_context", "POST", "/cameras/{camera_id}/scene/approve",
      path_map={"camera_id": "camera_id"}, body={"context": "context"}),
    R("request_scene_remap", "POST", "/cameras/{camera_id}/scene/remap",
      path_map={"camera_id": "camera_id"}),
    R("enqueue_scene_mapping", "POST", "/scene-mapping/queue",
      body={"camera_ids": "camera_ids"}),
    R("scene_mapping_progress", "GET", "/scene-mapping/progress"),
    R("scene_review_summary", "GET", "/scene-mapping/review"),
    # --- areas ---
    R("list_areas", "GET", "/areas"),
    R("create_area", "POST", "/areas", body={"area": "area"}, status=201),
    R("area_context", "GET", "/areas/{area_id}/context",
      path_map={"area_id": "area_id"}),
    R("approve_area_context", "POST", "/areas/{area_id}/context/approve",
      path_map={"area_id": "area_id"}, body={"context": "context"}),
    # --- engine control ---
    R("start_monitoring", "POST", "/engine/start"),
    R("stop_monitoring", "POST", "/engine/stop"),
    R("gate_status", "GET", "/engine/gate",
      query={"model": ("model", str)}),
    R("pull_model", "POST", "/engine/models/pull", body={"model": "model"}),
    R("pull_progress", "GET", "/engine/models/pull",
      query={"model": ("model", str)}),
    R("feed_sources", "GET", "/engine/feeds"),
    R("switch_feed", "POST", "/engine/feeds/switch", body={"key": "key"}),
    R("feed_switch_status", "GET", "/engine/feeds/switch"),
    # --- triage writes ---
    R("acknowledge_alert", "POST", "/events/{event_id}/acknowledge",
      path_map={"event_id": "event_id"}),
    R("resolve_alert", "POST", "/events/{event_id}/resolve",
      path_map={"event_id": "event_id"},
      body={"outcome": "outcome", "note": "note"}),
    # --- users, audit, retention, ops ---
    R("list_users", "GET", "/users"),
    R("add_user", "POST", "/users",
      body={"username": "username", "password": "password", "role": "role"},
      status=201),
    R("remove_user", "DELETE", "/users/{username}",
      path_map={"username": "username"}),
    R("audit_entries", "GET", "/audit", query={"limit": ("limit", int)}),
    R("retention_status", "GET", "/retention"),
    R("set_retention", "PUT", "/retention", body={"days": "days"}),
    R("backup_now", "POST", "/backups", status=201),
    R("download_diagnostics", "POST", "/diagnostics/bundle", status=201),
    R("value_summary", "GET", "/value/summary", query={"days": ("days", int)}),
    R("disk_encryption", "GET", "/system/disk-encryption"),
]


def register_writes(app, host: _ApiBackend, require_principal,
                    api_prefix: str, error) -> None:
    """Mount every table row on `app`.

    `require_principal` is the app's bearer dependency; `error` its JSON error
    envelope. Handlers are generated, so each row stays one line of truth."""
    from cvti.security.permissions import PermissionDenied
    from cvti.serving.onboarding import HierarchyConflict

    def _make(route: R):
        async def handler(request: Request, principal=None):
            kwargs: dict[str, Any] = {}
            for url_param, kwarg in route.path_map.items():
                if kwarg is not None:
                    kwargs[kwarg] = request.path_params[url_param]
            if route.body:
                try:
                    body = await request.json()
                except Exception:  # noqa: BLE001 - absent/invalid body
                    log.debug("request body absent or not JSON; using {}",
                              exc_info=True)
                    body = {}
                if not isinstance(body, dict):
                    return error(400, "bad_request", "JSON object body expected")
                for field, kwarg in route.body.items():
                    if field in body:
                        kwargs[kwarg] = body[field]
            for q_param, (kwarg, cast) in route.query.items():
                raw = request.query_params.get(q_param)
                if raw is not None:
                    try:
                        kwargs[kwarg] = cast(raw)
                    except (TypeError, ValueError):
                        return error(400, "bad_request",
                                     f"query param '{q_param}' is not a {cast.__name__}")
            try:
                result = host.call(principal, route.bridge, **kwargs)
            except PermissionDenied as exc:
                return error(403, "forbidden", str(exc),
                             {"permission": exc.permission})
            except TypeError as exc:
                # a required kwarg the request did not supply
                return error(400, "bad_request", str(exc))
            except (KeyError, LookupError) as exc:
                return error(404, "not_found", str(exc))
            except HierarchyConflict as exc:
                return error(409, "conflict", str(exc))
            except ValueError as exc:
                return error(400, "bad_request", str(exc))
            # Backends signal domain failure as {"ok": False, "error": ...}
            if isinstance(result, dict) and result.get("ok") is False:
                msg = str(result.get("error") or "operation refused")
                low = msg.lower()
                if "no such" in low or "unknown" in low or "not found" in low:
                    return error(404, "not_found", msg)
                return error(400, "bad_request", msg)
            return result

        async def auth_handler(request: Request,
                               principal=Depends(require_principal)):
            return await handler(request, principal)

        async def public_handler(request: Request):
            return await handler(request, None)

        chosen = public_handler if route.public else auth_handler
        chosen.__name__ = f"w4_{route.bridge}_{route.verb.lower()}"
        return chosen

    for route in ROUTES:
        app.add_api_route(api_prefix + route.path, _make(route),
                          methods=[route.verb], status_code=route.status,
                          name=f"{route.verb} {route.path} ({route.bridge})")
