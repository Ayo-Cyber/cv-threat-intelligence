"""The Argus Engine API app (FastAPI).

Read-only surface + auth + live WebSocket, per API contract v0.2. Auth is
bearer-token against the account store; data comes from cvti.api.sources.
Write/config endpoints are intentionally absent until the contract freezes.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from cvti.api import sources
from cvti.api.tokens import TokenStore
from cvti.logging_setup import get_logger

log = get_logger(__name__)

API_PREFIX = "/api/v1"


def _error(status: int, code: str, message: str, detail: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message,
                                           "detail": detail or {}}})


def register_index(app: FastAPI, *, mock: bool = False) -> None:
    """A self-describing root, so hitting the base URL isn't a bare 404.

    GET / and GET /api/v1 return the API's name, version, where the docs live,
    and the live endpoint list — no auth, no data, just discovery.
    """
    def _payload() -> dict:
        paths = sorted({
            r.path for r in app.routes
            if getattr(r, "methods", None) and r.path.startswith(API_PREFIX)
        })
        return {
            "name": "Argus Engine API",
            "version": app.version,
            "mock": mock,
            "status": "ok",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "websocket": f"{API_PREFIX}/stream",
            "endpoints": paths,
        }

    @app.get("/", include_in_schema=False)
    async def root():
        return _payload()

    @app.get(API_PREFIX, include_in_schema=False)
    async def api_root():
        return _payload()


def create_app(*, db_path: str = "runs/site/events.db",
               site_path: str = "configs/site_live.json") -> FastAPI:
    app = FastAPI(title="Argus Engine API", version="0.2.0")
    app.state.db_path = db_path
    app.state.site_path = site_path
    app.state.tokens = TokenStore()

    # Auth built lazily: the account store lives beside the events db, and a
    # test or a fresh box may not have one yet.
    def _accounts():
        from cvti.security.accounts import AccountStore
        return AccountStore(Path(app.state.db_path).parent / "auth.db")

    def require_principal(authorization: Optional[str] = Header(default=None)):
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        principal = app.state.tokens.resolve(token)
        if principal is None:
            raise HTTPException(status_code=401, detail="unauthorized")
        return principal

    # ---- auth ---------------------------------------------------------------
    @app.post(f"{API_PREFIX}/auth/session")
    async def sign_in(body: dict[str, Any]):
        from cvti.security.accounts import AuthError
        from cvti.security import permissions as perms
        username = str(body.get("username", ""))
        password = str(body.get("password", ""))
        try:
            user = _accounts().authenticate(username, password)
        except AuthError as exc:
            return _error(401, "unauthorized", str(exc))
        except Exception:  # noqa: BLE001 - no store yet / unreadable
            log.debug("sign-in failed (no/unreadable account store)", exc_info=True)
            return _error(401, "unauthorized", "invalid username or password")
        token, principal = app.state.tokens.mint(user.username, user.role)
        return {"token": token,
                "expires_at": _iso(principal.expires_at),
                "user": {"username": user.username, "role": user.role,
                         "permissions": sorted(perms.permissions_for(user.role))}}

    @app.get(f"{API_PREFIX}/auth/me")
    async def me(principal=Depends(require_principal)):
        from cvti.security import permissions as perms
        return {"username": principal.username, "role": principal.role,
                "permissions": sorted(perms.permissions_for(principal.role))}

    @app.delete(f"{API_PREFIX}/auth/session", status_code=204)
    async def sign_out(authorization: Optional[str] = Header(default=None)):
        if authorization and authorization.lower().startswith("bearer "):
            app.state.tokens.revoke(authorization[7:].strip())
        return JSONResponse(status_code=204, content=None)

    @app.get(f"{API_PREFIX}/roles")
    async def roles(principal=Depends(require_principal)):
        from cvti.security import permissions as perms
        return perms.describe()

    # ---- system -------------------------------------------------------------
    @app.get(f"{API_PREFIX}/system/health")
    async def health(principal=Depends(require_principal)):
        return sources.read_health(app.state.db_path)

    @app.get(f"{API_PREFIX}/system/info")
    async def info(principal=Depends(require_principal)):
        import platform
        from cvti.utils import argus_version
        return {"version": argus_version(), "node_id": "local",
                "platform": platform.system(),
                "capabilities": ["events", "health", "live_mjpeg", "websocket"]}

    @app.get(f"{API_PREFIX}/monitor")
    async def monitor(principal=Depends(require_principal)):
        return sources.monitor_state(app.state.db_path)

    # ---- cameras ------------------------------------------------------------
    @app.get(f"{API_PREFIX}/cameras")
    async def cameras(principal=Depends(require_principal)):
        return sources.read_cameras(app.state.site_path, app.state.db_path)

    @app.get(f"{API_PREFIX}/cameras/{{camera_id}}")
    async def camera(camera_id: str, principal=Depends(require_principal)):
        for c in sources.read_cameras(app.state.site_path, app.state.db_path):
            if c["id"] == camera_id:
                return c
        return _error(404, "not_found", f"no such camera '{camera_id}'")

    # ---- events & triage ----------------------------------------------------
    @app.get(f"{API_PREFIX}/events")
    async def events(principal=Depends(require_principal),
                     limit: int = Query(50, ge=1, le=200),
                     cursor: Optional[int] = Query(None),
                     camera: Optional[str] = Query(None),
                     priority: Optional[str] = Query(None)):
        return sources.read_events(app.state.db_path, limit=limit, cursor=cursor,
                                   camera=camera, priority=priority)

    @app.get(f"{API_PREFIX}/events/{{event_id}}")
    async def event(event_id: str, principal=Depends(require_principal)):
        got = sources.read_event(app.state.db_path, event_id)
        if got is None:
            return _error(404, "not_found", f"no such event '{event_id}'")
        return got

    @app.get(f"{API_PREFIX}/triage")
    async def triage(principal=Depends(require_principal)):
        return sources.read_triage(app.state.db_path)

    # ---- live video (transport descriptor) ----------------------------------
    @app.get(f"{API_PREFIX}/cameras/{{camera_id}}/stream")
    async def stream(camera_id: str, principal=Depends(require_principal)):
        # Today: point at the engine's MJPEG publisher (frames.json). When
        # go2rtc lands, only kind+url change; the player switches on `kind`.
        frames = Path(app.state.db_path).parent / "frames.json"
        try:
            pub = json.loads(frames.read_text())
            base = f"http://127.0.0.1:{pub['port']}/stream/{camera_id}?token={pub['token']}"
            return {"kind": "mjpeg", "url": base}
        except (OSError, ValueError, KeyError):
            return _error(503, "engine_unavailable",
                          "no live stream — engine not publishing frames",
                          {"phase": sources.monitor_state(app.state.db_path)["phase"]})

    # ---- websocket ----------------------------------------------------------
    @app.websocket(f"{API_PREFIX}/stream")
    async def ws_stream(ws: WebSocket, token: Optional[str] = Query(None)):
        if app.state.tokens.resolve(token) is None:
            await ws.close(code=4401)   # unauthorized
            return
        await ws.accept()
        db = app.state.db_path
        # Hydrate on connect (§16): one health + triage snapshot so the UI
        # paints without a separate poll.
        await _send(ws, "health", sources.read_health(db))
        await _send(ws, "triage", sources.read_triage(db))
        last_id = sources.max_event_id(db)
        last_health_gen = (sources.read_health(db).get("generated_at"))
        try:
            while True:
                await asyncio.sleep(1.0)
                # new alerts
                newest = sources.max_event_id(db)
                if newest > last_id:
                    batch = sources.read_events(db, limit=min(50, newest - last_id))
                    for ev in reversed(batch.get("events", [])):
                        if int(ev["id"].removeprefix("evt_")) > last_id:
                            await _send(ws, "alert.new", ev)
                    last_id = newest
                # health refresh
                doc = sources.read_health(db)
                if doc.get("generated_at") != last_health_gen:
                    last_health_gen = doc.get("generated_at")
                    await _send(ws, "health", doc)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 - a push loop must not crash the server
            log.debug("websocket push loop ended on error", exc_info=True)
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                log.debug("websocket close after error also failed", exc_info=True)

    register_index(app, mock=False)
    return app


async def _send(ws: WebSocket, type_: str, data: Any) -> None:
    await ws.send_text(json.dumps({"type": type_, "ts": time.time(), "data": data}))


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))
